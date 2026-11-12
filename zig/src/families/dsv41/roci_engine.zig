//! The Rocinante transport inside the engine: the arena in registered host memory, the queue pairs on both
//! rails, the start-up exchange over the engine's own links, and the sends and waits of the prompt path.

const std = @import("std");
const cuda = @import("cuda");
const fabric = @import("fabric");
const roci = @import("rocinante.zig");
const rv = @import("rocinante_verbs.zig");
const cd = @import("rocinante_cd.zig");
const mx = @import("mlx5_devx.zig");
const nap = @import("rdma_nap.zig");
const prompt2d = @import("prompt2d.zig");

pub const Region = roci.Region;
pub const Class = roci.Class;

const host_portable: c_uint = 0x01;
const host_mapped: c_uint = 0x02;
const host_wc: c_uint = 0x04;

/// A mapped host allocation and its device address.
const HostMem = struct {
    bytes: []align(64) u8,
    dev: u64,

    fn alloc(d: *const cuda.Driver, len: usize, flags: c_uint) !HostMem {
        var p: ?*anyopaque = null;
        try d.check(d.api.cuMemHostAlloc(&p, len, flags), "cuMemHostAlloc");
        errdefer _ = d.api.cuMemFreeHost(p);
        var dev: u64 = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, p, 0), "cuMemHostGetDevicePointer");
        const base: [*]align(64) u8 = @ptrCast(@alignCast(p.?));
        return .{ .bytes = base[0..len], .dev = dev };
    }
};

/// How the transport's arena (the regions it sends from and lands into) is allocated: `mmap` anonymous pages, locked, registered with
/// cuMemHostRegister (mapped, portable), the type the decode rings use; `wc` cuMemHostAlloc, mapped and write-combined, which only the
/// transport's own host tests open. The bytes the engine computes do not depend on it.
/// The NIC a lane runs on: the verbs one, or (TF_DS_ROCI_WAIT=1) the one whose queue pairs wait in the card.
const NicBox = union(enum) {
    verbs: *rv.VerbsNic,
    cd: *cd.CdNic,

    fn info(b: NicBox) !rv.Info {
        return switch (b) {
            .verbs => |n| n.info(),
            .cd => |n| n.info(),
        };
    }

    fn connect(b: NicBox, all: *const [roci.world]rv.Info) !void {
        return switch (b) {
            .verbs => |n| n.connect(all),
            .cd => |n| n.connect(all),
        };
    }

    fn nic(b: NicBox) roci.Nic {
        return switch (b) {
            .verbs => |n| n.nic(),
            .cd => |n| n.nic(),
        };
    }

    fn close(b: NicBox) void {
        switch (b) {
            .verbs => |n| n.close(),
            .cd => |n| n.close(),
        }
    }
};

pub const Roci = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    g: u8,
    layout: roci.Layout,
    arena: HostMem,
    /// The second lane of a rank borrows the first lane's allocation (its second half): only the first frees it.
    owns_arena: bool = true,
    /// This lane (0, 1 or 2 for the prompt lanes of a rank, 3 the host lane) and the device addresses [lo, hi) of the whole allocation the
    /// prompt lanes' arenas lie in.
    lane: u8 = 0,
    span: [2]u64 = .{ 0, 0 },
    ctl_mem: HostMem,
    verbs: fabric.verbs.Verbs,
    /// libmlx5, loaded when the queue pairs wait in the card (TF_DS_ROCI_WAIT=1).
    mlx: ?mx.Mlx5 = null,
    nic: NicBox,
    t: *roci.Transport,
    passes: u64 = 0,
    prev: roci.Counters = .{},

    /// open for the lanes of a rank: `lanes` 2 or 3 allocates the lanes' arenas as one block (so mx.Kernels' one range tests them all) and
    /// lane 0 takes the first part; with `first` (the lane 0 Roci) `lanes` is this lane's place in the group (2: lane 1, 3: lane 2) and it opens
    /// on that part of the block, with its own control block, queue pairs and proxies, and pairs the transports (a failure on one stops
    /// the others: the caller links three with Transport.ring3). The last lane is closed first, lane 0 last.
    pub fn openLane(gpa: std.mem.Allocator, d: *const cuda.Driver, rank: u32, node: u8, world: u32, order: [4]u32, fan: *const prompt2d.Fan, devices: [roci.rails][]const u8, gid: u8, lanes: u8, first: ?*Roci, layout_set: roci.LayoutSet) !*Roci {
        const pages: Pages = .mmap; // (locked anonymous pages registered with cuMemHostRegister: the arena the transport's kernels run at device speed on)
        if (world != roci.world) return error.NotFourRanks;
        if (lanes < 1 or lanes > 3) return error.BadLanes;
        const r = try gpa.create(Roci);
        errdefer gpa.destroy(r);
        const total: usize = @intCast(r.layout.total);
            r.lane = 3;
            // the second or third lane: the first lane's block, the part of its place (`lanes` is 2 or 3 here)
            if (lanes < 2 or f.lane != 0 or f.span[1] - f.span[0] < lanes * total) return error.NoSecondArena;
            r.lane = lanes - 1;
            r.owns_arena = false;
            r.arena_mem = f.arena_mem;
            r.arena = .{ .bytes = @alignCast(f.arena_mem.mem.bytes[r.lane * total ..][0..total]), .dev = f.arena_mem.mem.dev + r.lane * total };
            r.span = f.span;
        } else {
            r.arena_mem = try Arena.alloc(d, total * lanes, pages);
            r.arena = .{ .bytes = r.arena_mem.mem.bytes[0..total], .dev = r.arena_mem.mem.dev };
            r.span = .{ r.arena_mem.mem.dev, r.arena_mem.mem.dev + total * lanes };
        }
        errdefer if (r.owns_arena) r.arena_mem.free(d);
        r.ctl_mem = try HostMem.alloc(d, @sizeOf(roci.Control), host_portable | host_mapped);
        errdefer _ = d.api.cuMemFreeHost(r.ctl_mem.bytes.ptr);
        const ctl: *roci.Control = @ptrCast(r.ctl_mem.bytes.ptr);
        ctl.* = .{};
        r.verbs = try fabric.verbs.Verbs.open(null);
        errdefer r.verbs.close();
        const t0 = nowNs();
        const sigtext: ?[]const u8 = if (std.c.getenv("TF_DS_ROCI_SIGNAL")) |v| std.mem.span(v) else null;
        const wanted = roci.signalFrom(sigtext) orelse {
            std.log.err("TF_DS_ROCI_SIGNAL={s}: neither write nor atomic", .{sigtext.?});
            return error.BadSignal;
        };
        var ab: [300]u8 = undefined;
        const atomic_why: ?[]const u8 = if (wanted == .atomic) rdma.probeAtomic(&ab, devices[0..]) else null;
        const sm = signalMode(sigtext, atomic_why).?;
        if (sm.refuse) {
            std.log.err("TF_DS_ROCI_SIGNAL=atomic: {s}: the arrival signal stays a WRITE with TF_DS_ROCI_SIGNAL=write", .{sm.why.?});
            return error.AtomicUnsupported;
        }
        if (sm.warn) std.log.warn("TF_DS_ROCI_SIGNAL: {s}: {s}", .{ sm.note, sm.why.? }) else std.log.info("TF_DS_ROCI_SIGNAL: {s}", .{sm.note});
        const signal = sm.signal;
        const wtext: ?[]const u8 = if (std.c.getenv("TF_DS_ROCI_WAIT")) |v| std.mem.span(v) else null;
        // libmlx5 is probed only when the card's WAIT is wanted; the probe is the library the lanes then use
        var probe: ?mx.Mlx5 = if (prompt2d.switchDefaultOn(wtext)) (mx.Mlx5.open() catch null) else null;
        errdefer if (probe) |*m| m.close();
        const mode = waitMode(wtext, signal, probe != null);
        const in_card = mode.on;
        if (mode.refuse) {
            std.log.err("TF_DS_ROCI_WAIT=1: {s} does not load: the queue pairs that wait in the card need its DEVX entry points", .{mx.library});
            return error.DevxUnavailable;
        }
        std.log.info("TF_DS_ROCI_WAIT: {s}", .{mode.note});
        if (!in_card) {
            if (probe) |*m| m.close();
            probe = null;
        }
        errdefer if (r.mlx) |*m| m.close();
        if (in_card) {
            r.mlx = probe;
            probe = null;
            r.nic = .{ .cd = try cd.CdNic.open(gpa, &r.verbs, &r.mlx.?, .{ .devices = devices, .gid_index = gid, .signal = signal }, r.g, r.arena.bytes, r.ctl_mem.bytes[0..@sizeOf(roci.Control)], r.layout.hash()) };
        } else r.nic = .{ .verbs = try rv.VerbsNic.open(gpa, &r.verbs, .{ .devices = devices, .gid_index = gid, .signal = signal }, r.g, r.arena.bytes, r.ctl_mem.bytes[0..@sizeOf(roci.Control)], r.layout.hash()) };
        errdefer r.nic.close();
        const mine = try r.nic.info();
        var table: [roci.world]rv.Info = undefined;
        try gatherOverLinks(fan, rank, world, std.mem.asBytes(&mine), std.mem.sliceAsBytes(&table));
        // the table is by rank; the plans name nodes: node g is rank order[g]
        var by_node: [roci.world]rv.Info = undefined;
        for (&by_node, order) |*x, rk| x.* = table[rk];
        try r.nic.connect(&by_node);
        // every node connected before any sends
        var one = [_]u8{1};
        var seen: [roci.world]u8 = undefined;
        try gatherOverLinks(fan, rank, world, &one, &seen);
        var o: roci.Order = undefined;
        for (&o, order) |*x, y| x.* = @intCast(y);
        const set: roci.Settings = .{ .nap = nap.wanted() and !in_card, .codes = true, .gathers = true, .park = layout_set == .host, .idle_park_us = envU32("TF_DS_ROCI_IDLE_PARK_US", 0), .housekeep_every = @max(1, envU32("TF_DS_ROCI_HOUSEKEEP", 16)), .backstop_us = backstopUs(envU32("TF_DS_ROCI_BACKSTOP_US", 100)), .wait_qp = in_card };
        r.t = try roci.Transport.init(gpa, r.g, o, ctl, r.ctl_mem.dev, r.nic.nic(), set);
        errdefer r.t.deinit();
        if (first) |f| roci.Transport.pair(f.t, r.t);
        var buf: [768]u8 = undefined;
        var w = std.Io.Writer.fixed(&buf);
        const rails_text = try std.fmt.allocPrint(gpa, "{s},{s}", .{ devices[0], devices[1] });
        defer gpa.free(rails_text);
        roci.startLine(&w, r.g, rails_text, gid, set, signal) catch {};
        std.log.info("{s}lane {d} of {d}: registration and exchange {d} ms, nap {}, arena {s}", .{ w.buffered(), r.lane, if (first) |f| @as(u8, @intCast((f.span[1] - f.span[0]) / total)) else lanes, (nowNs() - t0) / std.time.ns_per_ms, set.nap, "locked anonymous pages registered with cuMemHostRegister" });
        return r;
    }

    pub fn close(r: *Roci) void {
        r.t.deinit();
        r.nic.close();
        if (r.mlx) |*m| m.close();
        r.verbs.close();
        _ = r.d.api.cuMemFreeHost(r.ctl_mem.bytes.ptr);
        if (r.owns_arena) r.arena_mem.free(r.d);
        r.gpa.destroy(r);
    }

    /// The device address of a region of the arena: where a producer kernel writes it and a consumer kernel reads it.
    pub fn addr(r: *const Roci, reg: Region) u64 {
        return r.arena.dev + r.layout.at(reg);
    }

    /// The device addresses [lo, hi) of the arena: a GEMM whose y lies in it stores whole lines (mx.Kernels.useStaged).
    pub fn range(r: *const Roci) [2]u64 {
        return r.span;
    }

    /// The plan a chunk of `cap` rows takes (roci.Transport.shape_caps), else it has none.
        return error.NoPlanForRows;
    }

    /// The send of an event on `s`, queued behind the producer: one stream write.
    }

        if (r.t.set.wait_qp) return .{ .ticket = r.t.go(ticket), .door = r.t.doorbellAddr(), .cnt = r.sent_cnt.ptr };
    /// The consumers on `s` wait for the event's words from every source, on both rails.
    pub fn wait(r: *Roci, s: cuda.Stream, class: Class) !void {
        try r.t.wait(s, class);
    }

    /// The counter line of the work since the last one (the engine reads it at a synchronisation point, when every
    /// proxy has posted what the stream sent): the gates read it per pass.
    pub fn passLine(r: *Roci) void {
        r.passes += 1;
        const now = r.t.counters();
        var d = now;
        inline for (@typeInfo(roci.Counters).@"struct".field_names) |name| {
            if (comptime !std.mem.eql(u8, name, "post_ns_max") and !std.mem.eql(u8, name, "watchdog_max_age_ms")) @field(d, name) = @field(now, name) - @field(r.prev, name);
        }
        r.prev = now;
        var buf: [768]u8 = undefined;
        var label: [32]u8 = undefined;
        const l = std.fmt.bufPrint(&label, "lane {d} line {d}", .{ r.lane, r.passes }) catch "line";
        var w = std.Io.Writer.fixed(&buf);
        w.print("ROCI {s}: events {d} doorbells {d} stream_writes {d} wait_groups {d} data_writes {d} words {d} posts {d} bytes_out {d} wakes {d} backstop_wakes {d} post_ns_max {d} watchdog_max_age_ms {d} abort {d}", .{ l, d.events, d.doorbells, d.stream_writes, d.wait_groups, d.data_writes, d.words, d.posts, d.bytes_out, d.wakes, d.backstop_wakes, d.post_ns_max, d.watchdog_max_age_ms, @atomicLoad(u64, &r.t.ctl.abort.v, .monotonic) }) catch return;
        std.log.info("{s}", .{w.buffered()});
    }
};

/// `TF_DS_ROCI_SIGNAL`: `write` (unset: the baseline) or `atomic`; other text is refused by name (null).
/// The arrival signal a start runs and why (the start's line says it): `refuse` the atomic was asked for outright and the card cannot, `warn` it was
/// not asked for (the default) and the card cannot, so the WRITE runs; `why` is the card's reason.
pub const SignalMode = struct { signal: roci.Signal, refuse: bool = false, warn: bool = false, why: ?[]const u8 = null, note: []const u8 };

/// `TF_DS_ROCI_SIGNAL`: unset or atomic is the fetch-and-add (D79: the card executes it only after the data WRITEs are placed), write the second
/// WRITE. `atomic_why` is the device check's reason the card cannot do atomics (null: it can). Unset on a card that cannot falls back to the WRITE
/// with a warning; `atomic` set outright refuses there. Null for text no signal has. Every rank alike (the nodes compare it at connect).
pub fn signalMode(text: ?[]const u8, atomic_why: ?[]const u8) ?SignalMode {
    const sig = roci.signalFrom(text) orelse return null;
    const explicit = if (text) |t| std.mem.trim(u8, t, " \t").len != 0 else false;
    if (sig == .write) return .{ .signal = .write, .note = "write: the arrival word is a second RDMA WRITE after the data" };
    if (atomic_why) |why| {
        if (explicit) return .{ .signal = .atomic, .refuse = true, .why = why, .note = "" };
        return .{ .signal = .write, .warn = true, .why = why, .note = "unset (the default is the atomic) but the card cannot, so the arrival word is a second RDMA WRITE" };
    }
    return .{ .signal = .atomic, .note = if (explicit) "atomic: an RDMA ATOMIC fetch-and-add after the data (the guaranteed arrival)" else "unset: an RDMA ATOMIC fetch-and-add after the data (the default, the guaranteed arrival; =write for the WRITE)" };
}

test "wait mode: a lane opens and closes on either NIC (the whole of openLane is type-checked here, not only by an engine build)" {
    _ = &Roci.openLane;
    _ = &Roci.close;
}

/// Which transport runs and why (the start line says it): `on` the queue pairs wait in the card (rocinante_cd.zig, no proxy threads), else the
/// verbs queue pairs and the proxies; `refuse` the switch was asked for with 1 and cannot run (the start refuses by name).
pub const WaitMode = struct { on: bool, refuse: bool = false, note: []const u8 };

/// `TF_DS_ROCI_WAIT`: unset or 1 is the card's WAIT, the default; 0 (or any other text) is the verbs path with the proxies. Unset (the default)
/// falls back to the verbs path, with its reason in the line, where the card's WAIT cannot run (libmlx5 not loadable); 1 asked for outright
/// refuses there, by name. Either arrival signal runs in the card (the chain ends in the word's WRITE or in an ATOMIC fetch-and-add entry), so
/// `signal` only decides what the line says. Every rank alike (the nodes compare it at connect).
pub fn waitMode(text: ?[]const u8, signal: roci.Signal, mlx_ok: bool) WaitMode {
    const explicit = text != null;
    if (!prompt2d.switchDefaultOn(text)) return .{ .on = false, .note = "off (set to something but 1): the verbs queue pairs and the proxy threads" };
    if (!mlx_ok) {
        if (explicit) return .{ .on = true, .refuse = true, .note = "" };
        return .{ .on = false, .note = "unset (the default is the card's WAIT) but libmlx5.so.1 does not load: the verbs queue pairs and the proxy threads run" };
    }
    return switch (signal) {
        .write => .{ .on = true, .note = if (explicit) "1: the queue pairs wait in the card (no proxy threads)" else "unset: the queue pairs wait in the card (the default; =0 for the proxy threads)" },
        .atomic => .{ .on = true, .note = if (explicit) "1: the queue pairs wait in the card (no proxy threads); each chain ends in an ATOMIC fetch-and-add (TF_DS_ROCI_SIGNAL=atomic)" else "unset: the queue pairs wait in the card (the default; =0 for the proxy threads); each chain ends in an ATOMIC fetch-and-add (TF_DS_ROCI_SIGNAL=atomic)" },
    };
}

/// `TF_DS_ROCI_CPUS_B=a,b`: the second lane's two proxy cores (default 1,2: small cores that neither the decode rings' proxies nor the
/// threads use).
fn cpusFromEnvB() ?[roci.rails]u16 {
    const v = std.c.getenv("TF_DS_ROCI_CPUS_B") orelse return .{ 1, 2 };
    var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
    var out: [roci.rails]u16 = undefined;
    for (&out) |*c| c.* = std.fmt.parseInt(u16, it.next() orelse return null, 10) catch return null;
    return out;
}

/// `TF_DS_ROCI_CPUS=a,b`: the two rails' proxy cores; unset, the scheduler's choice.
fn cpusFromEnv() ?[roci.rails]u16 {
    const v = std.c.getenv("TF_DS_ROCI_CPUS") orelse return null;
    var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
    var out: [roci.rails]u16 = undefined;
    for (&out) |*c| c.* = std.fmt.parseInt(u16, it.next() orelse return null, 10) catch return null;
    return out;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// An all-gather of `mine` (one fixed length on every rank) over the engine's links: rank 0 reads each follower's in
/// rank order, then sends the table whole to each. `table` holds `world` entries.
pub fn gatherOverLinks(fan: *const prompt2d.Fan, rank: u32, world: u32, mine: []const u8, table: []u8) !void {
    const n = mine.len;
    std.debug.assert(table.len == n * world);
    var buf: [4096]u8 = undefined;
    if (n > buf.len) return error.InfoTooLong;
    if (rank == 0) {
        @memcpy(table[0..n], mine);
        for (1..world) |g| {
            const got = try fan.links[g - 1].?.recv(&buf);
            if (got.len != n) return error.BadInfo;
            @memcpy(table[g * n ..][0..n], got);
        }
        for (1..world) |g| try fan.links[g - 1].?.send(table);
    } else {
        const l = fan.links[0].?;
        try l.send(mine);
        var big: [4 * 4096]u8 = undefined;
        const got = try l.recv(&big);
        if (got.len != table.len) return error.BadInfo;
        @memcpy(table, got);
    }
}

test "TF_DS_ROCI_WAIT: unset and 1 are the card's WAIT, 0 is the proxies; unset falls back to the proxies with its reason where the WAIT cannot run, 1 refuses there" {
    // unset: the default
    const d = waitMode(null, .write, true);
    try std.testing.expect(d.on and !d.refuse and std.mem.indexOf(u8, d.note, "the default") != null);
    const one = waitMode("1", .write, true);
    try std.testing.expect(one.on and !one.refuse);
    // 0, empty or any other text: the verbs path
    for ([_][]const u8{ "0", "", "no" }) |t| {
        const m = waitMode(t, .write, true);
        try std.testing.expect(!m.on and !m.refuse);
    }
    // the atomic signal runs in the card too (the chain ends in an ATOMIC entry): unset and 1 are on, neither falls back nor refuses, and the
    // line says the signal; 0 is just the proxies
    const a = waitMode(null, .atomic, true);
    try std.testing.expect(a.on and !a.refuse and std.mem.indexOf(u8, a.note, "ATOMIC") != null);
    const a1 = waitMode("1", .atomic, true);
    try std.testing.expect(a1.on and !a1.refuse and std.mem.indexOf(u8, a1.note, "ATOMIC") != null);
    try std.testing.expect(!waitMode("0", .atomic, true).on);
    // the write signal's line does not mention one
    try std.testing.expect(std.mem.indexOf(u8, d.note, "ATOMIC") == null and std.mem.indexOf(u8, one.note, "ATOMIC") == null);
    // the libmlx5 fallback is kept for the atomic signal as for the write one: unset falls back with its reason, 1 refuses
    const am = waitMode(null, .atomic, false);
    try std.testing.expect(!am.on and !am.refuse and std.mem.indexOf(u8, am.note, "libmlx5") != null);
    try std.testing.expect(waitMode("1", .atomic, false).refuse);
    // no libmlx5: the same
    const n = waitMode(null, .write, false);
    try std.testing.expect(!n.on and !n.refuse and std.mem.indexOf(u8, n.note, "libmlx5") != null);
    try std.testing.expect(waitMode("1", .write, false).refuse);
}

test "TF_DS_ROCI_SIGNAL: unset is the atomic where the card can; unset on a card without atomics is the WRITE with a warning; atomic set outright refuses there; write is always the WRITE" {
    // a card that can: unset and atomic are the atomic, write the WRITE
    const d = signalMode(null, null).?;
    try std.testing.expect(d.signal == .atomic and !d.refuse and !d.warn and std.mem.indexOf(u8, d.note, "default") != null);
    try std.testing.expect(signalMode("", null).?.signal == .atomic);
    try std.testing.expect(signalMode("atomic", null).?.signal == .atomic);
    try std.testing.expect(signalMode("write", null).?.signal == .write);
    // a card that cannot (the reason from the device check)
    const why = "device mlx5_0 reports atomic_cap 0: it does no ATOMIC operations";
    const f = signalMode(null, why).?;
    try std.testing.expect(f.signal == .write and !f.refuse and f.warn and std.mem.eql(u8, f.why.?, why));
    const e = signalMode("atomic", why).?;
    try std.testing.expect(e.refuse and !f.refuse and std.mem.eql(u8, e.why.?, why));
    const w = signalMode("write", why).?;
    try std.testing.expect(w.signal == .write and !w.refuse and !w.warn);
    // text no signal has
    try std.testing.expect(signalMode("cas", null) == null);
}
