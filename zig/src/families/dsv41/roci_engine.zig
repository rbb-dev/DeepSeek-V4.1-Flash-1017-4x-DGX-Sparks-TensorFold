//! The Rocinante transport inside the engine: the arena in registered host memory, the queue pairs on both
//! rails, the start-up exchange over the engine's own links, and the sends and waits of the prompt path.

const std = @import("std");
const cuda = @import("cuda");
const fabric = @import("fabric");
const roci = @import("rocinante.zig");
const rv = @import("rocinante_verbs.zig");
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
pub const Roci = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    g: u8,
    layout: roci.Layout,
    arena: HostMem,
    ctl_mem: HostMem,
    verbs: fabric.verbs.Verbs,
    t: *roci.Transport,
    passes: u64 = 0,
    prev: roci.Counters = .{},

    pub fn openLane(gpa: std.mem.Allocator, d: *const cuda.Driver, rank: u32, node: u8, world: u32, order: [4]u32, fan: *const prompt2d.Fan, devices: [roci.rails][]const u8, gid: u8, lanes: u8, first: ?*Roci, layout_set: roci.LayoutSet) !*Roci {
        const pages: Pages = .mmap; // (locked anonymous pages registered with cuMemHostRegister: the arena the transport's kernels run at device speed on)
        if (world != roci.world) return error.NotFourRanks;
        const r = try gpa.create(Roci);
        errdefer gpa.destroy(r);
        r.ctl_mem = try HostMem.alloc(d, @sizeOf(roci.Control), host_portable | host_mapped);
        errdefer _ = d.api.cuMemFreeHost(r.ctl_mem.bytes.ptr);
        const ctl: *roci.Control = @ptrCast(r.ctl_mem.bytes.ptr);
        ctl.* = .{};
        r.verbs = try fabric.verbs.Verbs.open(null);
        errdefer r.verbs.close();
        const t0 = nowNs();
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
        r.t = try roci.Transport.init(gpa, r.g, o, ctl, r.ctl_mem.dev, r.nic.nic(), set);
        errdefer r.t.deinit();
        var buf: [768]u8 = undefined;
        var w = std.Io.Writer.fixed(&buf);
        const rails_text = try std.fmt.allocPrint(gpa, "{s},{s}", .{ devices[0], devices[1] });
        defer gpa.free(rails_text);
        return r;
    }

    pub fn close(r: *Roci) void {
        r.t.deinit();
        r.nic.close();
        r.verbs.close();
        _ = r.d.api.cuMemFreeHost(r.ctl_mem.bytes.ptr);
        r.gpa.destroy(r);
    }

    /// The device address of a region of the arena: where a producer kernel writes it and a consumer kernel reads it.
    pub fn addr(r: *const Roci, reg: Region) u64 {
        return r.arena.dev + r.layout.at(reg);
    }

    /// The device addresses [lo, hi) of the arena: a GEMM whose y lies in it stores whole lines (mx.Kernels.useStaged).
    pub fn range(r: *const Roci) [2]u64 {
    }

    /// The plan a chunk of `cap` rows takes (roci.Transport.shape_caps), else it has none.
        return error.NoPlanForRows;
    }

    /// The send of an event on `s`, queued behind the producer: one stream write.
    }

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
        var w = std.Io.Writer.fixed(&buf);
        w.print("ROCI {s}: events {d} doorbells {d} stream_writes {d} wait_groups {d} data_writes {d} words {d} posts {d} bytes_out {d} wakes {d} backstop_wakes {d} post_ns_max {d} watchdog_max_age_ms {d} abort {d}", .{ l, d.events, d.doorbells, d.stream_writes, d.wait_groups, d.data_writes, d.words, d.posts, d.bytes_out, d.wakes, d.backstop_wakes, d.post_ns_max, d.watchdog_max_age_ms, @atomicLoad(u64, &r.t.ctl.abort.v, .monotonic) }) catch return;
        std.log.info("{s}", .{w.buffered()});
    }
};

/// `TF_DS_ROCI_CPUS_B=a,b`: the second lane's two proxy cores (default 1,2: small cores that neither the decode rings' proxies nor the
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