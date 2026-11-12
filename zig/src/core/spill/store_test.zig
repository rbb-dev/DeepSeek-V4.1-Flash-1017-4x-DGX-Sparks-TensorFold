//! Port of wojo's GLM 0110 `tools/spill_blocks_check.py` and `tools/spill_blocks_mutants.sh` (PR #124) as Zig tests,
//! plus cases of its own for the native store: format, lineage, dedup, references, eviction, quotas, and a durability
//! fault campaign. A real store over a scratch directory, a fake "device" of rows per
//! source the test engine packs blocks from, and the engine's side of every call (stage, staged, captured,
//! readReady, readRelease) played by the test. Each check returns error.CheckFailed when a property fails; the
//! mutant test switches each authored fix back off (`mutants.zig`) and requires some check to fail.
const std = @import("std");
const c = std.c;
const format = @import("format.zig");
const fsio = @import("fsio.zig");
const slots = @import("slots.zig");
const store = @import("store.zig");
const mutants = @import("mutants.zig");
const Allocator = std.mem.Allocator;
const Store = store.Store;
const Digest = format.Digest;
const Lineage = format.Lineage;
const testing = std.testing;
const io = std.testing.io;

const block_tokens = 16;
const bpt = 256; // bytes a token on this rank
const block_bytes = block_tokens * bpt;
const max_tokens = 1024;
const small_max = 8192;
const secs = [_]store.SectionSpec{ .{ .name = "ring", .len = 300 }, .{ .name = "raw", .len = 200 } };

var quiet = false; // a mutant run: failures are expected, not printed
var serial: u32 = 0;

const CheckFailed = error.CheckFailed;

fn ensure(ok: bool, src: std.builtin.SourceLocation) !void {
    if (ok) return;
    if (!quiet) std.debug.print("     check failed at {s}:{d}\n", .{ src.file, src.line });
    return CheckFailed;
}

const Opts = struct {
    tag: []const u8,
    write_slots: usize = 8,
    read_slots: usize = 8,
    small_slots: usize = 4,
    restore_slots: usize = 2,
    cap: u64 = 64 << 20,
    floor: u64 = 0,
    turns: u32 = 4,
    quota: f64 = 0,
    max_reads: u32 = 2,
    keep_builds: u32 = 1,
    backlog: u64 = 1 << 30,
    free: ?u64 = null,
    no_direct: bool = false,
    io_priority: ?u16 = null,
};

const Ev = struct { ev: fsio.Event, name: [96]u8, len: usize };

/// The test's side of the hooks: event log, an action at a step, a crash point, a pwrite script, a disk-free value.
const Hk = struct {
    mutex: std.Io.Mutex = .init,
    events: std.ArrayList(Ev) = .empty,
    gpa: Allocator,
    action_step: ?fsio.Step = null,
    action_suffix: []const u8 = "",
    action: ?*const fn (h: *H) void = null,
    observe: ?*const fn (h: *H, step: fsio.Step, name: []const u8) void = null, // every step, before the rest
    crash_step: ?fsio.Step = null,
    crash_suffix: []const u8 = "",
    crash_skip: u32 = 0,
    nospace_writes: u32 = 0, // the next pwrites failing with ENOSPC
    short_intr: bool = false, // every pwrite short (at most 1000 bytes), and every other one EINTR first
    flip: bool = false,
    chown_fail: bool = false, // every chown of a file or folder the store makes fails (EPERM)
    free: ?u64 = null,
    h: *H = undefined,

    fn crash(ctx: ?*anyopaque, step: fsio.Step, name: []const u8) bool {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        if (std.c.getenv("SPILL_DEBUG") != null) std.debug.print("hook {t} {s}\n", .{ step, name });
        if (k.observe) |f| f(k.h, step, name);
        if (k.action) |f| if (k.action_step == step and std.mem.endsWith(u8, name, k.action_suffix)) {
            k.action = null;
            f(k.h);
        };
        if (k.crash_step) |cs| if (cs == step and std.mem.endsWith(u8, name, k.crash_suffix)) {
            if (k.crash_skip > 0) {
                k.crash_skip -= 1;
                return false;
            }
            k.crash_step = null;
            return true;
        };
        return false;
    }

    fn event(ctx: ?*anyopaque, ev: fsio.Event, name: []const u8) void {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        var e: Ev = .{ .ev = ev, .name = undefined, .len = @min(name.len, 96) };
        @memcpy(e.name[0..e.len], name[0..e.len]);
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        k.events.append(k.gpa, e) catch {};
    }

    fn pwrite(ctx: ?*anyopaque, fd: c.fd_t, buf: []const u8, off: u64) fsio.Error!usize {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        if (k.nospace_writes > 0) {
            k.nospace_writes -= 1;
            return error.NoSpace;
        }
        var want = buf.len;
        if (k.short_intr) {
            k.flip = !k.flip;
            if (k.flip) return error.Interrupted;
            if (off % fsio.alignment == 0 and want > 1000) want = 1000;
        }
        const n = fsio.testPwrite(fd, buf, want, off); // testPwrite can make a direct write short off the grid
        if (n < 0) return error.Io;
        return @intCast(n);
    }

    fn chown(ctx: ?*anyopaque, fd: c.fd_t, owner: fsio.Owner) fsio.Error!void {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        if (k.chown_fail) return error.Io;
        if (c.fchown(fd, owner.uid, owner.gid) != 0) return error.Io;
    }

    fn freeBytes(ctx: ?*anyopaque) u64 {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        return k.free orelse (1 << 40);
    }

    fn clear(k: *Hk) void {
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        k.events.clearRetainingCapacity();
    }
};

/// A store over a fresh scratch directory and the test's fake device.
const H = struct {
    gpa: Allocator,
    path: [256]u8 = undefined,
    root: [:0]const u8 = undefined,
    bufs: [4][]align(slots.alignment) u8 = @splat(&.{}),
    lists: [4][1]slots.Buffer = undefined,
    cfg: store.Config = undefined,
    s: *Store = undefined,
    rows: std.AutoHashMapUnmanaged(u64, []u8) = .empty,
    hk: Hk,
    opened: bool = false,

    fn init(gpa: Allocator, o: Opts) !*H {
        const h = try gpa.create(H);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .hk = .{ .gpa = gpa, .free = o.free } };
        h.hk.h = h;
        serial += 1;
        var tb: [64]u8 = undefined;
        const tag = try std.fmt.bufPrint(&tb, "{s}-{d}", .{ o.tag, serial });
        const r = try fsio.testRoot(&h.path, tag);
        h.root = r;
        const dfd = try fsio.freshDir(gpa, r);
        _ = c.close(dfd);
        const counts = [4]usize{ o.write_slots, o.read_slots, o.small_slots, o.restore_slots };
        const per = [4]usize{ block_bytes, block_bytes, small_max, small_max };
        for (0..4) |i| {
            h.bufs[i] = try gpa.alignedAlloc(u8, .fromByteUnits(slots.alignment), @max(counts[i] * per[i], 0));
            h.lists[i] = .{h.bufs[i]};
        }
        h.cfg = .{
            .root = h.root,
            .rank = 0,
            .compat = @splat(0x5a),
            .limits = .{ .max_tokens = max_tokens, .block_tokens = block_tokens, .block_bytes = block_bytes, .max_small_bytes = small_max },
            .cap_bytes = o.cap,
            .floor_bytes = o.floor,
            .quota = o.quota,
            .turns = o.turns,
            .min_tokens = 16,
            .backlog_bytes = o.backlog,
            .staging = .{ .write = &h.lists[0], .read = &h.lists[1], .small = &h.lists[2], .restore = &h.lists[3] },
            .readers = 3,
            .max_reads = o.max_reads,
            .keep_builds = o.keep_builds,
            .io_priority = o.io_priority,
            .hooks = .{ .ctx = &h.hk, .crash = Hk.crash, .event = Hk.event, .pwrite = Hk.pwrite, .free = Hk.freeBytes, .chown = Hk.chown, .no_direct = o.no_direct },
        };
        try h.open();
        return h;
    }

    fn open(h: *H) !void {
        h.s = try Store.open(h.gpa, io, h.cfg);
        h.opened = true;
    }

    fn closeStore(h: *H) void {
        if (h.opened) h.s.close();
        h.opened = false;
    }

    fn reopen(h: *H) !void {
        h.closeStore();
        try h.open();
    }

    fn deinit(h: *H) void {
        h.closeStore();
        fsio.dropDir(h.gpa, h.root);
        for (h.bufs) |b| h.gpa.free(b);
        var it = h.rows.valueIterator();
        while (it.next()) |r| h.gpa.free(r.*);
        h.rows.deinit(h.gpa);
        h.hk.events.deinit(h.gpa);
        h.gpa.destroy(h);
    }

    /// Source `id`'s rows on the fake device (deterministic bytes, made on first use).
    fn src(h: *H, id: u64) []u8 {
        const gop = h.rows.getOrPut(h.gpa, id) catch @panic("oom");
        if (!gop.found_existing) {
            const r = h.gpa.alloc(u8, max_tokens * bpt) catch @panic("oom");
            var prng = std.Random.DefaultPrng.init(id *% 0x9e3779b97f4a7c15 +% 1);
            prng.random().bytes(r);
            gop.value_ptr.* = r;
        }
        return gop.value_ptr.*;
    }

    fn dir(h: *H, which: enum { blocks, points }) c.fd_t {
        return if (which == .blocks) h.s.blocks_fd else h.s.points_fd;
    }
};

fn smallByte(source: u64, at: usize, i: usize) u8 {
    return @truncate(i *% 131 +% source *% 7 +% at);
}

const KO = struct {
    lineage: Lineage,
    source: u64,
    gen: u64 = 1,
    shared: bool = false,
    group: u64 = 0,
    media: []const format.Media = &.{},
    raw: ?[]const u32 = null, // with media: the raw ids (the same length as the keyed ones)
    capture: bool = true,
    chain: ?[]const Digest = null, // the caller's id chain
};

/// The engine at a keep: `keep`, then the small state and tail captured from the device, reported.
fn keepIds(h: *H, ids: []const u32, o: KO) store.KeepResult {
    const at = ids.len;
    const tail_tokens = at % block_tokens;
    const r = h.s.keep(.{
        .keyed = ids,
        .raw = if (o.raw) |r| r[0..ids.len] else null,
        .media = o.media,
        .lineage = o.lineage,
        .shared = o.shared,
        .group = o.group,
        .sections = &secs,
        .tail_bytes = tail_tokens * bpt,
        .source = .{ .id = o.source, .generation = o.gen },
        .chain = o.chain,
    });
    if (r == .queued) {
        const q = r.queued;
        for (q.sections) |sec| {
            for (0..@intCast(sec.len)) |i| q.small[@intCast(sec.offset + i)] = smallByte(o.source, at, i);
        }
        if (q.tail) |t| @memcpy(t, h.src(o.source)[(at - tail_tokens) * bpt .. at * bpt]);
        if (o.capture) h.s.captured(q.job, true);
    }
    return r;
}

/// The engine between rounds: every requested block packed from its source and reported.
fn pump(h: *H) void {
    var reqs: [16]store.StageReq = undefined;
    while (true) {
        const n = h.s.stage(&reqs);
        if (n == 0) return;
        for (reqs[0..n]) |r| {
            @memcpy(r.dst, h.src(r.source.id)[r.first * bpt ..][0..r.dst.len]);
            h.s.staged(r.ticket, true);
        }
    }
}

/// The engine until nothing waits on it: blocks staged as write slots come back, then a drain.
fn settle(h: *H) store.Store.Drain {
    var i: usize = 0;
    while (i < 5000) : (i += 1) {
        pump(h);
        if (h.s.pendingStage() == 0) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return h.s.drain(5 * std.time.ns_per_s);
}

/// Keep and write `ids` at once; the point's key once it is published.
fn write(h: *H, ids: []const u32, o: KO) !Digest {
    const r = keepIds(h, ids, o);
    try ensure(r == .queued, @src());
    const d = settle(h);
    try ensure(d.done, @src());
    const key = format.pointKey(ids, block_tokens, o.media);
    try ensure(h.s.presence(key), @src());
    return key;
}

/// A restore of `key` through the engine's side: rows (at * bpt bytes, gpa-owned) or null when it fails. Once it
/// commits, the engine's first round on the state runs and the tier says so (restored: its marker goes).
fn restore(h: *H, key: Digest) !?[]u8 {
    const got = try restoreCommit(h, key);
    if (got != null) h.s.restored(key);
    return got;
}

/// ... committed, but before the state served a round (no restored(key): its marker stays).
fn restoreCommit(h: *H, key: Digest) !?[]u8 {
    const id = h.s.beginRead(key) catch return null;
    const p = h.s.points.get(key) orelse return null;
    const at = p.at;
    const out = try h.gpa.alloc(u8, at * bpt);
    errdefer h.gpa.free(out);
    var i: u32 = 0;
    const total: u32 = @intCast(p.blocks.len);
    while (i < total) {
        if (!h.s.readAwait(id, i, 5 * std.time.ns_per_s, .{})) break;
        var rb: [4]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, null, &rb);
        if (got.status == .failed and got.n == 0) break;
        for (rb[0..got.n]) |b| {
            @memcpy(out[b.first * bpt ..][0..b.data.len], b.data);
            h.s.readRelease(b.slot);
            i += 1;
        }
    }
    var small_ok = false;
    for (0..2500) |_| {
        if (h.s.readSmallView(id) != null) {
            small_ok = true;
            break;
        }
        if (h.s.readFailure(id).err != null) break;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    if (i < total or !small_ok) {
        _ = h.s.endRead(id, .abort) catch {};
        h.gpa.free(out);
        return null;
    }
    var done = (h.s.endRead(id, .commit) catch {
        h.gpa.free(out);
        return null;
    }) orelse {
        h.gpa.free(out);
        return null;
    };
    done.lineage.deinit(h.gpa);
    return out;
}

/// A restore that fails, its failure forgotten the way every rank forgets it: true when it failed.
fn restoreFails(h: *H, key: Digest) !bool {
    const id = h.s.beginRead(key) catch return true;
    const total: u32 = @intCast((h.s.points.get(key) orelse return true).blocks.len);
    var failed = false;
    var i: u32 = 0;
    while (i < total) {
        _ = h.s.readAwait(id, i, 5 * std.time.ns_per_s, .{});
        var rb: [4]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, null, &rb);
        for (rb[0..got.n]) |b| h.s.readRelease(b.slot);
        i += @intCast(got.n);
        if (got.status == .failed) {
            failed = true;
            break;
        }
        if (got.n == 0 and got.status == .waiting and h.s.readFailure(id).err != null) {
            failed = true;
            break;
        }
    }
    const f = h.s.readFailure(id);
    if (f.err != null) failed = true;
    if (failed) {
        const bad = try h.gpa.dupe(Digest, f.bad);
        defer h.gpa.free(bad);
        const lost = try h.gpa.dupe(Digest, f.lost);
        defer h.gpa.free(lost);
        h.s.forget(key, bad, lost);
    }
    _ = h.s.endRead(id, if (failed) .abort else .cancel) catch {};
    return failed;
}

fn idsOf(gpa: Allocator, n: usize, seed: u64) ![]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const ids = try gpa.alloc(u32, n);
    for (ids) |*x| x.* = prng.random().intRangeAtMost(u32, 1, 150000);
    return ids;
}

fn cat(gpa: Allocator, a: []const u32, b: []const u32) ![]u32 {
    const out = try gpa.alloc(u32, a.len + b.len);
    @memcpy(out[0..a.len], a);
    @memcpy(out[a.len..], b);
    return out;
}

fn lin(n: u8) Lineage {
    return .{ .root = @splat(n) };
}

fn manifestShared(h: *H, key: Digest) ?bool {
    var nb: [80]u8 = undefined;
    const h64 = format.hex(&key);
    const name = std.fmt.bufPrintSentinel(&nb, "{s}.man", .{h64}, 0) catch return null;
    var file = fsio.readSmall(&h.s.writer, h.gpa, h.s.points_fd, name, 1 << 20) catch return null;
    defer file.deinit(h.gpa);
    var m = format.Manifest.parse(h.gpa, file.bytes(), h.cfg.compat, h.cfg.limits) catch return null;
    defer m.deinit(h.gpa);
    return m.flags.shared;
}

fn fileExists(dirfd: c.fd_t, name: []const u8) bool {
    var nb: [128]u8 = undefined;
    const z = std.fmt.bufPrintSentinel(&nb, "{s}", .{name}, 0) catch return false;
    _ = fsio.statAt(dirfd, z) catch return false;
    return true;
}

fn blockFile(buf: []u8, name: Digest) []const u8 {
    const x = format.hex(&name);
    return std.fmt.bufPrint(buf, "{s}.blk", .{x}) catch unreachable;
}

fn pointFile(buf: []u8, key: Digest, suffix: []const u8) []const u8 {
    const x = format.hex(&key);
    return std.fmt.bufPrint(buf, "{s}{s}", .{ x, suffix }) catch unreachable;
}

fn countFiles(gpa: Allocator, dirfd: c.fd_t, temps_only: bool) usize {
    var names = fsio.list(gpa, dirfd) catch return 0;
    defer fsio.freeList(gpa, &names);
    var n: usize = 0;
    for (names.items) |x| {
        if (temps_only and x[0] != '.') continue;
        n += 1;
    }
    return n;
}

fn waitUntil(cond: anytype, ctx: anytype, ms: u32) bool {
    var i: u32 = 0;
    while (i < ms) : (i += 2) {
        if (cond(ctx)) return true;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    return cond(ctx);
}

// -- the checks ------------------------------------------------------------------------------------------------------

/// the block a kept state's end cuts holds the rows of the keep, not what the stream writes after
/// it; it is captured with the small state, never staged later from the source.
fn checkTail(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "tail" });
    defer h.deinit();
    const n = 2 * block_tokens + 6;
    const ids = try idsOf(gpa, n, 1);
    defer gpa.free(ids);
    const kept = try gpa.dupe(u8, h.src(1)[0 .. n * bpt]);
    defer gpa.free(kept);
    const r = keepIds(h, ids, .{ .lineage = lin(1), .source = 1 });
    try ensure(r == .queued, @src());
    const had_tail = r.queued.tail != null;
    // the stream goes on decoding in its extent past the keep: the tail block's rows change
    var prng = std.Random.DefaultPrng.init(9);
    prng.random().bytes(h.src(1)[2 * block_tokens * bpt .. (n + 8) * bpt]);
    try ensure(settle(h).done, @src());
    const got = (try restore(h, format.pointKey(ids, block_tokens, &.{}))) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, kept), @src());
    try ensure(had_tail and h.s.counts().staged_blocks == 2, @src());
}

/// Every read of a point is timed: each block's open (with its size check), read, CRC and close, and the read's wall
/// time from its begin to its last block read, once a read.
fn checkReadTimes(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "read times" });
    defer h.deinit();
    const n = 2 * block_tokens;
    const ids = try idsOf(gpa, n, 7);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(1), .source = 7 });
    const before = h.s.counts();
    try ensure(before.block_read.count == 0 and before.read_wall.count == 0, @src());
    const got = (try restore(h, key)) orelse return CheckFailed;
    defer gpa.free(got);
    const t = h.s.counts();
    const blocks = h.s.points.get(key).?.blocks.len;
    try ensure(blocks == 2, @src());
    try ensure(t.block_open.count == blocks and t.block_read.count == blocks, @src());
    try ensure(t.block_crc.count == blocks and t.block_close.count == blocks, @src());
    try ensure(t.read_wall.count == 1, @src());
    try ensure(t.block_read.sum_ns > 0 and t.block_crc.sum_ns > 0 and t.read_wall.sum_ns > 0, @src());
    // the wall holds its blocks' reads (two readers at once at most here: never less than the longest one)
    try ensure(t.read_wall.max_ns >= t.block_read.max_ns, @src());
}

/// The reader reads; the engine copies, only as far as it is told (LOADPART's boundary), the rest after.
fn checkUpto(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "upto" });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 2);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(2), .source = 2 });
    const id = try h.s.beginRead(key);
    for (0..3) |i| try ensure(h.s.readAwait(id, @intCast(i), 5 * std.time.ns_per_s, .{}), @src()); // (readers finish in any order)
    var rb: [8]store.ReadyBlock = undefined;
    const a = h.s.readReady(id, 2, &rb);
    try ensure(a.n == 2 and a.status == .done, @src());
    for (rb[0..a.n]) |b| h.s.readRelease(b.slot);
    const b2 = h.s.readReady(id, null, &rb);
    try ensure(b2.n == 1 and b2.status == .done and rb[0].index == 2, @src());
    h.s.readRelease(rb[0].slot);
    try ensure(waitUntil(struct {
        fn f(x: struct { *H, store.ReadId }) bool {
            return x[0].s.readSmallView(x[1]) != null;
        }
    }.f, .{ h, id }, 5000), @src());
    const v = h.s.readSmallView(id).?;
    try ensure(v.sections.len == 2 and std.mem.eql(u8, v.sections[0].name.slice(), "ring"), @src());
    for (v.sections) |sec| for (0..@intCast(sec.len)) |i| {
        try ensure(v.bytes[@intCast(sec.offset + i)] == smallByte(2, 3 * block_tokens, i), @src());
    };
    var done = (try h.s.endRead(id, .commit)) orelse return CheckFailed;
    defer done.lineage.deinit(gpa);
    try ensure(done.at == 3 * block_tokens and done.lineage.eql(lin(2)), @src());
}

/// Last use survives a restart (the manifest's time): the least recently used conversation is still evicted first.
fn checkLastUse(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lastuse" });
    defer h.deinit();
    const a = try idsOf(gpa, block_tokens, 2100);
    defer gpa.free(a);
    const b = try idsOf(gpa, block_tokens, 2101);
    defer gpa.free(b);
    const ka = try write(h, a, .{ .lineage = lin(21), .source = 21 });
    const kb = try write(h, b, .{ .lineage = lin(22), .source = 22 });
    std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
    h.s.touch(ka);
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try h.reopen();
    const order = h.s.evictionOrder() catch return CheckFailed;
    defer gpa.free(order);
    try ensure(order.len == 2 and std.mem.eql(u8, &order[0].key, &kb) and std.mem.eql(u8, &order[1].key, &ka), @src());
}

/// Keeps the store refuses say why and hold nothing: too short, invalid, out of staging, past the backlog.
fn checkRefusals(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "refuse", .small_slots = 1, .backlog = 3 * block_bytes + 2048 });
    defer h.deinit();
    const ids = try idsOf(gpa, 2 * block_tokens, 2200);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids[0..8], .{ .lineage = lin(1), .source = 1 }) == .short, @src());
    const bad = h.s.keep(.{ .keyed = ids, .lineage = lin(1), .sections = &.{.{ .name = "a-section-name-longer-than-32-bytes", .len = 8 }}, .source = .{ .id = 1, .generation = 1 } });
    try ensure(bad == .invalid, @src());
    const tail = h.s.keep(.{ .keyed = ids[0 .. block_tokens + 3], .lineage = lin(1), .sections = &secs, .tail_bytes = 0, .source = .{ .id = 1, .generation = 1 } });
    try ensure(tail == .invalid, @src());
    const r1 = keepIds(h, ids, .{ .lineage = lin(1), .source = 1, .capture = false });
    try ensure(r1 == .queued, @src());
    const other = try idsOf(gpa, block_tokens, 2201);
    defer gpa.free(other);
    try ensure(keepIds(h, other, .{ .lineage = lin(2), .source = 2 }) == .no_staging, @src());
    h.s.captured(r1.queued.job, true);
    try ensure(settle(h).done, @src());
    try ensure(keepIds(h, other, .{ .lineage = lin(2), .source = 2 }) == .queued, @src());
    try ensure(settle(h).done, @src());
    const big = try idsOf(gpa, 4 * block_tokens, 2202);
    defer gpa.free(big);
    try ensure(keepIds(h, big, .{ .lineage = lin(3), .source = 3 }) == .backlog, @src());
    const ct = h.s.counts();
    try ensure(ct.short == 1 and ct.invalid == 2 and ct.no_staging == 1 and ct.backlog_refused == 1 and ct.staging_bytes == 0, @src());
    try ensure(ct.committed == 2 and ct.points == 2, @src());
}

/// Shared flag 3: the same ids kept again as a shared prefix make the stored point shared, its manifest rewritten
/// through a temporary name, fsynced, renamed, then the folder fsynced; a restart reads it shared.
fn checkSharedUpgrade(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "upgrade" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 51);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(51), .source = 51 });
    h.hk.clear();
    try ensure(keepIds(h, ids, .{ .lineage = lin(51), .source = 51, .shared = true }) == .stored, @src());
    try ensure(h.s.points.get(key).?.shared, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure(manifestShared(h, key) == true, @src());
    // the rewrite's own events: written to a temporary name, fsynced, pages dropped, renamed, the folder fsynced
    h.hk.mutex.lockUncancelable(io);
    var saw: [8]fsio.Event = undefined;
    var k: usize = 0;
    for (h.hk.events.items) |e| if (std.mem.endsWith(u8, e.name[0..e.len], ".man") or std.mem.eql(u8, e.name[0..e.len], "points")) {
        if (k < saw.len) saw[k] = e.ev;
        k += 1;
    };
    h.hk.mutex.unlock(io);
    try ensure(k == 6 and std.mem.eql(fsio.Event, saw[0..6], &.{ .write, .chown, .fsync, .drop_pages, .rename, .dir_fsync }), @src());
    try h.reopen();
    const p = h.s.points.get(key) orelse return CheckFailed;
    try ensure(p.shared, @src());
}

/// ... also when they were still waiting to be written.
fn checkSharedWaiting(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "waiting" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 52);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids, .{ .lineage = lin(52), .source = 52 }) == .queued, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(52), .source = 52, .shared = true }) == .stored, @src());
    try ensure(settle(h).done, @src());
    const key = format.pointKey(ids, block_tokens, &.{});
    try ensure((h.s.points.get(key) orelse return CheckFailed).shared and manifestShared(h, key) == true, @src());
}

fn markWhileWriting(h: *H) void {
    const ids = h.rows.get(0xdead).?; // the ids stashed as bytes
    const s: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, ids));
    _ = keepIds(h, s, .{ .lineage = lin(53), .source = 53, .shared = true });
}

/// ... and when kept again as shared while its own files were being written: its manifest is rewritten shared.
fn checkSharedDuringWrite(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "during" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 53);
    defer gpa.free(ids);
    try h.rows.put(gpa, 0xdead, try gpa.dupe(u8, std.mem.sliceAsBytes(ids)));
    h.hk.action_step = .written;
    h.hk.action_suffix = ".ids";
    h.hk.action = markWhileWriting;
    const key = try write(h, ids, .{ .lineage = lin(53), .source = 53 });
    try ensure(h.hk.action == null, @src()); // (it ran)
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure((h.s.points.get(key) orelse return CheckFailed).shared and manifestShared(h, key) == true, @src());
}

var drop_key: Digest = undefined;
var drop_then_stop = false;
fn dropDuring(h: *H) void {
    h.s.forget(drop_key, &.{}, &.{});
    if (drop_then_stop) { // the process ends with this rewrite: the deletions the forget queued never run
        h.s.mutex.lockUncancelable(io);
        h.s.crashed = true;
        h.s.mutex.unlock(io);
    }
}

/// Whether the writer runs no task and has none queued.
fn writerIdle(h: *H) bool {
    h.s.mutex.lockUncancelable(io);
    defer h.s.mutex.unlock(io);
    return h.s.busy == 0 and h.s.task_head >= h.s.tasks.items.len;
}

/// A point dropped while its manifest is rewritten does not get the manifest back (no resurrection), also when the
/// process ends right after that rewrite, before the deletions its drop queued ran: the rewrite removes what it wrote
/// back itself, so the next open does not recover the point.
fn checkDropDuringRewrite(gpa: Allocator) !void {
    for ([_]bool{ false, true }) |stop| {
        const h = try H.init(gpa, .{ .tag = "resurrect" });
        defer h.deinit();
        const ids = try idsOf(gpa, block_tokens + 5, 54);
        defer gpa.free(ids);
        const key = try write(h, ids, .{ .lineage = lin(54), .source = 54 });
        drop_key = key;
        drop_then_stop = stop;
        defer drop_then_stop = false;
        h.hk.action_step = .written;
        h.hk.action_suffix = ".man";
        h.hk.action = dropDuring;
        try ensure(keepIds(h, ids, .{ .lineage = lin(54), .source = 54, .shared = true }) == .stored, @src());
        if (stop) try ensure(waitUntil(writerIdle, h, 3000), @src()) else _ = h.s.drain(5 * std.time.ns_per_s);
        try ensure(h.hk.action == null and !h.s.points.contains(key), @src());
        var nb: [96]u8 = undefined;
        try ensure(!fileExists(h.s.points_fd, pointFile(&nb, key, ".man")), @src());
        if (stop) {
            try h.reopen();
            try ensure(!h.s.presence(key) and h.s.counts().recovered_points == 0, @src());
            try ensure(countFiles(gpa, h.s.points_fd, false) == 0 and countFiles(gpa, h.s.blocks_fd, false) == 0, @src());
        }
    }
}

fn slowRewrite(_: *H) void {
    std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
}

/// A drain (a clean stop) waits for a queued metadata rewrite.
fn checkDrainWaitsRewrite(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "drainrw" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 55);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(55), .source = 55 });
    h.hk.action_step = .written;
    h.hk.action_suffix = ".man";
    h.hk.action = slowRewrite;
    try ensure(keepIds(h, ids, .{ .lineage = lin(55), .source = 55, .shared = true }) == .stored, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure(manifestShared(h, key) == true, @src());
}

var stuck_in_hook = std.atomic.Value(bool).init(false);
fn stuckRewrite(_: *H) void {
    stuck_in_hook.store(true, .release);
    std.Io.sleep(io, .fromMilliseconds(1000), .awake) catch {};
}

fn mtimeOf(h: *H, key: Digest) !i96 {
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    return (try fsio.statAt(h.s.points_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, key, ".man")}, 0))).mtime_ns;
}

fn exited(h: *H) bool {
    h.s.mutex.lockUncancelable(io);
    defer h.s.mutex.unlock(io);
    return h.s.writer_exited and h.s.readers_exited == h.s.reader_threads.items.len;
}

/// A stop never waits on a stuck disk: with the writer held in a hook inside a shared rewrite of P's manifest and
/// touches of Q queued behind it, closeWithin returns by its deadline, not clean, naming what it left; once the hook
/// returns the writer exits without running the touches (counted). The store is left allocated (its threads may still
/// run): the check uses the page allocator and keeps H until the threads report their exit.
fn checkCloseWithin(_: Allocator) !void {
    const gpa = std.heap.page_allocator;
    const h = try H.init(gpa, .{ .tag = "closewithin" });
    defer h.deinit();
    const ip = try idsOf(gpa, block_tokens + 5, 3000);
    defer gpa.free(ip);
    const iq = try idsOf(gpa, block_tokens + 5, 3001);
    defer gpa.free(iq);
    _ = try write(h, ip, .{ .lineage = lin(31), .source = 300 });
    const kq = try write(h, iq, .{ .lineage = lin(32), .source = 301 });
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    const q_mtime = try mtimeOf(h, kq);
    stuck_in_hook.store(false, .release);
    h.hk.action_step = .written;
    h.hk.action_suffix = ".man";
    h.hk.action = stuckRewrite;
    try ensure(keepIds(h, ip, .{ .lineage = lin(31), .source = 300, .shared = true }) == .stored, @src());
    try ensure(waitUntil(struct {
        fn f(_: void) bool {
            return stuck_in_hook.load(.acquire);
        }
    }.f, {}, 3000), @src());
    for (0..3) |_| h.s.touch(kq);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const r = h.s.closeWithin(100 * std.time.ns_per_ms);
    const took = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    h.opened = false; // closed when clean; abandoned otherwise: left allocated, never closed again
    try ensure(!r.clean and r.dropped >= 3 and took < 500 * std.time.ns_per_ms, @src());
    try ensure(waitUntil(exited, h, 3000), @src());
    try ensure(try mtimeOf(h, kq) == q_mtime, @src());
    try ensure(h.s.counts().close_dropped >= 3, @src());
}

/// ... and with a responsive writer it waits for what is queued: the touch and the shared rewrite are durable
/// after a reopen, and the close is clean.
fn checkCloseRunsMetadata(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "closemeta" });
    defer h.deinit();
    const ip = try idsOf(gpa, block_tokens + 5, 3002);
    defer gpa.free(ip);
    const iq = try idsOf(gpa, block_tokens + 5, 3003);
    defer gpa.free(iq);
    const kp = try write(h, ip, .{ .lineage = lin(33), .source = 302 });
    const kq = try write(h, iq, .{ .lineage = lin(34), .source = 303 });
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure(keepIds(h, ip, .{ .lineage = lin(33), .source = 302, .shared = true }) == .stored, @src());
    h.s.touch(kq);
    const q_used = (h.s.points.get(kq) orelse return CheckFailed).used;
    const r = h.s.closeWithin(5 * std.time.ns_per_s);
    h.opened = false;
    try ensure(r.clean and r.dropped == 0, @src());
    try h.open();
    try ensure((h.s.points.get(kp) orelse return CheckFailed).shared and manifestShared(h, kp) == true, @src());
    try ensure((h.s.points.get(kq) orelse return CheckFailed).used == q_used, @src());
}

/// A (2 blocks), B (other ids, the same rows: its blocks links to A's) and C (A's ids and a block more, A's lineage:
/// it names A's blocks).
fn linked(h: *H) ![3]Digest {
    const gpa = h.gpa;
    const ia = try idsOf(gpa, 2 * block_tokens, 61);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, 2 * block_tokens, 62);
    defer gpa.free(ib);
    const more = try idsOf(gpa, block_tokens, 63);
    defer gpa.free(more);
    const ic = try cat(gpa, ia, more);
    defer gpa.free(ic);
    const ka = try write(h, ia, .{ .lineage = lin(1), .source = 60 });
    const kb = try write(h, ib, .{ .lineage = lin(2), .source = 60 });
    const kc = try write(h, ic, .{ .lineage = lin(1), .source = 60 });
    return .{ ka, kb, kc };
}

/// a read failing on a missing block file drops that name and the points naming it, not the points
/// whose blocks are links to the same bytes (they still read back).
fn checkForgetLost(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lost" });
    defer h.deinit();
    const k = try linked(h);
    const pa = h.s.points.get(k[0]) orelse return CheckFailed;
    const pb = h.s.points.get(k[1]) orelse return CheckFailed;
    try ensure(h.s.counts().dedup_blocks == 2, @src());
    for (0..2) |i| try ensure(h.s.have.get(pa.blocks[i].name).?.ino == h.s.have.get(pb.blocks[i].name).?.ino, @src());
    const pb_names = [2]Digest{ pb.blocks[0].name, pb.blocks[1].name };
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    _ = c.unlinkat(h.s.blocks_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{blockFile(&nb, pa.blocks[0].name)}, 0), 0);
    try ensure(try restoreFails(h, k[0]), @src());
    try ensure(!h.s.points.contains(k[0]) and !h.s.points.contains(k[2]), @src());
    try ensure(h.s.points.contains(k[1]), @src());
    _ = h.s.drain(5 * std.time.ns_per_s);
    for (pb_names) |n| try ensure(fileExists(h.s.blocks_fd, blockFile(&nb, n)), @src());
    const got = (try restore(h, k[1])) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(60)[0 .. 2 * block_tokens * bpt]), @src());
}

/// ... one failing on damaged bytes drops every name of them (A, B and C).
fn checkForgetDamaged(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "damaged" });
    defer h.deinit();
    const k = try linked(h);
    const pa = h.s.points.get(k[0]) orelse return CheckFailed;
    const pb = h.s.points.get(k[1]) orelse return CheckFailed;
    const b0 = pb.blocks[0].name;
    // C names A's blocks and is being read when A's read fails: it goes too (unlisted, freed when its read ends)
    const rc = try h.s.beginRead(k[2]);
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&zb, "{s}", .{blockFile(&nb, pa.blocks[0].name)}, 0);
    const fd = c.openat(h.s.blocks_fd, name, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    try ensure(fd >= 0, @src());
    const junk: [64]u8 = @splat(0xff);
    _ = c.pwrite(fd, &junk, junk.len, 0);
    _ = c.close(fd);
    try ensure(try restoreFails(h, k[0]), @src());
    const gone = !h.s.points.contains(k[0]) and !h.s.points.contains(k[1]) and !h.s.points.contains(k[2]);
    _ = h.s.endRead(rc, .cancel) catch {};
    try ensure(gone, @src());
    try ensure(h.s.counts().points == 0, @src());
    _ = h.s.drain(5 * std.time.ns_per_s);
    var nb2: [96]u8 = undefined;
    try ensure(!fileExists(h.s.blocks_fd, blockFile(&nb2, b0)), @src());
}

/// A point's small state that cannot be read for an I/O reason (here: no permission) is not damage, the point stays
/// and no CRC error is counted; one of the wrong size is damage, counted in size_errors.
fn checkSmallReadErrors(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "smallerr" });
    defer h.deinit();
    const ids = try idsOf(gpa, 2 * block_tokens, 1800);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(80), .source = 80 });
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const st = try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, key, ".st")}, 0);
    try ensure(c.fchmodat(h.s.points_fd, st, 0, 0) == 0, @src()); // (an I/O-class error, as EIO or EMFILE would be)
    const before = h.s.counts();
    const id = try h.s.beginRead(key);
    var why: ?store.ReadError = null;
    for (0..2500) |_| {
        why = h.s.readFailure(id).err;
        if (why != null) break;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    _ = h.s.endRead(id, .abort) catch {};
    _ = c.fchmodat(h.s.points_fd, st, 0o644, 0);
    try ensure(why == .io, @src());
    try ensure(h.s.counts().crc_errors == before.crc_errors, @src());
    try ensure(h.s.points.contains(key), @src());
    // the wrong size: damage, named as such
    const fd = c.openat(h.s.points_fd, st, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    try ensure(fd >= 0, @src());
    _ = c.ftruncate(fd, 7);
    _ = c.close(fd);
    const id2 = try h.s.beginRead(key);
    why = null;
    for (0..2500) |_| {
        why = h.s.readFailure(id2).err;
        if (why != null) break;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    _ = h.s.endRead(id2, .abort) catch {};
    try ensure(why == .small, @src());
    try ensure(h.s.counts().size_errors == before.size_errors + 1, @src());
}

/// A point's small files (.ids, .st, .man) each have their pages dropped after their fsync.
fn checkPages(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "pages" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 7, 71);
    defer gpa.free(ids);
    h.hk.clear();
    const key = try write(h, ids, .{ .lineage = lin(71), .source = 71 });
    var nb: [96]u8 = undefined;
    for ([_][]const u8{ ".ids", ".st", ".man" }) |suffix| {
        const want = pointFile(&nb, key, suffix);
        var seq: [4]fsio.Event = undefined;
        var k: usize = 0;
        h.hk.mutex.lockUncancelable(io);
        for (h.hk.events.items) |e| if (std.mem.eql(u8, e.name[0..e.len], want) and (e.ev == .fsync or e.ev == .drop_pages)) {
            if (k < seq.len) seq[k] = e.ev;
            k += 1;
        };
        h.hk.mutex.unlock(io);
        try ensure(k == 2 and seq[0] == .fsync and seq[1] == .drop_pages, @src());
    }
}

/// Pages of file `name` in `dirfd` in the page cache now: a shared read-only mapping without populate, then mincore
/// (which reports the page cache for a file mapping, and reads nothing).
fn residentPages(dirfd: c.fd_t, name: []const u8) !usize {
    var zb: [128]u8 = undefined;
    const z = try std.fmt.bufPrintSentinel(&zb, "{s}", .{name}, 0);
    const fd = c.openat(dirfd, z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    if (fd < 0) return error.Unexpected;
    defer _ = c.close(fd);
    const size: usize = @intCast((try fsio.statFd(fd)).size);
    if (size == 0) return 0;
    const map = c.mmap(null, size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
    if (map == c.MAP_FAILED) return error.Unexpected;
    const base: *align(std.heap.page_size_min) anyopaque = @alignCast(map);
    defer _ = c.munmap(base, size);
    var vec: [4096]u8 = undefined;
    const pages = std.math.divCeil(usize, size, std.heap.pageSize()) catch unreachable;
    if (pages > vec.len) return error.Unexpected;
    if (c.mincore(base, size, &vec) != 0) return error.Unexpected;
    var n: usize = 0;
    for (vec[0..pages]) |v| n += v & 1;
    return n;
}

/// Whether this file system shows page-cache residency, which the residency checks need: a file written, fsynced and
/// dropped shows no page; read back buffered, it shows its pages. tmpfs (always resident) does not.
fn residencyShows(h: *H) !bool {
    const name = ".residency-probe";
    const fd = c.openat(h.s.points_fd, name, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    if (fd < 0) return error.Unexpected;
    defer {
        _ = c.close(fd);
        _ = c.unlinkat(h.s.points_fd, name, 0);
    }
    var buf: [3 * 4096]u8 = undefined;
    for (&buf, 0..) |*b, i| b.* = @truncate(i *% 7);
    if (c.pwrite(fd, &buf, buf.len, 0) != @as(isize, buf.len) or c.fsync(fd) != 0) return error.Unexpected;
    const no_hooks: fsio.Hooks = .{};
    fsio.dropPages(&no_hooks, fd, name);
    if (try residentPages(h.s.points_fd, name) != 0) return false;
    if (c.pread(fd, &buf, buf.len, 0) != @as(isize, buf.len)) return error.Unexpected;
    return try residentPages(h.s.points_fd, name) > 0;
}

/// Pages of every file in `dirfd` that the page cache holds now (none expected: the store reads past it).
fn residentAll(gpa: Allocator, dirfd: c.fd_t, suffixes: []const []const u8) !usize {
    var names = try fsio.list(gpa, dirfd);
    defer fsio.freeList(gpa, &names);
    var n: usize = 0;
    for (names.items) |x| for (suffixes) |sfx| if (std.mem.endsWith(u8, x, sfx)) {
        n += try residentPages(dirfd, x);
    };
    return n;
}

/// The open's scan reads every .man and .ids past the page cache: after a reopen of a text point, a tailed point and a
/// picture point, none of their pages is resident.
fn checkSmallPages(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "smallpages", .turns = 0 });
    defer h.deinit();
    if (!try residencyShows(h)) return error.SkipZigTest;
    try writeThree(h);
    try h.reopen();
    try ensure(h.s.counts().recovered_points == 3, @src());
    try ensure(try residentAll(gpa, h.s.points_fd, &.{ ".man", ".ids" }) == 0, @src());
}

/// A text point (whole blocks), a tailed one and one with a picture (its raw ids beside the keyed ones).
fn writeThree(h: *H) !void {
    const gpa = h.gpa;
    const a = try idsOf(gpa, 2 * block_tokens, 2700);
    defer gpa.free(a);
    _ = try write(h, a, .{ .lineage = lin(27), .source = 270 });
    const b = try idsOf(gpa, 2 * block_tokens + 5, 2701);
    defer gpa.free(b);
    _ = try write(h, b, .{ .lineage = lin(28), .source = 271 });
    const m = [_]format.Media{.{ .start = 3, .len = 4, .digest = @splat(7) }};
    const keyed = try idsOf(gpa, 2 * block_tokens, 2702);
    defer gpa.free(keyed);
    const raw = try gpa.dupe(u32, keyed);
    defer gpa.free(raw);
    for (3..7) |j| {
        keyed[j] = format.pictureId(&m[0].digest, @intCast(j - 3));
        raw[j] = 99;
    }
    _ = try write(h, keyed, .{ .lineage = lin(29), .source = 272, .media = &m, .raw = raw });
}

/// A file system that refuses O_DIRECT reads buffered, counts the refusal (direct_refused) and drops each file's pages
/// after the read: after a reopen and a restore, no page of any file stays, and the restored rows are the kept ones.
fn checkReadFallbacks(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "readfallback", .turns = 0, .no_direct = true });
    defer h.deinit();
    if (!try residencyShows(h)) return error.SkipZigTest;
    const ids = try idsOf(gpa, 3 * block_tokens + 5, 2710);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(30), .source = 273 });
    try h.reopen();
    const got = (try restore(h, key)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(273)[0 .. ids.len * bpt]), @src());
    try ensure(h.s.counts().direct_refused >= 1, @src());
    try ensure(try residentAll(gpa, h.s.points_fd, &.{ ".man", ".ids", ".st" }) == 0, @src());
    try ensure(try residentAll(gpa, h.s.blocks_fd, &.{".blk"}) == 0, @src());
}

/// Publication order: every block renamed and its folder fsynced before the point's files; the ids and small state
/// renamed and the points folder fsynced before the manifest, which is renamed last.
fn checkPublishOrder(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "order" });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 72);
    defer gpa.free(ids);
    h.hk.clear();
    const key = try write(h, ids, .{ .lineage = lin(72), .source = 72 });
    var nb: [96]u8 = undefined;
    const man = pointFile(&nb, key, ".man");
    var last_blk: ?usize = null;
    var blocks_sync: ?usize = null;
    var ids_ren: ?usize = null;
    var st_ren: ?usize = null;
    var points_sync: ?usize = null;
    var man_ren: ?usize = null;
    h.hk.mutex.lockUncancelable(io);
    for (h.hk.events.items, 0..) |e, i| {
        const n = e.name[0..e.len];
        if (e.ev == .rename and std.mem.endsWith(u8, n, ".blk")) last_blk = i;
        if (e.ev == .dir_fsync and std.mem.eql(u8, n, "blocks") and blocks_sync == null) blocks_sync = i;
        if (e.ev == .rename and std.mem.endsWith(u8, n, ".ids")) ids_ren = i;
        if (e.ev == .rename and std.mem.endsWith(u8, n, ".st")) st_ren = i;
        if (e.ev == .dir_fsync and std.mem.eql(u8, n, "points") and points_sync == null and st_ren != null) points_sync = i;
        if (e.ev == .rename and std.mem.eql(u8, n, man)) man_ren = i;
    }
    h.hk.mutex.unlock(io);
    const all = last_blk != null and blocks_sync != null and ids_ren != null and st_ren != null and points_sync != null and man_ren != null;
    try ensure(all, @src());
    try ensure(last_blk.? < blocks_sync.? and blocks_sync.? < ids_ren.? and ids_ren.? < st_ren.? and st_ren.? < points_sync.? and points_sync.? < man_ren.?, @src());
}

const Pending = struct {
    h: *H,
    held: std.ArrayList(slots.Slot) = .empty,

    fn recycle(ctx: ?*anyopaque) void {
        const p: *Pending = @ptrCast(@alignCast(ctx.?));
        for (p.held.items) |s| p.h.s.readRelease(s);
        p.held.clearRetainingCapacity();
    }
};

/// PR 124's hang: a follower whose peer failed early (LOADED with nothing copied before) and more blocks left than
/// read slots. Its copies complete only when it recycles; its wait for a block gives those slots back each time.
fn checkHang(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "hang", .read_slots = 4 });
    defer h.deinit();
    const n = 12 * block_tokens;
    const ids = try idsOf(gpa, n, 501);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(50), .source = 500 });
    const id = try h.s.beginRead(key);
    var pend: Pending = .{ .h = h };
    defer pend.held.deinit(gpa);
    const out = try gpa.alloc(u8, n * bpt);
    defer gpa.free(out);
    var ok = true;
    for (0..12) |i| {
        if (!h.s.readAwait(id, @intCast(i), 2 * std.time.ns_per_s, .{ .ctx = &pend, .recycle = Pending.recycle })) {
            ok = false;
            break;
        }
        var rb: [1]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, @intCast(i + 1), &rb);
        if (got.n != 1) {
            ok = false;
            break;
        }
        @memcpy(out[rb[0].first * bpt ..][0..rb[0].data.len], rb[0].data);
        try pend.held.append(gpa, rb[0].slot); // its copy's event not complete yet
    }
    Pending.recycle(&pend);
    if (!ok) _ = h.s.endRead(id, .abort) catch {};
    try ensure(ok, @src());
    _ = waitUntil(struct {
        fn f(x: struct { *H, store.ReadId }) bool {
            return x[0].s.readSmallView(x[1]) != null;
        }
    }.f, .{ h, id }, 5000);
    var done = (h.s.endRead(id, .commit) catch return CheckFailed) orelse return CheckFailed;
    done.lineage.deinit(gpa);
    try ensure(std.mem.eql(u8, out, h.src(500)[0 .. n * bpt]), @src());
}

/// Read-ahead accounting: preallocated buffers are kept and only the shortfall is charged.
fn checkCarve(gpa: Allocator) !void {
    const per = 3 * slots.alignment;
    const pre = try gpa.alignedAlloc(u8, .fromByteUnits(slots.alignment), 2 * per + per / 2);
    defer gpa.free(pre);
    try ensure(slots.extraBytes(&.{pre}, per, 7) == 5 * per, @src());
    try ensure(slots.extraBytes(&.{pre}, per, 2) == 0, @src());
}

const Slow = struct {
    h: *H,
    calls: std.ArrayList(store.ReadId) = .empty,

    fn copy(ctx: ?*anyopaque, id: store.ReadId, b: store.ReadyBlock) bool {
        const s: *Slow = @ptrCast(@alignCast(ctx.?));
        s.calls.append(s.h.gpa, id) catch {};
        std.Io.sleep(io, .fromMilliseconds(30), .awake) catch {};
        s.h.s.readRelease(b.slot);
        return true;
    }
};

/// the restores in flight share one copy budget a pause, each pause starting with the next.
fn checkBudget(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "budget", .read_slots = 16, .restore_slots = 3, .max_reads = 3 });
    defer h.deinit();
    var ids: [3]store.ReadId = undefined;
    for (0..3) |i| {
        const x = try idsOf(gpa, 5 * block_tokens, 80 + i);
        defer gpa.free(x);
        const key = try write(h, x, .{ .lineage = lin(@intCast(80 + i)), .source = 80 + i });
        ids[i] = try h.s.beginRead(key);
    }
    for (ids) |id| for (0..5) |i| try ensure(h.s.readAwait(id, @intCast(i), 5 * std.time.ns_per_s, .{}), @src());
    var slow: Slow = .{ .h = h };
    defer slow.calls.deinit(gpa);
    var turn: usize = 0;
    _ = h.s.stepReads(&ids, &turn, 40 * std.time.ns_per_ms, &slow, Slow.copy);
    // copies counted, not wall time (a loaded machine stretches each 30 ms sleep, which let the every-read-its-own-
    // budget mutant fit under a time bound): one shared 40 ms budget takes 1 or 2 copies; a budget each read takes
    // at least one copy per read, 3
    const first_calls = slow.calls.items.len;
    var firsts: [3]store.ReadId = undefined;
    for (0..3) |k| {
        slow.calls.clearRetainingCapacity();
        _ = h.s.stepReads(&ids, &turn, 40 * std.time.ns_per_ms, &slow, Slow.copy);
        firsts[k] = if (slow.calls.items.len > 0) slow.calls.items[0] else 0;
    }
    for (ids) |id| _ = h.s.endRead(id, .cancel) catch {};
    try ensure(first_calls >= 1 and first_calls <= 2, @src()); // (every read the whole budget: 3 or more)
    std.mem.sort(store.ReadId, &firsts, {}, std.sort.asc(store.ReadId));
    var want = ids;
    std.mem.sort(store.ReadId, &want, {}, std.sort.asc(store.ReadId));
    try ensure(std.mem.eql(store.ReadId, &firsts, &want), @src());
}

const CrashMode = enum {
    publish, // a new point's publication
    link, // ... whose first blocks equal a stored point's bytes (written as hard links)
    rewrite, // a published point's manifest rewritten with the shared flag
};

const crash_points = [_]struct { step: fsio.Step, suffix: []const u8, skip: u32 = 0, mode: CrashMode = .publish }{
    .{ .step = .written, .suffix = ".blk" },
    .{ .step = .synced, .suffix = ".blk" },
    .{ .step = .renamed, .suffix = ".blk" },
    .{ .step = .renamed, .suffix = ".blk", .skip = 1 },
    .{ .step = .dir_synced, .suffix = "blocks" },
    .{ .step = .written, .suffix = ".ids" },
    .{ .step = .renamed, .suffix = ".ids" },
    .{ .step = .written, .suffix = ".st" },
    .{ .step = .renamed, .suffix = ".st" },
    .{ .step = .dir_synced, .suffix = "points" },
    .{ .step = .written, .suffix = ".man" },
    .{ .step = .synced, .suffix = ".man" },
    .{ .step = .renamed, .suffix = ".man" },
    .{ .step = .published, .suffix = "points" },
    // the durability campaign: while a block is linked, and while shared metadata is rewritten
    .{ .step = .linked, .suffix = ".blk", .mode = .link },
    .{ .step = .linked, .suffix = ".blk", .skip = 1, .mode = .link },
    .{ .step = .written, .suffix = ".man", .mode = .rewrite },
    .{ .step = .synced, .suffix = ".man", .mode = .rewrite },
    .{ .step = .renamed, .suffix = ".man", .mode = .rewrite },
    .{ .step = .rewritten, .suffix = ".man", .mode = .rewrite },
};

fn crashed(h: *H) bool {
    return h.s.isCrashed();
}

/// The durability fault campaign: a kill after every write, fsync and rename of a point's publication, after a
/// dedup link is made (not yet renamed), and at every step of a shared-flag manifest rewrite. On restart an earlier
/// point verifies and restores; the cut point is either whole (published, shared or not) or gone; no temporary file,
/// orphan block or slot is left.
fn checkCrashes(gpa: Allocator) !void {
    for (crash_points, 0..) |cp, ci| {
        const h = try H.init(gpa, .{ .tag = "crash" });
        defer h.deinit();
        const a = try idsOf(gpa, 2 * block_tokens, 900 + ci);
        defer gpa.free(a);
        const ka = try write(h, a, .{ .lineage = lin(90), .source = 90 });
        const b = try idsOf(gpa, 3 * block_tokens + 3, 950 + ci);
        defer gpa.free(b);
        const src_b: u64 = if (cp.mode == .link) 90 else 91; // link: B's rows are A's (other ids, other names)
        var kb = format.pointKey(b, block_tokens, &.{});
        if (cp.mode == .rewrite) kb = try write(h, b, .{ .lineage = lin(91), .source = src_b });
        h.hk.crash_step = cp.step;
        h.hk.crash_suffix = cp.suffix;
        h.hk.crash_skip = cp.skip;
        if (cp.mode == .rewrite) {
            h.s.markShared(kb);
        } else {
            try ensure(keepIds(h, b, .{ .lineage = lin(91), .source = src_b }) == .queued, @src());
            pump(h);
        }
        if (!waitUntil(crashed, h, 3000)) {
            if (!quiet) std.debug.print("     no crash at {t} {s} (+{d}, {t})\n", .{ cp.step, cp.suffix, cp.skip, cp.mode });
            return CheckFailed;
        }
        const temps_before = countFiles(gpa, h.s.blocks_fd, true) + countFiles(gpa, h.s.points_fd, true);
        h.hk.crash_step = null;
        try h.reopen();
        try ensure(countFiles(gpa, h.s.blocks_fd, true) == 0 and countFiles(gpa, h.s.points_fd, true) == 0, @src());
        if (temps_before > 0) try ensure(h.s.counts().temps_removed >= temps_before, @src());
        try ensure(countFiles(gpa, h.s.blocks_fd, false) == h.s.have.count(), @src());
        try ensure(countFiles(gpa, h.s.points_fd, false) == 3 * h.s.points.count(), @src());
        const got = (try restore(h, ka)) orelse return CheckFailed;
        defer gpa.free(got);
        try ensure(std.mem.eql(u8, got, h.src(90)[0 .. 2 * block_tokens * bpt]), @src());
        const published = cp.mode == .rewrite or (cp.step == .renamed and std.mem.eql(u8, cp.suffix, ".man")) or cp.step == .published;
        try ensure(h.s.presence(kb) == published, @src());
        if (published) {
            const gb = (try restore(h, kb)) orelse return CheckFailed;
            defer gpa.free(gb);
            try ensure(std.mem.eql(u8, gb, h.src(src_b)[0 .. b.len * bpt]), @src());
        }
        if (cp.mode == .rewrite) { // whole either way; durably shared once the rewrite's directory fsync ran
            const on_disk = manifestShared(h, kb) orelse return CheckFailed;
            try ensure((h.s.points.get(kb) orelse return CheckFailed).shared == on_disk, @src());
            if (cp.step == .rewritten) try ensure(on_disk, @src());
            if (cp.step == .written or cp.step == .synced) try ensure(!on_disk, @src());
        }
    }
}

/// Recovery: a truncated block, a missing small state and a damaged manifest each drop only their point; a stale
/// temporary file and unreferenced blocks are collected.
fn checkRecovery(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "recover" });
    defer h.deinit();
    var keys: [4]Digest = undefined;
    for (0..4) |i| {
        const x = try idsOf(gpa, 2 * block_tokens + 4, 300 + i);
        defer gpa.free(x);
        keys[i] = try write(h, x, .{ .lineage = lin(@intCast(30 + i)), .source = 300 + i });
    }
    const p0 = h.s.points.get(keys[0]) orelse return CheckFailed;
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const bname = try std.fmt.bufPrintSentinel(&zb, "{s}", .{blockFile(&nb, p0.blocks[0].name)}, 0);
    const fd = c.openat(h.s.blocks_fd, bname, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    _ = c.ftruncate(fd, block_bytes - 100);
    _ = c.close(fd);
    var nb2: [96]u8 = undefined;
    var zb2: [96]u8 = undefined;
    _ = c.unlinkat(h.s.points_fd, try std.fmt.bufPrintSentinel(&zb2, "{s}", .{pointFile(&nb2, keys[1], ".st")}, 0), 0);
    const mname = try std.fmt.bufPrintSentinel(&zb2, "{s}", .{pointFile(&nb2, keys[2], ".man")}, 0);
    const mfd = c.openat(h.s.points_fd, mname, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    const junk: [4]u8 = @splat(0x33);
    _ = c.pwrite(mfd, &junk, 4, 40);
    _ = c.close(mfd);
    const tfd = c.openat(h.s.blocks_fd, ".stale.blk.1.1.tmp", .{ .ACCMODE = .WRONLY, .CREAT = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    _ = c.close(tfd);
    try h.reopen();
    try ensure(!h.s.points.contains(keys[0]) and !h.s.points.contains(keys[1]) and !h.s.points.contains(keys[2]), @src());
    try ensure(h.s.points.contains(keys[3]), @src());
    try ensure(h.s.counts().recovered_dropped == 3 and h.s.counts().temps_removed >= 1, @src());
    try ensure(countFiles(gpa, h.s.blocks_fd, false) == h.s.have.count() and h.s.have.count() == 3, @src());
    const got = (try restore(h, keys[3])) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(303)[0 .. (2 * block_tokens + 4) * bpt]), @src());
}

/// CRC-32 forging: the 4 bytes that make `data[0 .. len - 4] ++ them` hash to `target` (reflected, ISO-HDLC).
fn forgeCrc(data: []u8, target: u32) void {
    var table: [256]u32 = undefined;
    for (&table, 0..) |*t, i| {
        var x: u32 = @intCast(i);
        for (0..8) |_| x = if (x & 1 != 0) (x >> 1) ^ 0xEDB88320 else x >> 1;
        t.* = x;
    }
    const head = data[0 .. data.len - 4];
    const before = ~std.hash.Crc32.hash(head); // the register after the head
    var x: u32 = ~target; // the register wanted after the last 4 bytes
    for (0..4) |_| {
        const top: u8 = @truncate(x >> 24);
        var idx: u32 = 0;
        for (table, 0..) |t, i| if (@as(u8, @truncate(t >> 24)) == top) {
            idx = @intCast(i);
            break;
        };
        x = ((x ^ table[idx]) << 8) | idx;
    }
    std.mem.writeInt(u32, data[data.len - 4 ..][0..4], x ^ before, .little);
}

/// Dedup links only equal bytes: a block with the same CRC-32 and length but other bytes is written, not linked.
fn checkDedupSha(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "dedup" });
    defer h.deinit();
    const ia = try idsOf(gpa, block_tokens, 41);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, block_tokens, 42);
    defer gpa.free(ib);
    const ka = try write(h, ia, .{ .lineage = lin(41), .source = 41 });
    const want = std.hash.Crc32.hash(h.src(41)[0..block_bytes]);
    const rows = h.src(42);
    @memcpy(rows[0..block_bytes], h.src(41)[0..block_bytes]);
    rows[17] ^= 0x5a;
    forgeCrc(rows[0..block_bytes], want);
    try ensure(std.hash.Crc32.hash(rows[0..block_bytes]) == want, @src());
    const kb = try write(h, ib, .{ .lineage = lin(42), .source = 42 });
    const pa = h.s.points.get(ka) orelse return CheckFailed;
    const pb = h.s.points.get(kb) orelse return CheckFailed;
    try ensure(h.s.have.get(pa.blocks[0].name).?.ino != h.s.have.get(pb.blocks[0].name).?.ino, @src());
    const got = (try restore(h, kb)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, rows[0..block_bytes]), @src());
}

/// A guard before source rows change: the point's blocks still to stage are staged now (the engine queues those
/// copies before the change), never cancelled while slots allow.
fn checkGuard(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "guard" });
    defer h.deinit();
    const ids = try idsOf(gpa, 4 * block_tokens, 7);
    defer gpa.free(ids);
    const kept = try gpa.dupe(u8, h.src(5)[0 .. 4 * block_tokens * bpt]);
    defer gpa.free(kept);
    try ensure(keepIds(h, ids, .{ .lineage = lin(7), .source = 5 }) == .queued, @src());
    try ensure(h.s.pins(.{ .id = 5, .generation = 1 }, 0), @src());
    var any: [3]bool = undefined;
    h.s.pinsAny(&.{ 6, 5, 9 }, &any); // (pins for many sources, one lock)
    try ensure(!any[0] and any[1] and !any[2], @src());
    var out: [8]store.StageReq = undefined;
    const g = h.s.guard(.{ .id = 5, .generation = 1 }, 0, &out);
    try ensure(g.staged == 4 and g.cancelled == 0, @src());
    h.s.pinsAny(&.{ 6, 5, 9 }, &any); // (every block staged: nothing left to pin, as pins says)
    try ensure(!any[1] and any[1] == h.s.pins(.{ .id = 5, .generation = 1 }, 0), @src());
    for (out[0..g.staged]) |r| @memcpy(r.dst, h.src(5)[r.first * bpt ..][0..r.dst.len]); // queued before the change
    @memset(h.src(5)[0 .. 4 * block_tokens * bpt], 0xee); // the change
    for (out[0..g.staged]) |r| h.s.staged(r.ticket, true);
    try ensure(settle(h).done, @src());
    const got = (try restore(h, format.pointKey(ids, block_tokens, &.{}))) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, kept), @src());
}

/// ... past the room it has, or for an older generation of the source, the unpublished point is cancelled, its
/// staging back, a completion saying why.
fn checkGuardCancel(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "guardc" });
    defer h.deinit();
    const ids = try idsOf(gpa, 4 * block_tokens, 8);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids, .{ .lineage = lin(8), .source = 6 }) == .queued, @src());
    var out: [2]store.StageReq = undefined;
    const g = h.s.guard(.{ .id = 6, .generation = 1 }, 0, &out);
    try ensure(g.staged == 0 and g.cancelled == 1, @src());
    const ids2 = try idsOf(gpa, 2 * block_tokens, 9);
    defer gpa.free(ids2);
    try ensure(keepIds(h, ids2, .{ .lineage = lin(9), .source = 7, .gen = 1 }) == .queued, @src());
    var o2: [8]store.StageReq = undefined;
    const g2 = h.s.guard(.{ .id = 7, .generation = 2 }, 0, &o2);
    try ensure(g2.cancelled == 1, @src());
    var comp: [4]store.Completion = undefined;
    const n = h.s.poll(&comp);
    try ensure(n == 2 and comp[0] == .cancelled and comp[0].cancelled.why == .guard and comp[1].cancelled.why == .source_gone, @src());
    _ = settle(h);
    const ct = h.s.counts();
    try ensure(ct.staging_bytes == 0 and ct.points == 0 and ct.pending_jobs == 0, @src());
}

/// The staging cap holds for a point larger than all of staging (the GPU's 250k-token point is ~220 MB against a
/// 128 MiB cap): its blocks are staged a few slots at a time as the writer frees them, its restore reads through a
/// few read slots, a guard that cannot stage what is left cancels instead of overshooting, and the peak counter is
/// what was held at once (at most the slots carved), never more.
fn checkStagingCap(gpa: Allocator) !void {
    const o: Opts = .{ .tag = "stagecap", .write_slots = 3, .read_slots = 2, .small_slots = 2, .restore_slots = 1 };
    const cap: u64 = (o.write_slots + o.read_slots) * block_bytes + (o.small_slots + o.restore_slots) * small_max;
    const h = try H.init(gpa, o);
    defer h.deinit();
    const ids = try idsOf(gpa, max_tokens, 3100); // 64 blocks: 256 KiB against 44 KiB of staging
    defer gpa.free(ids);
    try ensure(max_tokens * bpt > 4 * cap, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(31), .source = 310 }) == .queued, @src());
    var reqs: [16]store.StageReq = undefined;
    var staged: usize = 0;
    var i: usize = 0;
    while (i < 20000 and h.s.pendingStage() > 0) : (i += 1) {
        const n = h.s.stage(&reqs);
        try ensure(n <= o.write_slots, @src());
        try ensure(h.s.counts().staging_bytes <= cap, @src());
        for (reqs[0..n]) |r| {
            @memcpy(r.dst, h.src(310)[r.first * bpt ..][0..r.dst.len]);
            h.s.staged(r.ticket, true);
        }
        staged += n;
        if (n == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try ensure(staged == max_tokens / block_tokens, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    const key = format.pointKey(ids, block_tokens, &.{});
    try ensure(h.s.presence(key), @src());
    const got = (try restore(h, key)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(310)[0 .. max_tokens * bpt]), @src());
    // a second large point guarded before any of it staged: more blocks left than free write slots, so it is cancelled
    const ids2 = try idsOf(gpa, max_tokens, 3101);
    defer gpa.free(ids2);
    try ensure(keepIds(h, ids2, .{ .lineage = lin(32), .source = 311 }) == .queued, @src());
    var out: [64]store.StageReq = undefined;
    const g = h.s.guard(.{ .id = 311, .generation = 1 }, 0, &out);
    try ensure(g.staged == 0 and g.cancelled == 1, @src());
    _ = settle(h);
    const ct = h.s.counts();
    try ensure(ct.staging_peak_bytes <= cap and ct.staging_peak_bytes >= block_bytes, @src());
    try ensure(ct.staging_bytes == 0 and ct.guard_staged == 0 and ct.cancelled_guard == 1, @src());
}

/// 0110's eviction order: old turns, abandoned forks, whole conversations least recently used first (longest state
/// first within one), shared prefixes last.
fn checkEvictionOrder(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "evict", .turns = 2 });
    defer h.deinit();
    const sys = try idsOf(gpa, 2 * block_tokens, 1000);
    defer gpa.free(sys);
    const r1 = try idsOf(gpa, block_tokens, 1001);
    defer gpa.free(r1);
    const r2 = try idsOf(gpa, block_tokens, 1002);
    defer gpa.free(r2);
    const r3 = try idsOf(gpa, block_tokens, 1003);
    defer gpa.free(r3);
    const r4 = try idsOf(gpa, block_tokens, 1004);
    defer gpa.free(r4);
    const b1 = try cat(gpa, sys, r1);
    defer gpa.free(b1);
    const a1 = try cat(gpa, sys, r2);
    defer gpa.free(a1);
    const a2f = try cat(gpa, a1, r3);
    defer gpa.free(a2f);
    const a2 = try cat(gpa, a1, r4);
    defer gpa.free(a2);
    const ks = try write(h, sys, .{ .lineage = lin(1), .source = 1, .shared = true });
    const kb1 = try write(h, b1, .{ .lineage = lin(2), .source = 2 });
    const ka1 = try write(h, a1, .{ .lineage = lin(3), .source = 3 });
    const ka2f = try write(h, a2f, .{ .lineage = lin(4), .source = 4 });
    const ka2 = try write(h, a2, .{ .lineage = lin(5), .source = 5 });
    const order = h.s.evictionOrder() catch return CheckFailed;
    defer gpa.free(order);
    const W = Store.Why;
    const want = [_]struct { k: Digest, w: W }{
        .{ .k = ka2f, .w = .pruned_forks },
        .{ .k = kb1, .w = .pruned_conversations },
        .{ .k = ka2, .w = .pruned_conversations },
        .{ .k = ka1, .w = .pruned_conversations },
        .{ .k = ks, .w = .pruned_shared },
    };
    try ensure(order.len == want.len, @src());
    for (order, want) |o, w| try ensure(std.mem.eql(u8, &o.key, &w.k) and o.why == w.w, @src());
    const sm = h.s.summary();
    try ensure(sm.conversations == 2 and sm.shared_states == 1 and sm.forks == 1 and sm.turns_kept_max == 2, @src());
}

/// Old turns past retention on the active path go as soon as a newer turn is written; their blocks stay (the
/// newer turn names them).
fn checkTurns(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "turns", .turns = 2 });
    defer h.deinit();
    const t3 = try idsOf(gpa, 3 * block_tokens, 1100);
    defer gpa.free(t3);
    const k1 = try write(h, t3[0..block_tokens], .{ .lineage = lin(1), .source = 11 });
    const k2 = try write(h, t3[0 .. 2 * block_tokens], .{ .lineage = lin(1), .source = 11 });
    const k3 = try write(h, t3, .{ .lineage = lin(1), .source = 11 });
    try ensure(!h.s.points.contains(k1) and h.s.points.contains(k2) and h.s.points.contains(k3), @src());
    const ct = h.s.counts();
    try ensure(ct.trimmed_turns == 1 and ct.blocks == 3 and ct.staged_blocks == 3, @src());
    const got = (try restore(h, k3)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(11)[0 .. 3 * block_tokens * bpt]), @src());
}

/// A resume forks the lineage at its cut: the same names before it (nothing restaged), its own after.
fn checkFork(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "fork" });
    defer h.deinit();
    const p = try idsOf(gpa, 3 * block_tokens, 1200);
    defer gpa.free(p);
    const more = try idsOf(gpa, 2 * block_tokens, 1201);
    defer gpa.free(more);
    const q = try cat(gpa, p[0 .. 2 * block_tokens], more);
    defer gpa.free(q);
    _ = try write(h, p, .{ .lineage = lin(12), .source = 12 });
    const before = h.s.counts().staged_blocks;
    var forked = try lin(12).fork(gpa, @splat(77), 2 * block_tokens);
    defer forked.deinit(gpa);
    const kq = try write(h, q, .{ .lineage = forked, .source = 12 });
    try ensure(h.s.counts().staged_blocks - before == 2, @src());
    const got = (try restore(h, kq)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(12)[0 .. 4 * block_tokens * bpt]), @src());
}

/// 0111 groups: branches of one conversation that share only a shared prefix count as one conversation.
fn checkGroups(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "groups" });
    defer h.deinit();
    const sys = try idsOf(gpa, block_tokens, 1300);
    defer gpa.free(sys);
    _ = try write(h, sys, .{ .lineage = lin(13), .source = 13, .shared = true });
    for (0..3) |i| {
        const x = try idsOf(gpa, block_tokens, 1301 + i);
        defer gpa.free(x);
        const y = try cat(gpa, sys, x);
        defer gpa.free(y);
        _ = try write(h, y, .{ .lineage = lin(@intCast(14 + i)), .source = 14 + i, .group = if (i < 2) 7 else 0 });
    }
    try ensure(h.s.summary().conversations == 2, @src());
}

/// A conversation over its quota trims its own old turns and forks first, not another conversation's states.
fn checkQuota(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "quota", .cap = 400 << 10, .quota = 0.1, .turns = 0 });
    defer h.deinit();
    const other = try idsOf(gpa, 2 * block_tokens, 1400);
    defer gpa.free(other);
    const ko = try write(h, other, .{ .lineage = lin(40), .source = 40 });
    const conv = try idsOf(gpa, 8 * block_tokens, 1401);
    defer gpa.free(conv);
    var keys: [8]Digest = undefined;
    for (1..9) |t| {
        const r = keepIds(h, conv[0 .. t * block_tokens], .{ .lineage = lin(41), .source = 41 });
        if (r != .queued) break;
        try ensure(settle(h).done, @src());
        keys[t - 1] = format.pointKey(conv[0 .. t * block_tokens], block_tokens, &.{});
    }
    const ct = h.s.counts();
    try ensure(ct.quota_trims > 0 and ct.pruned_turns > 0, @src());
    try ensure(h.s.points.contains(ko), @src());
    try ensure(h.s.points.contains(keys[7]), @src());
}

/// The free-disk floor: no write leaves less free; room is made in eviction order, else the write is refused.
fn checkFloor(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "floor", .floor = 1 << 20, .free = (1 << 20) + 200 * 1024 });
    defer h.deinit();
    const a = try idsOf(gpa, 4 * block_tokens, 1500);
    defer gpa.free(a);
    const ka = try write(h, a, .{ .lineage = lin(50), .source = 50 });
    // the disk fills with other files: the next write evicts A first
    h.hk.free = (1 << 20) + 2 * 1024;
    h.s.refreshDiskFree();
    const b = try idsOf(gpa, 2 * block_tokens, 1501);
    defer gpa.free(b);
    try ensure(keepIds(h, b, .{ .lineage = lin(51), .source = 51 }) == .queued, @src());
    try ensure(!h.s.points.contains(ka), @src());
    _ = settle(h);
    // full of other things: nothing could make room, the write is refused
    h.hk.free = 1 << 10;
    h.s.refreshDiskFree();
    const c2 = try idsOf(gpa, 2 * block_tokens, 1502);
    defer gpa.free(c2);
    try ensure(keepIds(h, c2, .{ .lineage = lin(52), .source = 52 }) == .no_space, @src());
}

/// The floor reads the disk as it is at each keep: free space something else took since the store's last write is
/// seen without a write of the store's to refresh it.
fn checkFloorFresh(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "floorfresh", .floor = 1 << 20, .free = 64 << 20 });
    defer h.deinit();
    const a = try idsOf(gpa, 2 * block_tokens, 1900);
    defer gpa.free(a);
    _ = try write(h, a, .{ .lineage = lin(90), .source = 90 });
    h.hk.free = 4 << 10; // another writer filled the disk; the store wrote nothing since
    const b = try idsOf(gpa, 2 * block_tokens, 1901);
    defer gpa.free(b);
    try ensure(keepIds(h, b, .{ .lineage = lin(91), .source = 91 }) != .queued, @src()); // not admitted below the floor
    try ensure(h.s.counts().disk_free_bytes == 4 << 10, @src());
}

/// The cap: a write that would pass it evicts the least recently used conversation, not a newer one.
fn checkCap(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "cap", .cap = 12 * 1024 });
    defer h.deinit();
    var keys: [3]Digest = undefined;
    for (0..3) |i| {
        const x = try idsOf(gpa, block_tokens, 1600 + i);
        defer gpa.free(x);
        keys[i] = try write(h, x, .{ .lineage = lin(@intCast(60 + i)), .source = 60 + i });
    }
    try ensure(!h.s.points.contains(keys[0]) and h.s.points.contains(keys[1]) and h.s.points.contains(keys[2]), @src());
    try ensure(h.s.counts().pruned_conversations >= 1 and h.s.counts().stored_bytes <= h.cfg.cap_bytes, @src());
}

/// The cap takes whole conversations, never a piece: the least recently used conversation A (turns of 2, 3 and 4
/// blocks) goes whole for a keep that needs less than its newest turn, B (newer) stays whole, and the counter counts
/// one conversation (its points beside it).
fn checkWholeConversation(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "whole" });
    defer h.deinit();
    const a1 = try idsOf(gpa, 2 * block_tokens, 1700);
    defer gpa.free(a1);
    const r1 = try idsOf(gpa, block_tokens, 1701);
    defer gpa.free(r1);
    const a2 = try cat(gpa, a1, r1);
    defer gpa.free(a2);
    const a3 = try cat(gpa, a2, r1);
    defer gpa.free(a3);
    const b1 = try idsOf(gpa, 2 * block_tokens, 1702);
    defer gpa.free(b1);
    const r2 = try idsOf(gpa, 2 * block_tokens, 1703);
    defer gpa.free(r2);
    const b2 = try cat(gpa, b1, r2);
    defer gpa.free(b2);
    const ka1 = try write(h, a1, .{ .lineage = lin(70), .source = 70 });
    const ka2 = try write(h, a2, .{ .lineage = lin(70), .source = 71 });
    const ka3 = try write(h, a3, .{ .lineage = lin(70), .source = 72 });
    const kb1 = try write(h, b1, .{ .lineage = lin(71), .source = 73 });
    const kb2 = try write(h, b2, .{ .lineage = lin(71), .source = 74 });
    // the store at its cap: one more block's keep must make room
    h.s.cfg.cap_bytes = h.s.counts().stored_bytes + 100;
    const cnew = try idsOf(gpa, block_tokens, 1704);
    defer gpa.free(cnew);
    try ensure(keepIds(h, cnew, .{ .lineage = lin(72), .source = 75 }) == .queued, @src());
    _ = settle(h);
    for ([_]Digest{ ka1, ka2, ka3 }) |k| try ensure(!h.s.points.contains(k), @src());
    for ([_]Digest{ kb1, kb2 }) |k| try ensure(h.s.points.contains(k), @src());
    const ct = h.s.counts();
    try ensure(ct.pruned_conversations == 1 and ct.pruned_conversation_points == 3, @src());
}

var held_ids = std.atomic.Value(bool).init(false);
fn holdIds(_: *H) void {
    held_ids.store(true, .release);
    std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
}

/// A block on disk is counted once when room is made: J's two blocks are written (in stored_bytes) and its point not
/// yet published (the writer held at its ids) when K is kept into a cap with room for J's files and K's estimate:
/// nothing is evicted (counted twice, J's blocks would make room for themselves again and take A).
fn checkRoomCountsOnce(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "roomonce" });
    defer h.deinit();
    const ia = try idsOf(gpa, block_tokens, 3400);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, block_tokens, 3401);
    defer gpa.free(ib);
    const ij = try idsOf(gpa, 2 * block_tokens, 3402);
    defer gpa.free(ij);
    const ik = try idsOf(gpa, block_tokens, 3403);
    defer gpa.free(ik);
    const ka = try write(h, ia, .{ .lineage = lin(34), .source = 340 });
    const kb = try write(h, ib, .{ .lineage = lin(35), .source = 341 });
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    // J's files: 2 blocks, small state (sections on 256 B), ids, a manifest of header, 2 blocks, 2 sections, trailer;
    // K's estimate (a keep's need): a block, small state, ids, the manifest's header and block list
    const j_files: u64 = 2 * block_bytes + 712 + 128 + (256 + 2 * 80 + 2 * 56 + 8);
    const k_need: u64 = block_bytes + 712 + 64 + 256 + 80;
    h.s.cfg.cap_bytes = h.s.counts().stored_bytes + j_files + k_need;
    held_ids.store(false, .release);
    h.hk.action_step = .written;
    h.hk.action_suffix = ".ids";
    h.hk.action = holdIds;
    try ensure(keepIds(h, ij, .{ .lineage = lin(36), .source = 342 }) == .queued, @src());
    pump(h);
    try ensure(waitUntil(struct {
        fn f(_: void) bool {
            return held_ids.load(.acquire);
        }
    }.f, {}, 3000), @src());
    try ensure(h.s.counts().blocks == 4, @src()); // (J's two blocks on disk, its point not yet)
    try ensure(keepIds(h, ik, .{ .lineage = lin(37), .source = 343 }) == .queued, @src());
    try ensure(settle(h).done, @src());
    const kj = format.pointKey(ij, block_tokens, &.{});
    const kk = format.pointKey(ik, block_tokens, &.{});
    try ensure(h.s.presence(ka) and h.s.presence(kb) and h.s.presence(kj) and h.s.presence(kk), @src());
    try ensure(h.s.counts().pruned_conversations == 0 and h.s.counts().queued_bytes == 0, @src());
}

/// A second owner of a live store is refused before it scans or deletes anything.
fn checkOwner(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "owner" });
    defer h.deinit();
    const tfd = c.openat(h.s.blocks_fd, ".pending.blk.9.9.tmp", .{ .ACCMODE = .WRONLY, .CREAT = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    _ = c.close(tfd);
    if (Store.open(gpa, io, h.cfg)) |other| {
        other.close();
        try ensure(false, @src());
    } else |e| try ensure(e == error.StoreBusy, @src());
    try ensure(fileExists(h.s.blocks_fd, ".pending.blk.9.9.tmp"), @src());
}

/// Other builds' namespaces: pruned unless live (locked).
fn checkPrune(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "prune", .keep_builds = 0 });
    defer h.deinit();
    const rfd = try fsio.openDir(h.root);
    defer _ = c.close(rfd);
    try fsio.ensureDirAt(rfd, "native-spill-00000000deadbeef", null);
    const ofd = try fsio.openDirAt(rfd, "native-spill-00000000deadbeef");
    defer _ = c.close(ofd);
    try fsio.ensureDirAt(ofd, "rank0", null);
    const r0 = try fsio.openDirAt(ofd, "rank0");
    var live = try fsio.Lock.take(r0, null);
    try h.reopen();
    try ensure(fileExists(ofd, "rank0"), @src());
    live.release();
    _ = c.close(r0);
    try h.reopen();
    try ensure(!fileExists(rfd, "native-spill-00000000deadbeef"), @src());
}

/// The namespace directory a compat digest names (`native-spill-` and its first 16 hex digits).
fn namespaceOf(buf: []u8, compat: Digest) []const u8 {
    const x = format.hex(&compat);
    return std.fmt.bufPrint(buf, "native-spill-{s}", .{x[0..16]}) catch unreachable;
}

/// Other builds are pruned by last use, not by creation: builds A, B, then A again, then C open (keep_builds 1): A,
/// used last before C, stays; B goes.
fn checkPruneByLastUse(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lastbuild", .keep_builds = 1 });
    defer h.deinit();
    const a: Digest = h.cfg.compat;
    const b: Digest = @splat(0xb0);
    const cc: Digest = @splat(0xc0);
    for ([_]Digest{ b, a, cc }) |next| {
        std.Io.sleep(io, .fromMilliseconds(15), .awake) catch {};
        h.closeStore();
        h.cfg.compat = next;
        try h.open();
    }
    const rfd = try fsio.openDir(h.root);
    defer _ = c.close(rfd);
    var nb: [64]u8 = undefined;
    try ensure(fileExists(rfd, namespaceOf(&nb, a)), @src());
    try ensure(!fileExists(rfd, namespaceOf(&nb, b)), @src());
    try ensure(fileExists(rfd, namespaceOf(&nb, cc)) and h.s.counts().builds_pruned == 1, @src());
}

extern "c" fn getgroups(size: c_int, list: [*]c.gid_t) c_int;

/// A group of this user other than its own (a file it makes there needs a chown to get it), or null.
fn otherGroup() ?c.gid_t {
    var groups: [64]c.gid_t = undefined;
    const n = getgroups(groups.len, &groups);
    if (n <= 0) return null;
    for (groups[0..@intCast(n)]) |g| if (g != c.getegid()) return g;
    return null;
}

fn gidAt(dirfd: c.fd_t, name: []const u8) !c.gid_t {
    var zb: [128]u8 = undefined;
    return (try fsio.statAt(dirfd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{name}, 0))).gid;
}

/// Every folder and file the store makes has the store root's owner before it is renamed into place. Without root,
/// the scratch root is given another of this user's groups, and every folder, the owner lock, and every block (one a
/// dedup link), ids, small state and manifest (one rewritten shared) must carry it; another build is pruned meanwhile.
fn checkOwnerFiles(gpa: Allocator) !void {
    const gid = otherGroup() orelse return error.SkipZigTest; // (a user in one group cannot show it)
    const h = try H.init(gpa, .{ .tag = "ownerfiles", .keep_builds = 0 });
    defer h.deinit();
    h.closeStore();
    const rfd = try fsio.openDir(h.root);
    defer _ = c.close(rfd);
    var nb: [64]u8 = undefined;
    const ns = namespaceOf(&nb, h.cfg.compat);
    var zb: [64]u8 = undefined;
    fsio.removeTree(gpa, rfd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{ns}, 0)); // (made by the first open)
    try ensure(c.fchown(rfd, c.geteuid(), gid) == 0, @src());
    var sx: std.os.linux.Statx = undefined;
    try ensure(c.statx(rfd, "", 0x1000, .{ .MODE = true }, &sx) == 0, @src());
    if (sx.mode & 0o2000 != 0) return error.SkipZigTest; // (a setgid folder gives its group without a chown)
    try fsio.ensureDirAt(rfd, "native-spill-00000000feedface", null); // another build, to be pruned
    const ofd = try fsio.openDirAt(rfd, "native-spill-00000000feedface");
    try fsio.ensureDirAt(ofd, "rank0", null);
    _ = c.close(ofd);
    try h.open();
    const ids = try idsOf(gpa, 2 * block_tokens + 5, 2800);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(28), .source = 280 });
    const other = try idsOf(gpa, 2 * block_tokens, 2801);
    defer gpa.free(other);
    _ = try write(h, other, .{ .lineage = lin(29), .source = 280 }); // (the same rows: its blocks are links)
    try ensure(h.s.counts().dedup_blocks == 2, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(28), .source = 280, .shared = true }) == .stored, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done and manifestShared(h, key) == true, @src());
    try ensure(!fileExists(rfd, "native-spill-00000000feedface") and h.s.counts().builds_pruned == 1, @src());
    try ensure(try gidAt(rfd, ns) == gid, @src());
    try ensure(try gidAt(h.s.top_fd, "rank0") == gid, @src());
    for ([_][]const u8{ "owner.lock", "blocks", "points" }) |n| try ensure(try gidAt(h.s.rank_fd, n) == gid, @src());
    var files: usize = 0;
    for ([_]c.fd_t{ h.s.blocks_fd, h.s.points_fd }) |dir| {
        var names = try fsio.list(gpa, dir);
        defer fsio.freeList(gpa, &names);
        for (names.items) |n| {
            try ensure(try gidAt(dir, n) == gid, @src());
            files += 1;
        }
    }
    try ensure(files == 3 + 2 * 3 + 2, @src()); // (3 blocks, 2 of them linked by the other point's 2; 2 points' 3 files)
    try ensure(h.s.counts().owner_failed == 0, @src());
}

/// A chown that fails never fails the store: the open and the write go on, the point publishes, and each failure is
/// counted (owner_failed).
fn checkOwnerFailed(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "ownerfail" });
    defer h.deinit();
    h.hk.chown_fail = true;
    try h.reopen();
    const ids = try idsOf(gpa, 2 * block_tokens + 5, 2810);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(30), .source = 281 });
    try ensure(h.s.presence(key), @src());
    // the open's 4 folders and its lock, the write's 3 blocks and 3 files
    try ensure(h.s.counts().owner_failed == 4 + 1 + 3 + 3, @src());
    try h.reopen();
    try ensure(h.s.presence(key), @src());
}

/// A manifest's bytes with its creation time zeroed (two stores' keeps of one state differ only there).
fn manifestNoTime(h: *H, key: Digest) ![]u8 {
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    var file = try fsio.readSmall(&h.s.writer, h.gpa, h.s.points_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, key, ".man")}, 0), 1 << 20);
    defer file.deinit(h.gpa);
    var m = try format.Manifest.parse(h.gpa, file.bytes(), h.cfg.compat, h.cfg.limits);
    defer m.deinit(h.gpa);
    m.created_ns = 0;
    return m.encode(h.gpa, h.cfg.compat);
}

fn pointBytes(h: *H, key: Digest, suffix: []const u8) ![]u8 {
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    var file = try fsio.readSmall(&h.s.writer, h.gpa, h.s.points_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, key, suffix)}, 0), 1 << 20);
    defer file.deinit(h.gpa);
    return h.gpa.dupe(u8, file.bytes());
}

/// A keep given the caller's id chain makes no pass over the ids of its own, and stores exactly what a keep without it
/// stores (key, block names, manifest, ids, restored rows); a chain of the wrong length is refused.
fn checkKeepChain(gpa: Allocator) !void {
    const ids = try idsOf(gpa, 3 * block_tokens + 5, 2900);
    defer gpa.free(ids);
    var ch: std.ArrayList(Digest) = .empty;
    defer ch.deinit(gpa);
    try format.chain(gpa, ids, block_tokens, &ch);
    var hs: [2]*H = undefined;
    var passes: [2]u64 = undefined;
    for (&hs, &passes, 0..) |*h, *n, i| {
        h.* = try H.init(gpa, .{ .tag = "keepchain" });
        const before = format.test_id_passes.load(.monotonic);
        const r = keepIds(h.*, ids, .{ .lineage = lin(29), .source = 290, .chain = if (i == 0) ch.items else null });
        n.* = format.test_id_passes.load(.monotonic) - before;
        try ensure(r == .queued and settle(h.*).done, @src());
    }
    defer for (hs) |h| h.deinit();
    try ensure(passes[0] == 0 and passes[1] == 1, @src());
    const key = format.pointKey(ids, block_tokens, &.{});
    const pa = hs[0].s.points.get(key) orelse return CheckFailed;
    const pb = hs[1].s.points.get(key) orelse return CheckFailed;
    for (pa.blocks, pb.blocks) |a, b| try ensure(std.mem.eql(u8, &a.name, &b.name), @src());
    var files: [2][3][]u8 = undefined;
    for (hs, 0..) |h, i| {
        files[i][0] = try manifestNoTime(h, key);
        files[i][1] = try pointBytes(h, key, ".ids");
        files[i][2] = (try restore(h, key)) orelse return CheckFailed;
    }
    defer for (files) |fs| for (fs) |f| gpa.free(f);
    for (0..3) |k| try ensure(std.mem.eql(u8, files[0][k], files[1][k]), @src());
    try ensure(std.mem.eql(u8, files[0][2], hs[0].src(290)[0 .. ids.len * bpt]), @src());
    // a chain of the wrong length: refused, counted
    const other = try idsOf(gpa, 2 * block_tokens, 2901);
    defer gpa.free(other);
    try ensure(keepIds(hs[0], other, .{ .lineage = lin(30), .source = 291, .chain = ch.items }) == .invalid, @src());
    try ensure(hs[0].s.counts().invalid == 1, @src());
}

/// Random stores: `n` points in memory only (`testInsert`: no files), grown as conversations grow (fresh ones, turns,
/// forks, tails, shared prefixes, groups merging conversations), with ties in `used` and in length.
fn randomStore(h: *H, rnd: std.Random, n: usize) !void {
    const gpa = h.gpa;
    var seqs: std.ArrayList([]u32) = .empty;
    defer {
        for (seqs.items) |x| gpa.free(x);
        seqs.deinit(gpa);
    }
    var shared_at: std.ArrayList(usize) = .empty;
    defer shared_at.deinit(gpa);
    var tries: usize = 0;
    while (h.s.points.count() < n and tries < 20 * n) : (tries += 1) {
        const roll = rnd.uintLessThan(u8, 100);
        var base: []const u32 = &.{};
        var shared = false;
        if (seqs.items.len == 0 or roll < 15) {
            if (shared_at.items.len > 0 and rnd.boolean()) base = seqs.items[shared_at.items[rnd.uintLessThan(usize, shared_at.items.len)]];
        } else if (roll < 20) {
            shared = true;
        } else if (roll < 80) {
            base = seqs.items[seqs.items.len - 1 - rnd.uintLessThan(usize, @min(3, seqs.items.len))];
        } else {
            base = seqs.items[rnd.uintLessThan(usize, seqs.items.len)];
        }
        const blocks = rnd.intRangeAtMost(usize, if (base.len == 0) 1 else 0, 3);
        const tail: usize = if (rnd.uintLessThan(u8, 4) == 0) rnd.intRangeAtMost(usize, 1, block_tokens - 1) else 0;
        const len = base.len + blocks * block_tokens + tail;
        if (len == base.len or len > max_tokens) continue;
        const ids = try gpa.alloc(u32, len);
        errdefer gpa.free(ids);
        @memcpy(ids[0..base.len], base);
        for (ids[base.len..]) |*x| x.* = rnd.intRangeAtMost(u32, 1, 150000);
        _ = try h.s.testInsert(ids, lin(rnd.int(u8)), shared, rnd.uintLessThan(u64, 4), rnd.intRangeAtMost(i96, 0, 30));
        try seqs.append(gpa, ids);
        if (shared) try shared_at.append(gpa, seqs.items.len - 1);
    }
}

fn sameVictims(gpa: Allocator, h: *H) !bool {
    const Same = struct {
        fn of(a: []const Store.Victim, b: []const Store.Victim) bool {
            if (a.len != b.len) return false;
            for (a, b) |x, y| if (!std.mem.eql(u8, &x.key, &y.key) or x.why != y.why) return false;
            return true;
        }
    };
    const convs = try h.s.testConversations();
    defer gpa.free(convs);
    for (0..convs.len + 1) |i| {
        const only: ?usize = if (i < convs.len) convs[i] else null;
        const a = try h.s.testVictims(only, false);
        defer gpa.free(a);
        const b = try h.s.testVictims(only, true);
        defer gpa.free(b);
        if (!Same.of(a, b)) return false;
    }
    return true;
}

/// victims() groups the points by conversation once, and its order is victimsScan's exactly: on seeded random stores
/// (`randomStore`; turns 0, 1, 2 or 4) victims(null) and victims(cv) for every conversation equal victimsScan's entry
/// for entry (key, why, order). The C allocator keeps this check fast in Debug; the checks that make room keep the
/// testing allocator's leak check.
fn checkVictimsGrouped(_: Allocator) !void {
    const gpa = std.heap.c_allocator;
    const h = try H.init(gpa, .{ .tag = "victims" });
    defer h.deinit();
    var prng = std.Random.DefaultPrng.init(0x4a33);
    const rnd = prng.random();
    for (0..500) |_| {
        h.s.testClear();
        h.s.cfg.turns = ([_]u32{ 0, 1, 2, 4 })[rnd.uintLessThan(usize, 4)];
        try randomStore(h, rnd, rnd.intRangeAtMost(usize, 1, 200));
        try ensure(try sameVictims(gpa, h), @src());
    }
}

/// The cost guard, in inner steps counted rather than time: a store of 2,000 conversations of two turns each costs
/// victims() a few steps a point, where victimsScan walks every point for every conversation.
fn checkVictimsCost(_: Allocator) !void {
    const gpa = std.heap.c_allocator; // as the differential test, for speed in Debug
    const h = try H.init(gpa, .{ .tag = "victimcost", .turns = 4 });
    defer h.deinit();
    var prng = std.Random.DefaultPrng.init(0x4a34);
    const rnd = prng.random();
    var ids: [2 * block_tokens]u32 = undefined;
    for (0..2000) |i| {
        for (&ids) |*x| x.* = rnd.intRangeAtMost(u32, 1, 150000);
        _ = try h.s.testInsert(ids[0..block_tokens], lin(@truncate(i)), false, 0, @intCast(i));
        _ = try h.s.testInsert(&ids, lin(@truncate(i)), false, 0, @intCast(i + 1));
    }
    try ensure(h.s.points.count() == 4000, @src());
    const before = Store.test_victim_steps.load(.monotonic);
    const v = try h.s.testVictims(null, false);
    defer gpa.free(v);
    const steps = Store.test_victim_steps.load(.monotonic) - before;
    // the grouped walk: two visits a member, one more for a whole conversation's rest, at most one path step a member
    try ensure(v.len == 4000 and steps <= 4 * 4000, @src());
}

/// Lookups: the longest complete point the prompt extends within its bounds; media must match.
fn checkLookup(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lookup", .turns = 0 });
    defer h.deinit();
    const ids = try idsOf(gpa, 5 * block_tokens, 1700);
    defer gpa.free(ids);
    const m = [_]format.Media{.{ .start = 3, .len = 4, .digest = @splat(1) }};
    const raw = try gpa.dupe(u32, ids);
    defer gpa.free(raw);
    for (3..7) |i| { // the picture's keyed ids made from its digest, its raw ids the placeholder
        ids[i] = format.pictureId(&m[0].digest, @intCast(i - 3));
        raw[i] = 99;
    }
    // a point whose ids do not agree with its media is refused at keep (and would be at recovery)
    try ensure(keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m }) == .invalid, @src());
    var forged = m;
    forged[0].digest[20] ^= 1; // (the first 8 bytes seed the ids: the same ids, another picture past them)
    const kf = try write(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &forged, .raw = raw });
    var seeded = m;
    seeded[0].digest[0] ^= 1; // (a seed byte changed: its ids no longer derive from it)
    try ensure(keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &seeded, .raw = raw }) == .invalid, @src());
    var placeholder = try gpa.dupe(u32, raw);
    defer gpa.free(placeholder);
    placeholder[4] = ids[4]; // a raw id inside the picture that is a picture's keyed id, not a model id
    try ensure(keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m, .raw = placeholder }) == .invalid, @src());
    const k2 = try write(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m, .raw = raw });
    const k4 = try write(h, ids[0 .. 4 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m, .raw = raw });
    const q = struct {
        fn at(hh: *H, x: store.Query) ?Digest {
            return if (hh.s.lookup(x)) |hit| hit.key else null;
        }
    };
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 70 }), k4), @src());
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 50 }), k2), @src());
    try ensure(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 50, .longer_than = 40 }) == null, @src());
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 80, .grid = 64 }), k4), @src());
    var other = m;
    other[0].digest[0] = 2;
    try ensure(q.at(h, .{ .keyed = ids, .media = &other, .max_at = 80 }) == null, @src());
    // the same shortened ids from another payload (only the full digest tells them apart): its own point, never m's
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &forged, .max_at = 80 }), kf), @src());
    try ensure(!std.meta.eql(kf, k2), @src());
    try ensure(q.at(h, .{ .keyed = ids, .media = &.{}, .max_at = 80 }) == null, @src());
    var changed = try gpa.dupe(u32, ids);
    defer gpa.free(changed);
    changed[20] +%= 1;
    try ensure(std.meta.eql(q.at(h, .{ .keyed = changed, .media = &m, .max_at = 80 }), null), @src());
    // across a restart: the points with pictures recover (their ids checked against their media) and match as before
    try h.reopen();
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 70 }), k4), @src());
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &forged, .max_at = 80 }), kf), @src());
    try ensure(q.at(h, .{ .keyed = ids, .media = &other, .max_at = 80 }) == null, @src());
}

/// Replace a point file's bytes (same name, truncated first).
fn rewritePointFile(h: *H, key: Digest, suffix: []const u8, bytes: []const u8) !void {
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, key, suffix)}, 0);
    const fd = c.openat(h.s.points_fd, name, .{ .ACCMODE = .WRONLY, .TRUNC = true, .CLOEXEC = true }, @as(c.mode_t, 0));
    if (fd < 0) return CheckFailed;
    defer _ = c.close(fd);
    if (c.pwrite(fd, bytes.ptr, bytes.len, 0) != @as(isize, @intCast(bytes.len))) return CheckFailed;
}

/// A picture point's raw ids altered on disk with every CRC made good (the .ids file's in the manifest, the
/// manifest's trailer): raw ids are not in the point key, so only recovery's `format.mediaIds` check catches it. A
/// raw id inside the picture made a picture id, and a raw id outside it changed, each drop their point; an untouched
/// picture point beside them recovers and still matches.
fn checkPictureRecovery(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "picrecover", .turns = 0 });
    defer h.deinit();
    const m = [_]format.Media{.{ .start = 3, .len = 4, .digest = @splat(7) }};
    const n = 2 * block_tokens;
    var keys: [3]Digest = undefined;
    var keyed: [3][]u32 = undefined;
    for (0..3) |i| {
        keyed[i] = try idsOf(gpa, n, 1900 + i);
        const raw = try gpa.dupe(u32, keyed[i]);
        defer gpa.free(raw);
        for (3..7) |j| {
            keyed[i][j] = format.pictureId(&m[0].digest, @intCast(j - 3));
            raw[j] = 99;
        }
        keys[i] = try write(h, keyed[i], .{ .lineage = lin(@intCast(80 + i)), .source = 80 + i, .media = &m, .raw = raw });
    }
    defer for (keyed) |k| gpa.free(k);
    for (0..2) |i| {
        var nb: [96]u8 = undefined;
        var zb: [96]u8 = undefined;
        var ids_file = try fsio.readSmall(&h.s.writer, gpa, h.s.points_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, keys[i], ".ids")}, 0), 1 << 20);
        defer ids_file.deinit(gpa);
        const ids_raw = ids_file.bytes();
        try ensure(ids_raw.len >= 2 * n * 4, @src());
        const raw_ids: []align(1) u32 = std.mem.bytesAsSlice(u32, ids_raw[n * 4 ..][0 .. n * 4]);
        if (i == 0) raw_ids[4] = keyed[i][4] else raw_ids[10] +%= 1; // a picture id inside, another model id outside
        try rewritePointFile(h, keys[i], ".ids", ids_raw);
        var man_file = try fsio.readSmall(&h.s.writer, gpa, h.s.points_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, keys[i], ".man")}, 0), 1 << 20);
        defer man_file.deinit(gpa);
        const man = man_file.bytes();
        var mf = try format.Manifest.parse(gpa, man, h.cfg.compat, h.cfg.limits);
        defer mf.deinit(gpa);
        const same = try mf.encode(gpa, h.cfg.compat);
        defer gpa.free(same);
        try ensure(std.mem.eql(u8, same, man), @src()); // (the re-encoding is the manifest's own bytes)
        mf.ids_crc = format.crc32(ids_raw);
        const fixed = try mf.encode(gpa, h.cfg.compat);
        defer gpa.free(fixed);
        try rewritePointFile(h, keys[i], ".man", fixed);
    }
    try h.reopen();
    try ensure(h.s.counts().recovered_dropped == 2, @src());
    try ensure(!h.s.presence(keys[0]) and !h.s.presence(keys[1]) and h.s.presence(keys[2]), @src());
    for (0..2) |i| try ensure(h.s.lookup(.{ .keyed = keyed[i], .media = &m, .max_at = 80 }) == null, @src());
    const hit = h.s.lookup(.{ .keyed = keyed[2], .media = &m, .max_at = 80 }) orelse return CheckFailed;
    try ensure(std.mem.eql(u8, &hit.key, &keys[2]), @src());
}

/// Disk faults on the write path: ENOSPC fails the point and leaves no temporary file or staging behind; short
/// writes and EINTR still write every byte; a file system without O_DIRECT is written buffered.
fn checkWriteFaults(gpa: Allocator) !void {
    {
        const h = try H.init(gpa, .{ .tag = "enospc" });
        defer h.deinit();
        const ids = try idsOf(gpa, 3 * block_tokens, 1800);
        defer gpa.free(ids);
        h.hk.nospace_writes = 1;
        try ensure(keepIds(h, ids, .{ .lineage = lin(80), .source = 80 }) == .queued, @src());
        _ = settle(h);
        var comp: [4]store.Completion = undefined;
        const n = h.s.poll(&comp);
        try ensure(n == 1 and comp[0] == .failed and comp[0].failed.why == .no_space, @src());
        try ensure(!h.s.presence(format.pointKey(ids, block_tokens, &.{})), @src());
        try ensure(countFiles(gpa, h.s.blocks_fd, true) == 0 and countFiles(gpa, h.s.points_fd, false) == 0, @src());
        try ensure(h.s.counts().staging_bytes == 0, @src());
    }
    inline for (.{ false, true }) |nd| {
        const h = try H.init(gpa, .{ .tag = "short", .no_direct = nd });
        defer h.deinit();
        const ids = try idsOf(gpa, 3 * block_tokens + 9, 1801);
        defer gpa.free(ids);
        h.hk.short_intr = true;
        const key = try write(h, ids, .{ .lineage = lin(81), .source = 81 });
        h.hk.short_intr = false;
        try h.reopen();
        const got = (try restore(h, key)) orelse return CheckFailed;
        defer gpa.free(got);
        try ensure(std.mem.eql(u8, got, h.src(81)[0 .. ids.len * bpt]), @src());
    }
}

// -- the restore quarantine marker ---------------------------------------------------------------------------------

/// Q's rows as stored: its first block is P's (shared by name, written once from P's source 310), its second its own.
fn qRows(h: *H) ![]u8 {
    const out = try h.gpa.alloc(u8, 2 * block_bytes);
    @memcpy(out[0..block_bytes], h.src(310)[0..block_bytes]);
    @memcpy(out[block_bytes..], h.src(311)[block_bytes .. 2 * block_bytes]);
    return out;
}

/// P (two blocks) and Q (P's first block and one of its own: the first block shared by name), both written.
fn sharedPair(h: *H) ![2]Digest {
    const gpa = h.gpa;
    const ip = try idsOf(gpa, 2 * block_tokens, 3100);
    defer gpa.free(ip);
    const more = try idsOf(gpa, block_tokens, 3101);
    defer gpa.free(more);
    const iq = try cat(gpa, ip[0..block_tokens], more);
    defer gpa.free(iq);
    const kp = try write(h, ip, .{ .lineage = lin(40), .source = 310 });
    const kq = try write(h, iq, .{ .lineage = lin(40), .source = 311 });
    try ensure(std.mem.eql(u8, &h.s.points.get(kp).?.blocks[0].name, &h.s.points.get(kq).?.blocks[0].name), @src());
    return .{ kp, kq };
}

fn smallReady(x: struct { *H, store.ReadId }) bool {
    return x[0].s.readSmallView(x[1]) != null;
}

fn hasFile(h: *H, key: Digest, suffix: []const u8) bool {
    var nb: [96]u8 = undefined;
    return fileExists(h.s.points_fd, pointFile(&nb, key, suffix));
}

/// No marker of `key` on disk and none in the store's books.
fn markerGone(x: struct { *H, Digest }) bool {
    const h = x[0];
    h.s.mutex.lockUncancelable(io);
    const booked = h.s.marks.contains(x[1]);
    h.s.mutex.unlock(io);
    return !booked and !hasFile(h, x[1], ".rst");
}

/// A restore that crashed the engine cannot crash it again from disk. P's restore has read its small state (its
/// marker is durable) when the process "dies" (reopened without endRead): the open forgets P (manifest, ids, small
/// state and marker gone, counted), and only P: Q, which shares P's first block, recovers and restores whole.
fn checkMarkerForgets(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "markerforget" });
    defer h.deinit();
    const k = try sharedPair(h);
    const id = try h.s.beginRead(k[0]);
    try ensure(waitUntil(smallReady, .{ h, id }, 5000), @src());
    try ensure(hasFile(h, k[0], ".rst"), @src());
    try h.reopen(); // (the crash: no endRead)
    for ([_][]const u8{ ".man", ".ids", ".st", ".rst" }) |sfx| try ensure(!hasFile(h, k[0], sfx), @src());
    const ct = h.s.counts();
    try ensure(ct.quarantined == 1 and ct.recovered_points == 1 and !h.s.presence(k[0]), @src());
    const ip = try idsOf(gpa, 2 * block_tokens, 3100);
    defer gpa.free(ip);
    if (h.s.lookup(.{ .keyed = ip, .max_at = max_tokens })) |hit| try ensure(!std.mem.eql(u8, &hit.key, &k[0]), @src());
    const got = (try restore(h, k[1])) orelse return CheckFailed;
    defer gpa.free(got);
    const want = try qRows(h);
    defer gpa.free(want);
    try ensure(std.mem.eql(u8, got, want), @src());
    try ensure(countFiles(gpa, h.s.blocks_fd, false) == h.s.have.count() and h.s.have.count() == 2, @src());
}

var cancel_id: store.ReadId = 0;
fn cancelDuring(h: *H) void {
    _ = h.s.endRead(cancel_id, .cancel) catch {};
}

/// ... the marker survives the commit until the tier says the restored state is in use (restored(key)): a reopen
/// between the two forgets P; after restored(P) it keeps P. An abort, a cancel, and a cancel while the marker is
/// being written leave no marker behind.
fn checkMarkerCleared(gpa: Allocator) !void {
    {
        const h = try H.init(gpa, .{ .tag = "markercommit" });
        defer h.deinit();
        const k = try sharedPair(h);
        const got = (try restoreCommit(h, k[0])) orelse return CheckFailed; // (commits; no round yet)
        gpa.free(got);
        try ensure(hasFile(h, k[0], ".rst"), @src());
        try h.reopen(); // (a crash after the commit, before the first round)
        try ensure(!h.s.presence(k[0]) and h.s.counts().quarantined == 1, @src());
    }
    {
        const h = try H.init(gpa, .{ .tag = "markerrestored" });
        defer h.deinit();
        const k = try sharedPair(h);
        const got = (try restoreCommit(h, k[0])) orelse return CheckFailed;
        gpa.free(got);
        try ensure(hasFile(h, k[0], ".rst"), @src());
        h.s.restored(k[0]);
        try ensure(waitUntil(markerGone, .{ h, k[0] }, 3000), @src());
        try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
        try h.reopen();
        try ensure(h.s.presence(k[0]) and h.s.counts().quarantined == 0, @src());
    }
    for ([_]Store.Outcome{ .abort, .cancel }) |how| {
        const h = try H.init(gpa, .{ .tag = "markerend" });
        defer h.deinit();
        const k = try sharedPair(h);
        const id = try h.s.beginRead(k[0]);
        try ensure(waitUntil(smallReady, .{ h, id }, 5000), @src());
        _ = try h.s.endRead(id, how);
        try ensure(waitUntil(markerGone, .{ h, k[0] }, 3000), @src());
        try h.reopen();
        try ensure(h.s.presence(k[0]) and h.s.counts().quarantined == 0, @src());
    }
    {
        const h = try H.init(gpa, .{ .tag = "markercancel" });
        defer h.deinit();
        const k = try sharedPair(h);
        h.hk.action_step = .written;
        h.hk.action_suffix = ".rst";
        h.hk.action = cancelDuring;
        cancel_id = h.s.next_id; // (set before beginRead returns: its reader may reach the marker first)
        try ensure(try h.s.beginRead(k[0]) == cancel_id, @src());
        try ensure(waitUntil(markerGone, .{ h, k[0] }, 3000) and h.hk.action == null, @src());
        try h.reopen();
        try ensure(h.s.presence(k[0]) and h.s.counts().quarantined == 0, @src());
    }
}

const Gate = struct { id: store.ReadId = 0, renamed_null: ?bool = null, marked_null: ?bool = null, marked_at: usize = 0 };
var gate: Gate = .{};
fn gateObserve(h: *H, step: fsio.Step, name: []const u8) void {
    if (!std.mem.endsWith(u8, name, ".rst")) return;
    const small_null = h.s.readSmallView(gate.id) == null;
    switch (step) {
        .renamed => gate.renamed_null = small_null,
        .marked => {
            gate.marked_null = small_null;
            h.hk.mutex.lockUncancelable(io);
            gate.marked_at = h.hk.events.items.len;
            h.hk.mutex.unlock(io);
        },
        else => {},
    }
}

/// ... the small state reads as ready only once the marker is durable: its write, chown, fsync, pages dropped,
/// rename and the points folder's fsync, in that order, all before the step that says it is marked; the small view
/// still null at the rename and at the mark.
fn checkMarkerGates(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "markergate" });
    defer h.deinit();
    const k = try sharedPair(h);
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    h.hk.clear();
    gate = .{};
    h.hk.observe = gateObserve;
    defer h.hk.observe = null;
    // (the id is known only once beginRead returns; the reader may start first: the observer reads gate.id late)
    gate.id = h.s.next_id;
    const id = try h.s.beginRead(k[0]);
    try ensure(id == gate.id, @src());
    try ensure(waitUntil(smallReady, .{ h, id }, 5000), @src());
    _ = try h.s.endRead(id, .cancel);
    var nb: [96]u8 = undefined;
    const marker = pointFile(&nb, k[0], ".rst");
    const want = [_]fsio.Event{ .write, .chown, .fsync, .drop_pages, .rename };
    var seen: usize = 0;
    var dir_sync: ?usize = null;
    h.hk.mutex.lockUncancelable(io);
    for (h.hk.events.items, 0..) |e, i| {
        const n = e.name[0..e.len];
        if (seen < want.len and std.mem.eql(u8, n, marker)) {
            if (e.ev == want[seen]) seen += 1;
        } else if (seen == want.len and dir_sync == null and e.ev == .dir_fsync and std.mem.eql(u8, n, "points")) dir_sync = i;
    }
    h.hk.mutex.unlock(io);
    try ensure(seen == want.len and dir_sync != null and dir_sync.? < gate.marked_at, @src());
    try ensure(gate.renamed_null == true and gate.marked_null == true, @src());
}

/// ... a marker that cannot be made (ENOSPC) fails the restore as an I/O error, never as damage: the point stays,
/// on disk and after a reopen, and the failure is counted.
fn checkMarkerFailure(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "markerfail" });
    defer h.deinit();
    const k = try sharedPair(h);
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    h.hk.nospace_writes = 1; // (the next pwrite is the marker's)
    const id = try h.s.beginRead(k[0]);
    try ensure(waitUntil(struct {
        fn f(x: struct { *H, store.ReadId }) bool {
            return x[0].s.readFailure(x[1]).err != null;
        }
    }.f, .{ h, id }, 5000), @src());
    try ensure(h.s.readFailure(id).err.? == .io and h.s.readSmallView(id) == null, @src());
    _ = try h.s.endRead(id, .abort);
    try ensure(h.s.counts().marker_failed == 1 and h.s.presence(k[0]) and !hasFile(h, k[0], ".rst"), @src());
    try h.reopen();
    try ensure(h.s.presence(k[0]) and h.s.counts().quarantined == 0, @src());
}

/// An open after a crash that left a restore marker (P's), a stale marker (a key with no point) and a truncated
/// manifest (R's) together: recovery forgets exactly the marked point, drops the damaged one, removes the stale
/// marker, and keeps every other point (Q, which shares P's first block, and S) whole.
fn checkMarkerWithDamage(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "markerdamage" });
    defer h.deinit();
    const k = try sharedPair(h);
    const ir = try idsOf(gpa, 2 * block_tokens + 3, 3102);
    defer gpa.free(ir);
    const kr = try write(h, ir, .{ .lineage = lin(41), .source = 312 });
    const is = try idsOf(gpa, block_tokens + 7, 3103);
    defer gpa.free(is);
    const ks = try write(h, is, .{ .lineage = lin(42), .source = 313 });
    const id = try h.s.beginRead(k[0]);
    try ensure(waitUntil(smallReady, .{ h, id }, 5000) and hasFile(h, k[0], ".rst"), @src());
    h.closeStore(); // (the crash: no endRead)
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const stale: Digest = @splat(0x77);
    const pfd = blk: { // (the store is closed: its folders by path)
        var pb: [512]u8 = undefined;
        var nsb: [64]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}/rank0/points", .{ h.root, namespaceOf(&nsb, h.cfg.compat) }, 0);
        break :blk try fsio.openDir(path);
    };
    defer _ = c.close(pfd);
    const mk = c.openat(pfd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, stale, ".rst")}, 0), .{ .ACCMODE = .WRONLY, .CREAT = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    try ensure(mk >= 0, @src());
    _ = c.pwrite(mk, &stale, stale.len, 0);
    _ = c.close(mk);
    const mfd = c.openat(pfd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, kr, ".man")}, 0), .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    try ensure(mfd >= 0 and c.ftruncate(mfd, 100) == 0, @src());
    _ = c.close(mfd);
    try h.open();
    const ct = h.s.counts();
    try ensure(ct.quarantined == 1 and ct.recovered_dropped == 1 and ct.recovered_points == 2, @src());
    try ensure(!h.s.presence(k[0]) and !h.s.presence(kr) and h.s.presence(k[1]) and h.s.presence(ks), @src());
    try ensure(!fileExists(h.s.points_fd, pointFile(&nb, stale, ".rst")) and !hasFile(h, k[0], ".rst"), @src());
    try ensure(countFiles(gpa, h.s.points_fd, false) == 2 * 3, @src());
    try ensure(countFiles(gpa, h.s.blocks_fd, false) == h.s.have.count() and h.s.have.count() == 2 + 2, @src());
    const q_rows = try qRows(h);
    defer gpa.free(q_rows);
    for ([_]struct { Digest, []const u8 }{ .{ k[1], q_rows }, .{ ks, h.src(313)[0 .. (block_tokens + 7) * bpt] } }) |x| {
        const got = (try restore(h, x[0])) orelse return CheckFailed;
        defer gpa.free(got);
        try ensure(std.mem.eql(u8, got, x[1]), @src());
    }
}

fn ioprioOf(tid: std.Thread.Id) !usize {
    const L = std.os.linux;
    const rc = L.syscall2(.ioprio_get, 1, tid); // IOPRIO_WHO_PROCESS: a thread id (0: the caller)
    if (L.errno(rc) != .SUCCESS) return error.Unexpected;
    return rc;
}

fn threadsStarted(h: *H) bool {
    h.s.mutex.lockUncancelable(io);
    defer h.s.mutex.unlock(io);
    return h.s.thread_ids.items.len == h.s.reader_threads.items.len + 1;
}

/// The store's writer and readers run at the configured I/O priority (TF_DS_SPILL_IOPRIO, best-effort 7 by default);
/// null leaves every thread at the process's priority; a value the kernel refuses is counted and never fatal.
fn checkIoPriority(gpa: Allocator) !void {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const be7: u16 = (2 << 13) | 7; // IOPRIO_CLASS_BE, level 7
    const own = try ioprioOf(0);
    for ([_]?u16{ be7, null, (5 << 13) | 1 }) |want| {
        const h = try H.init(gpa, .{ .tag = "ioprio", .io_priority = want });
        defer h.deinit();
        try ensure(waitUntil(threadsStarted, h, 3000), @src());
        h.s.mutex.lockUncancelable(io);
        const tids = try gpa.dupe(std.Thread.Id, h.s.thread_ids.items);
        h.s.mutex.unlock(io);
        defer gpa.free(tids);
        const invalid = want != null and want.? >> 13 == 5;
        for (tids) |tid| {
            const got = try ioprioOf(tid);
            try ensure(got == (if (want != null and !invalid) want.? else own), @src());
        }
        try ensure(h.s.counts().ioprio_refused == (if (invalid) tids.len else 0), @src());
        const ids = try idsOf(gpa, 2 * block_tokens, 3200);
        defer gpa.free(ids);
        const key = try write(h, ids, .{ .lineage = lin(43), .source = 320 });
        const back = (try restore(h, key)) orelse return CheckFailed;
        defer gpa.free(back);
        try ensure(std.mem.eql(u8, back, h.src(320)[0 .. 2 * block_tokens * bpt]), @src());
    }
}

/// The store lock's wait and hold times: a holder keeping the lock 80 ms on another thread makes an engine-thread call
/// wait at least 50 ms (the call starts after the holder has it), and records a hold of at least 80 ms; the buckets
/// count every time once.
fn checkLockTimes(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "locktimes" });
    defer h.deinit();
    var held = std.atomic.Value(bool).init(false);
    const t = try std.Thread.spawn(.{}, Store.testHoldLock, .{ h.s, 80 * std.time.ns_per_ms, &held });
    try ensure(waitUntil(struct {
        fn f(x: *std.atomic.Value(bool)) bool {
            return x.load(.acquire);
        }
    }.f, &held, 3000), @src());
    const ct = h.s.counts(); // (waits for the holder)
    t.join();
    try ensure(ct.lock_wait.max_ns >= 50 * std.time.ns_per_ms and ct.lock_hold.max_ns >= 80 * std.time.ns_per_ms, @src());
    for ([_]store.LockTimes{ ct.lock_wait, ct.lock_hold }) |lt| {
        var in_buckets: u64 = 0;
        for (lt.le_us) |b| in_buckets += b;
        try ensure(lt.count > 0 and in_buckets == lt.count and lt.sum_ns >= lt.max_ns, @src());
    }
}

fn stuckLong(_: *H) void {
    std.Io.sleep(io, .fromMilliseconds(600), .awake) catch {};
}

/// ... and a thread waiting on one of the store's conditions does not count as holding it: an idle store, and a drain
/// that gives up (in timed slices) on a writer held in a hook, show no wait or hold of 100 ms or more (a third of the
/// drain's 300 ms, which leaves room for a loaded machine).
fn checkLockIdle(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lockidle" });
    defer h.deinit();
    std.Io.sleep(io, .fromMilliseconds(200), .awake) catch {};
    const ids = try idsOf(gpa, 2 * block_tokens, 3300);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(44), .source = 330 }); // (the writer and readers woken, idle again)
    std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
    try ensure(h.s.counts().lock_hold.max_ns < 100 * std.time.ns_per_ms, @src());
    h.hk.action_step = .written;
    h.hk.action_suffix = ".man";
    h.hk.action = stuckLong;
    try ensure(keepIds(h, ids, .{ .lineage = lin(44), .source = 330, .shared = true }) == .stored, @src());
    try ensure(!h.s.drain(300 * std.time.ns_per_ms).done, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done and manifestShared(h, key) == true, @src());
    const ct = h.s.counts();
    try ensure(ct.lock_wait.max_ns < 100 * std.time.ns_per_ms and ct.lock_hold.max_ns < 100 * std.time.ns_per_ms, @src());
}

/// A restore cancelled while its reads are in flight gives every slot back, and the point back to eviction.
fn checkCancelRead(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "cancelread", .read_slots = 3 });
    defer h.deinit();
    const ids = try idsOf(gpa, 8 * block_tokens, 1900);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(90), .source = 90 });
    const id = try h.s.beginRead(key);
    var rb: [1]store.ReadyBlock = undefined;
    _ = h.s.readAwait(id, 0, 5 * std.time.ns_per_s, .{});
    const got = h.s.readReady(id, null, &rb);
    _ = try h.s.endRead(id, .cancel);
    for (rb[0..got.n]) |b| h.s.readRelease(b.slot);
    try ensure(waitUntil(struct {
        fn f(hh: *H) bool {
            const ct = hh.s.counts();
            return ct.staging_bytes == 0 and hh.s.reads.items.len == 0;
        }
    }.f, h, 3000), @src());
    try ensure(!h.s.reading.contains(key), @src());
}

/// Two restores share the read slots: each gets blocks in turn.
fn checkFairReads(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "fair", .read_slots = 2 });
    defer h.deinit();
    var ids: [2]store.ReadId = undefined;
    for (0..2) |i| {
        const x = try idsOf(gpa, 6 * block_tokens, 2000 + i);
        defer gpa.free(x);
        const key = try write(h, x, .{ .lineage = lin(@intCast(100 + i)), .source = 100 + i });
        ids[i] = try h.s.beginRead(key);
    }
    // the first restore took both slots (the second was not there yet); the slots it gives back go to the second
    try ensure(h.s.readAwait(ids[0], 0, 5 * std.time.ns_per_s, .{}) and h.s.readAwait(ids[0], 1, 5 * std.time.ns_per_s, .{}), @src());
    var rb: [2]store.ReadyBlock = undefined;
    const got = h.s.readReady(ids[0], null, &rb);
    try ensure(got.n == 2, @src());
    h.s.readRelease(rb[0].slot);
    try ensure(h.s.readAwait(ids[1], 0, 5 * std.time.ns_per_s, .{}), @src());
    h.s.readRelease(rb[1].slot);
    for (ids) |id| _ = try h.s.endRead(id, .cancel);
    try ensure(waitUntil(struct {
        fn f(hh: *H) bool {
            return hh.s.counts().staging_bytes == 0;
        }
    }.f, h, 3000), @src());
}

/// Replicated and owner-sharded points never read as each other: a store of another layout drops them as foreign.
fn checkLayout(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "layout" });
    defer h.deinit();
    const ids = try idsOf(gpa, 2 * block_tokens, 2300);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(23), .source = 23 });
    h.cfg.owner_layout = .owner_sharded;
    try h.reopen();
    try ensure(!h.s.presence(key) and h.s.counts().recovered_dropped == 1, @src());
}

/// 0118's owner-sharded store: a rank keeps a point of its own part with full blocks of every plane and a point of
/// another part with only the replicated planes (foreign_block_bytes); every point names its part (a part-less keep is
/// invalid), the sizes survive a restart, a lookup and partOf say the part, the same ids under two parts are two
/// block names (the salt), and a rank that owns another part reads the first rank's manifests as damaged.
fn checkOwnerParts(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "parts" });
    defer h.deinit();
    const foreign = block_bytes / 4;
    h.cfg.owner_layout = .owner_sharded;
    h.cfg.limits.foreign_block_bytes = foreign;
    h.cfg.limits.own_part = 0;
    try h.reopen();
    const ids = try idsOf(gpa, 2 * block_tokens, 2500);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids, .{ .lineage = lin(25), .source = 25 }) == .invalid, @src()); // (no part)
    const own = try write(h, ids, .{ .lineage = lin(25).inPart(0), .source = 25 });
    const other_ids = try idsOf(gpa, 3 * block_tokens, 2501);
    defer gpa.free(other_ids);
    const other = try write(h, other_ids, .{ .lineage = lin(26).inPart(1), .source = 26 });
    for (h.s.points.get(own).?.blocks) |b| try ensure(b.bytes == block_bytes, @src());
    for (h.s.points.get(other).?.blocks) |b| try ensure(b.bytes == foreign, @src());
    // the same ids from another part: other names (never the first point's blocks by name)
    const pa = h.s.points.get(own).?;
    var spans: [4]format.Span = undefined;
    var ch: std.ArrayList(Digest) = .empty;
    defer ch.deinit(gpa);
    try format.chain(gpa, ids, block_tokens, &ch);
    format.blockNames(lin(25).inPart(1), ch.items, ids, block_tokens, spans[0..2]);
    for (pa.blocks, spans[0..2]) |b, sp| try ensure(!std.mem.eql(u8, &b.name, &sp.name), @src());
    format.blockNames(lin(25).inPart(1).inPart(0), ch.items, ids, block_tokens, spans[0..2]); // (re-salted: replaced)
    for (pa.blocks, spans[0..2]) |b, sp| try ensure(std.mem.eql(u8, &b.name, &sp.name), @src());
    // the restored rows are the stored sizes: a foreign block is its source's first bytes of the block's rows
    const got = (try restore(h, other)) orelse return CheckFailed;
    defer gpa.free(got);
    for (0..3) |k| try ensure(std.mem.eql(u8, got[k * block_tokens * bpt ..][0..foreign], h.src(26)[k * block_tokens * bpt ..][0..foreign]), @src());
    try h.reopen();
    try ensure(h.s.presence(own) and h.s.presence(other), @src());
    try ensure(h.s.partOf(own).?.part == 0 and h.s.partOf(other).?.part == 1 and h.s.partOf(@splat(9)) == null, @src());
    const hit = h.s.lookup(.{ .keyed = other_ids, .max_at = max_tokens }) orelse return CheckFailed;
    try ensure(hit.part == 1 and std.mem.eql(u8, &hit.key, &other), @src());
    // a rank owning part 1 with these files: the sizes are not its own, both points damaged
    h.cfg.limits.own_part = 1;
    try h.reopen();
    try ensure(!h.s.presence(own) and !h.s.presence(other) and h.s.counts().recovered_dropped == 2, @src());
    // a foreign size past the block's, or one on a replicated store, is no configuration
    h.closeStore();
    h.cfg.limits.foreign_block_bytes = block_bytes + 1;
    try ensure(if (Store.open(gpa, io, h.cfg)) |_| false else |e| e == error.BadConfig, @src());
    h.cfg.limits.foreign_block_bytes = foreign;
    h.cfg.owner_layout = .replicated;
    try ensure(if (Store.open(gpa, io, h.cfg)) |_| false else |e| e == error.BadConfig, @src());
}

/// GLM 0110's pin (review of the native port): a resumed conversation that is the least recently used one, kept
/// under cap pressure, makes room without evicting the blocks it reuses. Its old turn goes (room), its blocks stay
/// (the new turn names them), the new turn stages only its new block, publishes and restores.
fn checkKeepReuse(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "reuse", .cap = 24 * 1024 });
    defer h.deinit();
    const a = try idsOf(gpa, 3 * block_tokens, 2400);
    defer gpa.free(a);
    const ka1 = try write(h, a[0 .. 2 * block_tokens], .{ .lineage = lin(24), .source = 24 });
    for (0..2) |i| {
        const x = try idsOf(gpa, block_tokens, 2401 + i);
        defer gpa.free(x);
        _ = try write(h, x, .{ .lineage = lin(@intCast(25 + i)), .source = 25 + i });
    }
    const before = h.s.counts();
    const ka2 = try write(h, a, .{ .lineage = lin(24), .source = 24 });
    const after = h.s.counts();
    try ensure(!h.s.points.contains(ka1) and after.pruned_conversations > before.pruned_conversations, @src()); // (the pressure)
    try ensure(after.staged_blocks - before.staged_blocks == 1 and after.stored_bytes <= h.cfg.cap_bytes, @src());
    const got = (try restore(h, ka2)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(24)[0 .. 3 * block_tokens * bpt]), @src());
}

/// A cancelled write's unstaged blocks go to another waiting write that names them, and so does their
/// reservation: the backlog, cap and floor keep counting those bytes until they are written.
fn checkHandOver(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "handover" });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 2600);
    defer gpa.free(ids);
    const r1 = keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(28), .source = 28 });
    try ensure(r1 == .queued, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(28), .source = 28 }) == .queued, @src());
    try ensure(h.s.cancel(r1.queued.job), @src());
    try ensure(h.s.counts().queued_bytes >= 3 * block_bytes, @src()); // the second write's 3 blocks, all reserved
    try ensure(settle(h).done, @src());
    const got = (try restore(h, format.pointKey(ids, block_tokens, &.{}))) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(28)[0 .. 3 * block_tokens * bpt]), @src());
    try ensure(h.s.counts().queued_bytes == 0, @src());
}

/// A quota that is not a fraction in [0, 1] is refused at open.
fn checkBadConfig(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "badcfg" });
    defer h.deinit();
    h.closeStore();
    for ([_]f64{ std.math.nan(f64), -0.1, 1.5, std.math.inf(f64) }) |q| {
        var cfg = h.cfg;
        cfg.quota = q;
        if (Store.open(gpa, io, cfg)) |s2| {
            s2.close();
            try ensure(false, @src());
        } else |e| try ensure(e == error.BadConfig, @src());
    }
}

fn chmodAt(dirfd: c.fd_t, name: []const u8, mode: c.mode_t) !void {
    var zb: [128]u8 = undefined;
    const z = try std.fmt.bufPrintSentinel(&zb, "{s}", .{name}, 0);
    if (c.fchmodat(dirfd, z, mode, 0) != 0) return error.Unexpected;
}

/// Recovery deletes only damage: a point it cannot read for another reason (here: its ids, or one of its blocks,
/// unreadable) keeps every file, and so does every block; the next open recovers it whole.
fn checkRecoverTransient(gpa: Allocator) !void {
    if (c.geteuid() == 0) return; // (permissions do not stop root)
    const h = try H.init(gpa, .{ .tag = "transient" });
    defer h.deinit();
    const ia = try idsOf(gpa, 2 * block_tokens + 3, 2500);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, 2 * block_tokens, 2501);
    defer gpa.free(ib);
    const ic = try idsOf(gpa, block_tokens, 2502);
    defer gpa.free(ic);
    const ka = try write(h, ia, .{ .lineage = lin(25), .source = 250 });
    const kb = try write(h, ib, .{ .lineage = lin(26), .source = 251 });
    const kc = try write(h, ic, .{ .lineage = lin(27), .source = 252 });
    const files = countFiles(gpa, h.s.blocks_fd, false) + countFiles(gpa, h.s.points_fd, false);
    var nb: [96]u8 = undefined;
    const b0 = (h.s.points.get(kb) orelse return CheckFailed).blocks[0].name;
    try chmodAt(h.s.points_fd, pointFile(&nb, ka, ".ids"), 0);
    try chmodAt(h.s.blocks_fd, blockFile(&nb, b0), 0);
    try h.reopen();
    const ct = h.s.counts();
    const deferred = !h.s.points.contains(ka) and !h.s.points.contains(kb) and h.s.points.contains(kc);
    const kept = countFiles(gpa, h.s.blocks_fd, false) + countFiles(gpa, h.s.points_fd, false);
    chmodAt(h.s.points_fd, pointFile(&nb, ka, ".ids"), 0o600) catch {}; // (gone if recovery deleted them)
    chmodAt(h.s.blocks_fd, blockFile(&nb, b0), 0o600) catch {};
    try ensure(deferred and ct.recover_deferred == 2 and ct.recovered_dropped == 0, @src());
    try ensure(kept == files, @src());
    try h.reopen();
    try ensure(h.s.points.contains(ka) and h.s.points.contains(kb) and h.s.points.contains(kc), @src());
    const got = (try restore(h, kb)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(251)[0 .. 2 * block_tokens * bpt]), @src());
}

// -- the tests -------------------------------------------------------------------------------------------------------
const Check = struct { name: []const u8, f: *const fn (Allocator) anyerror!void };
const checks = [_]Check{
    .{ .name = "tail", .f = checkTail },
    .{ .name = "read times", .f = checkReadTimes },
    .{ .name = "upto", .f = checkUpto },
    .{ .name = "shared upgrade", .f = checkSharedUpgrade },
    .{ .name = "shared waiting", .f = checkSharedWaiting },
    .{ .name = "shared during write", .f = checkSharedDuringWrite },
    .{ .name = "drop during rewrite", .f = checkDropDuringRewrite },
    .{ .name = "drain waits for rewrites", .f = checkDrainWaitsRewrite },
    .{ .name = "close within its deadline", .f = checkCloseWithin },
    .{ .name = "close runs queued metadata", .f = checkCloseRunsMetadata },
    .{ .name = "forget lost", .f = checkForgetLost },
    .{ .name = "forget damaged", .f = checkForgetDamaged },
    .{ .name = "pages", .f = checkPages },
    .{ .name = "small files leave no pages", .f = checkSmallPages },
    .{ .name = "read fallbacks counted", .f = checkReadFallbacks },
    .{ .name = "publish order", .f = checkPublishOrder },
    .{ .name = "hang", .f = checkHang },
    .{ .name = "carve", .f = checkCarve },
    .{ .name = "budget", .f = checkBudget },
    .{ .name = "crashes", .f = checkCrashes },
    .{ .name = "recovery", .f = checkRecovery },
    .{ .name = "dedup sha", .f = checkDedupSha },
    .{ .name = "guard", .f = checkGuard },
    .{ .name = "guard cancel", .f = checkGuardCancel },
    .{ .name = "eviction order", .f = checkEvictionOrder },
    .{ .name = "turns", .f = checkTurns },
    .{ .name = "fork", .f = checkFork },
    .{ .name = "groups", .f = checkGroups },
    .{ .name = "quota", .f = checkQuota },
    .{ .name = "floor", .f = checkFloor },
    .{ .name = "cap", .f = checkCap },
    .{ .name = "whole conversations at the cap", .f = checkWholeConversation },
    .{ .name = "the floor reads the disk at each keep", .f = checkFloorFresh },
    .{ .name = "small state read errors: I/O is not damage", .f = checkSmallReadErrors },
    .{ .name = "room counts a written block once", .f = checkRoomCountsOnce },
    .{ .name = "owner", .f = checkOwner },
    .{ .name = "prune", .f = checkPrune },
    .{ .name = "prune by last use", .f = checkPruneByLastUse },
    .{ .name = "files owned by the folder owner", .f = checkOwnerFiles },
    .{ .name = "owner failures never fail a write", .f = checkOwnerFailed },
    .{ .name = "keep takes the caller's chain", .f = checkKeepChain },
    .{ .name = "victims grouped = scan", .f = checkVictimsGrouped },
    .{ .name = "victims cost grows with the points", .f = checkVictimsCost },
    .{ .name = "lookup", .f = checkLookup },
    .{ .name = "picture recovery", .f = checkPictureRecovery },
    .{ .name = "write faults", .f = checkWriteFaults },
    .{ .name = "cancel read", .f = checkCancelRead },
    .{ .name = "restore marker forgets at open", .f = checkMarkerForgets },
    .{ .name = "restore marker cleared", .f = checkMarkerCleared },
    .{ .name = "restore marker gates the small state", .f = checkMarkerGates },
    .{ .name = "restore marker failure is not damage", .f = checkMarkerFailure },
    .{ .name = "restore marker with damage", .f = checkMarkerWithDamage },
    .{ .name = "io priority of the store's threads", .f = checkIoPriority },
    .{ .name = "lock wait and hold times", .f = checkLockTimes },
    .{ .name = "lock idle is not a hold", .f = checkLockIdle },
    .{ .name = "fair reads", .f = checkFairReads },
    .{ .name = "last use", .f = checkLastUse },
    .{ .name = "refusals", .f = checkRefusals },
    .{ .name = "layout", .f = checkLayout },
    .{ .name = "keep reuse", .f = checkKeepReuse },
    .{ .name = "recover transient", .f = checkRecoverTransient },
    .{ .name = "hand over", .f = checkHandOver },
    .{ .name = "bad config", .f = checkBadConfig },
    .{ .name = "owner parts", .f = checkOwnerParts },
    .{ .name = "staging cap", .f = checkStagingCap },
};

/// The checks' expected logs stay quiet: three checks provoke the quarantine's warning, and `zig build test` fails a
/// test step whose process writes to stderr.
fn quietLogs() std.log.Level {
    const was = std.testing.log_level;
    std.testing.log_level = .err;
    return was;
}

test "spill store: the GLM 0110 checks and the native port's own, all pass" {
    const was = quietLogs();
    defer std.testing.log_level = was;
    var failed: usize = 0;
    for (checks) |ch| {
        ch.f(testing.allocator) catch |e| {
            if (e == error.SkipZigTest) { // (a file system that cannot show what the check measures, e.g. tmpfs)
                if (std.c.getenv("SPILL_VERBOSE") != null) std.debug.print("SKIP {s}: this file system cannot show it\n", .{ch.name});
                continue;
            }
            std.debug.print("FAIL {s}: {t}\n", .{ ch.name, e });
            failed += 1;
        };
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

/// Which checks each mutant must fail (the GLM mutants that apply to a single-rank store, and the port's own).
const kills = [_]struct { m: mutants.Mutant, checks: []const []const u8 }{
    .{ .m = .tail_staged_later, .checks = &.{"tail"} },
    .{ .m = .loadpart_past_upto, .checks = &.{"upto"} },
    .{ .m = .no_upgrade, .checks = &.{ "shared upgrade", "shared waiting", "shared during write" } },
    .{ .m = .upgrade_not_durable, .checks = &.{"shared upgrade"} },
    .{ .m = .waiting_not_flagged, .checks = &.{"shared waiting"} },
    .{ .m = .written_no_rewrite, .checks = &.{"shared during write"} },
    .{ .m = .rewrite_resurrects, .checks = &.{"drop during rewrite"} },
    .{ .m = .rewrite_not_put, .checks = &.{"shared upgrade"} },
    .{ .m = .rewrite_no_dir_fsync, .checks = &.{"shared upgrade"} },
    .{ .m = .drain_skips_rewrites, .checks = &.{"drain waits for rewrites"} },
    .{ .m = .lost_is_damage, .checks = &.{"forget lost"} },
    .{ .m = .lost_names_kept, .checks = &.{"forget lost"} },
    .{ .m = .damage_not_widened, .checks = &.{"forget damaged"} },
    .{ .m = .small_files_keep_pages, .checks = &.{"pages"} },
    .{ .m = .pages_dropped_before_fsync, .checks = &.{"pages"} },
    .{ .m = .hang_no_recycle_in_wait, .checks = &.{"hang"} },
    .{ .m = .budget_each_read, .checks = &.{"budget"} },
    .{ .m = .budget_same_first, .checks = &.{"budget"} },
    .{ .m = .read_ahead_charges_preallocate, .checks = &.{"carve"} },
    .{ .m = .manifest_not_last, .checks = &.{"publish order"} },
    .{ .m = .scan_keeps_temps, .checks = &.{ "crashes", "recovery" } },
    .{ .m = .recover_no_size_check, .checks = &.{"recovery"} },
    .{ .m = .dedup_without_sha, .checks = &.{"dedup sha"} },
    .{ .m = .guard_cancels_only, .checks = &.{"guard"} },
    .{ .m = .evict_shared_first, .checks = &.{"eviction order"} },
    .{ .m = .floor_ignored, .checks = &.{"floor"} },
    .{ .m = .owner_lock_ignored, .checks = &.{"owner"} },
    .{ .m = .keep_unpinned, .checks = &.{"keep reuse"} },
    .{ .m = .forget_skips_reading, .checks = &.{"forget damaged"} },
    .{ .m = .recover_drops_on_io, .checks = &.{"recover transient"} },
    .{ .m = .handover_unreserved, .checks = &.{"hand over"} },
    .{ .m = .small_files_buffered, .checks = &.{"small files leave no pages"} },
    .{ .m = .prune_by_creation, .checks = &.{"prune by last use"} },
    .{ .m = .owner_not_set, .checks = &.{"files owned by the folder owner"} },
    .{ .m = .keep_rehashes, .checks = &.{"keep takes the caller's chain"} },
    .{ .m = .victims_tie_order, .checks = &.{"victims grouped = scan"} },
    .{ .m = .close_waits, .checks = &.{"close within its deadline"} },
    .{ .m = .marker_skipped, .checks = &.{"restore marker forgets at open"} },
    .{ .m = .marker_not_durable, .checks = &.{"restore marker gates the small state"} },
    .{ .m = .written_still_reserved, .checks = &.{"room counts a written block once"} },
    .{ .m = .conversation_pieces, .checks = &.{"whole conversations at the cap"} },
    .{ .m = .floor_stale, .checks = &.{"the floor reads the disk at each keep"} },
};

fn checkNamed(name: []const u8) *const fn (Allocator) anyerror!void {
    for (checks) |ch| if (std.mem.eql(u8, ch.name, name)) return ch.f;
    @panic("no such check");
}

test "spill store mutants: every authored fix switched back off is caught" {
    // a mutant may leak what its broken path never frees: its runs use the page allocator
    const gpa = std.heap.page_allocator;
    quiet = std.c.getenv("SPILL_VERBOSE") == null;
    const was = quietLogs();
    defer {
        quiet = false;
        mutants.current = .none;
        std.testing.log_level = was;
    }
    var survived: usize = 0;
    var crashes: usize = 0;
    for (kills) |k| {
        mutants.current = k.m;
        if (!quiet) std.debug.print("mutant {t}\n", .{k.m});
        var killed = false;
        var skipped: usize = 0;
        for (k.checks) |name| {
            // a kill is a failed assertion (CheckFailed) only; any other error is the harness or the check
            // breaking on its way (GLM's harness requires a FAIL line, not a crash)
            if (checkNamed(name)(gpa)) |_| {} else |e| if (e == CheckFailed) {
                killed = true;
            } else if (e == error.SkipZigTest) {
                skipped += 1; // (this file system cannot show what the check measures)
            } else {
                std.debug.print("CRASHED {t} in {s}: {t}\n", .{ k.m, name, e });
                crashes += 1;
            }
        }
        mutants.current = .none;
        if (!killed and skipped == k.checks.len) {
            if (!quiet) std.debug.print("SKIP mutant {t}: its checks cannot run on this file system\n", .{k.m});
        } else if (!killed) {
            std.debug.print("SURVIVED {t}\n", .{k.m});
            survived += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), survived);
    try testing.expectEqual(@as(usize, 0), crashes);
}
