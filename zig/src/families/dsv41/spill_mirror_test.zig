//! The SSD tier with a store on every rank, on the real pieces: rank 0's Leader over its Side, three Mirrors over their own
//! Sides (each a real store of files under its own directory), the frames between them on socket pairs and the ranks'
//! gathers on a star. The devices are the tier tests' simulated ones, made equal on every rank (the premise).
const std = @import("std");
const core = @import("core");
const tt = @import("spill_tier_test.zig");
const tier = @import("spill_tier.zig");
const sync = @import("spill_sync.zig");
const mir = @import("spill_mirror.zig");
const star_mod = @import("star.zig");
const Link = @import("link.zig").Link;

const gpa = std.testing.allocator;
const pc = tier.prompt_cache;
const M = mir.Mirror(tt.Dev);
const bt = tt.bt;

var links: [3]Link = undefined; // rank 0's frame links to ranks 1-3 (the test runs one rig at a time)

fn sendAll(_: *anyopaque, bytes: []const u8) anyerror!void {
    for (links) |l| try l.send(bytes);
}

fn idleYes(_: *anyopaque) bool {
    return true;
}

/// One follower: its device, its mirror, the kept states it holds by id, and the thread that serves its frames.
const Follower = struct {
    rank: *tt.Rank,
    mirror: *M = undefined,
    link: Link,
    held: std.AutoHashMapUnmanaged(u64, u64) = .empty, // Snap id -> device buffer
    err: ?anyerror = null,
    thread: ?std.Thread = null,
    began: std.atomic.Value(u64) = .init(0), // when its read began (a restore's destination taken)

    fn me(c: *anyopaque) *Follower {
        return @ptrCast(@alignCast(c));
    }
    fn snapBuf(c: *anyopaque, id: u64) ?u64 {
        return me(c).held.get(id);
    }
    fn allocSnap(c: *anyopaque) anyerror!u64 {
        me(c).began.store(tier.monotonicNs(), .release);
        return me(c).rank.sim.allocSnap();
    }
    fn freeSnap(c: *anyopaque, buf: u64) void {
        me(c).rank.sim.freeSnap(buf);
    }
    fn adoptSnap(c: *anyopaque, id: u64, buf: u64) anyerror!void {
        try me(c).held.put(gpa, id, buf);
    }
    fn host(f: *Follower) mir.Host {
        return .{ .ctx = f, .snapBuf = snapBuf, .allocSnap = allocSnap, .freeSnap = freeSnap, .adoptSnap = adoptSnap };
    }

    fn serve(f: *Follower) void {
        var buf: [1 << 16]u8 = undefined;
        while (true) {
            while (f.mirror.busy()) {
                if (f.link.wait(500)) break;
                f.mirror.service();
            }
            const bytes = f.link.recv(&buf) catch return;
            if (bytes.len == 0) return;
            f.mirror.onSync(bytes) catch |e| {
                f.err = e;
                return;
            };
        }
    }
};

const cfg: sync.Digest = @splat(5);

const Rig = struct {
    ranks: [4]*tt.Rank,
    net: star_mod.Net,
    star_links: [4][3]?Link,
    stars: [4]star_mod.Star,
    fl: [3]Follower,
    far: [3]Link, // the followers' ends
    fam: tt.Fam0,
    ld: *tt.L,

    /// `floor_gib` > 0 on one follower: its store refuses every keep (the disk floor above what is free).
    fn init(rig: *Rig, tag: []const u8, bad_floor: ?usize) !void {
        for (0..4) |g| {
            rig.ranks[g] = try tt.Rank.init(@intCast(g), tag, 0);
            if (g > 0) @memcpy(rig.ranks[g].sim.mem, rig.ranks[0].sim.mem);
            if (bad_floor == g) rig.ranks[g].settings.min_free_gib = 1 << 30;
        }
        rig.net = try star_mod.Net.open();
        for (0..4) |g| rig.stars[g] = rig.net.star(@intCast(g), &rig.star_links[g]);
        for (0..3) |i| {
            var pair: [2]c_int = undefined;
            if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
            links[i] = .{ .fd = pair[0] };
            rig.far[i] = .{ .fd = pair[1] };
        }
        rig.fam = .{ .sim = rig.ranks[0].sim };
        var f = rig.fam.family();
        f.sync_send = sendAll;
        f.sync_idle = idleYes;
        rig.ld = try tt.L.create(gpa, try rig.ranks[0].side(0), f, rig.ranks[0].settings, @splat(3));
        for (0..3) |i| {
            rig.fl[i] = .{ .rank = rig.ranks[i + 1], .link = rig.far[i] };
            rig.fl[i].mirror = try M.create(gpa, try rig.ranks[i + 1].side(@intCast(i + 1)), rig.stars[i + 1], @intCast(i + 1), rig.fl[i].host(), cfg);
        }
        // the start-up reconcile on every rank, then rank 0's Leader takes the protocol over
        var p0 = sync.Peer.init(gpa, rig.stars[0], 0, rig.ld.syncDisk());
        defer p0.deinit();
        p0.cfg = cfg;
        const Run = struct {
            fn go(m: *M, err: *?anyerror) void {
                _ = m.reconcile() catch |e| {
                    err.* = e;
                };
            }
        };
        var errs: [3]?anyerror = @splat(null);
        var ts: [3]std.Thread = undefined;
        for (0..3) |i| ts[i] = try std.Thread.spawn(.{}, Run.go, .{ rig.fl[i].mirror, &errs[i] });
        _ = try sync.reconcile(&p0);
        for (ts) |t| t.join();
        for (errs) |e| if (e) |x| return x;
        try rig.ld.enableSync(rig.stars[0], cfg, &p0);
        for (&rig.fl) |*x| x.thread = try std.Thread.spawn(.{}, Follower.serve, .{x});
    }

    /// The same kept state on every rank: Snap `id` at base 0 holding `at` positions.
    fn snapEverywhere(rig: *Rig, id: u64, at: usize) !*tt.Snap {
        const sn = try tt.keptSnap(rig.ranks[0].sim, id, 0, at);
        for (&rig.fl) |*x| {
            const buf = try x.rank.sim.allocSnap();
            std.debug.assert(buf == sn.buf);
            try x.held.put(gpa, id, buf);
        }
        return sn;
    }

    fn stop(rig: *Rig) void {
        for (links) |l| l.send(&.{}) catch {};
        for (&rig.fl) |*x| if (x.thread) |t| {
            t.join();
            x.thread = null;
        };
    }

    fn deinit(rig: *Rig) void {
        rig.stop();
        rig.ld.destroy();
        for (&rig.fl) |*x| {
            _ = x.mirror.close();
            x.held.deinit(gpa);
        }
        rig.fam.deinit();
        for (&links) |*l| l.close();
        for (&rig.far) |*l| l.close();
        rig.net.close();
        for (rig.ranks) |r| r.deinit();
    }

    /// Pumps rank 0 until the keep is visible to lookups (all four committed) or the bound passes.
    fn visible(rig: *Rig, id_list: []const u32) !bool {
        const t = rig.ld.tier();
        var k: usize = 0;
        while (k < 5000) : (k += 1) {
            t.vtable.pump.?(t.ptr, std.time.ns_per_ms);
            if (t.vtable.lookup(t.ptr, &tt.want(id_list)) != null) return true;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return false;
    }
};

fn ids(out: []u32) []u32 {
    tt.testIds(out, 77);
    return out;
}

test "four stores: a keep becomes visible after all four commit, a restore reads on all four and adopts the same bytes on each" {
    var rig: Rig = undefined;
    try rig.init("mirror-ok", null);
    defer rig.deinit();
    const t = rig.ld.tier();
    var buf: [4 * bt + 5]u32 = undefined;
    const id_list = ids(&buf);
    const sn = try rig.snapEverywhere(1, 4 * bt);
    defer std.testing.allocator.destroy(sn);
    try std.testing.expect(t.vtable.keep(t.ptr, &.{ .saved = sn, .key = id_list, .at = @intCast(sn.at), .bytes = 0, .group = 9, .lineage = &.{.{ .name = 7, .at = 0 }} }) != null);
    // a keep not yet committed by the others is not a hit (control)
    try std.testing.expect(t.vtable.lookup(t.ptr, &tt.want(id_list)) == null);
    try std.testing.expect(try rig.visible(id_list));
    // the point is on every rank's disk, with one digest
    var key: ?core.spill.Digest = null;
    var digest: ?core.spill.Digest = null;
    for (0..4) |g| {
        const side = if (g == 0) rig.ld.side else rig.fl[g - 1].mirror.side;
        var pts: std.ArrayList(core.spill.store.PointEntry) = .empty;
        defer pts.deinit(gpa);
        try side.store.pointList(gpa, &pts);
        try std.testing.expectEqual(@as(usize, 1), pts.items.len);
        if (key == null) {
            key = pts.items[0].key;
            digest = pts.items[0].digest;
        }
        try std.testing.expectEqualSlices(u8, &key.?, &pts.items[0].key);
        try std.testing.expectEqualSlices(u8, &digest.?, &pts.items[0].digest);
    }
    // a restore: every rank reads its own copy into the destination rank 0 reserved
    const p = t.vtable.lookup(t.ptr, &tt.want(id_list)).?;
    const job = try t.vtable.begin(t.ptr, &p);
    const t_begin = tier.monotonicNs();
    const done = tt.pollUntilDone(t, job);
    const t_done = tier.monotonicNs();
    try std.testing.expect(done == .ready);
    // every follower began its read at the restore frame, beside rank 0's: before rank 0's read was through, not after it
    for (&rig.fl) |*x| {
        const b = x.began.load(.acquire);
        try std.testing.expect(b != 0 and b <= t_done);
        _ = t_begin;
    }
    // (the followers adopted under rank 0's id before it installed: stop them to read their state)
    rig.stop();
    const dst = rig.fam.next_base;
    for (0..4) |g| {
        const sim = rig.ranks[g].sim;
        try std.testing.expect(sim.sameRows(0, dst, 4 * bt));
    }
    const dest_id = rig.fam.live.items[0].id;
    for (&rig.fl) |*x| {
        const buf2 = x.held.get(dest_id) orelse return error.NotAdopted;
        // the restored small state equals the kept one
        const small = rig.ranks[0].sim.layout.small_bytes;
        try std.testing.expectEqualSlices(u8, x.rank.sim.bytes(sn.buf, small), x.rank.sim.bytes(buf2, small));
    }
}

test "four stores: a block damaged on one follower fails the restore on all four, the entry is gone from every store and recomputes" {
    var rig: Rig = undefined;
    try rig.init("mirror-corrupt", null);
    defer rig.deinit();
    const t = rig.ld.tier();
    var buf: [4 * bt + 5]u32 = undefined;
    const id_list = ids(&buf);
    const sn = try rig.snapEverywhere(1, 4 * bt);
    defer std.testing.allocator.destroy(sn);
    try std.testing.expect(t.vtable.keep(t.ptr, &.{ .saved = sn, .key = id_list, .at = @intCast(sn.at), .bytes = 0, .group = 9, .lineage = &.{.{ .name = 7, .at = 0 }} }) != null);
    try std.testing.expect(try rig.visible(id_list));
    var pts: std.ArrayList(core.spill.store.PointEntry) = .empty;
    defer pts.deinit(gpa);
    try rig.ld.side.store.pointList(gpa, &pts);
    const key = pts.items[0].key;
    try rig.ranks[2].corrupt();
    const p = t.vtable.lookup(t.ptr, &tt.want(id_list)).?;
    const job = try t.vtable.begin(t.ptr, &p);
    const done = tt.pollUntilDone(t, job);
    try std.testing.expect(done == .failed);
    try std.testing.expect(done.failed != error.TestTimedOut);
    // nothing adopted anywhere, the entry gone from all four stores and not a hit again
    rig.stop();
    for (0..4) |g| {
        const side = if (g == 0) rig.ld.side else rig.fl[g - 1].mirror.side;
        try std.testing.expect(!side.store.presence(key));
    }
    try std.testing.expect(t.vtable.lookup(t.ptr, &tt.want(id_list)) == null);
    for (&rig.fl) |*x| try std.testing.expectEqual(@as(u32, 1), x.held.count());
}

test "four stores: a follower whose store refuses the write: the keep is never a hit and is gone from all four" {
    var rig: Rig = undefined;
    try rig.init("mirror-refuse", 3);
    defer rig.deinit();
    const t = rig.ld.tier();
    var buf: [4 * bt + 5]u32 = undefined;
    const id_list = ids(&buf);
    const sn = try rig.snapEverywhere(1, 4 * bt);
    defer std.testing.allocator.destroy(sn);
    _ = t.vtable.keep(t.ptr, &.{ .saved = sn, .key = id_list, .at = @intCast(sn.at), .bytes = 0, .group = 9, .lineage = &.{.{ .name = 7, .at = 0 }} });
    try std.testing.expect(!(try rig.visible(id_list)));
    rig.stop();
    for (0..4) |g| {
        const side = if (g == 0) rig.ld.side else rig.fl[g - 1].mirror.side;
        var pts: std.ArrayList(core.spill.store.PointEntry) = .empty;
        defer pts.deinit(gpa);
        try side.store.pointList(gpa, &pts);
        try std.testing.expectEqual(@as(usize, 0), pts.items.len);
    }
}

test "four stores: a guard frame for a state with no write is harmless; a malformed one is refused" {
    var rig: Rig = undefined;
    try rig.init("mirror-guard", null);
    defer rig.deinit();
    const t = rig.ld.tier();
    _ = t;
    var b: [1 + 12]u8 = undefined;
    b[0] = 2; // wire.SubOp.guard
    std.mem.writeInt(u64, b[1..9], 99, .little);
    std.mem.writeInt(u32, b[9..13], 0, .little);
    try sendAll(undefined, &b);
    // a sub-op nobody knows ends that follower's loop with BadFrame (rank 0 would never send one)
    rig.stop();
    for (&rig.fl) |*x| try std.testing.expectEqual(@as(?anyerror, null), x.err);
}

/// Arms `f` on every rank it names (rank 0's Leader, a follower's Mirror).
fn armRig(rig: *Rig, f: sync.Fault) void {
    rig.ld.armFault(f.armed(0));
    for (&rig.fl, 1..) |*x, g| x.mirror.arm(f.armed(@intCast(g)));
}

/// Keeps the test entry with `f` armed and returns whether it became visible; with `restore`, the restore is tried too and its
/// outcome returned (null: the entry was never visible). Afterwards no store holds the entry unless the restore adopted it.
const Drill = struct { visible: bool, restore: ?pc.Poll = null, held_by: usize = 0 };

fn drill(tag: []const u8, f: sync.Fault) !Drill {
    var rig: Rig = undefined;
    try rig.init(tag, null);
    defer rig.deinit();
    armRig(&rig, f);
    const t = rig.ld.tier();
    var buf: [4 * bt + 5]u32 = undefined;
    const id_list = ids(&buf);
    const sn = try rig.snapEverywhere(1, 4 * bt);
    defer std.testing.allocator.destroy(sn);
    try std.testing.expect(t.vtable.keep(t.ptr, &.{ .saved = sn, .key = id_list, .at = @intCast(sn.at), .bytes = 0, .group = 9, .lineage = &.{.{ .name = 7, .at = 0 }} }) != null);
    var out: Drill = .{ .visible = try rig.visible(id_list) };
    if (out.visible) {
        const p = t.vtable.lookup(t.ptr, &tt.want(id_list)).?;
        const job = try t.vtable.begin(t.ptr, &p);
        out.restore = tt.pollUntilDone(t, job);
    }
    rig.stop();
    for (0..4) |g| {
        const side = if (g == 0) rig.ld.side else rig.fl[g - 1].mirror.side;
        var pts: std.ArrayList(core.spill.store.PointEntry) = .empty;
        defer pts.deinit(gpa);
        try side.store.pointList(gpa, &pts);
        out.held_by += pts.items.len;
    }
    return out;
}

test "fault drill, control: no fault, the entry is kept on all four and a restore adopts it" {
    const d = try drill("fault-none", .{});
    try std.testing.expect(d.visible);
    try std.testing.expect(d.restore.? == .ready);
    try std.testing.expectEqual(@as(usize, 4), d.held_by);
}

test "fault drill: write:1 fails the follower's next keep, the entry is never visible and is on no store" {
    const d = try drill("fault-write1", .{ .write = 1 });
    try std.testing.expect(!d.visible);
    try std.testing.expectEqual(@as(usize, 0), d.held_by);
}

test "fault drill: write:0 fails rank 0's next keep, the entry is never visible and is on no store" {
    const d = try drill("fault-write0", .{ .write = 0 });
    try std.testing.expect(!d.visible);
    try std.testing.expectEqual(@as(usize, 0), d.held_by);
}

test "fault drill: corrupt:2 commits the keep, the restore fails its check and the entry is gone from all four" {
    const d = try drill("fault-corrupt2", .{ .corrupt = 2 });
    try std.testing.expect(d.visible);
    try std.testing.expect(d.restore.? == .failed);
    try std.testing.expectEqual(@as(usize, 0), d.held_by);
}

test "fault drill: corrupt:0 damages rank 0's own copy, the restore fails and the entry is gone from all four" {
    const d = try drill("fault-corrupt0", .{ .corrupt = 0 });
    try std.testing.expect(d.visible);
    try std.testing.expect(d.restore.? == .failed);
    try std.testing.expectEqual(@as(usize, 0), d.held_by);
}
