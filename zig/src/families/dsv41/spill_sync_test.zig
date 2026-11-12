//! spill_sync.zig on four ranks: four threads, each with its own directory of plain files for a store, rank 0's frames
//! over socket pairs and the ranks' collectives over a star on socket pairs. Every test ends by comparing the four
//! ranks' disks and visible sets; the decisions each follower reached must equal rank 0's.
const std = @import("std");
const c = std.c;
const sync = @import("spill_sync.zig");
const star_mod = @import("star.zig");
const Link = @import("link.zig").Link;

const gpa = std.testing.allocator;
const Key = sync.Key;
const Digest = sync.Digest;

fn keyOf(n: u8) Key {
    var k: Key = @splat(n);
    k[0] = 0xA0 +% n;
    return k;
}

/// The bytes entry `n` holds on a healthy rank.
fn bodyOf(n: u8, out: []u8) []u8 {
    for (out, 0..) |*b, i| b.* = @truncate(i *% 31 +% n *% 7 +% 1);
    return out;
}

fn hex(key: Key) [64]u8 {
    return std.fmt.bytesToHex(key, .lower);
}

// -- a store of plain files ---------------------------------------------------------------------------------------

/// One entry is one file: the body, then its SHA-256. Written under a temporary name, flushed, renamed.
const FileDisk = struct {
    dir: [:0]u8,
    // faults
    refuse_keep: bool = false, // keep() says no
    fail_commit: bool = false, // the write reports fail
    pending_reports: u32 = 0, // reports pending this many polls first
    forever_pending: bool = false,
    skew: bool = false, // writes other bytes than rank 0's (the premise broken)
    unreadable: bool = false, // list() fails
    // what happened
    reads: std.AutoHashMapUnmanaged(u64, ?bool) = .empty, // restore id -> the adopt decision (null: undecided)
    removes: u32 = 0,

    fn init(tag: []const u8, rank: usize) !*FileDisk {
        const d = try gpa.create(FileDisk);
        d.* = .{ .dir = try std.fmt.allocPrintSentinel(gpa, "/tmp/tf-sync-{s}-{d}-r{d}", .{ tag, c.getpid(), rank }, 0) };
        rmTree(d.dir);
        if (c.mkdir(d.dir, 0o700) != 0) return error.Mkdir;
        return d;
    }

    fn deinit(d: *FileDisk) void {
        rmTree(d.dir);
        gpa.free(d.dir);
        d.reads.deinit(gpa);
        gpa.destroy(d);
    }

    fn rmTree(root: [:0]const u8) void {
        const dir = c.opendir(root) orelse return;
        var names: std.ArrayList([:0]u8) = .empty;
        while (c.readdir(dir)) |ent| {
            const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
            names.append(gpa, gpa.dupeSentinel(u8, name, 0) catch continue) catch {};
        }
        _ = c.closedir(dir);
        for (names.items) |n| {
            var b: [512]u8 = undefined;
            const p = std.fmt.bufPrintSentinel(&b, "{s}/{s}", .{ root, n }, 0) catch continue;
            _ = c.unlink(p);
            gpa.free(n);
        }
        names.deinit(gpa);
        _ = c.rmdir(root);
    }

    fn path(d: *const FileDisk, buf: []u8, key: Key, ext: []const u8) [:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{s}.{s}", .{ d.dir, hex(key), ext }, 0) catch unreachable;
    }

    /// The entry's file written whole: temp name, fsync, rename.
    fn put(d: *FileDisk, key: Key, body: []const u8) bool {
        var b1: [512]u8 = undefined;
        var b2: [512]u8 = undefined;
        const tmp = d.path(&b1, key, "tmp");
        const fin = d.path(&b2, key, "ent");
        const fd = c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o600));
        if (fd < 0) return false;
        defer _ = c.close(fd);
        var dig: Digest = undefined;
        std.crypto.hash.sha2.Sha256.hash(body, &dig, .{});
        if (c.write(fd, body.ptr, body.len) != body.len) return false;
        if (c.write(fd, &dig, 32) != 32) return false;
        if (c.fsync(fd) != 0) return false;
        return c.rename(tmp, fin) == 0;
    }

    /// The entry's digest if its file is whole and its trailer matches its body; null otherwise.
    fn verified(d: *const FileDisk, key: Key) ?Digest {
        var b: [512]u8 = undefined;
        const fd = c.open(d.path(&b, key, "ent"), .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
        if (fd < 0) return null;
        defer _ = c.close(fd);
        var buf: [8192]u8 = undefined;
        const n = c.read(fd, &buf, buf.len);
        if (n < 32) return null;
        const len: usize = @intCast(n);
        var dig: Digest = undefined;
        std.crypto.hash.sha2.Sha256.hash(buf[0 .. len - 32], &dig, .{});
        if (!std.mem.eql(u8, &dig, buf[len - 32 .. len])) return null;
        return dig;
    }

    fn corruptEntry(d: *FileDisk, key: Key) void {
        var b: [512]u8 = undefined;
        const fd = c.open(d.path(&b, key, "ent"), .{ .ACCMODE = .RDWR }, @as(c.mode_t, 0));
        if (fd < 0) return;
        defer _ = c.close(fd);
        var one: [1]u8 = undefined;
        _ = c.pread(fd, &one, 1, 5);
        one[0] ^= 0xff;
        _ = c.pwrite(fd, &one, 1, 5);
    }

    fn has(d: *const FileDisk, key: Key) bool {
        var b: [512]u8 = undefined;
        return c.access(d.path(&b, key, "ent"), 0) == 0;
    }

    fn count(d: *const FileDisk) usize {
        const dir = c.opendir(d.dir) orelse return 0;
        defer _ = c.closedir(dir);
        var n: usize = 0;
        while (c.readdir(dir)) |ent| {
            const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
            if (std.mem.endsWith(u8, name, ".ent")) n += 1;
        }
        return n;
    }

    fn me(p: *anyopaque) *FileDisk {
        return @ptrCast(@alignCast(p));
    }

    fn keepFn(p: *anyopaque, key: Key, body: []const u8) bool {
        const d = me(p);
        if (d.refuse_keep) return false;
        if (d.skew) {
            var other: [256]u8 = undefined;
            @memcpy(other[0..body.len], body);
            other[0] ^= 0x55;
            return d.put(key, other[0..body.len]);
        }
        return d.put(key, body);
    }

    fn polled(d: *FileDisk) ?sync.Report {
        if (d.forever_pending) return .{ .status = .pending };
        if (d.pending_reports > 0) {
            d.pending_reports -= 1;
            return .{ .status = .pending };
        }
        return null;
    }

    fn keepReportFn(p: *anyopaque, key: Key) sync.Report {
        const d = me(p);
        if (d.polled()) |r| return r;
        if (d.fail_commit) return .{ .status = .fail };
        const dig = d.verified(key) orelse return .{ .status = .fail };
        return .{ .status = .ok, .digest = dig };
    }

    var read_keys: [64]Key = undefined;

    fn readFn(p: *anyopaque, key: Key, id: u64, _: []const u8) bool {
        const d = me(p);
        if (!d.has(key)) return false;
        d.reads.put(gpa, id, null) catch return false;
        read_keys[id % 64] = key;
        return true;
    }

    fn readReportFn(p: *anyopaque, id: u64) sync.Report {
        const d = me(p);
        if (d.polled()) |r| return r;
        const dig = d.verified(read_keys[id % 64]) orelse return .{ .status = .fail };
        return .{ .status = .ok, .digest = dig };
    }

    fn finishFn(p: *anyopaque, id: u64, adopt: bool) void {
        me(p).reads.put(gpa, id, adopt) catch {};
    }

    fn removeFn(p: *anyopaque, key: Key) void {
        const d = me(p);
        var b: [512]u8 = undefined;
        _ = c.unlink(d.path(&b, key, "ent"));
        _ = c.unlink(d.path(&b, key, "tmp"));
        d.removes += 1;
    }

    fn listFn(p: *anyopaque, g: std.mem.Allocator, out: *std.ArrayList(sync.Listed)) anyerror!void {
        const d = me(p);
        if (d.unreadable) return error.Io;
        const dir = c.opendir(d.dir) orelse return error.Io;
        defer _ = c.closedir(dir);
        while (c.readdir(dir)) |ent| {
            const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
            if (!std.mem.endsWith(u8, name, ".ent") or name.len != 68) continue;
            var key: Key = undefined;
            _ = std.fmt.hexToBytes(&key, name[0..64]) catch continue;
            try out.append(g, .{ .key = key, .digest = d.verified(key) orelse @splat(0) });
        }
    }

    fn disk(d: *FileDisk) sync.Disk {
        return .{ .ptr = d, .keep = keepFn, .keepReport = keepReportFn, .read = readFn, .readReport = readReportFn, .finishRead = finishFn, .remove = removeFn, .list = listFn };
    }
};

// -- four ranks --------------------------------------------------------------------------------------------------

const Rig = struct {
    net: star_mod.Net,
    star_links: [4][3]?Link,
    stars: [4]star_mod.Star,
    tx: [3]Link, // rank 0's end of each follower's frame stream
    rx: [3]Link,
    disks: [4]*FileDisk,
    peers: [4]sync.Peer,
    leader: sync.Leader,
    out: [4]std.ArrayList(sync.Decided) = @splat(.empty),
    errs: [4]?anyerror = @splat(null),
    threads: [3]?std.Thread = @splat(null),
    send_fails: ?usize = null, // frames to send before the next send fails (a link that went)

    fn init(tag: []const u8) !*Rig {
        const r = try gpa.create(Rig);
        r.* = undefined;
        r.out = @splat(.empty);
        r.errs = @splat(null);
        r.threads = @splat(null);
        r.send_fails = null;
        r.net = try star_mod.Net.open();
        for (0..4) |g| r.stars[g] = r.net.star(@intCast(g), &r.star_links[g]);
        for (0..3) |i| {
            var pair: [2]c_int = undefined;
            if (c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
            r.tx[i] = .{ .fd = pair[0] };
            r.rx[i] = .{ .fd = pair[1] };
        }
        for (0..4) |g| {
            r.disks[g] = try FileDisk.init(tag, g);
            r.peers[g] = sync.Peer.init(gpa, r.stars[g], @intCast(g), r.disks[g].disk());
        }
        r.leader = .{ .peer = &r.peers[0], .tx = .{ .ptr = r, .send = sendAll } };
        return r;
    }

    fn sendAll(p: *anyopaque, frame: []const u8) anyerror!void {
        const r: *Rig = @ptrCast(@alignCast(p));
        if (r.send_fails) |*n| {
            if (n.* == 0) return error.PeerDown;
            n.* -= 1;
        }
        for (r.tx) |l| try l.send(frame);
    }

    fn start(r: *Rig) !void {
        for (1..4) |g| r.threads[g - 1] = try std.Thread.spawn(.{}, follow, .{ r, g });
    }

    fn follow(r: *Rig, g: usize) void {
        var buf: [1 << 16]u8 = undefined;
        while (true) {
            const bytes = r.rx[g - 1].recv(&buf) catch |e| {
                r.errs[g] = e;
                return;
            };
            const f = sync.Frame.decode(bytes) catch |e| {
                r.errs[g] = e;
                return;
            };
            if (f.op == .end) return;
            r.peers[g].onFrame(f, &r.out[g]) catch |e| {
                r.errs[g] = e;
                return;
            };
        }
    }

    /// The followers stopped and joined; their state may be read.
    fn finish(r: *Rig) void {
        r.leader.end();
        for (&r.threads) |*t| if (t.*) |th| {
            th.join();
            t.* = null;
        };
    }

    fn deinit(r: *Rig) void {
        r.finish();
        for (0..4) |g| {
            r.peers[g].deinit();
            r.out[g].deinit(gpa);
            r.disks[g].deinit();
        }
        r.leader.deinit();
        for (&r.tx) |*l| l.close();
        for (&r.rx) |*l| l.close();
        r.net.close();
        gpa.destroy(r);
    }

    fn keep(r: *Rig, n: u8) sync.Leader.Keep {
        var b: [200]u8 = undefined;
        return r.leader.keep(keyOf(n), bodyOf(n, &b), &r.out[0]);
    }

    /// Settles until nothing is in flight (or `max` rounds).
    fn settleAll(r: *Rig, max: usize) !void {
        var i: usize = 0;
        while (r.peers[0].n > 0 and i < max) : (i += 1) try r.leader.settle(&r.out[0]);
    }

    fn kinds(r: *Rig, g: usize, kind: sync.Decided.Kind) usize {
        var n: usize = 0;
        for (r.out[g].items) |d| if (d.kind == kind) {
            n += 1;
        };
        return n;
    }

    /// After finish(): every follower reached rank 0's decisions, in order.
    fn expectSameDecisions(r: *Rig) !void {
        for (1..4) |g| {
            try std.testing.expectEqual(r.out[0].items.len, r.out[g].items.len);
            for (r.out[0].items, r.out[g].items) |a, b| {
                try std.testing.expectEqual(a.kind, b.kind);
                try std.testing.expectEqualSlices(u8, &a.key, &b.key);
                try std.testing.expectEqual(a.id, b.id);
            }
            try std.testing.expectEqual(@as(?anyerror, null), r.errs[g]);
        }
    }

    /// After finish(): the entry is on every disk, or on none.
    fn expectEverywhere(r: *Rig, n: u8, there: bool) !void {
        for (0..4) |g| try std.testing.expectEqual(there, r.disks[g].has(keyOf(n)));
        for (0..4) |g| try std.testing.expectEqual(there, r.peers[g].visible.contains(keyOf(n)));
    }
};

// -- the protocol -------------------------------------------------------------------------------------------------

test "a correct run: keeps become visible on all four, a restore adopts on all four, an evict removes it everywhere" {
    const r = try Rig.init("ok");
    defer r.deinit();
    try r.start();
    for ([_]u8{ 1, 2, 3 }) |n| try std.testing.expectEqual(sync.Leader.Keep.queued, r.keep(n));
    // not visible before the commit acks
    try std.testing.expect(!r.peers[0].isVisible(keyOf(1)));
    try std.testing.expect(!r.leader.restore(keyOf(1), 1, "", &r.out[0]));
    try r.settleAll(8);
    for ([_]u8{ 1, 2, 3 }) |n| try std.testing.expect(r.peers[0].isVisible(keyOf(n)));
    try std.testing.expectEqual(sync.Leader.Keep.already, r.keep(2));
    // a restore
    try std.testing.expect(r.leader.restore(keyOf(2), 2, "", &r.out[0]));
    try r.settleAll(8);
    // an evict
    r.leader.evict(keyOf(3), &r.out[0]);
    try std.testing.expect(!r.leader.restore(keyOf(3), 3, "", &r.out[0]));
    r.finish();
    try std.testing.expectEqual(@as(usize, 3), r.kinds(0, .kept));
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .adopt));
    try r.expectSameDecisions();
    try r.expectEverywhere(1, true);
    try r.expectEverywhere(2, true);
    try r.expectEverywhere(3, false);
    for (0..4) |g| {
        try std.testing.expectEqual(@as(?bool, true), r.disks[g].reads.get(2).?);
        try std.testing.expectEqual(@as(usize, 2), r.disks[g].count());
    }
    // the four disks hold the same bytes under the same names
    for ([_]u8{ 1, 2 }) |n| for (1..4) |g| {
        try std.testing.expectEqualSlices(u8, &r.disks[0].verified(keyOf(n)).?, &r.disks[g].verified(keyOf(n)).?);
    };
}

test "a commit that fails on one rank: the entry is visible nowhere and gone from all four disks; the others commit" {
    const r = try Rig.init("failcommit");
    defer r.deinit();
    try r.start();
    r.disks[2].fail_commit = true;
    _ = r.keep(1);
    try r.settleAll(8);
    r.disks[2].fail_commit = false;
    _ = r.keep(2);
    try r.settleAll(8);
    try std.testing.expect(!r.peers[0].isVisible(keyOf(1)));
    try std.testing.expect(!r.leader.restore(keyOf(1), 1, "", &r.out[0]));
    r.finish();
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .keep_dropped));
    try r.expectSameDecisions();
    try r.expectEverywhere(1, false);
    try r.expectEverywhere(2, true);
}

test "a write refused at once on rank 0's own disk, or on a follower's, drops the entry on all four" {
    for ([_]usize{ 0, 3 }) |bad| {
        const r = try Rig.init("refuse");
        defer r.deinit();
        try r.start();
        r.disks[bad].refuse_keep = true;
        _ = r.keep(1);
        try r.settleAll(8);
        r.finish();
        try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .keep_dropped));
        try r.expectSameDecisions();
        try r.expectEverywhere(1, false);
    }
}

test "a block corrupted on one rank: the restore is dropped on all four, the request recomputes, the other entries stay" {
    const r = try Rig.init("corrupt");
    defer r.deinit();
    try r.start();
    _ = r.keep(1);
    _ = r.keep(2);
    try r.settleAll(8);
    r.finish(); // (rank 1's file is damaged while the ranks are idle; the stream resumes below)
    // a fresh stream over the same disks
    r.disks[1].corruptEntry(keyOf(1));
    for (&r.threads, 1..) |*t, g| t.* = try std.Thread.spawn(.{}, Rig.follow, .{ r, g });
    try std.testing.expect(r.leader.restore(keyOf(1), 1, "", &r.out[0]));
    try r.settleAll(8);
    // the entry is gone: a second restore recomputes at once
    try std.testing.expect(!r.leader.restore(keyOf(1), 2, "", &r.out[0]));
    // the intact entry still restores
    try std.testing.expect(r.leader.restore(keyOf(2), 3, "", &r.out[0]));
    try r.settleAll(8);
    r.finish();
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .restore_dropped));
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .adopt));
    try r.expectSameDecisions();
    try r.expectEverywhere(1, false);
    try r.expectEverywhere(2, true);
    for (0..4) |g| try std.testing.expectEqual(@as(?bool, false), r.disks[g].reads.get(1).?);
    for (0..4) |g| try std.testing.expectEqual(@as(?bool, true), r.disks[g].reads.get(3).?);
}

test "a restore whose four reads are clean and equal but are not the bytes the keep committed is dropped on all four" {
    const r = try Rig.init("moved");
    defer r.deinit();
    try r.start();
    _ = r.keep(1);
    try r.settleAll(8);
    r.finish();
    // all four disks later hold the same other bytes under the name (each file whole, its trailer right)
    for (0..4) |g| _ = r.disks[g].put(keyOf(1), &.{ 9, 9, 9, 9 });
    for (&r.threads, 1..) |*t, g| t.* = try std.Thread.spawn(.{}, Rig.follow, .{ r, g });
    try std.testing.expect(r.leader.restore(keyOf(1), 1, "", &r.out[0]));
    try r.settleAll(8);
    r.finish();
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .restore_dropped));
    try std.testing.expectEqual(@as(usize, 0), r.kinds(0, .adopt));
    try r.expectSameDecisions();
    try r.expectEverywhere(1, false);
}

test "a restore on a rank that lost the file: that rank cannot read it, and the entry is dropped on all four" {
    const r = try Rig.init("lost");
    defer r.deinit();
    try r.start();
    _ = r.keep(1);
    try r.settleAll(8);
    r.finish();
    FileDisk.removeFn(r.disks[2], keyOf(1));
    for (&r.threads, 1..) |*t, g| t.* = try std.Thread.spawn(.{}, Rig.follow, .{ r, g });
    try std.testing.expect(r.leader.restore(keyOf(1), 1, "", &r.out[0]));
    try r.settleAll(8);
    r.finish();
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .restore_dropped));
    try r.expectSameDecisions();
    try r.expectEverywhere(1, false);
}

test "control: a rank whose bytes differ while it reports ok (the premise broken) is caught by the digest and dropped everywhere" {
    const r = try Rig.init("skew");
    defer r.deinit();
    try r.start();
    r.disks[3].skew = true;
    _ = r.keep(1);
    try r.settleAll(8);
    r.finish();
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .keep_dropped));
    try r.expectSameDecisions();
    try r.expectEverywhere(1, false);
}

test "a slow disk: the entry stays invisible while any rank is pending and becomes visible once all four ack" {
    const r = try Rig.init("slow");
    defer r.deinit();
    try r.start();
    r.disks[2].pending_reports = 3;
    _ = r.keep(1);
    try r.leader.settle(&r.out[0]);
    try std.testing.expect(!r.peers[0].isVisible(keyOf(1)));
    try std.testing.expectEqual(@as(usize, 1), r.peers[0].pending());
    try r.settleAll(8);
    try std.testing.expect(r.peers[0].isVisible(keyOf(1)));
    r.finish();
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .kept));
    try r.expectSameDecisions();
    try r.expectEverywhere(1, true);
}

test "a keep evicted while it settles never becomes visible" {
    const r = try Rig.init("evictpending");
    defer r.deinit();
    try r.start();
    r.disks[1].pending_reports = 2;
    _ = r.keep(1);
    try r.leader.settle(&r.out[0]);
    r.leader.evict(keyOf(1), &r.out[0]);
    try r.settleAll(8);
    r.finish();
    try std.testing.expectEqual(@as(usize, 0), r.kinds(0, .kept));
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .keep_dropped));
    try r.expectSameDecisions();
    try r.expectEverywhere(1, false);
}

test "a write that never settles is counted failed on every rank after the bound, and dropped everywhere" {
    const r = try Rig.init("never");
    defer r.deinit();
    try r.start();
    r.disks[1].forever_pending = true;
    _ = r.keep(1);
    try r.settleAll(sync.max_pending_settles + 8);
    r.finish();
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .keep_dropped));
    try r.expectSameDecisions();
    try r.expectEverywhere(1, false);
}

test "a link that goes: the peer is down, the restore in flight is dropped, nothing is visible and the engine goes on" {
    const r = try Rig.init("linkdown");
    defer r.deinit();
    try r.start();
    _ = r.keep(1);
    try r.settleAll(8);
    try std.testing.expect(r.leader.restore(keyOf(1), 1, "", &r.out[0]));
    r.send_fails = 0; // the settle frame cannot go
    try std.testing.expectError(error.Down, r.leader.settle(&r.out[0]));
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .restore_dropped));
    try std.testing.expect(!r.peers[0].isVisible(keyOf(1)));
    try std.testing.expectEqual(sync.Leader.Keep.refused, r.keep(2));
    try std.testing.expect(!r.leader.restore(keyOf(1), 2, "", &r.out[0]));
    r.send_fails = null;
    r.finish();
}

test "a follower whose collective ends puts rank 0 down at its next settle, not a hang" {
    const r = try Rig.init("peerdown");
    defer r.deinit();
    try r.start();
    _ = r.keep(1);
    try r.settleAll(8);
    try std.testing.expect(r.leader.restore(keyOf(1), 1, "", &r.out[0]));
    // rank 2's star link to rank 0 closes: rank 0's gather fails
    _ = c.shutdown(r.net.follower[1].?.fd, 2);
    try std.testing.expectError(error.Down, r.leader.settle(&r.out[0]));
    try std.testing.expectEqual(@as(usize, 1), r.kinds(0, .restore_dropped));
    try std.testing.expect(r.peers[0].down);
    // (the star's own rule: rank 0 closing its links is what ends the followers' reads)
    for (r.net.rank0) |l| if (l) |x| {
        _ = c.shutdown(x.fd, 2);
    };
    r.finish();
}

// -- start-up -----------------------------------------------------------------------------------------------------

fn seed(r: *Rig, g: usize, n: u8) void {
    var b: [200]u8 = undefined;
    _ = r.disks[g].put(keyOf(n), bodyOf(n, &b));
}

fn reconcileAll(r: *Rig, res: *[4]?sync.Reconciled) !void {
    const Run = struct {
        fn go(p: *sync.Peer, out: *?sync.Reconciled) void {
            out.* = sync.reconcile(p) catch null;
        }
    };
    var ts: [3]std.Thread = undefined;
    for (1..4) |g| ts[g - 1] = try std.Thread.spawn(.{}, Run.go, .{ &r.peers[g], &res[g] });
    Run.go(&r.peers[0], &res[0]);
    for (ts) |t| t.join();
}

test "start-up: entries equal on all four stay and are visible; one held by three, one that differs, are deleted everywhere" {
    const r = try Rig.init("reconcile");
    defer r.deinit();
    for (0..4) |g| seed(r, g, 1);
    for (0..3) |g| seed(r, g, 2); // rank 3 lacks it
    for (0..4) |g| seed(r, g, 3);
    r.disks[1].corruptEntry(keyOf(3)); // rank 1's copy is damaged: its digest differs
    var res: [4]?sync.Reconciled = @splat(null);
    try reconcileAll(r, &res);
    for (res) |x| try std.testing.expect(x != null);
    try r.expectEverywhere(1, true);
    try r.expectEverywhere(2, false);
    try r.expectEverywhere(3, false);
    try std.testing.expectEqual(@as(u32, 1), res[0].?.kept);
    try std.testing.expectEqual(@as(u32, 2), res[0].?.deleted);
    try std.testing.expectEqual(@as(u32, 1), res[3].?.kept);
    try std.testing.expectEqual(@as(u32, 1), res[3].?.deleted);
}

test "start-up control: four equal stores lose nothing" {
    const r = try Rig.init("reconcileok");
    defer r.deinit();
    for (0..4) |g| for ([_]u8{ 1, 2, 3 }) |n| seed(r, g, n);
    var res: [4]?sync.Reconciled = @splat(null);
    try reconcileAll(r, &res);
    for (res) |x| try std.testing.expectEqual(@as(u32, 3), x.?.kept);
    for ([_]u8{ 1, 2, 3 }) |n| try r.expectEverywhere(n, true);
    for (0..4) |g| try std.testing.expectEqual(@as(u32, 0), r.disks[g].removes);
}

test "start-up: a rank with an empty store (a lost disk) makes every entry go from the other three" {
    const r = try Rig.init("reconcileempty");
    defer r.deinit();
    for (0..3) |g| for ([_]u8{ 1, 2 }) |n| seed(r, g, n);
    var res: [4]?sync.Reconciled = @splat(null);
    try reconcileAll(r, &res);
    for ([_]u8{ 1, 2 }) |n| try r.expectEverywhere(n, false);
    try std.testing.expectEqual(@as(u32, 2), res[0].?.deleted);
    try std.testing.expectEqual(@as(u32, 0), res[3].?.deleted);
}

test "start-up: a rank whose store cannot be read makes every entry go everywhere" {
    const r = try Rig.init("reconcileunread");
    defer r.deinit();
    for (0..4) |g| for ([_]u8{ 1, 2 }) |n| seed(r, g, n);
    r.disks[2].unreadable = true;
    var res: [4]?sync.Reconciled = @splat(null);
    try reconcileAll(r, &res);
    for ([_]u8{ 1, 2 }) |n| {
        for ([_]usize{ 0, 1, 3 }) |g| try std.testing.expect(!r.disks[g].has(keyOf(n)));
        for (0..4) |g| try std.testing.expect(!r.peers[g].visible.contains(keyOf(n)));
    }
    try std.testing.expect(r.disks[2].has(keyOf(1))); // (it could not be read, so nothing could be deleted there; it is not visible)
}

test "start-up: more entries than a page are compared page by page" {
    const r = try Rig.init("reconcilepages");
    defer r.deinit();
    const n_entries = sync.page_entries * 2 + 5;
    var i: usize = 0;
    while (i < n_entries) : (i += 1) {
        var k: Key = @splat(0);
        std.mem.writeInt(u32, k[0..4], @intCast(i), .big);
        for (0..4) |g| {
            if (g == 3 and i % 50 == 7) continue; // rank 3 lacks every 50th
            _ = r.disks[g].put(k, &.{ 1, 2, 3, @truncate(i) });
        }
    }
    var res: [4]?sync.Reconciled = @splat(null);
    try reconcileAll(r, &res);
    const missing = (n_entries + 42) / 50;
    try std.testing.expectEqual(@as(u32, @intCast(n_entries - missing)), res[0].?.kept);
    try std.testing.expectEqual(@as(u32, @intCast(missing)), res[0].?.deleted);
    try std.testing.expectEqual(@as(usize, n_entries - missing), r.disks[1].count());
}

// -- frames -------------------------------------------------------------------------------------------------------

test "a frame reads back as written; a short, long or unnamed one is refused" {
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    const f: sync.Frame = .{ .op = .keep, .key = keyOf(9), .id = 0x0102030405, .body = "hello" };
    try f.encode(gpa, &list);
    const g = try sync.Frame.decode(list.items);
    try std.testing.expectEqual(f.op, g.op);
    try std.testing.expectEqualSlices(u8, &f.key, &g.key);
    try std.testing.expectEqual(f.id, g.id);
    try std.testing.expectEqualStrings("hello", g.body);
    try std.testing.expectError(error.BadFrame, sync.Frame.decode(list.items[0 .. list.items.len - 1]));
    try std.testing.expectError(error.BadFrame, sync.Frame.decode(list.items[0..10]));
    list.items[0] = 99;
    try std.testing.expectError(error.BadFrame, sync.Frame.decode(list.items));
    list.items[0] = 1;
    try list.append(gpa, 0);
    try std.testing.expectError(error.BadFrame, sync.Frame.decode(list.items));
}

test "start-up: a rank that abstains (its store did not open) ends the reconcile on every rank and deletes nothing" {
    const r = try Rig.init("abstain");
    defer r.deinit();
    for (0..4) |g| for ([_]u8{ 1, 2 }) |n| seed(r, g, n);
    const Run = struct {
        fn go(p: *sync.Peer, out: *?anyerror) void {
            const res = if (p.rank == 2) sync.abstain(p) else sync.reconcile(p);
            out.* = if (res) |_| null else |e| e;
        }
    };
    var errs: [4]?anyerror = @splat(null);
    var ts: [3]std.Thread = undefined;
    for (1..4) |g| ts[g - 1] = try std.Thread.spawn(.{}, Run.go, .{ &r.peers[g], &errs[g] });
    Run.go(&r.peers[0], &errs[0]);
    for (ts) |t| t.join();
    for (errs) |e| try std.testing.expectEqual(@as(?anyerror, error.Abstained), e);
    for ([_]u8{ 1, 2 }) |n| for (0..4) |g| try std.testing.expect(r.disks[g].has(keyOf(n)));
}

test "start-up: ranks whose settings differ are refused on every rank and delete nothing" {
    const r = try Rig.init("cfg");
    defer r.deinit();
    for (0..4) |g| seed(r, g, 1);
    r.peers[3].cfg = @splat(7);
    const Run = struct {
        fn go(p: *sync.Peer, out: *?anyerror) void {
            out.* = if (sync.reconcile(p)) |_| null else |e| e;
        }
    };
    var errs: [4]?anyerror = @splat(null);
    var ts: [3]std.Thread = undefined;
    for (1..4) |g| ts[g - 1] = try std.Thread.spawn(.{}, Run.go, .{ &r.peers[g], &errs[g] });
    Run.go(&r.peers[0], &errs[0]);
    for (ts) |t| t.join();
    for (errs) |e| try std.testing.expectEqual(@as(?anyerror, error.ConfigDiffers), e);
    for (0..4) |g| try std.testing.expect(r.disks[g].has(keyOf(1)));
}

test "TF_DS_SSD_FAULT: write:R, corrupt:S and both parse; anything else, or a rank past the world, is refused" {
    const F = sync.Fault;
    try std.testing.expectEqual(F{}, try F.parse(null, 4));
    try std.testing.expectEqual(F{}, try F.parse("", 4));
    try std.testing.expectEqual(F{ .write = 1 }, try F.parse("write:1", 4));
    try std.testing.expectEqual(F{ .corrupt = 2 }, try F.parse("corrupt:2", 4));
    try std.testing.expectEqual(F{ .write = 1, .corrupt = 2 }, try F.parse("write:1,corrupt:2", 4));
    try std.testing.expectEqual(F{ .write = 1, .corrupt = 2 }, try F.parse("corrupt:2,write:1", 4));
    for ([_][]const u8{ "bogus", "write", "write:", "write:x", "write:9", "write:4", "corrupt:-1", "write:1,write:2", " write:1", "write:1,", "write:1;corrupt:2", "Write:1", "evict:1" }) |bad|
        try std.testing.expectError(error.BadFault, F.parse(bad, 4));
    try std.testing.expectError(error.BadFault, F.parse("write:2", 2));
}

test "TF_DS_SSD_FAULT: each rank carries out only the faults that name it, and the start line's text round-trips" {
    const F = sync.Fault;
    const f = try F.parse("write:1,corrupt:2", 4);
    try std.testing.expectEqual(sync.Armed{}, f.armed(0));
    try std.testing.expectEqual(sync.Armed{ .write = true }, f.armed(1));
    try std.testing.expectEqual(sync.Armed{ .corrupt = true }, f.armed(2));
    try std.testing.expectEqual(sync.Armed{}, f.armed(3));
    var buf: [48]u8 = undefined;
    try std.testing.expectEqualStrings("write:1,corrupt:2", f.text(&buf));
    try std.testing.expectEqualStrings("corrupt:0", (F{ .corrupt = 0 }).text(&buf));
    try std.testing.expectEqualStrings("off", (F{}).text(&buf));
    try std.testing.expectEqual(f, try F.parse(f.text(&buf), 4));
}
