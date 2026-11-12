//! Collectives of the ranks' host threads over the engine's own TCP links: a star from rank 0, which reads every
//! follower's bytes in rank order and sends the result back to each. They carry what the ranks must agree on before or
//! beside the device work (a switch, a queue-pair table, a vote, a digest): small host bytes, no device memory, no
//! communicator. Every rank calls a collective in the same order; a rank that cannot go on ends the call with an error
//! on every rank (its link closes), never a hang for good.
const std = @import("std");
const Link = @import("link.zig").Link;

/// A part is at most this many bytes a rank (a frame of the table is `world` parts).
pub const max_part = 16 * 1024;
const max_world = 4;

pub const Star = struct {
    /// Rank 0: its followers' links in rank order (links[g - 1] to rank g). A follower: links[0] to rank 0.
    links: *const [3]?Link,
    rank: u32,
    world: u32,
    /// Set while a collective runs: two threads in one are a defect (the ranks would pair the wrong frames), reported
    /// at once rather than left to corrupt a frame.
    busy: *std.atomic.Value(bool),

    pub const Error = error{ StarBusy, BadPart, WorldTooLarge, PeerDown, FrameTooLong };

    fn enter(s: Star) Error!void {
        if (s.world > max_world) return error.WorldTooLarge;
        if (s.busy.cmpxchgStrong(false, true, .acquire, .monotonic) != null) return error.StarBusy;
    }

    fn leave(s: Star) void {
        s.busy.store(false, .release);
    }

    /// Every rank's `mine` (all the same length) into `table` rank after rank (`table.len == world * mine.len`).
    pub fn allGather(s: Star, mine: []const u8, table: []u8) !void {
        if (mine.len > max_part or table.len != s.world * mine.len) return error.BadPart;
        try s.enter();
        defer s.leave();
        const n = mine.len;
        if (s.world == 1) {
            @memcpy(table[0..n], mine);
            return;
        }
        if (s.rank == 0) {
            var buf: [max_part]u8 = undefined;
            @memcpy(table[0..n], mine);
            for (1..s.world) |g| {
                const got = try s.links[g - 1].?.recv(&buf);
                if (got.len != n) return error.BadPart;
                @memcpy(table[g * n ..][0..n], got);
            }
            for (1..s.world) |g| try s.links[g - 1].?.send(table);
        } else {
            const l = s.links[0].?;
            try l.send(mine);
            var big: [max_world * max_part]u8 = undefined;
            const got = try l.recv(&big);
            if (got.len != table.len) return error.BadPart;
            @memcpy(table, got);
        }
    }

    /// `buf` made, byte by byte, the largest of the ranks' (the same length on every rank), in place: the vote of
    /// ballots whose larger byte is the worse outcome.
    pub fn maxBytes(s: Star, buf: []u8) !void {
        if (buf.len > max_part) return error.BadPart;
        try s.enter();
        defer s.leave();
        if (s.world == 1) return;
        if (s.rank == 0) {
            var one: [max_part]u8 = undefined;
            for (1..s.world) |g| {
                const got = try s.links[g - 1].?.recv(&one);
                if (got.len != buf.len) return error.BadPart;
                for (buf, got) |*b, x| b.* = @max(b.*, x);
            }
            for (1..s.world) |g| try s.links[g - 1].?.send(buf);
        } else {
            const l = s.links[0].?;
            try l.send(buf);
            var one: [max_part]u8 = undefined;
            const got = try l.recv(&one);
            if (got.len != buf.len) return error.BadPart;
            @memcpy(buf, got);
        }
    }
};

// -- tests: four ranks on socket pairs --------------------------------------------------------------------------------

/// Four ranks' links on socket pairs, for the tests of the callers of a Star.
pub const Net = struct {
    rank0: [3]?Link,
    follower: [3]?Link,
    busy: [4]std.atomic.Value(bool) = @splat(.init(false)),

    pub fn open() !Net {
        var n: Net = .{ .rank0 = .{ null, null, null }, .follower = .{ null, null, null } };
        for (0..3) |i| {
            var pair: [2]c_int = undefined;
            if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
            n.rank0[i] = .{ .fd = pair[0] };
            n.follower[i] = .{ .fd = pair[1] };
        }
        return n;
    }

    pub fn close(n: *Net) void {
        for (&n.rank0) |*l| if (l.*) |*x| x.close();
        for (&n.follower) |*l| if (l.*) |*x| x.close();
    }

    /// Rank g's Star: rank 0 holds the three links, follower g its own to rank 0.
    pub fn star(n: *Net, g: u32, only: *[3]?Link) Star {
        only.* = .{ null, null, null };
        if (g == 0) only.* = n.rank0 else only[0] = n.follower[g - 1];
        return .{ .links = only, .rank = g, .world = 4, .busy = &n.busy[g] };
    }
};

/// `body` on four threads, rank 0 on the caller's, each with its Star; the first error any rank returned.
pub fn runRanks(comptime body: fn (Star, u32) anyerror!void) !void {
    var net = try Net.open();
    defer net.close();
    var links: [4][3]?Link = undefined;
    var stars: [4]Star = undefined;
    for (0..4) |g| stars[g] = net.star(@intCast(g), &links[g]);
    var errs: [4]?anyerror = @splat(null);
    const Run = struct {
        fn go(s: Star, g: u32, out: *?anyerror) void {
            body(s, g) catch |e| {
                out.* = e;
            };
        }
    };
    var threads: [3]std.Thread = undefined;
    for (1..4) |g| threads[g - 1] = try std.Thread.spawn(.{}, Run.go, .{ stars[g], @as(u32, @intCast(g)), &errs[g] });
    Run.go(stars[0], 0, &errs[0]);
    for (threads) |t| t.join();
    for (errs) |e| if (e) |x| return x;
}

test "an all-gather returns every rank's bytes in rank order on every rank" {
    try runRanks(struct {
        fn f(s: Star, g: u32) !void {
            const mine = [_]u8{ @intCast(g), @intCast(g + 10), @intCast(g + 20) };
            var table: [12]u8 = undefined;
            try s.allGather(&mine, &table);
            try std.testing.expectEqualSlices(u8, &.{ 0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23 }, &table);
            // a second one over the same links: frames stay paired
            const again = [_]u8{ @intCast(g * 2), 7, 7 };
            try s.allGather(&again, &table);
            try std.testing.expectEqualSlices(u8, &.{ 0, 7, 7, 2, 7, 7, 4, 7, 7, 6, 7, 7 }, &table);
        }
    }.f);
}

test "a vote is the byte-wise largest ballot on every rank, and a part of a table is any size up to the bound" {
    try runRanks(struct {
        fn f(s: Star, g: u32) !void {
            var ballot = [_]u8{ 0, 0, 0, 0 };
            ballot[g] = @intCast(g + 1);
            if (g == 2) ballot[0] = 9;
            try s.maxBytes(&ballot);
            try std.testing.expectEqualSlices(u8, &.{ 9, 2, 3, 4 }, &ballot);
            var big: [max_part]u8 = @splat(@intCast(g));
            var table: [4 * max_part]u8 = undefined;
            try s.allGather(&big, &table);
            for (0..4) |r| try std.testing.expect(table[r * max_part] == r and table[(r + 1) * max_part - 1] == r);
        }
    }.f);
}

test "control: a rank whose part has another length ends the collective on every rank instead of shifting the table" {
    // rank 2 sends 2 bytes where the others send 3: rank 0 refuses it and closes its links, so the followers' reads fail
    // (no rank is left waiting), and no table is built from the short part
    var net = try Net.open();
    defer net.close();
    var links: [4][3]?Link = undefined;
    var stars: [4]Star = undefined;
    for (0..4) |g| stars[g] = net.star(@intCast(g), &links[g]);
    var errs: [4]?anyerror = @splat(null);
    const Run = struct {
        fn follower(s: Star, out: *?anyerror) void {
            const mine = [_]u8{ 1, 2, 3 };
            var table: [12]u8 = undefined;
            s.allGather(&mine, &table) catch |e| {
                out.* = e;
            };
        }
        fn short(s: Star) void {
            s.links[0].?.send(&.{ 1, 2 }) catch {};
        }
    };
    const t1 = try std.Thread.spawn(.{}, Run.follower, .{ stars[1], &errs[1] });
    const t2 = try std.Thread.spawn(.{}, Run.short, .{stars[2]});
    const t3 = try std.Thread.spawn(.{}, Run.follower, .{ stars[3], &errs[3] });
    const mine = [_]u8{ 1, 2, 3 };
    var table: [12]u8 = undefined;
    try std.testing.expectError(error.BadPart, stars[0].allGather(&mine, &table));
    for (stars[0].links) |l| {
        if (l) |x| _ = std.c.shutdown(x.fd, 2);
    }
    t1.join();
    t2.join();
    t3.join();
    try std.testing.expectEqual(@as(?anyerror, error.PeerDown), errs[1]);
    try std.testing.expectEqual(@as(?anyerror, error.PeerDown), errs[3]);
}

test "two threads in one collective are refused" {
    var net = try Net.open();
    defer net.close();
    var links: [3]?Link = undefined;
    const s = net.star(0, &links);
    net.busy[0].store(true, .release);
    var t: [4]u8 = undefined;
    try std.testing.expectError(error.StarBusy, s.allGather(&.{1}, &t));
}
