//! The shape of the SSD landing's broadcast on the Rocinante transport: what rank 0 posts for one piece of the
//! landing's stream (land.zig: the spans' bytes in order, cut into pieces), as a plan of RDMA WRITEs a proxy would post.
//! A piece goes to each follower as two WRITEs, its first half on rail 0 and its second on rail 1, the cut on a cache-line
//! boundary so no line is written from two rails; a word after both halves tells the follower the piece is whole.
//! Pure host logic: where the bytes land (a registered host landing, then a copy into the pool, or the pool itself
//! registered) is the memory-layout question this module does not decide.
const std = @import("std");

pub const rails = 2;
/// The cut between a piece's halves is a multiple of this.
pub const line = 64;
pub const max_followers = 3;

/// One WRITE: bytes [offset, offset + len) of the piece to `peer` (1..3) on `rail`.
pub const Write = struct { peer: u8, rail: u8, offset: usize, len: usize };

/// The first half's length: half the piece, rounded up to a line, never past the piece.
pub fn firstHalf(len: usize) usize {
    return @min(len, std.mem.alignForward(usize, (len + 1) / 2, line));
}

/// The WRITEs of a piece of `len` bytes to `followers` followers (peers 1 ..= followers): per follower the first half on
/// rail 0 then the second on rail 1; a half with no bytes posts nothing.
pub fn pieceWrites(len: usize, followers: u8, out: []Write) error{ TooManyFollowers, NoRoom }![]Write {
    if (followers == 0 or followers > max_followers) return error.TooManyFollowers;
    if (out.len < followers * rails) return error.NoRoom;
    const a = firstHalf(len);
    var n: usize = 0;
    for (1..@as(usize, followers) + 1) |p| {
        if (a > 0) {
            out[n] = .{ .peer = @intCast(p), .rail = 0, .offset = 0, .len = a };
            n += 1;
        }
        if (len > a) {
            out[n] = .{ .peer = @intCast(p), .rail = 1, .offset = a, .len = len - a };
            n += 1;
        }
    }
    return out[0..n];
}

/// What a landing of `total` bytes in pieces of `piece` bytes posts to `followers` followers.
pub const Counts = struct {
    pieces: usize,
    writes: usize,
    words: usize,
    /// Bytes rank 0 sends on each rail (every follower gets the whole stream: no relay).
    rail_bytes: [rails]usize,
};

pub fn counts(total: usize, piece: usize, followers: u8) Counts {
    var c: Counts = .{ .pieces = 0, .writes = 0, .words = 0, .rail_bytes = @splat(0) };
    if (piece == 0) return c;
    var off: usize = 0;
    var buf: [max_followers * rails]Write = undefined;
    while (off < total) : (off += piece) {
        const len = @min(piece, total - off);
        const w = pieceWrites(len, followers, &buf) catch return c;
        c.pieces += 1;
        c.writes += w.len;
        c.words += followers * rails; // one word a follower on each rail, after that rail's WRITE
        for (w) |x| c.rail_bytes[x.rail] += x.len;
    }
    return c;
}

test "a piece goes to each follower as two line-aligned halves on the two rails, covering it once" {
    var buf: [max_followers * rails]Write = undefined;
    for ([_]usize{ 1, 63, 64, 65, 127, 128, 129, 4096, 1 << 20, (1 << 20) + 7 }) |len| {
        for ([_]u8{ 1, 2, 3 }) |f| {
            const w = try pieceWrites(len, f, &buf);
            for (1..@as(usize, f) + 1) |p| {
                var covered: usize = 0;
                var next: usize = 0;
                var rails_seen: [rails]bool = @splat(false);
                for (w) |x| if (x.peer == p) {
                    try std.testing.expectEqual(next, x.offset); // contiguous, in order, no overlap
                    try std.testing.expect(x.len > 0);
                    try std.testing.expect(!rails_seen[x.rail]);
                    rails_seen[x.rail] = true;
                    next = x.offset + x.len;
                    covered += x.len;
                    if (x.offset + x.len < len) try std.testing.expectEqual(@as(usize, 0), (x.offset + x.len) % line); // the cut
                };
                try std.testing.expectEqual(len, covered);
            }
        }
    }
}

test "the halves are balanced to a line, a piece under a line uses one rail, an empty piece posts nothing" {
    var buf: [max_followers * rails]Write = undefined;
    const w = try pieceWrites(1 << 20, 3, &buf);
    try std.testing.expectEqual(@as(usize, 6), w.len);
    for (w) |x| try std.testing.expect(x.len == (1 << 19));
    const odd = try pieceWrites((1 << 20) + 100, 1, &buf);
    try std.testing.expect(odd[0].len >= odd[1].len and odd[0].len - odd[1].len <= line + 100);
    const tiny = try pieceWrites(10, 2, &buf);
    try std.testing.expectEqual(@as(usize, 2), tiny.len); // one WRITE a follower, both on rail 0
    for (tiny) |x| try std.testing.expectEqual(@as(u8, 0), x.rail);
    try std.testing.expectEqual(@as(usize, 0), (try pieceWrites(0, 3, &buf)).len);
}

test "a landing's counts: every follower gets the whole stream, split over the rails, a word a follower a rail a piece" {
    // 1 GiB in 16 MiB pieces to three followers
    const c = counts(1 << 30, 16 << 20, 3);
    try std.testing.expectEqual(@as(usize, 64), c.pieces);
    try std.testing.expectEqual(@as(usize, 64 * 6), c.writes);
    try std.testing.expectEqual(@as(usize, 64 * 6), c.words);
    try std.testing.expectEqual(@as(usize, 3 * (1 << 29)), c.rail_bytes[0]);
    try std.testing.expectEqual(@as(usize, 3 * (1 << 29)), c.rail_bytes[1]);
    // control: the rails together carry the stream once a follower
    try std.testing.expectEqual(3 * (@as(usize, 1) << 30), c.rail_bytes[0] + c.rail_bytes[1]);
    try std.testing.expectEqual(@as(usize, 0), counts(1 << 20, 0, 3).pieces);
}

test "control: no follower, or more than the engine has, is refused; a short output is refused" {
    var buf: [max_followers * rails]Write = undefined;
    try std.testing.expectError(error.TooManyFollowers, pieceWrites(100, 0, &buf));
    try std.testing.expectError(error.TooManyFollowers, pieceWrites(100, 4, &buf));
    try std.testing.expectError(error.NoRoom, pieceWrites(100, 3, buf[0..5]));
}
