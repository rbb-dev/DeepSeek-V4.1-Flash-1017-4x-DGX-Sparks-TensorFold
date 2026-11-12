//! The grouped experts' tiles in an order that starts on the rows a node already holds: the chunk's rows are cut into
//! segments (the node's own quarter first), and a tile takes rows of one segment only, so the tiles of segment 0 can
//! be launched before the other segments' rows have arrived and the rest after.
//!
//! One grouping (tf_dsn_group_place's off / ent / erow, expert after expert) is laid out again: each expert's block is
//! its segments' rows one after the other, every segment padded to whole tiles by repeating the segment's first row
//! (that entry is computed and stored twice, with the same bytes), so a kernel that takes (expert, first row in the
//! expert's block) tiles and the block's length runs the layout unchanged. Entries keep their numbers: the outputs
//! are addressed by `ent`, never by position.

const std = @import("std");

/// A tile of the grouped kernels' list: (expert, first row in the expert's block).
pub const Tile = [2]i32;

/// The grouped tiles' rows (mx.pf_bm, native.cu pf::kBm).
pub const tile_rows: usize = 64;

pub const Layout = struct {
    /// [experts + 1]: each expert's block start in the padded arrays, the last the total.
    off: []i32,
    ent: []i32,
    erow: []i32,
    /// The tiles of segment 0, expert after expert, and the tiles of every other segment, expert after expert and
    /// segment after segment.
    local: []Tile,
    rest: []Tile,
};

/// The segment of a chunk row: `segments` equal parts of `seg_rows` rows each (two: segment 0 is the first part, segment 1
/// everything after it).
pub fn segmentOf(row: usize, segments: usize, seg_rows: usize) usize {
    if (segments == 2) return @intFromBool(row >= seg_rows);
    return @min(row / seg_rows, segments - 1);
}

/// The grouping (off [experts + 1], ent and erow [off[experts]]) re-laid by segment. Memory from `a` (an arena).
pub fn segmented(a: std.mem.Allocator, off: []const i32, ent: []const i32, erow: []const i32, experts: usize, segments: usize, seg_rows: usize) !Layout {
    const o = try a.alloc(i32, experts + 1);
    var en: std.ArrayList(i32) = .empty;
    var er: std.ArrayList(i32) = .empty;
    var lo: std.ArrayList(Tile) = .empty;
    var re: std.ArrayList(Tile) = .empty;
    for (0..experts) |e| {
        o[e] = @intCast(en.items.len);
        const block: usize = en.items.len;
        const b: usize = @intCast(off[e]);
        const c: usize = @intCast(off[e + 1] - off[e]);
        for (0..segments) |sg| {
            const start = en.items.len;
            for (0..c) |i| {
                if (segmentOf(@intCast(erow[b + i]), segments, seg_rows) != sg) continue;
                try en.append(a, ent[b + i]);
                try er.append(a, erow[b + i]);
            }
            if (en.items.len == start) continue;
            const first_ent = en.items[start];
            const first_row = er.items[start];
            while ((en.items.len - start) % tile_rows != 0) {
                try en.append(a, first_ent);
                try er.append(a, first_row);
            }
            var m0 = start - block;
            while (m0 < en.items.len - block) : (m0 += tile_rows) try (if (sg == 0) lo else re).append(a, .{ @intCast(e), @intCast(m0) });
        }
    }
    o[experts] = @intCast(en.items.len);
    return .{ .off = o, .ent = en.items, .erow = er.items, .local = lo.items, .rest = re.items };
}

/// The baseline tile list of a grouping: every expert with a row, in id order, a tile for each `tile_rows` rows.
pub fn todays(a: std.mem.Allocator, off: []const i32, experts: usize) ![]Tile {
    var t: std.ArrayList(Tile) = .empty;
    for (0..experts) |e| {
        const c: usize = @intCast(off[e + 1] - off[e]);
        var m0: usize = 0;
        while (m0 < c) : (m0 += tile_rows) try t.append(a, .{ @intCast(e), @intCast(m0) });
    }
    return t.items;
}

/// Whether a layout is what `segmented` promises over the grouping it was made from: every entry of the grouping in its
/// expert's block, only padding beyond them, every tile's rows from one segment, the local list holding segment 0's tiles
/// and the rest list the others', every row of every block under exactly one tile.
pub fn holds(l: Layout, off: []const i32, ent: []const i32, erow: []const i32, experts: usize, segments: usize, seg_rows: usize) bool {
    if (l.off[0] != 0 or l.ent.len != l.erow.len or l.off[experts] != l.ent.len) return false;
    if ((l.local.len + l.rest.len) * tile_rows != l.ent.len) return false;
    for ([_][]const Tile{ l.local, l.rest }, 0..) |list, which| for (list) |t| {
        const e: usize = @intCast(t[0]);
        const m0: usize = @intCast(t[1]);
        const bs: usize = @intCast(l.off[e]);
        const be: usize = @intCast(l.off[e + 1]);
        if (m0 % tile_rows != 0 or bs + m0 + tile_rows > be) return false;
        const sg = segmentOf(@intCast(l.erow[bs + m0]), segments, seg_rows);
        if ((sg == 0) != (which == 0)) return false;
        for (0..tile_rows) |i| if (segmentOf(@intCast(l.erow[bs + m0 + i]), segments, seg_rows) != sg) return false;
    };
    // the entries: each grouping row once at least in its expert's block, with its row, and nothing else in it
    for (0..experts) |e| {
        const bs: usize = @intCast(l.off[e]);
        const be: usize = @intCast(l.off[e + 1]);
        if (be < bs or (be - bs) % tile_rows != 0) return false;
        for (@intCast(off[e])..@intCast(off[e + 1])) |i| {
            var found = false;
            for (bs..be) |j| if (l.ent[j] == ent[i] and l.erow[j] == erow[i]) {
                found = true;
                break;
            };
            if (!found) return false;
        }
        for (bs..be) |j| {
            var found = false;
            for (@intCast(off[e])..@intCast(off[e + 1])) |i| if (l.ent[j] == ent[i] and l.erow[j] == erow[i]) {
                found = true;
                break;
            };
            if (!found) return false;
        }
    }
    return true;
}

// A small grouping: 4 chunk quarters of 32 rows, three experts. Expert 0 has 5 rows (quarters 2, 0, 3, 0, 1), expert 1 none,
// expert 2 has 130 rows (every quarter, uneven).
fn sample(a: std.mem.Allocator) !struct { off: []i32, ent: []i32, erow: []i32 } {
    const rows = [_]i32{ 70, 3, 100, 31, 33 };
    var ent: std.ArrayList(i32) = .empty;
    var erow: std.ArrayList(i32) = .empty;
    for (rows, 0..) |r, i| {
        try ent.append(a, @intCast(i * 7));
        try erow.append(a, r);
    }
    var prng = std.Random.DefaultPrng.init(5);
    for (0..130) |i| {
        try ent.append(a, @intCast(100 + i));
        try erow.append(a, @intCast(prng.random().uintLessThan(usize, 128)));
    }
    const off = try a.dupe(i32, &.{ 0, 5, 5, 135 });
    return .{ .off = off, .ent = ent.items, .erow = erow.items };
}

test "segments of one: the layout is the baseline tile list" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = try sample(a);
    const l = try segmented(a, g.off, g.ent, g.erow, 3, 1, 32);
    const want = try todays(a, g.off, 3);
    try std.testing.expectEqual(want.len, l.local.len);
    for (want, l.local) |w, t| try std.testing.expectEqual(w, t);
    try std.testing.expectEqual(@as(usize, 0), l.rest.len);
    try std.testing.expect(holds(l, g.off, g.ent, g.erow, 3, 1, 32));
}

test "four segments: a tile's rows come from one quarter, the node's own quarter apart" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = try sample(a);
    const l = try segmented(a, g.off, g.ent, g.erow, 3, 4, 32);
    try std.testing.expect(holds(l, g.off, g.ent, g.erow, 3, 4, 32));
    // expert 0: quarters 0 (2 rows), 1 (1), 2 (1), 3 (1): one tile each, one local and three rest; expert 1 none
    var in_e0: [2]usize = .{ 0, 0 };
    for (l.local) |t| if (t[0] == 0) {
        in_e0[0] += 1;
    };
    for (l.rest) |t| if (t[0] == 0) {
        in_e0[1] += 1;
    };
    try std.testing.expectEqual([2]usize{ 1, 3 }, in_e0);
    for ([_][]const Tile{ l.local, l.rest }) |list| for (list) |t| try std.testing.expect(t[0] != 1);
    // the padding repeats the segment's first row: expert 0's block is four tiles of 64 rows
    try std.testing.expectEqual(@as(i32, 256), l.off[1] - l.off[0]);
    // (segment 0 holds two rows, entries 7 and 21: the rest of its tile is the first one again)
    try std.testing.expectEqual(@as(i32, 7), l.ent[0]);
    try std.testing.expectEqual(@as(i32, 21), l.ent[1]);
    for (2..64) |i| {
        try std.testing.expectEqual(l.ent[0], l.ent[i]);
        try std.testing.expectEqual(l.erow[0], l.erow[i]);
    }
}

test "two segments: the node's quarter against everything after it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = try sample(a);
    const l2 = try segmented(a, g.off, g.ent, g.erow, 3, 2, 32);
    const l4 = try segmented(a, g.off, g.ent, g.erow, 3, 4, 32);
    try std.testing.expect(holds(l2, g.off, g.ent, g.erow, 3, 2, 32));
    // the same local tiles (segment 0), fewer or as many rest tiles
    try std.testing.expectEqual(l4.local.len, l2.local.len);
    try std.testing.expect(l2.rest.len <= l4.rest.len);
    // never fewer tiles than the baseline
    const want = try todays(a, g.off, 3);
    try std.testing.expect(l2.local.len + l2.rest.len >= want.len);
}

test "a layout that was changed is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = try sample(a);
    // a row moved to another quarter inside a tile
    var l = try segmented(a, g.off, g.ent, g.erow, 3, 4, 32);
    const at: usize = @intCast(l.off[2] + 1);
    l.erow[at] = if (l.erow[at] < 32) 100 else 3;
    try std.testing.expect(!holds(l, g.off, g.ent, g.erow, 3, 4, 32));
    // a tile dropped
    var m = try segmented(a, g.off, g.ent, g.erow, 3, 4, 32);
    m.rest = m.rest[1..];
    try std.testing.expect(!holds(m, g.off, g.ent, g.erow, 3, 4, 32));
    // a rest tile listed as local
    var n = try segmented(a, g.off, g.ent, g.erow, 3, 4, 32);
    n.local = try std.mem.concat(a, Tile, &.{ n.local, n.rest[0..1] });
    n.rest = n.rest[1..];
    try std.testing.expect(!holds(n, g.off, g.ent, g.erow, 3, 4, 32));
    // an entry lost from its block
    var p = try segmented(a, g.off, g.ent, g.erow, 3, 4, 32);
    p.ent[@intCast(p.off[2] + 70)] = 9999;
    try std.testing.expect(!holds(p, g.off, g.ent, g.erow, 3, 4, 32));
}
