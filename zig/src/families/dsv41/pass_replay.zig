//! tf-dsv41-rows-replay's passes (its hc, wp4 and idx cases): the pass sizes it replays (--rows) and, at each, the four
//! nodes' owned quarters, the offsets a quarter's repack reads in the gathered planes, and the row a control changes.
//! Its kvsrc case: what an in-place all-gather takes of one rank's array (a rank's span), on the host.
const std = @import("std");
const comm = @import("comm.zig");
const model = @import("model.zig");
const prompt = @import("prompt.zig");

/// The pass sizes replayed without --rows: a 1,024-row wave's (q = 256), a chunk's (512) and a 4,096-row pass's (1,024).
pub const default_rows = [_]usize{ 1024, 2048, 4096 };

/// The most pass sizes one --rows list names.
pub const max_sizes = 8;

/// A replay's pass sizes, in the order given.
pub const Sizes = struct {
    buf: [max_sizes]usize = undefined,
    len: usize = 0,

    pub fn slice(s: *const Sizes) []const usize {
        return s.buf[0..s.len];
    }
};

/// --rows' text (null: default_rows): comma-separated pass sizes, each a whole number of 1,024-row waves up to the
/// engine's largest pass (model.pass_rows_max), so a node's quarter is whole 256-row blocks.
pub fn parseRows(text: ?[]const u8) error{BadRows}!Sizes {
    var s: Sizes = .{};
    const t = text orelse {
        @memcpy(s.buf[0..default_rows.len], &default_rows);
        s.len = default_rows.len;
        return s;
    };
    var it = std.mem.splitScalar(u8, t, ',');
    while (it.next()) |item| {
        const rows = std.fmt.parseInt(usize, item, 10) catch return error.BadRows;
        if (rows == 0 or rows % 1024 != 0 or rows > model.pass_rows_max or s.len == max_sizes) return error.BadRows;
        s.buf[s.len] = rows;
        s.len += 1;
    }
    return s;
}

/// Node k's owned rows in a full pass of `rows` rows: [k q, (k + 1) q), q = rows / 4 (qhcRange's and idxShardRange's
/// at n == cap over the four ranks).
pub fn quarter(rows: usize, k: usize) prompt.QRows {
    const q = rows / 4;
    return .{ .start = k * q, .n = q };
}

/// The byte offset of plane `w`'s row `row` in a pass's gathered planes, fp32 [2, rows, d] (ga or gm): the planes lie
/// `rows` rows apart, the stride a quarter's repack (qhcRepackG, ch.cap = the pass's rows) reads them at.
pub fn planeAt(w: usize, rows: usize, row: usize, d: usize) usize {
    return (w * rows + row) * d * 4;
}

/// The row a control changes: 5 rows into the last node's quarter.
pub fn controlRow(rows: usize) usize {
    return quarter(rows, 3).start + 5;
}

/// A byte range of an array.
pub const Span = struct { off: usize, len: usize };

/// Rank `rank`'s part of the in-place all-gather `p` (every rank's part at p.buf + its rank x p.bytes: Comm.allGatherMany)
/// as a byte range of the array at `base`.
pub fn spanOf(p: comm.InPlace, base: u64, rank: usize) Span {
    return .{ .off = @intCast(p.buf - base + rank * p.bytes), .len = p.bytes };
}

/// What the all-gather `p` leaves in `dst` from rank `rank`'s whole array `src` (both from `base`): its span, copied.
/// Returns the bytes of `src` outside its span that are not `poison`: written where the gather does not take them.
pub fn takeSpan(dst: []u8, src: []const u8, p: comm.InPlace, base: u64, rank: usize, poison: u8) usize {
    const sp = spanOf(p, base, rank);
    @memcpy(dst[sp.off..][0..sp.len], src[sp.off..][0..sp.len]);
    var stray: usize = 0;
    for (src[0..sp.off]) |b| stray += @intFromBool(b != poison);
    for (src[sp.off + sp.len ..]) |b| stray += @intFromBool(b != poison);
    return stray;
}

test "the replay's passes: unset gives a 1,024-row wave's, a chunk's and a 4,096-row pass's" {
    const s = try parseRows(null);
    try std.testing.expectEqualSlices(usize, &.{ 1024, 2048, 4096 }, s.slice());
}

test "--rows: whole 1,024-row waves up to the engine's largest pass, in the order given" {
    try std.testing.expectEqualSlices(usize, &.{4096}, (try parseRows("4096")).slice());
    try std.testing.expectEqualSlices(usize, &.{ 2048, 1024, 3072 }, (try parseRows("2048,1024,3072")).slice());
    for (bad) |text| try std.testing.expectError(error.BadRows, parseRows(text));
    try std.testing.expectError(error.BadRows, parseRows("1024,1024,1024,1024,1024,1024,1024,1024,1024")); // nine
}

test "a pass's quarters: node k's rows [k q, (k + 1) q), every start on a 16-row block" {
    for (default_rows) |rows| {
        var next: usize = 0;
        for (0..4) |k| {
            const r = quarter(rows, k);
            try std.testing.expectEqual(next, r.start);
            try std.testing.expectEqual(rows / 4, r.n);
            try std.testing.expectEqual(@as(usize, 0), r.start % 16);
            next += r.n;
        }
        try std.testing.expectEqual(rows, next);
    }
    try std.testing.expectEqual(prompt.QRows{ .start = 3072, .n = 1024 }, quarter(4096, 3));
}

test "a 4,096-row pass's plane 1 starts 4,096 rows in: the repack's stride is the pass's rows" {
    try std.testing.expectEqual(@as(usize, 4096 * 5120 * 4), planeAt(1, 4096, 0, 5120));
    try std.testing.expectEqual(@as(usize, (4096 + 3072) * 5120 * 4), planeAt(1, 4096, 3072, 5120));
    try std.testing.expectEqual(@as(usize, 3072 * 5120 * 4), planeAt(0, 4096, 3072, 5120));
    try std.testing.expectEqual(@as(usize, (2048 + 512) * 5120 * 4), planeAt(1, 2048, 512, 5120));
}

test "a control's row lies in the last node's quarter" {
    for (default_rows) |rows| {
        const c = controlRow(rows);
        try std.testing.expect(c > quarter(rows, 3).start and c < rows);
    }
    try std.testing.expectEqual(@as(usize, 3077), controlRow(4096));
}

test "a rank's span of an in-place all-gather: its part at buf + rank x bytes, a range of its array" {
    const p: comm.InPlace = .{ .buf = 0x1000 + 40, .bytes = 8 };
    try std.testing.expectEqual(Span{ .off = 40, .len = 8 }, spanOf(p, 0x1000, 0));
    try std.testing.expectEqual(Span{ .off = 56, .len = 8 }, spanOf(p, 0x1000, 2));
}

test "the boundary's gathers take from block k of q rows its own groups of each cache array" {
    var cs: prompt.Caches = .{};
    const base = [4]u64{ 0x100_0000, 0x200_0000, 0x300_0000, 0x400_0000 };
    cs.comp_codes[20] = base[0];
    cs.comp_scales[20] = base[1];
    cs.idx_codes[20] = base[2];
    cs.idx_scales[20] = base[3];
    const row = [4]usize{ 256, 32, 64, 4 }; // a group's bytes: head_dim 512 / 2, / 16; index_head_dim 128 / 2, / 32
    for ([_]usize{ 512, 256, 1024 }) |q| for (0..2048 / q) |k| {
        const parts = prompt.boundaryGathers(&cs, 20, 4096, .{ .start = k * q, .n = q }, 512, 128);
        for (parts, base, row) |p, b, w| try std.testing.expectEqual(Span{ .off = (4096 + k * q) * w, .len = q * w }, spanOf(p, b, k));
    };
}

test "takeSpan: a rank's span into the gathered array; its bytes written outside the span are counted" {
    const p: comm.InPlace = .{ .buf = 0x1000 + 4, .bytes = 4 };
    var src: [16]u8 = @splat(0xa5);
    for (src[8..12], 1..) |*b, i| b.* = @intCast(i); // rank 1's span: [8, 12)
    var dst: [16]u8 = @splat(0xa5);
    try std.testing.expectEqual(@as(usize, 0), takeSpan(&dst, &src, p, 0x1000, 1, 0xa5));
    try std.testing.expectEqualSlices(u8, &src, &dst);
    src[3] = 0; // outside the span
    src[15] = 7;
    @memset(&dst, 0xa5);
    try std.testing.expectEqual(@as(usize, 2), takeSpan(&dst, &src, p, 0x1000, 1, 0xa5));
    var want: [16]u8 = @splat(0xa5);
    @memcpy(want[8..12], &[_]u8{ 1, 2, 3, 4 });
    try std.testing.expectEqualSlices(u8, &want, &dst);
}