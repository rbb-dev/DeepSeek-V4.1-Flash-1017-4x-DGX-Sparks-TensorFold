//! The owned-rows plan of a prompt chunk of any row count: how many rows each rank owns and where its slab starts.
//! A pure function of the chunk's row count and the rank's place, identical on every rank.

const std = @import("std");

/// Ranks that share a chunk's rows.
pub const world: usize = 4;

/// The owner kernels' row blocks: every slab but the last holds whole blocks (the indexer's 16-row blocks, and rows
/// whose byte size is a multiple of 16 for every row tensor that crosses the wire).
pub const block: usize = 16;

/// The rows of a full slab for a chunk of `n` rows: a quarter, rounded up to a whole block.
pub fn q(n: usize) usize {
    const quarter = (n + world - 1) / world;
    return (quarter + block - 1) / block * block;
}

/// A slab: the chunk rows [start, start + n) owned by one rank.
pub const Slab = struct { start: usize, n: usize };

/// Slab `k` of a chunk of `n` rows: rows [k q, k q + n_k) with n_k = clamp(n - k q, 0, q).
pub fn slab(n: usize, k: usize) Slab {
    const w = q(n);
    const start = @min(k * w, n);
    return .{ .start = k * w, .n = @min(n - start, w) };
}

/// All ranks' slabs of a chunk of `n` rows.
pub fn slabs(n: usize) [world]Slab {
    var out: [world]Slab = undefined;
    for (&out, 0..) |*s, k| s.* = slab(n, k);
    return out;
}

/// The slabs' non-empty rows, in order, cover [0, n) exactly once.
fn covers(n: usize) bool {
    var next: usize = 0;
    for (slabs(n)) |s| {
        if (s.n == 0) continue;
        if (s.start != next) return false;
        next += s.n;
    }
    return next == n;
}

test "the slabs of every chunk from 1 to 4,096 rows cover it exactly once, in whole blocks but the last" {
    var rows: usize = 1;
    while (rows <= 4096) : (rows += 1) {
        try std.testing.expect(covers(rows));
        const w = q(rows);
        try std.testing.expectEqual(@as(usize, 0), w % block);
        try std.testing.expect(w * world >= rows);
        // a quarter, rounded up to one block, and no larger
        try std.testing.expect(w < (rows + world - 1) / world + block);
        const ss = slabs(rows);
        for (ss, 0..) |s, k| {
            try std.testing.expectEqual(k * w, s.start);
            try std.testing.expect(s.n <= w);
            // every slab but the last non-empty one is full
            if (s.n != 0 and k + 1 < world and ss[k + 1].n != 0) try std.testing.expectEqual(w, s.n);
            // a slab's byte offset in any row tensor of a multiple-of-16 row size is 16-aligned
            try std.testing.expectEqual(@as(usize, 0), (s.start * 4) % 16);
        }
    }
}

test "the production shapes: 4,096 and 2,048 rows are the engine's full quarters; the examples" {
    try std.testing.expectEqual(@as(usize, 1024), q(4096));
    try std.testing.expectEqual(@as(usize, 512), q(2048));
    const s500 = slabs(500);
    try std.testing.expectEqual(@as(usize, 128), q(500));
    try std.testing.expectEqual(@as(usize, 128), s500[2].n);
    try std.testing.expectEqual(@as(usize, 116), s500[3].n);
    try std.testing.expectEqual(@as(usize, 464), q(1800));
    try std.testing.expectEqual(@as(usize, 408), slabs(1800)[3].n);
    try std.testing.expectEqual(@as(usize, 752), q(3000));
    try std.testing.expectEqual(@as(usize, 744), slabs(3000)[3].n);
    const s13 = slabs(13);
    try std.testing.expectEqual(@as(usize, 16), q(13));
    try std.testing.expectEqual(@as(usize, 13), s13[0].n);
    try std.testing.expectEqual(@as(usize, 0), s13[1].n);
    try std.testing.expectEqual(@as(usize, 0), s13[3].n);
}

test "negative control: a plan that rounds the quarter to 8 breaks the 16-row indexer blocks" {
    var broke = false;
    var rows: usize = 1;
    while (rows <= 4096) : (rows += 1) {
        const w8 = 8 * ((rows + 3) / 4 + 7) / 8;
        if (w8 % block != 0) broke = true;
    }
    try std.testing.expect(broke);
}
