//! TF_DS_HEAD_CAND: a sampled decode round's first candidates from each node's own columns of the head. Every node runs
//! tf_ds_top_cand_rows on its part of the vocabulary (`k` of each row); one exchange brings the four parts to every node;
//! rank 0 merges them here into the first `c` of each row in sampling.topCandidates' order: the set and the order
//! the baseline kernel gives on the whole rows, or false where the baseline kernel could flag a row (the round then reads the
//! whole head, the baseline path).
const std = @import("std");
const sampling = @import("sampling.zig");

/// The count each node selects for a round's count `c` (1 .. sampling.max_candidates): 32 or 64, so the round's last
/// stretch has two captured variants at most.
pub fn kFor(c: usize) u8 {
    return if (c <= 32) 32 else 64;
}

/// Where each part's columns start in the vocabulary, the parts in the order a one-row quarters exchange lands them
/// (node g = TP2 rank r + 2 x pair p lands at part 2r + p): g0, g2, g1, g3. A row of logits is the TP2 ranks' halves
/// in rank order, each half its pairs' columns in pair order (hw: the head's columns a pair).
pub fn vocabOffsets(hw: [2]usize) [4]u32 {
    const w0: u32 = @intCast(hw[0]);
    const w1: u32 = @intCast(hw[1]);
    return .{ 0, w0, w0 + w1, 2 * w0 + w1 };
}

/// One node's part of a round: vals [rows, k] f32, ids [rows, k] i64 (its own column numbers), flags [rows] i32, the
/// arrays tf_ds_top_cand_rows writes, each at a 16-byte boundary (the part travels as whole 16-byte words).
pub const Layout = struct {
    rows: usize,
    k: usize,
    ids: usize, // the ids' byte offset
    flags: usize, // the flags' byte offset
    bytes: usize, // the part's bytes, a multiple of 16
};

pub fn layout(rows: usize, k: usize) Layout {
    const ids = std.mem.alignForward(usize, rows * k * 4, 16);
    const flags = ids + std.mem.alignForward(usize, rows * k * 8, 16);
    return .{ .rows = rows, .k = k, .ids = ids, .flags = flags, .bytes = flags + std.mem.alignForward(usize, rows * 4, 16) };
}

/// The four parts as they land, one after another (`lay.bytes` each, the vocabulary's order: part j's column i is
/// id off[j] + i), merged into each row's first `c` (vals and ids [rows, c], the row's own value bits): true, the set
/// and the order of the baseline kernel on the whole rows (no row of which it would flag). False when a node flagged a row
/// (a NaN, more ties at its cut than its kernel holds) or when the baseline kernel could have flagged one: the row's c-th
/// value T is some node's last (k-th) value and at least two nodes' lists hold T (only then can the whole row hold more
/// than 2,048 values equal to T without a node flagging).
pub fn merge(parts: []const u8, lay: Layout, off: [4]u32, c: usize, vals: []f32, ids: []i64) bool {
    const k = lay.k;
    std.debug.assert(c >= 1 and c <= k and parts.len >= 4 * lay.bytes and vals.len >= lay.rows * c and ids.len >= lay.rows * c);
    for (0..lay.rows) |r| {
        var list: [4]Node = undefined;
        for (&list, 0..) |*n, j| {
            n.* = .{ .p = parts[j * lay.bytes ..][0..lay.bytes], .lay = lay, .r = r, .off = off[j] };
            if (n.flag() != 0) return false;
        }
        // the first c of the four sorted lists: each list's next, the best by sampling's order (a list of k >= c
        // never runs out)
        var at: [4]usize = @splat(0);
        for (0..c) |i| {
            var best: usize = 0;
            for (1..4) |j| {
                if (before(list[j].val(at[j]), list[j].id(at[j]), list[best].val(at[best]), list[best].id(at[best]))) best = j;
            }
            vals[r * c + i] = list[best].val(at[best]);
            ids[r * c + i] = list[best].id(at[best]);
            at[best] += 1;
        }
        // the baseline kernel flags the row when more than 2,048 values equal its c-th (T): only possible here when T is
        // some node's k-th value (that node's own count of T is not in its list) and another node's list holds T too
        const t = vals[r * c + c - 1];
        var cut = false;
        var holders: usize = 0;
        for (list) |n| {
            cut = cut or n.val(k - 1) == t;
            for (0..k) |i| {
                if (n.val(i) == t) {
                    holders += 1;
                    break;
                }
            }
        }
        if (cut and holders >= 2) return false;
    }
    return true;
}

/// One node's list of one row in its part (vals and ids as the kernel wrote them; ids made the vocabulary's).
const Node = struct {
    p: []const u8,
    lay: Layout,
    r: usize,
    off: u32,

    fn val(n: Node, i: usize) f32 {
        return @bitCast(std.mem.readInt(u32, n.p[(n.r * n.lay.k + i) * 4 ..][0..4], .little));
    }

    fn id(n: Node, i: usize) i64 {
        return std.mem.readInt(i64, n.p[n.lay.ids + (n.r * n.lay.k + i) * 8 ..][0..8], .little) + n.off;
    }

    fn flag(n: Node) i32 {
        return std.mem.readInt(i32, n.p[n.lay.flags + n.r * 4 ..][0..4], .little);
    }
};

/// sampling.topCandidates' order (tf_ds_top_cand_rows' cand_before): the larger value, then the lower id; +0 == -0.
fn before(va: f32, ia: i64, vb: f32, ib: i64) bool {
    return va > vb or (va == vb and ia < ib);
}

// ---- TF_DS_HEAD_KEY: a greedy round's tokens from the nodes' bests

/// The verify frame's head byte for a greedy round by the nodes' bests (head_cand's counts are 32 and 64).
pub const key: u8 = 1;

/// A node's best of each row (tf_ds_argmax_part): the value f32, its index i32 (0x7fffffff: none), whether the row's
/// first column is a NaN i32, a zero word.
pub const key_row: usize = 16;

/// No index: a node whose columns hold no number.
const no_index: i32 = 0x7fffffff;

/// The four nodes' bests of `rows` rows (`key_row` bytes each, a part per node, part j's column i is id off[j] + i)
/// merged into each row's token as tf_ds_argmax_rows gives it on the whole row: the first index of the largest value, a
/// NaN never larger; 0 when the row's first value (column 0 of the part at offset 0) is a NaN.
pub fn mergeKeys(parts: []const u8, rows: usize, off: [4]u32, tokens: []u32) void {
    std.debug.assert(parts.len >= 4 * rows * key_row and tokens.len >= rows);
    const first = std.mem.indexOfScalar(u32, &off, 0) orelse unreachable; // the part holding the row's first column
    for (0..rows) |r| {
        if (std.mem.readInt(i32, parts[(first * rows + r) * key_row + 8 ..][0..4], .little) != 0) {
            tokens[r] = 0;
            continue;
        }
        var bv: f32 = 0;
        var bid: i64 = -1; // none yet
        for (0..4) |j| {
            const p = parts[(j * rows + r) * key_row ..][0..key_row];
            const i = std.mem.readInt(i32, p[4..8], .little);
            if (i == no_index) continue; // no number in this node's columns
            const v: f32 = @bitCast(std.mem.readInt(u32, p[0..4], .little));
            const id = @as(i64, off[j]) + i;
            if (bid < 0 or v > bv or (v == bv and id < bid)) {
                bv = v;
                bid = id;
            }
        }
        tokens[r] = if (bid < 0) 0 else @intCast(bid);
    }
}

// ---- host models of tf_ds_top_cand_rows (the tests' reference)

const ties_max = 2048; // the kernel's tie buffer

/// tf_ds_top_cand_rows on one row (count `count`, as the kernel's `k` = min(count, n)): 1 when the row holds a NaN, 2
/// when more than 2,048 values equal its count-th, else 0 with the first count in vals / ids.
fn kernelModel(row: []const f32, count: usize, vals: []f32, ids: []i64) i32 {
    for (row) |v| if (v != v) return 1;
    const k = @min(count, row.len);
    _ = sampling.topCandidates(row, k, vals[0..k], ids[0..k]);
    const t = vals[k - 1];
    var ties: usize = 0;
    for (row) |v| ties += @intFromBool(v == t);
    return if (ties > ties_max) 2 else 0;
}

/// The four nodes' parts of `rows` rows (each [n], the nodes' columns at off[j] .. off[j] + widths[j]) as the kernel
/// writes them (a flagged row's vals and ids left as they were: here 0x7f bytes).
fn partsModel(gpa: std.mem.Allocator, rows: []const []const f32, widths: [4]usize, off: [4]u32, k: usize) ![]u8 {
    const lay = layout(rows.len, k);
    const out = try gpa.alloc(u8, 4 * lay.bytes);
    @memset(out, 0x7f);
    var v: [64]f32 = undefined;
    var id: [64]i64 = undefined;
    for (0..4) |j| {
        const p = out[j * lay.bytes ..][0..lay.bytes];
        for (rows, 0..) |row, r| {
            const flag = kernelModel(row[off[j]..][0..widths[j]], k, &v, &id);
            std.mem.writeInt(i32, p[lay.flags + r * 4 ..][0..4], flag, .little);
            if (flag != 0) continue;
            for (0..k) |i| {
                std.mem.writeInt(u32, p[(r * k + i) * 4 ..][0..4], @bitCast(v[i]), .little);
                std.mem.writeInt(i64, p[lay.ids + (r * k + i) * 8 ..][0..8], id[i], .little);
            }
        }
    }
    return out;
}

/// The merge of a round against the baseline kernel on its whole rows: whether the merge took the round, and how many rows
/// the baseline kernel flags. A merge that takes a round the baseline would flag, or whose bytes differ from the baseline: an error.
const Got = struct { merged: bool, flagged: usize };

fn check(gpa: std.mem.Allocator, rows: []const []const f32, widths: [4]usize, c: usize) !Got {
    return checkAt(gpa, rows, widths, .{ 0, @intCast(widths[0]), @intCast(widths[0] + widths[1]), @intCast(widths[0] + widths[1] + widths[2]) }, c);
}

/// check with the parts at any offsets (part j: columns off[j] .. off[j] + widths[j] of the rows).
fn checkAt(gpa: std.mem.Allocator, rows: []const []const f32, widths: [4]usize, off: [4]u32, c: usize) !Got {
    const k = kFor(c);
    const parts = try partsModel(gpa, rows, widths, off, k);
    defer gpa.free(parts);
    const vals = try gpa.alloc(f32, rows.len * c);
    defer gpa.free(vals);
    const ids = try gpa.alloc(i64, rows.len * c);
    defer gpa.free(ids);
    @memset(vals, std.math.nan(f32));
    @memset(ids, -1);
    const merged = merge(parts, layout(rows.len, k), off, c, vals, ids);
    var flagged: usize = 0;
    var tv: [64]f32 = undefined;
    var ti: [64]i64 = undefined;
    for (rows, 0..) |row, r| {
        const flag = kernelModel(row, c, &tv, &ti);
        if (flag != 0) {
            flagged += 1;
            continue;
        }
        if (!merged) continue;
        for (0..c) |i| {
            const got: u32 = @bitCast(vals[r * c + i]);
            const want: u32 = @bitCast(tv[i]);
            if (got != want or ids[r * c + i] != ti[i]) {
                std.debug.print("row {d} candidate {d}: merged ({x:0>8}, {d}), the baseline ({x:0>8}, {d})\n", .{ r, i, got, ids[r * c + i], want, ti[i] });
                return error.TestUnexpectedResult;
            }
        }
    }
    if (merged and flagged > 0) {
        std.debug.print("the merge took a round the baseline kernel flags ({d} rows)\n", .{flagged});
        return error.TestUnexpectedResult;
    }
    return .{ .merged = merged, .flagged = flagged };
}

const prod: [4]usize = .{ 32384, 32256, 32384, 32256 }; // production's nodes in vocabulary order: g0, g2, g1, g3

fn randomRows(gpa: std.mem.Allocator, rnd: std.Random, n: usize, rows: usize, levels: ?u32) ![][]f32 {
    const out = try gpa.alloc([]f32, rows);
    for (out) |*row| {
        row.* = try gpa.alloc(f32, n);
        for (row.*) |*v| v.* = if (levels) |l| @as(f32, @floatFromInt(rnd.uintLessThan(u32, l))) * 0.25 - 2.0 else @floatCast(rnd.floatNorm(f64) * 3.0);
    }
    return out;
}

fn freeRows(gpa: std.mem.Allocator, rows: [][]f32) void {
    for (rows) |row| gpa.free(row);
    gpa.free(rows);
}

test "the parts land in the vocabulary's order: a one-row quarters exchange puts node g's part where its columns start" {
    const ring2d = @import("ring2d.zig");
    const hw: [2]usize = .{ 32384, 32256 };
    const off = vocabOffsets(hw);
    try std.testing.expectEqual([4]u32{ 0, 32384, 64640, 97024 }, off);
    const lay = layout(34, 32);
    const pw = lay.bytes / 4;
    const dst: u64 = 1 << 32;
    const d = try ring2d.quartersDsts(dst, 1, 0, .{ pw, pw }, 4);
    for (0..4) |g| {
        const r = g % 2; // node g's TP2 rank and pair (ring2d.owner)
        const p = g / 2;
        const part = (d[g].ptr - dst) / lay.bytes;
        try std.testing.expectEqual(@as(u64, 0), (d[g].ptr - dst) % lay.bytes);
        try std.testing.expectEqual(@as(u32, @intCast(lay.bytes / 16)), d[g].row4);
        // round.zig's head: rank r's half at r x (w0 + w1), its pair p's columns at p x w0 within it
        try std.testing.expectEqual(@as(u32, @intCast(r * (hw[0] + hw[1]) + p * hw[0])), off[part]);
    }
}

test "a node's part: three arrays at 16-byte boundaries, whole 16-byte words; k 32 up to a count of 32, else 64" {
    const l = layout(2, 32);
    try std.testing.expectEqual(@as(usize, 256), l.ids);
    try std.testing.expectEqual(@as(usize, 256 + 512), l.flags);
    try std.testing.expectEqual(@as(usize, 256 + 512 + 16), l.bytes);
    const odd = layout(3, 32);
    try std.testing.expectEqual(@as(usize, 384), odd.ids);
    try std.testing.expectEqual(@as(usize, 384 + 768), odd.flags);
    try std.testing.expectEqual(@as(usize, 384 + 768 + 16), odd.bytes);
    for ([_]usize{ 1, 2, 3, 5, 34, 47, 48 }) |r| for ([_]usize{ 32, 64 }) |k| {
        const x = layout(r, k);
        try std.testing.expect(x.ids % 16 == 0 and x.flags % 16 == 0 and x.bytes % 16 == 0);
        try std.testing.expect(x.ids >= r * k * 4 and x.flags - x.ids >= r * k * 8 and x.bytes - x.flags >= r * 4);
    };
    try std.testing.expectEqual(@as(usize, 37056), layout(48, 64).bytes); // the largest part a round sends
    try std.testing.expectEqual(@as(u8, 32), kFor(1));
    try std.testing.expectEqual(@as(u8, 32), kFor(28)); // the server's default top_k 20 + the margin
    try std.testing.expectEqual(@as(u8, 32), kFor(32));
    try std.testing.expectEqual(@as(u8, 64), kFor(33));
    try std.testing.expectEqual(@as(u8, 64), kFor(sampling.max_candidates));
}

test "the merged candidates are the baseline kernel's on the whole rows: production's split, random rows, every count" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0xb1_0001);
    const rnd = prng.random();
    const V = prod[0] + prod[1] + prod[2] + prod[3];
    for ([_]usize{ 1, 2, 5 }) |R| for ([_]usize{ 1, 9, 28, 33, 64 }) |c| {
        const rows = try randomRows(gpa, rnd, V, R, null);
        defer freeRows(gpa, rows);
        const got = try check(gpa, rows, prod, c);
        try std.testing.expect(got.merged); // continuous values: no ties at any cut
        try std.testing.expectEqual(@as(usize, 0), got.flagged);
    };
}

test "ties across the nodes at the cut, signed zeros and infinities: the baseline set, order and value bits, or a fallback where the baseline could flag" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0xb1_0002);
    const rnd = prng.random();
    const w: [4]usize = .{ 300, 260, 300, 260 };
    var merged: usize = 0;
    var fell: usize = 0;
    for (0..400) |trial| {
        const R = 1 + trial % 3;
        // few levels: ties everywhere, across nodes and at the cut
        const rows = try randomRows(gpa, rnd, w[0] + w[1] + w[2] + w[3], R, @intCast(4 + trial % 13));
        defer freeRows(gpa, rows);
        for (rows) |row| for (row) |*v| {
            if (v.* == 0 and rnd.boolean()) v.* = -0.0; // a -0 beside +0: one key, their own bits kept
            if (rnd.uintLessThan(u32, 97) == 0) v.* = std.math.inf(f32);
            if (rnd.uintLessThan(u32, 89) == 0) v.* = -std.math.inf(f32);
        };
        const counts = [_]usize{ 1, 7, 28, 32, 40, 64 };
        const c = counts[trial % counts.len];
        const got = try check(gpa, rows, w, c);
        if (got.merged) merged += 1 else fell += 1;
        // the parts in another order than the vocabulary's: ties across the nodes still go to the lower id
        const again = try checkAt(gpa, rows, .{ 300, 260, 300, 260 }, .{ 560, 860, 0, 300 }, c);
        try std.testing.expectEqual(got.merged, again.merged);
    }
    // both happen: a cut value held by two nodes, one of them at its k-th, falls back
    try std.testing.expect(merged > 50 and fell > 50);
}

test "more than 2,048 ties at the cut spread over the nodes: the baseline kernel flags the row, the merge falls back" {
    const gpa = std.testing.allocator;
    const V = prod[0] + prod[1] + prod[2] + prod[3];
    const row = try gpa.alloc(f32, V);
    defer gpa.free(row);
    // 600 values of 1.0 in each node (2,400 > 2,048, under 2,048 in every node), the rest below; the 28th value is 1.0
    var prng: std.Random.DefaultPrng = .init(0xb1_0003);
    const rnd = prng.random();
    for (row) |*v| v.* = @floatCast(rnd.float(f64) * 0.5 - 1.0);
    var at: usize = 0;
    for (prod) |n| {
        for (0..600) |i| row[at + i * (n / 600)] = 1.0;
        at += n;
    }
    const rows = [_][]const f32{row};
    const got = try check(gpa, &rows, prod, 28);
    try std.testing.expectEqual(@as(usize, 1), got.flagged);
    try std.testing.expect(!got.merged);
    // two nodes of 1,100 each (2,200: each under the buffer, together over it): a fallback as well
    for (row) |*v| if (v.* == 1.0) {
        v.* = -2.0;
    };
    for (0..1100) |i| {
        row[i * 29] = 1.0; // node g0
        row[prod[0] + i * 29] = 1.0; // node g2
    }
    const two = try check(gpa, &rows, prod, 28);
    try std.testing.expectEqual(@as(usize, 1), two.flagged);
    try std.testing.expect(!two.merged);
    // 2,000 ties in one node alone (under the kernel's buffer, nowhere else): the baseline does not flag, the merge takes it
    for (row) |*v| if (v.* == 1.0) {
        v.* = -2.0;
    };
    for (0..2000) |i| row[i * 16] = 1.0; // all inside node g0 (32,384 columns)
    const one = try check(gpa, &rows, prod, 28);
    try std.testing.expectEqual(@as(usize, 0), one.flagged);
    try std.testing.expect(one.merged);
    // 2,049 there: node g0's kernel and the baseline both flag
    for (0..49) |i| row[i * 16 + 8] = 1.0;
    const over = try check(gpa, &rows, prod, 28);
    try std.testing.expectEqual(@as(usize, 1), over.flagged);
    try std.testing.expect(!over.merged);
}

test "a NaN in any node's columns: the node flags it, the merge falls back (the baseline kernel flags the row)" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0xb1_0004);
    const rnd = prng.random();
    const w: [4]usize = .{ 300, 260, 300, 260 };
    for (0..4) |j| {
        const rows = try randomRows(gpa, rnd, 1120, 2, null);
        defer freeRows(gpa, rows);
        const starts = [_]usize{ 0, 300, 560, 860 };
        const start = starts[j];
        rows[1][start + 17] = std.math.nan(f32);
        const got = try check(gpa, rows, w, 28);
        try std.testing.expectEqual(@as(usize, 1), got.flagged);
        try std.testing.expect(!got.merged);
    }
}

// ---- TF_DS_HEAD_KEY's tests: the baseline argmax kernel on the whole row against the nodes' bests merged

/// tf_ds_argmax_rows on one row: the first index of the largest number; 0 when the first value is a NaN.
fn argmaxModel(row: []const f32) u32 {
    if (row[0] != row[0]) return 0;
    var bv = -std.math.inf(f32);
    var bi: u32 = 0x7fffffff;
    for (row, 0..) |v, i| {
        if (v == v and (v > bv or (v == bv and i < bi))) {
            bv = v;
            bi = @intCast(i);
        }
    }
    return bi;
}

/// The four nodes' parts of `rows` (each [n]) as tf_ds_argmax_part writes them.
fn keyPartsModel(gpa: std.mem.Allocator, rows: []const []const f32, widths: [4]usize, off: [4]u32) ![]u8 {
    const out = try gpa.alloc(u8, 4 * rows.len * key_row);
    @memset(out, 0x7f);
    for (0..4) |j| for (rows, 0..) |row, r| {
        const cols = row[off[j]..][0..widths[j]];
        var bv = -std.math.inf(f32);
        var bi: i32 = no_index;
        for (cols, 0..) |v, i| {
            if (v == v and (v > bv or (v == bv and i < bi))) {
                bv = v;
                bi = @intCast(i);
            }
        }
        const p = out[(j * rows.len + r) * key_row ..][0..key_row];
        std.mem.writeInt(u32, p[0..4], @bitCast(bv), .little);
        std.mem.writeInt(i32, p[4..8], bi, .little);
        std.mem.writeInt(i32, p[8..12], @intFromBool(cols[0] != cols[0]), .little);
        std.mem.writeInt(i32, p[12..16], 0, .little);
    };
    return out;
}

fn checkKeys(gpa: std.mem.Allocator, rows: []const []const f32, widths: [4]usize, off: [4]u32) !void {
    const parts = try keyPartsModel(gpa, rows, widths, off);
    defer gpa.free(parts);
    const tokens = try gpa.alloc(u32, rows.len);
    defer gpa.free(tokens);
    @memset(tokens, 0xdead);
    mergeKeys(parts, rows.len, off, tokens);
    for (rows, tokens, 0..) |row, t, r| {
        // the vocabulary's row: the parts' columns at their offsets
        const whole = try gpa.alloc(f32, widths[0] + widths[1] + widths[2] + widths[3]);
        defer gpa.free(whole);
        for (0..4) |j| @memcpy(whole[off[j]..][0..widths[j]], row[off[j]..][0..widths[j]]);
        const want = argmaxModel(whole);
        if (t != want) {
            std.debug.print("row {d}: merged token {d}, the baseline {d}\n", .{ r, t, want });
            return error.TestUnexpectedResult;
        }
    }
}

test "a greedy round's tokens from the nodes' bests are the baseline argmax on the whole rows: production's split, random rows" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0xb2_0001);
    const rnd = prng.random();
    const V = prod[0] + prod[1] + prod[2] + prod[3];
    const off = vocabOffsets(.{ prod[0], prod[1] });
    for ([_]usize{ 1, 2, 5, 17 }) |R| {
        const rows = try randomRows(gpa, rnd, V, R, null);
        defer freeRows(gpa, rows);
        try checkKeys(gpa, @ptrCast(rows), prod, off);
    }
}

test "the nodes' bests: equal largest values across the nodes go to the lower id; NaNs never larger; a NaN first value is token 0" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0xb2_0002);
    const rnd = prng.random();
    const w: [4]usize = .{ 300, 260, 300, 260 };
    const off: [4]u32 = .{ 0, 300, 560, 860 };
    for (0..300) |trial| {
        const rows = try randomRows(gpa, rnd, 1120, 3, @intCast(2 + trial % 9)); // few levels: the largest held by several nodes
        defer freeRows(gpa, rows);
        for (rows) |row| for (row) |*v| {
            if (rnd.uintLessThan(u32, 50) == 0) v.* = std.math.nan(f32);
            if (rnd.uintLessThan(u32, 200) == 0) v.* = std.math.inf(f32);
            if (rnd.uintLessThan(u32, 100) == 0) v.* = -std.math.inf(f32);
            if (v.* == 0 and rnd.boolean()) v.* = -0.0;
        };
        switch (trial % 6) {
            0 => rows[0][0] = std.math.nan(f32), // the vocabulary's first value: token 0
            1 => rows[1][off[1]] = std.math.nan(f32), // another node's first column: an ordinary NaN
            2 => @memset(rows[2], std.math.nan(f32)), // no number at all: token 0
            3 => @memset(rows[0], -std.math.inf(f32)), // every value -inf: the first index
            4 => for (rows[1][off[2]..][0..w[2]]) |*v| {
                v.* = std.math.nan(f32); // a node without a number
            },
            else => {},
        }
        try checkKeys(gpa, @ptrCast(rows), w, off);
        // the parts in another order than the vocabulary's: the lower id still wins a tie
        try checkKeys(gpa, @ptrCast(rows), .{ 300, 260, 300, 260 }, .{ 560, 860, 0, 300 });
    }
}
