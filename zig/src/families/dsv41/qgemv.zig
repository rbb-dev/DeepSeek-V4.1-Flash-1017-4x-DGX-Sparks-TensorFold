//! The quant-into-GEMV micro-benchmark's launches and verdict (tf-dsv41-quant-gemv-bench; no engine path runs them): a
//! 2D node's dense decode linears with the prototype tf_dsn_dec_fp8x_r1, the decode GEMV that quantizes its own bf16
//! input rows after its wait (native.cu dec_dense_qx), against the baseline tf_dsn_dec_quant then tf_dsn_dec_fp8_r1.
const std = @import("std");
const mx = @import("mx.zig");
const weights = @import("weights.zig");

/// native.cu's QxArgs: the bf16 input rows the prototype quantizes (row stride ldx elements; a block-diagonal launch's
/// group g at x + g * gk elements).
pub const QxArgs = extern struct { x: u64 = 0, ldx: i64 = 0, gk: i64 = 0 };

comptime {
    std.debug.assert(@sizeOf(QxArgs) == 24);
}

/// A 2D node's decode linears on one input (K inputs; the outputs of each, one launch), or wo_a's FP8 rows (`groups`
/// groups of N / groups outputs, group g reading inputs g K .. (g + 1) K - 1 of a row of groups x K), with the output
/// type the round writes.
pub const Shape = struct { name: []const u8, k: usize, ns: []const usize, groups: usize = 1, y: mx.DType };

pub const shapes = [_]Shape{
    .{ .name = "wq_a+wkv", .k = 5120, .ns = &.{ 1280, 512 }, .y = .bf16 },
    .{ .name = "wq_b", .k = 1280, .ns = &.{8192}, .y = .bf16 },
    .{ .name = "wo_a", .k = 4096, .ns = &.{2048}, .groups = 2, .y = .bf16 },
    .{ .name = "wo_b", .k = 4096, .ns = &.{2560}, .y = .f32 },
    // the drafter's FP8 head on a node's vocabulary quarter (129,280 / 4): the widest grid, the most CTAs quantizing
    .{ .name = "head", .k = 5120, .ns = &.{32320}, .y = .f32 },
};

/// A decode round's rows the cells take (one stream's windows).
pub const rows = [_]usize{ 1, 2, 6 };

/// The prototype's rows at most (its template's 8-row group).
pub const max_rows = 8;

/// A launch of a shape's linears on m rows: the decode tensor-core kernels' arguments (the split, the CTAs each
/// linear takes; one input), as mx's dense decode launch builds them. `codes`: the baseline GEMV reading quantized rows at
/// q, s (row strides ldq bytes, lds floats; a group's at g K, g K / 32 of them); else the prototype's (q, s unread).
pub const Launch = struct { mm: mx.DecMulti, ctas: usize };

pub fn launchOf(sh: Shape, ls: []const weights.Linear, m: usize, ys: []const u64, codes: ?struct { q: u64, s: u64 }) !Launch {
    if (ls.len != sh.ns.len or ys.len != ls.len or ls.len > 4 or m == 0 or m > max_rows) return error.UnsupportedShape;
    var mm: mx.DecMulti = .{ .n = @intCast(ls.len) };
    var ctas: usize = 0;
    const kk = sh.k * sh.groups; // a row's inputs
    for (ls, ys, sh.ns, 0..) |l, y, n, i| {
        if (l.fmt != .fp8 or l.k != sh.k or l.n != n or n % (16 * sh.groups) != 0 or sh.k % 128 != 0) return error.UnsupportedShape;
        const sp = mx.decSplit(0, n, sh.k);
        if (sp.cs != 1) return error.UnsupportedShape;
        mm.p[i] = .{
            .q = if (codes) |c| c.q else 0,
            .s = if (codes) |c| c.s else 0,
            .ldq = if (codes != null) @intCast(kk) else 0,
            .lds = if (codes != null) @intCast(kk / 32) else 0,
            .w = l.w,
            .ws = l.s,
            .ldw = @intCast(sh.k),
            .ldws = @intCast(sh.k / 32),
            .fmt = 0,
            .M = @intCast(m),
            .N = @intCast(n),
            .K = @intCast(sh.k),
            .ks = @intCast(sp.ks),
            .cs = 1,
            .y = y,
            .ldy = @intCast(n),
            .y_dtype = @backingInt(sh.y),
            .gn = if (sh.groups > 1) @intCast(n / sh.groups) else 0,
            .gq = if (sh.groups > 1) @intCast(sh.k) else 0,
            .gs = if (sh.groups > 1) @intCast(sh.k / 32) else 0,
        };
        mm.cta0[i] = @intCast(ctas);
        ctas += std.math.divCeil(usize, n / 16, mx.dec_warps / sp.ks) catch unreachable;
    }
    mm.cta0[ls.len] = @intCast(ctas);
    return .{ .mm = mm, .ctas = ctas };
}

/// tf_dsn_dec_fp8_r1's dynamic shared memory (mx's decShared(dec_dense_cfg, 1)): the 8 warps' rings.
pub fn ringShared() u32 {
    const c = mx.dec_dense_cfg;
    const warp: usize = @max(c.stages * (16 * c.rowb + 128), 4 * 32 * 4);
    return @intCast(mx.dec_warps * warp);
}

/// The prototype's: the rings, then the m rows' codes (K + 16 bytes a row) and scales (K / 32 floats a row).
pub fn protoShared(m: usize, k: usize) u32 {
    return ringShared() + @as(u32, @intCast(m * (k + 16) + m * (k / 32) * 4));
}

/// One arm's 20 blocks of a cell (us a chain): median, min, max.
pub const Stat = struct { med: f64, min: f64, max: f64 };

pub fn statOf(us: []f64) Stat {
    std.mem.sort(f64, us, {}, std.sort.asc(f64));
    const n = us.len;
    const med = if (n % 2 == 1) us[n / 2] else (us[n / 2 - 1] + us[n / 2]) / 2;
    return .{ .med = med, .min = us[0], .max = us[n - 1] };
}

/// The fusion's gate for a shape: build it where the prototype is shorter than the baseline chain by at
/// least 1 us at every row count, and each cell's spread (the larger of the two arms' min-max) is under half that gain.
pub fn builds(a: []const Stat, b: []const Stat) bool {
    if (a.len == 0 or a.len != b.len) return false;
    for (a, b) |x, y| {
        const gain = x.med - y.med;
        const spread = @max(x.max - x.min, y.max - y.min);
        if (gain < 1.0 or spread >= gain / 2) return false;
    }
    return true;
}

test "the prototype's launches take the baseline grid and split: a 2D node's CTAs at the round's rows" {
    // (a decode round's CTAs: wq_a + wkv 112, wq_b 512, wo_a 128, wo_b 160; the head's 2,020 tiles)
    const want = [_]usize{ 112, 512, 128, 160, 2020 };
    for (shapes, want) |sh, w| {
        var ls: [4]weights.Linear = undefined;
        var ys: [4]u64 = undefined;
        for (sh.ns, 0..) |n, i| {
            ls[i] = .{ .words = 0, .k = @intCast(sh.k), .n = @intCast(n), .fmt = .fp8, .w = 0x1000 * (i + 1), .s = 0x9000 * (i + 1) };
            ys[i] = 0x100000 * (i + 1);
        }
        for (rows) |m| {
            const p = try launchOf(sh, ls[0..sh.ns.len], m, ys[0..sh.ns.len], null);
            const t = try launchOf(sh, ls[0..sh.ns.len], m, ys[0..sh.ns.len], .{ .q = 0x5000, .s = 0x6000 });
            try std.testing.expectEqual(w, p.ctas);
            try std.testing.expectEqual(p.ctas, t.ctas);
            for (0..sh.ns.len) |i| {
                const x = p.mm.p[i];
                try std.testing.expectEqual(@as(c_int, @intCast(mx.decSplit(0, sh.ns[i], sh.k).ks)), x.ks);
                try std.testing.expectEqual(@as(c_int, @intCast(m)), x.M);
                // the prototype reads no codes; the baseline GEMV the quantized rows of the whole input row
                try std.testing.expectEqual(@as(u64, 0), x.q);
                try std.testing.expectEqual(@as(i64, @intCast(sh.k * sh.groups)), t.mm.p[i].ldq);
                try std.testing.expectEqual(@as(c_int, if (sh.groups > 1) @intCast(sh.ns[i] / sh.groups) else 0), x.gn);
            }
        }
    }
    // more rows than the prototype's template, or a shape it does not take, refuse
    var l: weights.Linear = .{ .words = 0, .k = 1280, .n = 8192, .fmt = .fp8, .w = 1, .s = 1 };
    try std.testing.expectError(error.UnsupportedShape, launchOf(shapes[1], &.{l}, 9, &.{0}, null));
    l.fmt = .fp4;
    try std.testing.expectError(error.UnsupportedShape, launchOf(shapes[1], &.{l}, 2, &.{0}, null));
}

test "the prototype's shared memory: the warps' rings, then the rows' codes and scales, under the opt-in limit" {
    try std.testing.expectEqual(@as(u32, 36864), ringShared());
    try std.testing.expectEqual(@as(u32, 36864 + 2 * 5136 + 2 * 160 * 4), protoShared(2, 5120));
    // every cell fits a GB10 CTA's opt-in shared memory (99 KiB)
    for (shapes) |sh| for (rows) |m| try std.testing.expect(protoShared(m, sh.k) <= 99 * 1024);
}

test "the fusion's gate: at least 1 us shorter at every row count, the spread under half the gain" {
    const s = struct {
        fn st(med: f64, spread: f64) Stat {
            return .{ .med = med, .min = med - spread / 2, .max = med + spread / 2 };
        }
    }.st;
    try std.testing.expect(builds(&.{ s(10, 0.2), s(11, 0.2), s(14, 0.4) }, &.{ s(8.5, 0.2), s(9.5, 0.2), s(12.5, 0.4) }));
    // a row count under 1 us, a spread at half the gain, a slower prototype: not built
    try std.testing.expect(!builds(&.{ s(10, 0.2), s(11, 0.2) }, &.{ s(8.5, 0.2), s(10.2, 0.2) }));
    try std.testing.expect(!builds(&.{s(10, 1.0)}, &.{s(8.0, 0.2)}));
    try std.testing.expect(!builds(&.{s(10, 0.2)}, &.{s(10.5, 0.2)}));
    var us = [_]f64{ 3, 1, 2, 5, 4 };
    const t = statOf(&us);
    try std.testing.expectEqual(Stat{ .med = 3, .min = 1, .max = 5 }, t);
}
