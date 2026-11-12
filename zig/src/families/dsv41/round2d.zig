//! The exact 2D-TP4 split on the port's decode rounds (round.zig; the served four-node lane's rounds.py with model.py
//! Comm2D, wo_2d with TF_DS_2D_WOB_FOLD, moe_2d's fused decode experts, esum and logits_full): a node computes its
//! columns of every split output as on the prompt path (prompt2d.zig), and every exchange only moves bytes, so after
//! each one the node holds its TP2 rank's bytes and the rest of the round is TP2's. round.zig takes it through thin
//! hooks behind Engine.two; a TP2 engine has none and runs as before.
//!
//! The exchanges of a round: the attention and MoE partials' and Engram's projection's quarters (TP2's [2, R, N] gather,
//! which the mixes and Engram's sum then take as on TP2), the head's vocabulary quarters (TP2's [2, R, half]), and two
//! cat_rank fills of a TP2 rank's whole input from the two pairs' parts: wo_b's rotated input rows (wo_a's epilogue
//! writes this node's half, wo_b's suh at that half) and the experts' intermediate (between gate / up and down). Here
//! they go over the lane's RDMA rings (prompt2d.Two's quarters and catRank), as round.zig's TP2
//! gathers do; the rings (ring2d.zig) come with the graphs.
const std = @import("std");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const weights = @import("weights.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");

const Two = prompt2d.Two;
const Engine = prompt.Engine;

/// wo_a's epilogue rotation for its group at u's column `col` (Exl3Group.rotated's rot, wo_2d's fold): wo_b's rotated
/// input rows of this node's half, into t.own [R, uw[p]] fp16 with wo_b's suh at that half (the rank's suh from this
/// pair's first column).
pub fn woRot(t: *const Two, lay: weights.Layer, col: usize) exl3_linear.RotOut {
    const at: usize = if (t.p == 0) 0 else t.uw[0];
    return .{ .suh = lay.wo_b.suh + at * 2, .xh = t.own, .ldr = @intCast(t.uw[t.p]), .off = @intCast(col) };
}

/// wo_2d's exchange after wo_a (cat_rank_into): the TP2 rank's whole rotated input rows xb [R, uw0 + uw1] fp16 from the
/// two pairs' halves (this node's in t.own), for wo_b's glinear on them.
pub fn woExchange(t: *const Two, e: *const Engine, xb: u64, rows: usize) !void {
    try t.catRank(e, t.own, xb, rows, t.uw, 2);
}

/// moe_2d's experts on a decode window, after the routing (experts2d.gateup_fused, cat_rank, down_fused): this node's
/// gate / up blocks with decode_prep, the rank's whole intermediate assembled from the two pairs' blocks, down for this
/// node's output columns with the combine: out fp32 [R, dw[p]].
pub fn experts(t: *const Two, e: *const Engine, ex: weights.Experts, sc: exl3_experts.DecodeScratch, x: u64, x_stride: usize, pick: u64, wts: u64, out: u64, rows: usize, limit: f32) !void {
    if (t.ir != 0) {
        // TF_DS_2D_GU=parity (experts2d gateup_rest_fused, pair_parts, assemble_rest): the main blocks and the rest
        // columns of the experts this pair computes them for (one launch with Engine.par), then the packs exchanged
        // with the column partner and the half assembled
        if (e.par) |pk| try exl3_experts2d.decodeGateUpOne(e.ex, pk, e.s, ex, sc, t.rest_dec.?, x, x_stride, pick, wts, out, rows, limit) else {
            try exl3_experts2d.decodeGateUp(e.ex, e.s, ex, sc, x, x_stride, pick, wts, out, rows, limit);
            try exl3_experts2d.decodeGateUpRest(e.ex, e.ops, e.s, ex, sc, t.rest_dec.?, pick, rows, limit);
        }
        try t.assembleParity(e, ex, sc.xd, pick, rows * sc.slots);
    } else {
        try exl3_experts2d.decodeGateUp(e.ex, e.s, ex, sc, x, x_stride, pick, wts, out, rows, limit);
        try t.catRank(e, sc.xd, t.xd_full, rows * sc.slots, t.gu, 2);
    }
    try exl3_experts2d.decodeDown(e.ex, e.s, ex, sc, t.xd_full, pick, wts, out, rows);
}

test "wo_b's rotated input: each pair's half of the rank's suh" {
    var t: Two = undefined;
    t.uw = .{ 2048, 2048 };
    t.own = 0x7000;
    var lay: weights.Layer = undefined;
    lay.wo_b.suh = 0x10000;
    t.p = 0;
    try std.testing.expectEqual(exl3_linear.RotOut{ .suh = 0x10000, .xh = 0x7000, .ldr = 2048, .off = 1024 }, woRot(&t, lay, 1024));
    t.p = 1;
    try std.testing.expectEqual(exl3_linear.RotOut{ .suh = 0x10000 + 2048 * 2, .xh = 0x7000, .ldr = 2048, .off = 0 }, woRot(&t, lay, 0));
}

test "TF_DS_DECODE_KSPLIT: wo_b's K-halves summed per TP2 rank equal the whole-K partial within the fp32 bound; the planes pair as r and 2 + r" {
    // a stand-in for wo_b on a small shape: K = 4 * 16 inputs (rank r's two pairs' halves are K/2 each), N = 6 outputs, the
    // activations and weights deterministic fp32 values of mixed sign and size
    const K = 64;
    const N = 6;
    var state: u32 = 0x9e3779b9;
    const Rng = struct {
        fn next(st: *u32) f32 {
            st.* = st.* *% 1664525 +% 1013904223;
            const m: f32 = @floatFromInt(@as(i32, @intCast(st.* >> 8)) - (1 << 23));
            return m / @as(f32, 1 << 23) * @as(f32, @floatFromInt(@as(u32, 1) << @intCast((st.* >> 3) % 6)));
        }
    };
    var x: [2][K]f32 = undefined; // TP2 rank r's wo_b input row
    var wt: [2][N][K]f32 = undefined;
    for (0..2) |r| {
        for (&x[r]) |*v| v.* = Rng.next(&state);
        for (&wt[r]) |*row| for (row) |*v| {
            v.* = Rng.next(&state);
        };
    }
    // baseline: a node's partial of rank r over the whole K (its N columns), the consumer adds rank 0 and rank 1
    // the K-split: node (pair p, rank r) over its half, the planes in node order 2 p + r, summed as plane r + plane 2 + r
    var planes: [4][N]f32 = undefined;
    for (0..2) |p| for (0..2) |r| for (0..N) |n| {
        var acc: f32 = 0;
        for (p * K / 2..(p + 1) * K / 2) |k| acc += x[r][k] * wt[r][n][k];
        planes[2 * p + r][n] = acc;
    };
    for (0..N) |n| {
        var whole: [2]f32 = undefined;
        var mag: [2]f32 = undefined; // sum |term|: the bound's scale
        for (0..2) |r| {
            var acc: f32 = 0;
            var m: f32 = 0;
            for (0..K) |k| {
                acc += x[r][k] * wt[r][n][k];
                m += @abs(x[r][k] * wt[r][n][k]);
            }
            whole[r] = acc;
            mag[r] = m;
        }
        for (0..2) |r| {
            const ks = planes[r][n] + planes[2 + r][n]; // the kernel's add: out[r] = in[r] + in[2 + r]
            // the standard bound of a K-term fp32 dot against any other order: 2 K u sum|term|, u = 2^-24
            const bound = 2.0 * @as(f32, K) * @as(f32, 1.0 / 16777216.0) * mag[r];
            try std.testing.expect(@abs(ks - whole[r]) <= bound);
        }
        // the consumer's rank-order sum of the two ranks' K-split sums stays within twice the bound of the baseline
        const got = (planes[0][n] + planes[2][n]) + (planes[1][n] + planes[3][n]);
        const want = whole[0] + whole[1];
        try std.testing.expect(@abs(got - want) <= 2.0 * 2.0 * @as(f32, K) * @as(f32, 1.0 / 16777216.0) * (mag[0] + mag[1]));
    }
    // control: the bound can fail: a sum that drops one half is off by far more than it
    const n0 = planes[0][0] + planes[0][0];
    var m0: f32 = 0;
    for (0..K) |k| m0 += @abs(x[0][k] * wt[0][0][k]);
    var w0: f32 = 0;
    for (0..K) |k| w0 += x[0][k] * wt[0][0][k];
    try std.testing.expect(@abs(n0 - w0) > 2.0 * @as(f32, K) * @as(f32, 1.0 / 16777216.0) * m0);
}
