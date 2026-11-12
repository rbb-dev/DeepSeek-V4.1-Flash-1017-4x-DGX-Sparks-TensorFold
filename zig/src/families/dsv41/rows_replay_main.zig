//! The owned-row paths against a whole chunk's, on the real kernels: each path's rows must be the bytes the 2,048-row
//! launches give them. On random bf16 activations and FP8 weights with UE8M0 scales:
//!   - front (TF_DS_FRONT_OWNED): prompt.frontRows at rows [k q, k q + q) for every k, q = 512, 256 and 1,024 (x's
//!     rows quantized once, wq_a and wkv on the codes, the q RMSNorm), against attention's whole-chunk front (mm's
//!     mx.linear of wq_a and of wkv, then the q RMSNorm): qa, y and qr byte-equal; a control with one x value set to
//!     64 (its 32-value group's scale another, so every code of the group) differs in its row.
//!   - routing (TF_DS_FFN_CODES): prompt.ffnRows at the same owned blocks (the router's fp32 logits, the text-bias
//!     routing, x's rows quantized) against moe's whole-chunk router and routing and group()'s act_quant: logits,
//!     picks, weights, codes and scales byte-equal; one x value set to 64 differs in the logits and the codes.
//!   - the pack (TF_DS_PACK_CODES): each pair's bf16 pack quantized (prompt2d.packQuant) and the codes and scales
//!     assembled (prompt2d.packAssemble) against the baseline assembly of the values (assembleRest) and down's quant of it,
//!     at a full chunk's slot rows and a partial chunk's: down's codes and scales byte-equal; one pack value set to 64
//!     differs.
//!   - the Engram gate (TF_DS_ENGRAM_OWNER_SUM): prompt.engramGateRows at every owned block of 512, 256 and 1,024
//!     rows against the whole chunk's launch (engramApply's), on random streams, kv and qk: the gated streams
//!     byte-equal; one stream value set to 64 differs.
//!   - wo_b's input as codes (TF_DS_CAT_CODES): two column partners' parts quantized (prompt2d.catCodesQuant), each
//!     one's codes and scales copied into the other's stage (the exchange), placed and wo_b's GEMM run
//!     (prompt2d.catCodesGemm), against mx.linear of the assembled bf16 input, at a full chunk's rows, a partial
//!     chunk's past the column ring's 1,088 and a small chunk's 40: wo_b's input codes, scales and output byte-equal
//!     on both nodes; one value of pair 1's part set to 64 differs.
//!   - the boundary's kv-source update (kvsrc, TF_DS_BOUNDARY_OWNED): prompt.kvSourceAt (the compressor's and the index
//!     keys' bf16 GEMMs, their RMSNorms (d 512 and 128), RoPE, both FP4 stores) at every owned block of 512, 256 and
//!     1,024 rows of a chunk from position 4,096, each block on its own 0xa5 caches as a rank holds them, and each
//!     block's span of the four cache arrays taken as prompt.boundaryGathers' in-place all-gather takes it (on the host),
//!     against the whole chunk's update: the four arrays byte-equal, nothing written outside a block's span; every array's
//!     groups written by the whole chunk's update (not the 0xa5 it starts from); one x value of row 700 set to 64 makes
//!     the four arrays differ. Its bf16 GEMMs take 16-row tiles, as with TF_DS_BF16_ROWS=1.
//! Then at each pass of R rows (--rows: 1,024, 2,048 and 4,096 unless named; a node's quarter q = R / 4):
//!   - quarter-HC's readers (hc): hc_pre_pf at the four nodes' quarters as attnMixes runs each node's (the streams,
//!     pre, x and the outputs at the quarter's rows of the pass's buffers, the quarter's rows of both gathered planes
//!     repacked into the tight planes from R rows apart, one tight part), against one launch of the whole pass, with the
//!     post of random planes and of a random earlier post and comb: h_out, x, pre_out, post and comb byte-equal; one
//!     value of plane 1's row 3 q + 5 set to 64 makes each of them differ.
//!   - the boundary after them (wp4): the next layer's hc_pre_pf on all R rows (the resync and replayCut's launch) on
//!     what the quarters left against on what the whole pass left: its five outputs byte-equal; with quarters 1-3's
//!     post and comb as they were before the layer (not resynced), each of them differs.
//!   - the indexer's scores (idx, TF_DS_IDX_SHARD): _index_score_rows at the four quarters' row offsets against one
//!     launch of the whole pass, over R keys, at RB 1 and 2, keyed (the int64 keys, the tile maxima) and plain (the fp32
//!     scores): byte-equal; one query value of row 3 q + 5 set to 64 makes each of them differ.
//! Exit 0 when every check holds, 1 otherwise, 2 for a usage or build problem.
//!   tf-dsv41-rows-replay AOT_DIR [--rows 1024,2048,4096]
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const mx = dsv41.mx;
const prompt = dsv41.prompt;
const tri_basic = dsv41.tri_basic;
const tri_hc = dsv41.tri_hc;
const tri_index = dsv41.tri_index;
const pass_replay = dsv41.pass_replay;
const weights = dsv41.weights;

const D: usize = 5120; // hidden
const QW: usize = 1280; // wq_a's output (q_lora)
const HD: usize = 512; // wkv's output (head_dim)
const CAP: usize = 2048; // a full chunk
const EPS: f32 = 1e-20; // rms_norm_eps
const QS = [_]usize{ 512, 256, 1024 };
const NE: usize = 384; // routed experts
const TOPK: usize = 6;
const SL: usize = TOPK + 1; // a row's slots: the routed picks, the shared expert last
const SCALE: f32 = 1.5; // routed_scaling_factor
const IG: usize = 512; // a pair's main block of a slot row (gate / up's pack)
const IR: usize = 128; // the rest block
const E: usize = NE + 1; // the experts' ids, the shared expert last
const UW: usize = 2048; // a pair's part of wo_b's input (its two groups of o_lora)
const OW: usize = 2560; // wo_b's output columns on a pair
const HC: usize = 4; // the hyper-connection streams
const HC_EPS: f32 = 1e-6; // hc_eps
const HC_ITERS: usize = 20; // hc_sinkhorn_iters
const IH: usize = 32; // the indexer's heads
const ID: usize = 128; // the indexer's head dim
const ROPE: usize = 64; // qk_rope_head_dim

/// CUDA's __float2bfloat16_rn of a non-NaN value: round to nearest, ties to even.
fn bf16Rne(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn upload(d: *const cuda.Driver, bytes: []const u8) !cuda.DeviceBuffer {
}

fn get(buf: u64, n: usize, d: *const cuda.Driver, a: std.mem.Allocator) ![]u8 {
}

fn clear(buf: cuda.DeviceBuffer, d: *const cuda.Driver, s: cuda.Stream) !void {
    try d.check(d.api.cuMemsetD8Async(buf.ptr, 0xa5, buf.len, s.handle), "cuMemsetD8Async");
}

/// bf16 values, normal * scale (bf16-rounded).
fn bf16s(n: usize, rng: std.Random, scale: f32, a: std.mem.Allocator) ![]u8 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = bf16Rne(rng.floatNorm(f32) * scale);
    return std.mem.sliceAsBytes(v);
}

/// fp32 values, normal * 0.5.
fn f32s(n: usize, rng: std.Random, a: std.mem.Allocator) ![]u8 {
    const v = try a.alloc(f32, n);
    for (v) |*x| x.* = rng.floatNorm(f32) * 0.5;
    return std.mem.sliceAsBytes(v);
}

/// A bf16 linear [n, k]: values normal * scale.
fn bf16Linear(d: *const cuda.Driver, n: usize, k: usize, rng: std.Random, scale: f32, a: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer)) !weights.Linear {
    const b = try upload(d, try bf16s(n * k, rng, scale, a));
    try keep.append(a, b);
    return .{ .words = 0, .k = @intCast(k), .n = @intCast(n), .fmt = .bf16, .w = b.ptr, .s = 0 };
}

/// A norm's bf16 weights [n]: 1 + normal * 0.25.
fn normW(d: *const cuda.Driver, n: usize, rng: std.Random, a: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer)) !u64 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = bf16Rne(1 + rng.floatNorm(f32) * 0.25);
    const b = try upload(d, std.mem.sliceAsBytes(v));
    try keep.append(a, b);
    return b.ptr;
}

/// An FP8 linear [n, k]: E4M3 codes (no NaN) and UE8M0 scales a 32 x 32 block (2^-9 .. 2^0).
fn fp8Linear(d: *const cuda.Driver, n: usize, k: usize, rng: std.Random, a: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer)) !weights.Linear {
    const w = try a.alloc(u8, n * k);
    for (w) |*x| {
        x.* = rng.int(u8);
        if (x.* & 0x7f == 0x7f) x.* -= 1;
    }
    const sc = try a.alloc(u8, (n / 32) * (k / 32));
    for (sc) |*x| x.* = 118 + rng.uintLessThan(u8, 10);
    const wb = try upload(d, w);
    const sb = try upload(d, sc);
    try keep.append(a, wb);
    try keep.append(a, sb);
    return .{ .words = 0, .k = @intCast(k), .n = @intCast(n), .fmt = .fp8, .w = wb.ptr, .s = sb.ptr };
}

fn differs(x: []const u8, y: []const u8) usize {
    var n: usize = 0;
    for (x, y) |p, q| n += @intFromBool(p != q);
    return n;
}

const Run = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    t: dsv41.tri.Tri,
    k: *const mx.Kernels,
    a: std.mem.Allocator,
    gpa: std.mem.Allocator, // under each pass case's own arena
    ok: bool = true,

    fn report(r: *Run, what: []const u8, n: usize, want_equal: bool) void {
        const pass = (n == 0) == want_equal;
        if (!pass) r.ok = false;
        std.debug.print("{s}: {d} bytes differ: {s}\n", .{ what, n, if (pass) "as it must" else "WRONG" });
    }
};

/// TF_DS_FRONT_OWNED's front: prompt.frontRows at every owned block of q rows against attention's whole-chunk front.
fn front(r: *Run, rng: std.Random, keep: *std.ArrayList(cuda.DeviceBuffer)) !void {
    const d = r.d;
    const a = r.a;
    var lay: weights.Layer = .{ .idx = 0, .ratio = 0 };
    lay.wq_a = try fp8Linear(d, QW, D, rng, a, keep);
    lay.wkv = try fp8Linear(d, HD, D, rng, a, keep);
    const nw = try a.alloc(u16, QW);
    for (nw) |*x| x.* = bf16Rne(1 + rng.floatNorm(f32) * 0.25);
    const norm = try upload(d, std.mem.sliceAsBytes(nw));
    try keep.append(a, norm);
    lay.q_norm = norm.ptr;
    const x_host = try bf16s(CAP * D, rng, 0.5, a);
    const x = try upload(d, x_host);
    try keep.append(a, x);
    const zf = @max(mx.zFloats(lay.wq_a, CAP), mx.zFloats(lay.wkv, CAP));
    const ls = mx.Scratch.sizes(CAP, D, zf);
    var sb: [4]cuda.DeviceBuffer = undefined;
    for (&sb, ls) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(a, b.*);
    }
    const sc: mx.Scratch = .{ .q = sb[0].ptr, .s = sb[1].ptr, .z = sb[2].ptr, .fake = sb[3].ptr, .rows = CAP, .max_k = D };
    var bufs: [6]cuda.DeviceBuffer = undefined; // qa, y, qr: the whole chunk's, then the owned path's
    for (&bufs, [_]usize{ QW, HD, QW, QW, HD, QW }) |*b, w| {
        b.* = try cuda.DeviceBuffer.alloc(d, CAP * w * 2);
        try keep.append(a, b.*);
    }
    // attention's whole-chunk front (mm -> mmRows -> mx.linear of each, then the q RMSNorm)
    try mx.linear(r.k, r.s, sc, lay.wq_a, x.ptr, .bf16, D, CAP, bufs[0].ptr, QW, .bf16, false, null);
    try mx.linear(r.k, r.s, sc, lay.wkv, x.ptr, .bf16, D, CAP, bufs[1].ptr, HD, .bf16, false, null);
    try tri_basic.rmsnorm(r.t, bufs[0].ptr, QW, lay.q_norm, bufs[2].ptr, QW, EPS, CAP, QW);
    try r.s.synchronize();
    var whole: [3][]u8 = undefined;
    for (&whole, bufs[0..3], [_]usize{ QW, HD, QW }) |*h, b, w| h.* = try get(b.ptr, CAP * w * 2, d, a);
    for (QS) |q| {
        for (bufs[3..]) |b| try clear(b, d, r.s);
        var k: usize = 0;
        while (k * q < CAP) : (k += 1) try prompt.frontRows(r.k, r.t, r.s, sc, lay, x.ptr, D, .{ .start = k * q, .n = q }, bufs[3].ptr, bufs[4].ptr, bufs[5].ptr, HD, EPS);
        try r.s.synchronize();
        for ([_][]const u8{ "qa", "y", "qr" }, 0..) |name, j| {
            var line: [96]u8 = undefined;
            const got = try get(bufs[3 + j].ptr, whole[j].len, d, a);
            r.report(try std.fmt.bufPrint(&line, "front q={d}: {s} at every owned block against the whole chunk's", .{ q, name }), differs(whole[j], got), true);
        }
    }
    // control: one x value of row 700 set to 64 (its group's scale another), its owned block of 512 rows again
    const off = try a.dupe(u8, x_host);
    const v = std.mem.bytesAsSlice(u16, off);
    v[700 * D + 77] = 0x4280;
    const x2 = try upload(d, off);
    try keep.append(a, x2);
    try prompt.frontRows(r.k, r.t, r.s, sc, lay, x2.ptr, D, .{ .start = 512, .n = 512 }, bufs[3].ptr, bufs[4].ptr, bufs[5].ptr, HD, EPS);
    try r.s.synchronize();
    const got = try get(bufs[5].ptr, whole[2].len, d, a);
    r.report("front control: qr with one x value of row 700 set to 64", differs(whole[2], got), false);
}

/// TF_DS_FFN_CODES's routing: prompt.ffnRows at every owned block of q rows against moe's whole-chunk router, routing
/// and group()'s quant of x.
fn ffn(r: *Run, rng: std.Random, keep: *std.ArrayList(cuda.DeviceBuffer)) !void {
    const d = r.d;
    const a = r.a;
    var lay: weights.Layer = .{ .idx = 0, .ratio = 0 };
    lay.experts.dims = D;
    const gw = try upload(d, try bf16s(NE * D, rng, 0.05, a));
    try keep.append(a, gw);
    lay.gate_w = gw.ptr;
    const gb_host = try a.alloc(f32, NE);
    for (gb_host) |*v| v.* = rng.floatNorm(f32) * 0.01;
    const gb = try upload(d, std.mem.sliceAsBytes(gb_host));
    try keep.append(a, gb);
    lay.gate_b = gb.ptr;
    const x_host = try bf16s(CAP * D, rng, 0.5, a);
    const x = try upload(d, x_host);
    try keep.append(a, x);
    // logits, pick, wts, q, s: the whole chunk's, then the owned path's
    const widths = [_]usize{ NE * 4, SL * 4, SL * 4, D, D / 32 * 4 };
    var bufs: [10]cuda.DeviceBuffer = undefined;
    for (&bufs, 0..) |*b, i| {
        b.* = try cuda.DeviceBuffer.alloc(d, CAP * widths[i % 5]);
        try keep.append(a, b.*);
    }
    const shared_id = NE;
    // moe's whole-chunk routing (mx.bf16Gemm, tri_norm.route) and group()'s quant of x
    try mx.bf16Gemm(r.k, r.s, x.ptr, D, lay.gate_w, D, CAP, NE, D, bufs[0].ptr, NE, .f32, 1, false, 1, 0, 0, 0, null);
    try dsv41.tri_norm.route(r.t, bufs[0].ptr, 0, lay.gate_b, TOPK, SCALE, shared_id, bufs[1].ptr, bufs[2].ptr, CAP, NE, SL);
    try mx.quant(r.k, r.s, x.ptr, .bf16, D, CAP, D, bufs[3].ptr, D, bufs[4].ptr, D / 32);
    try r.s.synchronize();
    var whole: [5][]u8 = undefined;
    for (&whole, 0..) |*h, j| h.* = try get(bufs[j].ptr, CAP * widths[j], d, a);
    var sc: mx.ExpertScratch = undefined;
    sc.rows = CAP;
    sc.slots = SL;
    sc.q = bufs[8].ptr;
    sc.s = bufs[9].ptr;
    for (QS) |q| {
        for (bufs[5..]) |b| try clear(b, d, r.s);
        var k: usize = 0;
        while (k * q < CAP) : (k += 1) try prompt.ffnRows(r.k, r.t, r.s, lay, x.ptr, D, .{ .start = k * q, .n = q }, NE, TOPK, SCALE, shared_id, SL, bufs[5].ptr, bufs[6].ptr, bufs[7].ptr, sc);
        try r.s.synchronize();
        for ([_][]const u8{ "logits", "pick", "wts", "x codes", "x scales" }, 0..) |name, j| {
            var line: [96]u8 = undefined;
            const got = try get(bufs[5 + j].ptr, whole[j].len, d, a);
            r.report(try std.fmt.bufPrint(&line, "routing q={d}: {s} at every owned block against the whole chunk's", .{ q, name }), differs(whole[j], got), true);
        }
    }
    // control: one x value of row 700 set to 64 (its group's scale another), its owned block of 512 rows again
    const off = try a.dupe(u8, x_host);
    std.mem.bytesAsSlice(u16, off)[700 * D + 1234] = 0x4280;
    const x2 = try upload(d, off);
    try keep.append(a, x2);
    try prompt.ffnRows(r.k, r.t, r.s, lay, x2.ptr, D, .{ .start = 512, .n = 512 }, NE, TOPK, SCALE, shared_id, SL, bufs[5].ptr, bufs[6].ptr, bufs[7].ptr, sc);
    try r.s.synchronize();
    const got = try get(bufs[5].ptr, whole[0].len, d, a);
    r.report("routing control: logits with one x value of row 700 set to 64", differs(whole[0], got), false);
    const codes = try get(bufs[8].ptr, whole[3].len, d, a);
    r.report("routing control: x codes with one x value of row 700 set to 64", differs(whole[3], codes), false);
}

/// TF_DS_PACK_CODES' pack: the codes path (packQuant of each pair's pack, packAssemble) against the baseline assembly of the
/// values and down's quant of it, at `rows` chunk rows.
fn pack(r: *Run, ops: *const dsv41.ops.Ops, rng: std.Random, keep: *std.ArrayList(cuda.DeviceBuffer), rows: usize) !void {
    const d = r.d;
    const a = r.a;
    const p = rows * SL;
    const i = 2 * IG + IR;
    const pick_host = try a.alloc(i32, p);
    for (pick_host) |*x| x.* = rng.intRangeLessThan(i32, 0, @intCast(E));
    const owner_host = try a.alloc(i32, E);
    for (owner_host) |*x| x.* = rng.intRangeLessThan(i32, 0, 2);
    const pick = try upload(d, std.mem.sliceAsBytes(pick_host));
    const owner = try upload(d, std.mem.sliceAsBytes(owner_host));
    var packs: [2]cuda.DeviceBuffer = undefined;
    var pack_host: [2][]u8 = undefined;
    for (&packs, &pack_host) |*b, *h| {
        h.* = try bf16s(p * (IG + IR), rng, 0.5, a);
        b.* = try upload(d, h.*);
    }
    const pc = try dsv41.prompt2d.packCodesPlan(p, IG, IR);
    var bufs: [7]cuda.DeviceBuffer = undefined; // xd_full, hq, hs (the baseline); codes 0, codes 1, hq, hs (the codes path)
    for (&bufs, [_]usize{ p * i * 2, p * i, p * i / 32 * 4, pc.bytes, pc.bytes, p * i, p * i / 32 * 4 }) |*b, n| b.* = try cuda.DeviceBuffer.alloc(d, n);
    for ([_]cuda.DeviceBuffer{ pick, owner, packs[0], packs[1] } ++ bufs) |b| try keep.append(a, b);
    // baseline: assembleParity's values, then down's quant of them
    try ops.assembleRest(r.s, packs[0].ptr, packs[1].ptr, pick.ptr, owner.ptr, bufs[0].ptr, p, IG, IR);
    try mx.quant(r.k, r.s, bufs[0].ptr, .bf16, i, p, i, bufs[1].ptr, i, bufs[2].ptr, i / 32);
    // the codes path
    var sc: mx.ExpertScratch = undefined;
    sc.hq = bufs[5].ptr;
    sc.hs = bufs[6].ptr;
    for (bufs[5..]) |b| try clear(b, d, r.s);
    try dsv41.prompt2d.packQuant(r.k, r.s, packs[0].ptr, bufs[3].ptr, pc);
    try dsv41.prompt2d.packQuant(r.k, r.s, packs[1].ptr, bufs[4].ptr, pc);
    try dsv41.prompt2d.packAssemble(ops, r.s, bufs[3].ptr, bufs[4].ptr, pick.ptr, owner.ptr, sc, pc);
    try r.s.synchronize();
    const hq = try get(bufs[1].ptr, p * i, d, a);
    const hs = try get(bufs[2].ptr, p * i / 32 * 4, d, a);
    var line: [96]u8 = undefined;
    r.report(try std.fmt.bufPrint(&line, "pack rows={d}: down's codes from the packs' codes against down's quant", .{rows}), differs(hq, try get(bufs[5].ptr, hq.len, d, a)), true);
    r.report(try std.fmt.bufPrint(&line, "pack rows={d}: down's scales from the packs' codes against down's quant", .{rows}), differs(hs, try get(bufs[6].ptr, hs.len, d, a)), true);
    // control: one main value of pair 1's slot row 5 set to 64 (its group's scale another)
    const off = try a.dupe(u8, pack_host[1]);
    std.mem.bytesAsSlice(u16, off)[5 * IG + 3] = 0x4280;
    try dsv41.prompt2d.packQuant(r.k, r.s, packs[1].ptr, bufs[4].ptr, pc);
    try dsv41.prompt2d.packAssemble(ops, r.s, bufs[3].ptr, bufs[4].ptr, pick.ptr, owner.ptr, sc, pc);
    try r.s.synchronize();
    const n_off = differs(hq, try get(bufs[5].ptr, hq.len, d, a)) + differs(hs, try get(bufs[6].ptr, hs.len, d, a));
    r.report(try std.fmt.bufPrint(&line, "pack rows={d} control: one pack value set to 64", .{rows}), n_off, false);
}

/// TF_DS_ENGRAM_OWNER_SUM's gate: prompt.engramGateRows at every owned block of q rows against the whole chunk's.
fn gate(r: *Run, rng: std.Random, keep: *std.ArrayList(cuda.DeviceBuffer)) !void {
    const d = r.d;
    const a = r.a;
    const hc: usize = 4;
    const h_row = hc * D * 2;
    const kv_row = (hc + 1) * D * 2;
    const h_host = try bf16s(CAP * hc * D, rng, 0.5, a);
    const h = try upload(d, h_host);
    const kv = try upload(d, try bf16s(CAP * (hc + 1) * D, rng, 0.5, a));
    const qk_host = try a.alloc(f32, hc * D);
    for (qk_host) |*v| v.* = rng.floatNorm(f32) * 0.05;
    const qk = try upload(d, std.mem.sliceAsBytes(qk_host));
    var outs: [2]cuda.DeviceBuffer = undefined; // the whole chunk's, the owned blocks'
    for (&outs) |*b| b.* = try cuda.DeviceBuffer.alloc(d, CAP * h_row);
    for ([_]cuda.DeviceBuffer{ h, kv, qk } ++ outs) |b| try keep.append(a, b);
    try prompt.engramGateRows(r.t, qk.ptr, h.ptr, kv.ptr, outs[0].ptr, .{ .start = 0, .n = CAP }, EPS, D, h_row, kv_row);
    try r.s.synchronize();
    const whole = try get(outs[0].ptr, CAP * h_row, d, a);
    for (QS) |q| {
        try clear(outs[1], d, r.s);
        var k: usize = 0;
        while (k * q < CAP) : (k += 1) try prompt.engramGateRows(r.t, qk.ptr, h.ptr, kv.ptr, outs[1].ptr, .{ .start = k * q, .n = q }, EPS, D, h_row, kv_row);
        try r.s.synchronize();
        var line: [96]u8 = undefined;
        r.report(try std.fmt.bufPrint(&line, "gate q={d}: the gated streams at every owned block against the whole chunk's", .{q}), differs(whole, try get(outs[1].ptr, whole.len, d, a)), true);
    }
    // control: one stream value of row 700 set to 64, its owned block of 512 rows again
    const off = try a.dupe(u8, h_host);
    std.mem.bytesAsSlice(u16, off)[700 * hc * D + 4321] = 0x4280;
    try prompt.engramGateRows(r.t, qk.ptr, h.ptr, kv.ptr, outs[1].ptr, .{ .start = 512, .n = 512 }, EPS, D, h_row, kv_row);
    try r.s.synchronize();
    r.report("gate control: one stream value of row 700 set to 64", differs(whole, try get(outs[1].ptr, whole.len, d, a)), false);
}

/// woBCodes' steps for two column partners on one device: each one's quant, the exchange (each send span into the
/// other's receive span), each one's copies and GEMM into its `pa`.
fn catRun(r: *Run, ops: *const dsv41.ops.Ops, wo_b: weights.Linear, plans: [2]dsv41.prompt2d.CatCodes, rows: usize, pa: [2]u64, z: u64) !void {
    for (plans) |pl| try dsv41.prompt2d.catCodesQuant(r.k, r.s, pl);
    for (plans, 0..) |pl, i| {
        const to = plans[1 - i].recv;
        if (to.len != pl.send.len) return error.UnsupportedShape;
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(to.addr, pl.send.addr, pl.send.len, r.s.handle), "cuMemcpyDtoDAsync");
    }
    for (plans, pa) |pl, y| try dsv41.prompt2d.catCodesGemm(r.k, ops, r.s, wo_b, pl, rows, y, z);
}

/// TF_DS_CAT_CODES' wo_b: the column partners' codes path (nodes 0 and 2, TP2 rank 0 of each pair) against mx.linear
/// of the rank's assembled input.
fn cat(r: *Run, ops: *const dsv41.ops.Ops, rng: std.Random, keep: *std.ArrayList(cuda.DeviceBuffer), rows: usize) !void {
    const d = r.d;
    const a = r.a;
    const kk = 2 * UW;
    const wo_b = try fp8Linear(d, OW, kk, rng, a, keep);
    var part_host: [2][]u8 = undefined;
    var parts: [2]cuda.DeviceBuffer = undefined;
    for (&part_host, &parts) |*h, *b| {
        h.* = try bf16s(rows * UW, rng, 0.5, a);
        b.* = try upload(d, h.*);
        try keep.append(a, b.*);
    }
    // the rank's whole input: row i is pair 0's part, then pair 1's (catRank's bytes)
    const full_host = try a.alloc(u8, rows * kk * 2);
    for (0..rows) |i| for (0..2) |pg| @memcpy(full_host[(i * kk + pg * UW) * 2 ..][0 .. UW * 2], part_host[pg][i * UW * 2 ..][0 .. UW * 2]);
    const full = try upload(d, full_host);
    try keep.append(a, full);
    // baseline: mx.linear (its quant into the scratch, then its GEMM)
    var bufs: [4]cuda.DeviceBuffer = undefined; // the scratch's q, s and z; the baseline pa
    for (&bufs, [_]usize{ CAP * kk, CAP * (kk / 32) * 4, @max(mx.zFloats(wo_b, CAP), 1) * 4, rows * OW * 4 }) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(a, b.*);
    }
    const sc: mx.Scratch = .{ .q = bufs[0].ptr, .s = bufs[1].ptr, .z = bufs[2].ptr, .fake = 0, .rows = CAP, .max_k = kk };
    try mx.linear(r.k, r.s, sc, wo_b, full.ptr, .bf16, kk, rows, bufs[3].ptr, OW, .f32, false, null);
    // the codes path: each node's codes, stage, wob_q, wob_s and pa (Chunk.init's sizes)
    var node: [2][5]cuda.DeviceBuffer = undefined;
    var plans: [2]dsv41.prompt2d.CatCodes = undefined;
    for (&node, &plans, 0..) |*nb, *pl, pg| {
        for (nb, [_]usize{ dsv41.prompt2d.catCodesBytes(CAP, UW), CAP * UW * 2, CAP * kk, CAP * (kk / 32) * 4, rows * OW * 4 }) |*b, n| {
            b.* = try cuda.DeviceBuffer.alloc(d, n);
            try keep.append(a, b.*);
            try clear(b.*, d, r.s);
        }
        pl.* = dsv41.prompt2d.catCodesPlan(@intCast(2 * pg), .{ 0, 1, 2, 3 }, rows, .{ UW, UW }, parts[pg].ptr, nb[0].ptr, nb[1].ptr, nb[2].ptr, nb[3].ptr);
    }
    try catRun(r, ops, wo_b, plans, rows, .{ node[0][4].ptr, node[1][4].ptr }, sc.z);
    try r.s.synchronize();
    const hq = try get(bufs[0].ptr, rows * kk, d, a);
    const hs = try get(bufs[1].ptr, rows * (kk / 32) * 4, d, a);
    const hp = try get(bufs[3].ptr, rows * OW * 4, d, a);
    var line: [128]u8 = undefined;
    for (node, 0..) |nb, pg| {
        r.report(try std.fmt.bufPrint(&line, "cat rows={d} node {d}: wo_b's input codes from the parts' codes against mx.linear's quant", .{ rows, 2 * pg }), differs(hq, try get(nb[2].ptr, hq.len, d, a)), true);
        r.report(try std.fmt.bufPrint(&line, "cat rows={d} node {d}: wo_b's input scales from the parts' codes against mx.linear's quant", .{ rows, 2 * pg }), differs(hs, try get(nb[3].ptr, hs.len, d, a)), true);
        r.report(try std.fmt.bufPrint(&line, "cat rows={d} node {d}: wo_b's output on them against mx.linear's", .{ rows, 2 * pg }), differs(hp, try get(nb[4].ptr, hp.len, d, a)), true);
    }
    // control: one value of pair 1's part, row 5, set to 64 (its group's scale another), the codes path again
    const off = try a.dupe(u8, part_host[1]);
    std.mem.bytesAsSlice(u16, off)[5 * UW + 3] = 0x4280;
    try catRun(r, ops, wo_b, plans, rows, .{ node[0][4].ptr, node[1][4].ptr }, sc.z);
    try r.s.synchronize();
    const n_off = differs(hq, try get(node[0][2].ptr, hq.len, d, a)) + differs(hp, try get(node[0][4].ptr, hp.len, d, a));
    r.report(try std.fmt.bufPrint(&line, "cat rows={d} control: one value of pair 1's part set to 64", .{rows}), n_off, false);
}

/// The boundary layer's kv-source update on one GPU: an Engine, a Chunk, Caches and Shared with what prompt.kvSourceAt
/// takes (layer 0 of the Weights: ratio 1, the compressor and the index keys), the caches' four arrays.
const KvSrc = struct {
    e: prompt.Engine,
    ch: prompt.Chunk,
    cs: prompt.Caches,
    sh: prompt.Shared,
    arrays: [4]cuda.DeviceBuffer, // the latents' codes and scales, the index keys' codes and scales
    bufs: [5]cuda.DeviceBuffer, // the chunk's lat2, lat, ik, groups and gpos

    /// kvSourceAt of `n` rows of x from row `row` (position start + row) on the four arrays; they and the chunk's
    /// latents and index keys 0xa5 first.
    fn update(kv: *KvSrc, r: *Run, x: u64, start: usize, row: usize, n: usize) !void {
        for (kv.arrays ++ kv.bufs[0..3].*) |b| try clear(b, r.d, r.s);
        try prompt.kvSourceAt(&kv.e, &kv.ch, &kv.cs, &kv.sh, 0, x + row * D * 2, start + row, n);
    }

    /// The whole chunk's update: the four arrays after it.
    fn whole(kv: *KvSrc, r: *Run, x: u64, start: usize, a: std.mem.Allocator) ![4][]u8 {
        try kv.update(r, x, start, 0, CAP);
        try r.s.synchronize();
        var got: [4][]u8 = undefined;
        for (&got, kv.arrays) |*g, b| g.* = try get(b.ptr, b.len, r.d, a);
        return got;
    }

    /// Each block of q rows updated on its own 0xa5 arrays, as rank k holds them (rows k q ..), and its span of every
    /// array taken as the boundary's in-place gathers take it: the arrays the gathers leave, and the bytes the blocks
    /// wrote outside their spans.
    fn owned(kv: *KvSrc, r: *Run, x: u64, start: usize, q: usize, a: std.mem.Allocator) !struct { arrays: [4][]u8, stray: usize } {
        var got: [4][]u8 = undefined;
        for (&got, kv.arrays) |*g, b| {
            g.* = try a.alloc(u8, b.len);
            @memset(g.*, 0xa5);
        }
        var stray: usize = 0;
        var k: usize = 0;
        while (k * q < CAP) : (k += 1) {
            try kv.update(r, x, start, k * q, q);
            try r.s.synchronize();
            const parts = prompt.boundaryGathers(&kv.cs, 0, start, .{ .start = k * q, .n = q }, HD, ID);
            for (got, kv.arrays, parts) |g, b, pt| stray += pass_replay.takeSpan(g, try get(b.ptr, b.len, r.d, a), pt, b.ptr, k, 0xa5);
        }
        return .{ .arrays = got, .stray = stray };
    }
};

/// TF_DS_BOUNDARY_OWNED's kv-source update at owned blocks of q rows (KvSrc.owned) against the whole chunk's, then with
/// one x value of row 700 set to 64.
fn kvsrc(r: *Run, ops: *const dsv41.ops.Ops, ex: *const dsv41.exact.Exact, rng: std.Random, keep: *std.ArrayList(cuda.DeviceBuffer)) !void {
    const d = r.d;
    const a = r.a;
    const start: usize = 2 * CAP; // the chunk's first position
    const groups = start + CAP; // the caches' groups: one a position at ratio 1
    const half = ROPE / 2;
    var lay: weights.Layer = .{ .idx = 0, .ratio = 1 };
    lay.comp_wkv = try bf16Linear(d, HD, D, rng, 0.02, a, keep);
    lay.comp_norm = try normW(d, HD, rng, a, keep);
    lay.idx_wk = try bf16Linear(d, ID, HD, rng, 0.05, a, keep);
    lay.idx_k_norm = try normW(d, ID, rng, a, keep);
    const layers = try a.alloc(weights.Layer, 1);
    layers[0] = lay;
    const w: weights.Weights = .{ .gpa = a, .layers = layers };
    var cfg: dsv41.Config = undefined;
    cfg.hidden = D;
    cfg.head_dim = HD;
    cfg.rope_dim = ROPE;
    cfg.index_head_dim = ID;
    cfg.eps = EPS;
    // the compressed positions' rotations: fp32 [groups, half] cos, then sin
    const rot = try a.alloc(f32, 2 * groups * half);
    for (0..groups) |pos| for (0..half) |j| {
        const ang = @as(f64, @floatFromInt(pos)) * std.math.pow(f64, 10000, -@as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(half)));
        rot[pos * half + j] = @floatCast(@cos(ang));
        rot[(groups + pos) * half + j] = @floatCast(@sin(ang));
    };
    const rope = try upload(d, std.mem.sliceAsBytes(rot));
    try keep.append(a, rope);
    var kv: KvSrc = .{ .e = undefined, .ch = undefined, .cs = .{}, .sh = .{}, .arrays = undefined, .bufs = undefined };
    kv.e.d = d;
    kv.e.s = r.s;
    kv.e.t = r.t;
    kv.e.ops = ops;
    kv.e.exact = ex;
    kv.e.c = &cfg;
    kv.e.w = &w;
    kv.e.mx = r.k;
    kv.e.two = null;
    kv.e.compressed = .{ .cos = rope.ptr, .sin = rope.ptr + groups * half * 4 };
    for (&kv.bufs, [_]usize{ CAP * HD * 2, CAP * HD * 2, CAP * ID * 2, CAP * 8, CAP * 8 }) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(a, b.*);
    }
    kv.ch.lat2 = kv.bufs[0].ptr;
    kv.ch.lat = kv.bufs[1].ptr;
    kv.ch.ik = kv.bufs[2].ptr;
    kv.ch.groups = kv.bufs[3].ptr;
    kv.ch.gpos = kv.bufs[4].ptr;
    kv.ch.mxs = .{ .q = 0, .s = 0, .z = 0, .fake = 0, .rows = 0, .max_k = 0 };
    const rows = [4]usize{ HD / 2, HD / 16, ID / 2, ID / 32 }; // a group's bytes in each array
    for (&kv.arrays, rows) |*b, row| {
        b.* = try cuda.DeviceBuffer.alloc(d, groups * row);
        try keep.append(a, b.*);
    }
    kv.cs.comp_codes[0] = kv.arrays[0].ptr;
    kv.cs.comp_scales[0] = kv.arrays[1].ptr;
    kv.cs.idx_codes[0] = kv.arrays[2].ptr;
    kv.cs.idx_scales[0] = kv.arrays[3].ptr;
    const x_host = try bf16s(CAP * D, rng, 0.5, a);
    const x = try upload(d, x_host);
    try keep.append(a, x);
    const names = [4][]const u8{ "the latents' codes", "the latents' scales", "the index keys' codes", "the index keys' scales" };
    const whole = try kv.whole(r, x.ptr, start, a);
    var line: [160]u8 = undefined;
    // every array's groups hold what the whole chunk's update wrote there, not the 0xa5 the arrays start from
    for (names, whole, rows) |name, y, row| {
        var written: usize = 0;
        for (y[start * row ..][0 .. CAP * row]) |v| written += @intFromBool(v != 0xa5);
        r.report(try std.fmt.bufPrint(&line, "kvsrc: {s} at the whole chunk's groups against 0xa5", .{name}), written, false);
    }
    for (QS) |q| {
        const got = try kv.owned(r, x.ptr, start, q, a);
        for (names, whole, got.arrays) |name, y, z| r.report(try std.fmt.bufPrint(&line, "kvsrc q={d}: {s} the gathers take from every owned block against the whole chunk's", .{ q, name }), differs(y, z), true);
        r.report(try std.fmt.bufPrint(&line, "kvsrc q={d}: the four arrays' bytes the blocks wrote outside their gathered spans", .{q}), got.stray, true);
    }
    // control: one x value of row 700 set to 64, the blocks of 512 again
    const off = try a.dupe(u8, x_host);
    std.mem.bytesAsSlice(u16, off)[700 * D + 77] = 0x4280;
    const x2 = try upload(d, off);
    try keep.append(a, x2);
    const got = try kv.owned(r, x2.ptr, start, 512, a);
    var n_off: usize = 0;
    for (whole, got.arrays) |y, z| n_off += differs(y, z);
    r.report("kvsrc control: the four arrays with one x value of row 700 set to 64", n_off, false);
}

/// One layer's hc_pre_pf constants: its mixes' weights, scales and bases, and its norm.
const HcLayer = struct {
    fn_w: cuda.DeviceBuffer, // fp32 [24, HC D]
    scale: cuda.DeviceBuffer, // fp32 [3]
    base: cuda.DeviceBuffer, // fp32 [24]
    norm_w: cuda.DeviceBuffer, // bf16 [D]

    fn init(r: *Run, rng: std.Random, keep: *std.ArrayList(cuda.DeviceBuffer)) !HcLayer {
        const l: HcLayer = .{
            .fn_w = try upload(r.d, try f32s(24 * HC * D, rng, r.a)),
            .scale = try upload(r.d, try f32s(3, rng, r.a)),
            .base = try upload(r.d, try f32s(24, rng, r.a)),
            .norm_w = try upload(r.d, try bf16s(D, rng, 0.5, r.a)),
        };
        for ([_]cuda.DeviceBuffer{ l.fn_w, l.scale, l.base, l.norm_w }) |b| try keep.append(r.a, b);
        return l;
    }
};

/// The names of hc_pre_pf's outputs, in Hc.outputs' order.
const hc_outputs = [_][]const u8{ "h_out", "x", "pre_out", "post", "comb" };

/// hc_pre_pf's buffers for passes of up to `cap` rows: the streams, pre_in, post and comb (the earlier mixes' going in,
/// put back before every launch, which reads and then overwrites them) and the gathered planes; h_out, x and pre_out;
/// the whole pass's part, and a quarter's tight part and planes (ch.qhc_part, ch.qhc_g).
const Hc = struct {
    h: cuda.DeviceBuffer, // bf16 [cap, HC, D]
    pre_in: cuda.DeviceBuffer, // fp32 [cap, HC]
    post: cuda.DeviceBuffer, // fp32 [cap, HC]
    comb: cuda.DeviceBuffer, // fp32 [cap, HC, HC]
    post_in: []const u8,
    comb_in: []const u8,
    g: cuda.DeviceBuffer, // fp32 [2, R, D] at a pass of R rows
    h_out: cuda.DeviceBuffer, // bf16 [cap, HC, D]
    x: cuda.DeviceBuffer, // bf16 [cap, D]
    pre_out: cuda.DeviceBuffer, // fp32 [cap, HC]
    part: cuda.DeviceBuffer, // fp32 [cap, hc_blocks, 32]
    part_q: cuda.DeviceBuffer, // fp32 [cap / 4, hc_blocks, 32]
    gq: cuda.DeviceBuffer, // fp32 [2, cap / 4, D]

    fn init(r: *Run, rng: std.Random, cap: usize, keep: *std.ArrayList(cuda.DeviceBuffer)) !Hc {
        const d = r.d;
        const a = r.a;
        var hc: Hc = undefined;
        hc.h = try upload(d, try bf16s(cap * HC * D, rng, 0.5, a));
        hc.pre_in = try upload(d, try f32s(cap * HC, rng, a));
        hc.post_in = try f32s(cap * HC, rng, a);
        hc.comb_in = try f32s(cap * HC * HC, rng, a);
        hc.post = try upload(d, hc.post_in);
        hc.comb = try upload(d, hc.comb_in);
        hc.g = try upload(d, try f32s(2 * cap * D, rng, a));
        hc.h_out = try cuda.DeviceBuffer.alloc(d, cap * HC * D * 2);
        hc.x = try cuda.DeviceBuffer.alloc(d, cap * D * 2);
        hc.pre_out = try cuda.DeviceBuffer.alloc(d, cap * HC * 4);
        hc.part = try cuda.DeviceBuffer.alloc(d, cap * tri_basic.hc_blocks * 32 * 4);
        hc.part_q = try cuda.DeviceBuffer.alloc(d, cap / 4 * tri_basic.hc_blocks * 32 * 4);
        hc.gq = try cuda.DeviceBuffer.alloc(d, 2 * (cap / 4) * D * 4);
        for ([_]cuda.DeviceBuffer{ hc.h, hc.pre_in, hc.post, hc.comb, hc.g, hc.h_out, hc.x, hc.pre_out, hc.part, hc.part_q, hc.gq }) |b| try keep.append(a, b);
        return hc;
    }

    /// The earlier mixes' post and comb back at a `rows`-row pass's rows, the outputs 0xa5.
    fn reset(hc: *const Hc, r: *Run, rows: usize) !void {
        try r.s.synchronize();
        for ([_]cuda.DeviceBuffer{ hc.h_out, hc.x, hc.pre_out }) |b| try clear(b, r.d, r.s);
    }

    /// hc_pre_pf of a whole `rows`-row pass, with the post of the planes `rows` rows apart (attnMixes' one launch).
    fn whole(hc: *const Hc, r: *Run, l: HcLayer, rows: usize) !void {
        try hc.reset(r, rows);
        _ = try tri_hc.hcPrePf(r.t, hc.h.ptr, l.fn_w.ptr, l.scale.ptr, l.base.ptr, hc.pre_in.ptr, l.norm_w.ptr, EPS, HC_EPS, HC_ITERS, hc.x.ptr, hc.pre_out.ptr, hc.post.ptr, hc.comb.ptr, hc.part.ptr, .{ .gathered = hc.g.ptr, .world = 2, .h_out = hc.h_out.ptr }, rows, D);
    }

    /// hc_pre_pf at the four nodes' quarters of a `rows`-row pass, as attnMixes' owned branch runs each node's: the
    /// streams, pre, x, pre_out, post, comb and h_out at the quarter's rows of the pass's buffers; the quarter's rows of
    /// both planes repacked into the tight planes first (qhcRepackG, the planes `rows` rows apart); one tight part.
    fn quarters(hc: *const Hc, r: *Run, l: HcLayer, rows: usize) !void {
        try hc.reset(r, rows);
        for (0..4) |k| {
            const q = pass_replay.quarter(rows, k);
            for (0..2) |w| try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(hc.gq.ptr + w * q.n * D * 4, hc.g.ptr + pass_replay.planeAt(w, rows, q.start, D), q.n * D * 4, r.s.handle), "cuMemcpyDtoDAsync");
            _ = try tri_hc.hcPrePf(r.t, hc.h.ptr + q.start * HC * D * 2, l.fn_w.ptr, l.scale.ptr, l.base.ptr, hc.pre_in.ptr + q.start * HC * 4, l.norm_w.ptr, EPS, HC_EPS, HC_ITERS, hc.x.ptr + q.start * D * 2, hc.pre_out.ptr + q.start * HC * 4, hc.post.ptr + q.start * HC * 4, hc.comb.ptr + q.start * HC * HC * 4, hc.part_q.ptr, .{ .gathered = hc.gq.ptr, .world = 2, .h_out = hc.h_out.ptr + q.start * HC * D * 2 }, q.n, D);
        }
    }

    /// A `rows`-row pass's outputs: h_out, x, pre_out, post and comb.
    fn outputs(hc: *const Hc, r: *Run, rows: usize, a: std.mem.Allocator) ![5][]u8 {
        try r.s.synchronize();
        return .{
            try get(hc.h_out.ptr, rows * HC * D * 2, r.d, a),
            try get(hc.x.ptr, rows * D * 2, r.d, a),
            try get(hc.pre_out.ptr, rows * HC * 4, r.d, a),
            try get(hc.post.ptr, rows * HC * 4, r.d, a),
            try get(hc.comb.ptr, rows * HC * HC * 4, r.d, a),
        };
    }
};

/// Quarter-HC's readers at a `rows`-row pass: hc_pre_pf at the four nodes' quarters (Hc.quarters) against the whole
/// pass's launch; then with one value of plane 1's row controlRow set to 64, which reaches the quarters through the repack.
fn hcCase(r: *Run, hc: *const Hc, l: HcLayer, rows: usize) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try hc.whole(r, l, rows);
    const whole = try hc.outputs(r, rows, a);
    try hc.quarters(r, l, rows);
    const got = try hc.outputs(r, rows, a);
    var line: [160]u8 = undefined;
    for (hc_outputs, whole, got) |name, x, y| r.report(try std.fmt.bufPrint(&line, "hc rows={d}: {s} at the four nodes' quarters against the whole pass's", .{ rows, name }), differs(x, y), true);
    // control: one value of plane 1's row c set to 64, the quarters again
    const c = pass_replay.controlRow(rows);
    const at = pass_replay.planeAt(1, rows, c, D) + 7 * 4;
    var was: [4]u8 = undefined;
    try hc.quarters(r, l, rows);
    const off = try hc.outputs(r, rows, a);
    for (hc_outputs, whole, off) |name, x, y| r.report(try std.fmt.bufPrint(&line, "hc rows={d} control: {s} with one value of plane 1's row {d} set to 64", .{ rows, name, c }), differs(x, y), false);
}

/// The boundary layer's own buffers: its pass's planes (the post it takes) and its h_out, x and pre_out.
const Boundary = struct {
    g: cuda.DeviceBuffer, // fp32 [2, R, D]
    h_out: cuda.DeviceBuffer,
    x: cuda.DeviceBuffer,
    pre_out: cuda.DeviceBuffer,

    fn init(r: *Run, rng: std.Random, cap: usize, keep: *std.ArrayList(cuda.DeviceBuffer)) !Boundary {
        const bx: Boundary = .{
            .g = try upload(r.d, try f32s(2 * cap * D, rng, r.a)),
            .h_out = try cuda.DeviceBuffer.alloc(r.d, cap * HC * D * 2),
            .x = try cuda.DeviceBuffer.alloc(r.d, cap * D * 2),
            .pre_out = try cuda.DeviceBuffer.alloc(r.d, cap * HC * 4),
        };
        for ([_]cuda.DeviceBuffer{ bx.g, bx.h_out, bx.x, bx.pre_out }) |b| try keep.append(r.a, b);
        return bx;
    }

    /// The boundary layer's hc_pre_pf on every row of a `rows`-row pass, on what the layer before it left in `hc` (its
    /// h_out as the streams, its pre_out as pre_in, post and comb in place), with the post of its own planes: h_out, x,
    /// pre_out, post and comb after it.
    fn run(bx: *const Boundary, r: *Run, hc: *const Hc, l: HcLayer, rows: usize, a: std.mem.Allocator) ![5][]u8 {
        for ([_]cuda.DeviceBuffer{ bx.h_out, bx.x, bx.pre_out }) |b| try clear(b, r.d, r.s);
        _ = try tri_hc.hcPrePf(r.t, hc.h_out.ptr, l.fn_w.ptr, l.scale.ptr, l.base.ptr, hc.pre_out.ptr, l.norm_w.ptr, EPS, HC_EPS, HC_ITERS, bx.x.ptr, bx.pre_out.ptr, hc.post.ptr, hc.comb.ptr, hc.part.ptr, .{ .gathered = bx.g.ptr, .world = 2, .h_out = bx.h_out.ptr }, rows, D);
        try r.s.synchronize();
        return .{
            try get(bx.h_out.ptr, rows * HC * D * 2, r.d, a),
            try get(bx.x.ptr, rows * D * 2, r.d, a),
            try get(bx.pre_out.ptr, rows * HC * 4, r.d, a),
            try get(hc.post.ptr, rows * HC * 4, r.d, a),
            try get(hc.comb.ptr, rows * HC * HC * 4, r.d, a),
        };
    }
};

/// WP4's boundary at a `rows`-row pass: the layer after the owned layers (TF_DS_BOUNDARY_OWNED off) runs on every row and
/// reads every quarter's streams, pre, post and comb: after the four quarters' launches it gives the bytes it gives
/// after the whole pass's launch; then with quarters 1-3's post and comb as they were before the layer, as a node holds
/// them without the resync (it writes only its own quarter's rows).
fn wp4(r: *Run, hc: *const Hc, bx: *const Boundary, la: HcLayer, lb: HcLayer, rows: usize) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try hc.whole(r, la, rows);
    const whole = try bx.run(r, hc, lb, rows, a);
    try hc.quarters(r, la, rows);
    const got = try bx.run(r, hc, lb, rows, a);
    var line: [160]u8 = undefined;
    for (hc_outputs, whole, got) |name, x, y| r.report(try std.fmt.bufPrint(&line, "wp4 rows={d}: the boundary's {s} after the quarters' launches against after the whole pass's", .{ rows, name }), differs(x, y), true);
    try hc.quarters(r, la, rows);
    try r.s.synchronize();
    const q1 = pass_replay.quarter(rows, 1).start;
    const off = try bx.run(r, hc, lb, rows, a);
    for (hc_outputs, whole, off) |name, x, y| r.report(try std.fmt.bufPrint(&line, "wp4 rows={d} control: the boundary's {s} with quarters 1-3's post and comb not resynced", .{ rows, name }), differs(x, y), false);
}

/// The indexer's scorer inputs and outputs for passes of up to `cap` rows over up to `cap` keys: the queries, FP4 keys,
/// head weights, visible counts; the whole pass's and the quarters' scores and tile maxima.
const Idx = struct {
    q: cuda.DeviceBuffer, // bf16 [cap, IH, ID]
    kc: cuda.DeviceBuffer, // [cap, ID / 2]
    ks: cuda.DeviceBuffer, // [cap, ID / 32]
    wt: cuda.DeviceBuffer, // bf16 [cap, IH]
    vis: cuda.DeviceBuffer, // int64 [cap]
    out: [2]cuda.DeviceBuffer, // int64 or fp32 [R, n]: the whole pass's, the quarters'
    tmax: [2]cuda.DeviceBuffer, // int64 [R, n / 64]

    fn init(r: *Run, rng: std.Random, cap: usize, keep: *std.ArrayList(cuda.DeviceBuffer)) !Idx {
        const d = r.d;
        const a = r.a;
        const kc = try a.alloc(u8, cap * ID / 2);
        for (kc) |*v| v.* = rng.int(u8);
        const ks = try a.alloc(u8, cap * ID / 32);
        for (ks) |*v| v.* = 23 + rng.uintLessThan(u8, 127 - 23 + 1);
        var ix: Idx = .{
            .q = try upload(d, try bf16s(cap * IH * ID, rng, 0.5, a)),
            .kc = try upload(d, kc),
            .ks = try upload(d, ks),
            .wt = try upload(d, try bf16s(cap * IH, rng, 0.5, a)),
            .vis = try cuda.DeviceBuffer.alloc(d, cap * 8),
            .out = undefined,
            .tmax = undefined,
        };
        for (&ix.out, &ix.tmax) |*o, *t| {
            o.* = try cuda.DeviceBuffer.alloc(d, cap * cap * 8);
            t.* = try cuda.DeviceBuffer.alloc(d, cap * (cap / tri_index.tile) * 8);
        }
        for ([_]cuda.DeviceBuffer{ ix.q, ix.kc, ix.ks, ix.wt, ix.vis } ++ ix.out ++ ix.tmax) |b| try keep.append(a, b);
        return ix;
    }

    /// _index_score_rows over `n` keys for `rows` rows from row `start` into out[i] (keyed: and tmax[i]), every view at
    /// the rows' offsets (the indexer's row-block loop).
    fn score(ix: *const Idx, r: *Run, start: usize, rows: usize, n: usize, keyed: bool, i: usize) !void {
        const k: tri_index.IndexK = .{ .fp4 = .{ .codes = ix.kc.ptr, .scales = ix.ks.ptr } };
        const oe: usize = if (keyed) 8 else 4;
        const tm: ?u64 = if (keyed) ix.tmax[i].ptr + start * (n / tri_index.tile) * 8 else null;
        try tri_index.indexScore(r.t, ix.q.ptr + start * IH * ID * 2, k, ix.wt.ptr + start * IH * 2, ix.vis.ptr + start * 8, n, ix.out[i].ptr + start * n * oe, null, keyed, tm, false, rows, IH, ID);
    }

    /// The whole pass's (i 0) or the quarters' (i 1) outputs of a `rows`-row pass over `n` keys: the scores (keyed: and
    /// the tile maxima).
    fn outputs(ix: *const Idx, r: *Run, rows: usize, n: usize, keyed: bool, i: usize, a: std.mem.Allocator) ![2][]const u8 {
        try r.s.synchronize();
        return .{
            try get(ix.out[i].ptr, rows * n * @as(usize, if (keyed) 8 else 4), r.d, a),
            if (keyed) try get(ix.tmax[i].ptr, rows * (n / tri_index.tile) * 8, r.d, a) else &.{},
        };
    }

    /// The four nodes' quarters of a `rows`-row pass into out[1] (and tmax[1]), 0xa5 first.
    fn quarters(ix: *const Idx, r: *Run, rows: usize, n: usize, keyed: bool) !void {
        try clear(ix.out[1], r.d, r.s);
        try clear(ix.tmax[1], r.d, r.s);
        for (0..4) |k| {
            const q = pass_replay.quarter(rows, k);
            try ix.score(r, q.start, q.n, n, keyed, 1);
        }
    }
};

/// TF_DS_IDX_SHARD's scores at a `rows`-row pass over `rows` keys (each row's visible count 64 + its row, at most
/// all): the four nodes' quarters against one launch of the whole pass, at RB 1 and 2, keyed and plain; then with one
/// query value of row controlRow set to 64.
fn idxCase(r: *Run, ix: *const Idx, rows: usize) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const n = rows;
    const vis = try a.alloc(i64, rows);
    for (vis, 0..) |*v, i| v.* = @intCast(@min(n, 64 + i));
    try r.s.synchronize();
    const rb_was = tri_index.idx_rb;
    defer tri_index.idx_rb = rb_was;
    const c = pass_replay.controlRow(rows);
    const at = (c * IH * ID + 3) * 2; // head 0, dim 3
    const names = [2][]const u8{ "the scores", "the tile maxima" };
    var line: [160]u8 = undefined;
    for ([_]usize{ 1, 2 }) |rb| for ([_]bool{ true, false }) |keyed| {
        tri_index.idx_rb = rb;
        const kind = if (keyed) "keyed" else "plain";
        try clear(ix.out[0], r.d, r.s);
        try clear(ix.tmax[0], r.d, r.s);
        try ix.score(r, 0, rows, n, keyed, 0);
        const whole = try ix.outputs(r, rows, n, keyed, 0, a);
        try ix.quarters(r, rows, n, keyed);
        const got = try ix.outputs(r, rows, n, keyed, 1, a);
        const shown: usize = if (keyed) 2 else 1;
        for (names[0..shown], whole[0..shown], got[0..shown]) |name, x, y| r.report(try std.fmt.bufPrint(&line, "idx rows={d} RB={d} {s}: {s} at the four nodes' quarters against the whole pass's", .{ rows, rb, kind, name }), differs(x, y), true);
        var was: [2]u8 = undefined;
        try ix.quarters(r, rows, n, keyed);
        const off = try ix.outputs(r, rows, n, keyed, 1, a);
        for (names[0..shown], whole[0..shown], off[0..shown]) |name, x, y| r.report(try std.fmt.bufPrint(&line, "idx rows={d} RB={d} {s} control: {s} with one query value of row {d} set to 64", .{ rows, rb, kind, name, c }), differs(x, y), false);
    };
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2 and !(args.len == 4 and std.mem.eql(u8, args[2], "--rows"))) {
        std.debug.print("usage: tf-dsv41-rows-replay AOT_DIR [--rows 1024,2048,4096]\n", .{});
        return 2;
    }
    const sizes = pass_replay.parseRows(if (args.len == 4) args[3] else null) catch {
        std.debug.print("tf-dsv41-rows-replay: --rows takes up to {d} pass sizes, comma-separated, each a multiple of 1024 up to {d}\n", .{ pass_replay.max_sizes, dsv41.model.pass_rows_max });
        return 2;
    };
    if (!cuda.kernels.available) {
        std.debug.print("tf-dsv41-rows-replay: this build has no kernel images: run the image's binary\n", .{});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var set = try cuda.aot.Set.load(init.gpa, io, &driver, ctx.device, args[1]);
    defer set.deinit();
    var k = try mx.Kernels.load(&driver, cuda.kernels.dsv41_native);
    defer k.unload();
    var ops = try dsv41.ops.Ops.load(&driver, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    defer ops.unload();
    var ex = try dsv41.exact.Exact.load(&driver, cuda.kernels.dsv41_torch);
    defer ex.unload();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var r: Run = .{ .d = &driver, .s = stream, .t = .{ .set = &set, .stream = stream, .pdl = true }, .k = &k, .a = a, .gpa = init.gpa };
    var prng = std.Random.DefaultPrng.init(0x0f1e5);
    try front(&r, prng.random(), &keep);
    try ffn(&r, prng.random(), &keep);
    for ([_]usize{ CAP, 1000 }) |rows| try pack(&r, &ops, prng.random(), &keep, rows);
    try gate(&r, prng.random(), &keep);
    for ([_]usize{ CAP, 1100, 40 }) |rows| try cat(&r, &ops, prng.random(), &keep, rows);
    k.bf16_rows = true; // TF_DS_BF16_ROWS=1, as the release runs the boundary's bf16 GEMMs
    try kvsrc(&r, &ops, &ex, prng.random(), &keep);
    k.bf16_rows = false;
    var cap: usize = 0;
    for (sizes.slice()) |rows| cap = @max(cap, rows);
    const hc = try Hc.init(&r, prng.random(), cap, &keep);
    const la = try HcLayer.init(&r, prng.random(), &keep);
    for (sizes.slice()) |rows| try hcCase(&r, &hc, la, rows);
    const lb = try HcLayer.init(&r, prng.random(), &keep);
    const bx = try Boundary.init(&r, prng.random(), cap, &keep);
    for (sizes.slice()) |rows| try wp4(&r, &hc, &bx, la, lb, rows);
    const ix = try Idx.init(&r, prng.random(), cap, &keep);
    for (sizes.slice()) |rows| try idxCase(&r, &ix, rows);
    if (r.ok) {
        var list: [80]u8 = undefined;
        var len: usize = 0;
        for (sizes.slice(), 0..) |rows, i| len += (try std.fmt.bufPrint(list[len..], "{s}{d}", .{ if (i == 0) "" else ",", rows })).len;
        std.debug.print("rows-replay: every owned-row path gives the whole chunk's bytes at q = 512, 256 and 1024, the packs' codes down's own at 2048 and 1000 rows, the Engram gate's, wo_b's codes and output mx.linear's own at 2048, 1100 and 40 rows, the boundary's kv-source update's four cache arrays the whole chunk's through the gathers' spans; hc_pre_pf and the indexer's scores at every node's quarter the whole pass's own, and the boundary after the quarters, at {s} rows; every control as it must\n", .{list[0..len]});
        return 0;
    }
    std.debug.print("rows-replay: FAILED (the lines above)\n", .{});
    return 1;
}
