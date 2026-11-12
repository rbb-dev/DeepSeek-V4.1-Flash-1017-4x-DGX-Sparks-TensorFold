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
//! Exit 0 when every check holds, 1 otherwise, 2 for a usage or build problem.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const mx = dsv41.mx;
const prompt = dsv41.prompt;
const tri_basic = dsv41.tri_basic;
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

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
        return 2;
    }
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
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var prng = std.Random.DefaultPrng.init(0x0f1e5);
    try front(&r, prng.random(), &keep);
    try ffn(&r, prng.random(), &keep);
    for ([_]usize{ CAP, 1000 }) |rows| try pack(&r, &ops, prng.random(), &keep, rows);
    if (r.ok) {
        return 0;
    }
    std.debug.print("rows-replay: FAILED (the lines above)\n", .{});
    return 1;
}
