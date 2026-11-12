//! tf-dsv41-quant-gemv-bench: whether a decode linear's activation quant should move into its GEMV (the quant-into-GEMV
//! fusion's gate), on a 2D node's dense decode shapes (qgemv.shapes: wq_a + wkv in one launch, wq_b, wo_a's two groups,
//! wo_b, the drafter's FP8 head; random FP8 weights and bf16 rows) at 1, 2 and 6 rows. Each chain follows a 2-CTA
//! producer of the linear's bf16 input rows (tf_dsn_bench_rows: it waits for its predecessor, then lets its successor
//! start, as the kit's kernels do), as a round runs it:
//!   A   the baseline chain: tf_dsn_dec_quant (an ordinary launch), then the decode GEMV (a programmatic dependent launch):
//!       mx.linearsDecode, mx.woA8;
//!   B   the prototype: tf_dsn_dec_fp8x_r1, the GEMV quantizing its own rows after its wait, a programmatic launch;
//!   B'  the prototype as an ordinary launch (where no programmatic edge exists: after a join);
//!   G   the GEMV alone (tf_dsn_dec_fp8_r1 on rows quantized before), an ordinary launch.
//! Each arm 100 chains in one CUDA graph; 2 graph launches to warm up, then 20 between CUDA events: us a chain, the
//! 20 blocks' median, min and max. the outputs of B and B' against A's, byte for byte (0 bytes must differ), and a control
//! (B on rows with one value changed must differ). One JSON line a cell; a shape's verdict (qgemv.builds: at least 1 us
//! shorter at every row count, the spread under half the gain). On a GB10 with its engine stopped and its GPU clock set
//! (the measurement); elsewhere its byte checks alone mean anything. Exit 0 when every check holds, 1 otherwise, 2 for
//! a usage or build problem.
//!   tf-dsv41-quant-gemv-bench [AOT_DIR]   (AOT_DIR unread: the replays' calling convention)
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const mx = dsv41.mx;
const weights = dsv41.weights;
const qgemv = dsv41.qgemv;

const chains = 100; // a graph's chains
const blocks = 20;
const warm = 2;

fn bf16Rne(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// An FP8 linear [n, k] of random E4M3 codes (no NaN) and UE8M0 scales a 32 x 32 block (2^-9 .. 2^0).
fn fp8Linear(d: *const cuda.Driver, n: usize, k: usize, rng: std.Random, a: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer)) !weights.Linear {
    const w = try a.alloc(u8, n * k);
    for (w) |*x| {
        x.* = rng.int(u8);
        if (x.* & 0x7f == 0x7f) x.* -= 1;
    }
    const sc = try a.alloc(u8, (n / 32) * (k / 32));
    for (sc) |*x| x.* = 118 + rng.uintLessThan(u8, 10);
    const wb = try cuda.DeviceBuffer.fromHost(d, w);
    try keep.append(a, wb);
    const sb = try cuda.DeviceBuffer.fromHost(d, sc);
    try keep.append(a, sb);
    return .{ .words = 0, .k = @intCast(k), .n = @intCast(n), .fmt = .fp8, .w = wb.ptr, .s = sb.ptr };
}

const Bench = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const mx.Kernels,
    proto: cuda.Function, // tf_dsn_dec_fp8x_r1
    gemv: cuda.Function, // tf_dsn_dec_fp8_r1
    rows_k: cuda.Function, // tf_dsn_bench_rows
    sc: mx.Scratch,
    z: mx.Z,
    ok: bool = true,

    fn produce(b: *const Bench, src: u64, dst: u64, bytes: usize) !void {
        var args: cuda.Args = .{};
        args.add(src);
        args.add(dst);
        args.add(@as(c_int, @intCast(bytes / 16)));
        try cuda.launch.launch(b.rows_k, .{ .grid = .{ .x = 2, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, b.s, &args);
    }
};

/// One cell's arms on a shape's linears at m rows: the producer's source and the rows it writes, the arms' outputs.
const Cell = struct {
    b: *Bench,
    sh: qgemv.Shape,
    ls: []const weights.Linear,
    m: usize,
    src: u64,
    x: u64,
    ys: [3][4]u64, // A, B, B'
    pro: qgemv.Launch, // the prototype's
    gl: qgemv.Launch, // the GEMV alone's (the baseline, reading A's codes)
    shared: u32,

    fn bytes(c: *const Cell) usize {
        return c.m * c.sh.k * c.sh.groups * 2;
    }

    fn armA(c: *const Cell) !void {
        const b = c.b;
        try b.produce(c.src, c.x, c.bytes());
        if (c.sh.groups > 1) return mx.woA8(b.k, b.s, b.sc, c.ls[0], c.sh.groups, c.x, c.sh.k * c.sh.groups, c.m, c.ys[0][0], c.ls[0].n, b.z);
        var xs: [4]u64 = undefined;
        var ldxs: [4]usize = undefined;
        var ldos: [4]usize = undefined;
        var types: [4]mx.DType = undefined;
        for (c.ls, 0..) |l, i| {
            xs[i] = c.x;
            ldxs[i] = c.sh.k;
            ldos[i] = l.n;
            types[i] = c.sh.y;
        }
        const n = c.ls.len;
        try mx.linearsDecode(b.k, b.s, b.sc, c.ls, xs[0..n], ldxs[0..n], c.m, c.ys[0][0..n], ldos[0..n], types[0..n], b.z);
    }

    fn armB(c: *const Cell, pdl: bool) !void {
        const b = c.b;
        try b.produce(c.src, c.x, c.bytes());
        var l = c.pro;
        for (0..c.ls.len) |i| l.mm.p[i].y = c.ys[if (pdl) 1 else 2][i];
        var args: cuda.Args = .{};
        args.add(l.mm);
        args.add(qgemv.QxArgs{ .x = c.x, .ldx = @intCast(c.sh.k * c.sh.groups), .gk = @intCast(c.sh.k) });
        try cuda.launch.launch(b.proto, .{ .grid = .{ .x = @intCast(l.ctas), .y = 1, .z = 1 }, .block = .{ .x = 32 * mx.dec_warps }, .shared = c.shared, .pdl = pdl }, b.s, &args);
    }

    fn armG(c: *const Cell) !void {
        const b = c.b;
        var args: cuda.Args = .{};
        args.add(c.gl.mm);
        try cuda.launch.launch(b.gemv, .{ .grid = .{ .x = @intCast(c.gl.ctas), .y = 1, .z = 1 }, .block = .{ .x = 32 * mx.dec_warps }, .shared = qgemv.ringShared() }, b.s, &args);
    }

    fn run(c: *const Cell, arm: u2) !void {
        return switch (arm) {
            0 => c.armA(),
            1 => c.armB(true),
            2 => c.armB(false),
            3 => c.armG(),
        };
    }

    /// The arm's 100 chains in one graph: us a chain over 20 blocks.
    fn timed(c: *const Cell, arm: u2, us: *[blocks]f64) !void {
        const b = c.b;
        try cuda.graph.beginCapture(b.s, .thread_local);
        for (0..chains) |_| try c.run(arm);
        var g = try cuda.graph.endCapture(b.s);
        defer g.deinit();
        var exec = try g.instantiate();
        defer exec.deinit();
        for (0..warm) |_| try exec.launchOn(b.s);
        var ev: [blocks + 1]cuda.Event = undefined;
        for (&ev) |*e| e.* = try cuda.Event.init(b.d, true);
        defer for (&ev) |*e| e.deinit();
        try ev[0].record(b.s);
        for (0..blocks) |i| {
            try exec.launchOn(b.s);
            try ev[i + 1].record(b.s);
        }
        try b.s.synchronize();
        for (0..blocks) |i| us[i] = @as(f64, try cuda.Event.elapsedMs(ev[i], ev[i + 1])) * 1000.0 / chains;
    }
};

fn outBytes(sh: qgemv.Shape, l: weights.Linear, m: usize) usize {
    return m * l.n * @as(usize, if (sh.y == .f32) 4 else 2);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    if (!cuda.kernels.available) {
        std.debug.print("tf-dsv41-quant-gemv-bench: this build has no kernel images: run the image's binary\n", .{});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var k = try mx.Kernels.load(&driver, cuda.kernels.dsv41_native);
    defer k.unload();
    if (!k.dec_on) {
        std.debug.print("tf-dsv41-quant-gemv-bench: the decode tensor-core kernels are off (TF_DS_DECODE_DEC=0): the baseline chain is not the GEMV's\n", .{});
        return 2;
    }
    var mod = try cuda.Module.load(&driver, cuda.kernels.dsv41_native);
    defer mod.unload();
    const proto = try mod.function("tf_dsn_dec_fp8x_r1_kernel");
    var most: u32 = 0;
    for (qgemv.shapes) |sh| for (qgemv.rows) |m| {
        most = @max(most, qgemv.protoShared(m, sh.k));
    };
    try proto.allowDynamicShared(most);
    const gemv = try mod.function("tf_dsn_dec_fp8_r1_kernel");
    const rows_k = try mod.function("tf_dsn_bench_rows_kernel");
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*x| x.free();
    // the linears' scratch: the codes and scales of 8 rows of the widest input row (wo_a's two groups)
    var max_k: usize = 0;
    for (qgemv.shapes) |sh| max_k = @max(max_k, sh.k * sh.groups);
    const sz = mx.Scratch.sizes(qgemv.max_rows, max_k, 1);
    var bufs: [4]u64 = undefined;
    for (sz, 0..) |n, i| {
        const x = try cuda.DeviceBuffer.alloc(&driver, n);
        try keep.append(a, x);
        bufs[i] = x.ptr;
    }
    const sc: mx.Scratch = .{ .q = bufs[0], .s = bufs[1], .z = bufs[2], .fake = bufs[3], .rows = qgemv.max_rows, .max_k = max_k };
    var b: Bench = .{ .d = &driver, .s = stream, .k = &k, .proto = proto, .gemv = gemv, .rows_k = rows_k, .sc = sc, .z = .{ .ptr = bufs[2], .floats = 1 } };
    var prng = std.Random.DefaultPrng.init(0x7167);
    const rng = prng.random();
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &out_buf);
    for (qgemv.shapes) |sh| {
        var ls: [4]weights.Linear = undefined;
        for (sh.ns, 0..) |n, i| ls[i] = try fp8Linear(&driver, n, sh.k, rng, a, &keep);
        const lin = ls[0..sh.ns.len];
        var st: [3][qgemv.rows.len]qgemv.Stat = undefined; // A, B, B'
        for (qgemv.rows, 0..) |m, ri| {
            const kk = sh.k * sh.groups;
            const host = try a.alloc(u16, m * kk);
            for (host) |*v| v.* = bf16Rne(rng.floatNorm(f32) * 0.5);
            const src = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(host));
            try keep.append(a, src);
            const x = try cuda.DeviceBuffer.alloc(&driver, m * kk * 2);
            try keep.append(a, x);
            var c: Cell = .{ .b = &b, .sh = sh, .ls = lin, .m = m, .src = src.ptr, .x = x.ptr, .ys = undefined, .pro = undefined, .gl = undefined, .shared = qgemv.protoShared(m, sh.k) };
            for (0..3) |arm| for (lin, 0..) |l, i| {
                const y = try cuda.DeviceBuffer.alloc(&driver, outBytes(sh, l, m));
                try keep.append(a, y);
                c.ys[arm][i] = y.ptr;
            };
            c.pro = try qgemv.launchOf(sh, lin, m, c.ys[1][0..lin.len], null);
            c.gl = try qgemv.launchOf(sh, lin, m, c.ys[0][0..lin.len], .{ .q = sc.q, .s = sc.s });
            // the byte check: each arm once on outputs set to 0xa5 first
            for (0..3) |arm| for (lin, 0..) |l, i| try driver.check(driver.api.cuMemsetD8Async(c.ys[arm][i], 0xa5, outBytes(sh, l, m), stream.handle), "cuMemsetD8Async");
            for (0..3) |arm| try c.run(@intCast(arm));
            try stream.synchronize();
            var differ: [2]usize = .{ 0, 0 };
            for (lin, 0..) |l, i| {
                const n = outBytes(sh, l, m);
                const ya = try a.alloc(u8, n);
                const yb = try a.alloc(u8, n);
                try driver.check(driver.api.cuMemcpyDtoH_v2(ya.ptr, c.ys[0][i], n), "cuMemcpyDtoH");
                for (0..2) |j| {
                    try driver.check(driver.api.cuMemcpyDtoH_v2(yb.ptr, c.ys[1 + j][i], n), "cuMemcpyDtoH");
                    for (ya, yb) |p, q| differ[j] += @intFromBool(p != q);
                }
            }
            for (differ, [_][]const u8{ "B", "B'" }) |n, name| {
                const pass = n == 0;
                if (!pass) b.ok = false;
                std.debug.print("{s}, {d} rows: {s} against A: {d} bytes differ: {s}\n", .{ sh.name, m, name, n, if (pass) "as it must" else "WRONG" });
            }
            // the control: B on rows with one value set to 64 differs from A
            {
                const one = try a.dupe(u16, host);
                one[one.len / 2] = bf16Rne(64.0);
                const src1 = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(one));
                try keep.append(a, src1);
                var c1 = c;
                c1.src = src1.ptr;
                try c1.armB(true);
                try stream.synchronize();
                var n: usize = 0;
                for (lin, 0..) |l, i| {
                    const nb = outBytes(sh, l, m);
                    const ya = try a.alloc(u8, nb);
                    const yb = try a.alloc(u8, nb);
                    try driver.check(driver.api.cuMemcpyDtoH_v2(ya.ptr, c.ys[0][i], nb), "cuMemcpyDtoH");
                    try driver.check(driver.api.cuMemcpyDtoH_v2(yb.ptr, c.ys[1][i], nb), "cuMemcpyDtoH");
                    for (ya, yb) |p, q| n += @intFromBool(p != q);
                }
                if (n == 0) b.ok = false;
                std.debug.print("{s}, {d} rows: the control (one input value 64) against A: {d} bytes differ: {s}\n", .{ sh.name, m, n, if (n != 0) "as it must" else "WRONG" });
            }
            // the four arms timed
            var t: [4]qgemv.Stat = undefined;
            for (0..4) |arm| {
                var us: [blocks]f64 = undefined;
                try c.timed(@intCast(arm), &us);
                t[arm] = qgemv.statOf(&us);
            }
            for (0..3) |arm| st[arm][ri] = t[arm];
            try out.interface.print("{{\"shape\": \"{s}\", \"rows\": {d}, \"ctas\": {d}, \"shared_b\": {d}, \"a_us\": {d:.3}, \"a_min\": {d:.3}, \"a_max\": {d:.3}, \"b_us\": {d:.3}, \"b_min\": {d:.3}, \"b_max\": {d:.3}, \"b2_us\": {d:.3}, \"b2_min\": {d:.3}, \"b2_max\": {d:.3}, \"g_us\": {d:.3}, \"g_min\": {d:.3}, \"g_max\": {d:.3}, \"b_vs_a_bytes\": {d}, \"b2_vs_a_bytes\": {d}}}\n", .{ sh.name, m, c.pro.ctas, c.shared, t[0].med, t[0].min, t[0].max, t[1].med, t[1].min, t[1].max, t[2].med, t[2].min, t[2].max, t[3].med, t[3].min, t[3].max, differ[0], differ[1] });
            try out.interface.flush();
            std.debug.print("{s}, {d} rows: A {d:.2} us a chain, B {d:.2} (A - B {d:.2}), B' {d:.2} (A - B' {d:.2}), the GEMV alone {d:.2}\n", .{ sh.name, m, t[0].med, t[1].med, t[0].med - t[1].med, t[2].med, t[0].med - t[2].med, t[3].med });
        }
        std.debug.print("{s}: the fusion {s} after a programmatic edge (B), {s} after a join (B')\n", .{ sh.name, if (qgemv.builds(&st[0], &st[1])) "pays" else "does not pay", if (qgemv.builds(&st[0], &st[2])) "pays" else "does not pay" });
    }
    if (b.ok) {
        std.debug.print("quant-gemv-bench: the prototype's outputs are the baseline bytes in every cell\n", .{});
        return 0;
    }
    std.debug.print("quant-gemv-bench: FAILED (the lines above)\n", .{});
    return 1;
}
