//! cuBLAS (not Lt) by dlopen, for the fp32 GEMMs the served build runs through torch.matmul: the MoE gate's logits and
//! the indexer's weights of prompt chunks, the DSpark confidence head. torch calls cublasSgemm_v2 on one handle with
//! CUBLAS_DEFAULT_MATH and a 32 MiB workspace (the recording's cuBLAS log); the same call on the same library picks the
//! same algorithm.
const std = @import("std");
const cuda = @import("cuda");

pub const Status = c_int;
pub const Handle = ?*opaque {};
pub const Op = enum(c_int) { n = 0, t = 1 };
pub const default_math: c_int = 0;

pub const Error = error{ LibraryUnavailable, MissingSymbol, CublasFailed };

const S = Status;
const D = u64;

pub const Api = struct {
    cublasCreate_v2: *const fn (*Handle) callconv(.c) S,
    cublasDestroy_v2: *const fn (Handle) callconv(.c) S,
    cublasSetStream_v2: *const fn (Handle, cuda.abi.Stream) callconv(.c) S,
    cublasSetWorkspace_v2: *const fn (Handle, D, usize) callconv(.c) S,
    cublasSetMathMode: *const fn (Handle, c_int) callconv(.c) S,
    cublasSgemm_v2: *const fn (Handle, Op, Op, c_int, c_int, c_int, *const f32, D, c_int, D, c_int, *const f32, D, c_int) callconv(.c) S,
    cublasGemmEx: *const fn (Handle, Op, Op, c_int, c_int, c_int, *const anyopaque, D, c_int, c_int, D, c_int, c_int, *const anyopaque, D, c_int, c_int, c_int, c_int) callconv(.c) S,
};

/// One handle bound to a stream, with torch's workspace size and math mode.
pub const Blas = struct {
    lib: std.DynLib,
    api: Api,
    handle: Handle = null,

    /// torch's default cuBLAS workspace on this GPU (the recorded cublasSetWorkspace_v2 size).
    pub const workspace_bytes: usize = 32 << 20;

    pub fn open(stream: cuda.Stream, workspace: u64) Error!Blas {
        var lib = std.DynLib.open("libcublas.so.13") catch std.DynLib.open("libcublas.so") catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        var b: Blas = .{ .lib = lib, .api = api };
        try b.check(api.cublasCreate_v2(&b.handle));
        try b.check(api.cublasSetStream_v2(b.handle, stream.handle));
        try b.check(api.cublasSetWorkspace_v2(b.handle, workspace, workspace_bytes));
        try b.check(api.cublasSetMathMode(b.handle, default_math));
        return b;
    }

    pub fn close(b: *Blas) void {
        _ = b.api.cublasDestroy_v2(b.handle);
        b.lib.close();
        b.* = undefined;
    }

    fn check(_: *const Blas, s: Status) Error!void {
        if (s != 0) {
            std.log.err("cuBLAS status {d}", .{s});
            return error.CublasFailed;
        }
    }

    /// nn.Linear without a bias of x bf16 [rows, k] and w bf16 [n, k]: out bf16 [rows, n], as torch issues it
    /// (cublasGemmEx: transa T, transb N, m = n, n = rows, lda = ldb = k, ldc = n, fp32 alpha 1 and beta 0, bf16
    /// operands, COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP; the recording's cuBLAS log for the vision MLP).
    pub fn linearBf16(b: *const Blas, x: u64, w: u64, out: u64, rows: usize, k: usize, n: usize) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        const bf16_type: c_int = 14; // CUDA_R_16BF
        try b.check(b.api.cublasGemmEx(b.handle, .t, .n, @intCast(n), @intCast(rows), @intCast(k), &one, w, bf16_type, @intCast(k), x, bf16_type, @intCast(k), &zero, out, bf16_type, @intCast(n), 68, 99));
    }

    /// torch.matmul of x [rows, k] fp32 (row-major) and w [n, k] fp32 transposed: out [rows, n] fp32, as torch issues
    /// it (column-major C^T = W x^T: transa T, transb N, m = n, n = rows, lda = ldb = k, ldc = n, alpha 1, beta 0).
    pub fn xwT(b: *const Blas, x: u64, w: u64, out: u64, rows: usize, k: usize, n: usize) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        try b.check(b.api.cublasSgemm_v2(b.handle, .t, .n, @intCast(n), @intCast(rows), @intCast(k), &one, w, @intCast(k), x, @intCast(k), &zero, out, @intCast(n)));
    }

    /// xwT in row blocks of `step` rows (the last block takes the rest): cuBLAS picks its algorithm (tiles, split-K) from
    /// the shape, so a pass longer than `step` gives each row the bits of the `step`-row call a `step`-row pass makes.
    pub fn xwTRows(b: *const Blas, x: u64, w: u64, out: u64, rows: usize, k: usize, n: usize, step: usize) Error!void {
        var at: usize = 0;
        while (at < rows) : (at += step) {
            const r = @min(step, rows - at);
            try b.xwT(x + at * k * 4, w, out + at * n * 4, r, k, n);
        }
    }

    /// torch.matmul of x [rows, k] fp32 (row-major) and v [1, k] fp32 transposed (the DSpark confidence head): out [rows]
    /// fp32. A [k, 1] column is contiguous by torch's rules (a size-1 dimension's stride is not checked), so torch
    /// issues C^T = v^T x^T untransposed: transa N, transb N, m = 1, n = rows, lda = 1, ldb = k, ldc = 1 (cuBLAS runs it
    /// as a batched dot product).
    pub fn xv(b: *const Blas, x: u64, v: u64, out: u64, rows: usize, k: usize) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        try b.check(b.api.cublasSgemm_v2(b.handle, .n, .n, 1, @intCast(rows), @intCast(k), &one, v, 1, x, @intCast(k), &zero, out, 1));
    }

    /// The same product as xv (out [rows] = x [rows, k] @ v [k], fp32) as an m = rows, n = 1 GEMM: cuBLAS's m = 1 shape
    /// runs ~2.5 ms on a GB10 for the drafter's 5 x 5376 confidence rows, this one microseconds.
    pub fn xvT(b: *const Blas, x: u64, v: u64, out: u64, rows: usize, k: usize) Error!void {
        const one: f32 = 1;
        const zero: f32 = 0;
        try b.check(b.api.cublasSgemm_v2(b.handle, .t, .n, @intCast(rows), 1, @intCast(k), &one, x, @intCast(k), v, @intCast(k), &zero, out, @intCast(rows)));
    }
};

test "two handles on two streams give a pass's bytes: the same GEMMs on the second handle (own workspace, own stream), run beside the first's, equal one handle's" {
    var driver = cuda.Driver.open() catch return error.SkipZigTest;
    defer driver.close();
    var cx = cuda.Context.init(&driver, 0) catch return error.SkipZigTest;
    defer cx.deinit();
    const gpa = std.testing.allocator;
    var s0 = try cuda.Stream.init(&driver, true);
    defer s0.deinit();
    var s1 = try cuda.Stream.init(&driver, true);
    defer s1.deinit();
    var ws0 = try cuda.DeviceBuffer.alloc(&driver, Blas.workspace_bytes);
    defer ws0.free();
    var ws1 = try cuda.DeviceBuffer.alloc(&driver, Blas.workspace_bytes);
    defer ws1.free();
    var b0 = Blas.open(s0, ws0.ptr) catch return error.SkipZigTest;
    defer b0.close();
    var b1 = try Blas.open(s1, ws1.ptr);
    defer b1.close();
    // shapes of a pass: bf16 linears (rows 4,096) and the fp32 matmul of the router
    const shapes = [_][3]usize{ .{ 4096, 1280, 2048 }, .{ 4096, 512, 1024 }, .{ 4096, 2048, 256 } };
    var prng = std.Random.DefaultPrng.init(0x2b7);
    const rng = prng.random();
    for (shapes) |sh| {
        const rows = sh[0];
        const k = sh[1];
        const n = sh[2];
        const xb = try gpa.alloc(u16, rows * k);
        defer gpa.free(xb);
        const wb = try gpa.alloc(u16, n * k);
        defer gpa.free(wb);
        // small bf16 values (exponent 0x3c..0x40) so no input is a NaN or an infinity
        for (xb) |*v| v.* = (@as(u16, rng.int(u8) & 0x80) << 8) | (@as(u16, 0x3c + rng.uintLessThan(u8, 5)) << 7) | rng.int(u7);
        for (wb) |*v| v.* = (@as(u16, rng.int(u8) & 0x80) << 8) | (@as(u16, 0x3c + rng.uintLessThan(u8, 5)) << 7) | rng.int(u7);
        var x = try cuda.DeviceBuffer.alloc(&driver, xb.len * 2);
        defer x.free();
        var w = try cuda.DeviceBuffer.alloc(&driver, wb.len * 2);
        defer w.free();
        try driver.copyHtoD(@src(), x.ptr, xb.ptr, xb.len * 2);
        try driver.copyHtoD(@src(), w.ptr, wb.ptr, wb.len * 2);
        var o: [3]cuda.DeviceBuffer = undefined;
        for (&o) |*b| b.* = try cuda.DeviceBuffer.alloc(&driver, rows * n * 2);
        defer for (&o) |*b| b.free();
        // the first handle alone, then both at once on their streams
        try b0.linearBf16(x.ptr, w.ptr, o[0].ptr, rows, k, n);
        try s0.synchronize();
        try b0.linearBf16(x.ptr, w.ptr, o[1].ptr, rows, k, n);
        try b1.linearBf16(x.ptr, w.ptr, o[2].ptr, rows, k, n);
        try s0.synchronize();
        try s1.synchronize();
        const got = try gpa.alloc(u8, rows * n * 2 * 3);
        defer gpa.free(got);
        for (o, 0..) |b, i| try driver.check(driver.api.cuMemcpyDtoH_v2(got[i * rows * n * 2 ..].ptr, b.ptr, rows * n * 2), "dtoh");
        const len = rows * n * 2;
        try std.testing.expectEqualSlices(u8, got[0..len], got[len .. 2 * len]);
        try std.testing.expectEqualSlices(u8, got[0..len], got[2 * len .. 3 * len]);
        // the control: one weight changed differs (the comparison is live)
        const one: u16 = 0x3f80;
        try driver.copyHtoD(@src(), w.ptr, &one, 2);
        try b1.linearBf16(x.ptr, w.ptr, o[2].ptr, rows, k, n);
        try s1.synchronize();
        const again = try gpa.alloc(u8, len);
        defer gpa.free(again);
        try driver.check(driver.api.cuMemcpyDtoH_v2(again.ptr, o[2].ptr, len), "dtoh");
        try std.testing.expect(!std.mem.eql(u8, again, got[0..len]));
    }
}
