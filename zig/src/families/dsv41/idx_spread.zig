//! The scale spreads of the indexer's FP4 dots on real prompts (a test-instruments build with TF_DS_IDX_SPREAD=1): beside a prompt chunk's index
//! score, a sampled count of how far apart the four block exponents of a query head's dot with a key lie (native_sf.cu's
//! tf_dsn_idx_spread_kernel), which decides how much of the scoring a runtime guard could leave on the FP4 MMA. A default build has none of it:
//! `enabled` is false and the call compiles away.
const std = @import("std");
const cuda = @import("cuda");
const mx = @import("mx.zig");
const ti = @import("test_instruments.zig");

pub const enabled = ti.enabled;

const bins = 32;
const words = 3 * bins + 2; // S, key spread, row spread, pairs, pairs a sufficient test passes
const every = 40; // calls a summary line
const rows_sampled = 64;
const keys_sampled = 2048;

var ctr: ?cuda.DeviceBuffer = null;
var func: ?cuda.Function = null;
var calls: u64 = 0;
var asked: ?bool = null;

fn on() bool {
    if (asked) |a| return a;
    const v = std.c.getenv("TF_DS_IDX_SPREAD");
    asked = if (v) |p| std.mem.eql(u8, std.mem.span(p), "1") else false;
    return asked.?;
}

/// The cumulative share of `h`'s samples at or below each bin edge in `edges`, as "<=e p%" pairs.
fn shares(w: *std.Io.Writer, h: []const u32, edges: []const usize) !void {
    var total: u64 = 0;
    for (h) |x| total += x;
    for (edges) |e| {
        var cum: u64 = 0;
        for (h[0 .. e + 1]) |x| cum += x;
        const permille = if (total == 0) 0 else cum * 1000 / total;
        try w.print(" <={d}: {d}.{d}%", .{ e, permille / 10, permille % 10 });
    }
}

/// A chunk's index score block: `iq` the pre-quantization queries [rows, 32, 128] bf16 (the packer's input), `kscales` the keys' E8M0 scale bytes
/// [n, 4], `vis` each row's visible keys. A sample of rows and keys, counted on the stream; every `every` calls a line with the totals so far.
pub fn observe(d: *const cuda.Driver, s: cuda.Stream, k: *const mx.Kernels, iq: u64, kscales: u64, vis: u64, rows: usize, n: usize) !void {
    if (!enabled or !on()) return;
    if (func == null) {
        const sk = k.sfk orelse return error.NoSpreadKernel;
        func = sk.module.function("tf_dsn_idx_spread_kernel") catch return error.NoSpreadKernel;
        const b = try cuda.DeviceBuffer.alloc(d, words * 4);
        try d.check(d.api.cuMemsetD8_v2(b.ptr, 0, words * 4), "cuMemsetD8");
        ctr = b;
    }
    const row_step: usize = @max(1, rows / rows_sampled);
    const key_step: usize = @max(1, n / keys_sampled);
    var args: cuda.Args = .{};
    args.add(iq);
    args.add(kscales);
    args.add(vis);
    args.add(@as(c_int, @intCast(rows)));
    args.add(@as(c_int, @intCast(row_step)));
    args.add(@as(c_int, @intCast(n)));
    args.add(@as(c_int, @intCast(key_step)));
    args.add(ctr.?.ptr);
    try cuda.launch.launch(func.?, .{ .grid = .{ .x = @intCast((rows + row_step - 1) / row_step), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &args);
    calls += 1;
    if (calls % every != 0) return;
    try s.synchronize();
    var h: [words]u32 = undefined;
    try d.check(d.api.cuMemcpyDtoH_v2(&h, ctr.?.ptr, words * 4), "cuMemcpyDtoH");
    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.print("{s}: TF_DS_IDX_SPREAD after {d} index scores, {d} sampled (row, key) pairs; a pair's largest head spread S of the four block exponents (all 32 heads):", .{ ti.marker, calls, h[3 * bins] }) catch {};
    shares(&w, h[0..bins], &[_]usize{ 4, 7, 10, 14 }) catch {};
    const suff = if (h[3 * bins] == 0) 0 else @as(u64, h[3 * bins + 1]) * 1000 / h[3 * bins];
    w.print("; the sufficient test (row spread + key spread <= 7) passes {d}.{d}%; the keys' own spread", .{ suff / 10, suff % 10 }) catch {};
    shares(&w, h[bins .. 2 * bins], &[_]usize{ 0, 2, 4, 7 }) catch {};
    w.print("; the rows' largest query-head spread", .{}) catch {};
    shares(&w, h[2 * bins .. 3 * bins], &[_]usize{ 0, 2, 4, 7 }) catch {};
    std.log.info("{s}", .{w.buffered()});
}
