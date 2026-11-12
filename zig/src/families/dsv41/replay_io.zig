//! The replay tools' copies and fills between the host and the device, each in the order of the tool's one stream.
//!
//! A tool's stream is non-blocking (cuda.Stream.init(d, true)), and a synchronous copy (cuMemcpyHtoD / DtoH,
//! DeviceBuffer's upload / download / fromHost) or fill (cuMemsetD8 / D32, which returns before its bytes land) runs on
//! the legacy stream, which a non-blocking stream does not wait for. A launch could then read a payload, a control's
//! changed or restored byte or a filled range before it lands, and a read could see a launch's output before it is
//! written. Copies in go on the stream and have landed when the call returns; fills go on the stream, before every later
//! launch and copy on it; copies out wait for every launch before them. A tool sets `stream` once, in main, before its
//! first copy.
const std = @import("std");
const cuda = @import("cuda");

/// The tool's one stream (main's); a copy before it is set is refused.
pub var stream: ?cuda.Stream = null;

/// n host bytes to the device at `dst`, on the stream, landed when it returns.
pub fn put(d: *const cuda.Driver, dst: u64, src: ?*const anyopaque, n: usize) !void {
    const s = stream orelse return error.NoReplayStream;
    try s.synchronize();
}

/// n device bytes at `src` to the host, after every launch on the stream before it.
pub fn take(d: *const cuda.Driver, dst: ?*anyopaque, src: u64, n: usize) !void {
    const s = stream orelse return error.NoReplayStream;
    try s.synchronize();
    try d.check(d.api.cuMemcpyDtoH_v2(dst, src, n), "cuMemcpyDtoH");
}

/// n device bytes at `dst` set to `byte`, on the stream: every later launch and copy on it sees them.
pub fn fill(d: *const cuda.Driver, dst: u64, byte: u8, n: usize) !void {
    const s = stream orelse return error.NoReplayStream;
    try d.check(d.api.cuMemsetD8Async(dst, byte, n, s.handle), "cuMemsetD8Async");
}

/// A device buffer holding `bytes` (put).
pub fn buffer(d: *const cuda.Driver, bytes: []const u8) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(d, bytes.len);
    errdefer b.free();
    if (bytes.len > 0) try put(d, b.ptr, bytes.ptr, bytes.len);
    return b;
}

/// `bytes` into `b` at `offset` (in its range), as put.
pub fn write(b: cuda.DeviceBuffer, offset: usize, bytes: []const u8) !void {
    if (offset > b.len or bytes.len > b.len - offset) return error.Invalid;
    if (bytes.len > 0) try put(b.d, b.ptr + offset, bytes.ptr, bytes.len);
}

/// `out.len` bytes of `b` from `offset` (in its range), as take.
pub fn read(b: cuda.DeviceBuffer, offset: usize, out: []u8) !void {
    if (offset > b.len or out.len > b.len - offset) return error.Invalid;
    if (out.len > 0) try take(b.d, out.ptr, b.ptr + offset, out.len);
}

/// A device range's n bytes read back into memory from `a` (the caller frees them), as take.
pub fn get(d: *const cuda.Driver, src: u64, n: usize, a: std.mem.Allocator) ![]u8 {
    const x = try a.alloc(u8, n);
    errdefer a.free(x);
    if (n > 0) try take(d, x.ptr, src, n);
    return x;
}

/// What a tool may not call itself: every synchronous copy and fill, and every asynchronous copy of its own (its
/// device-to-device copies and its fills on its own stream are in its stream's order already).

/// For each tool's test: the copies and fills of `src` (a tool's source) that are not through this file, and whether its
/// main sets `stream`.
pub fn audit(src: []const u8) struct { outside: usize, sets_stream: bool } {
    var outside: usize = 0;
    for (forbidden) |f| outside += std.mem.count(u8, src, f);
    return .{ .outside = outside, .sets_stream = std.mem.count(u8, src, "replay_io.stream = ") - std.mem.count(u8, src, "replay_io.stream = null") == 1 };
}

/// A tool's copies and fills, each outside this file, printed with its line (a test's failure says where).
fn printOutside(name: []const u8, src: []const u8) void {
    for (forbidden) |f| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, src, at, f)) |i| : (at = i + 1) {
            const ls = (std.mem.lastIndexOfScalar(u8, src[0..i], '\n') orelse 0) + 1;
            const le = std.mem.indexOfScalarPos(u8, src, i, '\n') orelse src.len;
            std.debug.print("{s}: a copy or fill outside replay_io: {s}\n", .{ name, std.mem.trim(u8, src[ls..le], " ") });
        }
    }
}

fn expectOnlyThroughHere(comptime name: []const u8) !void {
    const src = @embedFile(name);
    const r = audit(src);
    if (r.outside != 0) printOutside(name, src);
    if (!r.sets_stream) std.debug.print("{s}: its main does not set replay_io.stream\n", .{name});
    try std.testing.expectEqual(@as(usize, 0), r.outside);
    try std.testing.expect(r.sets_stream);
}

test "the replay tools' copies: in on the stream and landed before the call returns, out after every launch before them" {
    const self = @embedFile("replay_io.zig");
    const Body = struct {
        fn of(text: []const u8, head: []const u8) []const u8 {
            const at = std.mem.indexOf(u8, text, head) orelse return "";
            const end = std.mem.indexOf(u8, text[at..], "\n}\n") orelse return "";
            return text[at .. at + end];
        }
        fn before(text: []const u8, first: []const u8, then: []const u8) bool {
            const x = std.mem.indexOf(u8, text, first) orelse return false;
            const y = std.mem.indexOf(u8, text, then) orelse return false;
            return x < y;
        }
    };
    const p = Body.of(self, "pub fn put(");
    const t = Body.of(self, "pub fn take(");
    try std.testing.expect(Body.before(t, "try s.synchronize();", "cuMemcpyDtoH_v2(dst, src, n)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, Body.of(self, "pub fn fill("), "cuMemsetD8Async(dst, byte, n, s.handle)"));
    // every other copy here is put's or take's
    for ([_][]const u8{ "pub fn buffer(", "pub fn write(", "pub fn read(", "pub fn get(" }) |h| {
        const b = Body.of(self, h);
        try std.testing.expect(b.len > 0);
        try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, b, "cuMemcpy"));
    }
    // the audit sees each kind of copy, and a tool that sets its stream
    try std.testing.expectEqual(@as(usize, 3), audit("try b.upload(0, x); try b.download(0, y); _ = fromHost(d, z);").outside);
    try std.testing.expectEqual(@as(usize, 2), audit("cuMemcpyHtoD_v2(a, b, c) cuMemcpyDtoH_v2(a, b, c)").outside);
    try std.testing.expectEqual(@as(usize, 2), audit("cuMemsetD8_v2(a, b, c) cuMemsetD32_v2(a, b, c) cuMemsetD8Async(a, b, c, s)").outside);
    try std.testing.expectEqual(@as(usize, 1), audit("cuMemcpyDtoD_v2(a, b, c) cuMemcpyDtoDAsync_v2(a, b, c, s)").outside);
    try std.testing.expectEqual(@as(usize, 0), audit("try replay_io.write(b, 0, x); try replay_io.read(b, 0, y);").outside);
    try std.testing.expect(audit("    dsv41.replay_io.stream = stream;\n").sets_stream);
}

test "the rows replay copies to and from the device in its stream's order alone" {
    try expectOnlyThroughHere("rows_replay_main.zig");
}

test "the quarter-HC boundary replay (wp4) copies to and from the device in its stream's order alone" {
    try expectOnlyThroughHere("wp4_replay_main.zig");
}

test "the indexer's score replay copies to and from the device in its stream's order alone" {
    try expectOnlyThroughHere("idx_replay_main.zig");
}

test "the indexer shard replay (wp6b) copies to and from the device in its stream's order alone" {
    try expectOnlyThroughHere("wp6b_idx_shard_replay_main.zig");
}

test "the quarter-HC mix replay copies to and from the device in its stream's order alone" {
    try expectOnlyThroughHere("hc_replay_main.zig");
}

test "the kernels replay copies to and from the device in its stream's order alone" {
    try expectOnlyThroughHere("kernels_replay_main.zig");
}

test "the kernels replay reads TF_DS_REPLAY_SAN and TF_DS_REPLAY_CONTROLS before its sections run, and a controls-only run is held to its own line counts" {
    const src = @embedFile("kernels_replay_main.zig");
    const at = std.mem.indexOf(u8, src, "pub fn main(init: std.process.Init) !u8 {") orelse return error.TestUnexpectedResult;
    const body = src[at..];
    const san = std.mem.indexOf(u8, body, "std.c.getenv(\"TF_DS_REPLAY_SAN\")") orelse return error.TestUnexpectedResult;
    const ctl = std.mem.indexOf(u8, body, "std.c.getenv(\"TF_DS_REPLAY_CONTROLS\")") orelse return error.TestUnexpectedResult;
    // the sections run through runSections (an error in one is a FAILED line and the rest still run), called after both
    const loop = std.mem.indexOf(u8, body, "const ran = runSections(sections, &r, &prng, only);") orelse return error.TestUnexpectedResult;
    try std.testing.expect(san < ctl and ctl < loop);
    const rs = std.mem.indexOf(u8, src, "fn runSections(") orelse return error.TestUnexpectedResult;
    const rbody = src[rs .. rs + (std.mem.indexOf(u8, src[rs..], "\n}\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expect(std.mem.indexOf(u8, rbody, "inline for (table) |sec| if (only == null or listed(only.?, sec[0])) {") != null);
    try std.testing.expect(std.mem.indexOf(u8, rbody, "else |err| {") != null and std.mem.indexOf(u8, rbody, " FAILED: error.") != null);
    try std.testing.expect(std.mem.indexOf(u8, rbody, "try sec[1](") == null);
    // the sanitizer mode shortens the soaks; the soaks read the variable, not a constant
    try std.testing.expect(std.mem.indexOf(u8, body[san..ctl], "soak_launches = 2;") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "\nvar soak_launches: usize = 200;") != null);
    // a section without a controls-only part refuses the run (exit 2) before anything runs
    const refuse = std.mem.indexOf(u8, body[ctl..loop], "return 2;") orelse return error.TestUnexpectedResult;
    try std.testing.expect(refuse > 0);
    // the count a section is held to is its controls-only count under the mode
    try std.testing.expect(std.mem.indexOf(u8, rbody, "if (controls_only) controlsLines(sec[0]) else sec[2]") != null);
    // pf_run128 has the part and leaves its normal lines out under it
    try std.testing.expect(std.mem.indexOf(u8, src, ".{ \"pf_run128\", 5 },") != null);
    const pf = std.mem.indexOf(u8, src, "\nfn pfRun128(") orelse return error.TestUnexpectedResult;
    const pb = src[pf .. pf + (std.mem.indexOf(u8, src[pf..], "\n}\n") orelse return error.TestUnexpectedResult)];
    const gate = std.mem.indexOf(u8, pb, "if (!controls_only) {") orelse return error.TestUnexpectedResult;
    const rows = std.mem.indexOf(u8, pb, "on the 128-byte-run tiles against the baseline's") orelse return error.TestUnexpectedResult;
    const ctrl = std.mem.indexOf(u8, pb, "pf_run128 control: gate / up's pack with one x value") orelse return error.TestUnexpectedResult;
    try std.testing.expect(gate < rows and rows < ctrl);
}

/// A control's pair of inputs over `src` (bf16 rows, `row_bytes` each): copies whose row 0 is finite (a word with exponent bits >= 0xB0 -- NaN, inf,
/// or so large its square overflows -- becomes 1.0) and whose word 0 is 1.0 in `clean` and 12.0 in `changed`, everything else equal. A random input
/// that carries specials norms row 0 to the same all-NaN row with or without the change, and a control on it cannot see the change.
pub fn finiteControl(a: std.mem.Allocator, src: []const u8, row_bytes: usize) !struct { clean: []u8, changed: []u8 } {
    const clean = try a.dupe(u8, src);
    errdefer a.free(clean);
    const changed = try a.dupe(u8, src);
    errdefer a.free(changed);
    for ([_][]u8{ clean, changed }) |buf| {
        const w = std.mem.bytesAsSlice(u16, buf[0..row_bytes]);
        for (w) |*x| {
            if (((x.* >> 7) & 0xff) >= 0xb0) x.* = 0x3f80;
        }
    }
    std.mem.bytesAsSlice(u16, clean)[0] = 0x3f80; // 1.0
    std.mem.bytesAsSlice(u16, changed)[0] = 0x4140; // 12.0
    return .{ .clean = clean, .changed = changed };
}

test "a control's pair of inputs: row 0 finite with word 0 the only difference, so a changed value is seen in the row's output (a row holding NaN / inf / huge words normed to the same all-NaN row hides it)" {
    const gpa = std.testing.allocator;
    const d: usize = 64;
    const src = try gpa.alloc(u16, 2 * d);
    defer gpa.free(src);
    for (src, 0..) |*w, i| w.* = 0x3f80 + @as(u16, @intCast(i % 7)); // normal values
    src[5] = 0x7fc0; // NaN
    src[9] = 0xff80; // -inf
    src[11] = 0x7f7f; // the largest finite: its square overflows
    src[d + 3] = 0x7fc0; // row 1 keeps its own words: the control reads row 0 only
    const pair = try finiteControl(gpa, std.mem.sliceAsBytes(src), d * 2);
    defer gpa.free(pair.clean);
    defer gpa.free(pair.changed);
    const c = std.mem.bytesAsSlice(u16, pair.clean);
    const x = std.mem.bytesAsSlice(u16, pair.changed);
    var ss_c: f32 = 0;
    var ss_x: f32 = 0;
    for (0..d) |i| {
        const vc: f32 = @bitCast(@as(u32, c[i]) << 16);
        const vx: f32 = @bitCast(@as(u32, x[i]) << 16);
        ss_c += vc * vc;
        ss_x += vx * vx;
        if (i != 0) try std.testing.expectEqual(c[i], x[i]);
    }
    try std.testing.expect(std.math.isFinite(ss_c) and std.math.isFinite(ss_x));
    try std.testing.expect(ss_c != ss_x); // the rms differs, so every output of row 0 differs
    try std.testing.expect(c[0] != x[0]);
    try std.testing.expectEqual(src[d + 3], c[d + 3]); // later rows untouched
    // the instrument that failed: the same changed word over the raw row, whose sum of squares is not a number either way
    var raw: f32 = 0;
    for (src[0..d]) |w| raw += @as(f32, @bitCast(@as(u32, w) << 16)) * @as(f32, @bitCast(@as(u32, w) << 16));
    try std.testing.expect(!std.math.isFinite(raw));
}
