    const rows = std.mem.indexOf(u8, pb, "on the 128-byte-run tiles against the baseline's") orelse return error.TestUnexpectedResult;

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
