/// What follows the 64 bytes in version 3: the second kernel word and the decode defaults' state.
pub const HeaderTail = extern struct {
    kern2: u32, // the kernel switches beyond kern's 32 bits (prompt2d.kern2Asked): HC_2S, DRAFT_VOCAB
    speed: u32, // speedWord: the state IN EFFECT (not asked) of the decode defaults that carry no kernel-word bit
};

comptime {
}

/// The decode defaults that carry no kernel-word bit, as the run has them in effect (a switch the kit or a rank could not
/// run is off here, whatever was asked).
pub const Speed = struct {
    twin: u2, // TF_DS_ATTN_TWIN: 0 u, 1 w, 2 x, 3 z (tri_attn.Twin's order)
    rope_w: bool, // TF_DS_ROPE_W
    merge_cols: usize, // TF_DS_MERGE_D in effect: the columns a program (128, 64, 16) or 0 (off)
    group2: bool, // TF_DS_DEC_GROUP2
    expert_exit: bool, // TF_DS_EXPERT_EXIT
    pack_epilogue: bool, // TF_DS_PACK_EPILOGUE
    hc_2s: bool, // TF_DS_HC_2S
};

/// The speed word's bits: 0-1 the twin, 2 ROPE_W, 3-4 MERGE_D (0 off, 1 = 128 columns, 2 = 64, 3 = 16), 5 DEC_GROUP2,
pub fn speedWord(s: Speed) u32 {
    var x: u32 = s.twin;
    if (s.rope_w) x |= 1 << 2;
    x |= @as(u32, switch (s.merge_cols) {
        128 => 1,
        64 => 2,
        16 => 3,
        else => 0,
    }) << 3;
    if (s.group2) x |= 1 << 5;
    if (s.expert_exit) x |= 1 << 6;
    if (s.pack_epilogue) x |= 1 << 7;
    if (s.hc_2s) x |= 1 << 8;
    return x;
}

/// Bytes before the first record.
pub const header_bytes = @sizeOf(Header) + @sizeOf(HeaderTail);

        try s.w.writeAll(std.mem.asBytes(&t));
        s.bytes += header_bytes;
    try testing.expectEqual(@as(usize, header_bytes + 3 * (prefix_bytes + 24)), bytes.len);
    var off: usize = header_bytes;
    var off: usize = header_bytes;
    one.merge_cols = 16;
    try testing.expectEqual(@as(u32, 3 << 3), speedWord(one));
    one.merge_cols = 32; // a width the word has no code for reads as off
    try testing.expectEqual(@as(u32, 0), speedWord(one));

test "the speed word's bit 9 is TF_DS_QKV_PLAIN in effect, and nothing else moves with it" {
    var s: Speed = .{ .twin = 0, .rope_w = false, .merge_cols = 0, .group2 = false, .expert_exit = false, .pack_epilogue = false, .hc_2s = false, .qkv_plain = true };
    try testing.expectEqual(@as(u32, 1 << 9), speedWord(s));
    s.qkv_plain = false;
    try testing.expectEqual(@as(u32, 0), speedWord(s));
    // T3's defaults with QKV_PLAIN on (T4's): 0x1EF plus bit 9
    try testing.expectEqual(@as(u32, 0x3EF), speedWord(.{ .twin = 3, .rope_w = true, .merge_cols = 128, .group2 = true, .expert_exit = true, .pack_epilogue = true, .hc_2s = true, .qkv_plain = true }));
    try testing.expectEqual(@as(usize, 8), speed_names.len);
}