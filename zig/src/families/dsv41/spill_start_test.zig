const math: start.Math = .{ .pair_sum = true, .qhc = true, .owner_gather = true, .idx_shard = true, .idx_rb = 2, .negzero = false, .replay_floor = true, .owner_yx = true, .cat_partner = true, .front_owned = true, .ffn_codes = true, .pack_codes = true, .engram_owner_sum = true, .boundary_owned = true, .cat_codes = true, .pf_split = "C7", .woa_w8 = true, .quant_once = true, .pf_gemm5 = true, .group_tiles = true, .topk_tiles = true, .idx_proj_owned = true, .bf16_rows = true, .pf_run128 = true, .pass_rows = 4096, .pass_from = 8192, .pass_tail = true, .decode_fuse = true, .attn_u = true, .idx_u = true, .mma_sf = true, .mma_sf_exact = true, .roci = true, .roci_collect = true, .idx_v = true, .gemm_w16 = true, .assemble_quant = true, .attn_epilogue = true, .hc_stride = true, .hc_fused = true, .attn_epi_fused = true, .gemm_r2 = true, .gemm_halves = true, .qgate_canary = 1, .hc_2s = true, .pf_raster = 8 };
test "the math text holds every switch at the widest TF_DS_IDX_RB" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    var widest = math;
    widest.idx_rb = std.math.maxInt(usize);
    var texts: start.Texts = .{};
    const c = try start.common(io, Env{ .pairs = &recipe_env }, facts(&tmp), widest, &texts);
    try std.testing.expect(std.mem.indexOf(u8, c.math, " engram_owner_sum=1 boundary_owned=1 cat_codes=1 ") != null);
    try std.testing.expect(std.mem.indexOf(u8, c.math, "idx_rb=18446744073709551615 ") != null);
}

    try std.testing.expectEqualStrings("packed=1 owned=0 instream=0 aio=1", c.engram_switches);
    try std.testing.expectEqualStrings("packed=1 owned=0 instream=0 aio=0", odd.engram_switches);
    try std.testing.expectEqualStrings("pair_sum=1 qhc=1 owner_gather=1 idx_shard=1 idx_rb=2 negzero=0 replay_floor=1 owner_yx=1 cat_partner=1 front_owned=1 ffn_codes=1 pack_codes=1 engram_owner_sum=1 boundary_owned=1 cat_codes=1 pf_split=C7 woa_w8=1 quant_once=1 pf_gemm5=1 group_tiles=1 topk_tiles=1 idx_proj_owned=1 bf16_rows=1 pf_run128=1 pass_rows=4096 pass_from=8192 pass_tail=1 decode_fuse=1 attn_u=1 idx_u=1 mma_sf=1 mma_sf_exact=1 roci=1 roci_collect=1 idx_v=1 gemm_w16=1 assemble_quant=1 attn_epilogue=1 hc_stride=1 hc_fused=1 attn_epi_fused=1 hc_2s=1 gemm_r2=1 gemm_halves=1 qgate_canary=1 pf_raster=8", c.math);
    // every field of the math in its text
    inline for (@typeInfo(start.Math).@"struct".field_names) |f| try std.testing.expect(std.mem.indexOf(u8, c.math, f ++ "=") != null);
    // every switch at its longest value fits the text
    var longest = math;
    longest.idx_rb = std.math.maxInt(usize);
    var texts3: start.Texts = .{};
    _ = try start.common(io, Env{ .pairs = &recipe_env }, facts(&tmp), longest, &texts3);

test "the math text names TF_DS_ROCI, TF_DS_ROCI_COLLECT, TF_DS_IDX_V, TF_DS_GEMM_W16, TF_DS_ASSEMBLE_QUANT, TF_DS_ATTN_EPILOGUE, TF_DS_HC_STRIDE, TF_DS_HC_FUSED, TF_DS_ATTN_EPI_FUSED, TF_DS_HC_2S, TF_DS_GEMM_R2 and TF_DS_GEMM_HALVES and TF_DS_GEMM_HALVES only when they are on: the names of states stored without it do not change" {
    var buf: [1024]u8 = undefined;
    var off = math;
    off.roci = false;
    off.roci_collect = false;
    off.idx_v = false;
    off.gemm_w16 = false;
    off.assemble_quant = false;
    off.attn_epilogue = false;
    off.hc_stride = false;
    off.hc_fused = false;
    off.attn_epi_fused = false;
    off.hc_2s = false;
    off.gemm_r2 = false;
    off.qgate_canary = 0;
    off.pf_raster = 0;
    const text = try start.mathText(&buf, off);
    try std.testing.expect(std.mem.indexOf(u8, text, "roci") == null);
    try std.testing.expect(std.mem.endsWith(u8, text, "mma_sf_exact=1"));
    var buf2: [1024]u8 = undefined;
    const on = try start.mathText(&buf2, math);
    try std.testing.expectEqualStrings(text, on[0..text.len]);
    try std.testing.expectEqualStrings(" roci=1 roci_collect=1 idx_v=1 gemm_w16=1 assemble_quant=1 attn_epilogue=1 hc_stride=1 hc_fused=1 attn_epi_fused=1 hc_2s=1 gemm_r2=1 gemm_halves=1 qgate_canary=1 pf_raster=8", on[text.len..]);
}

test "the math text names TF_DS_PF_RASTER with its group only when it is on, after the other optional switches" {
    var off_buf: [1024]u8 = undefined;
    var on_buf: [1024]u8 = undefined;
    var m = math;
    m.pf_raster = 0;
    const off = try start.mathText(&off_buf, m);
    try std.testing.expect(std.mem.indexOf(u8, off, "pf_raster") == null);
    for ([_]u32{ 2, 4, 8, 16 }) |g| {
        m.pf_raster = g;
        const on = try start.mathText(&on_buf, m);
        try std.testing.expectEqualStrings(off, on[0..off.len]);
        var want: [16]u8 = undefined;
        try std.testing.expectEqualStrings(try std.fmt.bufPrint(&want, " pf_raster={d}", .{g}), on[off.len..]);
    }
}
    on.qgate_canary = 1;

test "the canary in the tier's math is a number: the head's 1 keeps its old name, the body's 2 and 3 name themselves, and none is the same text as before" {
    var base_buf: [1024]u8 = undefined;
    var m = math;
    m.pf_raster = 0; // (its group comes last: the canary joins before it)
    m.qgate_canary = 0;
    const none = try start.mathText(&base_buf, m);
    try std.testing.expect(std.mem.indexOf(u8, none, "qgate_canary") == null);
    for ([_]u8{ 1, 2, 3 }) |c| {
        var buf: [1024]u8 = undefined;
        m.qgate_canary = c;
        const t = try start.mathText(&buf, m);
        var want: [32]u8 = undefined;
        // after the other optional switches, before pf_raster
        try std.testing.expect(std.mem.indexOf(u8, t, try std.fmt.bufPrint(&want, " qgate_canary={d}", .{c})) != null);
        try std.testing.expectEqualStrings(none, t[0..none.len]);
    }
}
