/// prompt.replay_floor, the kernel switches: pf_split as TF_DS_PF_SPLIT's text). pair_sum, owner_gather and negzero
/// are no longer switches: they keep the values the shipped settings gave them, so a store keeps its name. Every kernel
/// switch a prompt pass computes with is one, same-bytes ones too (native.zig's test ties prompt2d.kern_names to it).
pub const Math = struct { pair_sum: bool, qhc: bool, owner_gather: bool, idx_shard: bool, idx_rb: usize, negzero: bool, replay_floor: bool, owner_yx: bool, cat_partner: bool, front_owned: bool, ffn_codes: bool, pack_codes: bool, engram_owner_sum: bool, boundary_owned: bool, cat_codes: bool, pf_split: []const u8, woa_w8: bool, quant_once: bool, pf_gemm5: bool, group_tiles: bool, topk_tiles: bool, idx_proj_owned: bool, bf16_rows: bool, pf_run128: bool, pass_rows: usize, pass_from: usize, pass_tail: bool, decode_fuse: bool, attn_u: bool, idx_u: bool, mma_sf: bool, mma_sf_exact: bool, roci: bool = false, roci_collect: bool = false, idx_v: bool = false, gemm_w16: bool = false, assemble_quant: bool = false, attn_epilogue: bool = false, hc_stride: bool = false, hc_fused: bool = false, gemm_r2: bool = false, gemm_halves: bool = false, pf_raster: u32 = 0, attn_epi_fused: bool = false, qgate_canary: u8 = 0, hc_2s: bool = false };
    math: [768]u8 = undefined,
/// The math switches in effect as the store name's text (`m`'s every field, its name=value in Math's order).
pub fn mathText(buf: []u8, m: Math) ![]const u8 {
    const base = try mathBase(buf, m);
    // the switches below name themselves only when they are on: the names of states stored without them stay as they were
    var n = base.len;
    inline for (optional) |o| {
        if (@field(m, o[1])) {
            const x = try std.fmt.bufPrint(buf[n..], " {s}=1", .{o[0]});
            n += x.len;
        }
    }
    // the canary in effect (TF_DS_QGATE_CANARY: the head's 1, the body's 2 or 3), after the other optional switches, as the
    // head's 1 always was: a state kept under one canary is never restored under another
    if (m.qgate_canary != 0) {
        const x = try std.fmt.bufPrint(buf[n..], " qgate_canary={d}", .{m.qgate_canary});
        n += x.len;
    }
    if (m.pf_raster != 0) {
        const x = try std.fmt.bufPrint(buf[n..], " pf_raster={d}", .{m.pf_raster});
        n += x.len;
    }
    return buf[0..n];
}

/// The switches whose text joins the math's only when they are on: (the text's name, Math's field), in the order they join.
const optional = [_][2][]const u8{
    .{ "roci", "roci" },
    .{ "roci_collect", "roci_collect" },
    .{ "idx_v", "idx_v" },
    .{ "gemm_w16", "gemm_w16" },
    .{ "assemble_quant", "assemble_quant" },
    .{ "attn_epilogue", "attn_epilogue" },
    .{ "hc_stride", "hc_stride" },
    .{ "hc_fused", "hc_fused" },
    .{ "attn_epi_fused", "attn_epi_fused" },
    .{ "hc_2s", "hc_2s" },
    .{ "gemm_r2", "gemm_r2" },
};

fn mathBase(buf: []u8, m: Math) ![]const u8 {
    return std.fmt.bufPrint(buf, "pair_sum={d} qhc={d} owner_gather={d} idx_shard={d} idx_rb={d} negzero={d} replay_floor={d} owner_yx={d} cat_partner={d} front_owned={d} ffn_codes={d} pack_codes={d} engram_owner_sum={d} boundary_owned={d} cat_codes={d} pf_split={s} woa_w8={d} quant_once={d} pf_gemm5={d} group_tiles={d} topk_tiles={d} idx_proj_owned={d} bf16_rows={d} pf_run128={d} pass_rows={d} pass_from={d} pass_tail={d} decode_fuse={d} attn_u={d} idx_u={d} mma_sf={d} mma_sf_exact={d}", .{
        @intFromBool(m.pair_sum),    @intFromBool(m.qhc),         @intFromBool(m.owner_gather), @intFromBool(m.idx_shard),   m.idx_rb,                         @intFromBool(m.negzero),        @intFromBool(m.replay_floor), @intFromBool(m.owner_yx),
        @intFromBool(m.cat_partner), @intFromBool(m.front_owned), @intFromBool(m.ffn_codes),    @intFromBool(m.pack_codes),  @intFromBool(m.engram_owner_sum), @intFromBool(m.boundary_owned), @intFromBool(m.cat_codes),    m.pf_split,
        @intFromBool(m.woa_w8),      @intFromBool(m.quant_once),  @intFromBool(m.pf_gemm5),     @intFromBool(m.group_tiles), @intFromBool(m.topk_tiles),       @intFromBool(m.idx_proj_owned), @intFromBool(m.bf16_rows),    @intFromBool(m.pf_run128),
        m.pass_rows,                 m.pass_from,                 @intFromBool(m.pass_tail),    @intFromBool(m.decode_fuse), @intFromBool(m.attn_u),           @intFromBool(m.idx_u),          @intFromBool(m.mma_sf),       @intFromBool(m.mma_sf_exact),
    });
}

        @intFromBool(set(env, "TF_DS_ENGRAM_PACKED")),  @intFromBool(false), // (the owned records do not exist: the name keeps its text)
    const math = try mathText(&t.math, m);