pub const Math = struct { pair_sum: bool, qhc: bool, owner_gather: bool, idx_shard: bool, idx_rb: usize, negzero: bool, replay_floor: bool, owner_yx: bool, cat_partner: bool, front_owned: bool, ffn_codes: bool, pack_codes: bool, engram_owner_sum: bool, boundary_owned: bool, cat_codes: bool, pf_split: []const u8, woa_w8: bool, quant_once: bool, pf_gemm5: bool, group_tiles: bool, topk_tiles: bool, idx_proj_owned: bool, bf16_rows: bool, pf_run128: bool, pass_rows: usize, pass_from: usize, pass_tail: bool, decode_fuse: bool, attn_u: bool, idx_u: bool, mma_sf: bool, mma_sf_exact: bool, roci: bool = false, roci_collect: bool = false, idx_v: bool = false, gemm_w16: bool = false, assemble_quant: bool = false, attn_epilogue: bool = false, hc_stride: bool = false, hc_fused: bool = false, gemm_r2: bool = false, gemm_halves: bool = false, pf_raster: u32 = 0, attn_epi_fused: bool = false, qgate_canary: u8 = 0, hc_2s: bool = false };
    const base = try mathBase(buf, m);
        }
    // the canary in effect (TF_DS_QGATE_CANARY: the head's 1, the body's 2 or 3), after the other optional switches, as the
    // head's 1 always was: a state kept under one canary is never restored under another
    if (m.qgate_canary != 0) {
        const x = try std.fmt.bufPrint(buf[n..], " qgate_canary={d}", .{m.qgate_canary});
        n += x.len;
    }

fn mathBase(buf: []u8, m: Math) ![]const u8 {
        @intFromBool(set(env, "TF_DS_ENGRAM_PACKED")),  @intFromBool(false), // (the owned records do not exist: the name keeps its text)