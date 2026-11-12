//! DeepSeek's own checkpoint (config.Format.native) on the GPU: the kernels of zig/kernels/cuda/dsv41/native.cu
//! (cuda.kernels.dsv41_native) for its linears and experts, as DeepSeek's reference computes them (inference/model.py
//! `linear`, `Expert`, `MoE`; kernel.py `act_quant`, `fp8_gemm`, `fp4_gemm`): each activation rounded to bf16 and
//! quantized to E4M3 with a power-of-two scale per 32 values, FP8 weights with a UE8M0 scale per 32 x 32 block, MXFP4
//! experts with one per 32 values of a row, every 32-wide K block's sum scaled into fp32. A row's bits never depend on
//! the other rows of a launch, and a layer's K split depends on its shape alone, so drafted, concurrent and resumed rows
//! get a row's own bits at any row count.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const exl3_linear = @import("exl3_linear.zig");
const ops_chunk = @import("ops.zig"); // (l2_chunk_bytes: TF_DS_L2_CHUNK_KIB)

pub const DType = exl3_linear.DType;

/// native.cu's MxArgs, field for field (C layout).
pub const MxArgs = extern struct {
    q: u64 = 0,
    s: u64 = 0,
    rows: u64 = 0,
    ldq: i64 = 0,
    lds: i64 = 0,
    w: u64 = 0,
    ws: u64 = 0,
    ldw: i64 = 0,
    ldws: i64 = 0,
    fmt: c_int = 0,
    M: c_int = 0,
    N: c_int = 0,
    K: c_int = 0,
    splits: c_int = 1,
    kblocks: c_int = 0,
    z: u64 = 0,
    y: u64 = 0,
    ldy: i64 = 0,
    y_dtype: c_int = 0,
    accumulate: c_int = 0,
    raster: c_int = 0, // pfRaster's group of row tiles for the tile GEMMs (native.cu pf_tile_of); 0: row tiles fastest
};

/// native.cu's GuArgs.
pub const GuArgs = extern struct {
    q: u64 = 0,
    s: u64 = 0,
    ldq: i64 = 0,
    lds: i64 = 0,
    ent: u64 = 0,
    erow: u64 = 0,
    off: u64 = 0,
    act: u64 = 0,
    nact: u64 = 0,
    w1: u64 = 0,
    w3: u64 = 0,
    s1: u64 = 0,
    s3: u64 = 0,
    ldw: i64 = 0,
    ldws: i64 = 0,
    fmt: c_int = 1,
    D: c_int = 0,
    shared: c_int = -1,
    sldw: i64 = 0,
    sldws: i64 = 0,
    owner: u64 = 0,
    n_tiles: c_int = 0,
    rest_tiles: c_int = 0,
    wts: u64 = 0,
    limit: f32 = 0,
    h: u64 = 0,
    ldh: i64 = 0,
    P: c_int = 0,
    splits: c_int = 1,
    kblocks: c_int = 0,
    z: u64 = 0,
    early_exit: c_int = 0, // Track C (TF_DS_EXPERT_EXIT)
};

/// native.cu's DnArgs.
pub const DnArgs = extern struct {
    q: u64 = 0,
    s: u64 = 0,
    ldq: i64 = 0,
    lds: i64 = 0,
    ent: u64 = 0,
    off: u64 = 0,
    act: u64 = 0,
    nact: u64 = 0,
    w2: u64 = 0,
    s2: u64 = 0,
    ldw: i64 = 0,
    ldws: i64 = 0,
    fmt: c_int = 1,
    shared: c_int = -1,
    sldw: i64 = 0,
    sldws: i64 = 0,
    I: c_int = 0,
    N: c_int = 0,
    round_bf16: c_int = 0,
    out: u64 = 0,
    ldo: i64 = 0,
    splits: c_int = 1,
    kblocks: c_int = 0,
    z: u64 = 0,
    P: c_int = 0,
    early_exit: c_int = 0, // Track C (TF_DS_EXPERT_EXIT)
};

/// native.cu's CombArgs: tf_dsn_combine's arguments for a down launch that sums the slots itself (TF_DS_DECODE_FUSE).
pub const CombArgs = extern struct {
    pick: u64 = 0,
    R: c_int = 0,
    S: c_int = 0,
    E: c_int = 0,
    accumulate: c_int = 0,
    y: u64 = 0,
    ldy: i64 = 0,
    cnt: u64 = 0,
};

/// native.cu's BfArgs.
pub const BfArgs = extern struct {
    x: u64 = 0,
    ldx: i64 = 0,
    gx: i64 = 0,
    w: u64 = 0,
    ldw: i64 = 0,
    gw: i64 = 0,
    M: c_int = 0,
    N: c_int = 0,
    K: c_int = 0,
    y: u64 = 0,
    ldy: i64 = 0,
    gy: i64 = 0,
    y_dtype: c_int = 0,
    scale: f32 = 1,
    round_first: c_int = 0,
    splits: c_int = 1,
    ksteps: c_int = 0,
    z: u64 = 0,
    raster: c_int = 0, // as MxArgs.raster, for tf_dsn_pf_bf16_kernel
};

/// native.cu's W8Args (tf_dsn_pf_bf16w8: wo_a's groups of a prompt chunk on their FP8 rows).
pub const W8Args = extern struct {
    q: u64 = 0,
    s: u64 = 0,
    ldq: i64 = 0,
    lds: i64 = 0,
    w: u64 = 0,
    ws: u64 = 0,
    ldw: i64 = 0,
    ldws: i64 = 0,
    M: c_int = 0,
    N: c_int = 0,
    K: c_int = 0,
    y_dtype: c_int = 0,
    y: u64 = 0,
    ldy: i64 = 0,
    raster: c_int = 0, // as MxArgs.raster, for tf_dsn_pf_bf16w8_kernel and its 16-warp twin
    gq: i64 = 0, // TF_DS_WOA_MERGE: group z's operands are the base ones + z times these byte strides (woaGroupStrides); 0: a one-group launch
    gs: i64 = 0,
    gw: i64 = 0,
    gws: i64 = 0,
    gy: i64 = 0,
};

/// native.cu's DecArgs (the decode tensor-core kernels' dense linears).
pub const DecArgs = extern struct {
    q: u64 = 0,
    s: u64 = 0,
    rows: u64 = 0,
    ldq: i64 = 0,
    lds: i64 = 0,
    w: u64 = 0,
    ws: u64 = 0,
    ldw: i64 = 0,
    ldws: i64 = 0,
    fmt: c_int = 0,
    M: c_int = 0,
    N: c_int = 0,
    K: c_int = 0,
    ks: c_int = 1,
    cs: c_int = 1,
    z: u64 = 0,
    cnt: u64 = 0,
    y: u64 = 0,
    ldy: i64 = 0,
    y_dtype: c_int = 0,
    accumulate: c_int = 0,
    scale: f32 = 1,
    round_first: c_int = 0,
    gn: c_int = 0,
    gq: i64 = 0,
    gs: i64 = 0,
};

/// native.cu's DecMulti: up to four dense linears of one format and row count in one launch.
pub const DecMulti = extern struct {
    p: [4]DecArgs = @splat(.{}),
    n: c_int = 0,
    cta0: [5]c_int = @splat(0),
};

comptime {
    std.debug.assert(@sizeOf(DecMulti) == 4 * 168 + 24 and @offsetOf(DecMulti, "cta0") == 676);
    std.debug.assert(@sizeOf(DecArgs) == 168 and @offsetOf(DecArgs, "z") == 96 and @offsetOf(DecArgs, "y") == 112 and @offsetOf(DecArgs, "gq") == 152);
}

/// The decode tensor-core kernels' shared memory (native.cu namespace dec: kRbD / kNsD for the dense linears, kRbE /
/// kNsE for the experts, slotBytes, warpBytes): a CTA of dec_warps warps, each warp its ring of stages or its partials.
pub const DecCfg = struct { rowb: usize, stages: usize };
pub const dec_dense_cfg: DecCfg = .{ .rowb = 64, .stages = 4 };
pub const dec_experts_cfg: DecCfg = .{ .rowb = 64, .stages = 2 }; // down
pub const dec_gu_cfg: DecCfg = .{ .rowb = 128, .stages = 2 }; // gate / up
pub const dec_warps = 8;
fn decWarpBytes(cfg: DecCfg, rg: usize) usize {
    return @max(cfg.stages * (16 * cfg.rowb + 128), rg * 4 * 32 * 4);
}
fn decShared(cfg: DecCfg, rg: usize) u32 {
    return @intCast(dec_warps * decWarpBytes(cfg, rg));
}
/// The row groups (of 8) a launch's template holds for m rows: 1, 2, 4, 8 (index 0..3); past 64 rows the experts' kernels
/// run passes of 64 (the same arithmetic a row).
fn decRg(m: usize) usize {
    return if (m <= 8) 0 else if (m <= 16) 1 else if (m <= 32) 2 else 3;
}
const dec_counters = 16384;

/// The column groups a down launch with the routed sum in it (TF_DS_DECODE_FUSE) counts in at most: a count each.
const comb_counters = 4096;

/// Whether a decode down launch of gx column groups sums the routed slots itself (TF_DS_DECODE_FUSE: the switch, and
/// a count for each group), else tf_dsn_combine after it.
pub fn decSumIn(fuse: bool, gx: usize) bool {
    return fuse and gx <= comb_counters;
}

comptime {
    // the layouts native.cu was compiled with (pointers and int64 8 bytes, ints and floats 4, C padding)
    std.debug.assert(@sizeOf(MxArgs) == 136 and @offsetOf(MxArgs, "z") == 96 and @offsetOf(MxArgs, "raster") == 128);
    std.debug.assert(@offsetOf(GuArgs, "sldw") == 136 and @offsetOf(GuArgs, "owner") == 152 and @offsetOf(GuArgs, "z") == 216 and @offsetOf(GuArgs, "early_exit") == 224 and @sizeOf(GuArgs) == 232);
    std.debug.assert(@offsetOf(DnArgs, "sldw") == 104 and @offsetOf(DnArgs, "out") == 136 and @offsetOf(DnArgs, "z") == 160 and @offsetOf(DnArgs, "early_exit") == 172 and @sizeOf(DnArgs) == 176);
    std.debug.assert(@offsetOf(BfArgs, "y") == 64 and @offsetOf(BfArgs, "z") == 112 and @offsetOf(BfArgs, "raster") == 120 and @sizeOf(BfArgs) == 128);
    std.debug.assert(@sizeOf(W8Args) == 144 and @offsetOf(W8Args, "M") == 64 and @offsetOf(W8Args, "y") == 80 and @offsetOf(W8Args, "raster") == 96 and @offsetOf(W8Args, "gq") == 104 and @offsetOf(W8Args, "gy") == 136);
    std.debug.assert(@sizeOf(CombArgs) == 48 and @offsetOf(CombArgs, "y") == 24 and @offsetOf(CombArgs, "cnt") == 40);
}

/// The kernels of cuda.kernels.dsv41_native.
pub const Kernels = struct {
    module: cuda.Module,
    act_quant: cuda.Function,
    /// tf_dsn_act_quant_vec_kernel (TF_DS_AQ_VEC): act_quant on four lanes a 32-group, 16-byte loads; null when the image lacks it (an older image: the baseline kernel runs).
    act_quant_vec: ?cuda.Function = null,
    // TF_DS_ATTN_EPILOGUE: the attention output's inverse RoPE and wo_a's quant in one launch
    attn_epilogue: cuda.Function,
    gemm: [4]cuda.Function, // row tiles of 1, 2, 4, 8 m16 tiles
    reduce: cuda.Function,
    count: cuda.Function,
    scan: cuda.Function,
    place: cuda.Function,
    gu: [3]cuda.Function, // 1, 2, 4
    dn: [3]cuda.Function,
    gu_finish: cuda.Function,
    dn_finish: cuda.Function,
    gu_gemv: cuda.Function,
    dn_gemv: cuda.Function,
    gemv: cuda.Function,
    mx_sab: cuda.Function, // the decode MMA with the operands swapped (16 weight rows by up to 8 activation rows)
    gu_sab: cuda.Function,
    dn_sab: cuda.Function,
    sab: bool, // decode launches on the swap-AB MMA (TF_DS_DECODE_SAB=1; else the GEMV kernels, faster at a round's
    // 1-3 rows on GB10: forward 45.7 against 49.9 ms a round, measured 2026-10-17)
    // the decode tensor-core kernels (tf_dsn_dec_*): [kind][row groups 1, 2, 4, 8], kinds fp8, fp4, bf16 (DecArgs), gate /
    // up (GuArgs), down (DnArgs); the cross-CTA split's counters (int zeros, each launch leaves them zero)
    dec: [5][4]cuda.Function,
    dec_group: cuda.Function, // a decode launch's grouping in one CTA
    nact_log: ?cuda.Function, // TF_DS_NACT_LOG's record kernel (a kit built before it: none)
    dec_group2: cuda.Function, // the same grouping by a scan of one expert a thread and a ranked placement (TF_DS_DEC_GROUP2)
    group2: bool, // TF_DS_DEC_GROUP2 (on unless 0): a decode launch's grouping runs dec_group2 (the same bytes; each rank reads its own)
    dec_quant: cuda.Function, // act_quant for decode launches (a programmatic dependent launch)
    dec_l2pf: cuda.Function, // the active experts' down weights into L2 (beside the 2D exchange)
    dec_on: bool, // TF_DS_DECODE_DEC=0: the GEMV kernels instead (the reference path, measured slower)
    // TF_DS_DEC_PDL=0: the decode tensor-core kernels (dense, gate / up, down) as ordinary launches, the same kernels (the
    // decode plan's G0.2 control (a)); unset or any other value: programmatic dependent launches as served
    dec_pdl: bool,
    // TF_DS_EXPERT_EXIT=1 (Track C): the decode expert kernels' grid rows past the active experts return at once
    expert_exit: bool,
    counters: cuda.DeviceBuffer,
    combine: cuda.Function,
    bf16: cuda.Function,
    bf_reduce: cuda.Function,
    to_f16: cuda.Function,
    // prompt chunks' tile GEMMs (native.cu tf_dsn_pf_*: the tile kernels' bits, operands staged through shared memory)
    pf_gemm: cuda.Function,
    pf_gemm_fold: cuda.Function,
    pf_gu: cuda.Function,
    pf_dn: cuda.Function,
    pf_tiles: cuda.Function,
    pf_bf16: cuda.Function,
    // TF_DS_PF_T64 in effect (usePfT64): the router's bf16 GEMM of a prompt chunk on a 64 x 64 tile; its kernel (null until then)
    pf_t64: bool = false,
    pf_bf16_t64: ?cuda.Function = null,
    pf_off: bool, // TF_DS_PF_TILES=0: prompt chunks on the tile kernels (the same bits)
    // TF_DS_PF_SPLIT in effect (useSplit): the split FP8 GEMMs' kernel; the configurations' kernels (null until then)
    split: PfSplit = .off,
    pf_split: [4]?cuda.Function = @splat(null),
    // wo_a on its FP8 rows (useW8: a layer the loader left without its bf16 rows, TF_DS_WOA_W8); null until then
    pf_bf16w8: ?cuda.Function = null,
    // TF_DS_GEMM_W16 in effect (useGemmW16): a prompt chunk's exact GEMMs and wo_a's bf16 rows on 16 warps a block
    // (512 threads); the kernels are loaded with the others (SfKernels.w16, pf_bf16w8_w16) and this one chooses them
    gemm_w16: bool = false,
    pf_bf16w8_w16: ?cuda.Function = null,
    // TF_DS_SPLIT64N in effect (useSplit64n): the exact arithmetic's 64-row split tile at 64 columns for N of 512 and 1,280
    split64n: bool = false,
    // TF_DS_PF_GEMM5 in effect (useGemm5): the dense FP8 GEMM without a K split at 5 stages; its kernel
    gemm5: bool = false,
    pf_gemm5: ?cuda.Function = null,
    // the transport's arena [wc[0], wc[1]) (useStaged): a dense GEMM whose fp32 y lies in it takes the _st kernels, which store
    // whole lines (write-combined host memory takes partial lines slowly); the same values
    wc: [2]u64 = .{ 0, 0 },
    st_gemm: ?cuda.Function = null,
    st_fold: ?cuda.Function = null,
    st_gemm5: ?cuda.Function = null,
    st_split: [4]?cuda.Function = @splat(null),
    // tf_dsn_combine over a row-quarter order with the doorbell rung from inside (useStaged)
    combine_door: ?cuda.Function = null,
    // TF_DS_GROUP_TILES in effect (useGroupTiles): a prompt chunk's tile list built once, by the grouping; its kernel
    group_tiles: bool = false,
    group_tiles_fn: ?cuda.Function = null,
    // TF_DS_TOPK_TILES in effect (useTopkTiles): the indexer's pruned top-k in one launch (topkTiles); its kernel
    topk_tiles: bool = false,
    topk_tiles_fn: ?cuda.Function = null,
    // TF_DS_BF16_ROWS in effect: a prompt launch of tf_dsn_bf16_gemm takes a CTA a 16-row tile (bfGridRows)
    bf16_rows: bool = false,
    // TF_DS_PF_RUN128 in effect (useRun128): a prompt chunk's MXFP4 expert tiles with their weights staged 128 bytes a
    // row (pfExpertsLaunch); their kernels
    run128: bool = false,
    pf_gu_r128: ?cuda.Function = null,
    /// TF_DS_PACK_EPILOGUE in effect: tf_dsn_sf_gu_exact_q_kernel, gate/up's launch that also writes the expert pack's codes and scales.
    gu_q: ?cuda.Function = null,
    pf_dn_r128: ?cuda.Function = null,
    // TF_DS_DECODE_FUSE in effect (useDecodeFuse): a decode round's routed sum in the down launch (decSumIn); its
    // kernels (one a row group) and their column groups' counts
    decode_fuse: bool = false,
    dec_dn_cb: ?[4]cuda.Function = null,
    comb_cnt: ?cuda.DeviceBuffer = null,
    // and a 2D node's assembly of the experts' intermediate with down's quant of it in one launch (assembleQuant)
    dec_asm_quant: ?cuda.Function = null,
    // TF_DS_QGATE_CANARY 2 or 3 in effect (useQgateBody): the body canary's kernel (qgateBody); null until then
    qgate_body_fn: ?cuda.Function = null,
    // TF_DS_ASSEMBLE_QUANT in effect (useAssembleQuant): the same assembly-and-quant launch for a prompt chunk's
    // parity experts, its kernel loaded on its own (not with TF_DS_DECODE_FUSE's routed sum)
    assemble_quant: bool = false,
    // TF_DS_MMA_SF in effect (useMmaSf): a prompt chunk's FP8 GEMMs on the block-scaled kernels (sfGemmOf, sfExperts);
    // their module, null until then
    sf: MmaSf = .off,
    sfk: ?SfKernels = null,
    // TF_DS_MMA_SF_EXACT: the GEMMs its guard keeps on the baseline kernels (sfKeep), each by its weight scales' key
    sf_kept: [sf_kept_max]u64 = undefined,
    sf_nkept: usize = 0,
    sf_proven: [sf_proven_max]u64 = undefined, // TF_DS_MMA_SF_EXACT's guard: the keys it proved, ascending
    sf_nproven: usize = 0,
    sf_seen: [3][2]u8 = @splat(.{ 255, 0 }), // the guard's scale bytes seen: FP8 dense, MXFP4, the shared expert
    sf_apart: usize = 0, // the layers whose experts the guard kept for their packing (other lengths than its reads)
    // TF_DS_GEMM_R2 in effect (useGemmR2): the exact arithmetic's split GEMMs and expert tiles on its kernels (SfKernels.r2)
    gemm_r2: bool = false,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var dec: [5][4]cuda.Function = undefined;
        const kinds = [_][]const u8{ "fp8", "fp4", "bf", "gu", "dn" };
        for (kinds, 0..) |kn, i| for (0..4) |r| {
            var nb: [64]u8 = undefined;
            const name = try std.fmt.bufPrintSentinel(&nb, "tf_dsn_dec_{s}_r{d}_kernel", .{ kn, @as(usize, 1) << @intCast(r) }, 0);
            dec[i][r] = try m.function(name);
            try dec[i][r].allowDynamicShared(decShared(if (i < 3) dec_dense_cfg else if (i == 3) dec_gu_cfg else dec_experts_cfg, @as(usize, 1) << @intCast(r)));
        };
        var counters = try cuda.DeviceBuffer.alloc(d, dec_counters * 4);
        errdefer counters.free();
        try d.check(d.api.cuMemsetD32_v2(counters.ptr, 0, dec_counters), "cuMemsetD32");
        return .{
            .dec = dec,
            .dec_group = try m.function("tf_dsn_dec_group_kernel"),
            .dec_group2 = try m.function("tf_dsn_dec_group2_kernel"),
            .nact_log = m.function("tf_dsn_nact_log_kernel") catch null,
            .group2 = if (std.c.getenv("TF_DS_DEC_GROUP2")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true,
            .dec_quant = try m.function("tf_dsn_dec_quant_kernel"),
            .dec_l2pf = try m.function("tf_dsn_dec_l2pf_kernel"),
            .dec_on = decOnEnv(),
            .dec_pdl = if (std.c.getenv("TF_DS_DEC_PDL")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true,
            .expert_exit = if (std.c.getenv("TF_DS_EXPERT_EXIT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false,
            .counters = counters,
            .module = m,
            .act_quant = try m.function("tf_dsn_act_quant_kernel"),
            .act_quant_vec = m.function("tf_dsn_act_quant_vec_kernel") catch null,
            .attn_epilogue = try m.function("tf_dsn_attn_epilogue_kernel"),
            .gemm = .{ try m.function("tf_dsn_mx_gemm1_kernel"), try m.function("tf_dsn_mx_gemm2_kernel"), try m.function("tf_dsn_mx_gemm4_kernel"), try m.function("tf_dsn_mx_gemm8_kernel") },
            .reduce = try m.function("tf_dsn_reduce_kernel"),
            .count = try m.function("tf_dsn_group_count_kernel"),
            .scan = try m.function("tf_dsn_group_scan_kernel"),
            .place = try m.function("tf_dsn_group_place_kernel"),
            .gu = .{ try m.function("tf_dsn_gu1_kernel"), try m.function("tf_dsn_gu2_kernel"), try m.function("tf_dsn_gu4_kernel") },
            .dn = .{ try m.function("tf_dsn_dn1_kernel"), try m.function("tf_dsn_dn2_kernel"), try m.function("tf_dsn_dn4_kernel") },
            .gu_finish = try m.function("tf_dsn_gu_finish_kernel"),
            .dn_finish = try m.function("tf_dsn_dn_finish_kernel"),
            .gu_gemv = try m.function("tf_dsn_gu_gemv_kernel"),
            .dn_gemv = try m.function("tf_dsn_dn_gemv_kernel"),
            .gemv = try m.function("tf_dsn_mx_gemv_kernel"),
            .mx_sab = try m.function("tf_dsn_mx_sab_kernel"),
            .gu_sab = try m.function("tf_dsn_gu_sab_kernel"),
            .dn_sab = try m.function("tf_dsn_dn_sab_kernel"),
            .sab = if (std.c.getenv("TF_DS_DECODE_SAB")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false,
            .combine = try m.function("tf_dsn_combine_kernel"),
            .bf16 = try m.function("tf_dsn_bf16_gemm_kernel"),
            .bf_reduce = try m.function("tf_dsn_bf_reduce_kernel"),
            .to_f16 = try m.function("tf_dsn_bf16_to_f16_kernel"),
            .pf_gemm = try pfFunction(m, "tf_dsn_pf_gemm_kernel", pf_gemm_smem),
            .pf_gemm_fold = try pfFunction(m, "tf_dsn_pf_gemm_fold_kernel", pf_gemm_smem),
            .pf_gu = try pfFunction(m, "tf_dsn_pf_gu_kernel", pf_gu_smem),
            .pf_dn = try pfFunction(m, "tf_dsn_pf_dn_kernel", pf_dn_smem),
            .pf_tiles = try m.function("tf_dsn_pf_tiles_kernel"),
            .pf_bf16 = try pfFunction(m, "tf_dsn_pf_bf16_kernel", pf_bf16_smem),
            .pf_off = pfOffEnv(),
        };
    }

    fn pfFunction(m: cuda.Module, name: [:0]const u8, smem: u32) !cuda.Function {
        const f = try m.function(name);
        try f.allowDynamicShared(smem);
        return f;
    }

    /// TF_DS_DECODE_FUSE in effect: tf_dsn_dec_dn_cb_r{1,2,4,8} (dec_dn's shared memory), their zeroed counts,
    /// tf_dsn_dec_asm_quant, then the switch. An error leaves it off.
    pub fn useDecodeFuse(k: *Kernels, d: *const cuda.Driver) !void {
        const aq = try k.module.function("tf_dsn_dec_asm_quant_kernel");
        var fs: [4]cuda.Function = undefined;
        for (&fs, 0..) |*f, r| {
            var nb: [64]u8 = undefined;
            f.* = try k.module.function(try std.fmt.bufPrintSentinel(&nb, "tf_dsn_dec_dn_cb_r{d}_kernel", .{@as(usize, 1) << @intCast(r)}, 0));
            try f.allowDynamicShared(decShared(dec_experts_cfg, @as(usize, 1) << @intCast(r)));
        }
        var cnt = try cuda.DeviceBuffer.alloc(d, comb_counters * 4);
        errdefer cnt.free();
        try d.check(d.api.cuMemsetD32_v2(cnt.ptr, 0, comb_counters), "cuMemsetD32");
        k.dec_dn_cb = fs;
        k.comb_cnt = cnt;
        k.dec_asm_quant = aq;
        k.decode_fuse = true;
    }

    /// TF_DS_ASSEMBLE_QUANT in effect: tf_dsn_dec_asm_quant (the parity assembly with down's quant in one launch),
    /// then the switch. An error leaves it off.
    pub fn useAssembleQuant(k: *Kernels) !void {
        k.dec_asm_quant = try k.module.function("tf_dsn_dec_asm_quant_kernel");
        k.assemble_quant = true;
    }

    /// TF_DS_PF_RUN128 in effect: tf_dsn_pf_gu_r128 and tf_dsn_pf_dn_r128, then the switch. An error leaves it off.
    pub fn useRun128(k: *Kernels) !void {
        const gu = try pfFunction(k.module, "tf_dsn_pf_gu_r128_kernel", pf_gu_r128_smem);
        const dn = try pfFunction(k.module, "tf_dsn_pf_dn_r128_kernel", pf_dn_r128_smem);
        k.pf_gu_r128 = gu;
        k.pf_dn_r128 = dn;
        k.run128 = true;
    }

    /// TF_DS_GEMM_R2 `asked` in effect: the exact arithmetic's expert tiles take the R2 kernels loaded beside the baseline
    /// (SfKernels.r2). They stage the activation rows as the baseline kernels do with a padding row's copies left out, and
    /// give down a 128-row tile, keeping every K block's MMA order and fp32 adds. Nothing loads here: an image built
    /// without both (or a GPU that did not load them, or another arithmetic than the exact form) leaves the switch off.
    pub fn useGemmR2(k: *Kernels, asked: bool) bool {
        const ok = if (k.sfk) |x| x.r2.gu != null and x.r2.dn != null else false;
        k.gemm_r2 = asked and ok and k.sf == .exact;
        return k.gemm_r2;
    }

    /// TF_DS_TOPK_TILES in effect: tf_dsn_topk_tiles, then the switch. An error leaves it off.
    pub fn useTopkTiles(k: *Kernels) !void {
        k.topk_tiles_fn = try k.module.function("tf_dsn_topk_tiles_kernel");
        k.topk_tiles = true;
    }

    /// TF_DS_QGATE_CANARY 2 or 3 (prompt.qgate_body): the body canary's kernel, for qgateBody.
    pub fn useQgateBody(k: *Kernels) !void {
        k.qgate_body_fn = try k.module.function("tf_dsn_qgate_body_kernel");
    }

    /// TF_DS_GROUP_TILES in effect: tf_dsn_group_tiles, then the switch. An error leaves it off.
    pub fn useGroupTiles(k: *Kernels) !void {
        k.group_tiles_fn = try k.module.function("tf_dsn_group_tiles_kernel");
        k.group_tiles = true;
    }

    /// The dense GEMMs' whole-line twins for a y in [lo, hi) (the transport's arena): the _st kernels of the tile, fold, 5-stage
    /// and split configurations, then the range. An error leaves the range empty.
    pub fn useStaged(k: *Kernels, lo: u64, hi: u64) !void {
        k.st_gemm = try pfFunction(k.module, "tf_dsn_pf_gemm_st_kernel", pf_gemm_smem);
        k.st_fold = try pfFunction(k.module, "tf_dsn_pf_gemm_fold_st_kernel", pf_gemm_smem);
        k.st_gemm5 = try pfFunction(k.module, "tf_dsn_pf_gemm5_st_kernel", pf_gemm5_smem);
        for (&k.st_split, split_st_kernels, split_smem) |*f, name, smem| f.* = try pfFunction(k.module, name, smem);
        if (k.sfk) |*sk| try sk.useStaged();
        // the GPU-scope release with the last arrival's system fence (door_arrive_cta<2>)
        k.combine_door = try k.module.function("tf_dsn_combine_door_g_kernel");
        k.wc = .{ lo, hi };
    }

    /// TF_DS_PF_GEMM5 in effect: tf_dsn_pf_gemm5, then the switch. An error leaves it off.
    pub fn useGemm5(k: *Kernels) !void {
        k.pf_gemm5 = try pfFunction(k.module, "tf_dsn_pf_gemm5_kernel", pf_gemm5_smem);
        k.gemm5 = true;
    }

    /// tf_dsn_pf_bf16w8 for woA8Prefill (the layers the loader left without their bf16 rows), and its 16-warp twin.
    pub fn useW8(k: *Kernels) !void {
        k.pf_bf16w8 = try pfFunction(k.module, "tf_dsn_pf_bf16w8_kernel", pf_w8_smem);
        k.pf_bf16w8_w16 = pfFunction(k.module, "tf_dsn_pf_bf16w8_w16_kernel", pf_w8_smem) catch null;
    }

    /// TF_DS_GEMM_W16 `asked` in effect: a prompt chunk's exact GEMMs and wo_a's bf16 rows launch the 16-warp (512
    /// thread) kernels loaded beside the baseline. Their tiles, stages and shared memory are the 8-warp kernels' and their
    /// bytes are theirs. Nothing loads here: an image built without the twins (or, for the block-scaled kernels, a GPU
    /// that did not load them) leaves the switch off.
    pub fn useGemmW16(k: *Kernels, asked: bool) bool {
        const sf = if (k.sfk) |x| x.w16.split64 != null and x.w16.split128 != null and x.w16.gu != null and x.w16.dn != null else true;
        k.gemm_w16 = asked and k.pf_bf16w8_w16 != null and sf;
        return k.gemm_w16;
    }

    /// TF_DS_PF_T64 `asked` in effect: tf_dsn_pf_bf16_t64 (the bf16 prompt GEMM on a 64 x 64 tile, two CTAs an SM), then the switch. An
    /// image built without the kernel, or a GPU that did not load it, leaves the switch off.
    pub fn usePfT64(k: *Kernels, asked: bool) bool {
        if (asked and k.pf_bf16_t64 == null) k.pf_bf16_t64 = pfFunction(k.module, "tf_dsn_pf_bf16_t64_kernel", pf_bf16_t64_smem) catch null;
        k.pf_t64 = asked and k.pf_bf16_t64 != null;
        return k.pf_t64;
    }

    /// TF_DS_SPLIT64N `asked` in effect: the exact arithmetic's 64-column split tile (SfKernels.split64n) is in this image and loaded.
    /// Nothing loads here: an image built without it leaves the switch off.
    pub fn useSplit64n(k: *Kernels, asked: bool) bool {
        k.split64n = asked and (if (k.sfk) |x| x.split64n != null else false);
        return k.split64n;
    }

    /// TF_DS_PF_SPLIT `sw` in effect (off: nothing changes, nothing loaded): the four configurations' kernels, then
    /// the switch. An error leaves it off.
    pub fn useSplit(k: *Kernels, sw: PfSplit) !void {
        if (sw == .off) return;
        var fs: [4]?cuda.Function = undefined;
        for (&fs, split_kernels, split_smem) |*f, name, smem| f.* = try pfFunction(k.module, name, smem);
        k.pf_split = fs;
        k.split = sw;
    }

    /// TF_DS_MMA_SF `sw` in effect (off: nothing changes, nothing loaded): native_sf.cu's kernels from `image`, then the
    /// switch. An error leaves it off.
    pub fn useMmaSf(k: *Kernels, d: *const cuda.Driver, image: []const u8, sw: MmaSf) !void {
        if (sw == .off) return;
        if (k.sfk == null) k.sfk = try SfKernels.load(d, image);
        k.sf = sw;
    }

    pub fn unload(k: *Kernels) void {
        k.counters.free();
        if (k.comb_cnt) |*c| c.free();
        if (k.sfk) |*x| x.module.unload();
        k.module.unload();
        k.* = undefined;
    }
};

fn launch(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, args);
}

/// TF_DS_DECODE_DEC as Kernels.load reads it: the decode tensor-core kernels unless "0".
pub fn decOnEnv() bool {
    return if (std.c.getenv("TF_DS_DECODE_DEC")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
}

/// TF_DS_PF_TILES as Kernels.load reads it: "0" turns the prompt chunks' tile kernels off.
pub fn pfOffEnv() bool {
    return if (std.c.getenv("TF_DS_PF_TILES")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
}

/// The prefill tile kernels' dynamic shared memory (native.cu asserts the same: pf::kGemmSmem, kGuSmem, kDnSmem).
pub const pf_gemm_smem: u32 = 76800;
pub const pf_gu_smem: u32 = 79872;
pub const pf_dn_smem: u32 = 53760;
pub const pf_bf16_smem: u32 = 81920;
/// tf_dsn_pf_bf16_t64 (pf::kBf64Smem: 4 stages of 128 rows of 80 bytes): two CTAs and their 1 KiB each reserve fit an SM's 100 KiB.
pub const pf_bf16_t64_smem: u32 = 40960;
pub const pf_w8_smem: u32 = 84480; // tf_dsn_pf_bf16w8 (pf::kW8Smem)
pub const pf_gemm5_smem: u32 = 87040; // tf_dsn_pf_gemm5 (pf::GemmFiveCfg::SMEM)
pub const pf_gu_r128_smem: u32 = 97280; // tf_dsn_pf_gu_r128 (pf::kGuR128Smem)
pub const pf_dn_r128_smem: u32 = 62464; // tf_dsn_pf_dn_r128 (pf::kDnR128Smem)

/// The grouped experts' prefill tile kernels a prompt chunk launches and their dynamic shared memory: the baseline
/// tf_dsn_pf_gu / tf_dsn_pf_dn, or with TF_DS_PF_RUN128 their 128-byte-run twins (the same grid and arguments).
pub const ExpertsLaunch = struct { r128: bool, gu_smem: u32, dn_smem: u32 };
pub fn pfExpertsLaunch(run128: bool) ExpertsLaunch {
    return if (run128) .{ .r128 = true, .gu_smem = pf_gu_r128_smem, .dn_smem = pf_dn_r128_smem } else .{ .r128 = false, .gu_smem = pf_gu_smem, .dn_smem = pf_dn_smem };
}
/// The grouped prefill tiles' rows (native.cu pf::kBm).
pub const pf_bm = 64;
/// Rows from which a dense prompt GEMM takes the prefill tile kernel (fewer: the tile kernel; the same bits either way).
pub const pf_min_rows = 64;

fn launchShared(f: cuda.Function, s: cuda.Stream, grid: [3]usize, smem: u32, args: *cuda.Args) !void {
    try launchSharedT(f, s, grid, smem, 256, args);
}

/// launchShared for a block of `threads` threads (the 16-warp kernels' 512).
fn launchSharedT(f: cuda.Function, s: cuda.Stream, grid: [3]usize, smem: u32, threads: u32, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads }, .shared = smem }, s, args);
}

fn aligned(p: u64, a: u64) bool {
    return p % a == 0;
}

/// Whether a dense FP8 linear on these operands fits the prefill GEMM (tf_dsn_pf_gemm: N a multiple of 128, K of 64 up
/// to 8192, cp.async-aligned rows).
fn pfDense(k: *const Kernels, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, accumulate: bool) bool {
    if (k.pf_off or m < pf_min_rows or l.fmt != .fp8) return false;
    if (l.n % 128 != 0 or l.k % 64 != 0 or l.k > 8192) return false;
    if (!aligned(q, 16) or ldq % 16 != 0 or !aligned(sc, 8) or lds % 2 != 0 or !aligned(l.w, 16) or l.k % 16 != 0) return false;
    if (accumulate and (!aligned(y, 8) or ldy % 2 != 0)) return false;
    return true;
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// `n` bf16 values into fp16 (round to nearest).
pub fn bf16ToF16(k: *const Kernels, s: cuda.Stream, in: u64, out: u64, n: usize) !void {
    var a: cuda.Args = .{};
    a.add(in);
    a.add(out);
    a.add(@as(i64, @intCast(n)));
    try launch(k.to_f16, s, .{ cdiv(n, 256), 1, 1 }, 256, &a);
}

/// The rows a GEMM launch takes at most (larger row counts run as launches of this many, each its own rows' bits):
/// it bounds the split partials' buffer.
pub const slab_rows = 1024;

/// TF_DS_GEMM_HALVES in effect (Model.open): a pass of more than `gemm_halves_rows` rows runs its dense FP8, bf16 and
/// pf GEMMs as row blocks of that many rows at most, each block the launch a pass of that many rows makes for its
/// rows; the experts keep the whole pass. Rows are independent in these GEMMs (a row's bits at every row count), so
/// every output byte is the whole-pass launch's.
pub var gemm_halves: bool = false;
pub const gemm_halves_rows: usize = 4096;

/// A row block of a GEMM launch's `m` rows: [start, start + n).
pub const Block = struct { start: usize, n: usize };

/// The row blocks TF_DS_GEMM_HALVES gives a GEMM of `m` rows (`on`: the switch in effect): blocks of gemm_halves_rows
/// rows from row 0, the last one the remainder; a launch of that many rows or fewer is one block, as is every launch
/// with the switch off.
pub fn blockCount(m: usize, on: bool) usize {
    return if (on and m > gemm_halves_rows) cdiv(m, gemm_halves_rows) else 1;
}

/// Block `i` of blockCount(m, on): its rows, never crossing the launch's m (i past the count: no rows).
pub fn blockAt(m: usize, on: bool, i: usize) Block {
    const bs = if (on and m > gemm_halves_rows) gemm_halves_rows else m;
    if (bs == 0 or i * bs >= m) return .{ .start = m, .n = 0 };
    return .{ .start = i * bs, .n = @min(bs, m - i * bs) };
}

/// The rows a GEMM launch picks its kernel, split and variant at under TF_DS_GEMM_HALVES: its first block's (the block
/// rows at more, 4,096 at the served setting), so every block takes the launch a 4,096-row pass takes for its rows.
pub fn routeRows(m: usize, on: bool) usize {
    return if (on and m > gemm_halves_rows) gemm_halves_rows else m;
}

/// TF_DS_PF_RASTER in effect (Model.open): the group of row tiles G (2, 4, 8 or 16) the prompt tile GEMMs walk their grid
/// in (native.cu pf_tile_of: G row tiles across all the column tiles, then the next G), or 0: row tiles fastest. It
/// changes only the order whole output tiles run in, so every output byte is the default order's.
pub var pf_raster: u32 = 0;

/// The activation slab a launch must read, in bytes, before the grouped order applies: below it the slab stays in the L2
/// whichever order the tiles run in. vLLM PR #55180's gate for the SM12x blockwise-FP8 GEMM (the 24 MiB L2 of a GB10
/// holds a slab of up to about this much beside the weight tiles in flight).
pub const raster_slab_bytes: usize = 14 << 20;

/// The `raster` argument of a tile GEMM launch of `m` rows whose activation takes `row_bytes` a row: `g` (pf_raster) where the
/// slab m * row_bytes reaches raster_slab_bytes, else 0.
pub fn rasterOf(g: u32, m: usize, row_bytes: usize) c_int {
    return if (g != 0 and m * row_bytes >= raster_slab_bytes) @intCast(g) else 0;
}

/// native.cu pf_tile_of: the (row tile, column tile) of the CTA at (bx, by) of a grid of nrow x ncol tiles under group `g`
/// (0: bx, by). Every tile of the grid is the image of exactly one CTA.
pub fn rasterTile(g: u32, nrow: u32, ncol: u32, bx: u32, by: u32) [2]u32 {
    if (g == 0) return .{ bx, by };
    const lin = bx + nrow * by;
    const gsize = g * ncol;
    const gid = lin / gsize;
    const r0 = gid * g;
    const rows = @min(g, nrow - r0);
    const w = lin - gid * gsize;
    return .{ r0 + w % rows, w / rows };
}

/// A layer's K splits, from its shape alone (a row's sum is the same at every row count): enough CTAs over N (four
/// 32-column warps a CTA) to fill the GPU twice, at least 8 K blocks a split, at most 16 splits.
pub fn splits(k: usize, n: usize) usize {
    const kb = k / 32;
    const ctas = cdiv(n / 32, 4);
    var s: usize = 1;
    while (ctas * s < 96 and s < 16 and kb / (s * 2) >= 8) s *= 2;
    return s;
}

/// TF_DS_PF_SPLIT: the kernel of a prompt chunk's split FP8 GEMMs (a K split, from pf_min_rows rows: wq_a, wkv, wq_b,
/// wo_b, the indexer's wq_b). off: tf_dsn_pf_gemm_fold (the baseline); a, b, c6, c7: that configuration of
/// tf_dsn_pf_gemm_split* for every shape it divides (pfSplitCfg); table: split_table's choice by shape and rows. Every
/// choice gives the fold kernel's bits: the same MMAs, scaling and split merge in the same order (native.cu
/// run_tile_split1).
pub const PfSplit = enum(u32) {
    off = 0,
    table = 1,
    a = 2,
    b = 3,
    c6 = 4,
    c7 = 5,

    const texts = [_][]const u8{ "0", "1", "A", "B", "C6", "C7" };

    /// TF_DS_PF_SPLIT's text: unset or "0" off, "1" the table, "A", "B", "C6" or "C7" that configuration; null for any
    /// other text (not a setting: the start says so and runs off).
    pub fn parse(text: ?[]const u8) ?PfSplit {
        const v = text orelse return .off;
        for (texts, 0..) |t, i| if (std.mem.eql(u8, v, t)) return @fromBackingInt(@intCast(i));
        return null;
    }

    /// The setting's text for a Settings word or the SSD tier's math (null: no setting of this build).
    pub fn name(word: u32) ?[]const u8 {
        return if (word < texts.len) texts[word] else null;
    }
};

/// The split configurations (native.cu GemmSplit*Cfg, kernels tf_dsn_pf_gemm_split{A,B,C6,C7}_kernel): rows a tile,
/// K blocks a stage, dynamic shared memory.
pub const SplitCfg = enum(u2) { a, b, c6, c7 };
pub const split_bm = [4]usize{ 128, 64, 64, 64 };
pub const split_kbs = [4]usize{ 2, 4, 2, 2 };
pub const split_smem = [4]u32{ 87040, 76800, 76800, 89600 };
const split_st_kernels = [4][:0]const u8{ "tf_dsn_pf_gemm_splitA_st_kernel", "tf_dsn_pf_gemm_splitB_st_kernel", "tf_dsn_pf_gemm_splitC6_st_kernel", "tf_dsn_pf_gemm_splitC7_st_kernel" };
const split_kernels = [4][:0]const u8{ "tf_dsn_pf_gemm_splitA_kernel", "tf_dsn_pf_gemm_splitB_kernel", "tf_dsn_pf_gemm_splitC6_kernel", "tf_dsn_pf_gemm_splitC7_kernel" };

/// TF_DS_PF_SPLIT=1's table: a split FP8 GEMM's shape (K, N) and its configuration by the launch's rows (at most 512,
/// at most 1,024, more); null and every shape not listed: the fold kernel.
pub const SplitRow = struct { k: usize, n: usize, cfg: [3]?SplitCfg };
pub const split_table = [_]SplitRow{
    .{ .k = 5120, .n = 1280, .cfg = .{ null, null, .a } }, // wq_a
    .{ .k = 5120, .n = 512, .cfg = .{ null, null, .a } }, // wkv
    .{ .k = 1280, .n = 8192, .cfg = .{ null, null, .a } }, // a 2D node's wq_b
    .{ .k = 4096, .n = 2560, .cfg = .{ null, null, .a } }, // a 2D node's wo_b
    .{ .k = 1280, .n = 4096, .cfg = .{ null, null, .a } }, // the indexer's wq_b
};

/// split_table's configuration for a launch of m rows of shape (K, N).
fn tableCfg(m: usize, kdim: usize, n: usize) ?SplitCfg {
    const class: usize = if (m <= 512) 0 else if (m <= 1024) 1 else 2;
    for (split_table) |row| if (row.k == kdim and row.n == n) return row.cfg[class];
    return null;
}

/// The configuration a split FP8 GEMM of m rows and shape (K, N) takes under TF_DS_PF_SPLIT (null: the fold kernel):
/// the forced one or the table's, only for a K split (splits > 1) whose fold (K blocks a split) and K blocks are
/// multiples of the configuration's K blocks a stage.
pub fn pfSplitCfg(sw: PfSplit, m: usize, kdim: usize, n: usize) ?SplitCfg {
    const cfg: SplitCfg = switch (sw) {
        .off => return null,
        .table => tableCfg(m, kdim, n) orelse return null,
        .a => .a,
        .b => .b,
        .c6 => .c6,
        .c7 => .c7,
    };
    const sp = splits(kdim, n);
    if (sp < 2) return null;
    const nkb = kdim / 32;
    const kbs = split_kbs[@backingInt(cfg)];
    if (cdiv(nkb, sp) % kbs != 0 or nkb % kbs != 0) return null;
    return cfg;
}

/// Why TF_DS_PF_SPLIT asked for cannot run (null: it can): the checkpoint's own FP8 / MXFP4 weights and the prompt
/// chunks' tile kernels.
pub fn splitOff(native: bool, pf_off: bool) ?[]const u8 {
    if (!native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    if (pf_off) return "needs the prompt chunks' tile kernels (TF_DS_PF_TILES=0 turns them off)";
    return null;
}

/// The prompt tile GEMM of a dense FP8 linear that pfDense admits: tf_dsn_pf_gemm without a K split, the fold kernel
/// with one, or the split configuration TF_DS_PF_SPLIT gives the shape (pfSplitCfg).
pub const PfGemm = union(enum) { gemm, fold, split: SplitCfg, gemm5 };
pub fn pfGemmOf(sw: PfSplit, m: usize, kdim: usize, n: usize) PfGemm {
    if (splits(kdim, n) < 2) return .gemm;
    if (pfSplitCfg(sw, m, kdim, n)) |c| return .{ .split = c };
    return .fold;
}

/// pfGemmOf with TF_DS_PF_GEMM5 (`gemm5`): a GEMM without a K split on tf_dsn_pf_gemm5 (the same tile, K blocks a
/// stage and K order at 5 stages: tf_dsn_pf_gemm's bits).
pub fn pfGemmKernel(sw: PfSplit, gemm5: bool, m: usize, kdim: usize, n: usize) PfGemm {
    const c = pfGemmOf(sw, m, kdim, n);
    return if (c == .gemm and gemm5) .gemm5 else c;
}

/// TF_DS_MMA_SF: a prompt chunk's FP8 GEMMs on the tensor core's block-scaled MMA (native_sf.cu, the dsv41_native_sf
/// image built for sm_121a): the dense FP8 GEMMs mx.gemm takes to its tile kernels and the grouped experts' gate / up
/// and down. chained: a split's K blocks accumulate in the MMA (the switch's arithmetic: not the baseline bits); exact: each
/// K block's MMA from zero, then an fp32 add (the baseline bits whenever every scaled block sum is a normal float: the
/// replay's check of the kernels, no setting).
pub const MmaSf = enum(u2) { off = 0, chained = 1, exact = 2 };

/// native_sf.cu's kernels, [0] the chained arithmetic's and [1] the exact one's where both exist.
pub const SfKernels = struct {
    module: cuda.Module,
    gemm: [2]cuda.Function,
    split64: [2]cuda.Function,
    split128: [2]cuda.Function,
    zsplit: cuda.Function,
    // the exact arithmetic's split64 at a 64-column tile (TF_DS_SPLIT64N); null in an image built without it
    split64n: ?cuda.Function = null,
    gu: [2]cuda.Function,
    dn: [2]cuda.Function,
    // the exact arithmetic's kernels on 16 warps a block (TF_DS_GEMM_W16); null each in an image built without it
    w16: W16 = .{},
    // TF_DS_GEMM_R2's kernels (the exact arithmetic's): null each in an image built without it
    r2: R2 = .{},
    ranges: ?cuda.Function = null, // tf_dsn_sf_ranges: TF_DS_MMA_SF_EXACT's guard
    // the whole-line twins for a y in the transport's arena (Kernels.useStaged)
    st_gemm: [2]?cuda.Function = @splat(null),
    st_split64: [2]?cuda.Function = @splat(null),
    st_split128: [2]?cuda.Function = @splat(null),
    // wo_b's exact whole-line kernel with the CTAs in a row-quarter order and the doorbell rung from inside (sentFirstTakes)
    st_door: ?cuda.Function = null,
    // the same kernel on the activation rows' two bases (gemmTwo)
    st_two: ?cuda.Function = null,
    // down's exact kernel on the pack's two parts and the rest block's owner part where they lie (Part3)
    dn_three: ?cuda.Function = null,
    // the same three kernels in the chained arithmetic (TF_DS_MMA_SF), null in an image built without them
    ch_door: ?cuda.Function = null,
    ch_two: ?cuda.Function = null,
    ch_dn_three: ?cuda.Function = null,

    /// wo_b's door kernel, the two-base kernel and down on the pack's parts in the arithmetic `sf` runs (null: none loaded, or no block-scaled arithmetic).
    pub fn door(sk: *const SfKernels, sf: MmaSf) ?cuda.Function {
        return switch (sf) {
            .exact => sk.st_door,
            .chained => sk.ch_door,
            .off => null,
        };
    }
    pub fn two(sk: *const SfKernels, sf: MmaSf) ?cuda.Function {
        return switch (sf) {
            .exact => sk.st_two,
            .chained => sk.ch_two,
            .off => null,
        };
    }
    pub fn three(sk: *const SfKernels, sf: MmaSf) ?cuda.Function {
        return switch (sf) {
            .exact => sk.dn_three,
            .chained => sk.ch_dn_three,
            .off => null,
        };
    }

    /// The kernel a split64 launch of N columns takes under TF_DS_SPLIT64N (`on`): the 64-column tile's, for the exact arithmetic (`x` 1)
    /// at N of 512 (wkv) and 1,280 (wq_a), when the output is not the whole-line twin's and the 16-warp twin is not taken; else null
    /// (the baseline launch). Its grid is (cdiv(m, 64), n / 64) at 128 threads.
    pub fn split64nOf(sk: *const SfKernels, on: bool, x: usize, n: usize, stage: bool, w16: bool) ?cuda.Function {
        if (!on or x != 1 or stage or w16) return null;
        if (n != 512 and n != 1280) return null;
        return sk.split64n;
    }

    fn useStaged(sk: *SfKernels) !void {
        // the GPU-scope release with the last arrival's system fence (door_arrive_cta<2>), as combine's door: the per-thread system fence of the
        // twin before it held every sent CTA's slot until its stores drained (LEDGER: 1.300 ms a launch against the plain launch's 1.143)
        sk.st_door = try Kernels.pfFunction(sk.module, "tf_dsn_sf_split128_exact_st_door_g_kernel", sf_split128_smem);
        sk.st_two = Kernels.pfFunction(sk.module, "tf_dsn_sf_split128_exact_st_two_kernel", sf_split128_smem) catch null;
        sk.ch_door = Kernels.pfFunction(sk.module, "tf_dsn_sf_split128_st_door_kernel", sf_split128_smem) catch null;
        sk.ch_two = Kernels.pfFunction(sk.module, "tf_dsn_sf_split128_st_two_kernel", sf_split128_smem) catch null;
        const pf = Kernels.pfFunction;
        sk.st_gemm = .{ try pf(sk.module, "tf_dsn_sf_gemm_st_kernel", sf_gemm_smem), try pf(sk.module, "tf_dsn_sf_gemm_exact_st_kernel", sf_gemm_smem) };
        sk.st_split64 = .{ try pf(sk.module, "tf_dsn_sf_split64_st_kernel", sf_split64_smem), try pf(sk.module, "tf_dsn_sf_split64_exact_st_kernel", sf_split64_smem) };
        sk.st_split128 = .{ try pf(sk.module, "tf_dsn_sf_split128_st_kernel", sf_split128_smem), try pf(sk.module, "tf_dsn_sf_split128_exact_st_kernel", sf_split128_smem) };
    }

    /// The kernels of `image` (cuda.kernels.dsv41_native_sf: sm_121a code alone, so another GPU does not load it).
    pub fn load(d: *const cuda.Driver, image: []const u8) !SfKernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        const pf = Kernels.pfFunction;
        var sk: SfKernels = .{
            .module = m,
            .gemm = .{ try pf(m, "tf_dsn_sf_gemm_kernel", sf_gemm_smem), try pf(m, "tf_dsn_sf_gemm_exact_kernel", sf_gemm_smem) },
            .split64 = .{ try pf(m, "tf_dsn_sf_split64_kernel", sf_split64_smem), try pf(m, "tf_dsn_sf_split64_exact_kernel", sf_split64_smem) },
            .split128 = .{ try pf(m, "tf_dsn_sf_split128_kernel", sf_split128_smem), try pf(m, "tf_dsn_sf_split128_exact_kernel", sf_split128_smem) },
            .zsplit = try pf(m, "tf_dsn_sf_zsplit_kernel", sf_split64_smem),
            .gu = .{ try pf(m, "tf_dsn_sf_gu_kernel", pf_gu_r128_smem), try pf(m, "tf_dsn_sf_gu_exact_kernel", pf_gu_r128_smem) },
            .dn = .{ try pf(m, "tf_dsn_sf_dn_kernel", pf_dn_r128_smem), try pf(m, "tf_dsn_sf_dn_exact_kernel", pf_dn_r128_smem) },
            .w16 = .{
                .split64 = pf(m, "tf_dsn_sf_split64_exact_w16_kernel", sf_split64_smem) catch null,
                .split128 = pf(m, "tf_dsn_sf_split128_exact_w16_kernel", sf_split128_smem) catch null,
                .gu = pf(m, "tf_dsn_sf_gu_exact_w16_kernel", pf_gu_r128_smem) catch null,
                .dn = pf(m, "tf_dsn_sf_dn_exact_w16_kernel", pf_dn_r128_smem) catch null,
            },
            .r2 = .{
                .gu = pf(m, "tf_dsn_sf_gu_exact_r2_kernel", pf_gu_r128_smem) catch null,
                .dn = pf(m, "tf_dsn_sf_dn_exact_r2_kernel", r2_dn_smem) catch null,
            },
            .ranges = try m.function("tf_dsn_sf_ranges_kernel"),
        };
        sk.split64n = pf(m, "tf_dsn_sf_split64n_exact_kernel", sf_split64n_smem) catch null;
        sk.dn_three = pf(m, "tf_dsn_sf_dn_exact_three_kernel", pf_dn_r128_smem) catch null;
        sk.ch_dn_three = pf(m, "tf_dsn_sf_dn_three_kernel", pf_dn_r128_smem) catch null;
        return sk;
    }
};

/// native_sf.cu's sf::Part3 for a pack of `p` slot rows (packCodesPlan): a part is its main block [p, ig] of codes then its rest block
/// [p, ir], the scales (fp32 a 32-value group) after the codes, main block then rest block; the first part is down's own base
/// (q, s with ldq ig, lds ig / 32). q1 / s1 the second part's main block, qr0 / sr0 and qr1 / sr1 the rest blocks of the two parts,
/// ldq_rest = ir and lds_rest = ir / 32 their row strides, the int32 [E + 1] owner table, kb_main the K blocks of one pair's main columns.
pub const Part3 = extern struct { q1: u64, s1: u64, qr0: u64, sr0: u64, qr1: u64, sr1: u64, owner: u64, ldq_rest: i64, lds_rest: i64, kb_main: c_int };

/// The pack's two parts in pair order (the codes of `rows` slot rows, a main block `ig` and a rest block `ir` wide, packCodesPlan's layout).
pub const Three = struct { p0: u64, p1: u64, rows: usize, ig: usize, ir: usize };

/// Whether down reads the pack from its two parts (the three-base kernel, native_sf.cu's THREE loader): the exact block-scaled expert kernels
/// are in use on weights whose scales are proven, the kernel is loaded, the rest blocks have their owner table, and down's K is the two pairs'
/// main blocks and one rest block. False: the parts are assembled into hq / hs and down reads those.
pub fn threeTakes(k: *const Kernels, ex: weights.Experts, sc: ExpertScratch, ig: usize, ir: usize) bool {
    const nx = ex.native orelse return false;
    const rest = ex.rest orelse return false;
    _ = rest;
    if (k.sf == .off or sfExperts(k, nx.s1, true) == null) return false;
    const sk = k.sfk orelse return false;
    if (sk.three(k.sf) == null) return false;
    if (!pfExperts(k, sc, ex.down_k, ex.down_n, 0)) return false;
    return ig % 32 == 0 and ir % 32 == 0 and ex.down_k == 2 * ig + ir;
}

/// The byte offsets, from a part's start, of a pack part's blocks for `p` slot rows of a main block `ig` and a rest block `ir` wide: the
/// main codes at 0, the rest codes after them, the main scales after all the codes, the rest scales after those.
pub const PackBlocks = struct { main_q: usize, rest_q: usize, main_s: usize, rest_s: usize };
pub fn packBlocks(p: usize, ig: usize, ir: usize) PackBlocks {
    const codes = p * (ig + ir);
    return .{ .main_q = 0, .rest_q = p * ig, .main_s = codes, .rest_s = codes + p * (ig / 32) * 4 };
}

/// Where down's K block `kb` of slot row `m` lies when read from the two parts and the owner's rest block (the mapping
/// native_sf.cu's THREE loader makes): (which part, byte offset in it, bytes in a block, scale float offset in it).
pub fn packAt(p: usize, ig: usize, ir: usize, owner_second: bool, kb: usize, m: usize) struct { part: u1, q: usize, s: usize } {
    const b = packBlocks(p, ig, ir);
    const kmain = ig / 32;
    if (kb < kmain) return .{ .part = 0, .q = b.main_q + m * ig + kb * 32, .s = b.main_s + (m * kmain + kb) * 4 };
    if (kb < 2 * kmain) return .{ .part = 1, .q = b.main_q + m * ig + (kb - kmain) * 32, .s = b.main_s + (m * kmain + kb - kmain) * 4 };
    const r = kb - 2 * kmain;
    return .{ .part = @intFromBool(owner_second), .q = b.rest_q + m * ir + r * 32, .s = b.rest_s + (m * (ir / 32) + r) * 4 };
}

/// The exact arithmetic's split tiles and experts on 16 warps a block (TF_DS_GEMM_W16), each null where the image
/// was built without it. Their tiles and stages are the 8-warp kernels', so their dynamic shared memory is theirs.
pub const W16 = struct {
    split64: ?cuda.Function = null,
    split128: ?cuda.Function = null,
    gu: ?cuda.Function = null,
    dn: ?cuda.Function = null,
};

/// TF_DS_GEMM_R2's kernels, each null where the image was built without it: the experts' gate / up and down on the
/// exact arithmetic. Their shared memory is pf_gu_r128_smem and r2_dn_smem.
pub const R2 = struct {
    gu: ?cuda.Function = null,
    dn: ?cuda.Function = null,
};

/// The block-scaled kernels' dynamic shared memory (native_sf.cu asserts its tiles'): tf_dsn_sf_gemm (3 stages of
/// 17,408 bytes), tf_dsn_sf_split64 and tf_dsn_sf_zsplit (6 of 12,800), tf_dsn_sf_split128 (5 of 17,408); the experts'
/// take their 128-byte-run twins' (pf_gu_r128_smem, pf_dn_r128_smem).
pub const sf_gemm_smem: u32 = 52224;
pub const sf_split64_smem: u32 = 76800;
pub const sf_split128_smem: u32 = 87040;
/// tf_dsn_sf_split64n (native_sf.cu sf::Split64NCfg: 5 stages of 8,704 bytes): two blocks fit an SM's 100 KiB with their static bytes.
pub const sf_split64n_smem: u32 = 43520;
/// TF_DS_GEMM_R2's kernels' dynamic shared memory (native_sf.cu asserts its tiles'): the experts' down at 128 rows
/// (90,112 bytes); gate / up takes the tile above.
pub const r2_dn_smem: u32 = 90112;

/// The block-scaled kernel of a dense FP8 GEMM of m rows whose K split is `sp` (splits()): gemm without a split; with
/// one, the split folded in the CTA on 64-row tiles up to 1,024 rows and on 128-row tiles past them, and (chained) one
/// split a CTA with tf_dsn_reduce after it below pf_min_rows rows: the same MMAs and adds in the same order, so a row's
/// bits at every row count. null: the baseline kernels (off; exact below pf_min_rows, where the slab kernels give its bits).
pub const SfGemm = enum { gemm, split64, split128, zsplit };
pub fn sfGemmOf(sw: MmaSf, m: usize, sp: usize) ?SfGemm {
    if (sw == .off) return null;
    if (sw == .exact and m < pf_min_rows) return null;
    if (sp < 2) return .gemm;
    if (m < pf_min_rows) return .zsplit;
    return if (m <= 1024) .split64 else .split128;
}

/// A dense FP8 shape TF_DS_MMA_SF's kernels take: N a multiple of 128, K of 64 up to 8,192 (the tile's weight scales'
/// slab), and with a K split each split's K blocks even (the split tiles' 2 K blocks a stage).
pub fn sfShape(kdim: usize, n: usize) bool {
    if (n % 128 != 0 or kdim % 64 != 0 or kdim > 8192) return false;
    const sp = splits(kdim, n);
    return sp < 2 or cdiv(kdim / 32, sp) % 2 == 0;
}

/// Whether TF_DS_MMA_SF's kernels take a dense FP8 linear on these operands: the switch in effect, sfShape, pfDense's
/// alignment of the operands, and with TF_DS_MMA_SF_EXACT a shape with a K split (sfExactShape) and a key its guard
/// proved (sfProvenKey: a linear the guard did not read stays on the baseline kernels), at any row count (a GEMM it does
/// not take keeps the baseline kernels at every one).
fn sfDense(k: *const Kernels, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, y: u64, ldy: usize, accumulate: bool) bool {
    if (k.sf == .off or k.pf_off or l.fmt != .fp8 or !sfShape(l.k, l.n)) return false;
    if (k.sf == .exact and (!sfExactShape(l.k, l.n) or !sfProvenKey(k, l.s))) return false;
    if (!aligned(q, 16) or ldq % 16 != 0 or !aligned(sc, 8) or lds % 2 != 0 or !aligned(l.w, 16) or l.k % 16 != 0) return false;
    if (accumulate and (!aligned(y, 8) or ldy % 2 != 0)) return false;
    return true;
}

/// The block-scaled kernel mx.gemm takes for a dense FP8 GEMM of m rows on these operands (sfDense, then sfGemmOf at
/// the shape's K split); null: the baseline kernels.
pub fn sfRoute(k: *const Kernels, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, y: u64, ldy: usize, accumulate: bool, m: usize) ?SfGemm {
    if (!sfDense(k, l, q, ldq, sc, lds, y, ldy, accumulate)) return null;
    return sfGemmOf(k.sf, m, splits(l.k, l.n));
}

/// Whether a dense GEMM's y takes the whole-line kernels: fp32, written (not added to), 16-byte aligned with whole float4 rows,
/// and in the transport's arena (Kernels.useStaged).
pub fn stagedY(k: *const Kernels, y: u64, ldy: usize, y_dtype: DType, accumulate: bool) bool {
    return k.wc[1] != 0 and !accumulate and y_dtype == .f32 and y >= k.wc[0] and y < k.wc[1] and y % 16 == 0 and ldy % 4 == 0;
}

/// TF_DS_MMA_SF's launch of a dense FP8 GEMM that sfDense takes (gemm's arguments; `kind` sfGemmOf's): its kernel on
/// the baseline grids; one split a CTA writes z [splits, m, n] for tf_dsn_reduce (gemm's slab launches' reduce).
fn sfGemm(k: *const Kernels, s: cuda.Stream, kind: SfGemm, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, y_dtype: DType, accumulate: bool, z: u64) !void {
    const sk = k.sfk orelse return error.NoMmaSfKernels;
    const x: usize = @intFromBool(k.sf == .exact);
    const stage = stagedY(k, y, ldy, y_dtype, accumulate) and kind != .zsplit and sk.st_gemm[x] != null;
    const sp = splits(l.k, l.n);
    const a: MxArgs = .{ .q = q, .s = sc, .ldq = @intCast(ldq), .lds = @intCast(lds), .w = l.w, .ws = l.s, .ldw = l.k, .ldws = l.k / 32, .fmt = 0, .M = @intCast(m), .N = @intCast(l.n), .K = @intCast(l.k), .splits = @intCast(sp), .kblocks = @intCast(cdiv(l.k / 32, sp)), .z = z, .y = y, .ldy = @intCast(ldy), .y_dtype = @backingInt(y_dtype), .accumulate = @intFromBool(accumulate), .raster = rasterOf(pf_raster, m, l.k) };
    var args: cuda.Args = .{};
    args.add(a);
    // the exact arithmetic's split tiles on 16 warps a block when TF_DS_GEMM_W16 is on and this image has its twins
    const w = if (k.gemm_w16 and x == 1) sk.w16 else W16{};
    switch (kind) {
        .gemm => try launchShared(if (stage) sk.st_gemm[x].? else sk.gemm[x], s, .{ cdiv(m, 128), l.n / 128, 1 }, sf_gemm_smem, &args),
        .split64 => if (sk.split64nOf(k.split64n, x, l.n, stage, w.split64 != null)) |f| try launchSharedT(f, s, .{ cdiv(m, 64), l.n / 64, 1 }, sf_split64n_smem, 128, &args) else if (stage) try launchShared(sk.st_split64[x].?, s, .{ cdiv(m, 64), l.n / 128, 1 }, sf_split64_smem, &args) else if (w.split64) |f| try launchSharedT(f, s, .{ cdiv(m, 64), l.n / 128, 1 }, sf_split64_smem, 512, &args) else try launchShared(sk.split64[x], s, .{ cdiv(m, 64), l.n / 128, 1 }, sf_split64_smem, &args),
        .split128 => if (stage) try launchShared(sk.st_split128[x].?, s, .{ cdiv(m, 128), l.n / 128, 1 }, sf_split128_smem, &args) else if (w.split128) |f| try launchSharedT(f, s, .{ cdiv(m, 128), l.n / 128, 1 }, sf_split128_smem, 512, &args) else try launchShared(sk.split128[x], s, .{ cdiv(m, 128), l.n / 128, 1 }, sf_split128_smem, &args),
        .zsplit => {
            try launchShared(sk.zsplit, s, .{ cdiv(m, 64), l.n / 128, sp }, sf_split64_smem, &args);
            var ra: cuda.Args = .{};
            ra.add(z);
            ra.add(@as(c_int, @intCast(sp)));
            ra.add(@as(c_int, @intCast(m)));
            ra.add(@as(c_int, @intCast(l.n)));
            ra.add(y);
            ra.add(@as(i64, @intCast(ldy)));
            ra.add(@as(c_int, @backingInt(y_dtype)));
            ra.add(@as(c_int, @intFromBool(accumulate)));
            try launch(k.reduce, s, .{ cdiv(m * l.n, 256), 1, 1 }, 256, &ra);
        },
    }
}

/// native_sf.cu's sf::RowOrder: the order of a dense launch's CTAs by row quarter (see sf::tile_row).
pub const RowOrder = extern struct { quarters: u32, tpq_log2: u32, per_log2: u32, span: u32 };
/// native.cu's QuarterOrder: the order of tf_dsn_combine_door's CTAs by row quarter.
pub const QuarterOrder = extern struct { quarters: u32, unit: u32, span: u32 };
/// native.cu's Door: where a producer kernel rings the transport's doorbell (the arrival count, the doorbell line, the ticket).
pub const Door = extern struct { cnt: u64, door: u64, ticket: u64 };

/// The quarters (row start / q) of the four row blocks in launch order, one byte each, or null unless they are the four quarters of
/// four `q`-row blocks, each once.
fn packQuarters(first: [2]usize, last: [2]usize, q: usize) ?u32 {
    if (q == 0) return null;
    var seen = [4]bool{ false, false, false, false };
    var x: u32 = 0;
    for ([_]usize{ first[0], first[1], last[0], last[1] }, 0..) |row0, i| {
        if (row0 % q != 0 or row0 / q >= 4 or seen[row0 / q]) return null;
        seen[row0 / q] = true;
        x |= @as(u32, @intCast(row0 / q)) << @intCast(8 * i);
    }
    return x;
}

fn pow2Log(n: usize) ?u32 {
    if (n == 0 or n & (n - 1) != 0) return null;
    return @ctz(n);
}

/// The CTA order of a dense launch of 4 `q`-row quarters (128-row tiles, `col_tiles` columns of them) that runs the two quarters
/// starting at rows `first` before the two at `last`: every column of the first two quarters' row tiles first. Null unless a
/// quarter is a power of two row tiles.
pub fn rowOrder(first: [2]usize, last: [2]usize, q: usize, col_tiles: usize) ?RowOrder {
    if (q % 128 != 0) return null;
    const tpq = pow2Log(q / 128) orelse return null;
    const quarters = packQuarters(first, last, q) orelse return null;
    return .{ .quarters = quarters, .tpq_log2 = tpq, .per_log2 = tpq + 1, .span = @intCast(2 * (q / 128) * col_tiles) };
}

/// The order of a launch of `rows` rows that is the baseline: every column's row tiles in turn (one phase; `rows` / 128 a power of two,
/// at least four).
pub fn identityRowOrder(rows: usize, col_tiles: usize) ?RowOrder {
    if (rows % 128 != 0) return null;
    const per = pow2Log(rows / 128) orelse return null;
    if (per < 2) return null;
    return .{ .quarters = 0x03020100, .tpq_log2 = per - 2, .per_log2 = per, .span = @intCast((rows / 128) * col_tiles) };
}

/// The row tile and column tile the CTA that is `order`th in a launch runs (sf::tile_row, sf::tile_col).
pub fn tileOf(ro: RowOrder, order: u32) struct { row: u32, col: u32 } {
    const phase: u32 = @intFromBool(order >= ro.span);
    const w = order - (if (phase != 0) ro.span else 0);
    const pos = (phase << @intCast(ro.per_log2)) + (w & ((@as(u32, 1) << @intCast(ro.per_log2)) - 1));
    const q = (ro.quarters >> @intCast(8 * (pos >> @intCast(ro.tpq_log2)))) & 0xff;
    return .{ .row = (q << @intCast(ro.tpq_log2)) | (pos & ((@as(u32, 1) << @intCast(ro.tpq_log2)) - 1)), .col = w >> @intCast(ro.per_log2) };
}

/// The order of tf_dsn_combine_door's CTAs for `q`-row quarters of `n` columns (a CTA is 256 elements): the quarters at rows `first`
/// before the ones at `last`. Null unless q * n is a multiple of 256 and the blocks are the four quarters.
pub fn quarterOrder(first: [2]usize, last: [2]usize, q: usize, n: usize) ?QuarterOrder {
    if ((q * n) % 256 != 0) return null;
    const quarters = packQuarters(first, last, q) orelse return null;
    const unit: u32 = @intCast(q * n / 256);
    return .{ .quarters = quarters, .unit = unit, .span = 2 * unit };
}

/// Whether wo_b's GEMM (`l` on the quantized rows q / sc, `m` rows, fp32 y in the arena) can ring its first part's doorbell
/// itself: the exact block-scaled kernels in effect and taking it on the 128-row tiles, y a whole-line store, the door kernel loaded.
pub fn sentFirstTakes(k: *const Kernels, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, y: u64, ldy: usize, m: usize) bool {
    const sk = k.sfk orelse return false;
    if (sk.door(k.sf) == null) return false;
    if (!stagedY(k, y, ldy, .f32, false)) return false;
    const kind = sfRoute(k, l, q, ldq, sc, lds, y, ldy, false, m) orelse return false;
    return kind == .split128;
}

/// sfGemm's split128 launch in the order `ro` with the doorbell rung by the first `ro.span` CTAs' last arrival (sentFirstTakes).
pub fn gemmSentFirst(k: *const Kernels, s: cuda.Stream, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, z: u64, ro: RowOrder, door: Door) !void {
    const sk = k.sfk orelse return error.NoMmaSfKernels;
    const f = sk.door(k.sf) orelse return error.NoMmaSfKernels;
    const sp = splits(l.k, l.n);
    const a: MxArgs = .{ .q = q, .s = sc, .ldq = @intCast(ldq), .lds = @intCast(lds), .w = l.w, .ws = l.s, .ldw = l.k, .ldws = l.k / 32, .fmt = 0, .M = @intCast(m), .N = @intCast(l.n), .K = @intCast(l.k), .splits = @intCast(sp), .kblocks = @intCast(cdiv(l.k / 32, sp)), .z = z, .y = y, .ldy = @intCast(ldy), .y_dtype = @backingInt(DType.f32), .accumulate = 0 };
    var args: cuda.Args = .{};
    args.add(a);
    args.add(ro);
    args.add(door);
    try launchShared(f, s, .{ cdiv(m, 128), l.n / 128, 1 }, sf_split128_smem, &args);
}

/// The second base of the activation rows (native_sf.cu sf::Part2): the K blocks from `kb_split` on.
pub const Part2 = extern struct { q1: u64, s1: u64, kb_split: u32 };

/// Whether gemmTwo takes a GEMM: the exact whole-line kernel (sfRoute's split128 route, y in the arena), and both bases' rows alike
/// strided.
pub fn twoTakes(k: *const Kernels, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, y: u64, ldy: usize, m: usize) bool {
    const sk = k.sfk orelse return false;
    if (sk.two(k.sf) == null) return false;
    if (!stagedY(k, y, ldy, .f32, false)) return false;
    const kind = sfRoute(k, l, q, ldq, sc, lds, y, ldy, false, m) orelse return false;
    return kind == .split128;
}

/// sfGemm's split128 exact whole-line launch with the activation rows' K blocks [0, kb_split) at (q, sc) and [kb_split, K / 32) at
/// (p.q1, p.s1) block kb - kb_split, both with the strides ldq / lds (twoTakes).
pub fn gemmTwo(k: *const Kernels, s: cuda.Stream, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, p: Part2, m: usize, y: u64, ldy: usize, z: u64) !void {
    const sk = k.sfk orelse return error.NoMmaSfKernels;
    const f = sk.two(k.sf) orelse return error.NoMmaSfKernels;
    const sp = splits(l.k, l.n);
    const a: MxArgs = .{ .q = q, .s = sc, .ldq = @intCast(ldq), .lds = @intCast(lds), .w = l.w, .ws = l.s, .ldw = l.k, .ldws = l.k / 32, .fmt = 0, .M = @intCast(m), .N = @intCast(l.n), .K = @intCast(l.k), .splits = @intCast(sp), .kblocks = @intCast(cdiv(l.k / 32, sp)), .z = z, .y = y, .ldy = @intCast(ldy), .y_dtype = @backingInt(DType.f32), .accumulate = 0 };
    var args: cuda.Args = .{};
    args.add(a);
    args.add(p);
    try launchShared(f, s, .{ cdiv(m, 128), l.n / 128, 1 }, sf_split128_smem, &args);
}

/// linear's quantization of x and then gemmSentFirst: the bytes `linear` writes into y, the doorbell rung from inside.
pub fn linearSentFirst(k: *const Kernels, s: cuda.Stream, sc: Scratch, l: weights.Linear, x: u64, x_dtype: DType, ldx: usize, m: usize, y: u64, ldy: usize, ro: RowOrder, door: Door) !void {
    if (m > sc.rows or l.k > sc.max_k) return error.ScratchTooSmall;
    try quant(k, s, x, x_dtype, ldx, m, l.k, sc.q, l.k, sc.s, l.k / 32);
    try gemmSentFirst(k, s, l, sc.q, l.k, sc.s, l.k / 32, m, y, ldy, sc.z, ro, door);
}

/// A kernel with the block it launches and its dynamic shared memory.
pub const Launch = struct { f: cuda.Function, threads: u32, smem: u32 };

/// TF_DS_MMA_SF's kernel of a prompt chunk's expert tiles (gate / up, else down: the 128-byte-run twins' arguments,
/// grid and tiles), the block it launches and its shared memory: the baseline 256 threads, with TF_DS_GEMM_R2 the exact
/// arithmetic's R2 tiles (which stage the same arguments and tiles), or with TF_DS_GEMM_W16 its 16-warp twin's 512; null: the baseline kernel (the switch off, or with TF_DS_MMA_SF_EXACT the layer's
/// experts not proven by its guard, `key` their gate scale table NativeExperts.s1).
pub fn sfExperts(k: *const Kernels, key: u64, of_down: bool) ?Launch {
    if (k.sf == .off or (k.sf == .exact and !sfProvenKey(k, key))) return null;
    const sk = k.sfk orelse return null;
    const x: usize = @intFromBool(k.sf == .exact);
    if (k.gemm_r2 and x == 1) {
        if (of_down) {
            if (sk.r2.dn) |f| return .{ .f = f, .threads = 256, .smem = r2_dn_smem };
        } else if (sk.r2.gu) |f| return .{ .f = f, .threads = 256, .smem = pf_gu_r128_smem };
    }
    if (k.gemm_w16 and x == 1) {
        const w = sk.w16;
        const smem = if (of_down) pf_dn_r128_smem else pf_gu_r128_smem;
        const f = (if (of_down) w.dn else w.gu) orelse return .{ .f = if (of_down) sk.dn[1] else sk.gu[1], .threads = 256, .smem = smem };
        return .{ .f = f, .threads = 512, .smem = smem };
    }
    return .{ .f = if (of_down) sk.dn[x] else sk.gu[x], .threads = 256, .smem = if (of_down) pf_dn_r128_smem else pf_gu_r128_smem };
}

/// TF_DS_MMA_SF_EXACT's proven ranges of weight scale bytes (UE8M0). Within them every scaled block sum of a GEMM is a
/// normal float, given activation scales 2^-22 (act_quant's floor) to 2^100 (the bound the baseline FAST scaling assumes):
/// the block-scaled MMA then scales the block's sum by a power of two exactly, and adding it to the fp32 sum is the baseline
/// fmaf of the same exact product, so the bits are the baseline. FP8 x FP8 (a nonzero block sum at least 2^-18): 41 .. 127,
/// 2^-86 .. 2^0, inside the baseline FAST range; E4M3 x E2M1 (at least 2^-10, at most 32 x 448 x 6): 33 .. 137, 2^-94 ..
/// 2^10.
pub const sf_fp8_scales = [2]u8{ 41, 127 };
pub const sf_fp4_scales = [2]u8{ 33, 137 };
/// The GEMMs the guard may keep on the baseline kernels at most (more: the switch stays off).
pub const sf_kept_max = 64;

/// A range of device bytes: its first byte and its length.
pub const ByteRange = struct { ptr: u64, len: u64 };

/// A dense FP8 linear's weight scales: [n / 32, k / 32] UE8M0 bytes from l.s.
pub fn linearScales(l: weights.Linear) ByteRange {
    return .{ .ptr = l.s, .len = (@as(u64, l.n) / 32) * (l.k / 32) };
}

/// A layer's experts' weight scales, three tensors an expert (gate's, up's, down's: out[3e .. 3e + 3]), from the
/// device's pointer tables read back (s1, s3, s2 [E]) and the owner table (null: no rest block): each tensor's own
/// bytes, never the 256-byte gaps of the packing. A routed expert's MXFP4 gate / up scales [rows, dims / 32] (rows: the
/// main block, and the rest block's when owner[e] is 1), down's [down_n, down_k / 32]; the shared expert's (the last)
/// FP8 [rows / 32, dims / 32] and [down_n / 32, down_k / 32].
pub fn expertScales(ex: weights.Experts, s1: []const u64, s3: []const u64, s2: []const u64, owner: ?[]const i32, out: []ByteRange) void {
    for (0..ex.count) |e| {
        const l = weights.expertScaleLens(ex, e, if (owner) |o| o[e] == 1 else false);
        out[3 * e] = .{ .ptr = s1[e], .len = l[0] };
        out[3 * e + 1] = .{ .ptr = s3[e], .len = l[1] };
        out[3 * e + 2] = .{ .ptr = s2[e], .len = l[2] };
    }
}

/// Whether a tensor's scale bytes, from range[0] to range[1], lie in the proven range (fp4: MXFP4's, else FP8's).
pub fn sfProven(range: [2]u8, fp4: bool) bool {
    const r = if (fp4) sf_fp4_scales else sf_fp8_scales;
    return range[0] <= range[1] and range[0] >= r[0] and range[1] <= r[1];
}

/// The keys TF_DS_MMA_SF_EXACT's guard may prove at most (more: the switch stays off).
pub const sf_proven_max = 1024;

/// A dense FP8 shape TF_DS_MMA_SF_EXACT takes: one sfShape takes, with a K split. Without one (Engram's wkv) the
/// block-scaled kernel is no faster than the baseline, so the shape stays there (its guard proves it no key).
pub fn sfExactShape(kdim: usize, n: usize) bool {
    return sfShape(kdim, n) and splits(kdim, n) >= 2;
}

/// The block-scaled kernels' mode after TF_DS_MMA_SF_EXACT's agreement: agreed, the exact arithmetic it loaded;
/// refused, its own arithmetic off, and TF_DS_MMA_SF's (which takes the kernels first) as it was.
pub fn sfAfterExact(sf: MmaSf, agreed: bool) MmaSf {
    return if (agreed or sf != .exact) sf else .off;
}

/// A range's smallest and largest byte as tf_dsn_sf_ranges wrote them (u32 each); a word past 255 (its output left
/// unwritten: scaleRanges fills it with 0xff first) reads as a range nothing was read of (lo > hi).
pub fn readRange(lo: u32, hi: u32) [2]u8 {
    return if (lo > 255 or hi > 255) .{ 1, 0 } else .{ @intCast(lo), @intCast(hi) };
}

/// The guard's verdict on one tensor of `len` scale bytes from what the GPU read (`range`): proven when the tensor is
/// empty (nothing to prove), else only from a real read (lo <= hi) inside the proven range. An unread or unwritten
/// result keeps the GEMM on the baseline kernels.
pub fn sfProvenRead(len: u64, range: [2]u8, fp4: bool) bool {
    return len == 0 or sfProven(range, fp4);
}

/// TF_DS_MMA_SF_EXACT's guard proves the GEMM of weight scales `key` (a dense linear's l.s, a layer's experts'
/// NativeExperts.s1): the set kept ascending; a key proven twice is there once; more than sf_proven_max refuses.
pub fn sfProve(k: *Kernels, key: u64) !void {
    const keys = k.sf_proven[0..k.sf_nproven];
    var i: usize = keys.len;
    while (i > 0 and keys[i - 1] > key) i -= 1;
    if (i > 0 and keys[i - 1] == key) return;
    if (k.sf_nproven == sf_proven_max) return error.TooManyProven;
    var j: usize = k.sf_nproven;
    while (j > i) : (j -= 1) k.sf_proven[j] = k.sf_proven[j - 1];
    k.sf_proven[i] = key;
    k.sf_nproven += 1;
}

/// Whether TF_DS_MMA_SF_EXACT's guard proved the GEMM of weight scales `key`.
pub fn sfProvenKey(k: *const Kernels, key: u64) bool {
    var lo: usize = 0;
    var hi: usize = k.sf_nproven;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (k.sf_proven[mid] < key) lo = mid + 1 else hi = mid;
    }
    return lo < k.sf_nproven and k.sf_proven[lo] == key;
}

/// TF_DS_MMA_SF_EXACT's guard keeps the GEMM of weight scales `key` on the baseline kernels (a dense linear's l.s, a layer's
/// experts' NativeExperts.s1); more than sf_kept_max refuses.
pub fn sfKeep(k: *Kernels, key: u64) !void {
    if (sfKept(k, key)) return;
    if (k.sf_nkept == sf_kept_max) return error.TooManyKept;
    k.sf_kept[k.sf_nkept] = key;
    k.sf_nkept += 1;
}

/// Whether TF_DS_MMA_SF_EXACT's guard keeps the GEMM of weight scales `key` on the baseline kernels.
pub fn sfKept(k: *const Kernels, key: u64) bool {
    for (k.sf_kept[0..k.sf_nkept]) |x| if (x == key) return true;
    return false;
}

/// The smallest and the largest byte of each of `ranges` (device bytes: tf_dsn_sf_ranges, a CTA a range, one launch and
/// one copy back) into out.
pub fn scaleRanges(k: *const Kernels, d: *const cuda.Driver, s: cuda.Stream, gpa: std.mem.Allocator, ranges: []const ByteRange, out: [][2]u8) !void {
    const f = (k.sfk orelse return error.NoMmaSfKernels).ranges orelse return error.NoMmaSfKernels;
    if (ranges.len == 0) return;
    const host = try gpa.alloc(u64, 2 * ranges.len);
    defer gpa.free(host);
    for (ranges, 0..) |r, i| {
        host[i] = r.ptr;
        host[ranges.len + i] = r.len;
    }
    var buf = try cuda.DeviceBuffer.alloc(d, host.len * 8 + ranges.len * 8);
    defer buf.free();
    // the tables on the stream that reads them (a pageable copy staged before the call returns), the results filled
    // with 0xff first: a range left unwritten reads as nothing read (readRange)
    try buf.uploadAsync(0, std.mem.sliceAsBytes(host), s.handle);
    const mm = buf.ptr + host.len * 8;
    try d.check(d.api.cuMemsetD8Async(mm, 0xff, ranges.len * 8, s.handle), "cuMemsetD8Async");
    var a: cuda.Args = .{};
    a.add(buf.ptr);
    a.add(buf.ptr + ranges.len * 8);
    a.add(@as(c_int, @intCast(ranges.len)));
    a.add(mm);
    try launch(f, s, .{ ranges.len, 1, 1 }, 256, &a);
    try s.synchronize();
    const got = try gpa.alloc(u32, 2 * ranges.len);
    defer gpa.free(got);
    try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(got.ptr), mm, got.len * 4), "cuMemcpyDtoH");
    for (out, 0..) |*o, i| o.* = readRange(got[2 * i], got[2 * i + 1]);
}

/// A decode-size bf16 linear's K splits (16-column steps), from its shape alone, as splits() for the MX GEMM.
pub fn bfSplits(k: usize, n: usize) usize {
    const steps = k / 16;
    const ctas = cdiv(n / 32, 4);
    var s: usize = 1;
    while (ctas * s < 96 and s < 16 and steps / (s * 2) >= 8) s *= 2;
    return s;
}

/// The decode tensor-core kernels' K split of a dense linear, from its shape alone (fmt 0 FP8, 1 FP4, 2 bf16; n
/// output columns, k inputs): ks warps a 16-column tile in a CTA of dec_warps (each warp keeping at least two stages of
/// a row), cs CTAs a tile (grid z, the last CTA summing their partials in order). Measured on GB10 (decode_bench.cu):
/// a tile's K over all 8 warps of its CTA was the fastest for every shape of the model, and splitting it over CTAs
/// as well was not faster.
pub const DecSplit = struct { ks: usize, cs: usize };
pub fn decSplit(fmt: u2, n: usize, k: usize) DecSplit {
    _ = n;
    const rb: usize = switch (fmt) {
        0 => k,
        1 => k / 2,
        else => 2 * k,
    };
    const nst = cdiv(rb, dec_dense_cfg.rowb);
    var ks: usize = dec_warps;
    while (ks > 1 and nst < 2 * ks) ks /= 2;
    return .{ .ks = ks, .cs = 1 };
}

/// Whether the decode tensor-core kernels take a dense linear of this shape (else the GEMV / the bf16 GEMM: a choice by
/// shape alone).
pub fn decTakes(k: *const Kernels, fmt: u2, n: usize, kdim: usize) bool {
    if (!k.dec_on or n % 16 != 0 or n == 0) return false;
    return switch (fmt) {
        0, 1 => kdim % 128 == 0,
        else => kdim % 32 == 0,
    };
}

/// Floats of split partials a decode tensor-core launch of `rows` rows needs.
pub fn decZFloats(fmt: u2, n: usize, kdim: usize, rows: usize) usize {
    const sp = decSplit(fmt, n, kdim);
    return if (sp.cs > 1) sp.cs * rows * n else 0;
}

/// Dense decode linears on the tensor-core kernels in one launch (each `as`: every field but the split's; one kind, 0
/// FP8, 1 FP4, 2 bf16, and one row count), each as its own launch would compute it.
/// pdl: a programmatic dependent launch (its weights' prologue beside the predecessor), only right after quantDec:
/// the predecessor must be an ordinary launch (all earlier work done when it started) that lets it start early.
/// Past decode_rows rows (the drafter's pass of 13 or more streams: five rows a stream) the rows run in slabs of
/// decode_rows, each its own launch with the slab's row pointers and the split, quantization and epilogue it has alone
/// (the templates store at most decode_rows rows; a row's bits do not depend on the launch's row count). Only the first
/// slab may be a programmatic dependent launch: the later ones follow a decode kernel, so they are ordinary launches.
fn decDenseMulti(k: *const Kernels, s: cuda.Stream, kind: u2, as: []const DecArgs, z: ?Z, pdl: bool) !void {
    if (as.len == 0 or as.len > 4) return error.UnsupportedShape;
    const m: usize = @intCast(as[0].M);
    if (m <= decode_rows) return decDenseLaunch(k, s, kind, as, z, pdl);
    for (as) |a| if (a.M != as[0].M) return error.UnsupportedShape;
    var r0: usize = 0;
    while (r0 < m) : (r0 += decode_rows) {
        var sl: [4]DecArgs = undefined;
        for (as, 0..) |a, i| sl[i] = decSlab(a, r0, @min(decode_rows, m - r0));
        try decDenseLaunch(k, s, kind, sl[0..as.len], z, pdl and r0 == 0);
    }
}

/// Rows r0 .. r0 + mm of a decode launch's arguments: the activation rows (a row list, else the codes and scales: FP8 /
/// FP4 codes ldq bytes a row, bf16 rows ldq elements; scales lds floats) and the output rows (ldy elements of y_dtype).
fn decSlab(a: DecArgs, r0: usize, mm: usize) DecArgs {
    var b = a;
    b.M = @intCast(mm);
    if (a.rows != 0) {
        b.rows = a.rows + r0 * 4;
    } else {
        const qsz: usize = if (a.fmt == 2) 2 else 1;
        b.q = a.q + r0 * @as(usize, @intCast(a.ldq)) * qsz;
        if (a.s != 0) b.s = a.s + r0 * @as(usize, @intCast(a.lds)) * 4;
    }
    const ysz: usize = if (a.accumulate != 0 or a.y_dtype == @backingInt(DType.f32)) 4 else 2;
    b.y = a.y + r0 * @as(usize, @intCast(a.ldy)) * ysz;
    return b;
}

fn decDenseLaunch(k: *const Kernels, s: cuda.Stream, kind: u2, as: []const DecArgs, z: ?Z, pdl: bool) !void {
    var mm: DecMulti = .{ .n = @intCast(as.len) };
    const m: usize = @intCast(as[0].M);
    if (m > decode_rows) return error.NotADecodeWindow;
    var ctas: usize = 0;
    var zoff: usize = 0;
    var cs_all: usize = 0;
    for (as, 0..) |a0, i| {
        var a = a0;
        if (a.M != as[0].M) return error.UnsupportedShape;
        const n: usize = @intCast(a.N);
        const sp = decSplit(kind, n, @intCast(a.K));
        if (i == 0) cs_all = sp.cs else if (sp.cs != cs_all) return error.UnsupportedShape;
        a.ks = @intCast(sp.ks);
        a.cs = @intCast(sp.cs);
        const groups = cdiv(n / 16, dec_warps / sp.ks);
        if (sp.cs > 1) {
            const zz = z orelse return error.ScratchTooSmall;
            if (zoff + sp.cs * m * n > zz.floats or ctas + groups > dec_counters) return error.ScratchTooSmall;
            a.z = zz.ptr + zoff * 4;
            a.cnt = k.counters.ptr + ctas * 4;
            zoff += sp.cs * m * n;
        }
        mm.p[i] = a;
        mm.cta0[i] = @intCast(ctas);
        ctas += groups;
    }
    mm.cta0[as.len] = @intCast(ctas);
    const rg = decRg(m);
    var args: cuda.Args = .{};
    args.add(mm);
    try cuda.launch.launch(k.dec[kind][rg], .{ .grid = .{ .x = @intCast(ctas), .y = 1, .z = @intCast(cs_all) }, .block = .{ .x = 32 * dec_warps }, .shared = decShared(dec_dense_cfg, @as(usize, 1) << @intCast(rg)), .pdl = pdl and k.dec_pdl }, s, &args);
}

fn decDense(k: *const Kernels, s: cuda.Stream, kind: u2, a: DecArgs, z: ?Z) !void {
    return decDenseMulti(k, s, kind, &.{a}, z, false);
}

/// The rows a decode-size launch takes at most (a round's window, the drafter's rows): what the split partials of
/// gate / up, down and the bf16 linears are sized for (decodeZFloats).
pub const decode_rows = 64;

/// Split partials (fp32) for decode-size launches: the place and its size in floats. Decode launches split K by their
/// shape alone (so a row's bits are the same at every row count); prompt chunks never split them.
pub const Z = struct { ptr: u64, floats: usize };

/// Floats of decode split partials a layer's experts and bf16 linears need, for `rows` rows of `slots` slots
/// (n_experts: the router's outputs; ih: the indexer's heads, weights_proj's outputs).
pub fn decodeZFloats(lay: weights.Layer, hidden: usize, n_experts: usize, ih: usize, rows: usize, slots: usize) usize {
    var need: usize = 0;
    const ex = lay.experts;
    if (ex.native) |nx| {
        const p = rows * slots;
        const ncols = nx.tiles * 32;
        need = @max(need, splits(ex.dims, ncols) * 2 * p * ncols);
        need = @max(need, splits(ex.down_k, ex.down_n) * p * ex.down_n);
    }
    if (lay.gate_w != 0) need = @max(need, bfSplits(hidden, n_experts) * rows * n_experts);
    if (lay.idx_proj != 0 and ih % 32 == 0) need = @max(need, bfSplits(hidden, ih) * rows * ih);
    for (lay.wo_a[0..lay.groups]) |wo| if (wo.fmt == .bf16) {
        need = @max(need, bfSplits(wo.k, wo.n) * rows * wo.n);
    };
    // the decode linears that may be bf16 (mx.linear decode)
    for ([_]?weights.Linear{ lay.wq_a, lay.wkv, lay.wq_b, lay.wo_b, lay.comp_wkv, lay.comp_wgate, lay.idx_wq_b, lay.idx_wk, lay.engram_wkv }) |ol| {
        const l = ol orelse continue;
        if (l.fmt == .bf16) need = @max(need, bfSplits(l.k, l.n) * rows * l.n);
    }
    // the decode tensor-core kernels' cross-CTA partials (every decode linear, the router, the indexer's weights_proj)
        const l = ol orelse continue;
        const f: u2 = switch (l.fmt) {
            .fp8 => 0,
            .fp4 => 1,
            .bf16 => 2,
            .exl3 => continue,
        };
        need = @max(need, decZFloats(f, l.n, l.k, rows));
    }
    if (lay.gate_w != 0) need = @max(need, decZFloats(2, n_experts, hidden, rows));
    if (lay.idx_proj != 0) need = @max(need, decZFloats(2, ih, hidden, rows));
    return need;
}

/// The m16 row tiles a warp holds (1, 2, 4, 8) for `m` rows: the launch's own choice, no bits change with it.
fn tiles(m: usize) usize {
    return if (m <= 16) 0 else if (m <= 32) 1 else if (m <= 64) 2 else 3;
}

/// A linear's scratch: the activations' codes and scales, the split partials, the quant-dequant rows (wo_a's input).
pub const Scratch = struct {
    q: u64, // u8 [rows, max K]
    s: u64, // f32 [rows, max K / 32]
    z: u64, // f32 [zFloats]
    fake: u64, // bf16 [rows, max K]
    rows: usize,
    max_k: usize,

    /// Bytes of q, s, z and fake for `rows` rows of K up to `max_k` and the largest split partials `zf` floats.
    pub fn sizes(rows: usize, max_k: usize, zf: usize) [4]usize {
        return .{ rows * max_k, rows * (max_k / 32) * 4, @max(zf, 1) * 4, rows * max_k * 2 };
    }
};

/// Floats of split partials a linear needs (a slab of rows at most).
pub fn zFloats(l: weights.Linear, rows: usize) usize {
    if (l.fmt != .fp8 and l.fmt != .fp4) return 0;
    const s = splits(l.k, l.n);
    return if (s > 1) s * @min(rows, slab_rows) * l.n else 0;
}

/// TF_DS_ATTN_EPILOGUE: the attention output o [m, H * HD] bf16, in place: rope_heads' inverse RoPE of each head's
/// last rd columns, then act_quant's codes and scales of the roped rows into q / sc (dense rows), or its fake rows
/// into `fake` (the other two 0 then). The roped columns and the rest are whole 32-value groups. `store_o`: the roped
/// columns are written back into o (else o is left as the attention wrote it: its readers are the codes or fake rows).
pub fn attnEpilogue(k: *const Kernels, s: cuda.Stream, o: u64, cos: u64, sin: u64, pos: u64, m: usize, h: usize, hd: usize, rd: usize, q: u64, sc: u64, fake_rows: u64, store_o: bool) !void {
    const kk = h * hd;
    if (kk % 32 != 0 or hd % 32 != 0 or rd == 0 or rd > hd or rd % 32 != 0 or (hd - rd) % 32 != 0) return error.UnsupportedShape;
    var a: cuda.Args = .{};
    a.add(o);
    a.add(cos);
    a.add(sin);
    a.add(pos);
    a.add(@as(c_int, @intCast(m)));
    a.add(@as(c_int, @intCast(h)));
    a.add(@as(c_int, @intCast(hd)));
    a.add(@as(c_int, @intCast(rd)));
    a.add(q);
    a.add(@as(i64, @intCast(kk)));
    a.add(sc);
    a.add(@as(i64, @intCast(kk / 32)));
    a.add(fake_rows);
    a.add(@as(i64, @intCast(kk)));
    a.add(@as(c_int, @intFromBool(store_o)));
    try launch(k.attn_epilogue, s, .{ m, cdiv(kk / 8, 256), 1 }, 256, &a);
}

/// TF_DS_AQ_VEC in effect (Model.open; the replay's act_quant section toggles it): act_quant runs the four-lane twin tf_dsn_act_quant_vec_kernel where the call fits
/// (aqVecFits) and the image has it; false runs tf_dsn_act_quant_kernel (a warp a 32-group) for every call. The twin writes the same bytes (the replay compares every one).
pub var aq_vec_on: bool = true;

/// TF_DS_AQ_VEC as asked: unset or empty and "1" are on, "0" is off, null for any other text (Model.open refuses the start with it, at its call site; this parse is pure and logs nothing).
pub fn aqVecWanted(v: ?[]const u8) ?bool {
    const t = v orelse return true;
    if (t.len == 0 or std.mem.eql(u8, t, "1")) return true;
    if (std.mem.eql(u8, t, "0")) return false;
    return null;
}

/// Whether an act_quant call may take the twin: bf16 x, K a multiple of 32, x's row stride a multiple of 8 elements and x 16-byte aligned (its 16-byte loads), the codes' row
/// stride a multiple of 8 and q 8-byte aligned (its 8-byte stores); with `fake` (non-zero: the dequantized rows into it) fake's row stride a multiple of 8 and fake 16-byte
/// aligned (its 16-byte stores). Any other call runs the baseline kernel, which has no such condition.
pub fn aqVecFits(x_dtype: DType, x: u64, ldx: usize, kdim: usize, q: u64, ldq: usize, fake_ptr: u64, ldf: usize) bool {
    if (x_dtype != .bf16 or kdim == 0 or kdim % 32 != 0 or ldx % 8 != 0 or x % 16 != 0) return false;
    if (fake_ptr != 0) return ldf % 8 == 0 and fake_ptr % 16 == 0;
    return ldq % 8 == 0 and q % 8 == 0;
}

/// Whether this act_quant call runs the twin: the switch on, the image has it, the call fits.
fn aqVecTakes(k: *const Kernels, x_dtype: DType, x: u64, ldx: usize, kdim: usize, q: u64, ldq: usize, fake_ptr: u64, ldf: usize) bool {
    return aq_vec_on and k.act_quant_vec != null and aqVecFits(x_dtype, x, ldx, kdim, q, ldq, fake_ptr, ldf);
}

/// The one launch of act_quant (quant's and fake's): the twin at 4 threads a 32-group, else the baseline kernel at a warp a group. The same arguments either way.
fn aqLaunch(k: *const Kernels, s: cuda.Stream, a: *cuda.Args, x_dtype: DType, x: u64, ldx: usize, m: usize, kdim: usize, q: u64, ldq: usize, fake_ptr: u64, ldf: usize) !void {
    if (aqVecTakes(k, x_dtype, x, ldx, kdim, q, ldq, fake_ptr, ldf)) return launch(k.act_quant_vec.?, s, .{ cdiv(m * (kdim / 32) * 4, 256), 1, 1 }, 256, a);
    try launch(k.act_quant, s, .{ cdiv(m * (kdim / 32) * 32, 256), 1, 1 }, 256, a);
}

/// act_quant: x [M, K] (row stride ldx, dtype x_dtype) -> q [M, K] E4M3 (row stride ldq) and s [M, K / 32] fp32 (row
/// stride lds).
pub fn quant(k: *const Kernels, s: cuda.Stream, x: u64, x_dtype: DType, ldx: usize, m: usize, kdim: usize, q: u64, ldq: usize, sc: u64, lds: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_int, @backingInt(x_dtype)));
    a.add(@as(i64, @intCast(ldx)));
    a.add(@as(c_int, @intCast(m)));
    a.add(@as(c_int, @intCast(kdim)));
    a.add(q);
    a.add(@as(i64, @intCast(ldq)));
    a.add(sc);
    a.add(@as(i64, @intCast(lds)));
    a.add(@as(u64, 0));
    a.add(@as(i64, 0));
    try aqLaunch(k, s, &a, x_dtype, x, ldx, m, kdim, q, ldq, 0, 0);
}

/// quant for a decode launch (the same codes and scales): with dec_on tf_dsn_dec_quant, an ordinary launch (every
/// earlier kernel done when it starts) that lets the decode tensor-core kernel after it, a programmatic dependent
/// launch, start streaming its first weights meanwhile (that kernel waits for it before reading the codes).
pub fn quantDec(k: *const Kernels, s: cuda.Stream, x: u64, x_dtype: DType, ldx: usize, m: usize, kdim: usize, q: u64, ldq: usize, sc: u64, lds: usize) !void {
    if (!k.dec_on) return quant(k, s, x, x_dtype, ldx, m, kdim, q, ldq, sc, lds);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_int, @backingInt(x_dtype)));
    a.add(@as(i64, @intCast(ldx)));
    a.add(@as(c_int, @intCast(m)));
    a.add(@as(c_int, @intCast(kdim)));
    a.add(q);
    a.add(@as(i64, @intCast(ldq)));
    a.add(sc);
    a.add(@as(i64, @intCast(lds)));
    try cuda.launch.launch(k.dec_quant, .{ .grid = .{ .x = @intCast(cdiv(m * (kdim / 32) * 32, 256)), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
}

/// A quality gate's body canary on x (fp32 [n] in place, n whole 32-value groups), `mode` (prompt.qgate_body): 2 one more
/// FP8 step, act_quant's quant-dequant of each 32-value group (the value rounded to bf16, its E4M3 code at the group's
/// power-of-two scale, back: exact in fp32, the value fake writes); 3 each nonzero finite value one ulp away from zero
/// (its bits plus one; the largest finite values, zeros, infinities and NaN kept). Its kernel from useQgateBody.
pub fn qgateBody(k: *const Kernels, s: cuda.Stream, x: u64, n: usize, mode: u8) !void {
    const f = k.qgate_body_fn orelse return error.NoQgateBodyKernel;
    if (mode != 2 and mode != 3) return error.BadCanaryMode;
    if (n % 32 != 0) return error.CanaryRows;
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(n)));
    a.add(@as(c_int, mode));
    try launch(f, s, .{ cdiv(n, 256), 1, 1 }, 256, &a);
}

/// act_quant(..., inplace=True): x's quant-dequant rows into out [m, kdim] bf16 (row stride ldo).
pub fn fake(k: *const Kernels, s: cuda.Stream, x: u64, x_dtype: DType, ldx: usize, m: usize, kdim: usize, out: u64, ldo: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_int, @backingInt(x_dtype)));
    a.add(@as(i64, @intCast(ldx)));
    a.add(@as(c_int, @intCast(m)));
    a.add(@as(c_int, @intCast(kdim)));
    a.add(@as(u64, 0));
    a.add(@as(i64, 0));
    a.add(@as(u64, 0));
    a.add(@as(i64, 0));
    a.add(out);
    a.add(@as(i64, @intCast(ldo)));
    try aqLaunch(k, s, &a, x_dtype, x, ldx, m, kdim, 0, 0, out, ldo);
}

/// The MX GEMM of a quantized activation (q, sc: row strides ldq, lds) with a native FP8 / FP4 linear: y [m, n]
/// (row stride ldy, dtype; accumulate: fp32 y +=), split as splits(K, N), slabs of slab_rows rows. With
/// TF_DS_GEMM_HALVES a launch of more than gemm_halves_rows rows runs as row blocks of them, each block's kernel and
/// split chosen at the block rows (routeRows) and its grid over its own rows: a row's bits are the same at every row
/// count, so the blocks give the whole launch's bytes.
pub fn gemm(k: *const Kernels, s: cuda.Stream, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, y_dtype: DType, accumulate: bool, z: u64) !void {
    if (l.fmt != .fp8 and l.fmt != .fp4) return error.NotAnMxLinear;
    const sel = routeRows(m, gemm_halves);
    if (blockCount(m, gemm_halves) == 1) return gemmBlock(k, s, l, q, ldq, sc, lds, m, y, ldy, y_dtype, accumulate, z, sel);
    const esz: usize = switch (y_dtype) {
        .f32 => 4,
        else => 2,
    };
    var i: usize = 0;
    while (i < blockCount(m, gemm_halves)) : (i += 1) {
        const b = blockAt(m, gemm_halves, i);
        try gemmBlock(k, s, l, q + b.start * ldq, ldq, sc + b.start * lds * 4, lds, b.n, y + b.start * ldy * esz, ldy, y_dtype, accumulate, z, sel);
    }
}

/// gemm's launch of `m` rows (one of its row blocks): its kernel, split and variant chosen at `sel` rows (routeRows).
fn gemmBlock(k: *const Kernels, s: cuda.Stream, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, y_dtype: DType, accumulate: bool, z: u64, sel: usize) !void {
    const sp = splits(l.k, l.n);
    const esz: usize = switch (y_dtype) {
        .f32 => 4,
        else => 2,
    };
    if (sfRoute(k, l, q, ldq, sc, lds, y, ldy, accumulate, sel)) |kind| return sfGemm(k, s, kind, l, q, ldq, sc, lds, m, y, ldy, y_dtype, accumulate, z);
    if (pfDense(k, l, q, ldq, sc, lds, sel, y, ldy, accumulate)) {
        // the prefill tile GEMM: every row at once, a split folded in the CTA (the tile kernel's and reduce's bits)
        const a: MxArgs = .{ .q = q, .s = sc, .ldq = @intCast(ldq), .lds = @intCast(lds), .w = l.w, .ws = l.s, .ldw = l.k, .ldws = l.k / 32, .fmt = 0, .M = @intCast(m), .N = @intCast(l.n), .K = @intCast(l.k), .splits = @intCast(sp), .kblocks = @intCast(cdiv(l.k / 32, sp)), .y = y, .ldy = @intCast(ldy), .y_dtype = @backingInt(y_dtype), .accumulate = @intFromBool(accumulate), .raster = rasterOf(pf_raster, m, l.k) };
        var args: cuda.Args = .{};
        args.add(a);
        const stage = stagedY(k, y, ldy, y_dtype, accumulate);
        return switch (pfGemmKernel(k.split, k.gemm5, sel, l.k, l.n)) {
            .gemm => launchShared(if (stage) k.st_gemm.? else k.pf_gemm, s, .{ cdiv(m, 128), l.n / 128, 1 }, pf_gemm_smem, &args),
            .gemm5 => launchShared((if (stage) k.st_gemm5 else k.pf_gemm5) orelse return error.NoGemm5Kernel, s, .{ cdiv(m, 128), l.n / 128, 1 }, pf_gemm5_smem, &args),
            .fold => launchShared(if (stage) k.st_fold.? else k.pf_gemm_fold, s, .{ cdiv(m, 64), l.n / 128, 1 }, pf_gemm_smem, &args),
            .split => |c| launchShared((if (stage) k.st_split[@backingInt(c)] else k.pf_split[@backingInt(c)]) orelse return error.NoSplitKernel, s, .{ cdiv(m, split_bm[@backingInt(c)]), l.n / 128, 1 }, split_smem[@backingInt(c)], &args),
        };
    }
    var r0: usize = 0;
    while (r0 < m) : (r0 += slab_rows) {
        const mm = @min(slab_rows, m - r0);
        const t = tiles(mm);
        const a: MxArgs = .{
            .q = q + r0 * ldq,
            .s = sc + r0 * lds * 4,
            .ldq = @intCast(ldq),
            .lds = @intCast(lds),
            .w = l.w,
            .ws = l.s,
            .ldw = if (l.fmt == .fp4) l.k / 2 else l.k,
            .ldws = if (l.fmt == .fp4) l.k / 32 else l.k / 32,
            .fmt = if (l.fmt == .fp4) 1 else 0,
            .M = @intCast(mm),
            .N = @intCast(l.n),
            .K = @intCast(l.k),
            .splits = @intCast(sp),
            .kblocks = @intCast(cdiv(l.k / 32, sp)),
            .z = z,
            .y = y + r0 * ldy * esz,
            .ldy = @intCast(ldy),
            .y_dtype = @backingInt(y_dtype),
            .accumulate = @intFromBool(accumulate),
        };
        var args: cuda.Args = .{};
        args.add(a);
        const rows_cta = (@as(usize, 1) << @intCast(t)) * 16;
        try launch(k.gemm[t], s, .{ cdiv(l.n / 32, 4), sp, cdiv(mm, rows_cta) }, 128, &args);
        if (sp > 1) {
            var ra: cuda.Args = .{};
            ra.add(z);
            ra.add(@as(c_int, @intCast(sp)));
            ra.add(@as(c_int, @intCast(mm)));
            ra.add(@as(c_int, @intCast(l.n)));
            ra.add(a.y);
            ra.add(@as(i64, @intCast(ldy)));
            ra.add(@as(c_int, @backingInt(y_dtype)));
            ra.add(@as(c_int, @intFromBool(accumulate)));
            try launch(k.reduce, s, .{ cdiv(mm * l.n, 256), 1, 1 }, 256, &ra);
        }
    }
}

/// The decode GEMV of a quantized activation with a native FP8 / FP4 linear (gemm's arguments, no split): a warp a
/// column, each row its own arithmetic, so a row's bits are the same at any row count.
pub fn gemv(k: *const Kernels, s: cuda.Stream, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, y_dtype: DType, accumulate: bool, z: ?Z) !void {
    if (l.fmt != .fp8 and l.fmt != .fp4) return error.NotAnMxLinear;
    const f: u2 = if (l.fmt == .fp4) 1 else 0;
    if (decTakes(k, f, l.n, l.k)) return decDense(k, s, f, .{
        .q = q,
        .s = sc,
        .ldq = @intCast(ldq),
        .lds = @intCast(lds),
        .w = l.w,
        .ws = l.s,
        .ldw = if (l.fmt == .fp4) l.k / 2 else l.k,
        .ldws = l.k / 32,
        .fmt = f,
        .M = @intCast(m),
        .N = @intCast(l.n),
        .K = @intCast(l.k),
        .y = y,
        .ldy = @intCast(ldy),
        .y_dtype = @backingInt(y_dtype),
        .accumulate = @intFromBool(accumulate),
    }, z);
    const a: MxArgs = .{
        .q = q,
        .s = sc,
        .ldq = @intCast(ldq),
        .lds = @intCast(lds),
        .w = l.w,
        .ws = l.s,
        .ldw = if (l.fmt == .fp4) l.k / 2 else l.k,
        .ldws = l.k / 32,
        .fmt = if (l.fmt == .fp4) 1 else 0,
        .M = @intCast(m),
        .N = @intCast(l.n),
        .K = @intCast(l.k),
        .y = y,
        .ldy = @intCast(ldy),
        .y_dtype = @backingInt(y_dtype),
        .accumulate = @intFromBool(accumulate),
    };
    var args: cuda.Args = .{};
    args.add(a);
    if (k.sab) return launch(k.mx_sab, s, .{ cdiv(l.n, 16), 1, 1 }, 256, &args);
    try launch(k.gemv, s, .{ cdiv(l.n, 8), 1, 1 }, 256, &args);
}

/// A bf16 linear with fp32 accumulation: y [m, n] = x [m, K] bf16 . w [n, K]^T (F.linear of bf16 values), scaled by
/// `scale` (round_first: of the bf16-rounded sum, as a bf16 Linear's output times a constant), `groups` block-diagonal
/// groups (x + g gx, w + g gw, y + g gy elements).
/// With `z` (a decode launch: one group, its rows within z) K is split as bfSplits(kdim, n), the partials summed in
/// split order; without it the K loop runs whole (prompt chunks). With TF_DS_GEMM_HALVES a launch of more than
/// gemm_halves_rows rows runs as row blocks of them, each block's kernel chosen at the pass's block cap (routeRows)
/// and its grid over its own rows.
pub fn bf16Gemm(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: u64, ldw: usize, m: usize, n: usize, kdim: usize, y: u64, ldy: usize, y_dtype: DType, scale: f32, round_first: bool, groups: usize, gx: usize, gw: usize, gy: usize, z: ?Z) !void {
    if (n % 32 != 0 or kdim % 16 != 0) return error.UnsupportedShape;
    const sel = routeRows(m, gemm_halves);
    if (blockCount(m, gemm_halves) == 1) return bf16Block(k, s, x, ldx, w, ldw, m, n, kdim, y, ldy, y_dtype, scale, round_first, groups, gx, gw, gy, z, sel);
    const ysz: usize = if (y_dtype == .f32) 4 else 2;
    var i: usize = 0;
    while (i < blockCount(m, gemm_halves)) : (i += 1) {
        const b = blockAt(m, gemm_halves, i);
        try bf16Block(k, s, x + b.start * ldx * 2, ldx, w, ldw, b.n, n, kdim, y + b.start * ldy * ysz, ldy, y_dtype, scale, round_first, groups, gx, gw, gy, z, sel);
    }
}

/// The launch of a prompt chunk's bf16 GEMM under TF_DS_PF_T64 (`on`): the 64 x 64 tile for the router's 384 columns of a single group,
/// else null (the 128 x 128 tile). Its grid is (cdiv(m, 64), n / 64, 1) at 128 threads.
pub fn pfT64Of(k: *const Kernels, on: bool, n: usize, groups: usize) ?cuda.Function {
    if (!on or groups != 1 or n != 384) return null;
    return k.pf_bf16_t64;
}

/// bf16Gemm's launch of `m` rows (one of its row blocks): its kernel chosen at `sel` rows (routeRows).
fn bf16Block(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: u64, ldw: usize, m: usize, n: usize, kdim: usize, y: u64, ldy: usize, y_dtype: DType, scale: f32, round_first: bool, groups: usize, gx: usize, gw: usize, gy: usize, z: ?Z, sel: usize) !void {
    if (z != null and groups == 1 and m > decode_rows) {
        // a decode launch past decode_rows rows (the drafter's router at 13+ streams): slabs of decode_rows rows, each
        // as a launch of its own rows (the split partials are sized for decode_rows rows; a row's bits are its own)
        const ysz: usize = if (y_dtype == .f32) 4 else 2;
        var r0: usize = 0;
        while (r0 < m) : (r0 += decode_rows)
            try bf16Block(k, s, x + r0 * ldx * 2, ldx, w, ldw, @min(decode_rows, m - r0), n, kdim, y + r0 * ldy * ysz, ldy, y_dtype, scale, round_first, 1, gx, gw, gy, z, sel);
        return;
    }
    if (z != null and groups == 1 and decTakes(k, 2, n, kdim)) return decDense(k, s, 2, .{
        .q = x,
        .ldq = @intCast(ldx),
        .w = w,
        .ldw = @intCast(ldw * 2),
        .fmt = 2,
        .M = @intCast(m),
        .N = @intCast(n),
        .K = @intCast(kdim),
        .y = y,
        .ldy = @intCast(ldy),
        .y_dtype = @backingInt(y_dtype),
        .scale = scale,
        .round_first = @intFromBool(round_first),
    }, z);
    var a: BfArgs = .{ .x = x, .ldx = @intCast(ldx), .gx = @intCast(gx), .w = w, .ldw = @intCast(ldw), .gw = @intCast(gw), .M = @intCast(m), .N = @intCast(n), .K = @intCast(kdim), .y = y, .ldy = @intCast(ldy), .gy = @intCast(gy), .y_dtype = @backingInt(y_dtype), .scale = scale, .round_first = @intFromBool(round_first), .raster = rasterOf(pf_raster, m, kdim * 2) };
    const sp = if (z != null) bfSplits(kdim, n) else 1;
    if (sp > 1) {
        const zz = z.?;
        if (groups != 1) return error.UnsupportedShape;
        // the partials' buffer bounds the rows (a round's window, or the drafter's pass of every stream's block)
        if (sp * m * n > zz.floats) return error.ScratchTooSmall;
        a.splits = @intCast(sp);
        a.ksteps = @intCast(cdiv(kdim / 16, sp));
        a.z = zz.ptr;
    }
    var args: cuda.Args = .{};
    args.add(a);
    if (sp == 1 and !k.pf_off and sel >= pf_min_rows and n % 128 == 0 and kdim % 32 == 0 and aligned(x, 16) and ldx % 8 == 0 and
        gx % 8 == 0 and aligned(w, 16) and ldw % 8 == 0 and gw % 8 == 0)
    {
        // the prefill bf16 GEMM: the same MMA chain (the tile kernel's bits), staged tiles
        if (pfT64Of(k, k.pf_t64, n, groups)) |f| return launchSharedT(f, s, .{ cdiv(m, 64), n / 64, 1 }, pf_bf16_t64_smem, 128, &args);
        return launchShared(k.pf_bf16, s, .{ cdiv(m, 128), n / 128, groups }, pf_bf16_smem, &args);
    }
    try launch(k.bf16, s, .{ cdiv(n / 32, 4), bfGridRows(k.bf16_rows, z == null, m), if (sp > 1) sp else groups }, 128, &args);
    if (sp > 1) {
        var ra: cuda.Args = .{};
        ra.add(a);
        try launch(k.bf_reduce, s, .{ cdiv(m * n, 256), 1, 1 }, 256, &ra);
    }
}

/// tf_dsn_bf16_gemm's CTAs over m rows (grid y): at most 64, each walking its 16-row tiles gridDim.y apart; with
/// TF_DS_BF16_ROWS (`rows_on`) a prompt launch (`prompt`: no decode split) takes one a tile. A tile's chain is the same
/// whichever CTA runs it, so either grid gives the same bits.
pub fn bfGridRows(rows_on: bool, prompt: bool, m: usize) usize {
    return if (rows_on and prompt) cdiv(m, 16) else @min(cdiv(m, 16), 64);
}

/// A native linear on m rows of x (dtype x_dtype, row stride ldx) into y (row stride ldy, y_dtype): FP8 / FP4 through
/// act_quant and the MX GEMM, bf16 through the bf16 GEMM (x must be bf16 then). decode (a decode launch: z its split
/// partials): FP8 / FP4 through the GEMV, bf16 split as bfSplits.
pub fn linear(k: *const Kernels, s: cuda.Stream, sc: Scratch, l: weights.Linear, x: u64, x_dtype: DType, ldx: usize, m: usize, y: u64, ldy: usize, y_dtype: DType, accumulate: bool, decode: ?Z) !void {
    switch (l.fmt) {
        .fp8, .fp4 => {
            if (m > sc.rows or l.k > sc.max_k) return error.ScratchTooSmall;
            try quant(k, s, x, x_dtype, ldx, m, l.k, sc.q, l.k, sc.s, l.k / 32);
            if (decode != null) return gemv(k, s, l, sc.q, l.k, sc.s, l.k / 32, m, y, ldy, y_dtype, accumulate, decode);
            try gemm(k, s, l, sc.q, l.k, sc.s, l.k / 32, m, y, ldy, y_dtype, accumulate, sc.z);
        },
        .bf16 => {
            if (x_dtype != .bf16 or accumulate) return error.UnsupportedInput;
            try bf16Gemm(k, s, x, ldx, l.w, l.k, m, l.n, l.k, y, ldy, y_dtype, 1, false, 1, 0, 0, 0, decode);
        },
        .exl3 => return error.NotANativeLinear,
    }
}

/// A native linear's DecArgs on m rows: FP8 / FP4 of quantized rows (q, sc; row strides ldq bytes, lds floats), bf16
/// of bf16 rows at q (ldq elements).
fn decArgsOf(l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, y_dtype: DType) DecArgs {
    return .{
        .q = q,
        .s = if (l.fmt == .bf16) 0 else sc,
        .ldq = @intCast(ldq),
        .lds = @intCast(lds),
        .w = l.w,
        .ws = l.s,
        .ldw = switch (l.fmt) {
            .fp4 => l.k / 2,
            .bf16 => l.k * 2,
            else => l.k,
        },
        .ldws = l.k / 32,
        .fmt = switch (l.fmt) {
            .fp4 => 1,
            .bf16 => 2,
            else => 0,
        },
        .M = @intCast(m),
        .N = @intCast(l.n),
        .K = @intCast(l.k),
        .y = y,
        .ldy = @intCast(ldy),
        .y_dtype = @backingInt(y_dtype),
    };
}

fn decKind(l: weights.Linear) ?u2 {
    return switch (l.fmt) {
        .fp8 => 0,
        .fp4 => 1,
        .bf16 => 2,
        .exl3 => null,
    };
}

/// Decode linears on m rows (mx.linear decode for each, bf16 x): the FP8 / FP4 ones sharing an input quantized once
/// (consecutive linears of the same x, row stride and K read the same codes: act_quant's are a function of the row),
/// and consecutive linears of one format and one input on the tensor-core kernels in one launch (each as alone).
/// Past decode_rows rows (the drafter's pass of 13 or more streams) in slabs of decode_rows rows, each slab as a call of
/// its own rows (codes and scales are functions of the row; the decode kernels' and GEMV's row bits do not depend on the
/// row count), never the prompt GEMMs.
pub fn linearsDecode(k: *const Kernels, s: cuda.Stream, sc: Scratch, ls: []const weights.Linear, xs: []const u64, ldxs: []const usize, m: usize, outs: []const u64, ldos: []const usize, types: []const DType, z: Z) !void {
    if (m > decode_rows) {
        if (ls.len > 8) return error.UnsupportedShape;
        var r0: usize = 0;
        while (r0 < m) : (r0 += decode_rows) {
            var sx: [8]u64 = undefined;
            var so: [8]u64 = undefined;
            for (0..ls.len) |i| {
                sx[i] = xs[i] + r0 * ldxs[i] * 2;
                so[i] = outs[i] + r0 * ldos[i] * @as(usize, if (types[i] == .f32) 4 else 2);
            }
            try linearsDecode(k, s, sc, ls, sx[0..ls.len], ldxs, @min(decode_rows, m - r0), so[0..ls.len], ldos, types, z);
        }
        return;
    }
    var i: usize = 0;
    while (i < ls.len) {
        const l = ls[i];
        const kind = decKind(l) orelse return error.NotANativeLinear;
        if (!decTakes(k, kind, l.n, l.k)) {
            try linear(k, s, sc, l, xs[i], .bf16, ldxs[i], m, outs[i], ldos[i], types[i], false, z);
            i += 1;
            continue;
        }
        // the run of linears on this input that the launch takes
        var j = i + 1;
        while (j < ls.len and j - i < 4) : (j += 1) {
            const o = ls[j];
            const ok = decKind(o) == kind and decTakes(k, kind, o.n, o.k) and o.k == l.k and xs[j] == xs[i] and ldxs[j] == ldxs[i];
            if (!ok) break;
            // (an earlier output over this input: not one launch)
            var clash = false;
            for (i..j) |p| clash = clash or outs[p] == xs[i];
            if (clash) break;
        }
        var as: [4]DecArgs = undefined;
        if (kind == 2) {
            for (i..j) |p| as[p - i] = decArgsOf(ls[p], xs[p], ldxs[p], 0, 0, m, outs[p], ldos[p], types[p]);
        } else {
            if (m > sc.rows or l.k > sc.max_k) return error.ScratchTooSmall;
            try quantDec(k, s, xs[i], .bf16, ldxs[i], m, l.k, sc.q, l.k, sc.s, l.k / 32);
            for (i..j) |p| as[p - i] = decArgsOf(ls[p], sc.q, l.k, sc.s, l.k / 32, m, outs[p], ldos[p], types[p]);
        }
        try decDenseMulti(k, s, kind, as[0 .. j - i], z, kind != 2);
        i = j;
    }
}

/// wo_a's groups (Attention.forward: act_quant(o, inplace) of the heads' rows, then the block-diagonal bf16 einsum):
/// o [m, groups * gk] bf16 (row stride ldo) quant-dequantized into sc.fake, each group's columns through its bf16
/// weight into u's column block (u row stride ldu, bf16).
pub fn woA(k: *const Kernels, s: cuda.Stream, sc: Scratch, wo: []const weights.Linear, o: u64, ldo: usize, m: usize, u: u64, ldu: usize, z: ?Z) !void {
    const gk: usize = wo[0].k;
    const kk = gk * wo.len;
    if (m > sc.rows or kk > sc.max_k) return error.ScratchTooSmall;
    try fake(k, s, o, .bf16, ldo, m, kk, sc.fake, kk);
    try woAGemm(k, s, sc, wo, m, kk, u, ldu, z);
}

/// woA's groups on sc.fake already holding its quant-dequant rows (TF_DS_ATTN_EPILOGUE wrote them): the same launches
/// as woA, without the fake quant's read of o.
pub fn woAFromFake(k: *const Kernels, s: cuda.Stream, sc: Scratch, wo: []const weights.Linear, m: usize, u: u64, ldu: usize, z: ?Z) !void {
    const gk: usize = wo[0].k;
    const kk = gk * wo.len;
    if (m > sc.rows or kk > sc.max_k) return error.ScratchTooSmall;
    try woAGemm(k, s, sc, wo, m, kk, u, ldu, z);
}

fn woAGemm(k: *const Kernels, s: cuda.Stream, sc: Scratch, wo: []const weights.Linear, m: usize, kk: usize, u: u64, ldu: usize, z: ?Z) !void {
    const gk: usize = wo[0].k;
    var col: usize = 0;
    for (wo, 0..) |l, g| {
        if (l.fmt != .bf16 or l.k != gk) return error.NotANativeLinear;
        try bf16Gemm(k, s, sc.fake + g * gk * 2, kk, l.w, l.k, m, l.n, l.k, u + col * 2, ldu, .bf16, 1, false, 1, 0, 0, 0, z);
        col += l.n;
    }
}

/// woA of a prompt chunk's layer the loader left without its bf16 rows (weights.Layer.wo_a_rows false: TF_DS_WOA_W8),
/// on its FP8 rows (Layer.wo_a8, every group stacked): o [m, groups * gk] bf16 (row stride ldo) quantized once
/// (act_quant: the codes and scales whose quant-dequant values woA's fake rows hold), then each group's columns through
/// tf_dsn_pf_bf16w8 into u's column block (row stride ldu, bf16). The kernel multiplies the values woA multiplies (each
/// code times its scale rounded to bf16; each weight's E4M3 value times its block scale cut to bf16, the loader's rows)
/// on pf_bf16's MMA chain, at every row count: woA's bits. With TF_DS_GEMM_HALVES a launch of more than
/// gemm_halves_rows rows runs as row blocks of them, each block its own launch over its rows.
pub fn woA8Prefill(k: *const Kernels, s: cuda.Stream, sc: Scratch, w8: weights.Linear, groups: usize, o: u64, ldo: usize, m: usize, u: u64, ldu: usize) !void {
    const gk: usize = w8.k;
    const kk = gk * groups;
    if (m > sc.rows or kk > sc.max_k) return error.ScratchTooSmall;
    if (!w8Shape(w8, groups)) return error.UnsupportedShape;
    try quant(k, s, o, .bf16, ldo, m, kk, sc.q, kk, sc.s, kk / 32);
    try woA8Groups(k, s, sc, w8, groups, m, u, ldu);
}

/// woA8Prefill's group launches on sc.q / sc.s already holding o's codes and scales (TF_DS_ATTN_EPILOGUE wrote them):
/// the same launches, without the quant's read of o.
pub fn woA8PrefillCodes(k: *const Kernels, s: cuda.Stream, sc: Scratch, w8: weights.Linear, groups: usize, m: usize, u: u64, ldu: usize) !void {
    const gk: usize = w8.k;
    const kk = gk * groups;
    if (m > sc.rows or kk > sc.max_k) return error.ScratchTooSmall;
    if (!w8Shape(w8, groups)) return error.UnsupportedShape;
    try woA8Groups(k, s, sc, w8, groups, m, u, ldu);
}

/// TF_DS_WOA_MERGE in effect (Model.open; the replay's woa_merge section toggles it): a prompt chunk's wo_a groups on their FP8 rows (woA8Groups) run as ONE launch of tf_dsn_pf_bf16w8 with grid.z =
/// groups instead of one launch a group: a layer's two 5.33-wave launches (256 CTAs on 48 SMs, 6 waves each) become one of 10.67 waves (11), the last wave of the first no longer idle. Each CTA runs its
/// group's code unchanged: the bytes are the group launches'.
pub var woa_merge: bool = true;

/// TF_DS_WOA_MERGE as asked: unset, empty and "1" on (the default since #417: x0.945 a layer on the Sparks, the same bytes), "0" off (the operator's choice), null for any other text (Model.open refuses
/// the start with it at its call site; this parse is pure and logs nothing).
pub fn woaMergeWanted(text: ?[]const u8) ?bool {
    const t = text orelse return true;
    if (t.len == 0 or std.mem.eql(u8, t, "1")) return true;
    if (std.mem.eql(u8, t, "0")) return false;
    return null;
}

/// The byte strides between wo_a's groups for group width `gn` (output columns) and group depth `gk` (K): the codes' columns, the activation scales, the weight codes, the weight scales (32 x 32 blocks) and
/// the bf16 output columns: the merged launch's per-group offsets; a test holds them to the per-group launches' (g * gn * gk, g * (gn / 32) * (gk / 32), ...).
pub const GroupStrides = struct { gq: i64, gs: i64, gw: i64, gws: i64, gy: i64 };
pub fn woaGroupStrides(gk: usize, gn: usize) GroupStrides {
    return .{
        .gq = @intCast(gk),
        .gs = @intCast(gk / 32 * 4),
        .gw = @intCast(gn * gk),
        .gws = @intCast(gn / 32 * (gk / 32)),
        .gy = @intCast(gn * 2),
    };
}

fn woA8Groups(k: *const Kernels, s: cuda.Stream, sc: Scratch, w8: weights.Linear, groups: usize, m: usize, u: u64, ldu: usize) !void {
    // TF_DS_GEMM_W16: the 16-warp twin (the same tile, stages and bytes, a warp's 32 rows of it)
    const w16 = if (k.gemm_w16) k.pf_bf16w8_w16 else null;
    const f = w16 orelse k.pf_bf16w8 orelse return error.NoW8Kernel;
    const gk: usize = w8.k;
    const kk = gk * groups;
    const gn = w8.n / groups;
    const st = woaGroupStrides(gk, gn);
    // TF_DS_GEMM_HALVES: row blocks of gemm_halves_rows at most, each block its own launch (the kernel's bits are a
    // row's own), the codes and scales read at the block's rows. TF_DS_PF_RASTER: each launch's slab is its rows of one
    // group's codes (b.n * gk bytes), the group applying from raster_slab_bytes
    var i: usize = 0;
    while (i < blockCount(m, gemm_halves)) : (i += 1) {
        const b = blockAt(m, gemm_halves, i);
        const merged = woa_merge and groups > 1;
        for (0..if (merged) 1 else groups) |g| {
            const a: W8Args = .{
                .q = sc.q + b.start * kk + g * gk,
                .s = sc.s + b.start * (kk / 32) * 4 + g * (gk / 32) * 4,
                .ldq = @intCast(kk),
                .lds = @intCast(kk / 32),
                .w = w8.w + g * gn * gk,
                .ws = w8.s + g * (gn / 32) * (gk / 32),
                .ldw = @intCast(gk),
                .ldws = @intCast(gk / 32),
                .M = @intCast(b.n),
                .N = @intCast(gn),
                .K = @intCast(gk),
                .y_dtype = @backingInt(DType.bf16),
                .y = u + b.start * ldu * 2 + g * gn * 2,
                .ldy = @intCast(ldu),
                .raster = rasterOf(pf_raster, b.n, gk),
                .gq = if (merged) st.gq else 0,
                .gs = if (merged) st.gs else 0,
                .gw = if (merged) st.gw else 0,
                .gws = if (merged) st.gws else 0,
                .gy = if (merged) st.gy else 0,
            };
            var args: cuda.Args = .{};
            args.add(a);
            try launchSharedT(f, s, .{ cdiv(b.n, 128), gn / 128, if (merged) groups else 1 }, pf_w8_smem, if (w16 != null) 512 else 256, &args);
        }
    }
}

/// Whether tf_dsn_pf_bf16w8 takes a layer's stacked FP8 wo_a: each group's outputs whole 128-column tiles, K whole
/// 32-column stages (16-byte rows) and at most weights.w8_max_k (its tile's scales in the kernel's static array), the
/// groups' rows whole 32-row scale blocks.
pub fn w8Shape(w8: weights.Linear, groups: usize) bool {
    if (groups == 0 or w8.fmt != .fp8 or w8.n % groups != 0) return false;
    const gn = w8.n / groups;
    return gn % 128 == 0 and w8.k % 32 == 0 and w8.k > 0 and w8.k <= weights.w8_max_k;
}

/// Why TF_DS_WOA_W8 asked for cannot run on this rank (null: it can): the checkpoint's own weights, the prompt chunks'
/// tile kernels and the decode tensor-core kernels (without them a round reads wo_a's bf16 rows).
pub fn woaW8Off(native: bool, pf_off: bool, dec_on: bool) ?[]const u8 {
    if (!native) return "needs the checkpoint's own FP8 / MXFP4 weights";
    if (pf_off) return "needs the prompt chunks' tile kernels (TF_DS_PF_TILES=0 turns them off)";
    if (!dec_on) return "needs the decode tensor-core kernels (TF_DS_DECODE_DEC=0 reads wo_a's bf16 rows)";
    return null;
}

/// wo_a of a decode launch on its FP8 rows (weights.Layer.wo_a8, every group stacked): o [m, groups * gk] bf16 (row
/// stride ldo) quantized once (act_quant: the codes and scales the reference's quant-dequant rows hold), every group's
/// columns in one launch, group g's output columns reading its input columns, into u [m, groups * o_lora] bf16 (row
/// stride ldu). The same products as woA's bf16 einsum of the quant-dequant rows (both exact), summed in fp32.
pub fn woA8(k: *const Kernels, s: cuda.Stream, sc: Scratch, w8: weights.Linear, groups: usize, o: u64, ldo: usize, m: usize, u: u64, ldu: usize, z: ?Z) !void {
    const gk: usize = w8.k;
    const kk = gk * groups;
    if (m > sc.rows or kk > sc.max_k) return error.ScratchTooSmall;
    if (!decTakes(k, 0, w8.n, gk) or (w8.n / groups) % 16 != 0) return error.UnsupportedShape;
    try quantDec(k, s, o, .bf16, ldo, m, kk, sc.q, kk, sc.s, kk / 32);
    try decDenseMulti(k, s, 0, &.{.{
        .q = sc.q,
        .s = sc.s,
        .ldq = @intCast(kk),
        .lds = @intCast(kk / 32),
        .w = w8.w,
        .ws = w8.s,
        .ldw = @intCast(gk),
        .ldws = @intCast(gk / 32),
        .fmt = 0,
        .M = @intCast(m),
        .N = @intCast(w8.n),
        .K = @intCast(gk),
        .y = u,
        .ldy = @intCast(ldu),
        .y_dtype = @backingInt(DType.bf16),
        .gn = @intCast(w8.n / groups),
        .gq = @intCast(gk),
        .gs = @intCast(gk / 32),
    }}, z, true);
}

/// The experts' scratch (the routed rows' groups, their quantized inputs and intermediate, the slots' outputs) for
/// up to `rows` rows of `slots` slots, `e` experts, D inputs, I the whole intermediate a node's down reads, N outputs.
pub const ExpertScratch = struct {
    rows: usize,
    slots: usize,
    q: u64, // u8 [rows, D]
    s: u64, // f32 [rows, D / 32]
    cnt: u64, // i32 [E]
    off: u64, // i32 [E + 1]
    fill: u64, // i32 [E]
    act: u64, // i32 [E]
    nact: u64, // i32 [1]
    ent: u64, // i32 [rows * slots]
    erow: u64,
    hq: u64, // u8 [rows * slots, I]
    hs: u64, // f32 [rows * slots, I / 32]
    out: u64, // f32 [rows * slots, N]
    z: Z = .{ .ptr = 0, .floats = 0 }, // decode split partials (decodeZFloats; none: no decode launches split)
    // the prefill tiles' list (int2 [rows * slots / pf_bm + E]) and its length (i32 [1]): none, prompt chunks take the
    // tile kernels
    tiles: u64 = 0,
    ntiles: u64 = 0,
    /// down's activation read from the pack's two parts where they lie (TF_DS_ROCI_CODES, threeTakes): none, down reads hq / hs
    three: ?Three = null,

    /// Bytes of the prefill tile list for `rows` rows of `slots` slots and `e` experts.
    pub fn tileBytes(rows: usize, slots: usize, e: usize) usize {
        return (rows * slots / pf_bm + e) * 8;
    }

    pub fn sizes(rows: usize, slots: usize, e: usize, d: usize, i: usize, n: usize) [12]usize {
        const p = rows * slots;
        return .{ rows * d, rows * d / 32 * 4, 4 * e, 4 * (e + 1), 4 * e, 4 * e, 4, 4 * p, 4 * p, p * i, p * i / 32 * 4, p * n * 4 };
    }
};

/// TF_DS_NACT_LOG (log only, off by default): the device ring tf_dsn_nact_log_kernel appends to (0: off: no launch, no copy), its records, and the
/// pure decode / render of them. See native.cu for the layout.
pub var nact_ring: u64 = 0;
pub const nact_records = 4096;
pub const NactRec = struct { call: u64, rows: u32, active: u32, owned: u32, shared: u32 };

pub fn nactDecode(words: []const u64, out: []NactRec) usize {
    var n: usize = 0;
    var i: usize = 8; // after the counter and its header
    while (i + 3 < words.len and n < out.len) : (i += 4) {
        if (words[i] == 0) continue;
        out[n] = .{ .call = words[i] - 1, .rows = @truncate(words[i + 1] >> 32), .active = @truncate(words[i + 1]), .owned = @truncate(words[i + 2]), .shared = @truncate(words[i + 2] >> 32) };
        n += 1;
    }
    return n;
}

pub fn nactRender(gpa: std.mem.Allocator, rank: u32, world: u32, recs: []NactRec) ![]u8 {
    std.mem.sort(NactRec, recs, {}, struct {
        fn less(_: void, a: NactRec, b: NactRec) bool {
            return a.call < b.call;
        }
    }.less);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.print(gpa, "# TF_DS_NACT_LOG v1 rank={d} world={d} records={d} columns=call rows active owned shared (shared: 0 not active, 1 active and unowned, 2 active and owned)\n", .{ rank, world, recs.len });
    for (recs) |q| try out.print(gpa, "{d} {d} {d} {d} {d}\n", .{ q.call, q.rows, q.active, q.owned, q.shared });
    return out.toOwnedSlice(gpa);
}

/// A decode launch's grouping (r rows of sc.slots picks, r * slots <= 4096) in one CTA: sc.off, act, nact, ent and erow.
/// k.group2 (TF_DS_DEC_GROUP2, on unless set to 0): the scan-and-rank kernel, the same bytes, for up to 1,024 experts; else the one-CTA loop.
pub fn decodeGrouping(k: *const Kernels, s: cuda.Stream, e: usize, sc: ExpertScratch, pick: u64, r: usize) !void {
    var g: cuda.Args = .{};
    g.add(pick);
    g.add(@as(c_int, @intCast(r * sc.slots)));
    g.add(@as(c_int, @intCast(sc.slots)));
    g.add(@as(c_int, @intCast(e)));
    g.add(sc.off);
    g.add(sc.act);
    g.add(sc.nact);
    g.add(sc.ent);
    g.add(sc.erow);
    const f = if (k.group2 and e <= 1024) k.dec_group2 else k.dec_group;
    try cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024 }, .shared = @intCast((2 * e + 1) * 4) }, s, &g);
}

/// TF_DS_NACT_LOG: this decode launch's grouping into the log (after decodeGrouping, on the same stream); only called with the log on.
fn nactLog(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, r: usize) !void {
    const f = k.nact_log orelse return;
    var g: cuda.Args = .{};
    g.add(sc.act);
    g.add(sc.nact);
    g.add(if (ex.native) |nx| nx.owner else @as(u64, 0));
    g.add(@as(c_int, @intCast(ex.count - 1)));
    g.add(@as(c_int, @intCast(r)));
    g.add(nact_ring);
    try cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &g);
}

/// The routed rows grouped by expert (picks pick [r, slots]; ids past the experts are another node's), and x's rows
/// quantized: what gate / up and down read. `coded`: x's codes and scales are in sc.q / sc.s already (a prompt
/// chunk's, not one_cta), the quant skipped.
fn group(k: *const Kernels, s: cuda.Stream, d: *const cuda.Driver, ex: weights.Experts, sc: ExpertScratch, x: u64, ldx: usize, pick: u64, r: usize, one_cta: bool, coded: bool) !void {
    const e: usize = ex.count;
    const rs = r * sc.slots;
    if (one_cta and rs <= 4096) {
        // a decode launch: the count, prefix and places in one CTA (the same groups, entries in entry order)
        try decodeGrouping(k, s, e, sc, pick, r);
        if (nact_ring != 0) try nactLog(k, s, ex, sc, r);
        // x's codes last: gate / up (a programmatic dependent launch) reads the groups before it waits for them
        return quantDec(k, s, x, .bf16, ldx, r, ex.dims, sc.q, ex.dims, sc.s, ex.dims / 32);
    }
    if (!coded) try quant(k, s, x, .bf16, ldx, r, ex.dims, sc.q, ex.dims, sc.s, ex.dims / 32);
    try grouping(k, s, d, e, sc, pick, r);
}

/// Whether a grouping on scratch `sc` builds the prefill tile list itself (TF_DS_GROUP_TILES with a tile list: the
/// prompt chunk's scratch), so gate / up and down take it as built; else each builds it (tileList).
pub fn tilesOnce(group_tiles: bool, sc: ExpertScratch) bool {
    return group_tiles and sc.tiles != 0;
}

/// A prompt chunk's grouping of the picks pick [r, slots] over `e` experts (ids past them another node's): the counts,
/// their prefix and the active list (tf_dsn_group_scan; with tilesOnce tf_dsn_group_tiles, which also writes the tile
/// list), then each pick's entry (tf_dsn_group_place).
pub fn grouping(k: *const Kernels, s: cuda.Stream, d: *const cuda.Driver, e: usize, sc: ExpertScratch, pick: u64, r: usize) !void {
    const rs = r * sc.slots;
    try d.check(d.api.cuMemsetD32Async(sc.cnt, 0, e, s.handle), "cuMemsetD32Async");
    var a: cuda.Args = .{};
    a.add(pick);
    a.add(@as(c_int, @intCast(rs)));
    a.add(@as(c_int, @intCast(e)));
    a.add(sc.cnt);
    try launch(k.count, s, .{ cdiv(rs, 256), 1, 1 }, 256, &a);
    a = .{};
    a.add(sc.cnt);
    a.add(@as(c_int, @intCast(e)));
    if (tilesOnce(k.group_tiles, sc)) {
        a.add(@as(c_int, pf_bm));
        a.add(sc.off);
        a.add(sc.fill);
        a.add(sc.act);
        a.add(sc.nact);
        a.add(sc.tiles);
        a.add(sc.ntiles);
        const f = k.group_tiles_fn orelse return error.NoGroupTilesKernel;
        try cuda.launch.launch(f, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024 }, .shared = @intCast(e * 4) }, s, &a);
    } else {
        a.add(sc.off);
        a.add(sc.fill);
        a.add(sc.act);
        a.add(sc.nact);
        try launch(k.scan, s, .{ 1, 1, 1 }, 32, &a);
    }
    a = .{};
    a.add(pick);
    a.add(@as(c_int, @intCast(rs)));
    a.add(@as(c_int, @intCast(sc.slots)));
    a.add(@as(c_int, @intCast(e)));
    a.add(sc.off);
    a.add(sc.fill);
    a.add(sc.ent);
    a.add(sc.erow);
    try launch(k.place, s, .{ cdiv(rs, 256), 1, 1 }, 256, &a);
}

/// The largest k tf_dsn_topk_tiles takes (its shared lists).
pub const topk_tiles_max = 512;

/// The indexer's pruned top-k of `rows` rows in one launch (tri_index.topkSelectPruned's result): keys [rows, n] int64
/// (row stride ks), their 64-key tiles' maxima tmax [rows, nt = cdiv(n, 64)], each row's visible count vis [rows] ->
/// out [rows, k] int64, the k largest keys' indices ascending, -1 at or past vis. k at most topk_tiles_max, more than
/// k tiles (the chain prunes only then).
pub fn topkTiles(k: *const Kernels, s: cuda.Stream, keys: u64, ks: usize, tmax: u64, n: usize, kk: usize, vis: u64, out: u64, rows: usize) !void {
    const f = k.topk_tiles_fn orelse return error.NoTopkTilesKernel;
    const nt = cdiv(n, 64);
    if (kk == 0 or kk > topk_tiles_max or nt <= kk or n > std.math.maxInt(c_int)) return error.UnsupportedShape;
    var a: cuda.Args = .{};
    a.add(keys);
    a.add(@as(i64, @intCast(ks)));
    a.add(tmax);
    a.add(@as(c_int, @intCast(nt)));
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    a.add(vis);
    a.add(out);
    try launch(f, s, .{ rows, 1, 1 }, 1024, &a);
}

/// The active experts' down weights (down's order) into L2 on stream `s`, up to `budget` bytes, paced at about
/// 230 GB/s: the 2D exchange between gate / up and down leaves DRAM idle.
pub fn prefetchDown(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, budget: usize) !void {
    const nx = ex.native orelse return;
    const i: usize = ex.down_k;
    const n: usize = ex.down_n;
    var a: cuda.Args = .{};
    a.add(sc.act);
    a.add(sc.nact);
    a.add(nx.w2);
    a.add(@as(c_longlong, @intCast(n * i / 2)));
    a.add(@as(c_longlong, @intCast(n * i)));
    a.add(@as(c_int, @intCast(ex.count - 1)));
    a.add(@as(c_longlong, @intCast(budget)));
    a.add(@as(c_longlong, @intFromFloat(128.0 * @as(f64, @floatFromInt(ops_chunk.l2_chunk_bytes)) / 230.0))); // (the served 230 GB/s, a wave of 128 chunks)
    a.add(@as(c_longlong, @intCast(ops_chunk.l2_chunk_bytes)));
    try cuda.launch.launch(k.dec_l2pf, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
}

/// The decode tensor-core kernels' warps a tile (and matrix) of the experts' gate / up and down (fixed: the experts'
/// shapes are the layer's).
pub const dec_gu_ks = 4;
pub const dec_dn_ks = 2;

/// The expert pack's codes and scales as gate/up's epilogue writes them (TF_DS_PACK_EPILOGUE): extern, the twin kernel's `sf::QOut`
/// (two pointers, then the pair's main width, the rest width and the pack's slot rows).
pub const PackQ = extern struct { codes: u64, scales: u64, ig: i32, ir: i32, P: i32 };

/// TF_DS_PACK_EPILOGUE's start: the twin kernel must be in this build's module, else the start ends by name (never a silent fallback to
/// gate/up and packQuant).
pub fn usePackEpilogue(k: *Kernels) !void {
    const sk = k.sfk orelse return error.NoPackEpilogueKernel;
    k.gu_q = Kernels.pfFunction(sk.module, "tf_dsn_sf_gu_exact_q_kernel", pf_gu_r128_smem) catch return error.NoPackEpilogueKernel;
}

/// Gate / up of a native layer's experts on r rows: x [r, D] bf16 (row stride ldx) routed by pick / wts [r, slots]
/// (the shared expert the last id, every row's last slot), each slot's h = bf16(silu(min(g, limit)) * clamp(u) * w)
/// into the pack h as the 2D exchange carries it: every slot's main columns [r * slots, ldh] (ldh: the main blocks'
/// width), then with a rest block every slot's rest columns [r * slots, rest width].
/// split: a decode launch, the GEMV kernels (a warp a column, each row its own arithmetic: its bits at any row count);
/// prompt chunks: false, the MMA tiles.
pub fn gateUp(k: *const Kernels, s: cuda.Stream, d: *const cuda.Driver, ex: weights.Experts, sc: ExpertScratch, x: u64, ldx: usize, pick: u64, wts: u64, r: usize, limit: f32, h: u64, ldh: usize, split: bool) !void {
    return gateUpOf(k, s, d, ex, sc, x, ldx, pick, wts, r, limit, h, ldh, split, false, null);
}

/// gateUp of a prompt chunk's r rows whose pack codes the launch writes itself into `pq` (TF_DS_PACK_EPILOGUE): the twin kernel in place of
/// the block-scaled gate/up, the bf16 pack `h` still written. Only the exact block-scaled prompt path has the twin: any other shape is an
/// error, never the baseline launch.
pub fn gateUpPack(k: *const Kernels, s: cuda.Stream, d: *const cuda.Driver, ex: weights.Experts, sc: ExpertScratch, x: u64, ldx: usize, pick: u64, wts: u64, r: usize, limit: f32, h: u64, ldh: usize, coded: bool, pq: PackQ) !void {
    return gateUpOf(k, s, d, ex, sc, x, ldx, pick, wts, r, limit, h, ldh, false, coded, pq);
}

/// gateUp of a prompt chunk's r rows whose codes and scales are in sc.q / sc.s already (act_quant of x's rows [r, D],
/// the bytes group() would write): the same grouping and launches, without the quant.
pub fn gateUpCodes(k: *const Kernels, s: cuda.Stream, d: *const cuda.Driver, ex: weights.Experts, sc: ExpertScratch, pick: u64, wts: u64, r: usize, limit: f32, h: u64, ldh: usize) !void {
    return gateUpOf(k, s, d, ex, sc, 0, ex.dims, pick, wts, r, limit, h, ldh, false, true, null);
}

fn gateUpOf(k: *const Kernels, s: cuda.Stream, d: *const cuda.Driver, ex: weights.Experts, sc: ExpertScratch, x: u64, ldx: usize, pick: u64, wts: u64, r: usize, limit: f32, h: u64, ldh: usize, split: bool, coded: bool, pq: ?PackQ) !void {
    const nx = ex.native orelse return error.NotNativeExperts;
    if (r > sc.rows) return error.ScratchTooSmall;
    if (pq != null and (split or k.gu_q == null)) return error.PackEpilogueNotApplicable;
    try group(k, s, d, ex, sc, x, ldx, pick, r, split and k.dec_on, coded);
    const dd: usize = ex.dims;
    const a: GuArgs = .{
        .q = sc.q,
        .s = sc.s,
        .ldq = @intCast(dd),
        .lds = @intCast(dd / 32),
        .ent = sc.ent,
        .erow = sc.erow,
        .off = sc.off,
        .act = sc.act,
        .nact = sc.nact,
        .w1 = nx.w1,
        .w3 = nx.w3,
        .s1 = nx.s1,
        .s3 = nx.s3,
        .ldw = @intCast(dd / 2),
        .ldws = @intCast(dd / 32),
        .fmt = 1,
        .D = @intCast(dd),
        .shared = @intCast(ex.count - 1),
        .sldw = @intCast(dd),
        .sldws = @intCast(dd / 32),
        .owner = nx.owner,
        .n_tiles = @intCast(nx.tiles),
        .rest_tiles = @intCast(nx.rest_tiles),
        .wts = wts,
        .limit = limit,
        .h = h,
        .ldh = @intCast(ldh),
        .P = @intCast(r * sc.slots),
    };
    const p = r * sc.slots;
    const ny = @min(@as(usize, ex.count), p);
    if (split) {
        var ga: cuda.Args = .{};
        if (k.dec_on and dd % 128 == 0) {
            // the decode tensor-core kernel: a tile's gate and up each on decGuSplit warps
            var b = a;
            b.splits = @intCast(dec_gu_ks);
            b.early_exit = @intFromBool(k.expert_exit);
            ga.add(b);
            const tpc = dec_warps / (2 * dec_gu_ks);
            const rg = decRg(r);
            return cuda.launch.launch(k.dec[3][rg], .{ .grid = .{ .x = @intCast(cdiv(nx.tiles * 2, tpc)), .y = @intCast(ny), .z = 1 }, .block = .{ .x = 32 * dec_warps }, .shared = decShared(dec_gu_cfg, @as(usize, 1) << @intCast(rg)), .pdl = k.dec_pdl }, s, &ga);
        }
        ga.add(a);
        if (k.sab) return launch(k.gu_sab, s, .{ cdiv(nx.tiles * 32, 16), ny, 1 }, 256, &ga);
        return launch(k.gu_gemv, s, .{ cdiv(nx.tiles * 32, 8), ny, 1 }, 256, &ga);
    }
    var args: cuda.Args = .{};
    args.add(a);
    const main = (nx.tiles - nx.rest_tiles) * 32;
    if (pfExperts(k, sc, dd, nx.tiles * 32, main) and dd % 128 == 0) {
        if (!tilesOnce(k.group_tiles, sc)) try tileList(k, s, sc);
        args.add(sc.tiles);
        args.add(sc.ntiles);
        if (pq) |q| {
            // the twin runs where the exact block-scaled gate/up does and nowhere else
            _ = sfExperts(k, nx.s1, false) orelse return error.PackEpilogueNotApplicable;
            if (k.sf != .exact or k.gemm_r2 or k.gemm_w16) return error.PackEpilogueNotApplicable;
            args.add(q);
            return launchSharedT(k.gu_q.?, s, .{ nx.tiles * 32 / 128, p / pf_bm + ex.count, 1 }, pf_gu_r128_smem, 256, &args);
        }
        if (sfExperts(k, nx.s1, false)) |fsf| return launchSharedT(fsf.f, s, .{ nx.tiles * 32 / 128, p / pf_bm + ex.count, 1 }, fsf.smem, fsf.threads, &args);
        const el = pfExpertsLaunch(k.run128);
        const f = if (el.r128) k.pf_gu_r128 orelse return error.NoRun128Kernel else k.pf_gu;
        return launchShared(f, s, .{ nx.tiles * 32 / 128, p / pf_bm + ex.count, 1 }, el.gu_smem, &args);
    }
    const t = @min(tiles(r), 2);
    try launch(k.gu[t], s, .{ cdiv(nx.tiles, 4), ny, 1 }, 128, &args);
}

/// Whether a prompt chunk's grouped experts take the prefill tiles: a tile list, K (`kdim`) a multiple of 128, the
/// output columns (`n`) and the rest block's start (`main`) multiples of 128, aligned rows.
fn pfExperts(k: *const Kernels, sc: ExpertScratch, kdim: usize, n: usize, main: usize) bool {
    if (k.pf_off or sc.tiles == 0) return false;
    return kdim % 128 == 0 and n % 128 == 0 and main % 128 == 0;
}

/// The prefill tiles of the current grouping (sc.off / act / nact) into sc.tiles, sc.ntiles.
pub fn tileList(k: *const Kernels, s: cuda.Stream, sc: ExpertScratch) !void {
    var a: cuda.Args = .{};
    a.add(sc.off);
    a.add(sc.act);
    a.add(sc.nact);
    a.add(sc.tiles);
    a.add(sc.ntiles);
    try launch(k.pf_tiles, s, .{ 1, 1, 1 }, 32, &a);
}

/// Down of a native layer's experts after gate / up (the same grouping): h [r * slots, ldh] bf16, the node's whole
/// intermediate part (down_k columns), quantized, through each expert's down part, the slots' outputs summed in expert
/// id order into out [r, down_n] fp32 (row stride ldo; accumulate: added to it).
/// split: a decode launch, the GEMV kernel (as gateUp).
pub fn down(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, h: u64, ldh: usize, pick: u64, r: usize, out: u64, ldo: usize, accumulate: bool, split: bool) !void {
    return downOf(k, s, ex, sc, h, ldh, pick, r, out, ldo, accumulate, split, false, null);
}

/// down of a decode window's r rows whose intermediate's codes and scales are in sc.hq / sc.hs already (assembleQuant's:
/// the bytes quantDec would write): down's decode launches without the quant.
pub fn downDecCodes(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, pick: u64, r: usize, out: u64, ldo: usize, accumulate: bool) !void {
    return downOf(k, s, ex, sc, 0, ex.down_k, pick, r, out, ldo, accumulate, true, true, null);
}

/// The width of a 2D node's assembled intermediate (pair 0's main ig, pair 1's, the rest ir) that assembleQuant takes:
/// whole 32-value groups in each part, else none.
pub fn asmQuantWidth(ig: usize, ir: usize) ?usize {
    if (ig % 32 != 0 or ir % 32 != 0 or ig == 0) return null;
    return 2 * ig + ir;
}

/// TF_DS_DECODE_FUSE: tf_ds_assemble_rest's assembly of a 2D node's expert intermediate into out [p, 2 ig + ir] bf16
/// (pair 0's main, pair 1's, each slot's rest from owner[pick]) and quantDec's codes and scales of it into sc.hq / sc.hs
/// (row strides the width and width / 32), in one ordinary launch: the bytes of the two launches.
pub fn assembleQuant(k: *const Kernels, s: cuda.Stream, part0: u64, part1: u64, pick: u64, owner: u64, out: u64, p: usize, ig: usize, ir: usize, sc: ExpertScratch) !void {
    const f = k.dec_asm_quant orelse return error.NoDecodeFuseKernel;
    if (p == 0) return;
    const w = asmQuantWidth(ig, ir) orelse return error.UnsupportedShape;
    if (p > sc.rows * sc.slots) return error.ScratchTooSmall;
    var a: cuda.Args = .{};
    for ([_]u64{ part0, part1, pick, owner, out }) |v| a.add(v);
    for ([_]usize{ p, ig, ir }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(sc.hq);
    a.add(@as(i64, @intCast(w)));
    a.add(sc.hs);
    a.add(@as(i64, @intCast(w / 32)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(cdiv(p * (w / 32) * 32, 256)), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
}

/// down of a prompt chunk's r rows whose intermediate's codes and scales are in sc.hq / sc.hs already (act_quant of
/// the assembled intermediate [r * slots, down_k], the bytes down's quant would write): the same launches without the
/// quant.
pub fn downCodes(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, pick: u64, r: usize, out: u64, ldo: usize, accumulate: bool) !void {
    return downOf(k, s, ex, sc, 0, ex.down_k, pick, r, out, ldo, accumulate, false, true, null);
}

/// down of a prompt chunk with its routed sum in two stages (TF_DS_ROCI_COMBINE_FIRST): the same launches as `down`, the
/// last one (tf_dsn_combine) over the row blocks of `cf` in its order.
pub fn downFirst(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, h: u64, ldh: usize, pick: u64, r: usize, out: u64, ldo: usize, accumulate: bool, cf: *const CombineFirst) !void {
    return downOf(k, s, ex, sc, h, ldh, pick, r, out, ldo, accumulate, false, false, cf);
}

/// downCodes with its routed sum in two stages: downFirst for an intermediate whose codes are in sc.hq / sc.hs already.
pub fn downCodesFirst(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, pick: u64, r: usize, out: u64, ldo: usize, accumulate: bool, cf: *const CombineFirst) !void {
    return downOf(k, s, ex, sc, 0, ex.down_k, pick, r, out, ldo, accumulate, false, true, cf);
}

fn downOf(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, h: u64, ldh: usize, pick: u64, r: usize, out: u64, ldo: usize, accumulate: bool, split: bool, coded: bool, cf: ?*const CombineFirst) !void {
    const nx = ex.native orelse return error.NotNativeExperts;
    if (cf != null and split) return error.CombineFirstOnDecode;
    const i: usize = ex.down_k;
    const n: usize = ex.down_n;
    const p = r * sc.slots;
    if (!coded) {
        if (split) try quantDec(k, s, h, .bf16, ldh, p, i, sc.hq, i, sc.hs, i / 32) else try quant(k, s, h, .bf16, ldh, p, i, sc.hq, i, sc.hs, i / 32);
    }
    const a: DnArgs = .{
        .q = sc.hq,
        .s = sc.hs,
        .ldq = @intCast(i),
        .lds = @intCast(i / 32),
        .ent = sc.ent,
        .off = sc.off,
        .act = sc.act,
        .nact = sc.nact,
        .w2 = nx.w2,
        .s2 = nx.s2,
        .ldw = @intCast(i / 2),
        .ldws = @intCast(i / 32),
        .fmt = 1,
        .shared = @intCast(ex.count - 1),
        .sldw = @intCast(i),
        .sldws = @intCast(i / 32),
        .I = @intCast(i),
        .N = @intCast(n),
        .round_bf16 = 0,
        .out = sc.out,
        .ldo = @intCast(n),
    };
    var args: cuda.Args = .{};
    args.add(a);
    const ny = @min(@as(usize, ex.count), p);
    if (split and k.dec_on and i % 128 == 0 and n % 16 == 0) {
        var b = a;
        b.splits = @intCast(dec_dn_ks);
        b.early_exit = @intFromBool(k.expert_exit);
        var da: cuda.Args = .{};
        da.add(b);
        const tpc = dec_warps / dec_dn_ks;
        const rg = decRg(r);
        const gx = cdiv(n / 16, tpc);
        const cfg: cuda.launch.Config = .{ .grid = .{ .x = @intCast(gx), .y = @intCast(ny), .z = 1 }, .block = .{ .x = 32 * dec_warps }, .shared = decShared(dec_experts_cfg, @as(usize, 1) << @intCast(rg)), .pdl = k.dec_pdl };
        if (decSumIn(k.decode_fuse, gx)) {
            // the routed sum in this launch (tf_dsn_dec_dn_cb: tf_dsn_combine's arithmetic): no combine launch after it
            const f = k.dec_dn_cb orelse return error.NoDecodeFuseKernel;
            da.add(CombArgs{ .pick = pick, .R = @intCast(r), .S = @intCast(sc.slots), .E = @intCast(ex.count), .accumulate = @intFromBool(accumulate), .y = out, .ldy = @intCast(ldo), .cnt = k.comb_cnt.?.ptr });
            return cuda.launch.launch(f[rg], cfg, s, &da);
        }
        try cuda.launch.launch(k.dec[4][rg], cfg, s, &da);
    } else if (split) {
        if (k.sab) try launch(k.dn_sab, s, .{ cdiv(n, 16), ny, 1 }, 256, &args) else try launch(k.dn_gemv, s, .{ cdiv(n, 8), ny, 1 }, 256, &args);
    } else if (pfExperts(k, sc, i, n, 0)) {
        if (!tilesOnce(k.group_tiles, sc)) try tileList(k, s, sc);
        if (sc.three) |th| {
            // the pack's parts where they lie: the first part is the base (its main block), the second's main block and both rest blocks ride along
            const f3 = (k.sfk orelse return error.NoMmaSfKernels).three(k.sf) orelse return error.NoMmaSfKernels;
            const rest = ex.rest orelse return error.NoRest;
            const b = packBlocks(th.rows, th.ig, th.ir);
            var a3 = a;
            a3.q = th.p0 + b.main_q;
            a3.s = th.p0 + b.main_s;
            a3.ldq = @intCast(th.ig);
            a3.lds = @intCast(th.ig / 32);
            var args3: cuda.Args = .{};
            args3.add(a3);
            args3.add(sc.tiles);
            args3.add(sc.ntiles);
            args3.add(Part3{ .q1 = th.p1 + b.main_q, .s1 = th.p1 + b.main_s, .qr0 = th.p0 + b.rest_q, .sr0 = th.p0 + b.rest_s, .qr1 = th.p1 + b.rest_q, .sr1 = th.p1 + b.rest_s, .owner = rest.owner, .ldq_rest = @intCast(th.ir), .lds_rest = @intCast(th.ir / 32), .kb_main = @intCast(th.ig / 32) });
            try launchShared(f3, s, .{ n / 128, p / pf_bm + ex.count, 1 }, pf_dn_r128_smem, &args3);
        } else {
            args.add(sc.tiles);
            args.add(sc.ntiles);
            if (sfExperts(k, nx.s1, true)) |fsf| try launchSharedT(fsf.f, s, .{ n / 128, p / pf_bm + ex.count, 1 }, fsf.smem, fsf.threads, &args) else {
                const el = pfExpertsLaunch(k.run128);
                const f = if (el.r128) k.pf_dn_r128 orelse return error.NoRun128Kernel else k.pf_dn;
                try launchShared(f, s, .{ n / 128, p / pf_bm + ex.count, 1 }, el.dn_smem, &args);
            }
        }
    } else {
        const t = @min(tiles(r), 2);
        try launch(k.dn[t], s, .{ cdiv(n / 32, 4), ny, 1 }, 128, &args);
    }
    if (cf) |f| return combineFirst(k, s, sc, pick, r, ex.count, n, out, ldo, accumulate, f);
    try combine(k, s, sc, pick, r, ex.count, n, out, ldo, accumulate);
}

/// down's last launch (tf_dsn_combine): the slots' outputs of r rows (sc.out [r * slots, n] fp32, by entry) summed in
/// expert id order over `experts` experts into out [r, n] fp32 (row stride ldo; accumulate: added to it).
pub fn combine(k: *const Kernels, s: cuda.Stream, sc: ExpertScratch, pick: u64, r: usize, experts: usize, n: usize, out: u64, ldo: usize, accumulate: bool) !void {
    var ca: cuda.Args = .{};
    ca.add(sc.out);
    ca.add(@as(i64, @intCast(n)));
    ca.add(pick);
    ca.add(@as(c_int, @intCast(r)));
    ca.add(@as(c_int, @intCast(sc.slots)));
    ca.add(@as(c_int, @intCast(experts)));
    ca.add(@as(c_int, @intCast(n)));
    ca.add(out);
    ca.add(@as(i64, @intCast(ldo)));
    ca.add(@as(c_int, @intFromBool(accumulate)));
    try launch(k.combine, s, .{ cdiv(r * n, 256), 1, 1 }, 256, &ca);
}

/// The routed sum of a prompt chunk in two stages (TF_DS_ROCI_COMBINE_FIRST). The chunk's rows are four blocks of `rows` rows;
/// `first` are the starts (in rows) of the two blocks the owner sums' first part sends, `last` the starts of the other two.
/// The sum of the `first` blocks is launched, then `between` runs on the host (it queues the send behind them on the stream),
/// then the sum of the `last` blocks is launched.
pub const CombineFirst = struct {
    rows: usize,
    first: [2]usize,
    last: [2]usize,
    ctx: *anyopaque,
    between: *const fn (ctx: *anyopaque) anyerror!void,
    /// TF_DS_ROCI_SENT_FIRST: one launch of tf_dsn_combine_door in this order instead of the four, the doorbell rung from inside
    /// it: called right before the launch for the Door (the ticket reserved there), `between` not called.
    door: ?*const fn (ctx: *anyopaque) anyerror!Door = null,

    /// Whether the four blocks are the four row quarters of `r` rows, each once.
    pub fn covers(cf: CombineFirst, r: usize) bool {
        if (cf.rows == 0 or cf.rows * 4 != r) return false;
        var seen: [4]bool = .{ false, false, false, false };
        for ([_][2]usize{ cf.first, cf.last }) |pair| for (pair) |row0| {
            if (row0 % cf.rows != 0 or row0 / cf.rows >= 4 or seen[row0 / cf.rows]) return false;
            seen[row0 / cf.rows] = true;
        };
        return true;
    }
};

/// tf_dsn_combine over the rows [row0, row0 + rows) of a chunk: the kernel indexes its rows relative to its arguments, so a row
/// start is an offset of the entries' outputs (sc.out, n columns a row and sc.slots rows an entry), of the picks and of the
/// output. Each output row is the sum the one launch over all the rows makes of it.
pub fn combineRows(k: *const Kernels, s: cuda.Stream, sc: ExpertScratch, pick: u64, row0: usize, rows: usize, experts: usize, n: usize, out: u64, ldo: usize, accumulate: bool) !void {
    var part = sc;
    part.out = sc.out + row0 * sc.slots * n * 4;
    try combine(k, s, part, pick + row0 * sc.slots * 4, rows, experts, n, out + row0 * ldo * 4, ldo, accumulate);
}

/// tf_dsn_combine over all `r` rows in the quarter order of `qo`, the doorbell rung by its first qo.span CTAs' last arrival.
pub fn combineDoor(k: *const Kernels, s: cuda.Stream, sc: ExpertScratch, pick: u64, r: usize, experts: usize, n: usize, out: u64, ldo: usize, accumulate: bool, qo: QuarterOrder, door: Door) !void {
    const f = k.combine_door orelse return error.NoMmaSfKernels;
    var ca: cuda.Args = .{};
    ca.add(sc.out);
    ca.add(@as(i64, @intCast(n)));
    ca.add(pick);
    ca.add(@as(c_int, @intCast(r)));
    ca.add(@as(c_int, @intCast(sc.slots)));
    ca.add(@as(c_int, @intCast(experts)));
    ca.add(@as(c_int, @intCast(n)));
    ca.add(out);
    ca.add(@as(i64, @intCast(ldo)));
    ca.add(@as(c_int, @intFromBool(accumulate)));
    ca.add(qo);
    ca.add(door);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(r * n / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &ca);
}

pub fn combineFirst(k: *const Kernels, s: cuda.Stream, sc: ExpertScratch, pick: u64, r: usize, experts: usize, n: usize, out: u64, ldo: usize, accumulate: bool, cf: *const CombineFirst) !void {
    if (!cf.covers(r)) return error.CombineFirstRows;
    if (cf.door) |get_door| {
        const qo = quarterOrder(cf.first, cf.last, cf.rows, n) orelse return error.CombineFirstRows;
        return combineDoor(k, s, sc, pick, r, experts, n, out, ldo, accumulate, qo, try get_door(cf.ctx));
    }
    for (cf.first) |row0| try combineRows(k, s, sc, pick, row0, cf.rows, experts, n, out, ldo, accumulate);
    try cf.between(cf.ctx);
    for (cf.last) |row0| try combineRows(k, s, sc, pick, row0, cf.rows, experts, n, out, ldo, accumulate);
}

test "splits depend on the shape alone and fill the GPU" {
    try std.testing.expectEqual(@as(usize, 16), splits(5120, 1280)); // wq_a: 10 CTAs, 10 K blocks a split
    try std.testing.expectEqual(@as(usize, 16), splits(5120, 512)); // wkv
    try std.testing.expectEqual(@as(usize, 2), splits(1280, 8192)); // a 2D node's wq_b
    try std.testing.expectEqual(@as(usize, 1), splits(3072, 12800)); // a 2D node's Engram wkv
    try std.testing.expectEqual(@as(usize, 4), splits(1280, 4096)); // the indexer's wq_b
    try std.testing.expectEqual(@as(usize, 16), splits(5120, 640)); // a 2D node's expert gate / up pack
    try std.testing.expectEqual(@as(usize, 16), bfSplits(5120, 384)); // the router
    try std.testing.expectEqual(@as(usize, 16), bfSplits(4096, 1024)); // a wo_a group
}

/// The split FP8 GEMMs a prompt chunk runs on a 2D node (K, N) with their folds (K blocks a split), and Engram's
/// unsplit wkv.
const split_shapes = [_]struct { k: usize, n: usize, fold: usize }{
    .{ .k = 5120, .n = 1280, .fold = 10 }, // wq_a
    .{ .k = 5120, .n = 512, .fold = 10 }, // wkv
    .{ .k = 1280, .n = 8192, .fold = 20 }, // wq_b
    .{ .k = 4096, .n = 2560, .fold = 16 }, // wo_b
    .{ .k = 1280, .n = 4096, .fold = 10 }, // the indexer's wq_b
};
const row_counts = [_]usize{ 64, 65, 127, 128, 129, 256, 512, 513, 1000, 1024, 1025, 2048, 4096 };

test "TF_DS_PF_SPLIT's text: unset or 0 off, 1 the table, A, B, C6 or C7 that configuration, any other text no setting" {
    try std.testing.expectEqual(@as(?PfSplit, .off), PfSplit.parse(null));
    try std.testing.expectEqual(@as(?PfSplit, .off), PfSplit.parse("0"));
    try std.testing.expectEqual(@as(?PfSplit, .table), PfSplit.parse("1"));
    try std.testing.expectEqual(@as(?PfSplit, .a), PfSplit.parse("A"));
    try std.testing.expectEqual(@as(?PfSplit, .b), PfSplit.parse("B"));
    try std.testing.expectEqual(@as(?PfSplit, .c6), PfSplit.parse("C6"));
    try std.testing.expectEqual(@as(?PfSplit, .c7), PfSplit.parse("C7"));
    for ([_][]const u8{ "", "a", "c7", "2", "A ", " 1", "C8", "on" }) |t| try std.testing.expectEqual(@as(?PfSplit, null), PfSplit.parse(t));
    // the words a Settings frame or the tier's text carries, and none past them
    for ([_][]const u8{ "0", "1", "A", "B", "C6", "C7" }, 0..) |t, i| try std.testing.expectEqualStrings(t, PfSplit.name(@intCast(i)).?);
    try std.testing.expectEqual(@as(?[]const u8, null), PfSplit.name(6));
}

test "a split configuration divides the fold of every production shape at every row count, B only wq_b's and wo_b's" {
    for (split_shapes) |sh| for (row_counts) |m| {
        try std.testing.expectEqual(sh.fold, cdiv(sh.k / 32, splits(sh.k, sh.n)));
        try std.testing.expectEqual(@as(?SplitCfg, .a), pfSplitCfg(.a, m, sh.k, sh.n));
        try std.testing.expectEqual(@as(?SplitCfg, .c6), pfSplitCfg(.c6, m, sh.k, sh.n));
        try std.testing.expectEqual(@as(?SplitCfg, .c7), pfSplitCfg(.c7, m, sh.k, sh.n));
        // B stages 4 K blocks: folds of 10 (wq_a, wkv, the indexer's wq_b) stay on the fold kernel
        try std.testing.expectEqual(if (sh.fold % 4 == 0) @as(?SplitCfg, .b) else null, pfSplitCfg(.b, m, sh.k, sh.n));
        // the table: the baseline kernel up to 1,024 rows, A past them
        try std.testing.expectEqual(if (m > 1024) @as(?SplitCfg, .a) else null, pfSplitCfg(.table, m, sh.k, sh.n));
        try std.testing.expectEqual(@as(?SplitCfg, null), pfSplitCfg(.off, m, sh.k, sh.n));
    };
    // no K split (Engram's wkv), a split shape outside the table, a fold of an odd count of K blocks
    for ([_]PfSplit{ .table, .a, .b, .c6, .c7 }) |sw| try std.testing.expectEqual(@as(?SplitCfg, null), pfSplitCfg(sw, 2048, 3072, 12800));
    try std.testing.expectEqual(@as(?SplitCfg, null), pfSplitCfg(.table, 2048, 5120, 640));
    try std.testing.expectEqual(@as(?SplitCfg, .a), pfSplitCfg(.a, 2048, 5120, 640));
    try std.testing.expectEqual(@as(usize, 4), splits(1152, 4096)); // a fold of 9 K blocks
    for ([_]PfSplit{ .a, .b, .c6, .c7 }) |sw| try std.testing.expectEqual(@as(?SplitCfg, null), pfSplitCfg(sw, 2048, 1152, 4096));
    // each configuration's tile rows, stage width and shared memory as native.cu's static_assert holds them
    try std.testing.expectEqualSlices(usize, &.{ 128, 64, 64, 64 }, &split_bm);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4, 2, 2 }, &split_kbs);
    try std.testing.expectEqualSlices(u32, &.{ 87040, 76800, 76800, 89600 }, &split_smem);
}

test "TF_DS_PF_SPLIT unset runs the baseline kernels: the fold kernel with a K split, tf_dsn_pf_gemm without" {
    for (split_shapes) |sh| for (row_counts) |m| {
        try std.testing.expectEqual(PfGemm.fold, pfGemmOf(.off, m, sh.k, sh.n));
        try std.testing.expectEqual(PfGemm{ .split = .c7 }, pfGemmOf(.c7, m, sh.k, sh.n));
    };
    for ([_]PfSplit{ .off, .table, .a, .b, .c6, .c7 }) |sw| try std.testing.expectEqual(PfGemm.gemm, pfGemmOf(sw, 2048, 3072, 12800));
    try std.testing.expectEqual(PfGemm.fold, pfGemmOf(.b, 2048, 5120, 1280));
    try std.testing.expectEqual(PfGemm{ .split = .b }, pfGemmOf(.b, 2048, 4096, 2560));
}

test "TF_DS_PF_GEMM5 takes the GEMMs without a K split to the 5-stage kernel and nothing else; unset, tf_dsn_pf_gemm" {
    for (row_counts) |m| {
        // Engram's wkv (no K split)
        try std.testing.expectEqual(PfGemm.gemm, pfGemmKernel(.off, false, m, 3072, 12800));
        try std.testing.expectEqual(PfGemm.gemm5, pfGemmKernel(.off, true, m, 3072, 12800));
        try std.testing.expectEqual(PfGemm.gemm5, pfGemmKernel(.a, true, m, 3072, 12800));
        // the split shapes keep the fold kernel or their split configuration
        for (split_shapes) |sh| {
            try std.testing.expectEqual(PfGemm.fold, pfGemmKernel(.off, true, m, sh.k, sh.n));
            try std.testing.expectEqual(PfGemm{ .split = .a }, pfGemmKernel(.a, true, m, sh.k, sh.n));
            try std.testing.expectEqual(pfGemmOf(.c7, m, sh.k, sh.n), pfGemmKernel(.c7, false, m, sh.k, sh.n));
        }
    }
    try std.testing.expectEqual(@as(u32, 87040), pf_gemm5_smem);
}

test "TF_DS_PF_SPLIT runs with the checkpoint's own weights and the prompt tile kernels, else off and why" {
    try std.testing.expectEqual(@as(?[]const u8, null), splitOff(true, false));
    try std.testing.expectEqualStrings("needs the checkpoint's own FP8 / MXFP4 weights", splitOff(false, false).?);
    try std.testing.expectEqualStrings("needs the prompt chunks' tile kernels (TF_DS_PF_TILES=0 turns them off)", splitOff(true, true).?);
}

test "TF_DS_WOA_W8 runs with the checkpoint's own weights, the prompt tile kernels and the decode kernels, else off and why" {
    try std.testing.expectEqual(@as(?[]const u8, null), woaW8Off(true, false, true));
    try std.testing.expectEqualStrings("needs the checkpoint's own FP8 / MXFP4 weights", woaW8Off(false, false, true) orelse "none");
    try std.testing.expectEqualStrings("needs the prompt chunks' tile kernels (TF_DS_PF_TILES=0 turns them off)", woaW8Off(true, true, true) orelse "none");
    try std.testing.expectEqualStrings("needs the decode tensor-core kernels (TF_DS_DECODE_DEC=0 reads wo_a's bf16 rows)", woaW8Off(true, false, false) orelse "none");
}

test "the FP8 wo_a kernel takes a 2D node's stacked groups: whole 128-column tiles a group, whole 32-column stages" {
    const w8: weights.Linear = .{ .words = 1, .w = 1, .s = 1, .k = 4096, .n = 2 * 1024, .fmt = .fp8 };
    try std.testing.expect(w8Shape(w8, 2));
    try std.testing.expect(w8Shape(w8, 4) and w8Shape(w8, 16) and !w8Shape(w8, 32)); // 64-column groups
    try std.testing.expect(!w8Shape(w8, 3) and !w8Shape(w8, 0));
    var odd = w8;
    odd.k = 4096 + 16;
    try std.testing.expect(!w8Shape(odd, 2));
    var bf = w8;
    bf.fmt = .bf16;
    try std.testing.expect(!w8Shape(bf, 2));
    var wide = w8;
    wide.k = weights.w8_max_k;
    try std.testing.expect(w8Shape(wide, 2));
    wide.k = weights.w8_max_k + 32;
    try std.testing.expect(!w8Shape(wide, 2));
}

test "a layer loaded without wo_a's bf16 rows sizes every buffer as with them (wo_a's Linears keep their shapes)" {
    var full: weights.Layer = .{ .idx = 2, .ratio = 2 };
    full.wq_a = .{ .words = 1, .w = 1, .s = 1, .k = 5120, .n = 1280, .fmt = .fp8 };
    full.wkv = .{ .words = 2, .w = 2, .s = 2, .k = 5120, .n = 512, .fmt = .fp8 };
    full.wq_b = .{ .words = 3, .w = 3, .s = 3, .k = 1280, .n = 8192, .fmt = .fp8 };
    full.wo_b = .{ .words = 4, .w = 4, .s = 4, .k = 4096, .n = 2560, .fmt = .fp8 };
    full.groups = 2;
    full.wo_a[0] = .{ .words = 5, .w = 5, .k = 4096, .n = 1024, .fmt = .bf16 };
    full.wo_a[1] = .{ .words = 6, .w = 6, .k = 4096, .n = 1024, .fmt = .bf16 };
    full.wo_a8 = .{ .words = 7, .w = 7, .s = 7, .k = 4096, .n = 2048, .fmt = .fp8 };
    full.experts = std.mem.zeroes(weights.Experts);
    var shape_only = full;
    for (shape_only.wo_a[0..2]) |*wo| wo.* = .{ .words = 0, .w = 0, .k = wo.k, .n = wo.n, .fmt = .bf16 };
    shape_only.wo_a_rows = false;
    for ([_]usize{ 1, 16, 64 }) |rows| try std.testing.expectEqual(decodeZFloats(full, 5120, 384, 32, rows, 7), decodeZFloats(shape_only, 5120, 384, 32, rows, 7));
    for (full.wo_a[0..2], shape_only.wo_a[0..2]) |a, b| {
        try std.testing.expectEqual(a.k, b.k);
        try std.testing.expectEqual(a.n, b.n);
        try std.testing.expectEqual(a.fmt, b.fmt);
    }
}

test "TF_DS_GROUP_TILES: the grouping builds the tile list on a prompt chunk's scratch only; unset, gate / up and down build it each" {
    var sc = std.mem.zeroes(ExpertScratch);
    try std.testing.expect(!tilesOnce(false, sc)); // the baseline: count, scan, place, then tileList in gate / up and in down
    try std.testing.expect(!tilesOnce(true, sc)); // no tile list (the drafter's scratch): nothing to build
    sc.tiles = 0x1000;
    try std.testing.expect(!tilesOnce(false, sc));
    try std.testing.expect(tilesOnce(true, sc));
}

test "gate / up and down build the tile list only when the grouping did not; the grouping launches one of the two kernels" {
    const src = @embedFile("mx.zig");
    const gu = std.mem.indexOf(u8, src, "fn gateUpOf(").?;
    const dn = std.mem.indexOf(u8, src, "fn downOf(").?;
    const gp = std.mem.indexOf(u8, src, "pub fn grouping(").?;
    for ([_]usize{ gu, dn }) |at| {
        const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "if (!tilesOnce(k.group_tiles, sc)) try tileList(k, s, sc);"));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "tileList("));
    }
    const body = src[gp .. gp + std.mem.indexOf(u8, src[gp..], "\n}\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "if (tilesOnce(k.group_tiles, sc)) {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "launch(k.scan,"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "k.group_tiles_fn orelse"));
}

/// tf_dsn_topk_tiles' radix select on the host, step for step (8-bit digits from the top, the bins counted from 255
/// down, the stop when every value of the chosen bin is needed): the smallest threshold exactly kk of the values reach
/// (null entries skipped).
fn selectKthModel(vals: []const ?u64, kk: usize) u64 {
    var prefix: u64 = 0;
    var need = kk;
    var shift: u6 = 56;
    while (true) : (shift -= 8) {
        var hist: [256]usize = @splat(0);
        const hi: u64 = if (shift == 56) 0 else ~@as(u64, 0) << (shift + 8);
        for (vals) |ov| {
            const u = ov orelse continue;
            if (u & hi == prefix) hist[@intCast((u >> shift) & 255)] += 1;
        }
        var above: usize = 0;
        var d: usize = 255;
        while (d > 0) : (d -= 1) {
            if (above + hist[d] >= need) break;
            above += hist[d];
        }
        prefix |= @as(u64, d) << shift;
        need -= above;
        if (hist[d] == need or shift == 0) return prefix;
    }
}

fn flipKey(key: i64) u64 {
    return @as(u64, @bitCast(key)) ^ (1 << 63);
}

/// tf_dsn_topk_tiles' three steps on the host for one row (its keys [n], tile maxima [nt]): the kk tiles of the
/// largest maxima, the kk largest of their keys, their positions ascending, -1 at or past vis.
fn topkTilesModel(gpa: std.mem.Allocator, keys: []const i64, tmax: []const i64, kk: usize, vis: i64, out: []i64) !void {
    const tv = try gpa.alloc(?u64, tmax.len);
    defer gpa.free(tv);
    for (tv, tmax) |*v, m| v.* = flipKey(m);
    const t1 = selectKthModel(tv, kk);
    var tl: std.ArrayList(usize) = .empty;
    defer tl.deinit(gpa);
    for (tmax, 0..) |m, i| if (flipKey(m) >= t1) try tl.append(gpa, i);
    if (tl.items.len != kk) return error.NotKk;
    const kv = try gpa.alloc(?u64, kk * 64);
    defer gpa.free(kv);
    for (kv, 0..) |*v, c| {
        const pos = tl.items[c / 64] * 64 + c % 64;
        v.* = if (pos < keys.len) flipKey(keys[pos]) else null;
    }
    const t2 = selectKthModel(kv, kk);
    var pos: std.ArrayList(i64) = .empty;
    defer pos.deinit(gpa);
    for (kv, 0..) |v, c| if (v) |u| if (u >= t2) try pos.append(gpa, @intCast(tl.items[c / 64] * 64 + c % 64));
    if (pos.items.len != kk) return error.NotKk;
    std.mem.sort(i64, pos.items, {}, std.sort.asc(i64));
    for (out, pos.items) |*o, p| o.* = if (p < vis) p else -1;
}

/// The pruned chain's result for one row, by its definition (kernels.py topk_select_pruned): the kk largest tile
/// maxima's tiles, those tiles' keys (past n none), their kk largest, each key's index (its low 32 bits inverted)
/// sorted ascending, -1 at or past vis.
fn prunedChainRef(gpa: std.mem.Allocator, keys: []const i64, tmax: []const i64, kk: usize, vis: i64, out: []i64) !void {
    const order = try gpa.alloc(usize, tmax.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, tmax, struct {
        fn gt(tm: []const i64, a: usize, b: usize) bool {
            return tm[a] > tm[b];
        }
    }.gt);
    var cand: std.ArrayList(i64) = .empty;
    defer cand.deinit(gpa);
    for (order[0..kk]) |t| for (t * 64..@min(t * 64 + 64, keys.len)) |p| try cand.append(gpa, keys[p]);
    std.mem.sort(i64, cand.items, {}, std.sort.desc(i64));
    const idx = try gpa.alloc(i64, kk);
    defer gpa.free(idx);
    for (idx, cand.items[0..kk]) |*x, key| x.* = 0xFFFFFFFF - (key & 0xFFFFFFFF);
    std.mem.sort(i64, idx, {}, std.sort.asc(i64));
    for (out, idx) |*o, x| o.* = if (x < vis) x else -1;
}

/// topk_indices' key of a score at position t (kernels.py: the score's bits in total order above, ~t below).
fn scoreKey(sc: f32, t: usize) i64 {
    const b: i32 = if (sc == 0) 0 else @bitCast(sc);
    const o: i32 = if (b < 0) b ^ 0x7FFFFFFF else b;
    const hi: u64 = @as(u64, @bitCast(@as(i64, o))) << 32;
    return @bitCast(hi | (0xFFFFFFFF - @as(u64, t)));
}

test "the pruned top-k in one kernel (its steps on the host) gives the pruned chain's indices: every vis, ties, a partial last tile, rows all -inf" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4b53);
    const rng = prng.random();
    const cases = [_]struct { n: usize, kk: usize }{ .{ .n = 4097, .kk = 64 }, .{ .n = 4160, .kk = 64 }, .{ .n = 32769, .kk = 512 }, .{ .n = 33000, .kk = 512 }, .{ .n = 65536, .kk = 512 } };
    for (cases) |cs| {
        const nt = cdiv(cs.n, 64);
        const keys = try gpa.alloc(i64, cs.n);
        defer gpa.free(keys);
        const tmax = try gpa.alloc(i64, nt);
        defer gpa.free(tmax);
        const want = try gpa.alloc(i64, cs.kk);
        defer gpa.free(want);
        const got = try gpa.alloc(i64, cs.kk);
        defer gpa.free(got);
        // every vis: all visible, past n, fewer than kk, inside the last (partial) tile, none (every score -inf)
        for ([_]usize{ cs.n, cs.n + 9, cs.kk / 2, cs.n - 3, 0, cs.n / 2 }, 0..) |vis, vi| {
            for (keys, 0..) |*key, t| {
                // scores from a few hundred values (ties), some negative, -inf at or past vis
                var sc: f32 = @as(f32, @floatFromInt(rng.uintLessThan(u32, 300))) * 0.125 - 8;
                if (vi == 5 and t % 7 == 0) sc = 0; // zeros (-0 as +0)
                key.* = scoreKey(if (t < vis) sc else -std.math.inf(f32), t);
            }
            for (tmax, 0..) |*m, b| m.* = std.mem.max(i64, keys[b * 64 .. @min(b * 64 + 64, cs.n)]);
            try prunedChainRef(gpa, keys, tmax, cs.kk, @intCast(vis), want);
            try topkTilesModel(gpa, keys, tmax, cs.kk, @intCast(vis), got);
            try std.testing.expectEqualSlices(i64, want, got);
        }
    }
}

test "TF_DS_BF16_ROWS: a prompt launch of the bf16 GEMM takes a CTA a 16-row tile; unset, and every decode launch, at most 64" {
    for ([_]usize{ 1, 16, 17, 256, 512, 1024, 1025, 2048, 4096 }) |m| {
        const today = @min((m + 15) / 16, 64);
        for ([_]bool{ false, true }) |prompt| try std.testing.expectEqual(today, bfGridRows(false, prompt, m));
        try std.testing.expectEqual(today, bfGridRows(true, false, m)); // a decode launch keeps its grid
        try std.testing.expectEqual((m + 15) / 16, bfGridRows(true, true, m));
    }
    // the same grid up to 1,024 rows (an owned quarter of 512: 32 CTAs either way)
    try std.testing.expectEqual(bfGridRows(false, true, 512), bfGridRows(true, true, 512));
    try std.testing.expectEqual(@as(usize, 128), bfGridRows(true, true, 2048));
}

test "the bf16 GEMM's plain launch takes its rows' grid from bfGridRows" {
    const src = @embedFile("mx.zig");
    const at = std.mem.indexOf(u8, src, "fn bf16Block(").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "try launch(k.bf16, s, .{ cdiv(n / 32, 4), bfGridRows(k.bf16_rows, z == null, m), if (sp > 1) sp else groups }, 128, &args);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "launch(k.bf16,"));
}

test "TF_DS_PF_RUN128: the experts' prefill tiles on their 128-byte-run twins with their shared memory; unset, the baseline kernels" {
    try std.testing.expectEqual(ExpertsLaunch{ .r128 = false, .gu_smem = 79872, .dn_smem = 53760 }, pfExpertsLaunch(false));
    try std.testing.expectEqual(ExpertsLaunch{ .r128 = true, .gu_smem = 97280, .dn_smem = 62464 }, pfExpertsLaunch(true));
    // native.cu's pf::kGuR128Smem / kDnR128Smem: 3 activation slots of 9,216 bytes and 2 weight pairs (gate and up:
    // 256 rows of 128 bytes and their 8 scale bytes; down: 128 rows), within GB10's 101,376 bytes a CTA with the
    // kernels' 3,328 static bytes
    try std.testing.expectEqual(@as(u32, 3 * 9216 + 2 * (256 * 128 + 256 * 8)), pf_gu_r128_smem);
    try std.testing.expectEqual(@as(u32, 3 * 9216 + 2 * (128 * 128 + 128 * 8)), pf_dn_r128_smem);
    try std.testing.expect(pf_gu_r128_smem + 3328 <= 101376);
}

test "gate / up and down launch their prefill tiles through pfExpertsLaunch" {
    const src = @embedFile("mx.zig");
    const fns = [_][]const u8{ "fn gateUpOf(", "fn downOf(" };
    const picks = [_][]const u8{
        "const f = if (el.r128) k.pf_gu_r128 orelse return error.NoRun128Kernel else k.pf_gu;\n",
        "const f = if (el.r128) k.pf_dn_r128 orelse return error.NoRun128Kernel else k.pf_dn;\n",
    };
    const launches = [_][]const u8{
        "return launchShared(f, s, .{ nx.tiles * 32 / 128, p / pf_bm + ex.count, 1 }, el.gu_smem, &args);",
        "try launchShared(f, s, .{ n / 128, p / pf_bm + ex.count, 1 }, el.dn_smem, &args);",
    };
    for (fns, picks, launches) |fname, pick, launch_| {
        const at = std.mem.indexOf(u8, src, fname).?;
        const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const el = pfExpertsLaunch(k.run128);"));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, pick));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, launch_));
    }
}

test "TF_DS_DECODE_FUSE: a decode down launch sums the routed slots itself with the switch and a count for each column group; unset, tf_dsn_combine after it" {
    try std.testing.expect(!decSumIn(false, 1));
    try std.testing.expect(!decSumIn(false, 40));
    try std.testing.expect(decSumIn(true, 1));
    try std.testing.expect(decSumIn(true, 40)); // a 2D node's down at 2,560 columns: 40 groups of 64
    try std.testing.expect(decSumIn(true, comb_counters));
    try std.testing.expect(!decSumIn(true, comb_counters + 1));
    try std.testing.expectEqual(@as(usize, 40), cdiv(2560 / 16, dec_warps / dec_dn_ks));
}

test "down's decode launch takes the summing kernel through decSumIn, the same configuration, and returns before the combine" {
    const src = @embedFile("mx.zig");
    const at = std.mem.indexOf(u8, src, "fn downOf(").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "if (decSumIn(k.decode_fuse, gx)) {"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "da.add(CombArgs{ .pick = pick, .R = @intCast(r), .S = @intCast(sc.slots), .E = @intCast(ex.count), .accumulate = @intFromBool(accumulate), .y = out, .ldy = @intCast(ldo), .cnt = k.comb_cnt.?.ptr });"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "return cuda.launch.launch(f[rg], cfg, s, &da);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "try cuda.launch.launch(k.dec[4][rg], cfg, s, &da);"));
    // the summing launch returns before the combine (mx.combine, down's last launch); the other paths reach it
    const fused = std.mem.indexOf(u8, body, "return cuda.launch.launch(f[rg], cfg, s, &da);").?;
    const comb = std.mem.indexOf(u8, body, "try combine(k, s, sc, pick, r, ex.count, n, out, ldo, accumulate);").?;
    try std.testing.expect(fused < comb);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "try combine("));
    // tf_dsn_combine's arguments are CombArgs' (the same pick, rows, slots, experts, output and accumulate)
    const cat = std.mem.indexOf(u8, src, "pub fn combine(").?;
    const cbody = src[cat .. cat + std.mem.indexOf(u8, src[cat..], "\n}\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, cbody, "try launch(k.combine, s, .{ cdiv(r * n, 256), 1, 1 }, 256, &ca);"));
    for ([_][]const u8{ "ca.add(pick);", "ca.add(@as(c_int, @intCast(r)));", "ca.add(@as(c_int, @intCast(sc.slots)));", "ca.add(@as(c_int, @intCast(experts)));", "ca.add(out);", "ca.add(@as(i64, @intCast(ldo)));", "ca.add(@as(c_int, @intFromBool(accumulate)));" }) |line| {
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, cbody, line));
    }
}

test "TF_DS_ROCI_COMBINE_FIRST: the two stages' blocks are the four row quarters, each once" {
    var dummy: u8 = 0;
    const F = struct {
        fn f(_: *anyopaque) anyerror!void {}
    };
    const base: CombineFirst = .{ .rows = 1024, .first = .{ 1024, 3072 }, .last = .{ 0, 2048 }, .ctx = &dummy, .between = F.f };
    try std.testing.expect(base.covers(4096));
    try std.testing.expect(!base.covers(4097)); // not four quarters of the rows
    try std.testing.expect(!base.covers(2048));
    var dup = base;
    dup.last = .{ 0, 1024 }; // a quarter twice, one never
    try std.testing.expect(!dup.covers(4096));
    var off = base;
    off.first = .{ 1000, 3072 }; // not a whole quarter's start
    try std.testing.expect(!off.covers(4096));
    var out = base;
    out.last = .{ 0, 4096 }; // past the rows
    try std.testing.expect(!out.covers(4096));
}

test "TF_DS_ROCI_COMBINE_FIRST: a stage's launch is the combine's launch at row offsets; the host step comes between the stages; decode launches refuse it" {
    const src = @embedFile("mx.zig");
    const at = std.mem.indexOf(u8, src, "pub fn combineRows(").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
    for ([_][]const u8{ "part.out = sc.out + row0 * sc.slots * n * 4;", "pick + row0 * sc.slots * 4, rows, experts, n, out + row0 * ldo * 4, ldo, accumulate);" }) |line| {
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, line));
    }
    const cf = src[std.mem.indexOf(u8, src, "fn combineFirst(").?..];
    const a = std.mem.indexOf(u8, cf, "for (cf.first) |row0|").?;
    const b = std.mem.indexOf(u8, cf, "try cf.between(cf.ctx);").?;
    const c = std.mem.indexOf(u8, cf, "for (cf.last) |row0|").?;
    try std.testing.expect(a < b and b < c);
    const dn = src[std.mem.indexOf(u8, src, "fn downOf(").?..];
    try std.testing.expect(std.mem.indexOf(u8, dn, "if (cf != null and split) return error.CombineFirstOnDecode;").? < 300);
}

test "TF_DS_DECODE_FUSE: the assembly with down's quant takes a 2D node's widths in whole 32-value groups" {
    try std.testing.expectEqual(@as(?usize, 1152), asmQuantWidth(512, 128)); // a 2D node's 512 main a pair and 128 rest
    try std.testing.expectEqual(@as(?usize, 1024), asmQuantWidth(512, 0));
    try std.testing.expectEqual(@as(?usize, null), asmQuantWidth(500, 128));
    try std.testing.expectEqual(@as(?usize, null), asmQuantWidth(512, 112));
    try std.testing.expectEqual(@as(?usize, null), asmQuantWidth(0, 128));
    // the codes' and scales' rows are down's quant rows: down_k bytes and down_k / 32 scales
    try std.testing.expectEqual(@as(usize, 36), asmQuantWidth(512, 128).? / 32);
}

test "a decode window's coded down is down's decode path without its quant" {
    const src = @embedFile("mx.zig");
    const at = std.mem.indexOf(u8, src, "pub fn downDecCodes(").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "return downOf(k, s, ex, sc, 0, ex.down_k, pick, r, out, ldo, accumulate, true, true, null);"));
    const dn = std.mem.indexOf(u8, src, "fn downOf(").?;
    const dbody = src[dn .. dn + std.mem.indexOf(u8, src[dn..], "\n}\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, dbody, "if (!coded) {"));
}

test "TF_DS_MMA_SF: a dense FP8 GEMM's block-scaled kernel by its split and rows; off, and exact below pf_min_rows, the baseline's" {
    for ([_]usize{ 1, 63, 64, 1024, 1025, 4096 }) |m| {
        try std.testing.expectEqual(@as(?SfGemm, null), sfGemmOf(.off, m, 16));
        try std.testing.expectEqual(@as(?SfGemm, null), sfGemmOf(.off, m, 1));
    }
    // no K split: K whole in the CTA at every row count
    for ([_]usize{ 1, 63, 64, 1025, 4096 }) |m| try std.testing.expectEqual(@as(?SfGemm, .gemm), sfGemmOf(.chained, m, 1));
    // a K split: one split a CTA below pf_min_rows (tf_dsn_reduce after), then 64-row tiles to 1,024 rows, 128-row past
    try std.testing.expectEqual(@as(?SfGemm, .zsplit), sfGemmOf(.chained, 1, 16));
    try std.testing.expectEqual(@as(?SfGemm, .zsplit), sfGemmOf(.chained, pf_min_rows - 1, 2));
    try std.testing.expectEqual(@as(?SfGemm, .split64), sfGemmOf(.chained, pf_min_rows, 2));
    try std.testing.expectEqual(@as(?SfGemm, .split64), sfGemmOf(.chained, 1024, 8));
    try std.testing.expectEqual(@as(?SfGemm, .split128), sfGemmOf(.chained, 1025, 8));
    try std.testing.expectEqual(@as(?SfGemm, .split128), sfGemmOf(.chained, 4096, 16));
    // exact: where the baseline tile kernels run (below pf_min_rows the baseline slab kernels give its bits)
    try std.testing.expectEqual(@as(?SfGemm, null), sfGemmOf(.exact, pf_min_rows - 1, 16));
    try std.testing.expectEqual(@as(?SfGemm, null), sfGemmOf(.exact, pf_min_rows - 1, 1));
    try std.testing.expectEqual(@as(?SfGemm, .gemm), sfGemmOf(.exact, pf_min_rows, 1));
    try std.testing.expectEqual(@as(?SfGemm, .split64), sfGemmOf(.exact, pf_min_rows, 16));
    try std.testing.expectEqual(@as(?SfGemm, .split128), sfGemmOf(.exact, 2048, 16));
}

test "TF_DS_MMA_SF takes every dense FP8 GEMM shape of a prompt chunk, and none whose split's K blocks are odd" {
    // wq_a, wkv, wq_b, wo_b, the indexer's wq_b (2D node), Engram's wkv (no K split)
    for ([_][2]usize{ .{ 5120, 1280 }, .{ 5120, 512 }, .{ 1280, 8192 }, .{ 4096, 2560 }, .{ 1280, 4096 }, .{ 3072, 12800 } }) |kn| {
        try std.testing.expect(sfShape(kn[0], kn[1]));
        // the split tiles' 2 K blocks a stage divide the split's K blocks (native_sf.cu tile_fp8's splits)
        const sp = splits(kn[0], kn[1]);
        try std.testing.expect(sp == 1 or cdiv(kn[0] / 32, sp) % 2 == 0);
    }
    try std.testing.expect(!sfShape(5120, 1280 - 64)); // N not a multiple of 128
    try std.testing.expect(!sfShape(5120 + 32, 1280)); // K not a multiple of 64
    try std.testing.expect(!sfShape(8192 + 64, 1280)); // K past 8,192 (the tiles' weight scales' slab)
    try std.testing.expect(sfShape(8192, 1280));
    // 2,304 x 128: 8 splits of 9 K blocks
    try std.testing.expectEqual(@as(usize, 8), splits(2304, 128));
    try std.testing.expect(!sfShape(2304, 128));
}

test "TF_DS_MMA_SF's dense kernels' shared memory: native_sf.cu's tiles; a small chunk's split partials fit a linear's scratch" {
    // sf::GemmCfg: 128 activation rows and 128 weight rows of 2 K blocks (64 bytes), 128 rows' 2 scales; 3 stages
    try std.testing.expectEqual(@as(u32, 3 * (128 * 64 + 128 * 64 + 128 * 2 * 4)), sf_gemm_smem);
    // sf::Split64Cfg: 64 activation rows; 6 stages. sf::Split128Cfg: GemmCfg's stage, 5 stages
    try std.testing.expectEqual(@as(u32, 6 * (64 * 64 + 128 * 64 + 64 * 2 * 4)), sf_split64_smem);
    try std.testing.expectEqual(@as(u32, 5 * (128 * 64 + 128 * 64 + 128 * 2 * 4)), sf_split128_smem);
    // TF_DS_GEMM_R2's expert tile: sf::DnR2Cfg's W2 ring (128 activation rows, 4 K blocks a stage, 3 stages, a
    // pair of 128-byte weight slots); its split GEMM takes sf_split128_smem's tile
    try std.testing.expectEqual(@as(u32, 3 * (128 * 128 + 128 * 4 * 4) + 2 * (128 * 128 + 128 * 8)), r2_dn_smem);
    // with the kernels' 2,560 static bytes, within GB10's 101,376 a CTA
    for ([_]u32{ sf_gemm_smem, sf_split64_smem, sf_split128_smem, pf_gu_r128_smem, pf_dn_r128_smem, r2_dn_smem }) |b| try std.testing.expect(b + 2560 <= 101376);
    // one split a CTA (below pf_min_rows): splits x rows x N floats, within zFloats of any scratch of at least those rows
    const l: weights.Linear = .{ .words = 0, .k = 5120, .n = 1280, .fmt = .fp8 };
    try std.testing.expect(splits(5120, 1280) * (pf_min_rows - 1) * 1280 <= zFloats(l, pf_min_rows - 1));
}

test "TF_DS_MMA_SF's expert kernels follow the switch: chained or exact, else none (the baseline's)" {
    var k: Kernels = undefined;
    k.sf = .off;
    k.sfk = null;
    try std.testing.expect(sfExperts(&k, 0, false) == null);
    const f = struct {
        fn of(i: usize) cuda.Function {
            return .{ .d = undefined, .handle = @ptrFromInt(0x1000 + i * 16) };
        }
    }.of;
    k.sfk = .{ .module = undefined, .gemm = .{ f(0), f(1) }, .split64 = .{ f(2), f(3) }, .split128 = .{ f(4), f(5) }, .zsplit = f(6), .gu = .{ f(7), f(8) }, .dn = .{ f(9), f(10) } };
    try std.testing.expect(sfExperts(&k, 0, false) == null);
    k.sf = .chained;
    try std.testing.expectEqual(f(7).handle, (sfExperts(&k, 0, false) orelse return error.TestUnexpectedResult).f.handle);
    try std.testing.expectEqual(f(9).handle, (sfExperts(&k, 0, true) orelse return error.TestUnexpectedResult).f.handle);
    k.sf = .exact;
    k.sf_nkept = 0;
    k.sf_nproven = 0;
    try std.testing.expect(sfExperts(&k, 0, false) == null); // not proven: the baseline
    try sfProve(&k, 0);
    try std.testing.expectEqual(f(8).handle, (sfExperts(&k, 0, false) orelse return error.TestUnexpectedResult).f.handle);
    try std.testing.expectEqual(f(10).handle, (sfExperts(&k, 0, true) orelse return error.TestUnexpectedResult).f.handle);
    // Kernels' defaults: off, no module
    const d: Kernels = .{ .module = undefined, .act_quant = undefined, .attn_epilogue = undefined, .gemm = undefined, .reduce = undefined, .count = undefined, .scan = undefined, .place = undefined, .gu = undefined, .dn = undefined, .gu_finish = undefined, .dn_finish = undefined, .gu_gemv = undefined, .dn_gemv = undefined, .gemv = undefined, .mx_sab = undefined, .gu_sab = undefined, .dn_sab = undefined, .sab = false, .dec = undefined, .dec_group = undefined, .dec_group2 = undefined, .nact_log = null, .group2 = false, .dec_quant = undefined, .dec_l2pf = undefined, .dec_on = true, .dec_pdl = true, .expert_exit = false, .counters = undefined, .combine = undefined, .bf16 = undefined, .bf_reduce = undefined, .to_f16 = undefined, .pf_gemm = undefined, .pf_gemm_fold = undefined, .pf_gu = undefined, .pf_dn = undefined, .pf_tiles = undefined, .pf_bf16 = undefined, .pf_off = false };
    try std.testing.expectEqual(MmaSf.off, d.sf);
    try std.testing.expect(d.sfk == null);
    try std.testing.expectEqual(@as(usize, 0), d.sf_nproven);
}

test "TF_DS_GEMM_W16 launches the exact kernels' 16-warp twins at 512 threads, the baseline at 256 elsewhere" {
    const f = struct {
        fn of(i: usize) cuda.Function {
            return .{ .d = undefined, .handle = @ptrFromInt(0x1000 + i * 16) };
        }
    }.of;
    var k: Kernels = undefined;
    k.sf = .exact;
    k.sf_nkept = 0;
    k.sf_nproven = 0;
    k.gemm_w16 = false;
    k.pf_bf16w8_w16 = null;
    k.sfk = .{ .module = undefined, .gemm = .{ f(0), f(1) }, .split64 = .{ f(2), f(3) }, .split128 = .{ f(4), f(5) }, .zsplit = f(6), .gu = .{ f(7), f(8) }, .dn = .{ f(9), f(10) }, .w16 = .{ .split64 = f(20), .split128 = f(21), .gu = f(22), .dn = f(23) } };
    try sfProve(&k, 0);
    // the switch off: the baseline exact kernels, 256 threads
    try std.testing.expectEqual(f(8).handle, sfExperts(&k, 0, false).?.f.handle);
    try std.testing.expectEqual(@as(u32, 256), sfExperts(&k, 0, false).?.threads);
    // the chained arithmetic: the baseline kernel at either setting (the twins are the exact form's)
    k.sf = .chained;
    k.gemm_w16 = true;
    try std.testing.expectEqual(f(9).handle, sfExperts(&k, 0, true).?.f.handle);
    try std.testing.expectEqual(@as(u32, 256), sfExperts(&k, 0, true).?.threads);
    // the exact arithmetic with the switch on: the twins, 512 threads
    k.sf = .exact;
    try std.testing.expectEqual(f(22).handle, sfExperts(&k, 0, false).?.f.handle);
    try std.testing.expectEqual(@as(u32, 512), sfExperts(&k, 0, false).?.threads);
    try std.testing.expectEqual(f(23).handle, sfExperts(&k, 0, true).?.f.handle);
    try std.testing.expectEqual(@as(u32, 512), sfExperts(&k, 0, true).?.threads);
    // an image without a twin keeps that kernel the baseline
    k.sfk.?.w16.gu = null;
    try std.testing.expectEqual(f(8).handle, sfExperts(&k, 0, false).?.f.handle);
    try std.testing.expectEqual(@as(u32, 256), sfExperts(&k, 0, false).?.threads);
    k.sfk.?.w16.gu = f(22);
    // the switch needs every twin: wo_a's, and the block-scaled kernels' where they loaded
    try std.testing.expect(!k.useGemmW16(true));
    k.pf_bf16w8_w16 = f(24);
    try std.testing.expect(k.useGemmW16(true));
    try std.testing.expect(k.gemm_w16);
    try std.testing.expect(!k.useGemmW16(false));
    try std.testing.expect(!k.gemm_w16);
    k.sfk.?.w16.dn = null;
    try std.testing.expect(!k.useGemmW16(true));
}

test "TF_DS_GEMM_W16's launches: the twins on the tiles' own shared memory, 512 threads" {
    const src = @embedFile("mx.zig");
    const code = src[0..std.mem.indexOf(u8, src, "\ntest \"").?];
    // the dense split tiles: the twin at 512 threads, else the baseline at 256, on the configurations' dynamic bytes
    for ([_][]const u8{
        "else if (w.split64) |f| try launchSharedT(f, s, .{ cdiv(m, 64), l.n / 128, 1 }, sf_split64_smem, 512, &args) else try launchShared(sk.split64[x], s, .{ cdiv(m, 64), l.n / 128, 1 }, sf_split64_smem, &args),",
        "else if (w.split128) |f| try launchSharedT(f, s, .{ cdiv(m, 128), l.n / 128, 1 }, sf_split128_smem, 512, &args) else try launchShared(sk.split128[x], s, .{ cdiv(m, 128), l.n / 128, 1 }, sf_split128_smem, &args),",
    }) |want| try std.testing.expect(std.mem.indexOf(u8, code, want) != null);
    // the twins are the exact form's alone: the switch is read with the arithmetic
    try std.testing.expect(std.mem.indexOf(u8, code, "const w = if (k.gemm_w16 and x == 1) sk.w16 else W16{};") != null);
    // wo_a's rows: the twin at 512, else the baseline at 256, on the kernel's dynamic bytes
    try std.testing.expect(std.mem.indexOf(u8, code, "try launchSharedT(f, s, .{ cdiv(b.n, 128), gn / 128, if (merged) groups else 1 }, pf_w8_smem, if (w16 != null) 512 else 256, &args);") != null);
    // the expert tiles launch the block sfExperts names (512 for a twin, 256 for the baseline)
    try std.testing.expect(std.mem.indexOf(u8, code, "return launchSharedT(fsf.f, s, .{ nx.tiles * 32 / 128, p / pf_bm + ex.count, 1 }, fsf.smem, fsf.threads, &args);") != null);
    try std.testing.expect(std.mem.indexOf(u8, code, "try launchSharedT(fsf.f, s, .{ n / 128, p / pf_bm + ex.count, 1 }, fsf.smem, fsf.threads, &args) else {") != null);
}

test "TF_DS_SPLIT64N takes the 64-column split tile for the exact arithmetic at N of 512 and 1,280 alone, and nothing else changes" {
    const f = struct {
        fn of(i: usize) cuda.Function {
            return .{ .d = undefined, .handle = @ptrFromInt(0x3000 + i * 16) };
        }
    }.of;
    var k: Kernels = undefined;
    k.sf = .exact;
    k.split64n = false;
    k.sfk = .{ .module = undefined, .gemm = .{ f(0), f(1) }, .split64 = .{ f(2), f(3) }, .split128 = .{ f(4), f(5) }, .zsplit = f(6), .gu = .{ f(7), f(8) }, .dn = .{ f(9), f(10) } };
    // an image without the kernel keeps the switch off
    try std.testing.expect(!k.useSplit64n(true));
    k.sfk.?.split64n = f(30);
    try std.testing.expect(!k.useSplit64n(false));
    try std.testing.expect(k.useSplit64n(true));
    const sk = k.sfk.?;
    // wkv (512) and wq_a (1,280) in the exact arithmetic: the 64-column kernel
    try std.testing.expectEqual(f(30).handle, sk.split64nOf(k.split64n, 1, 512, false, false).?.handle);
    try std.testing.expectEqual(f(30).handle, sk.split64nOf(k.split64n, 1, 1280, false, false).?.handle);
    // every other N, the chained arithmetic, the whole-line twin's output and the 16-warp twin keep the baseline launch
    for ([_]usize{ 128, 256, 1024, 2560, 4096, 8192, 12800 }) |n| try std.testing.expect(sk.split64nOf(k.split64n, 1, n, false, false) == null);
    try std.testing.expect(sk.split64nOf(k.split64n, 0, 512, false, false) == null);
    try std.testing.expect(sk.split64nOf(k.split64n, 1, 512, true, false) == null);
    try std.testing.expect(sk.split64nOf(k.split64n, 1, 1280, false, true) == null);
    // the switch off: none
    try std.testing.expect(sk.split64nOf(false, 1, 512, false, false) == null);
    // the launch: the 64-column grid at 128 threads on the kernel's own dynamic bytes, ahead of every other split64 launch
    const src = @embedFile("mx.zig");
    const code = src[0..std.mem.indexOf(u8, src, "\ntest \"").?];
    const at = std.mem.indexOf(u8, code, "." ++ "split64 => if (sk.split64nOf(k.split64n, x, l.n, stage, w.split64 != null)) |f| try launchSharedT(f, s, .{ cdiv(m, 64), l.n / 64, 1 }, sf_split64n_smem, 128, &args) else if (stage)") orelse return error.TestUnexpectedResult;
    _ = at;
    // two blocks an SM: the dynamic bytes (native_sf.cu asserts the same) and the kernel's static 2,304 and 1 KiB each within 100 KiB
    try std.testing.expectEqual(@as(u32, 5 * (64 * 64 + 64 * 64 + 64 * 2 * 4)), sf_split64n_smem);
    try std.testing.expect(2 * (sf_split64n_smem + 2304 + 1024) <= 100 * 1024);
}

test "TF_DS_PF_T64 takes the 64 x 64 bf16 tile for the router's 384 columns of one group alone, and nothing else changes" {
    var k: Kernels = undefined;
    k.pf_t64 = false;
    k.pf_bf16_t64 = .{ .d = undefined, .handle = @ptrFromInt(0x5000) };
    // on, N 384, one group: the 64 x 64 kernel
    try std.testing.expectEqual(k.pf_bf16_t64.?.handle, pfT64Of(&k, true, 384, 1).?.handle);
    // off, another N, or a grouped launch keeps the 128 x 128 tile
    try std.testing.expect(pfT64Of(&k, false, 384, 1) == null);
    for ([_]usize{ 64, 128, 256, 512, 1280, 2560, 4096 }) |n| try std.testing.expect(pfT64Of(&k, true, n, 1) == null);
    try std.testing.expect(pfT64Of(&k, true, 384, 2) == null);
    // an image without the kernel: none
    k.pf_bf16_t64 = null;
    try std.testing.expect(pfT64Of(&k, true, 384, 1) == null);
    // the launch: the 64 x 64 grid at 128 threads on the kernel's own dynamic bytes, ahead of the 128 x 128 launch
    const src = @embedFile("mx.zig");
    const code = src[0..std.mem.indexOf(u8, src, "\ntest \"").?];
    const at = std.mem.indexOf(u8, code, "if (pfT64Of(k, k.pf_t64, n, groups)) |f| return " ++ "launchSharedT(f, s, .{ cdiv(m, 64), n / 64, 1 }, pf_bf16_t64_smem, 128, &args)") orelse return error.TestUnexpectedResult;
    const shipping = std.mem.indexOf(u8, code, "return " ++ "launchShared(k.pf_bf16, s, .{ cdiv(m, 128), n / 128, groups }, pf_bf16_smem, &args)") orelse return error.TestUnexpectedResult;
    try std.testing.expect(at < shipping);
    // two CTAs an SM: 4 stages of (64 + 64) rows of 80 bytes (native_sf asserts the same), each CTA with its 1 KiB reserve within 100 KiB
    try std.testing.expectEqual(@as(u32, 4 * (64 + 64) * 80), pf_bf16_t64_smem);
    try std.testing.expect(2 * (pf_bf16_t64_smem + 1024) <= 100 * 1024);
}

test "TF_DS_GEMM_R2 takes its exact expert kernels, and only those; without every kernel or another arithmetic, the baseline's" {
    const f = struct {
        fn of(i: usize) cuda.Function {
            return .{ .d = undefined, .handle = @ptrFromInt(0x2000 + i * 16) };
        }
    }.of;
    var k: Kernels = undefined;
    k.sf = .exact;
    k.gemm_r2 = false;
    k.sf_nkept = 0;
    k.sf_nproven = 0;
    k.sfk = .{ .module = undefined, .gemm = .{ f(0), f(1) }, .split64 = .{ f(2), f(3) }, .split128 = .{ f(4), f(5) }, .zsplit = f(6), .gu = .{ f(7), f(8) }, .dn = .{ f(9), f(10) }, .r2 = .{ .gu = f(22), .dn = f(23) } };
    try sfProve(&k, 0);
    // off: the baseline kernels and shared memory
    try std.testing.expectEqual(f(8).handle, sfExperts(&k, 0, false).?.f.handle);
    try std.testing.expectEqual(pf_gu_r128_smem, sfExperts(&k, 0, false).?.smem);
    try std.testing.expectEqual(f(10).handle, sfExperts(&k, 0, true).?.f.handle);
    try std.testing.expectEqual(pf_dn_r128_smem, sfExperts(&k, 0, true).?.smem);
    // on: the R2 kernels, their shared memory
    try std.testing.expect(k.useGemmR2(true));
    try std.testing.expectEqual(f(22).handle, sfExperts(&k, 0, false).?.f.handle);
    try std.testing.expectEqual(pf_gu_r128_smem, sfExperts(&k, 0, false).?.smem);
    try std.testing.expectEqual(f(23).handle, sfExperts(&k, 0, true).?.f.handle);
    try std.testing.expectEqual(r2_dn_smem, sfExperts(&k, 0, true).?.smem);
    // the chained arithmetic is not R2's: off, the baseline kernels
    k.sf = .chained;
    try std.testing.expect(!k.useGemmR2(true));
    try std.testing.expectEqual(f(9).handle, sfExperts(&k, 0, true).?.f.handle);
    // an image without one R2 kernel: off
    k.sf = .exact;
    k.sfk.?.r2.dn = null;
    try std.testing.expect(!k.useGemmR2(true));
    try std.testing.expectEqual(f(10).handle, sfExperts(&k, 0, true).?.f.handle);
    k.sfk.?.r2.dn = f(23);
    try std.testing.expect(k.useGemmR2(true));
    try std.testing.expect(!k.useGemmR2(false));
    try std.testing.expect(!k.gemm_r2);
}

test "the door, two-base and down-on-the-parts kernels follow the block-scaled arithmetic in effect: the chained twins under TF_DS_MMA_SF, the exact ones under its exact form" {
    const f = struct {
        fn of(i: usize) cuda.Function {
            return .{ .d = undefined, .handle = @ptrFromInt(0x5000 + i * 16) };
        }
    }.of;
    var sk: SfKernels = undefined;
    sk.st_door = f(1);
    sk.st_two = f(2);
    sk.dn_three = f(3);
    sk.ch_door = f(11);
    sk.ch_two = f(12);
    sk.ch_dn_three = f(13);
    try std.testing.expectEqual(f(1).handle, sk.door(.exact).?.handle);
    try std.testing.expectEqual(f(2).handle, sk.two(.exact).?.handle);
    try std.testing.expectEqual(f(3).handle, sk.three(.exact).?.handle);
    try std.testing.expectEqual(f(11).handle, sk.door(.chained).?.handle);
    try std.testing.expectEqual(f(12).handle, sk.two(.chained).?.handle);
    try std.testing.expectEqual(f(13).handle, sk.three(.chained).?.handle);
    try std.testing.expect(sk.door(.off) == null and sk.two(.off) == null and sk.three(.off) == null);
    // an image without the chained twins: none under TF_DS_MMA_SF, the baseline copying path
    sk.ch_door = null;
    sk.ch_two = null;
    sk.ch_dn_three = null;
    try std.testing.expect(sk.door(.chained) == null and sk.two(.chained) == null and sk.three(.chained) == null);
    try std.testing.expect(sk.door(.exact) != null);
}

test "gemm, gate / up and down take TF_DS_MMA_SF's kernels before the baseline's, on the baseline grids and the twins' shared memory" {
    const src = @embedFile("mx.zig");
    const code = src[0..std.mem.indexOf(u8, src, "\ntest \"").?];
    const gat = std.mem.indexOf(u8, code, "fn gemmBlock(").?;
    const gemm_body = code[gat .. gat + std.mem.indexOf(u8, code[gat..], "\n}\n").?];
    const sfl = std.mem.indexOf(u8, gemm_body, "if (sfRoute(k, l, q, ldq, sc, lds, y, ldy, accumulate, sel)) |kind| return sfGemm(k, s, kind, l, q, ldq, sc, lds, m, y, ldy, y_dtype, accumulate, z);") orelse return error.TestUnexpectedResult;
    const pfl = std.mem.indexOf(u8, gemm_body, "if (pfDense(k, l, q, ldq, sc, lds, sel, y, ldy, accumulate)) {").?;
    try std.testing.expect(sfl < pfl);
    const fns = [_][]const u8{ "fn gateUpOf(", "fn downOf(" };
    const sfs = [_][]const u8{
        "if (sfExperts(k, nx.s1, false)) |fsf| return launchSharedT(fsf.f, s, .{ nx.tiles * 32 / 128, p / pf_bm + ex.count, 1 }, fsf.smem, fsf.threads, &args);",
        "if (sfExperts(k, nx.s1, true)) |fsf| try launchSharedT(fsf.f, s, .{ n / 128, p / pf_bm + ex.count, 1 }, fsf.smem, fsf.threads, &args) else {",
    };
    for (fns, sfs) |fname, sfline| {
        const at = std.mem.indexOf(u8, code, fname).?;
        const body = code[at .. at + std.mem.indexOf(u8, code[at..], "\n}\n").?];
        const tl = std.mem.indexOf(u8, body, "args.add(sc.ntiles);").?;
        const sf_at = std.mem.indexOf(u8, body, sfline) orelse return error.TestUnexpectedResult;
        const el = std.mem.indexOf(u8, body, "const el = pfExpertsLaunch(k.run128);").?;
        try std.testing.expect(tl < sf_at and sf_at < el);
    }
}

test "TF_DS_MMA_SF_EXACT's proven scale ranges: within them every scaled block sum is a normal float" {
    const min_normal = std.math.floatMin(f32);
    // FP8 x FP8: a nonzero block sum at least 2^-18 (two E4M3 subnormals' product), an activation scale at least 2^-22
    // (act_quant's floor); the upper end is the baseline FAST range (2^0)
    const lo8: i32 = sf_fp8_scales[0];
    try std.testing.expect(std.math.ldexp(@as(f64, 1), -18 - 22 + lo8 - 127) >= min_normal);
    try std.testing.expect(std.math.ldexp(@as(f64, 1), -18 - 22 + (lo8 - 1) - 127) < min_normal);
    try std.testing.expectEqual(@as(u8, 127), sf_fp8_scales[1]);
    // E4M3 x E2M1: a nonzero block sum at least 2^-10 (an E4M3 subnormal times 0.5); at most 32 x 448 x 6, with an
    // activation scale at most 2^100 (the baseline FAST assumption) below fp32's largest
    const lo4: i32 = sf_fp4_scales[0];
    const hi4: i32 = sf_fp4_scales[1];
    try std.testing.expect(std.math.ldexp(@as(f64, 1), -10 - 22 + lo4 - 127) >= min_normal);
    try std.testing.expect(std.math.ldexp(@as(f64, 1), -10 - 22 + (lo4 - 1) - 127) < min_normal);
    try std.testing.expect(32.0 * 448.0 * 6.0 * std.math.ldexp(@as(f64, 1), 100 + hi4 - 127) < std.math.floatMax(f32));
    // sfProven: both ends inside
    try std.testing.expect(sfProven(.{ 41, 127 }, false) and !sfProven(.{ 40, 100 }, false) and !sfProven(.{ 60, 128 }, false));
    try std.testing.expect(sfProven(.{ 33, 137 }, true) and !sfProven(.{ 32, 100 }, true) and !sfProven(.{ 60, 138 }, true));
}

test "linearScales and expertScales give each scale tensor's exact bytes: owner's rest rows, the shared expert's FP8 blocks, never the packing's gaps" {
    const l: weights.Linear = .{ .words = 1, .w = 1, .s = 0x5000, .k = 5120, .n = 1280, .fmt = .fp8 };
    try std.testing.expectEqual(ByteRange{ .ptr = 0x5000, .len = 40 * 160 }, linearScales(l));
    var ex = std.mem.zeroes(weights.Experts);
    ex.count = 3; // two routed experts and the shared one
    ex.dims = 5120;
    ex.width = 512;
    ex.down_k = 1152;
    ex.down_n = 2560;
    ex.native = .{ .w1 = 0, .w3 = 0, .s1 = 0, .s3 = 0, .w2 = 0, .s2 = 0, .tiles = 20, .rest_tiles = 4 };
    const s1 = [_]u64{ 0x10000, 0x20000, 0x30000 };
    const s3 = [_]u64{ 0x11000, 0x21000, 0x31000 };
    const s2 = [_]u64{ 0x12000, 0x22000, 0x32000 };
    var out: [9]ByteRange = undefined;
    expertScales(ex, &s1, &s3, &s2, &[_]i32{ 1, 0, 1 }, &out);
    // expert 0 computes its rest (640 rows), expert 1 does not (512); MXFP4 [rows, 160], down [2,560, 36]
    try std.testing.expectEqual(ByteRange{ .ptr = 0x10000, .len = 640 * 160 }, out[0]);
    try std.testing.expectEqual(ByteRange{ .ptr = 0x11000, .len = 640 * 160 }, out[1]);
    try std.testing.expectEqual(ByteRange{ .ptr = 0x12000, .len = 2560 * 36 }, out[2]);
    try std.testing.expectEqual(ByteRange{ .ptr = 0x20000, .len = 512 * 160 }, out[3]);
    try std.testing.expectEqual(ByteRange{ .ptr = 0x22000, .len = 2560 * 36 }, out[5]);
    // the shared expert (the last) FP8: [rows / 32, 160] and [2,560 / 32, 36]
    try std.testing.expectEqual(ByteRange{ .ptr = 0x30000, .len = 20 * 160 }, out[6]);
    try std.testing.expectEqual(ByteRange{ .ptr = 0x32000, .len = 80 * 36 }, out[8]);
    // without a rest (no owner table): the main block alone
    expertScales(ex, &s1, &s3, &s2, null, &out);
    try std.testing.expectEqual(@as(u64, 512 * 160), out[0].len);
    try std.testing.expectEqual(@as(u64, 16 * 160), out[6].len);
}

test "TF_DS_MMA_SF_EXACT runs only what its guard proved: a dense linear by its scale pointer, a layer's experts by their gate scale table; the kept and the unread on the baseline kernels" {
    var k: Kernels = undefined;
    k.sf = .exact;
    k.sf_nkept = 0;
    k.sf_nproven = 0;
    k.pf_off = false;
    k.sfk = .{ .module = undefined, .gemm = undefined, .split64 = undefined, .split128 = undefined, .zsplit = undefined, .gu = .{ .{ .d = undefined, .handle = @ptrFromInt(0x10) }, .{ .d = undefined, .handle = @ptrFromInt(0x20) } }, .dn = .{ .{ .d = undefined, .handle = @ptrFromInt(0x30) }, .{ .d = undefined, .handle = @ptrFromInt(0x40) } } };
    const l: weights.Linear = .{ .words = 1, .w = 0x1000, .s = 0x5000, .k = 5120, .n = 1280, .fmt = .fp8 };
    // nothing proven: the baseline kernels
    try std.testing.expect(!sfDense(&k, l, 0x2000, 5120, 0x3000, 160, 0, 0, false));
    try std.testing.expect(sfExperts(&k, 0x7000, false) == null);
    // the proven keys (each once): the block-scaled kernels
    try sfProve(&k, 0x7000);
    try sfProve(&k, 0x5000);
    try sfProve(&k, 0x5000);
    try std.testing.expectEqual(@as(usize, 2), k.sf_nproven);
    try std.testing.expect(sfProvenKey(&k, 0x5000) and sfProvenKey(&k, 0x7000) and !sfProvenKey(&k, 0x6000));
    try std.testing.expect(sfDense(&k, l, 0x2000, 5120, 0x3000, 160, 0, 0, false));
    try std.testing.expect(sfExperts(&k, 0x7000, false) != null and sfExperts(&k, 0x7000, true) != null);
    try std.testing.expect(sfExperts(&k, 0x8000, true) == null);
    // a kept key (the start line's words) is no proven one
    try sfKeep(&k, 0x6000);
    try std.testing.expect(sfKept(&k, 0x6000) and !sfKept(&k, 0x5000) and !sfProvenKey(&k, 0x6000));
    // the chained arithmetic (TF_DS_MMA_SF) has no guard: neither set is read
    k.sf = .chained;
    k.sf_nproven = 0;
    try std.testing.expect(sfDense(&k, l, 0x2000, 5120, 0x3000, 160, 0, 0, false));
    try std.testing.expect(sfExperts(&k, 0x8000, false) != null);
    // keys proven in any order are found; the sets' bounds: one more is refused
    k.sf_nproven = 0;
    for ([_]u64{ 0x900, 0x100, 0x500, 0x300 }) |x| try sfProve(&k, x);
    for ([_]u64{ 0x100, 0x300, 0x500, 0x900 }) |x| try std.testing.expect(sfProvenKey(&k, x));
    try std.testing.expect(!sfProvenKey(&k, 0x200) and !sfProvenKey(&k, 0x1000) and !sfProvenKey(&k, 0x50));
    k.sf_nproven = 0;
    for (0..sf_proven_max) |i| try sfProve(&k, 0x10_0000 + i * 64);
    try std.testing.expectError(error.TooManyProven, sfProve(&k, 0x9999));
    k.sf_nkept = 0;
    for (0..sf_kept_max) |i| try sfKeep(&k, 0x100 + i);
    try std.testing.expectError(error.TooManyKept, sfKeep(&k, 0x9999));
}

test "TF_DS_MMA_SF_EXACT's route: sfDense takes a dense shape only with a K split, even with its key proven; sfRoute is mx.gemm's kernel at m rows" {
    var k: Kernels = undefined;
    k.sf = .exact;
    k.sf_nkept = 0;
    k.sf_nproven = 0;
    k.pf_off = false;
    // Engram's wkv (3,072 x 12,800): no K split; its key proven, still the baseline kernel in the exact arithmetic
    const unsplit: weights.Linear = .{ .words = 1, .w = 0x1000, .s = 0x5000, .k = 3072, .n = 12800, .fmt = .fp8 };
    try std.testing.expect(sfShape(3072, 12800) and !sfExactShape(3072, 12800));
    try sfProve(&k, unsplit.s);
    try std.testing.expect(!sfDense(&k, unsplit, 0x2000, 3072, 0x3000, 96, 0, 0, false));
    for ([_]usize{ 1, 64, 512, 4096 }) |m| try std.testing.expectEqual(@as(?SfGemm, null), sfRoute(&k, unsplit, 0x2000, 3072, 0x3000, 96, 0, 0, false, m));
    // the chained arithmetic takes it (its own kernel, no guard)
    k.sf = .chained;
    try std.testing.expect(sfDense(&k, unsplit, 0x2000, 3072, 0x3000, 96, 0, 0, false));
    try std.testing.expectEqual(@as(?SfGemm, .gemm), sfRoute(&k, unsplit, 0x2000, 3072, 0x3000, 96, 0, 0, false, 512));
    // a split shape (wq_a), proven: the baseline below 64 rows, 64-row tiles to 1,024, 128-row ones past; unproven: the baseline
    k.sf = .exact;
    const l: weights.Linear = .{ .words = 1, .w = 0x1000, .s = 0x6000, .k = 5120, .n = 1280, .fmt = .fp8 };
    for ([_]usize{ 1, 63, 64, 512, 1000, 1024, 1025, 4096 }) |m| try std.testing.expectEqual(@as(?SfGemm, null), sfRoute(&k, l, 0x2000, 5120, 0x3000, 160, 0, 0, false, m));
    try sfProve(&k, l.s);
    for ([_]usize{ 1, 63 }) |m| try std.testing.expectEqual(@as(?SfGemm, null), sfRoute(&k, l, 0x2000, 5120, 0x3000, 160, 0, 0, false, m));
    for ([_]usize{ 64, 512, 1000, 1024 }) |m| try std.testing.expectEqual(@as(?SfGemm, .split64), sfRoute(&k, l, 0x2000, 5120, 0x3000, 160, 0, 0, false, m));
    for ([_]usize{ 1025, 4096 }) |m| try std.testing.expectEqual(@as(?SfGemm, .split128), sfRoute(&k, l, 0x2000, 5120, 0x3000, 160, 0, 0, false, m));
    // mx.gemm takes the route sfRoute names
    const src = @embedFile("mx.zig");
    // (TF_DS_GEMM_HALVES off: routeRows gives the launch's own m, so this is still mx.gemm's kernel at m rows)
    for ([_]usize{ 1, 64, 1025, 4096, 8192 }) |m| try std.testing.expectEqual(m, routeRows(m, false));
    const gat = std.mem.indexOf(u8, src, "const sel = routeRows(m, gemm_halves);") orelse return error.TestUnexpectedResult;
    const at = std.mem.indexOf(u8, src, "fn gemmBlock(k: *const Kernels") orelse return error.TestUnexpectedResult;
    try std.testing.expect(gat < at);
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n}\n") orelse return error.TestUnexpectedResult)];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "if (sfRoute(k, l, q, ldq, sc, lds, y, ldy, accumulate, sel)) |kind| return sfGemm(k, s, kind, l, q, ldq, sc, lds, m, y, ldy, y_dtype, accumulate, z);"));
}

test "TF_DS_MMA_SF_EXACT's guard fails closed: a tensor proven only by a real read inside the bounds, or by a length of 0" {
    // the kernel's answer for a range it read nothing of (255, 0); a result left unwritten (0xff filled) reads as one
    try std.testing.expectEqual([2]u8{ 255, 0 }, readRange(255, 0));
    try std.testing.expectEqual([2]u8{ 1, 0 }, readRange(0xffff_ffff, 0xffff_ffff));
    try std.testing.expectEqual([2]u8{ 1, 0 }, readRange(41, 0x1ff));
    try std.testing.expectEqual([2]u8{ 41, 127 }, readRange(41, 127));
    try std.testing.expect(sfProvenRead(0, .{ 255, 0 }, false)); // nothing to prove
    try std.testing.expect(!sfProvenRead(1024, .{ 255, 0 }, false)); // a tensor the kernel read nothing of
    try std.testing.expect(!sfProvenRead(1024, readRange(0xffff_ffff, 0xffff_ffff), true)); // an unwritten result
    try std.testing.expect(sfProvenRead(1024, .{ 41, 127 }, false) and !sfProvenRead(1024, .{ 40, 127 }, false) and !sfProvenRead(1024, .{ 41, 128 }, false));
    try std.testing.expect(sfProvenRead(1024, .{ 33, 137 }, true) and !sfProvenRead(1024, .{ 32, 137 }, true) and !sfProvenRead(1024, .{ 33, 138 }, true));
    // sfProven itself needs a real range (lo <= hi)
    try std.testing.expect(!sfProven(.{ 255, 0 }, false) and !sfProven(.{ 255, 0 }, true) and !sfProven(.{ 100, 99 }, false));
}

test "TF_DS_MMA_SF_EXACT's mode after its agreement: refused, it turns only its own arithmetic off and leaves TF_DS_MMA_SF's" {
    try std.testing.expectEqual(MmaSf.chained, sfAfterExact(.chained, false));
    try std.testing.expectEqual(MmaSf.off, sfAfterExact(.exact, false));
    try std.testing.expectEqual(MmaSf.exact, sfAfterExact(.exact, true));
    try std.testing.expectEqual(MmaSf.off, sfAfterExact(.off, false));
    try std.testing.expectEqual(MmaSf.chained, sfAfterExact(.chained, true));
}

test "TF_DS_MMA_SF_EXACT takes a dense shape only with a K split: Engram's wkv (none) stays on the baseline kernel" {
    try std.testing.expect(sfExactShape(5120, 1280) and sfExactShape(5120, 512) and sfExactShape(1280, 8192) and sfExactShape(4096, 2560) and sfExactShape(1280, 4096));
    try std.testing.expect(sfShape(3072, 12800) and !sfExactShape(3072, 12800));
    try std.testing.expect(!sfExactShape(5120, 1280 - 64));
}

test "the guard's tables go up on the stream that reads them, its results filled with 0xff first" {
    const src = @embedFile("mx.zig");
    const at = std.mem.indexOf(u8, src, "pub fn scaleRanges(") orelse return error.TestUnexpectedResult;
    const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n}\n") orelse return error.TestUnexpectedResult)];
    const up = std.mem.indexOf(u8, body, "try buf.uploadAsync(0, std.mem.sliceAsBytes(host), s.handle);") orelse return error.TestUnexpectedResult;
    const fill = std.mem.indexOf(u8, body, "try d.check(d.api.cuMemsetD8Async(mm, 0xff, ranges.len * 8, s.handle), \"cuMemsetD8Async\");") orelse return error.TestUnexpectedResult;
    const go = std.mem.indexOf(u8, body, "try launch(f, s,") orelse return error.TestUnexpectedResult;
    try std.testing.expect(up < fill and fill < go);
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, body, "buf.upload(0,"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "o.* = readRange(got[2 * i], got[2 * i + 1]);"));
}

test "TF_DS_ROCI_SENT_FIRST: a launch in a row-quarter order runs every tile once, the sent quarters' tiles in its first phase" {
    for ([_]usize{ 4096, 2048 }) |rows| {
        const q = rows / 4;
        const cols: usize = 20;
        const row_tiles = rows / 128;
        for (0..4) |g| {
            // the plan's sent quarters of node g (rows of g ^ 1 and g ^ 3), the kept ones (g, g ^ 2)
            const first = [2]usize{ (g ^ 1) * q, (g ^ 3) * q };
            const last = [2]usize{ g * q, (g ^ 2) * q };
            const ro = rowOrder(first, last, q, cols).?;
            try std.testing.expectEqual(@as(u32, @intCast(row_tiles / 2 * cols)), ro.span);
            var seen = try std.testing.allocator.alloc(bool, row_tiles * cols);
            defer std.testing.allocator.free(seen);
            @memset(seen, false);
            for (0..row_tiles * cols) |o| {
                const t = tileOf(ro, @intCast(o));
                try std.testing.expect(t.row < row_tiles and t.col < cols);
                const at = t.row * cols + t.col;
                try std.testing.expect(!seen[at]);
                seen[at] = true;
                // the first span CTAs hold the rows of the two sent quarters, every column of them
                const quarter = t.row / (row_tiles / 4);
                const sent = quarter == (g ^ 1) or quarter == (g ^ 3);
                try std.testing.expectEqual(o < ro.span, sent);
            }
            for (seen) |b| try std.testing.expect(b);
            // within a phase the row tiles go fastest: the first per CTAs are the first column's tiles
            try std.testing.expectEqual(@as(u32, 0), tileOf(ro, 0).col);
            try std.testing.expectEqual(@as(u32, 0), tileOf(ro, @intCast(row_tiles / 2 - 1)).col);
            try std.testing.expectEqual(@as(u32, 1), tileOf(ro, @intCast(row_tiles / 2)).col);
            try std.testing.expectEqual(@as(u32, 0), tileOf(ro, ro.span).col);
        }
        // the baseline order is the identity order
        const id = identityRowOrder(rows, cols).?;
        for (0..row_tiles * cols) |o| {
            const t = tileOf(id, @intCast(o));
            try std.testing.expectEqual(@as(u32, @intCast(o % row_tiles)), t.row);
            try std.testing.expectEqual(@as(u32, @intCast(o / row_tiles)), t.col);
        }
    }
}

test "TF_DS_ROCI_SENT_FIRST: an order refuses what is not four quarters each once, a quarter that is not whole power-of-two row tiles" {
    try std.testing.expect(rowOrder(.{ 1024, 3072 }, .{ 0, 2048 }, 1024, 20) != null);
    try std.testing.expectEqual(@as(?RowOrder, null), rowOrder(.{ 1024, 3072 }, .{ 0, 0 }, 1024, 20)); // a quarter twice
    try std.testing.expectEqual(@as(?RowOrder, null), rowOrder(.{ 1024, 3072 }, .{ 0, 4096 }, 1024, 20)); // a fifth
    try std.testing.expectEqual(@as(?RowOrder, null), rowOrder(.{ 1024, 3072 }, .{ 0, 2000 }, 1024, 20)); // not on a quarter
    try std.testing.expectEqual(@as(?RowOrder, null), rowOrder(.{ 384, 1152 }, .{ 0, 768 }, 384, 20)); // 3 row tiles a quarter
    try std.testing.expectEqual(@as(?RowOrder, null), rowOrder(.{ 100, 300 }, .{ 0, 200 }, 100, 20)); // not whole row tiles
    try std.testing.expectEqual(@as(?RowOrder, null), identityRowOrder(3 * 128, 20));
    // the combine's: 1,024 rows x 2,560 columns = 10,240 CTAs a quarter, the sent ones first
    const qo = quarterOrder(.{ 1024, 3072 }, .{ 0, 2048 }, 1024, 2560).?;
    try std.testing.expectEqual(@as(u32, 10240), qo.unit);
    try std.testing.expectEqual(@as(u32, 20480), qo.span);
    try std.testing.expectEqual(@as(u32, 0x02_00_03_01), qo.quarters);
    try std.testing.expectEqual(@as(?QuarterOrder, null), quarterOrder(.{ 100, 300 }, .{ 0, 200 }, 100, 2561)); // 256000+ elements not whole CTAs
    try std.testing.expectEqual(@as(?QuarterOrder, null), quarterOrder(.{ 1024, 1024 }, .{ 0, 2048 }, 1024, 2560));
}

test "TF_DS_ROCI_SENT_FIRST: the kernels' argument structs are the device structs' size" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(RowOrder));
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(QuarterOrder));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Door));
}

test "TF_DS_GEMM_HALVES's row blocks: 4,096 rows each at most, never past the launch's m, the last one the remainder" {
    // off (or at the block rows and fewer): one block of the launch's rows
    for ([_]usize{ 1, 4096, 4097, 6144, 8192 }) |m| {
        try std.testing.expectEqual(@as(usize, 1), blockCount(m, false));
        try std.testing.expectEqual(Block{ .start = 0, .n = m }, blockAt(m, false, 0));
    }
    try std.testing.expectEqual(@as(usize, 1), blockCount(4096, true));
    try std.testing.expectEqual(@as(usize, 2), blockCount(4097, true));
    try std.testing.expectEqual(@as(usize, 2), blockCount(6144, true));
    try std.testing.expectEqual(@as(usize, 2), blockCount(8192, true));
    // on: the blocks start at 0, stay within the launch's rows and cover them; no block crosses the pass's last row
    for ([_]usize{ 4097, 4100, 6144, 8192 }) |m| {
        var at: usize = 0;
        var i: usize = 0;
        while (i < blockCount(m, true)) : (i += 1) {
            const b = blockAt(m, true, i);
            try std.testing.expectEqual(at, b.start);
            try std.testing.expect(b.n > 0 and b.n <= gemm_halves_rows);
            at += b.n;
        }
        try std.testing.expectEqual(m, at);
        try std.testing.expectEqual(@as(usize, 0), blockAt(m, true, blockCount(m, true)).n);
    }
    try std.testing.expectEqual(Block{ .start = 0, .n = 4096 }, blockAt(8192, true, 0));
    try std.testing.expectEqual(Block{ .start = 4096, .n = 4096 }, blockAt(8192, true, 1));
    try std.testing.expectEqual(Block{ .start = 4096, .n = 2048 }, blockAt(6144, true, 1));
    try std.testing.expectEqual(Block{ .start = 4096, .n = 4 }, blockAt(4100, true, 1)); // a merged final pass's tail
    try std.testing.expectEqual(Block{ .start = 4096, .n = 1 }, blockAt(4097, true, 1));
    // routeRows: the pass's block cap past it, the launch's own rows otherwise
    try std.testing.expectEqual(@as(usize, 4096), routeRows(8192, true));
    try std.testing.expectEqual(@as(usize, 4096), routeRows(4097, true));
    try std.testing.expectEqual(@as(usize, 4096), routeRows(4096, true));
    try std.testing.expectEqual(@as(usize, 4096), routeRows(4096, false));
    try std.testing.expectEqual(@as(usize, 8192), routeRows(8192, false));
    // every block of an 8,192-row pass picks the kernel, split and variant the 4,096-row pass picks (the pass's block
    // cap), though a shorter block's own rows would pick another (2,048 rows: a 64-row-tile split past 1,024; 512: a
    // 64-row-tile split; 4 rows: the fold kernel, or the baseline kernels below 64)
    for ([_]usize{ 5120, 3072, 1280 }) |kdim| for ([_]usize{ 1280, 512, 8192 }) |n| {
        if (n % 128 != 0 or kdim % 64 != 0) continue;
        for ([_]PfSplit{ .off, .table, .a, .b, .c6, .c7 }) |sw| try std.testing.expect(std.meta.eql(pfGemmKernel(sw, true, 4096, kdim, n), pfGemmKernel(sw, true, routeRows(8192, true), kdim, n)));
        for ([_]MmaSf{ .chained, .exact }) |sw| try std.testing.expectEqual(sfGemmOf(sw, 4096, splits(kdim, n)), sfGemmOf(sw, routeRows(8192, true), splits(kdim, n)));
    };
    try std.testing.expectEqual(SfGemm.split128, sfGemmOf(.exact, 4096, 16).?);
    try std.testing.expectEqual(SfGemm.split64, sfGemmOf(.exact, 512, 16).?); // where a tail block's own rows would go
    try std.testing.expect(std.meta.eql(PfGemm.fold, pfGemmOf(.table, 512, 5120, 1280)));
    try std.testing.expect(std.meta.eql(PfGemm{ .split = .a }, pfGemmOf(.table, 4096, 5120, 1280)));
}

test "TF_DS_GEMM_HALVES's launches: gemm, bf16Gemm and woA8Prefill run the row blocks; the experts keep the pass" {
    const src = @embedFile("mx.zig");
    // one block loop in each launch (the experts' gateUp / down launch no dense GEMM: they keep the pass)
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, src, "\n        const b = blockAt(m, gemm_halves, i);"));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, src, "\n    while (i < blockCount(m, gemm_halves))"));
    // gemm and bf16Gemm choose their kernel, split and variant at the pass's block cap (sel), and read their rows at
    // the block's offsets: the codes, the scales and the output alike
    for ([_][]const u8{
        "const sel = routeRows(m, gemm_halves);",
        "q + b.start * ldq, ldq, sc + b.start * lds * 4, lds, b.n, y + b.start * ldy * esz",
        "sfRoute(k, l, q, ldq, sc, lds, y, ldy, accumulate, sel)",
        "pfDense(k, l, q, ldq, sc, lds, sel, y, ldy, accumulate)",
        "pfGemmKernel(k.split, k.gemm5, sel, l.k, l.n)",
        "x + b.start * ldx * 2, ldx, w, ldw, b.n, n, kdim, y + b.start * ldy * ysz",
        "sel >= pf_min_rows",
        "q = sc.q + b.start * kk + g * gk,",
        "y = u + b.start * ldu * 2 + g * gn * 2,",
    }) |line| try std.testing.expect(std.mem.indexOf(u8, src, line) != null);
    // the experts' launches take the whole pass (their scratch and tiles are the pass's)
    for ([_][]const u8{ "pub fn gateUp(", "pub fn down(" }) |sig| {
        const at = std.mem.indexOf(u8, src, sig) orelse return error.TestUnexpectedResult;
        const body = src[at .. at + (std.mem.indexOf(u8, src[at..], "\n}\n") orelse return error.TestUnexpectedResult)];
        try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, body, "gemm_halves"));
    }
}

test "TF_DS_PF_RASTER's tile order: every (row tile, column tile) is visited exactly once, group by group, for odd grids too" {
    const a = std.testing.allocator;
    for ([_]u32{ 0, 2, 4, 8, 16 }) |g| for ([_][2]u32{ .{ 1, 1 }, .{ 1, 9 }, .{ 3, 5 }, .{ 7, 1 }, .{ 16, 10 }, .{ 17, 3 }, .{ 33, 7 }, .{ 64, 10 }, .{ 65, 20 }, .{ 129, 9 } }) |grid| {
        const seen = try a.alloc(u8, grid[0] * grid[1]);
        defer a.free(seen);
        @memset(seen, 0);
        for (0..grid[1]) |by| for (0..grid[0]) |bx| {
            const t = rasterTile(g, grid[0], grid[1], @intCast(bx), @intCast(by));
            try std.testing.expect(t[0] < grid[0] and t[1] < grid[1]);
            seen[t[1] * grid[0] + t[0]] += 1;
        };
        for (seen) |c| try std.testing.expectEqual(@as(u8, 1), c);
    };
    // the walk: group 8 on 64 x 10 tiles takes rows 0 .. 7 of column 0, then rows 0 .. 7 of column 1, and the second group
    // of rows only after the tenth column
    try std.testing.expectEqual([2]u32{ 7, 0 }, rasterTile(8, 64, 10, 7, 0));
    try std.testing.expectEqual([2]u32{ 0, 1 }, rasterTile(8, 64, 10, 8, 0));
    try std.testing.expectEqual([2]u32{ 7, 9 }, rasterTile(8, 64, 10, 79, 0));
    try std.testing.expectEqual([2]u32{ 8, 0 }, rasterTile(8, 64, 10, 80, 0));
    // the last group is the remaining row tiles: 65 row tiles at group 8 end with one row tile across the columns
    try std.testing.expectEqual([2]u32{ 64, 0 }, rasterTile(8, 65, 3, 65 * 3 - 3, 0));
    try std.testing.expectEqual([2]u32{ 64, 2 }, rasterTile(8, 65, 3, 65 * 3 - 1, 0));
    // off: the default order
    try std.testing.expectEqual([2]u32{ 5, 2 }, rasterTile(0, 64, 10, 5, 2));
}

test "TF_DS_PF_RASTER's gate: the group applies from an activation slab of 14 MiB, never when off" {
    try std.testing.expectEqual(@as(c_int, 8), rasterOf(8, 4096, 5120)); // wq_a's input at a 4,096-row pass: 20 MiB
    try std.testing.expectEqual(@as(c_int, 8), rasterOf(8, 16384, 4096));
    try std.testing.expectEqual(@as(c_int, 0), rasterOf(8, 4096, 1280)); // wq_b's input: 5 MiB stays in the L2
    try std.testing.expectEqual(@as(c_int, 0), rasterOf(8, 2048, 5120)); // a 2,048-row pass: 10 MiB
    try std.testing.expectEqual(@as(c_int, 0), rasterOf(8, 2867, 5120)); // 14,679,040 B: one row short of 14 MiB
    try std.testing.expectEqual(@as(c_int, 4), rasterOf(4, 2868, 5120));
    try std.testing.expectEqual(@as(c_int, 0), rasterOf(0, 16384, 5120));
    // wo_a's FP8 rows: a group's codes are gk = 4,096 bytes a row (woA8Prefill's slab per launch, a block's rows under halves)
    try std.testing.expectEqual(@as(c_int, 8), rasterOf(8, 4096, 4096)); // 16 MiB
    try std.testing.expectEqual(@as(c_int, 8), rasterOf(8, 3584, 4096)); // exactly 14 MiB
    try std.testing.expectEqual(@as(c_int, 0), rasterOf(8, 3583, 4096));
    try std.testing.expectEqual(@as(c_int, 0), rasterOf(8, 2048, 4096)); // 8 MiB
    try std.testing.expectEqual(@as(usize, 14 << 20), raster_slab_bytes);
}

test "TF_DS_PF_RASTER's launches: the tile GEMMs get the group, the argument structs carry it last" {
    const src = @embedFile("mx.zig");
    // the dense FP8 launches (the prefill tile kernels, the block-scaled MMA) and the bf16 tile GEMM; the experts, the
    // TWO / ROWMAP launches and the decode GEMMs keep the default order
    // (each count includes the literal in this test)
    try std.testing.expectEqual(@as(usize, 2 + 1), std.mem.count(u8, src, ".raster = rasterOf(pf_raster, m, l.k) }"));
    try std.testing.expectEqual(@as(usize, 1 + 1), std.mem.count(u8, src, ".raster = rasterOf(pf_raster, m, kdim * 2) }"));
    // wo_a's FP8 rows (woA8Prefill: one tf_dsn_pf_bf16w8 launch a group and row block, the slab a group's codes) take it too,
    // so TF_DS_GEMM_HALVES splits no prompt launch the group leaves in the default order
    try std.testing.expectEqual(@as(usize, 1 + 1), std.mem.count(u8, src, ".raster = rasterOf(pf_raster, b.n, gk),"));
    try std.testing.expectEqual(@as(usize, 96), @offsetOf(W8Args, "raster"));
    try std.testing.expectEqual(@as(usize, 144), @sizeOf(W8Args));
    try std.testing.expectEqual(@as(usize, 104), @offsetOf(W8Args, "gq"));
    try std.testing.expectEqual(@as(usize, 136), @offsetOf(W8Args, "gy"));
    // the argument structs end in the field, in step with the device structs (native.cu MxArgs / BfArgs / W8Args)
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(MxArgs, "raster") % 4);
    try std.testing.expectEqual(@offsetOf(MxArgs, "accumulate") + 4, @offsetOf(MxArgs, "raster"));
    try std.testing.expectEqual(@offsetOf(BfArgs, "z") + 8, @offsetOf(BfArgs, "raster"));
}

test "the expert-count log: device words decode to records in call order, empty slots left out, and out.len bounds them" {
    // the kernel's ring: word 0 the call counter, 8 words of header, then four words a record {index + 1, rows << 32 | active, shared << 32 | owned, 0}
    var words: [4 * 8]u64 = @splat(0);
    words[4 * 2] = 3; // call index 2 (stored + 1)
    words[4 * 2 + 1] = (3 << 32) | 14;
    words[4 * 2 + 2] = (2 << 32) | 6;
    words[4 * 5] = 6;
    words[4 * 5 + 1] = (3 << 32) | 15;
    words[4 * 5 + 2] = (1 << 32) | 7;
    var out: [4]NactRec = undefined;
    try std.testing.expectEqual(@as(usize, 2), nactDecode(&words, &out));
    try std.testing.expectEqual(NactRec{ .call = 2, .rows = 3, .active = 14, .owned = 6, .shared = 2 }, out[0]);
    try std.testing.expectEqual(NactRec{ .call = 5, .rows = 3, .active = 15, .owned = 7, .shared = 1 }, out[1]);
    var one: [1]NactRec = undefined;
    try std.testing.expectEqual(@as(usize, 1), nactDecode(&words, &one));
}

test "the expert-count log file: a header naming the rank and the layout, then the records in call order" {
    const gpa = std.testing.allocator;
    var recs = [_]NactRec{ .{ .call = 9, .rows = 2, .active = 11, .owned = 5, .shared = 2 }, .{ .call = 8, .rows = 2, .active = 10, .owned = 4, .shared = 0 } };
    const text = try nactRender(gpa, 2, 4, &recs);
    defer gpa.free(text);
    try std.testing.expectEqualStrings("# TF_DS_NACT_LOG v1 rank=2 world=4 records=2 columns=call rows active owned shared (shared: 0 not active, 1 active and unowned, 2 active and owned)\n8 2 10 4 0\n9 2 11 5 2\n", text);
}

test "wo_b's door GEMM is the twin whose CTAs release the count at GPU scope (the system fence by the last arrival alone), as combine's door is" {
    const src = @embedFile("mx.zig");
    const at = std.mem.indexOf(u8, src, "fn useStaged(sk: *SfKernels) !void {").?;
    const body = src[at..][0..900];
    try std.testing.expect(std.mem.indexOf(u8, body, "sk.st_door = try Kernels.pfFunction(sk.module, \"tf_dsn_sf_split128_exact_st_door_g_kernel\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tf_dsn_sf_split128_exact_st_door_kernel") == null);
    // and combine's door is the same release
    try std.testing.expect(std.mem.indexOf(u8, src, "k.combine_door = try k.module.function(\"tf_dsn_combine_door_g_kernel\")") != null);
}

test "TF_DS_AQ_VEC: unset, empty and 1 are on, 0 is off, any other text is null (the start is refused at its call site)" {
    try std.testing.expectEqual(@as(?bool, true), aqVecWanted(null));
    try std.testing.expectEqual(@as(?bool, true), aqVecWanted(""));
    try std.testing.expectEqual(@as(?bool, true), aqVecWanted("1"));
    try std.testing.expectEqual(@as(?bool, false), aqVecWanted("0"));
    for ([_][]const u8{ "2", "on", "off", "true", " 1", "01", "-1", "10" }) |bad| try std.testing.expectEqual(@as(?bool, null), aqVecWanted(bad));
}

test "TF_DS_AQ_VEC: act_quant takes the four-lane twin for bf16 x, K a multiple of 32, 16-byte x with ldx a multiple of 8 and 8-byte q with ldq a multiple of 8 (fake rows: 16-byte, ldf a multiple of 8); anything else keeps the baseline kernel" {
    // the prompt pass's calls: the front and wo_b's input, the experts' pack rows, the pack quant's 32-wide groups
    try std.testing.expect(aqVecFits(.bf16, 0x1000, 5120, 5120, 0x2000, 5120, 0, 0));
    try std.testing.expect(aqVecFits(.bf16, 0x1000, 8192, 8192, 0x2000, 8192, 0, 0));
    try std.testing.expect(aqVecFits(.bf16, 0x1000, 640, 640, 0x2000, 640, 0, 0));
    try std.testing.expect(aqVecFits(.bf16, 0x1000, 32, 32, 0x2008, 32, 0, 0));
    // fake rows (wo_a's input): q is not read, fake's alignment and stride are
    try std.testing.expect(aqVecFits(.bf16, 0x1000, 640, 640, 0x2001, 641, 0x4000, 640));
    // x: another dtype, K not a multiple of 32 (or zero), a row stride off 8 elements, an address off 16 bytes
    try std.testing.expect(!aqVecFits(.f32, 0x1000, 640, 640, 0x2000, 640, 0, 0));
    try std.testing.expect(!aqVecFits(.f16, 0x1000, 640, 640, 0x2000, 640, 0, 0));
    try std.testing.expect(!aqVecFits(.bf16, 0x1000, 640, 100, 0x2000, 640, 0, 0));
    try std.testing.expect(!aqVecFits(.bf16, 0x1000, 640, 0, 0x2000, 640, 0, 0));
    try std.testing.expect(!aqVecFits(.bf16, 0x1000, 644, 640, 0x2000, 640, 0, 0));
    try std.testing.expect(!aqVecFits(.bf16, 0x1008, 640, 640, 0x2000, 640, 0, 0));
    // q: a row stride off 8, an address off 8 bytes
    try std.testing.expect(!aqVecFits(.bf16, 0x1000, 640, 640, 0x2000, 644, 0, 0));
    try std.testing.expect(!aqVecFits(.bf16, 0x1000, 640, 640, 0x2004, 640, 0, 0));
    // fake: a row stride off 8, an address off 16 bytes
    try std.testing.expect(!aqVecFits(.bf16, 0x1000, 640, 640, 0, 0, 0x4000, 644));
    try std.testing.expect(!aqVecFits(.bf16, 0x1000, 640, 640, 0, 0, 0x4008, 640));
}

test "TF_DS_AQ_VEC: quant's and fake's launches go through aqLaunch, the one place that names an act_quant kernel; the twin's grid is 4 threads a group, the baseline a warp" {
    const src = @embedFile("mx.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "try aqLaunch(k, s, &a, x_dtype, x, ldx, m, kdim, q, ldq, 0, " ++ "0);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "try aqLaunch(k, s, &a, x_dtype, x, ldx, m, kdim, 0, 0, out, " ++ "ldo);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "try launch(k.act_" ++ "quant, s, .{ cdiv(m * (kdim / 32) * 32, 256), 1, 1 }, 256, a);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "launch(k.act_quant_" ++ "vec.?, s, .{ cdiv(m * (kdim / 32) * 4, 256), 1, 1 }, 256, a);"));
    // no other launch of either kernel
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "launch(k.act_" ++ "quant,"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "launch(k.act_quant_" ++ "vec"));
}

test "TF_DS_WOA_MERGE: unset, empty and 1 are on, 0 is off, any other text is null (the start is refused at its call site)" {
    try std.testing.expectEqual(@as(?bool, true), woaMergeWanted(null));
    try std.testing.expectEqual(@as(?bool, true), woaMergeWanted(""));
    try std.testing.expectEqual(@as(?bool, true), woaMergeWanted("1"));
    try std.testing.expectEqual(@as(?bool, false), woaMergeWanted("0"));
    for ([_][]const u8{ "2", "on", "true", " 1", "01", "-1", "10" }) |bad| try std.testing.expectEqual(@as(?bool, null), woaMergeWanted(bad));
}

test "TF_DS_WOA_MERGE: the merged launch's group strides are the per-group launches' offsets (wo_a's 2 groups of 1,024 outputs x 4,096 K, and a 4-group shape)" {
    for ([_][2]usize{ .{ 4096, 1024 }, .{ 2048, 512 }, .{ 4096, 128 } }) |sh| {
        const gk = sh[0];
        const gn = sh[1];
        const st = woaGroupStrides(gk, gn);
        try std.testing.expectEqual(@as(i64, @intCast(gk)), st.gq);
        try std.testing.expectEqual(@as(i64, @intCast((gk / 32) * 4)), st.gs);
        try std.testing.expectEqual(@as(i64, @intCast(1 * gn * gk)), st.gw);
        try std.testing.expectEqual(@as(i64, @intCast(1 * (gn / 32) * (gk / 32))), st.gws);
        try std.testing.expectEqual(@as(i64, @intCast(1 * gn * 2)), st.gy);
    }
    const st = woaGroupStrides(4096, 1024);
    try std.testing.expectEqual(@as(i64, 4096), st.gq);
    try std.testing.expectEqual(@as(i64, 512), st.gs);
    try std.testing.expectEqual(@as(i64, 4194304), st.gw);
    try std.testing.expectEqual(@as(i64, 4096), st.gws);
    try std.testing.expectEqual(@as(i64, 2048), st.gy);
}

test "TF_DS_WOA_MERGE: woA8Groups launches one grid.z = groups launch a row block when it is on and a launch a group otherwise; the offsets are the strides' (every needle split: a literal of this file is the file's text)" {
    const src = @embedFile("mx.zig");
    const at = std.mem.indexOf(u8, src, "fn woA8" ++ "Groups(").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n}\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, ".{ cdiv(b.n, 128), gn / 128, if (merged) " ++ "groups else 1 }"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const merged = woa_merge and groups " ++ "> 1;"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "for (0..if (merged) 1 else " ++ "groups) |g|"));
    // no strides on the per-group launches: every one is zero unless merged
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, body, "= if (merged) st."));
    // the kernel side takes them (native.cu's W8Args carries the five strides; mx.zig's W8Args mirrors it field for field)
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "gq: i64 = 0, // TF_DS_WOA_MERGE" ++ ": group z's operands"));
}
