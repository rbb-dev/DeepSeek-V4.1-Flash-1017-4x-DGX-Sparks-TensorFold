//! DeepSeek-V4.1-Flash behind the native CUDA family interface, tensor parallel over two GPUs: the server's process is
//! rank 0 (the lane core over lanes.zig's Backend on model.zig), rank 1 is `tf-dsv41-lanes` following it on the other
//! node (the same model directory, rank cache, kit, pool and port). The ranks link over TCP and Rocinante; the inputs
//! beyond the checkpoint come from the environment:
//!   TF_TP_WORLD (4, the only world), TF_TP_MASTER, TF_TP_PORT   the ranks: rank 0's fabric address and port (the followers connect)
//!   TF_TP_LISTEN                                world 4: rank 0's address for each follower, "a,b,c" (ranks 1-3: the
//!                                               address each was given; unset: TF_TP_MASTER for all)
//!   TF_DS_RANK_CACHE                            the lane's per-rank weight files (the Python lane's TF_DS_RANK_CACHE)
//!   TF_DS_KIT                                   cubins/ (the served extension cubins), the served RoPE tables
//!                                               (rope-*.f32, as many rows as the context) and engram.json
//!   TF_DS_ENGRAM, TF_DS_TOKEN_MAP               the Engram tables and the compressed token map
//!   TF_DS_ARENA_GIB                             a first block of device memory for the buffers (unset: they take what
//!                                               they need, the pool's caches by the context)
//!   TF_DS_ROUND_MS                              the lane core's round costs by rows (ms, comma separated; the
//!                                               served ROUND_MS unless set: a four-node lane's rounds cost less)
//!   TF_DS_WINDOW                                the streams' shared pool of positions (unset: --context; the
//!                                               four-node lane's 2,097,152 with requests of up to --context)
//!   TF_DS_ROUND_ROWS, TF_DS_STREAMS             a round's rows at most (16; the four-node lane's 48) and the pool's
//!                                               streams (4; its 16): --parallel at most the streams. Every rank
//!                                               the same (the followers: --round-rows, --streams, or these variables):
//!                                               a rank that differs from rank 0 stops at the start, as one with
//!                                               other graphs, graph cap, RDMA devices or drafter does
//!   TF_RDMA_DEVICES                             the decode gathers over the RDMA ring on these devices (the rings are required)
//!   TF_DS_RDMA_COLLECT_PEERS=1                  (with TF_RDMA_DEVICES at world 4) the rings' gather_into copy-out
//!                                               reads every peer's slot at once: a row of blocks a peer, each waiting
//!                                               for its own peer's flag (dsv41_rdma_collect_peers); the same bytes, so
//!                                               each rank's own choice; one line says it is on, or why it is off
//!   TF_DS_RING_WC=1                             (with TF_RDMA_DEVICES) the rings' send and landing data slots from
//!                                               cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocWriteCombined),
//!                                               registered with ibv_reg_mr as in the baseline; the flags, sizes, sent words,
//!                                               sequence and doorbell stay in the baseline memory (the CPU reads those);
//!                                               the same bytes, one line at the start; every rank alike
//!   TF_DS_RING_INPLACE=1                        (with TF_DS_RING_WC) the decode exchanges' producers write their
//!                                               partial straight into a send slot and publish it, so the staging copy
//!                                               is gone; the collecting copy-out stays (its consumers re-read the
//!                                               destination); the same bytes, one line at the start; every rank alike
//!   TF_DS_RING_WORD=1                         (with TF_RDMA_DEVICES) each lane's data is a plain RDMA WRITE and its
//!                                               64-bit generation word is written after it on the same queue pair (an
//!                                               inline 8-byte WRITE, or with TF_DS_RING_WORD_SIGNAL=atomic an RDMA ATOMIC
//!                                               fetch-and-add); the collecting kernel waits on every lane's word,
//!                                               bounded by TF_DS_RING_WORD_DEADLINE_MS (default 120,000) and the abort word,
//!                                               and a ring no sequence reaches is poisoned; the same bytes, one line at
//!                                               the start; every rank alike
//!   TF_DS_DIRECT_WEIGHTS=0                      the weights read through the page cache (default: past it, O_DIRECT)
//!   TF_DS_RDMA_WFE=1                            each RDMA proxy with nothing in flight naps on its doorbell (Arm WFET,
//!                                               at most 50 us a nap) instead of reading it in a loop, on a ring with
//!                                               no relays and a CPU with WFET (one line at the start says which); a
//!                                               proxy logs its naps when its ring parks; the replies' bits as without
//!   TF_DS_RDMA_NAP_US=N                         (with TF_DS_RDMA_WFE) a nap's longest length in us, default 50 (a
//!                                               positive value, at most 1,000,000; start() caps the effective length
//!                                               at 2,500 us, the teardown budget, so the usable range is 1..2,500):
//!                                               the deadline that ends a nap nothing else ends; longer means fewer
//!                                               deadline wakes and a longer tail if the GPU's write is lost (the
//!                                               knob for a measured comparison, not a default)
//!   TF_DS_GRAPHS=1                              the rounds' stretches as CUDA graphs
//!   TF_DS_GRAPH_CAP=N                           (with TF_DS_GRAPHS) the captured stretches kept at most, the least
//!                                               recently replayed destroyed first (0 or unset: all); rank 1's
//!                                               --graph-cap, the same number on both ranks
//!   TF_DS_HC_SIDE=1                             the mixes' side work on a stream of its own
//!   TF_DS_L2_PREFETCH=1                         the paced L2 prefetch of the next kernels' weights
//!   TF_DS_ENGRAM_AIO=1                          a round's Engram reads by Linux AIO on O_DIRECT
//!   TF_DS_ENGRAM_PACKED=DIR                     Engram rows from tf-dsv41-engram-pack's shards in DIR: a record a row,
//!                                               a row cache (TF_DS_ENGRAM_CACHE_MIB, 2048), decoded on the GPU;
//!                                               TF_DS_ENGRAM_INSTREAM=1: a round's landing as a stream host step;
//!                                               TF_DS_ENGRAM_PREFAULT=1: the row cache faulted in at the start (2 MiB
//!                                               pages where the kernel gives them), not by its first rows (the
//!                                               replies' bits as without; one line at the start says how long)
//!   TF_DS_FIX_K=K                               a measurement: every drafted stream verifies K drafts a round
//!   TF_DS_SERVED_K=all|solo                     a stream verifies at most the served lane's choice of drafts (its
//!                                               round table and draft ms): at any streams, or one stream alone
//!   TF_DS_NO_COPIES=1                           decode windows take no copied continuation of the stream's own
//!                                               context (the drafter's drafts alone; the replies' tokens as without);
//!                                               rank 0's setting (it plans every window)
//!   TF_DS_CAND_FIRST=1                          rank 0's round reads (the device's argmax, or a sampled round's first
//!                                               candidates) run before the drafter's absorb on the stream, and the
//!                                               host waits for them alone and draws while the absorb runs: the same
//!                                               kernels on the same inputs, so the same replies; one line says so
//!   TF_DS_COPY_MATCH=N                          a copied continuation needs N matching tokens, 4 .. 64 (unset: 8);
//!                                               rank 0's setting
//!   TF_DS_COPY_ROWS=W                           a lone stream's copied continuation spans up to W rows, 7 .. 17
//!                                               (unset: the drafter's 6): from 8 drafts, doubling while every copied
//!                                               token lands, on rungs of 8, 12 and 16 (the replies' tokens as
//!                                               without); rank 0's setting
//!   TF_DS_PF_SPLIT=A|B|C6|C7|1                  a prompt chunk's split FP8 GEMMs (wq_a, wkv, wq_b, wo_b, the
//!                                               indexer's wq_b) on a restructured split loop: that tile
//!                                               configuration wherever it divides the shape's split, or (1) the
//!                                               per-shape table (mx.split_table); the fold kernel's bits; unset or 0:
//!                                               the fold kernel; with the checkpoint's own weights and the tile
//!                                               kernels (one line at the start says which is missing); every rank the
//!                                               same
//!   TF_DS_WOA_W8=1                              the main layers' wo_a loaded from its FP8 rows alone (the bf16 rows
//!                                               not loaded: about 16.8 MB a layer); a prompt chunk's wo_a converts
//!                                               them to the bf16 rows' values in its kernel (the replies' bits as
//!                                               without); a layer whose FP8 bytes the kernel cannot take keeps its
//!                                               rows (the load's line says how many); with the checkpoint's own
//!                                               weights, the tile kernels and the decode kernels (one line at the
//!                                               start says which is missing); every rank the same
//!   TF_DS_QUANT_ONCE=1                          a prompt chunk's attention front quantizes x once for wq_a and wkv
//!                                               (baseline: once for each) where TF_DS_FRONT_OWNED does not run it on
//!                                               owned rows (the replies' bits as without); with the checkpoint's own
//!                                               weights; every rank the same
//!   TF_DS_ASSEMBLE_QUANT=1                      world 4: a parity chunk's expert intermediate is assembled and
//!                                               down's quant of it computed in one launch (the assembled rows and
//!                                               down's codes and scales, the bytes of the two launches), and down
//!                                               reads the codes without quantizing again; in effect with the
//!                                               checkpoint's own weights and the 2D split's parity experts; every
//!                                               rank the same
//!   TF_DS_PF_GEMM5=1                            a prompt chunk's dense FP8 GEMMs without a K split (Engram's wkv) at 5
//!                                               pipeline stages (tf_dsn_pf_gemm's tile, its bits); with the
//!                                               checkpoint's own weights and the tile kernels; every rank the same
//!   TF_DS_GROUP_TILES=1                         a prompt chunk's experts' grouping builds their tile list in its own
//!                                               launch (the prefix, the active list and the tiles at once), which
//!                                               gate / up and down read instead of building it each (the same lists);
//!                                               with the checkpoint's own weights and the tile kernels; every rank
//!                                               the same
//!   TF_DS_TOPK_TILES=1                          the indexer's pruned top-k of a prompt chunk (rows past 4,096 keys) in
//!                                               one launch a row block: the tiles of the largest maxima, their keys'
//!                                               top k, the indices sorted (the chain's picks); with the checkpoint's
//!                                               own weights and an index_topk of at most 512; every rank the same
//!   TF_DS_IDX_PROJ_OWNED=1                      world 4: the indexer's queries (wq_b, RoPE, fp4_qd) and head weights
//!                                               (weights_proj) of a layer TF_DS_IDX_SHARD shards run on the node's
//!                                               owned rows alone, the rows its scorer reads (the replies' bits as
//!                                               without); in effect with TF_DS_IDX_SHARD=1, the checkpoint's own
//!                                               weights and the owned rows' kernel variants on every rank (one line at
//!                                               the start says which rank cannot); every rank the same
//!   TF_DS_BF16_ROWS=1                           a prompt chunk's plain bf16 GEMMs (the indexer's weights_proj) take a
//!                                               CTA a 16-row tile instead of at most 64 CTAs (each tile's chain as
//!                                               without: the same bits); decode launches as without; with the
//!                                               checkpoint's own weights; every rank the same
//!   TF_DS_PF_RUN128=1                           a prompt chunk's MXFP4 expert tiles (gate / up and down) stage each
//!                                               weight row 128 bytes at a time (two stages' K blocks in one run;
//!                                               without it 64 bytes, one stage's), the same arithmetic in the same
//!                                               order (the replies' bits as without); with the checkpoint's own
//!                                               weights and the tile kernels; every rank the same
//!   TF_DS_DECODE_FUSE=1                         a decode round's routed sum (the slots' outputs summed in expert id
//!                                               order) in the down kernel's launch instead of a launch of its own, and a
//!                                               2D node's assembly of the experts' intermediate writing down's codes
//!                                               (down's quant launch goes), the same arithmetic in the same order (the
//!                                               replies' bits as without); with the checkpoint's own weights and the
//!                                               decode kernels; every rank the same
//!   TF_DS_DRAFT_MAIN_4WAY=1                     a 2D node's drafter main_proj on its quarter of the columns (rows of
//!                                               the rank file's whole entry), an absorb's rows gathered by the
//!                                               backbone's quarters; the drafts' bits as without; with the
//!                                               checkpoint's own weights and the decode kernels; every rank the same
//!   TF_DS_DRAFT_4WAY=1                          a 2D node's drafter stages split four ways like the backbone's layers
//!                                               (their parts of the rank file's TP2 entries; their sublayer gathers the
//!                                               backbone's quarters, no row pair ring), the drafts' bits as without;
//!                                               with the checkpoint's own weights, the decode kernels and the kit's
//!                                               extra set (aot-extra/: the stages' sparse attention at their heads);
//!                                               decided before the weights load, every rank the same
//!   TF_DS_HC_LATE_FORK=1                        a decode mix's side work (the stored streams' dots and the Sinkhorn,
//!                                               on the mixes' side stream) launched after the main stream's next
//!                                               launch, the router or the attention's input projections, so a captured
//!                                               round holds that kernel as the finish's first successor; the same
//!                                               kernels on the same inputs (the replies' bits as without); with the
//!                                               checkpoint's own weights and the side stream; every rank the same
//!   TF_DS_ATTN_EPILOGUE=1                      a prompt chunk's attention output takes the inverse RoPE and wo_a's
//!                                               quant (its codes and scales, or wo_a's quant-dequant rows) in one
//!                                               launch, wo_a then reading those bytes (the same arithmetic and the
//!                                               same bytes; the quant's read of o goes); with the checkpoint's own
//!                                               weights; every rank the same
//!   TF_DS_HC_STRIDE=1                           world 4: a quarter-HC layer's mixes read the gathered planes where
//!                                               they lie in ga / gm, at the chunk's own stride, instead of the tight
//!                                               repack (the same bytes read, no device-to-device copy); in effect
//!                                               with TF_DS_QUARTER_HC=1 and the four-node 2D split; every rank the
//!                                               same
//!   TF_DS_ATTN_U=1                              a prompt chunk's sparse attention with packed latents from the kit's
//!                                               twin _sparse_attn_part_u (aot-extra/: each code byte's two FP4 values
//!                                               unpacked in hardware), the same keys and the same arithmetic (the
//!                                               replies' bits as without); a variant without its twin keeps the baseline
//!                                               kernel; every rank the same
//!   TF_DS_IDX_U=1                               a prompt chunk's indexer scores from the kit's twin
//!                                               _index_score_rows_u (aot-extra/: each packed key code byte's two FP4
//!                                               values unpacked in hardware), the same scores and keys (the replies'
//!                                               bits as without); a variant without its twin keeps the baseline kernel;
//!                                               every rank the same
//!   TF_DS_IDX_V=1                               a prompt chunk's indexer scores from the kit's twin
//!                                               _index_score_rows_v (aot-extra/: the key tile dequantized once a
//!                                               program and each of its rows dotted in turn, at four warps: fewer
//!                                               registers a thread, more warps an SM), the same scores and keys (the
//!                                               replies' bits as without); it replaces TF_DS_IDX_U's twin where both
//!                                               are on, and a variant without its twin falls back to that and then to
//!                                               the baseline kernel; every rank the same
//!   TF_DS_ATTN_U_DECODE=1                       a decode round's sparse attention with packed latents from the kit's
//!                                               twin _sparse_attn_part_u (as TF_DS_ATTN_U's prompt call; the
//!                                               drafter's passes read bf16 keys and are unchanged), the same keys
//!                                               and arithmetic; a variant without its twin keeps the baseline kernel;
//!                                               every rank the same
//!   TF_DS_MMA_SF=1                              a prompt chunk's FP8 GEMMs (the dense ones the tile kernels take and
//!                                               the experts' gate / up and down) on the tensor core's block-scaled
//!                                               MMA (native_sf.cu, built for sm_121a): each K block's sum scaled by
//!                                               its UE8M0 scales in the MMA and a split's K blocks accumulated there:
//!                                               NOT the replies' bits without it; a row's bits at every chunk size;
//!                                               decode rounds as without; with the checkpoint's own weights, the
//!                                               tile kernels and a GPU that loads the kernels, on every rank or none
//!                                               (one line says which and names the GEMMs); every rank the same; the
//!                                               SSD tier's math names it
//!   TF_DS_MMA_SF_EXACT=1                        the same GEMMs on the block-scaled MMA with each K block's MMA from
//!                                               zero, then the baseline fp32 add: the replies' bits without it; its guard
//!                                               keeps on the baseline kernels every GEMM with a weight scale outside the
//!                                               range where that is proven (each scaled block sum a normal float), and
//!                                               one line names the GEMMs and the kept ones; TF_DS_MMA_SF=1 takes the
//!                                               kernels first; with the checkpoint's own weights, the tile kernels and
//!                                               a GPU that loads the kernels, on every rank or none; every rank the
//!                                               same; the SSD tier's math names it
//!   TF_DS_GEMM_W16=1                            a prompt chunk's exact GEMMs (TF_DS_MMA_SF_EXACT's arithmetic: the
//!                                               dense FP8 split GEMMs and the experts' gate / up and down) and wo_a's
//!                                               bf16 rows on 16 warps a block (512 threads, a warp's tile half as
//!                                               wide) instead of 8: the same tiles, stages and shared memory, the
//!                                               same arithmetic in the same order, so the replies' bits without it;
//!                                               needs TF_DS_MMA_SF=1 (exact) for the GEMMs, and runs wo_a's rows
//!                                               either way where the layer has its FP8 rows; every rank the same;
//!                                               the SSD tier's math names it
//!   TF_DS_GEMM_R2=1                             TF_DS_MMA_SF_EXACT's expert GEMMs (gate / up and down) on kernels that
//!                                               leave padding rows' copies out and give down a 128-row tile: the
//!                                               replies' bits without it; needs TF_DS_MMA_SF_EXACT=1 in effect and an
//!                                               image with the kernels; one line says whether it runs; every rank the
//!                                               same; the SSD tier's math names it
//!   TF_DS_DECODE_KSPLIT=1                       a decode round's attention on one exchange a layer instead of two:
//!                                               wo_b takes this node's own wo_a groups' outputs (its K-half of the
//!                                               rank's rows) over every output column, so the column-ring cat before
//!                                               it is gone, and the four nodes' full-width fp32 partials are gathered
//!                                               and added per TP2 rank in pairs. The order of the fp32 additions
//!                                               changes, so the last bits of some values differ from the baseline: a
//!                                               reply is as good, but may take another equally valid path after some
//!                                               tokens (byte-equality tests do not apply; the quality gate does).
//!                                               Needs the 2D split on four nodes, the native FP8 checkpoint and a
//!                                               small file beside the rank's, ksplit{R}of4-<hash>.bin, written with
//!                                               tf-dsv41-rank-cache --ksplit (wo_b's K-half of every main layer, ~0.39
//!                                               GiB a node; the rank file and the prompt path are untouched); refuses
//!                                               to start without it; every rank the same; not in the SSD tier's math
//!   TF_DS_GEMM_HALVES=1                         a prompt pass of more than 4,096 rows runs its dense FP8, bf16 and pf
//!                                               GEMMs as row blocks of at most 4,096, each block the launch a
//!                                               4,096-row pass makes for its rows (the same kernel, split, variant
//!                                               and grid rule); the experts keep the whole pass. A row's bits are the
//!                                               same at every row count, so the replies' bits are unchanged; the SSD
//!                                               tier's math names it; every rank the same
//!   TF_DS_PF_RASTER=G                           (G = 2, 4, 8 or 16; other text refuses the start) a prompt GEMM whose
//!                                               activation slab is 14 MiB or more (the tile FP8 GEMMs, the block-scaled
//!                                               MMA's and the bf16 tile GEMM) walks its grid G row tiles at a time
//!                                               across all the column tiles instead of row tiles fastest, so a wave
//!                                               shares a G-tile slab of the activation in the L2 where the default
//!                                               order re-reads all of it per column; only the order of whole output
//!                                               tiles changes, so the replies' bits are unchanged. It does not need
//!                                               TF_DS_GEMM_HALVES and may run with it (each 4,096-row block is a
//!                                               launch of its own); the SSD tier's math names it; every rank the same
//!   TF_DS_QGATE_CANARY=1                        not a serving setting: the head reads its rows after one more FP8
//!                                               step (act_quant's quant-dequant), every logit perturbed on purpose,
//!                                               the ruler of a quality gate's runs (tf-dsv41-lanes --logits-log);
//!                                               with the checkpoint's own kernels; every rank the same
//!   TF_DS_QGATE_CANARY=2 or 3                   not a serving setting: layer 0's MoE partials, each rank's before
//!                                               their gather, in every prompt chunk and every round, take one more
//!                                               FP8 step (2: act_quant's quant-dequant of each 32-value group) or
//!                                               move one fp32 ulp away from zero (3: a reduction order's size): the
//!                                               body perturbed on purpose, a quality gate's floors; with the
//!                                               checkpoint's own kernels; every rank the same canary (a start whose
//!                                               ranks differ is refused)
//! The kernel set (TENSORFOLD_CUDA_KERNELS) is the recorded Triton set (aot_pack.py). The pool holds --context
//! positions for every stream together (each takes an extent: its prompt, its max_tokens and a round's rows).
//! TF_TP_WORLD=4: the four nodes of the exact 2D split (prompt2d.zig), ranks 1-3 `tf-dsv41-lanes` following on the
//! other nodes; TF_RDMA_DEVICES (and TF_DS_RDMA_KERNELS: rdma_gather.cu's fatbin, unset: the embedded one) put its
//! decode-size exchanges on the RDMA rings, as the four-node Python lane's TF_RDMA_DEVICES does.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const model = @import("model.zig");
const lanes_mod = @import("lanes.zig");
const rows_mod = @import("rows.zig");
const round = @import("round.zig");
const link = @import("link.zig");
const rank_cache = @import("rank_cache.zig");
const vision_mod = @import("vision.zig");
const api = @import("engine_api");
const core = @import("core");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const tri_index = @import("tri_index.zig");
const tri_attn = @import("tri_attn.zig");
const tri_hc = @import("tri_hc.zig");
const mx = @import("mx.zig");
const weights = @import("weights.zig");
const spill_state = @import("spill_state.zig");
const config = @import("config.zig");
const spill_tier = @import("spill_tier.zig");
const spill_sync = @import("spill_sync.zig");
const spill_mirror = @import("spill_mirror.zig");
const spill_start = @import("spill_start.zig");
const spill_open = @import("spill_open.zig");
const spill_identity = @import("spill_identity.zig");

pub const model_type = "deepseek_v41";
pub const formats: []const []const u8 = &.{"exl3-mul1"};
/// The Python lane's default window; the server's --context moves it up to the model's 1,048,576.
pub const default_context: i64 = 262144;

/// The window the checkpoint reaches with the RoPE factor the engine runs, for the generic --context check before the family
/// loads: config.json's max_position_embeddings (`base`), or original_max_position_embeddings x TF_DS_YARN_FACTOR when that
/// asks for another factor (the start's checkRope then refuses a kit cut for a different one).
pub fn nativeWindow(a: std.mem.Allocator, io: std.Io, dir: []const u8, base: i64) i64 {
    const text: ?[]const u8 = if (std.c.getenv("TF_DS_YARN_FACTOR")) |v| std.mem.span(v) else null;
    const asked = (config.yarnFactorAsked(text) catch return base) orelse return base;
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return base;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return base;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return base;
    if (doc != .object) return base;
    // the served checkpoint keeps its text model's settings in text_config (as modelContext reads them), else at the top
    const model_cfg = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const rope = model_cfg.object.get("rope_scaling") orelse doc.object.get("rope_scaling") orelse return base;
    if (rope != .object) return base;
    const orig = rope.object.get("original_max_position_embeddings") orelse return base;
    if (orig != .integer or orig.integer <= 0) return base;
    return windowFor(base, orig.integer, asked);
}

/// `base` unless a factor is asked: then original x factor (the positions YaRN reaches with it).
pub fn windowFor(base: i64, original: i64, asked: ?f64) i64 {
    const f = asked orelse return base;
    return @intFromFloat(@round(f * @as(f64, @floatFromInt(original))));
}

test "the window follows TF_DS_YARN_FACTOR: unset is the checkpoint's own, factor 32 over 65,536 reaches 2,097,152" {
    try std.testing.expectEqual(@as(i64, 1048576), windowFor(1048576, 65536, null));
    try std.testing.expectEqual(@as(i64, 2097152), windowFor(1048576, 65536, 32));
    try std.testing.expectEqual(@as(i64, 1048576), windowFor(1048576, 65536, 16));
}
pub const max_segments: u32 = 1;
/// The served build's prompt chunk (TF_DS_PREFILL_CHUNK).
pub const prompt_rows: u32 = 4096;
/// The served lane's reasoning effort numbers (text.zig): the server passes each name and the template hears its number.
pub const efforts = @import("text.zig").efforts;

pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    /// The SSD tier's settings (engine_api.Spill.fromEnv, parsed once by the server; not asked: off).
    spill: api.Spill = .{},
    slide: bool = false, // no learner here
};

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence.
    stream_bytes: usize,
    /// Streams that allocate nothing (the CUDA host admits them free): none, every stream has its own sequence.
    free_streams: u32 = 0,
    /// Device bytes a trainer keeps out of the streams' room (--slide): none.
    learn_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    /// Kept prompt states for the host's prompt cache (core/prompt_cache Snapshots: `ptr` and its functions).
    snaps: ?Snaps = null,
    /// Images in requests (the server renders their placeholders; the tower runs on this rank).
    lane_vision: ?lanes.stream.LaneVision = null,
    /// The most prompt plus reply tokens a stream holds (rows.usableRows: the window less a decode round's rows): the
    /// server's context window, so a request with no max_tokens gets all of it and one that asks too much is refused there.
    usable_window: ?u32 = null,
    /// The SSD tier (the prompt cache's Tier) when it opened; null otherwise (`spill_state` says why).
    tier: ?api.prompt_cache.Tier = null,
    /// The prompt cache's native rules with the tier: Rules.grid (the prompt chunk) and Rules.window (the replay).
    grid: u32 = 0,
    replay: u32 = 0,
    /// The rows a prompt pass may run while a stream decodes (TF_DS_PASS_BESIDE; the host's step rows). 0: the grid.
    beside: u32 = 0,
    /// The rounds that follow each long prompt step while a stream decodes (TF_DS_ROUNDS_BESIDE; 0: unset).
    rounds: u32 = 0,
    /// The tier's state at the start and its settings with what the family derives (null: not asked).
    spill_state: ?api.SpillState = null,
    spill_info: ?api.SpillInfo = null,
    /// The host's order for prompt passes while several prompts fill together (the family's switches, applied by rank
    /// 0's lane host).
    fill: api.fill_order.Policy = .{},
    /// Whether the model's stream is free of a CUDA graph capture (the lane host's service fence), with `ctx`.
    quiet: ?*const fn (ctx: *anyopaque) bool = null,
    /// The orderly close, once the host's engine thread has ended (no request left, no frame in flight): the followers
    /// hear done and end. It frees nothing; deinit after it does not repeat it (null: deinit alone tells them).
    finish: ?*const fn (*anyopaque) void = null,
    /// The other streams' rounds between two pieces of a prompt pass (TF_FILL_ROUNDS; 0 while passes run whole).
    fill_rounds: u32 = 0,
};

pub const Snaps = struct {
    ptr: *anyopaque,
    bytes: *const fn (*anyopaque, u32) u64,
    save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque,
    restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void,
    drop: *const fn (*anyopaque, *anyopaque) void,
    /// Rows a resumed prompt pass must still run past a kept state (core/prompt_cache Rules.window).
    tail: u32 = 0,
    /// Whether a kept state still restores (Lanes.usable): given with the SSD tier only.
    usable: ?*const fn (*anyopaque, *anyopaque, u32) bool = null,
};

/// The rank's model and the backend over it (what `open` hands the server, freed by `deinit`).
const State = struct {
    gpa: std.mem.Allocator,
    m: *model.Model,
    lanes: lanes_mod.Lanes,
    vision: ?*vision_mod.Vision = null,
    // the SSD tier (rank 0): its pinned staging (made before the pool is sized), a staging slot's events, the store's
    // open (abandoned past TF_DS_SPILL_OPEN_MS: then staging, events and the open's context live until the process
    // ends, the opener's thread closing the store it finally opens)
    staging: ?cuda.HostBuffer = null,
    events: []cuda.Event = &.{},
    abandoned: bool = false,
    spill_state: ?api.SpillState = null,
    spill_info: ?api.SpillInfo = null,
    staging_formula: [200]u8 = undefined,
    off_buf: [160]u8 = undefined,

    /// The tier off for this start, the cause named (/health spill.state).
    fn off(st: *State, comptime fmt: []const u8, args: anytype) void {
        const why = std.fmt.bufPrint(&st.off_buf, fmt, args) catch fmt;
        st.spill_state = api.SpillState.offFor(why);
        std.log.warn("dsv41 spill: the SSD tier is off for this start: {s}", .{why});
    }
};

/// What the store's open (on the opener's thread) needs; it outlives an abandoned opener.
const OpenCtx = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    m: *model.Model,
    layout: spill_state.StateLayout,
    settings: spill_tier.Settings,
    compat: spill_tier.Digest,
    max_tokens: u32,
    staging: []align(4096) u8,
    events: []cuda.Event,
    rank: u32 = 0,

    fn open(ctx: *anyopaque) anyerror!*lanes_mod.SpillTier.S {
        const oc: *OpenCtx = @ptrCast(@alignCast(ctx));
        var gens = try oc.m.spillGenerations(oc.gpa);
        return lanes_mod.SpillTier.S.create(oc.gpa, oc.io, .{ .m = oc.m }, oc.layout, gens, oc.settings, oc.rank, oc.compat, oc.max_tokens, oc.staging, oc.events, .{}) catch |e| {
            gens.deinit(oc.gpa);
            return e;
        };
    }

    fn close(_: *anyopaque, side: *lanes_mod.SpillTier.S) void {
        side.destroy();
    }
};

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// An empty variable as none (a container passes TF_RDMA_DEVICES empty when it is unset).
fn nonEmpty(v: ?[]const u8) ?[]const u8 {
    const s = v orelse return null;
    return if (s.len == 0) null else s;
}

/// The device bytes this rank's weights take, for the host's memory check: its file in TF_DS_RANK_CACHE (the
/// checkpoint holds both ranks'); 0 when there is none to read (open then says what to set).
pub fn weightBytes(io: std.Io, dir: []const u8) u64 {
    _ = dir;
    var d = std.Io.Dir.cwd().openDir(io, getenv("TF_DS_RANK_CACHE") orelse return 0, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var buf: [512]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return 0;
    const name = rank_cache.find(fba.allocator(), io, d, 0, world) catch return 0;
    const st = d.statFile(io, name, .{}) catch return 0;
    return st.size;
}

/// Loads rank 0 and links rank 1 (which must start against TF_TP_MASTER:TF_TP_PORT); the pool holds TF_DS_WINDOW
/// positions (else `o.context`), a stream at most `o.context`. Needs `ctx` current.
/// The engine's own process settings the shared CUDA context reads (host threads and event waits sleep for the GPU, 16 hardware queues):
/// set here, before the context is made, for this family alone.
pub fn applyFamilyDefaults() void {
    api.speed_defaults.applyFamily();
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return error.BadWorld;
    if (world != 2 and world != 4) return error.NotPortedYet; // TP2, or the four-node 2D split
    const master = link.parseIp(getenv("TF_TP_MASTER") orelse return error.NoMaster) catch return error.BadMaster;
    const port = std.fmt.parseInt(u16, getenv("TF_TP_PORT") orelse "29620", 10) catch return error.BadPort;
    const listen: ?[3][4]u8 = if (getenv("TF_TP_LISTEN")) |t| blk: {
        var out: [3][4]u8 = undefined;
        var it = std.mem.splitScalar(u8, t, ',');
        for (&out) |*a| a.* = link.parseIp(it.next() orelse return error.BadMaster) catch return error.BadMaster;
        if (it.next() != null) return error.BadMaster;
        break :blk out;
    } else null;
    const arena_gib = if (getenv("TF_DS_ARENA_GIB")) |t| std.fmt.parseInt(usize, t, 10) catch return error.BadArena else 0;
    const window = if (getenv("TF_DS_WINDOW")) |t| std.fmt.parseInt(usize, t, 10) catch return error.BadWindow else o.context;
    if (window < o.context) return error.BadWindow;
    // the free target (TF_DS_KV_FREE_TARGET_GIB) sizes the pool from the memory the ranks have free; a pool named too is two
    // answers to one question: refused, never one clamped to the other
    const kv_target: u64 = if (getenv("TF_DS_KV_FREE_TARGET_GIB")) |t| api.kv_pool.parseTargetGib(t) catch return error.BadKvTarget else 0;
    const kv_growth_text = getenv("TF_DS_KV_GROWTH_MIB");
    const kv_growth: u64 = if (kv_growth_text) |t| api.kv_pool.parseGrowthMib(t) catch return error.BadKvGrowth else 0;
    if (kv_target != 0 and getenv("TF_DS_WINDOW") != null) return error.KvTargetAndWindow;
    const st = try gpa.create(State);
    errdefer gpa.destroy(st);
    st.* = .{ .gpa = gpa, .m = undefined, .lanes = undefined };
    // the SSD tier's decisions from its settings alone, and its pinned staging out of memory before the pool is sized
    // (the tier never refuses the start: every problem from here is an off reason on /health)
    const pre = spill_start.before(o.spill, world);
    var tier_settings = pre.settings;
    if (pre.asked) {
        st.spill_info = .{ .settings = o.spill };
        st.spill_state = if (pre.off) |why| api.SpillState.offFor(why) else .{ .state = .opening };
    }
    if (tier_settings) |ts| st.staging = cuda.HostBuffer.alloc(ctx.d, @intCast(ts.stagingBytes())) catch |e| blk: {
        tier_settings = null;
        st.off("its pinned staging (TF_DS_SPILL_STAGING_MIB={d}): {s}", .{ ts.staging_mib, @errorName(e) });
        break :blk null;
    };
    errdefer if (st.staging) |*b| b.free();
    if (st.staging) |b| if (@intFromPtr(b.bytes.ptr) % 4096 != 0) {
        st.staging.?.free();
        st.staging = null;
        tier_settings = null;
        st.off("its pinned staging is not 4 KiB aligned", .{});
    };
    // the copy settings need nothing from the model: a bad one refuses the start before the weights load
    const copies = !lanes_mod.noCopies(getenv("TF_DS_NO_COPIES"));
    const copy_match = lanes_mod.copyMatch(getenv("TF_DS_COPY_MATCH")) orelse return error.BadCopyMatch;
    const copy_rows = lanes_mod.copyRows(getenv("TF_DS_COPY_ROWS")) orelse return error.BadCopyRows;
    st.m = try model.Model.open(gpa, io, ctx, .{
        .model_dir = dir,
        .cache_dir = getenv("TF_DS_RANK_CACHE") orelse return error.NoRankCache,
        .kit_dir = getenv("TF_DS_KIT") orelse return error.NoKit,
        .aot_dir = kernels, // null (TENSORFOLD_CUDA_KERNELS=native, or no set beside the binary): the kit's aot/
        .rank = 0,
        .world = world,
        .master = master,
        .listen = listen,
        .port = port,
        .engram_dir = getenv("TF_DS_ENGRAM"),
        .token_map = getenv("TF_DS_TOKEN_MAP"),
        .rdma_devices = nonEmpty(getenv("TF_RDMA_DEVICES")),
        .rdma_kernels = getenv("TF_DS_RDMA_KERNELS"),
        .graphs = if (getenv("TF_DS_GRAPHS")) |v| !std.mem.eql(u8, v, "0") else false,
        .graph_cap = if (getenv("TF_DS_GRAPH_CAP")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadGraphCap else 0,
        .side = if (getenv("TF_DS_HC_SIDE")) |v| !std.mem.eql(u8, v, "0") else false,
        .prefetch = if (getenv("TF_DS_L2_PREFETCH")) |v| !std.mem.eql(u8, v, "0") else false,
        .engram_aio = if (getenv("TF_DS_ENGRAM_AIO")) |v| !std.mem.eql(u8, v, "0") else false,
        .pool = window,
        .kv_target_bytes = kv_target,
        .kv_growth_bytes = kv_growth,
        .kv_growth_set = kv_growth_text != null,
        .kv_vmm = if (getenv("TF_DS_KV_VMM")) |v| !std.mem.eql(u8, v, "0") else false,
        .context = o.context,
        .drafts = o.drafts,
        .arena_bytes = arena_gib << 30,
        .round_rows = if (getenv("TF_DS_ROUND_ROWS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadRoundRows else round.default_rows,
        .round_ms = if (getenv("TF_DS_ROUND_MS")) |v| lanes_mod.parseMs(std.heap.page_allocator, v) catch return error.BadRoundMs else null,
        .streams = if (getenv("TF_DS_STREAMS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadStreams else model.max_streams_default,
        .land_slots = if (tier_settings != null) pre.land_slots else 0,
    });
    errdefer st.m.close();
    st.lanes = lanes_mod.Lanes.init(gpa, st.m);
    errdefer st.lanes.deinit();
    if (getenv("TF_DS_FIX_K")) |t| st.lanes.fix_k = std.fmt.parseInt(usize, t, 10) catch return error.BadFixK;
    if (getenv("TF_DS_SERVED_K")) |t| st.lanes.served_k = if (std.mem.eql(u8, t, "1") or std.mem.eql(u8, t, "all")) .all else if (std.mem.eql(u8, t, "solo")) .solo else .off;
    // copied continuations: rank 0 plans every window (a verify frame carries the window's rows, not its plan), so its
    // own settings are the engine's
    st.lanes.copies = copies;
    st.lanes.copy_match = copy_match;
    st.lanes.copy_rows = copy_rows;
    if (!st.lanes.copies) std.log.info("TF_DS_NO_COPIES=1: decode windows take no copied continuation (the drafter's drafts alone)", .{});
    if (st.lanes.copy_match != lanes_mod.copy_match_default) std.log.info("TF_DS_COPY_MATCH={d}: a copied continuation needs {d} matching tokens ({d} without the setting)", .{ st.lanes.copy_match, st.lanes.copy_match, lanes_mod.copy_match_default });
    if (st.lanes.copy_rows != 0) {
        var line: [192]u8 = undefined;
        std.log.info("{s}", .{lanes_mod.copyRowsLine(&line, st.lanes.copy_rows) catch "TF_DS_COPY_ROWS"});
    }
    // a vision checkpoint's tower on this rank, from the kit's vision/ (torch's attention cubin, the routing bias)
    st.vision = null;
    if (st.m.cfg.vision) st.vision = try vision_mod.Vision.open(gpa, io, st.m, dir, getenv("TF_DS_KIT").?);
    errdefer if (st.vision) |v| v.close();
    st.lanes.vision = st.vision;
    // the pool is the model's (preallocated): a stream takes no device memory of its own
    const L = lanes_mod.Lanes;
    var snaps: Snaps = .{ .ptr = &st.lanes, .bytes = L.snapBytesFn, .save = L.snapSaveFn, .restore = L.snapRestoreFn, .drop = L.snapDropFn, .tail = @intCast(st.m.cfg.window) };
    if (tier_settings) |ts| tierStart(st, io, ctx, dir, ts, o.spill, world);
    errdefer tierStop(st);
    var tier: ?api.prompt_cache.Tier = null;
    if (st.lanes.spill) |t| {
        tier = t.tier();
        snaps.usable = L.usableFn;
    }
    try st.lanes.idleRdma(); // park even before the first API request
    // a follower that stops stops this server too, with a line that names it, instead of a wait on it for good
    st.m.watchPeers() catch |err| std.log.warn("dsv41: rank 0 cannot watch its followers: {s}", .{@errorName(err)});
    return .{ .backend = st.lanes.backend(), .facts = st.lanes.facts(), .rows = @intCast(st.m.eng.round_rows), .stream_bytes = 0, .ctx = st, .deinit = deinitFn, .snaps = snaps, .lane_vision = if (st.vision) |v| v.hook() else null, .usable_window = @as(u32, @intCast(rows_mod.usableRows(st.m.ctx_cap, st.m.eng.round_rows, model.chunk_rows))), .tier = tier, .grid = model.chunk_rows, .beside = prompt2d.pass_beside_asked, .rounds = prompt2d.rounds_beside_asked, .replay = @intCast(st.m.cfg.window), .spill_state = st.spill_state, .spill_info = st.spill_info, .fill = prompt2d.prompt_order, .quiet = quietFn, .finish = finishFn };
}

/// Rank 0's SSD tier: the layout and the staging plan, a staging slot's events, the store's name from every rank's
/// weights and the common inputs, the store opened on its own thread within TF_DS_SPILL_OPEN_MS, then the landing,
/// the start probe and the Leader on the lanes. Any problem turns the tier off, named; it never refuses the start.
/// The math switches in effect for the SSD tier's store name: the 2D node's agreed flags (all off without one), the
/// indexer's score rows, the pair sums' zero, the replay floor and the native kernels' switches in effect (all off
/// without them).
pub fn tierMath(two: ?*const prompt2d.Two, k: ?*const mx.Kernels) spill_start.Math {
    return .{
        // the switches that no longer exist keep the values the shipped settings gave them, so a store written before the NCCL
        // removal keeps its name (the numerics are the same bytes): pair sums off, quarter-HC and the owner gather on
        .pair_sum = false,
        .qhc = two != null,
        .owner_gather = two != null,
        .idx_shard = if (two) |t| t.idx_shard else false,
        .idx_rb = tri_index.idx_rb,
        .negzero = false,
        .replay_floor = prompt.replay_floor,
        .owner_yx = two != null,
        .cat_partner = two != null,
        .front_owned = two != null,
        .ffn_codes = two != null,
        .pack_codes = two != null,
        .engram_owner_sum = two != null,
        .boundary_owned = two != null,
        .cat_codes = two != null,
        .pf_split = mx.PfSplit.name(@backingInt(if (k) |x| x.split else .off)).?,
        .woa_w8 = weights.woa_w8,
        .quant_once = prompt.quant_once,
        .pf_gemm5 = if (k) |x| x.gemm5 else false,
        .group_tiles = if (k) |x| x.group_tiles else false,
        .topk_tiles = if (k) |x| x.topk_tiles else false,
        .idx_proj_owned = if (two) |t| t.idx_proj_owned else false,
        .bf16_rows = if (k) |x| x.bf16_rows else false,
        .pf_run128 = if (k) |x| x.run128 else false,
        .pass_rows = prompt2d.pass_rows_asked,
        .pass_from = prompt2d.pass_from_asked,
        .pass_tail = prompt2d.pass_tail_asked,
        .decode_fuse = if (k) |x| x.decode_fuse else false,
        .mma_sf = if (k) |x| x.sf == .chained else false,
        .mma_sf_exact = if (k) |x| x.sf == .exact else false,
        .gemm_w16 = if (k) |x| x.gemm_w16 else false,
        .gemm_r2 = if (k) |x| x.gemm_r2 else false,
        .attn_u = tri_attn.attn_u,
        .idx_u = tri_index.idx_u,
        .roci = two != null,
        .roci_collect = false,
        .idx_v = tri_index.idx_v,
        .assemble_quant = if (k) |x| x.assemble_quant else false,
        .attn_epilogue = prompt.attn_epilogue,
        .hc_stride = prompt.hc_stride,
        .hc_fused = tri_hc.hc_fused,
        .attn_epi_fused = tri_attn.attn_epi_fused,
        .hc_2s = tri_hc.hc_2s,
        .gemm_halves = mx.gemm_halves,
        .pf_raster = @intCast(mx.pf_raster),
        .qgate_canary = if (prompt.qgate_canary) 1 else @backingInt(prompt.qgate_body),
    };
}

/// The store's name from the common inputs and every rank's weight digest (the start-up exchange's: Model.weights): the
/// same on every rank that started from the same checkpoint, kit and switches. Null name: why not, written into `buf`.
const Named = union(enum) { name: spill_tier.Digest, why: []const u8 };

fn storeNameFor(io: std.Io, ctx: *const cuda.Context, m: *model.Model, dir: []const u8, layout: *const spill_state.StateLayout, world: u32, buf: *[160]u8) Named {
    var texts: spill_start.Texts = .{};
    var model_dir = std.Io.Dir.cwd().openDir(io, dir, .{}) catch |e| return .{ .why = std.fmt.bufPrint(buf, "the checkpoint folder: {s}", .{@errorName(e)}) catch "the checkpoint folder" };
    defer model_dir.close(io);
    var kit_dir = std.Io.Dir.cwd().openDir(io, getenv("TF_DS_KIT") orelse "", .{}) catch |e| return .{ .why = std.fmt.bufPrint(buf, "the kit folder (TF_DS_KIT): {s}", .{@errorName(e)}) catch "the kit folder" };
    defer kit_dir.close(io);
    const math = tierMath(if (m.two) |*t| t else null, if (m.mxk) |*x| x else null);
    const facts: spill_start.Facts = .{
        .model = model_dir,
        .kit = kit_dir,
        .token_map = getenv("TF_DS_TOKEN_MAP"),
        .driver = @intCast(@max(ctx.d.version() catch 0, 0)),
        .capability = ctx.capability() catch 0,
        .layout = layout.digest(),
        .namespace_version = core.spill.format.namespace_version,
        .state_version = spill_state.format_version,
    };
    const common = spill_start.common(io, api.ProcessEnv{}, facts, math, &texts) catch |e| return .{ .why = std.fmt.bufPrint(buf, "the store's name: {s}", .{@errorName(e)}) catch "the store's name" };
    var missing: spill_identity.Missing = .{};
    const name = spill_identity.name(common, m.weights[0..world], &missing) catch {
        var mb: [120]u8 = undefined;
        return .{ .why = std.fmt.bufPrint(buf, "the store's name: {s}", .{missing.text(&mb)}) catch "the store's name" };
    };
    return .{ .name = name };
}

/// Rank 0's tier start. With a store on every rank (TF_DS_SSD_EVERY_RANK) every rank enters one reconcile exactly once, so a
/// start that ends early on rank 0 abstains for it: the followers' reconcile then returns and no store is touched.
fn tierStart(st: *State, io: std.Io, ctx: *const cuda.Context, dir: []const u8, s: spill_tier.Settings, sp: api.Spill, world: u32) void {
    var reconciled = false;
    tierStartInner(st, io, ctx, dir, s, sp, world, &reconciled);
    if (world > 1 and !reconciled) abstainBare(st.gpa, st.m);
}

/// A rank without a usable store takes part in the reconcile by abstaining, once.
fn abstainBare(gpa: std.mem.Allocator, m: *model.Model) void {
    const host = m.comm.host.star orelse return;
    var p = spill_sync.Peer.init(gpa, host, m.rank, spill_sync.nullDisk());
    defer p.deinit();
    _ = spill_sync.abstain(&p) catch {};
}

fn tierStartInner(st: *State, io: std.Io, ctx: *const cuda.Context, dir: []const u8, s: spill_tier.Settings, sp: api.Spill, world: u32, reconciled: *bool) void {
    const gpa = st.gpa;
    const m = st.m;
    const layout = m.spillLayout() catch |e| return st.off("the state layout: {s}", .{@errorName(e)});
    const min = spill_start.stagingMin(&layout, s.smalls, s.restores, &st.staging_formula);
    if (st.spill_info) |*si| {
        si.staging_min_bytes = min.bytes;
        si.staging_formula = min.formula;
    }
    if (world > 1) {
        var pb: [160]u8 = undefined;
        if (spill_start.probeTokens(sp.probe_tokens, model.chunk_rows, rows_mod.usableRows(m.ctx_cap, m.eng.round_rows, model.chunk_rows), &pb)) |why| return st.off("{s}", .{why});
    }
    const plan = spill_tier.stagingPlan(s, &layout) catch return st.off("TF_DS_SPILL_STAGING_MIB={d}: below the {d} MiB this layout needs ({s})", .{ s.staging_mib, (min.bytes + (1 << 20) - 1) >> 20, min.formula });
    st.events = spillEvents(gpa, ctx, spill_tier.eventCount(plan)) catch |e| return st.off("its events: {s}", .{@errorName(e)});
    var nb: [160]u8 = undefined;
    const name = switch (storeNameFor(io, ctx, m, dir, &layout, world, &nb)) {
        .name => |n| n,
        .why => |w| return st.off("{s}", .{w}),
    };
    // the store opened on its own thread; the start waits at most TF_DS_SPILL_OPEN_MS for it
    const oc = gpa.create(OpenCtx) catch |e| return st.off("{s}", .{@errorName(e)});
    m.pool_ceiling = @min(m.pool_ceiling, m.pool_cap); // (the tier's tables are sized by the pool: a live resize may not take it beyond)
    oc.* = .{ .gpa = gpa, .io = io, .m = m, .layout = layout, .settings = s, .compat = spill_tier.storeCompat(name, &layout), .max_tokens = @intCast(@min(m.pool_cap, std.math.maxInt(u32))), .staging = @alignCast(st.staging.?.bytes), .events = st.events };
    const compat = oc.compat;
    const opener = spill_open.Opener(lanes_mod.SpillTier.S).start(gpa, io, oc, OpenCtx.open, OpenCtx.close) catch |e| {
        gpa.destroy(oc);
        return st.off("its opener: {s}", .{@errorName(e)});
    };
    var rb: [120]u8 = undefined;
    const side = switch (opener.wait(@as(u64, sp.open_ms) * std.time.ns_per_ms)) {
        .opened => |x| x,
        .failed => |e| {
            gpa.destroy(oc);
            return st.off("{s}", .{spill_open.reason(.{ .failed = e }, sp.open_ms, &rb)});
        },
        .timeout => {
            st.abandoned = true; // oc, the staging and the events stay: the opener's thread still reads them
            return st.off("{s}", .{spill_open.reason(.timeout, sp.open_ms, &rb)});
        },
    };
    gpa.destroy(oc);
    // the landing's probe (and, without a store on every rank, the landing; followers made their pools from
    // Settings.land_slots), then the start probe: every rank's bytes of one fixed prompt's state compared before the tier
    // serves anything
    var probe_ns: ?u64 = null; // the start probe's time, to /metrics through the Leader once it exists
    var boot: [16]u8 = undefined;
    io.random(&boot);
    if (world > 1) {
        st.lanes.startLanding(@as(usize, sp.land_piece_mib) << 20, sp.probe_rows) catch |e| {
            side.destroy();
            return st.off("the landing: {s}", .{@errorName(e)});
        };
    }
    // the Leader owns the store from here (its end closes it)
    const t = lanes_mod.SpillTier.create(gpa, side, st.lanes.spillFamily(), s, boot) catch |e| {
        side.destroy();
        st.lanes.stopLanding();
        return st.off("its Leader: {s}", .{@errorName(e)});
    };
    if (world > 1) {
        // the four stores made one: an entry survives only if all four hold it, equal (every other entry is deleted)
        const cfg = spill_mirror.cfgDigest(compat, s);
        const star = m.comm.host.star orelse {
            t.destroy();
            st.lanes.stopLanding();
            return st.off("the four stores need the start-up links (TF_DS_SSD_EVERY_RANK)", .{});
        };
        var p0 = spill_sync.Peer.init(gpa, star, 0, t.syncDisk());
        defer p0.deinit();
        p0.cfg = cfg;
        reconciled.* = true;
        const rec = spill_sync.reconcile(&p0) catch |e| {
            t.destroy();
            st.lanes.stopLanding();
            return st.off("the four stores could not be made one: {s}", .{@errorName(e)});
        };
        t.enableSync(star, cfg, &p0) catch |e| {
            t.destroy();
            st.lanes.stopLanding();
            return st.off("the four stores: {s}", .{@errorName(e)});
        };
        t.armFault(prompt2d.ssd_fault.armed(0));
        std.log.info("dsv41 spill: the four stores reconciled: {d} entr{s} held by all four, {d} deleted", .{ rec.kept, if (rec.kept == 1) "y" else "ies", rec.deleted });
    }
    if (world > 1) {
        var seed: u64 = undefined;
        io.random(std.mem.asBytes(&seed));
        var t0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t0);
        const verdict = st.lanes.probeStart(sp.probe_tokens, seed) catch |e| spill_tier.ProbeVerdict{ .failed = e };
        var t1: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t1);
        if (verdict != .busy) probe_ns = @intCast(@max((t1.sec - t0.sec) * std.time.ns_per_s + (t1.nsec - t0.nsec), 0));
        switch (verdict) {
            .same, .busy => {},
            .differs => |rank| {
                t.destroy();
                st.lanes.stopLanding();
                return st.off("the start probe: rank {d}'s bytes differ from rank 0's (TF_DS_SPILL_PROBE_TOKENS={d})", .{ rank, sp.probe_tokens });
            },
            .failed => |e| {
                t.destroy();
                st.lanes.stopLanding();
                return st.off("the start probe could not run: {s}", .{@errorName(e)});
            },
        }
    }
    if (probe_ns) |ns| t.noteProbe(ns);
    st.lanes.spill = t;
    st.spill_state = .{ .state = .on };
    var fault_buf: [48]u8 = undefined;
    if (world > 1)
        std.log.info("dsv41 spill: the SSD tier is on: a store on every rank under {s} ({d} GiB, staging {d} MiB; a restore is read by each rank from its own disk, no landing); TF_DS_SSD_FAULT {s}", .{ s.dir, s.gib, s.stagingBytes() >> 20, prompt2d.ssd_fault.text(&fault_buf) });
}

/// A follower's tier when every rank keeps its own store: its Side opened over its own directory, wrapped in a Mirror, and the
/// ranks' reconcile entered once (rank 0's tierStart enters it too, abstaining on every path that ends early). Null: no
/// tier here, whatever the cause (this rank abstained, or the ranks refused each other); rank 0 is then told to turn it off.
pub fn followerTierFromEnv(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, m: *model.Model, dir: []const u8, env: anytype, buf: []u8) ?*lanes_mod.FollowMirror {
    return followerTier(gpa, io, ctx, m, dir, api.Spill.fromEnv(env, buf));
}

pub fn followerTier(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, m: *model.Model, dir: []const u8, sp: api.Spill) ?*lanes_mod.FollowMirror {
    if (m.world < 2 or m.lead.land_slots == 0) return null;
    const star = m.comm.host.star orelse return null;
    const mirror = openFollowerStore(gpa, io, ctx, m, dir, sp, star) catch |e| {
        std.log.warn("dsv41 spill: rank {d} abstains from the four stores: {s}", .{ m.rank, @errorName(e) });
        var p = spill_sync.Peer.init(gpa, star, m.rank, spill_sync.nullDisk());
        defer p.deinit();
        _ = spill_sync.abstain(&p) catch {};
        return null;
    };
    var fault_buf: [48]u8 = undefined;
    mirror.arm(prompt2d.ssd_fault.armed(m.rank));
    const rec = mirror.reconcile() catch |e| {
        std.log.warn("dsv41 spill: rank {d}: the four stores could not be made one: {s}", .{ m.rank, @errorName(e) });
        _ = mirror.close();
        return null;
    };
    std.log.info("dsv41 spill: rank {d}: a store of its own under {s}: {d} entr{s} held by all four, {d} deleted; TF_DS_SSD_FAULT {s}", .{ m.rank, sp.dir orelse "", rec.kept, if (rec.kept == 1) "y" else "ies", rec.deleted, prompt2d.ssd_fault.text(&fault_buf) });
    return mirror;
}

fn openFollowerStore(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, m: *model.Model, dir: []const u8, sp: api.Spill, star: spill_sync.Star) !*lanes_mod.FollowMirror {
    const pre = spill_start.before(sp, m.world);
    const s = pre.settings orelse return error.NoSpillSettings;
    const layout = try m.spillLayout();
    const plan = try spill_tier.stagingPlan(s, &layout);
    var staging = try cuda.HostBuffer.alloc(ctx.d, @intCast(s.stagingBytes()));
    errdefer staging.free();
    if (@intFromPtr(staging.bytes.ptr) % 4096 != 0) return error.StagingNotAligned;
    const events = try spillEvents(gpa, ctx, spill_tier.eventCount(plan));
    var nb: [160]u8 = undefined;
    const name = switch (storeNameFor(io, ctx, m, dir, &layout, m.world, &nb)) {
        .name => |n| n,
        .why => |w| {
            std.log.warn("dsv41 spill: rank {d}: {s}", .{ m.rank, w });
            return error.NoStoreName;
        },
    };
    const compat = spill_tier.storeCompat(name, &layout);
    const oc = try gpa.create(OpenCtx);
    m.pool_ceiling = @min(m.pool_ceiling, m.pool_cap); // (the tier's tables are sized by the pool: a live resize may not take it beyond)
    oc.* = .{ .gpa = gpa, .io = io, .m = m, .layout = layout, .settings = s, .compat = compat, .max_tokens = @intCast(@min(m.pool_cap, std.math.maxInt(u32))), .staging = @alignCast(staging.bytes), .events = events, .rank = m.rank };
    const opener = try spill_open.Opener(lanes_mod.SpillTier.S).start(gpa, io, oc, OpenCtx.open, OpenCtx.close);
    const side = switch (opener.wait(@as(u64, sp.open_ms) * std.time.ns_per_ms)) {
        .opened => |x| x,
        .failed => |e| {
            gpa.destroy(oc);
            return e;
        },
        .timeout => return error.StoreOpenTimeout, // (the opener's thread still reads oc, the staging and the events: left to it)
    };
    gpa.destroy(oc);
    // (the staging and the events live for the process: the store's close is bounded and may leave threads reading them)
    return lanes_mod.FollowMirror.create(gpa, side, star, m.rank, undefined, spill_mirror.cfgDigest(compat, s));
}

/// The tier's end (deinit, or a start failing after it): the Leader first (the store closed within what is left of
/// the stop's flush bound), the landing's buffers, then the events and the staging, kept when the opener was
/// abandoned or the store's close gave up on a disk call (those threads may read them until the process exits).
/// Nothing here enters a collective: after the orderly close the followers have heard done.
fn tierStop(st: *State) void {
    const clean = if (st.lanes.spill) |t| t.close() else true;
    st.lanes.spill = null;
    st.lanes.stopLanding();
    if (st.abandoned or !clean) return;
    for (st.events) |*e| e.deinit();
    if (st.events.len > 0) st.gpa.free(st.events);
    st.events = &.{};
    if (st.staging) |*b| b.free();
    st.staging = null;
}

/// One reusable event a staging slot (spill_tier.eventCount).
fn spillEvents(gpa: std.mem.Allocator, ctx: *const cuda.Context, n: usize) ![]cuda.Event {
    const evs = try gpa.alloc(cuda.Event, n);
    var made: usize = 0;
    errdefer {
        for (evs[0..made]) |*e| e.deinit();
        gpa.free(evs);
    }
    for (evs) |*e| {
        e.* = try cuda.Event.init(ctx.d, false);
        made += 1;
    }
    return evs;
}

/// Loaded.quiet: no CUDA graph capture in progress on the model's stream (a driver error reads as quiet: the fence
/// then rests on the lane host's own round and chunk rules).
fn quietFn(ctx: *anyopaque) bool {
    const st: *State = @ptrCast(@alignCast(ctx));
    const status = cuda.graph.captureStatus(st.m.stream) catch return true;
    return status == .none;
}

/// Loaded.finish: the followers hear done (their logs end "followed": true); the model and the lanes stay for deinit.
fn finishFn(ptr: *anyopaque) void {
    const st: *State = @ptrCast(@alignCast(ptr));
    // a landing still in flight is aborted while the followers listen (after done no frame reaches them); the lane
    // host's flush has ended the tier's restores by now, so this is the safety net
    if (st.lanes.lander) |*l| l.abort();
    st.lanes.finish();
}

/// The work this engine ran eagerly after a CUDA graph capture failed on a rank (every rank runs it so, and counts
/// alike; /metrics shows rank 0's); null without graphs.
pub fn graphFallbacks(ctx: *anyopaque) ?*const std.atomic.Value(u64) {
    const st: *State = @ptrCast(@alignCast(ctx));
    return if (st.m.graphs != null) &st.m.graph_fallbacks else null;
}

/// Rank 0's decode rounds by the head's path (lanes.HeadPath), read at any time.
pub fn headRounds(ctx: *anyopaque) ?*const [6]std.atomic.Value(u64) {
    const st: *State = @ptrCast(@alignCast(ctx));
    return &st.lanes.head_rounds;
}

/// TF_DS_DRAFT_VOCAB's counters (rank 0's: its guard decides each pass, the followers run it); null unless the list
/// is in effect.
pub fn draftVocab(ctx: *anyopaque) ?*const api.DraftVocabCounters {
    const st: *State = @ptrCast(@alignCast(ctx));
    return if (st.m.dr) |dr| (if (dr.vocab != null) &st.m.vocab_stats else null) else null;
}

fn deinitFn(ptr: *anyopaque) void {
    const st: *State = @ptrCast(@alignCast(ptr));
    // the tier after the prompt cache's store (cuda.zig frees it first). The followers heard done at Loaded.finish when
    // the orderly close ran, so nothing here may talk to them: tierStop sends no frame that matters and enters no
    // collective; without finish (a start that failed), lanes.deinit's done still comes after the tier's last frame
    tierStop(st);
    st.lanes.deinit(); // the followers hear done, unless finish told them
    if (st.vision) |v| v.close();
    st.m.close();
    st.gpa.destroy(st);
}

/// A request or a start this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NotPortedYet => "the native DeepSeek-V4.1 engine runs tensor parallel over two GPUs or the four nodes of the 2D split: set TF_TP_WORLD=2 or 4",
        error.NoMaster, error.BadMaster => "set TF_TP_MASTER to rank 0's fabric address (rank 1: tf-dsv41-lanes MODEL CACHE 1 2 MASTER PORT KIT; ranks 1-3 of four: ... G 4 ...)",
        error.NoRdmaKernels => "TF_RDMA_DEVICES at world 4 needs TF_DS_RDMA_KERNELS (rdma_gather.cu's fatbin) in a build without the kernel images",
        error.NoRankCache => "set TF_DS_RANK_CACHE to the lane's per-rank weight files",
        error.NoKit => "set TF_DS_KIT to the kernel kit (cubins/, the RoPE tables, engram.json)",
        error.NoFreeSlot => "this engine serves TF_DS_STREAMS streams at once (default 4): --parallel at most that",
        error.BadRoundRows => "TF_DS_ROUND_ROWS: a round's rows at most, 16 .. 64",
        error.BadCopyMatch => "TF_DS_COPY_MATCH: the matching tokens a copied continuation needs, 4 .. 64 (unset: 8)",
        error.BadCopyRows => std.fmt.comptimePrint("TF_DS_COPY_ROWS: a lone stream's copied window's rows at most, {d} .. {d} (unset: {d})", .{ lanes_mod.window_rows + 1, lanes_mod.copy_rows_most, lanes_mod.window_rows }),
        error.BadGraphCap => "TF_DS_GRAPH_CAP: the captured CUDA graphs kept at most, a count (0: all)",
        error.BadStreams => "TF_DS_STREAMS: the pool's streams, 1 .. 32",
        error.BadWindow => "TF_DS_WINDOW (the recipe's POOL): the streams' shared positions, at least --context: the pool must hold one full request, the context is never lowered",
        error.SettingsDiffer => "a follower rank runs other settings than this server (the line above says which): give every rank the same TF_DS_STREAMS (the followers: --streams), TF_DS_ROUND_ROWS (--round-rows), RDMA devices, drafter, TF_DS_GRAPHS (--graphs), TF_DS_GRAPH_CAP (--graph-cap), TF_DS_REPLAY_FLOOR, TF_DS_IDX_RB, TF_DS_PF_SPLIT, TF_DS_WOA_W8, TF_DS_QUANT_ONCE, TF_DS_PF_GEMM5, TF_DS_GROUP_TILES, TF_DS_TOPK_TILES, TF_DS_IDX_PROJ_OWNED, TF_DS_BF16_ROWS, TF_DS_PF_RUN128, TF_DS_DECODE_FUSE, TF_DS_DRAFT_4WAY, TF_DS_HC_LATE_FORK, TF_DS_ATTN_U, TF_DS_IDX_U, TF_DS_IDX_V, TF_DS_ATTN_U_DECODE, TF_DS_MMA_SF, TF_DS_MMA_SF_EXACT, TF_DS_GEMM_W16, TF_DS_GEMM_R2, TF_DS_ASSEMBLE_QUANT, TF_DS_ATTN_EPILOGUE, TF_DS_HC_STRIDE, TF_DS_HC_FUSED, TF_DS_RING_WC, TF_DS_RING_INPLACE, TF_DS_RING_WORD, TF_DS_RING_WORD_SIGNAL, TF_DS_DRAFT_MAIN_4WAY, TF_DS_HEAD_CAND, TF_DS_GEMM_HALVES, TF_DS_PF_RASTER, TF_DS_DECODE_KSPLIT, TF_DS_HEAD_KEY and TF_DS_QGATE_CANARY, and every follower a --pool of at least the window (--context, or TF_DS_WINDOW)",
        error.BadKvTarget => "TF_DS_KV_FREE_TARGET_GIB: the free memory to keep on each Spark, a number of GiB above 0 and at most 1024 (unset: the pool is TF_DS_WINDOW's, or the context)",
        error.BadKvGrowth => "TF_DS_KV_GROWTH_MIB: the memory left for what the engine takes after the pool, a whole number of MiB (unset: 6144)",
        error.KvTargetAndWindow => "TF_DS_KV_FREE_TARGET_GIB sizes the pool from the free memory; TF_DS_WINDOW (the recipe's POOL) names a fixed one: set one of them",
        error.NoHostLinks => "the ranks' host links (the star the start's votes and digests go over) are not open: every rank must join rank 0 at TF_TP_MASTER before the weights load",
        error.PoolRefused => "the KV pool does not fit the free memory the ranks have: the line above names the shortfall, the rank and what its start has taken (lower TF_DS_KV_FREE_TARGET_GIB, or free memory), or a rank was given other KV pool settings than rank 0",
        error.ContextFull => "the streams' prompts and budgets fill the context's pool: lower max_tokens, or retry when a stream ends",
        error.PromptTooLong => "the prompt and its max_tokens do not fit the context window",
        error.CanariesDiffer => "TF_DS_QGATE_CANARY: the ranks asked different canaries (the lines above say which): set the same on every rank (a quality gate's runs, never a serving setting)",
        error.DraftersDiffer => "TF_DS_DRAFT_4WAY=1 or TF_DS_DRAFT_MAIN_4WAY=1: the ranks' drafters are not split the same way, or a rank cannot launch the four-way stages (the lines above say which): give every rank the same kit (its aot-extra/), TF_DS_DECODE_DEC, TF_DS_DRAFT_4WAY and TF_DS_DRAFT_MAIN_4WAY",
        error.NoVisionKit => "this checkpoint reads images: its kit (TF_DS_KIT) needs vision/torch_fmha_sm120.cubin and vision/gate_bias_vl.safetensors",
        error.TooManyRingStreams => "TF_DS_RING_INPLACE=1: more than 2 CUDA streams gather on one ring (each stream owns a fixed send slot): run the decode on one stream, or leave the switch off",
        else => null,
    };
}

test "every kernel switch the start-up settings carry and a prompt pass computes with is in the SSD tier's math" {
    @setEvalBranchQuota(20000);
    // a switch the store's name leaves out lets one setting restore the chunk-end states another wrote; the switches
    // of decode rounds and the drafter's passes alone compute no state a prompt pass stores
    const decode_only = [_][]const u8{ "TF_DS_DRAFT_4WAY", "TF_DS_HEAD_PACK12", "TF_DS_HC_LATE_FORK", "TF_DS_ATTN_U_DECODE", "TF_DS_DRAFT_MAIN_4WAY", "TF_DS_HEAD_CAND", "TF_DS_HEAD_KEY", "TF_DS_DECODE_KSPLIT", "TF_DS_RING_WC", "TF_DS_RING_INPLACE", "TF_DS_RING_WORD", "TF_DS_RING_WORD_SIGNAL", "TF_DS_DRAFT_VOCAB" };
    inline for (prompt2d.kern_names ++ prompt2d.kern2_names) |name| {
        if (name.len == 0) continue; // a reserved bit: no switch here
        const field = comptime blk: {
            var b: [name.len - "TF_DS_".len]u8 = undefined;
            for (name["TF_DS_".len..], &b) |c, *o| o.* = std.ascii.toLower(c);
            const f = b;
            break :blk &f;
        };
        const exempt = comptime for (decode_only) |x| {
            if (std.mem.eql(u8, x, name)) break true;
        } else false;
        if (exempt == @hasField(spill_start.Math, field)) {
            std.debug.print("{s}: {s}\n", .{ name, if (exempt) "decode-only, yet in the tier's math" else "not in the tier's math (spill_start.Math, its text, tierMath)" });
            return error.TestUnexpectedResult;
        }
    }
    // TF_DS_ATTN_U: the effective switch
    const keep = tri_attn.attn_u;
    defer tri_attn.attn_u = keep;
    tri_attn.attn_u = true;
    try std.testing.expect(tierMath(null, null).attn_u);
    tri_attn.attn_u = false;
    try std.testing.expect(!tierMath(null, null).attn_u);
    // TF_DS_IDX_U likewise
    const keep_iu = tri_index.idx_u;
    defer tri_index.idx_u = keep_iu;
    tri_index.idx_u = true;
    try std.testing.expect(tierMath(null, null).idx_u);
    tri_index.idx_u = false;
    try std.testing.expect(!tierMath(null, null).idx_u);
    // TF_DS_IDX_V likewise
    const keep_iv = tri_index.idx_v;
    defer tri_index.idx_v = keep_iv;
    tri_index.idx_v = true;
    try std.testing.expect(tierMath(null, null).idx_v);
    tri_index.idx_v = false;
    try std.testing.expect(!tierMath(null, null).idx_v);
    // TF_DS_QGATE_CANARY: the canary in effect, the head's 1, the body's 2 or 3, none 0
    const keep_qc = prompt.qgate_canary;
    defer prompt.qgate_canary = keep_qc;
    const keep_qb = prompt.qgate_body;
    defer prompt.qgate_body = keep_qb;
    prompt.qgate_body = .off;
    prompt.qgate_canary = true;
    try std.testing.expectEqual(@as(u8, 1), tierMath(null, null).qgate_canary);
    prompt.qgate_canary = false;
    try std.testing.expectEqual(@as(u8, 0), tierMath(null, null).qgate_canary);
    prompt.qgate_body = .body_fp8;
    try std.testing.expectEqual(@as(u8, 2), tierMath(null, null).qgate_canary);
    prompt.qgate_body = .body_ulp;
    try std.testing.expectEqual(@as(u8, 3), tierMath(null, null).qgate_canary);
}

test "the tier's math carries TF_DS_QUANT_ONCE in effect" {
    const was = prompt.quant_once;
    defer prompt.quant_once = was;
    prompt.quant_once = false;
    try std.testing.expect(!tierMath(null, null).quant_once);
    prompt.quant_once = true;
    try std.testing.expect(tierMath(null, null).quant_once);
}

test "the tier's math carries TF_DS_WOA_W8 as the load ran with it" {
    const was = weights.woa_w8;
    defer weights.woa_w8 = was;
    weights.woa_w8 = false;
    try std.testing.expect(!tierMath(null, null).woa_w8);
    weights.woa_w8 = true;
    try std.testing.expect(tierMath(null, null).woa_w8);
}

test "the tier's math carries TF_DS_PF_SPLIT in effect: the native kernels' setting, off without them" {
    try std.testing.expectEqualStrings("0", tierMath(null, null).pf_split);
    var k: mx.Kernels = undefined;
    k.gemm5 = false;
    k.group_tiles = false;
    k.topk_tiles = false;
    k.bf16_rows = false;
    k.run128 = false;
    k.decode_fuse = false;
    k.sf = .off;
    for ([_]mx.PfSplit{ .off, .table, .a, .b, .c6, .c7 }, [_][]const u8{ "0", "1", "A", "B", "C6", "C7" }) |sw, text| {
        k.split = sw;
        try std.testing.expectEqualStrings(text, tierMath(null, &k).pf_split);
    }
    // TF_DS_PF_GEMM5 likewise: the kernels' setting, off without them
    try std.testing.expect(!tierMath(null, null).pf_gemm5);
    k.gemm5 = false;
    try std.testing.expect(!tierMath(null, &k).pf_gemm5);
    k.gemm5 = true;
    try std.testing.expect(tierMath(null, &k).pf_gemm5);
    // TF_DS_GROUP_TILES likewise
    try std.testing.expect(!tierMath(null, null).group_tiles and !tierMath(null, &k).group_tiles);
    k.group_tiles = true;
    try std.testing.expect(tierMath(null, &k).group_tiles);
    // TF_DS_TOPK_TILES likewise
    try std.testing.expect(!tierMath(null, null).topk_tiles and !tierMath(null, &k).topk_tiles);
    k.topk_tiles = true;
    try std.testing.expect(tierMath(null, &k).topk_tiles);
    // TF_DS_BF16_ROWS likewise
    try std.testing.expect(!tierMath(null, null).bf16_rows and !tierMath(null, &k).bf16_rows);
    k.bf16_rows = true;
    try std.testing.expect(tierMath(null, &k).bf16_rows);
    // TF_DS_PF_RUN128 likewise
    try std.testing.expect(!tierMath(null, null).pf_run128 and !tierMath(null, &k).pf_run128);
    k.run128 = true;
    try std.testing.expect(tierMath(null, &k).pf_run128);
    // TF_DS_DECODE_FUSE likewise
    try std.testing.expect(!tierMath(null, null).decode_fuse and !tierMath(null, &k).decode_fuse);
    k.decode_fuse = true;
    try std.testing.expect(tierMath(null, &k).decode_fuse);
    // TF_DS_MMA_SF likewise: the arithmetic in effect after the agreement, off without the native kernels
    k.sf = .off;
    try std.testing.expect(!tierMath(null, null).mma_sf and !tierMath(null, &k).mma_sf);
    k.sf = .chained;
    try std.testing.expect(tierMath(null, &k).mma_sf and !tierMath(null, &k).mma_sf_exact);
    // TF_DS_MMA_SF_EXACT: its own field (another kernel, the baseline bits)
    try std.testing.expect(!tierMath(null, null).mma_sf_exact);
    k.sf = .exact;
    try std.testing.expect(tierMath(null, &k).mma_sf_exact and !tierMath(null, &k).mma_sf);
    // TF_DS_GEMM_R2 likewise
    try std.testing.expect(!tierMath(null, null).gemm_r2 and !tierMath(null, &k).gemm_r2);
    k.gemm_r2 = true;
    try std.testing.expect(tierMath(null, &k).gemm_r2);
}

test "refusals name what to set" {
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoMaster).?, "TF_TP_MASTER") != null);
    try std.testing.expectEqualStrings("TF_DS_COPY_MATCH: the matching tokens a copied continuation needs, 4 .. 64 (unset: 8)", explain(null, error.BadCopyMatch).?);
    try std.testing.expectEqualStrings("TF_DS_COPY_ROWS: a lone stream's copied window's rows at most, 7 .. 17 (unset: 6)", explain(null, error.BadCopyRows).?);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoFreeSlot).?, "TF_DS_STREAMS") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "--streams") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "--graph-cap") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_REPLAY_FLOOR") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_IDX_RB") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_PF_SPLIT") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_WOA_W8") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_QUANT_ONCE") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_PF_GEMM5") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_GROUP_TILES") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_TOPK_TILES") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_IDX_PROJ_OWNED") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_BF16_ROWS") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_PF_RUN128") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_DECODE_FUSE") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_DRAFT_4WAY") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_DRAFT_MAIN_4WAY") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.DraftersDiffer).?, "TF_DS_DRAFT_MAIN_4WAY") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_HC_LATE_FORK") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_ATTN_U") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_IDX_U") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_IDX_V") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_ATTN_U_DECODE") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_MMA_SF") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_MMA_SF_EXACT") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_GEMM_R2") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_HEAD_CAND") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_HEAD_KEY") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_DECODE_KSPLIT") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_HC_FUSED") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_RING_WORD") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_RING_WORD_SIGNAL") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_QGATE_CANARY") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.DraftersDiffer).?, "aot-extra") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.CanariesDiffer) orelse "", "TF_DS_QGATE_CANARY") != null);
    try std.testing.expectEqual(@as(?[]const u8, null), explain(null, error.OutOfMemory));
}

test "an engine with graphs counts its work run eagerly after failed captures; one without reports no count" {
    var m: model.Model = undefined;
    m.graphs = null;
    m.graph_fallbacks = .init(0);
    var st: State = undefined;
    st.m = &m;
    try std.testing.expect(graphFallbacks(&st) == null);
    m.graphs = round.Graphs.init(std.testing.allocator, 0);
    defer m.graphs.?.deinit();
    try std.testing.expect(graphFallbacks(&st).? == &m.graph_fallbacks);
}

test "TF_DS_MMA_SF and TF_DS_MMA_SF_EXACT both set: TF_DS_MMA_SF's arithmetic stays, the exact form is off, and the tier's text says so" {
    var k: mx.Kernels = undefined;
    k.split = .off;
    k.gemm5 = false;
    k.group_tiles = false;
    k.topk_tiles = false;
    k.bf16_rows = false;
    k.run128 = false;
    k.decode_fuse = false;
    // TF_DS_MMA_SF took the kernels (useMmaSf), then TF_DS_MMA_SF_EXACT was refused on every rank (its own reason)
    k.sf = .chained;
    k.sf = mx.sfAfterExact(k.sf, false);
    try std.testing.expectEqual(mx.MmaSf.chained, k.sf);
    const tm = tierMath(null, &k);
    try std.testing.expect(tm.mma_sf and !tm.mma_sf_exact);
    var buf: [768]u8 = undefined;
    const text = try spill_start.mathText(&buf, tm);
    try std.testing.expect(std.mem.endsWith(u8, text, " mma_sf=1 mma_sf_exact=0"));
}

test "the engine's decode rounds by the head's path are its lanes' own counters" {
    var st: State = undefined;
    try std.testing.expectEqual(@as(?*const [6]std.atomic.Value(u64), &st.lanes.head_rounds), headRounds(&st));
}

test "the tier's math keeps the values the shipped settings gave the switches that no longer exist, so a store written before keeps its name; all off without a 2D node" {
    const off = tierMath(null, null);
    try std.testing.expect(!off.owner_yx and !off.qhc and !off.owner_gather and !off.pair_sum and !off.cat_partner and !off.front_owned and !off.ffn_codes);
    try std.testing.expect(!off.pack_codes and !off.engram_owner_sum and !off.boundary_owned and !off.cat_codes and !off.roci and !off.negzero);
    var t: prompt2d.Two = undefined;
    t.idx_proj_owned = false;
    t.idx_shard = false;
    const on = tierMath(&t, null);
    try std.testing.expect(on.qhc and on.owner_gather and on.owner_yx and on.cat_partner and on.front_owned and on.ffn_codes);
    try std.testing.expect(on.pack_codes and on.engram_owner_sum and on.boundary_owned and on.cat_codes and on.roci);
    try std.testing.expect(!on.pair_sum and !on.negzero and !on.roci_collect and !on.idx_shard);
    // the name's text, as it was written when these were agreed flags
    var buf: [768]u8 = undefined;
    const text = try spill_start.mathText(&buf, on);
    try std.testing.expect(std.mem.startsWith(u8, text, "pair_sum=0 qhc=1 owner_gather=1 idx_shard=0 idx_rb="));
    try std.testing.expect(std.mem.indexOf(u8, text, " negzero=0 replay_floor=") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, " owner_yx=1 cat_partner=1 front_owned=1 ffn_codes=1 pack_codes=1 engram_owner_sum=1 boundary_owned=1 cat_codes=1 pf_split=") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, " roci=1") != null and std.mem.indexOf(u8, text, "roci_collect") == null);
}

test "every start error of the pool, the follower settings and the engine.cfg has its sentence, and the window and the free target are told apart" {
    inline for (.{ error.BadKvTarget, error.BadKvGrowth, error.KvTargetAndWindow, error.PoolRefused, error.NoHostLinks }) |e| {
        const words = explain(null, e) orelse return error.TestUnexpectedResult;
        try std.testing.expect(words.len > 20);
    }
    const both = explain(null, error.KvTargetAndWindow).?;
    try std.testing.expect(std.mem.indexOf(u8, both, "TF_DS_KV_FREE_TARGET_GIB sizes the pool from the free memory") != null);
    try std.testing.expect(std.mem.indexOf(u8, both, "TF_DS_WINDOW (the recipe's POOL) names a fixed one: set one of them") != null);
}
