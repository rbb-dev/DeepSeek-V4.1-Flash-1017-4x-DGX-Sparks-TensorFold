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
//!   TF_DS_GRAPHS=1                              the rounds' stretches as CUDA graphs
//!   TF_DS_GRAPH_CAP=N                           (with TF_DS_GRAPHS) the captured stretches kept at most, the least
//!                                               recently replayed destroyed first (0 or unset: all); rank 1's
//!                                               --graph-cap, the same number on both ranks
//!   TF_DS_HC_SIDE=1                             the mixes' side work on a stream of its own
//!   TF_DS_L2_PREFETCH=1                         the paced L2 prefetch of the next kernels' weights
//!   TF_DS_ENGRAM_AIO=1                          a round's Engram reads by Linux AIO on O_DIRECT
//!   TF_DS_FIX_K=K                               a measurement: every drafted stream verifies K drafts a round
//!   TF_DS_SERVED_K=all|solo                     a stream verifies at most the served lane's choice of drafts (its
//!                                               round table and draft ms): at any streams, or one stream alone
//!   TF_FILL_CHUNKS=N                            a prompt pass runs N chunks (of 2,048) at a time, the other streams'
//!   TF_FILL_ROUNDS, TF_FILL_FIRST               rounds between (TF_FILL_ROUNDS of them, 4 unless set); its first
//!                                               piece TF_FILL_FIRST chunks (unset: N), so a prompt that short runs
//!                                               whole. One pass at a time. Unset or 0: each pass whole
//!   TF_FILL_MANY=1                              with TF_FILL_CHUNKS: a request is admitted while a pass fills and
//!                                               its pass runs between the other's pieces (fewest tokens left first)
//!   TF_DS_SOLO_OVERHEAD=1                       a lone drafted stream's head windows cost the fixed draft allowance
//!   TF_DS_NO_COPIES=1                           a drafted stream proposes no suffix-lookup copy windows
//!   TF_DS_PASS_ROWS=N, TF_DS_PASS_FROM=P        a prompt pass's rows at most (a multiple of 2,048 up to the build's
//!                                               pass rows), passes of 2,048 below position P (8192 unless set)
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
//!   TF_DS_PF_RASTER=G                           (G = 2, 4, 8 or 16; other text refuses the start) a prompt GEMM whose
//!                                               activation slab is 14 MiB or more (the tile FP8 GEMMs, the block-scaled
//!                                               MMA's and the bf16 tile GEMM) walks its grid G row tiles at a time
//!                                               across all the column tiles instead of row tiles fastest, so a wave
//!                                               shares a G-tile slab of the activation in the L2 where the default
//!                                               order re-reads all of it per column; only the order of whole output
//!                                               tiles changes, so the replies' bits are unchanged. It does not need
//!                                               TF_DS_GEMM_HALVES and may run with it (each 4,096-row block is a
//!                                               launch of its own); the SSD tier's math names it; every rank the same
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
const round = @import("round.zig");
const link = @import("link.zig");
const rank_cache = @import("rank_cache.zig");
const vision_mod = @import("vision.zig");
const tri_attn = @import("tri_attn.zig");
const tri_hc = @import("tri_hc.zig");
const mx = @import("mx.zig");
const weights = @import("weights.zig");

pub const model_type = "deepseek_v41";
pub const formats: []const []const u8 = &.{"exl3-mul1"};
/// The Python lane's default window; the server's --context moves it up to the model's 1,048,576.
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
/// The served build's prompt chunk (TF_DS_PREFILL_CHUNK).
pub const prompt_rows: u32 = 4096;
/// The served lane's reasoning effort numbers (text.zig): the server passes each name and the template hears its number.
pub const efforts = @import("text.zig").efforts;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false }; // slide: no learner here

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

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
    /// The rows a prompt pass may run while a stream decodes (TF_DS_PASS_BESIDE; the host's step rows). 0: the grid.
    beside: u32 = 0,
    /// The rounds that follow each long prompt step while a stream decodes (TF_DS_ROUNDS_BESIDE; 0: unset).
    rounds: u32 = 0,
    /// The host's order for prompt passes while several prompts fill together (the family's switches, applied by rank
    /// 0's lane host).
    fill: api.fill_order.Policy = .{},
    /// The other streams' rounds between two pieces of a prompt pass (TF_FILL_ROUNDS; 0 while passes run whole).
    fill_rounds: u32 = 0,
};

pub const Snaps = struct {
    ptr: *anyopaque,
    bytes: *const fn (*anyopaque, u32) u64,
    save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque,
    restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void,
    drop: *const fn (*anyopaque, *anyopaque) void,
    /// Whether a kept state restores for a prompt of this many tokens, and whether it still holds its rows.
    resumes: *const fn (*anyopaque, *anyopaque, u32) bool,
    live: *const fn (*anyopaque, *anyopaque) bool,
    tail: u32 = 0,
};

/// The rank's model and the backend over it (what `open` hands the server, freed by `deinit`).
const State = struct {
    gpa: std.mem.Allocator,
    m: *model.Model,
    lanes: lanes_mod.Lanes,
    vision: ?*vision_mod.Vision = null,
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
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
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
    const st = try gpa.create(State);
    errdefer gpa.destroy(st);
    st.gpa = gpa;
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
        .context = o.context,
        .drafts = o.drafts,
        .arena_bytes = arena_gib << 30,
        .round_rows = if (getenv("TF_DS_ROUND_ROWS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadRoundRows else round.default_rows,
        .round_ms = if (getenv("TF_DS_ROUND_MS")) |v| lanes_mod.parseMs(std.heap.page_allocator, v) catch return error.BadRoundMs else null,
        .streams = if (getenv("TF_DS_STREAMS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadStreams else model.max_streams_default,
    });
    errdefer st.m.close();
    st.lanes = lanes_mod.Lanes.init(gpa, st.m);
    errdefer st.lanes.deinit();
    if (getenv("TF_DS_FIX_K")) |t| st.lanes.fix_k = std.fmt.parseInt(usize, t, 10) catch return error.BadFixK;
    if (getenv("TF_DS_SERVED_K")) |t| st.lanes.served_k = if (std.mem.eql(u8, t, "1") or std.mem.eql(u8, t, "all")) .all else if (std.mem.eql(u8, t, "solo")) .solo else .off;
    if (getenv("TF_FILL_CHUNKS")) |t| st.lanes.fill_chunks = std.fmt.parseInt(usize, t, 10) catch return error.BadFill;
    if (getenv("TF_FILL_FIRST")) |t| st.lanes.fill_first = std.fmt.parseInt(usize, t, 10) catch return error.BadFill;
    st.lanes.fill_many = if (getenv("TF_FILL_MANY")) |v| !std.mem.eql(u8, v, "0") else false;
    const fill_rounds: u32 = if (getenv("TF_FILL_ROUNDS")) |t| std.fmt.parseInt(u32, t, 10) catch return error.BadFill else 4;
    st.lanes.solo_overhead = if (getenv("TF_DS_SOLO_OVERHEAD")) |v| !std.mem.eql(u8, v, "0") else false;
    st.lanes.no_copies = if (getenv("TF_DS_NO_COPIES")) |v| !std.mem.eql(u8, v, "0") else false;
    if (getenv("TF_DS_PASS_ROWS")) |t| st.lanes.pass_rows = try passRows(t);
    if (getenv("TF_DS_PASS_FROM")) |t| st.lanes.pass_from = std.fmt.parseInt(usize, t, 10) catch return error.BadPassRows;
    // a vision checkpoint's tower on this rank, from the kit's vision/ (torch's attention cubin, the routing bias)
    st.vision = null;
    if (st.m.cfg.vision) st.vision = try vision_mod.Vision.open(gpa, io, st.m, dir, getenv("TF_DS_KIT").?);
    errdefer if (st.vision) |v| v.close();
    st.lanes.vision = st.vision;
    // the pool is the model's (preallocated): a stream takes no device memory of its own
    const L = lanes_mod.Lanes;
    const snaps: Snaps = .{ .ptr = &st.lanes, .bytes = L.snapBytesFn, .save = L.snapSaveFn, .restore = L.snapRestoreFn, .drop = L.snapDropFn, .resumes = L.snapResumesFn, .live = L.snapLiveFn };
    try st.lanes.idleRdma(); // park even before the first API request
    // a follower that stops stops this server too, with a line that names it, instead of a wait on it for good
    st.m.watchPeers() catch |err| std.log.warn("dsv41: rank 0 cannot watch its followers: {s}", .{@errorName(err)});
    return .{ .backend = st.lanes.backend(), .facts = st.lanes.facts(), .rows = @intCast(st.m.eng.round_rows), .stream_bytes = 0, .ctx = st, .deinit = deinitFn, .snaps = snaps, .lane_vision = if (st.vision) |v| v.hook() else null, .usable_window = @as(u32, @intCast(rows_mod.usableRows(st.m.ctx_cap, st.m.eng.round_rows, model.chunk_rows))), .tier = tier, .grid = model.chunk_rows, .beside = prompt2d.pass_beside_asked, .rounds = prompt2d.rounds_beside_asked, .replay = @intCast(st.m.cfg.window), .spill_state = st.spill_state, .spill_info = st.spill_info, .fill = prompt2d.prompt_order, .quiet = quietFn, .finish = finishFn };
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
        .pf_raster = @intCast(mx.pf_raster),
        .qgate_canary = if (prompt.qgate_canary) 1 else @backingInt(prompt.qgate_body),
    };
}

    const math = tierMath(if (m.two) |*t| t else null, if (m.mxk) |*x| x else null);
    if (world > 1 and !reconciled) abstainBare(st.gpa, st.m);
    if (world > 1) {
    if (world > 1)
        std.log.info("dsv41 spill: the SSD tier is on: a store on every rank under {s} ({d} GiB, staging {d} MiB; a restore is read by each rank from its own disk, no landing); TF_DS_SSD_FAULT {s}", .{ s.dir, s.gib, s.stagingBytes() >> 20, prompt2d.ssd_fault.text(&fault_buf) });
    if (m.world < 2 or m.lead.land_slots == 0) return null;
/// The work this engine ran eagerly after a CUDA graph capture failed on a rank (every rank runs it so, and counts
/// alike; /metrics shows rank 0's); null without graphs.
pub fn graphFallbacks(ctx: *anyopaque) ?*const std.atomic.Value(u64) {
    const st: *State = @ptrCast(@alignCast(ctx));
    return if (st.m.graphs != null) &st.m.graph_fallbacks else null;
}

}

fn deinitFn(ptr: *anyopaque) void {
    const st: *State = @ptrCast(@alignCast(ptr));
    st.lanes.deinit(); // rank 1 hears done
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
        error.BadGraphCap => "TF_DS_GRAPH_CAP: the captured CUDA graphs kept at most, a count (0: all)",
        error.BadStreams => "TF_DS_STREAMS: the pool's streams, 1 .. 32",
        error.BadPassRows => "TF_DS_PASS_ROWS: a multiple of 2,048 from 2,048 to the build's pass rows; TF_DS_PASS_FROM: a position",
        error.BadFill => "TF_FILL_CHUNKS, TF_FILL_FIRST and TF_FILL_ROUNDS: whole numbers (a prompt piece's chunks of 2,048 tokens, the first piece's, the other streams' rounds between two pieces)",
        error.BadWindow => "TF_DS_WINDOW: the streams' shared positions, at least --context",
        error.SettingsDiffer => "a follower rank runs other settings than this server (the line above says which): give every rank the same TF_DS_STREAMS (the followers: --streams), TF_DS_ROUND_ROWS (--round-rows), RDMA devices, drafter, TF_DS_REPLAY_FLOOR and TF_DS_IDX_SPLIT, and every follower a --pool of at least the window (--context, or TF_DS_WINDOW)",
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
    const keep_qb = prompt.qgate_body;
    defer prompt.qgate_body = keep_qb;
    prompt.qgate_body = .off;
    try std.testing.expectEqual(@as(u8, 1), tierMath(null, null).qgate_canary);
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
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoFreeSlot).?, "TF_DS_STREAMS") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "--streams") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "--graph-cap") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_REPLAY_FLOOR") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_PF_SPLIT") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_WOA_W8") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_QUANT_ONCE") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_PF_GEMM5") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_GROUP_TILES") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_TOPK_TILES") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_IDX_PROJ_OWNED") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_BF16_ROWS") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_PF_RUN128") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_DRAFT_4WAY") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_DRAFT_MAIN_4WAY") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.DraftersDiffer).?, "TF_DS_DRAFT_MAIN_4WAY") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_ATTN_U") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_IDX_U") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_IDX_V") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_ATTN_U_DECODE") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_MMA_SF") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_MMA_SF_EXACT") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_GEMM_R2") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_HC_FUSED") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_RING_WORD") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.SettingsDiffer).?, "TF_DS_RING_WORD_SIGNAL") != null);
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

test "passRows takes multiples of the resume grid up to the pass rows" {
    try std.testing.expectEqual(@as(usize, model.resume_rows), try passRows("2048"));
    try std.testing.expectEqual(@as(usize, model.chunk_rows), try passRows("4096"));
    try std.testing.expectError(error.BadPassRows, passRows("0"));
    try std.testing.expectError(error.BadPassRows, passRows("3000"));
    try std.testing.expectError(error.BadPassRows, passRows("8192"));
}
