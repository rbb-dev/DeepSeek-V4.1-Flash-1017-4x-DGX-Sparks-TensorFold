//! The engine a server drives (``Engine``), and ``LaneHost``: the lane core served on one thread, rounds left to it.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// The server's name for one request, unique while the process lives.
pub const tfvideo = @import("tfvideo.zig");
/// ``cache_salt``: a salted request's cache key.
pub const cache_salt = @import("cache_salt.zig");

pub const Id = u64;

/// One request's sampling (Python's exact_sampling.Sampling); a request without one decodes greedily.
pub const Sampling = lanes.Sampling;

/// One image of a request (``Request.images``): the tower's input and the placeholder rows its output replaces.
pub const Image = lanes.stream.Image;

/// A prompt's reproducible seed (``seed_for``), for requests that name none.
pub const seedFor = lanes.sampling.seedFor;

/// A committed token's log probability under the target's raw distribution, with the best tokens at its position.
pub const LogprobRow = lanes.LogprobRow;

/// The request's stop strings, checked on the engine's thread after each committed token (the server decodes the tail).
pub const Stop = struct {
    ctx: *anyopaque,
    check: *const fn (ctx: *anyopaque, emitted: []const u32) bool,
};

/// A prompt's image for a lane-vision family (Info.lane_vision): its span and the family's payload
/// (lanes.stream.LaneImage).
pub const LaneImage = lanes.stream.LaneImage;

/// A lane-vision family's image input (lanes.stream.LaneVision): the server renders each image part as the placeholder
/// and the family's prepare sizes its span.
pub const LaneVision = lanes.stream.LaneVision;

/// tool_choice "required" or a named function: outside a think block the answer opens a call to an offered tool.
pub const CallGate = struct {
    /// The token that opens a call, the template text before the name, and the mark that ends the name.
    opener: u32,
    lead: []const u8 = "",
    tail: []const u8 = "",
    /// Offered names (empty: any); the gate completes a name its prefix starts.
    names: []const []const u8 = &.{},
    think_open: ?u32 = null,
    think_end: ?u32 = null,
};

/// response_format and the guided_* fields: every token keeps the reply inside this grammar.
pub const Structure = struct {
    kind: enum { json, json_schema, regex, choice, grammar, structural_tag },
    /// The schema (JSON text), regex, choices (a JSON array), EBNF or structural tag; empty for any JSON object.
    text: []const u8 = "",
    /// With thinking on, the grammar starts after this token.
    after: ?u32 = null,
    /// The server's compiler, which has compiled `text` once already (the engine's compile is then a cache hit).
    compiler: ?*grammar.Compiler = null,
    /// Tokens the reply never draws while the grammar holds it, unless a row allows nothing else (a forced DSML call:
    /// the marker spelled in plain characters, so that every marker is its own token).
    ban: []const u32 = &.{},
};

pub const grammar = lanes.grammar;

/// A reply to decode. The request and every slice in it stay valid until its ``finished`` event.
pub const Request = struct {
    prompt: []const u32,
    /// A lane-vision family's images in order, each span inside the prompt (its positions hold the image token).
    lane_images: []const LaneImage = &.{},
    /// With `cache_key` (below): each lane image's identity for the durable tier (its span and the full SHA-256 its cache ids
    /// derive from, prompt_cache.pictureId), in prompt order; empty: none known (the tier then keeps and finds nothing with them).
    cache_media: []const prompt_cache.Media = &.{},
    max_tokens: u32,
    sampling: ?Sampling = null,
    /// Tokens that end the reply as ``stop`` (the token is delivered); empty when the request ignores EOS.
    eos: []const u32 = &.{},
    stop: ?Stop = null,
    /// false: one token a round and no drafts, the serial reference drafted replies must equal.
    drafts: bool = true,
    /// Background work yields to foreground arrivals.
    background: bool = false,
    /// The prompt length its kept states count as (0: its own); a pass prefilling a reply carries that request's.
    born: u32 = 0,
    /// Prompt prefix lengths worth keeping for a later turn: the rendered history, then shared system blocks.
    history_len: u32 = 0,
    shared_prefixes: []const u32 = &.{},
    /// Where the latest user message's text starts: a state kept before it lets an edit of that turn resume (0: none).
    rewind_len: u32 = 0,
    /// Where the prompt's prefill chunks start after 0 (Python's PrefillPlan at ``Info.prefill_step``); empty: the engine's own.
    chunks: []const u32 = &.{},
    /// Decoded intervals [start,end), sorted, nonempty and strictly separated; an empty list uses prompt arithmetic.
    decode_spans: []const [2]u32 = &.{},
    /// Most reply tokens inside a think block before the engine writes ``think_close`` (0: no limit).
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: ?u32 = null,
    /// The offered tools as JSON text, for drafters that propose a call's structure; empty: no tools.
    tools_json: []const u8 = "",
    /// Set only when ``Info`` says the engine enforces it.
    call: ?CallGate = null,
    structure: ?Structure = null,
    /// Stop a short exact cycle while the think block is open.
    loop_guard: bool = false,
    /// With ``loop_guard``, also close the think block after this many 1024-token windows in a row with under 2% new
    /// token 8-grams (a templated loop that no exact cycle shows); 0: exact cycles only.
    loop_windows: u32 = 0,
    /// The prompt row (the assistant header's last) where routed lessons decide whether they act (null: none acts).
    decide: ?u32 = null,
    /// The model as shipped: no learned neuron acts on its rows, no prompt state is kept or reused (a lesson's writer).
    untaught: bool = false,
    /// Logprob rows for each reply token with this many best tokens (0..20), when ``Info.logprobs``; null: none.
    logprobs: ?u8 = null,
    /// Rows for each prompt token after the first, with ranks and this many best tokens, when ``Info.prompt_logprobs``.
    prompt_logprobs: ?u8 = null,
    /// Images whose rows replace placeholder tokens of ``prompt``, in order; only when ``Info.vision`` is set.
    images: []const Image = &.{},
    /// The prompt as the prompt cache matches it (each image's placeholders by its content); empty: ``prompt``.
    cache_key: []const u32 = &.{},

    /// The prompt as the prompt cache matches it: an image request's placeholders stand for their images.
    pub fn cacheKey(r: *const Request) []const u32 {
        return if (r.cache_key.len > 0) r.cache_key else r.prompt;
    }

    /// The prompt length its kept states count as: its own, or for a warm pass the request whose reply it prefills.
    pub fn bornLen(r: *const Request) u32 {
        return if (r.born > 0) r.born else @intCast(r.prompt.len);
    }
};

/// An image prepared for the engine's tower (``Vision.prepare``): patches, patch grid, tokens and a content hash.
pub const PreparedImage = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

/// Image input an engine serves (``Info.vision``): image bytes to tower input, and what a request may hold.
pub const Vision = struct {
    ctx: *anyopaque,
    /// The image's bytes (any format the platform decodes) at most ``max_tokens`` tokens; allocated in ``a``.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!PreparedImage,
    /// The token the chat template writes once per image, expanded to the image's tokens.
    image_token: u32,
    /// The tokens the template wraps an image in (the image token among them): text that spells one stays text.
    markers: []const u32 = &.{},
    /// Tokens the server writes around each image's rows (Nemotron Omni's <img> and </img>); null: the template's own.
    wrap: ?[2]u32 = null,
    /// Image tokens one request may hold in all, and images.
    image_tokens: u32,
    max_images: u32,
    /// Video input, when the engine's family reads it.
    video: ?VideoOffer = null,
    /// The engine's words for a prepare error the server cannot word (a decoder library it lacks); null: the server's.
    explain: ?*const fn (err: anyerror) ?[]const u8 = null,
};

/// A video's facts as the server's decoder reads them, and its frames on request.
pub const VideoSource = struct {
    frames: u64, // the stream's frame count
    rate_num: u64, // its average frame rate, num / den
    rate_den: u64,
    width: u32,
    height: u32,
    ctx: *anyopaque,
    /// The frames at ``indices`` (ascending, distinct), each to ``sink`` in order.
    decode: *const fn (ctx: *anyopaque, indices: []const u32, sink: FrameSink) anyerror!void,
};

/// Where a decoded frame goes: RGB rows of ``width`` * 3 bytes, ``stride`` bytes apart.
pub const FrameSink = struct {
    ctx: *anyopaque,
    take: *const fn (ctx: *anyopaque, k: usize, rgb: []const u8, width: u32, height: u32, stride: usize) anyerror!void,
};

/// A video prepared for the tower: each group of frames an image, and each group's time in seconds.
pub const PreparedVideo = struct { groups: []PreparedImage, times: []f64 };

/// Video input an engine serves (``Vision.video``): how a decoded video becomes images and how the prompt holds them.
pub const VideoOffer = struct {
    /// ``src`` at most ``max_tokens`` tokens and ``max_frames`` sampled frames; allocated in ``a``.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, src: VideoSource, max_tokens: u32, max_frames: u32) anyerror!PreparedVideo,
    /// The token the chat template writes once per video, expanded to its groups.
    video_token: u32,
    /// The tokens around each group's image tokens; the group's time follows as text.
    group_open: u32,
    group_close: u32,
    /// The tokens the template wraps a video in (the video token among them): text that spells one stays text.
    markers: []const u32 = &.{},
    /// Video tokens one request may hold in all, videos, and frames sampled a video.
    video_tokens: u32,
    max_videos: u32,
    max_frames: u32,
};

/// --vision: image and video input's limits (``Open.vision``); image_tokens_set: the flag given, over a family default.
pub const VisionOpen = struct { image_tokens: u32 = 4096, image_tokens_set: bool = false, max_images: u32 = 4, video_tokens: u32 = 16384, max_videos: u32 = 2, max_frames: u32 = 256 };

pub const Reason = enum { stop, length, cancelled, failed };

/// Depths the per-depth draft counts keep (the lane engine's deepest head chain); a deeper draft counts at the last.
pub const draft_depths = 16;

/// One depth of a draft head's chains: the rounds whose chain drafted that deep, those that reached the depth (kept
/// every draft above it), and those that kept its draft too. kept / drafted is DeepInfer's per-position accept rate
/// (deepinfer/engine/stats_logger.py:211-236); kept / reached, the rate the depth rule learns (lanes/depth.zig).
pub const DraftDepth = struct { drafted: u32 = 0, reached: u32 = 0, kept: u32 = 0 };

/// What a finished reply's rounds did, for its runtime fields and /metrics.
/// A reply's decode rounds by the plan its window ran: copied continuations and the head's drafts, each the rounds, the
/// drafts verified and those kept (lanes.stream.Plans).
pub const Plans = lanes.stream.Plans;

/// A drafter's pruned vocabulary (DeepSeek's TF_DS_DRAFT_VOCAB), as the family counts it: its passes on the list and
/// on the whole head, the committed tokens its guard counted and how many of them were outside the list, and whether
/// the passes take the whole head. The family writes them; the server reads them at any time.
pub const DraftVocabCounters = struct {
    list_passes: std.atomic.Value(u64) = .init(0),
    full_passes: std.atomic.Value(u64) = .init(0),
    tokens: std.atomic.Value(u64) = .init(0),
    misses: std.atomic.Value(u64) = .init(0),
    full: std.atomic.Value(bool) = .init(false),

    pub fn read(c: *const DraftVocabCounters) DraftVocab {
        return .{ .list_passes = c.list_passes.load(.monotonic), .full_passes = c.full_passes.load(.monotonic), .tokens = c.tokens.load(.monotonic), .misses = c.misses.load(.monotonic), .full = c.full.load(.monotonic) };
    }
};

/// A read of DraftVocabCounters (Status.draft_vocab).
pub const DraftVocab = struct { list_passes: u64 = 0, full_passes: u64 = 0, tokens: u64 = 0, misses: u64 = 0, full: bool = false };

pub const Stats = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    plans: Plans = .{},
    /// The draft head's chains by depth (index 0: the first draft); all zero where the engine does not count them.
    depths: [draft_depths]DraftDepth = @splat(.{}),
    /// The narrowest verify window of the reply's rounds (drafted replies: 2 or more).
    min_rows: u32 = 0,
    prefill_widths: []const u32 = &.{},
    prefill_raised: []const bool = &.{},
    /// The loop period that ended or interrupted the think block, if any.
    loop_period: ?u32 = null,
    /// The dry window's share of new token 8-grams, when the loop guard's novelty signal closed the think block.
    loop_novelty: ?f64 = null,
    /// Prompt prefill duration measured by the engine host, in seconds.
    prefill_seconds: ?f64 = null,
    /// The drafter's own counters as a JSON object, or empty.
    telemetry_json: []const u8 = "",
};

pub const Event = union(enum) {
    /// The prompt is in the cache: ``cached`` of its tokens came from a kept prefix.
    prefilled: u32,
    /// The next ``tokens`` event's logprob rows, one a token (``logprobs`` requests); valid only during the call.
    logprobs: []const LogprobRow,
    /// The prompt's rows, one a prompt token after the first, right after ``prefilled`` (``prompt_logprobs`` requests).
    prompt_logprobs: []const LogprobRow,
    /// Tokens committed for this request, in order; valid only during the call.
    tokens: []const u32,
    /// The reply ended; nothing follows. ``message`` says why a failed reply failed.
    finished: struct { reason: Reason, stats: Stats = .{}, message: []const u8 = "" },
};

/// Where a request's events go. Called on the engine's thread: it must return at once and not re-enter the engine.
pub const Sink = struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, id: Id, event: *const Event) void,
};

/// A reasoning effort name and the number a family's chat template takes for it (DeepSeek-V4.1's: "medium" -> 62).
pub const Effort = struct { name: []const u8, value: i64 };

/// The order prompt steps are served in while several prompts fill together (fill_order.zig).
pub const fill_order = @import("fill_order.zig");

/// What the engine serves, fixed once it has loaded.
pub const Info = struct {
    name: []const u8 = "lanes",
    lanes: u32 = 1,
    /// Prompt plus reply tokens one request may use (0: no limit).
    context_window: u32 = 0,
    context_fitted: bool = false,
    /// What the engine enforces; the server refuses requests that need more.
    call_gates: bool = false,
    /// The engine observes loop_guard requests in its lane stream.
    loop_guard: bool = false,
    /// The engine enforces a token budget while the think block is open.
    thinking_budget: bool = true,
    structures: bool = false,
    /// Requests may ask for ``logprobs`` (the engine's lanes give the target's rows).
    logprobs: bool = false,
    /// Requests may ask for ``prompt_logprobs`` (a prompt pass from 0 that reads every row).
    prompt_logprobs: bool = false,
    /// Prompt rows a prefill chunk at most, for the server's chunk starts (0: the engine cuts prompts itself).
    prefill_step: u32 = 0,
    /// The family's resume grid (its prompt chunk): the rows a prompt step runs between two rounds at the default
    /// cadence, the host's step rows while a stream decodes (lanes.stream.Facts). 0: the host plans none.
    resume_rows: u32 = 0,
    /// The rows a prompt step may run between two rounds while a stream decodes (the host's step rows): a multiple of
    /// the resume rows, no more than the family's pass rows. 0: the resume rows.
    beside_rows: u32 = 0,
    /// TF_DS_ROUNDS_BESIDE: the rounds that follow each long prompt step while a stream decodes (0: unset; the share's rule).
    rounds_beside: u32 = 0,
    /// The order prompt steps are served in while several prompts fill together (off: one step each in turn).
    fill: fill_order.Policy = .{},
    /// The live streams' rounds the lane host runs between two steps of a prompt pass (0: a step every turn of its
    /// loop, a round between); none waits while no stream decodes.
    fill_rounds: u32 = 0,
    /// The engine keeps prompt states between requests, so ``prefilled`` counts the tokens it restored.
    prompt_cache: bool = false,
    /// The policy the engine runs under, one line (`/health` shows it); empty: none.
    policy: []const u8 = "",
    /// A line the server prints once at startup (the engine's memory plan); empty: none.
    startup: []const u8 = "",
    /// The reasoning efforts the family's template takes as numbers: each name passes as asked and the template hears
    /// its number. Empty: the names the template quotes, as text (the nearest one for any other name).
    efforts: []const Effort = &.{},
    /// A lane-vision family's image input (LaneVision: each image part its placeholder, the family's prepare its span);
    /// null: none (upstream's image input is `vision`).
    lane_vision: ?LaneVision = null,
    /// A prompt-only request (max_tokens 0) keeps its end state: the server prefills each reply for the next turn.
    warm_turns: bool = false,
    /// The engine decodes plain whatever a request asks: no drafter loaded, or this chip is faster plain.
    plain_only: bool = false,
    /// The immutable retained-prefix plan applied at startup, not a process memory limit or current cache occupancy.
    prompt_cache_plan: ?PromptCachePlan = null,
    /// The engine scores decisions (`score`) and writes no text: the server refuses text requests.
    decides: bool = false,
    /// Decisions may carry pictures (`Decision.pictures`).
    pictures: bool = false,
    /// Image input (--vision); null: text only.
    vision: ?Vision = null,
    /// The engine decodes greedily only: the server refuses a request with a temperature before admitting it.
    greedy_only: bool = false,
    /// Whether a request's caches fit beside the running streams' now (null: a free lane does); else it waits in order.
    admit: ?Admit = null,
    /// The prompt cache's disk tier as the engine was opened with it (null: the backend has no tier to bind).
    spill: ?SpillInfo = null,
};

pub const Admit = struct {
    ctx: *anyopaque,
    fits: *const fn (ctx: *anyopaque, prompt: u32, max_tokens: u32) bool,
};

pub const PromptCachePlan = struct {
    source: enum { physical_footprint, explicit, metal_working_set },
    budget_bytes: u64,
    explicit_budget: bool = false,
    over_cap: ?bool = null,
    ram_bytes: ?u64 = null,
    ready_footprint_bytes: ?u64 = null,
    cap_bytes: ?u64 = null,
    room_bytes: ?u64 = null,
    margin_bytes: ?u64 = null,
};

/// The disk tier's settings as an engine took them, with what the family derives from them (Info.spill).
pub const SpillInfo = struct {
    settings: Spill,
    /// The pinned staging the state layout needs at these smalls and restores (null: not known), with its formula.
    staging_min_bytes: ?u64 = null,
    staging_formula: []const u8 = "",
};

/// The disk tier's state (Status.spill): opening while the store's open runs, on, or off with its cause (a bad
/// setting, the open's error or timeout, a failed probe, TF_DS_SPILL_OFF_REASON). The reason is copied into the
/// struct, so a status read on another thread never points into the engine's memory.
pub const SpillState = struct {
    state: State = .opening,
    reason_len: u8 = 0,
    reason_buf: [120]u8 = undefined,

    pub const State = enum { opening, on, off };

    /// Off for `why` (cut to 120 bytes).
    pub fn offFor(why: []const u8) SpillState {
        var s: SpillState = .{ .state = .off };
        const n = @min(why.len, s.reason_buf.len);
        @memcpy(s.reason_buf[0..n], why[0..n]);
        s.reason_len = @intCast(n);
        return s;
    }

    pub fn reason(s: *const SpillState) []const u8 {
        return if (s.state == .off) s.reason_buf[0..s.reason_len] else "";
    }
};

/// A checkpoint family an engine reads: its config ``model_type`` and weight formats, as gate entries name them.
pub const Family = struct { model_type: []const u8, formats: []const []const u8 };

/// The math an NVFP4 checkpoint runs: its own (FP4 rows into NVFP4 layers, FP8 into FP8 ones), or bf16 rows throughout.
pub const Precision = enum { checkpoint, full };

/// What a server asks of the engine it opens: the checkpoint, and the serve flags an engine reads.
pub const Open = struct {
    dir: []const u8,
    model_type: []const u8,
    context: ?i64 = null,
    lanes: u32 = 8,
    /// --parallel named a number: an engine whose memory fits fewer streams refuses instead of serving fewer.
    lanes_fixed: bool = false,
    drafts: bool = true,
    /// --drafter: a draft model's directory for families that load one (Qwen3.8-27B's DFlash2); null: none.
    drafter: ?[]const u8 = null,
    /// --drafter-bits: 4 prepares the drafter's weights in 4 bits; 0 keeps them bf16.
    drafter_bits: u8 = 4,
    /// --record: each stream's committed rows and tapped features written under this folder, for drafter training.
    record: ?[]const u8 = null,
    speed_up: ?[]const u8 = null,
    prompt_cache_gib: ?f64 = null,
    prompt_cache_over_cap: bool = false,
    /// --learn: where shared prompt states are kept on disk for later sessions and servers (null: off).
    learn: ?[]const u8 = null,
    /// --learn-min-free-gib: filesystem space kept free before any learned-state write.
    learn_min_free_gib: f64 = 4,
    /// --learn-gib: what learned states may take on disk, every model and build together.
    learn_gib: f64 = 32,
    /// --learn-conversations: every kept state written behind to disk too, not only shared cuts (under `learn`).
    learn_conversations: bool = false,
    /// --learn-daily-gib: what learned states may write a day under the root, every model together.
    learn_daily_gib: f64 = 32,
    /// --slide: Sliding Weights learns into the served weights, live (off: learn requests are refused).
    slide: bool = false,
    /// --recall: the server attaches Recall's record once the engine loads; an engine that reads none refuses first.
    recall: bool = false,
    /// --device and --segments (CUDA); null: the backend's environment fallback, then its default.
    device: ?u32 = null,
    segments: ?u32 = null,
    /// --precision (CUDA): an NVFP4 checkpoint's math, NVIDIA's own or full precision; null: the family's default.
    precision: ?Precision = null,
    /// --mtp-drafts and --mtp-confidence; null: the engine's defaults.
    mtp_drafts: ?u32 = null,
    mtp_confidence: ?f64 = null,
    /// Kept prompt entries (--checkpoint-slots); null: the engine's plan.
    keep: ?i64 = null,
    /// Tensor parallelism: this process is `rank` of `tp`; rank 0 listens on `master`:`master_port` for the others.
    tp: u32 = 1,
    rank: u32 = 0,
    master: []const u8 = "",
    master_port: u16 = 29551,
    /// --policy: the engine's policy as `key=value,...`; empty: its defaults.
    policy: []const u8 = "",
    /// --vision: load the checkpoint's image tower (null: text only; an engine without one refuses).
    vision: ?VisionOpen = null,
    /// The prompt cache's disk tier and parked-request scheduling, read from the environment (`Spill.fromEnv`).
    spill: Spill = .{},
};

/// The process environment, for `Spill.fromEnv`.
pub const ProcessEnv = struct {
    pub fn get(_: ProcessEnv, name: [:0]const u8) ?[]const u8 {
        const v = std.c.getenv(name) orelse return null;
        return std.mem.span(v);
    }
};

/// One call's time distribution (prompt_cache.Timing: log2 buckets, count, sum, max).
pub const Timing = prompt_cache.Timing;

/// The prompt cache's durable tier and its scheduling, read once from the environment by `fromEnv` on rank 0 (the
/// store's rank; followers hold no tier state). A value that does not parse or is out of range turns the tier off
/// naming it (`off`), never clamped, and the engine starts either way. An empty value counts as unset.
pub const Spill = struct {
    /// The tier was asked for: TF_DS_SPILL_GIB other than 0 (or not a number), or TF_DS_SPILL_OFF_REASON set.
    asked: bool = false,
    /// Why the tier is off although asked: a bad setting ("NAME=value: rule") or TF_DS_SPILL_OFF_REASON; null: on.
    off: ?[]const u8 = null,
    /// TF_DS_SPILL_DIR: the store's directory (absolute).
    dir: ?[]const u8 = null,
    /// TF_DS_SPILL_GIB: what the store may hold (0, the default: off).
    gib: f64 = 0,
    /// TF_DS_SPILL_MIN_FREE_GIB: no write leaves less free on the store's file system.
    min_free_gib: f64 = 50,
    /// TF_DS_SPILL_QUOTA: one conversation's share of the cap (0: none).
    quota: f64 = 0,
    /// TF_DS_SPILL_MIN_TOKENS: shorter kept states are not written.
    min_tokens: u32 = 8192,
    /// TF_SPILL_TURNS: old turns kept on a conversation's active path before they go first (0: until room is needed).
    turns: u32 = 4,
    /// TF_SPILL_LOAD_MS: one service's budget for every restore together, then staged writes.
    load_ms: u32 = 40,
    /// TF_SPILL_LOAD_WAIT_MS: how long a request waits for its restore in all (deferred while the tier has no room,
    /// then parked on it) before it is admitted without it.
    load_wait_ms: u32 = 30_000,
    /// TF_SPILL_DEFER_MS: how long a request whose restore the tier refused for capacity waits with no restore
    /// finishing (reset when one ends) before it is admitted on what memory holds (0: at once).
    defer_ms: u32 = 5_000,
    /// TF_SPILL_FLUSH_MS: a clean stop's bound for draining writes.
    flush_ms: u32 = 15_000,
    /// TF_SPILL_PUMP_MS: staging's share of a service while streams decode (at most load_ms).
    pump_ms: u32 = 2,
    /// TF_SPILL_POLL_MS: the idle host's poll and the other loop granularities.
    poll_ms: u32 = 2,
    /// TF_SPILL_DEFER_RETRY_MS: how soon a deferred request asks again.
    defer_retry_ms: u32 = 25,
    /// TF_SPILL_DEFER_MIN: the shortest point worth deferring for, in tokens (0: the family's grid).
    defer_min: u32 = 0,
    /// TF_DS_SPILL_STAGING_MIB: pinned host staging for writes, reads and small states together (its minimum, from
    /// the state layout, smalls and restores, is checked where the layout is known).
    staging_mib: u32 = 128,
    /// TF_DS_SPILL_SMALLS: small-state slots for queued writes.
    smalls: u32 = 4,
    /// TF_DS_SPILL_RESTORES: restores at once (also the store's reads at once).
    restores: u32 = 2,
    /// TF_DS_SPILL_READERS: reader threads.
    readers: u32 = 4,
    /// TF_DS_SPILL_BACKLOG_GIB: unpublished write bytes before a keep is refused.
    backlog_gib: f64 = 16,
    /// TF_DS_SPILL_KEEP_BUILDS: earlier builds' stores kept beside this one's.
    keep_builds: u32 = 1,
    /// TF_DS_SPILL_OPEN_MS: how long the start waits for the store's open before it serves without the tier.
    open_ms: u32 = 300_000,
    /// TF_DS_SPILL_IOPRIO: the tier threads' I/O priority (be/0..be/7, idle; none leaves it as it is).
    ioprio: IoPrio = .{},
    /// TF_DS_SPILL_LAND_SLOTS: landing buffers preallocated on every rank.
    land_slots: u32 = 2,
    /// TF_DS_SPILL_LAND_PIECE_MIB: one broadcast piece of a landing.
    land_piece_mib: u32 = 32,
    /// TF_DS_SPILL_PROBE_TOKENS: the start-up probe's prompt.
    probe_tokens: u32 = 8192,
    /// TF_DS_SPILL_PROBE_EVERY: a sampled re-check every N keeps (0: at start only).
    probe_every: u32 = 0,
    /// TF_DS_SPILL_PROBE_ROWS: the rows a probe samples in each plane besides its first and last.
    probe_rows: u32 = 8,
    /// TF_DS_IMAGE_CACHE: pictures kept and found (on unless "0", the server's reading).
    image_cache: bool = true,

    pub const IoPrio = struct {
        class: enum { none, be, idle } = .be,
        level: u3 = 7,
    };

    pub const Int = struct { name: [:0]const u8, field: []const u8, min: u32, max: u32, unit: []const u8 };
    pub const Float = struct { name: [:0]const u8, field: []const u8, min: f64, max: f64, unit: []const u8 };

    /// The whole-number settings with their ranges (a value outside turns the tier off).
    pub const ints = [_]Int{
        .{ .name = "TF_DS_SPILL_MIN_TOKENS", .field = "min_tokens", .min = 0, .max = 1 << 24, .unit = "tokens" },
        .{ .name = "TF_SPILL_TURNS", .field = "turns", .min = 0, .max = 1 << 16, .unit = "turns" },
        .{ .name = "TF_SPILL_LOAD_MS", .field = "load_ms", .min = 1, .max = 10_000, .unit = "milliseconds" },
        .{ .name = "TF_SPILL_LOAD_WAIT_MS", .field = "load_wait_ms", .min = 1, .max = 3_600_000, .unit = "milliseconds" }, // not 0: every restore begun would be dropped unused
        .{ .name = "TF_SPILL_DEFER_MS", .field = "defer_ms", .min = 0, .max = 3_600_000, .unit = "milliseconds" },
        .{ .name = "TF_SPILL_FLUSH_MS", .field = "flush_ms", .min = 0, .max = 3_600_000, .unit = "milliseconds" },
        .{ .name = "TF_SPILL_PUMP_MS", .field = "pump_ms", .min = 0, .max = 10_000, .unit = "milliseconds" },
        .{ .name = "TF_SPILL_POLL_MS", .field = "poll_ms", .min = 1, .max = 1_000, .unit = "milliseconds" },
        .{ .name = "TF_SPILL_DEFER_RETRY_MS", .field = "defer_retry_ms", .min = 1, .max = 60_000, .unit = "milliseconds" },
        .{ .name = "TF_SPILL_DEFER_MIN", .field = "defer_min", .min = 0, .max = 1 << 24, .unit = "tokens" },
        .{ .name = "TF_DS_SPILL_STAGING_MIB", .field = "staging_mib", .min = 1, .max = 65_536, .unit = "MiB" },
        .{ .name = "TF_DS_SPILL_SMALLS", .field = "smalls", .min = 1, .max = 64, .unit = "slots" },
        .{ .name = "TF_DS_SPILL_RESTORES", .field = "restores", .min = 1, .max = 16, .unit = "restores" }, // the tier's bound (spill_tier.max_restores)
        .{ .name = "TF_DS_SPILL_READERS", .field = "readers", .min = 1, .max = 64, .unit = "threads" },
        .{ .name = "TF_DS_SPILL_KEEP_BUILDS", .field = "keep_builds", .min = 0, .max = 16, .unit = "builds" },
        .{ .name = "TF_DS_SPILL_OPEN_MS", .field = "open_ms", .min = 1, .max = 86_400_000, .unit = "milliseconds" },
        .{ .name = "TF_DS_SPILL_LAND_SLOTS", .field = "land_slots", .min = 1, .max = 16, .unit = "slots" },
        .{ .name = "TF_DS_SPILL_LAND_PIECE_MIB", .field = "land_piece_mib", .min = 1, .max = 1024, .unit = "MiB" },
        .{ .name = "TF_DS_SPILL_PROBE_TOKENS", .field = "probe_tokens", .min = 1, .max = 1 << 24, .unit = "tokens" },
        .{ .name = "TF_DS_SPILL_PROBE_EVERY", .field = "probe_every", .min = 0, .max = 1 << 30, .unit = "keeps" },
        .{ .name = "TF_DS_SPILL_PROBE_ROWS", .field = "probe_rows", .min = 0, .max = 62, .unit = "rows a plane" },
    };
    /// The fractional settings with their ranges.
    pub const floats = [_]Float{
        .{ .name = "TF_DS_SPILL_MIN_FREE_GIB", .field = "min_free_gib", .min = 0, .max = 1 << 20, .unit = "GiB" },
        .{ .name = "TF_DS_SPILL_QUOTA", .field = "quota", .min = 0, .max = 1, .unit = "of the cap" },
        .{ .name = "TF_DS_SPILL_BACKLOG_GIB", .field = "backlog_gib", .min = 1.0 / 1024.0, .max = 1 << 20, .unit = "GiB" },
    };
    /// The most TF_DS_SPILL_GIB may be.
    pub const max_gib: f64 = 1 << 20;
    /// Every name `fromEnv` reads (capabilities lists them).
    pub const env_names = blk: {
        var names: []const []const u8 = &.{ "TF_DS_SPILL_GIB", "TF_DS_SPILL_DIR", "TF_DS_SPILL_IOPRIO", "TF_DS_SPILL_OFF_REASON", "TF_DS_IMAGE_CACHE" };
        for (ints) |n| names = names ++ &[_][]const u8{n.name};
        for (floats) |n| names = names ++ &[_][]const u8{n.name};
        break :blk names[0..names.len].*;
    };

    /// The tier is asked for, nothing turned it off, and it has a directory and a cap.
    pub fn enabled(s: Spill) bool {
        return s.asked and s.off == null and s.dir != null and s.gib > 0;
    }

    /// What the store's file system must have room for: this build's store, the kept builds', and the floor.
    pub fn roomGib(s: Spill) f64 {
        return @as(f64, @floatFromInt(1 + s.keep_builds)) * s.gib + s.min_free_gib;
    }

    /// The settings from `env` (anything with `get(name) ?[]const u8`, the process environment in the server). A bad
    /// value's reason is written into `buf` (at least 200 bytes, living as long as the result).
    pub fn fromEnv(env: anytype, buf: []u8) Spill {
        var s: Spill = .{};
        if (value(env, "TF_DS_SPILL_OFF_REASON")) |why| return .{ .asked = true, .off = why };
        const gib_text = value(env, "TF_DS_SPILL_GIB") orelse return s;
        const gib = std.fmt.parseFloat(f64, gib_text) catch null;
        if (gib) |g| if (g == 0) return s;
        s.asked = true;
        s.gib = gib orelse return s.bad(buf, "TF_DS_SPILL_GIB", gib_text, "GiB, more than 0 up to 1048576");
        if (!std.math.isFinite(s.gib) or s.gib < 0 or s.gib > max_gib) return s.bad(buf, "TF_DS_SPILL_GIB", gib_text, "GiB, more than 0 up to 1048576");
        const dir = value(env, "TF_DS_SPILL_DIR") orelse {
            s.off = std.fmt.bufPrint(buf, "TF_DS_SPILL_DIR: not set (TF_DS_SPILL_GIB={s} asks for the tier)", .{clip(gib_text)}) catch "TF_DS_SPILL_DIR: not set";
            return s;
        };
        if (dir[0] != '/') return s.bad(buf, "TF_DS_SPILL_DIR", dir, "an absolute path");
        s.dir = dir;
        inline for (ints) |n| if (value(env, n.name)) |t| {
            const v = std.fmt.parseInt(u32, t, 10) catch null;
            if (v == null or v.? < n.min or v.? > n.max) return s.bad(buf, n.name, t, std.fmt.comptimePrint("{s}, a whole number from {d} to {d}", .{ n.unit, n.min, n.max }));
            @field(s, n.field) = v.?;
        };
        inline for (floats) |n| if (value(env, n.name)) |t| {
            const v = std.fmt.parseFloat(f64, t) catch null;
            if (v == null or !std.math.isFinite(v.?) or v.? < n.min or v.? > n.max) return s.bad(buf, n.name, t, std.fmt.comptimePrint("{s}, from {d} to {d}", .{ n.unit, n.min, n.max }));
            @field(s, n.field) = v.?;
        };
        if (value(env, "TF_DS_SPILL_IOPRIO")) |t| s.ioprio = ioPrio(t) orelse return s.bad(buf, "TF_DS_SPILL_IOPRIO", t, "be/0 to be/7, idle or none");
        // untrimmed, as the server's picture keys read it (prompt_cache.imageCacheOn: the one reading)
        s.image_cache = @import("prompt_cache.zig").imageCacheOn(env.get("TF_DS_IMAGE_CACHE"));
        if (s.pump_ms > s.load_ms) {
            s.off = std.fmt.bufPrint(buf, "TF_SPILL_PUMP_MS={d}: more than TF_SPILL_LOAD_MS={d} (staging's share of a service cannot pass the service)", .{ s.pump_ms, s.load_ms }) catch "TF_SPILL_PUMP_MS: more than TF_SPILL_LOAD_MS";
            return s;
        }
        return s;
    }

    /// The state to show: the engine's when it reported one, else off for a bad setting, else opening.
    pub fn stateOf(s: Spill, now: ?SpillState) SpillState {
        if (now) |n| return n;
        if (s.off) |why| return SpillState.offFor(why);
        return .{ .state = .opening };
    }

    /// The derived room with its formula, as /health and the start-up line show it.
    pub fn roomFormula(s: Spill, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("(1 + keep_builds {d}) x gib {d} + min_free_gib {d} = {d} GiB", .{ s.keep_builds, s.gib, s.min_free_gib, s.roomGib() });
    }

    /// TF_DS_SPILL_IOPRIO's text.
    pub fn ioprioText(s: Spill, buf: *[8]u8) []const u8 {
        return switch (s.ioprio.class) {
            .be => std.fmt.bufPrint(buf, "be/{d}", .{s.ioprio.level}) catch unreachable,
            .idle => "idle",
            .none => "none",
        };
    }

    /// The start-up log line (and /health's settings): the state, every value, and the derived room with its formula.
    pub fn describe(s: Spill, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (!s.asked) return w.writeAll("spill: not asked (TF_DS_SPILL_GIB unset or 0)");
        if (s.off) |why| return w.print("spill: off: {s}", .{why});
        try w.print("spill: on: dir={s} gib={d} min_free_gib={d} room=", .{ s.dir.?, s.gib, s.min_free_gib });
        try s.roomFormula(w);
        inline for (ints) |n| try w.print(" {s}={d}", .{ n.field, @field(s, n.field) });
        inline for (floats) |n| if (comptime !std.mem.eql(u8, n.field, "min_free_gib")) try w.print(" {s}={d}", .{ n.field, @field(s, n.field) });
        var io_buf: [8]u8 = undefined;
        try w.print(" ioprio={s} image_cache={s}", .{ s.ioprioText(&io_buf), if (s.image_cache) "on" else "off" });
    }

    fn bad(s: Spill, buf: []u8, comptime name: []const u8, text: []const u8, comptime rule: []const u8) Spill {
        var out = s;
        out.off = std.fmt.bufPrint(buf, name ++ "={s}: " ++ rule, .{clip(text)}) catch name ++ ": a bad value (" ++ rule ++ ")";
        return out;
    }

    /// A set value, trimmed of spaces; null when unset or empty.
    fn value(env: anytype, comptime name: [:0]const u8) ?[]const u8 {
        const t = std.mem.trim(u8, env.get(name) orelse return null, " ");
        return if (t.len == 0) null else t;
    }

    fn clip(t: []const u8) []const u8 {
        return t[0..@min(t.len, 64)];
    }

        if (std.mem.eql(u8, t, "idle")) return .{ .class = .idle, .level = 0 };
        if (std.mem.eql(u8, t, "none")) return .{ .class = .none, .level = 0 };
        if (!std.mem.startsWith(u8, t, "be/") or t.len != 4 or t[3] < '0' or t[3] > '7') return null;
        return .{ .class = .be, .level = @intCast(t[3] - '0') };
    }
};

/// A packed decision: a trunk (pictures expand at `at`), a branch a question, and each branch's labels (`counts`).
pub const Decision = struct {
    trunk: []const u32,
    at: usize = 0,
    pictures: []const []const u8 = &.{}, // data URLs, in prompt order
    branches: []const []const u32 = &.{},
    counts: []const usize = &.{},
    labels: []const u32 = &.{},
    logits: []f64 = &.{},
    read: usize = 0, // out: the trunk's tokens with its pictures expanded, plus every branch's
    problem: []const u8 = "", // out: why the engine refused (error.Refused), in `arena`
    arena: ?std.mem.Allocator = null, // the request's, for what the engine hands back
};

/// An opened engine; ``close`` stops its thread and frees its backend. ``quiesce``, when the engine has one, is the
/// orderly close without the frees, for a server's stop before its connections end: every request ends (cancelled),
/// the engine's own last work runs (a family's followers hear done), and connection threads' calls stay valid (submit
/// refused as Closed; cancel, status and memory unchanged). ``close`` after it frees the rest without repeating it.
pub const Opened = struct { engine: Engine, close: *const fn (ctx: *anyopaque) void, ctx: *anyopaque, quiesce: ?*const fn (ctx: *anyopaque) void = null };

pub const Memory = struct { active: u64 = 0, cache: u64 = 0, peak: u64 = 0, pinned: u64 = 0, limit: ?u64 = null };

/// A backend's words for its own refusals (a request it cannot serve); null: the error's name.
pub const Explain = struct {
    ctx: ?*anyopaque = null,
    text: *const fn (ctx: ?*anyopaque, err: anyerror) ?[]const u8,
};

/// A backend's own memory counts for LaneHost; read from HTTP threads, so it must not touch the GPU.
pub const MemorySource = struct {
    ctx: ?*anyopaque = null,
    read: *const fn (ctx: ?*anyopaque, reset_peak: bool) ?Memory,
};

/// The prompt cache's counters for /health and /metrics, read from HTTP threads as one copy (LaneHost publishes it on
/// its engine thread). Null in Status unless a tier or grouping is on. Totals unless in `gauges`.
pub const CacheStats = struct {
    memory_hits: u64 = 0,
    memory_misses: u64 = 0,
    kept: u64 = 0,
    evicted: u64 = 0,
    refused: u64 = 0,
    failed: u64 = 0,
    entries: u64 = 0,
    held_bytes: u64 = 0,
    disk_lookups: u64 = 0,
    disk_hits: u64 = 0,
    disk_hits_committed: u64 = 0,
    restores_begun: u64 = 0,
    restores_shared: u64 = 0,
    restores_failed: u64 = 0,
    restores_cancelled: u64 = 0,
    restores_dropped: u64 = 0,
    restores_refused: u64 = 0,
    restores_active: u64 = 0,
    restoring_bytes: u64 = 0,
    restore_blocks_done: u64 = 0,
    restore_blocks_total: u64 = 0,
    restored_tokens: u64 = 0,
    restore_waits: u64 = 0,
    restore_wait_timeouts: u64 = 0,
    restore_wait_ns: u64 = 0,
    restore_wait_max_ns: u64 = 0,
    restores_deferred: u64 = 0,
    restores_deferred_expired: u64 = 0,
    writes_enqueued: u64 = 0,
    shared_promoted: u64 = 0,
    flushes: u64 = 0,
    flush_timeouts: u64 = 0,
    flush_unpublished: u64 = 0,
    parked: u64 = 0,
    chats_continued: u64 = 0,
    chats_continued_prefill_tokens: u64 = 0,
    cached_tokens: u64 = 0,
    computed_tokens: u64 = 0,
    service_refused: u64 = 0,
    token_bytes: u64 = 0,
    /// The tier's own counters, each rank's (null: no tier).
    spill: ?prompt_cache.RankCounts = null,
    /// The tier's call timings (the lane host's store calls and the tier's own work).
    times: prompt_cache.TierTimes = .{},

    pub const gauges = [_][]const u8{ "entries", "held_bytes", "restores_active", "restoring_bytes", "restore_blocks_done", "restore_blocks_total", "parked", "token_bytes" };

    /// Each field's /metrics help (a missing one fails the build).
    pub const help = .{
        .memory_hits = "Prompt passes that resumed a kept state in memory.",
        .memory_misses = "Prompt lookups that found no usable kept state.",
        .kept = "Prompt states kept in memory.",
        .evicted = "Kept states evicted from memory for room.",
        .refused = "Prompt states not kept: past the memory budget.",
        .failed = "Kept states that failed to save or restore.",
        .entries = "Prompt states kept in memory now.",
        .held_bytes = "Bytes the kept states hold now.",
        .disk_lookups = "Lookups that asked the disk tier.",
        .disk_hits = "Lookups whose durable point was longer than the best usable memory state.",
        .disk_hits_committed = "Disk restores every rank installed, adopted as kept states.",
        .restores_begun = "Disk restore jobs started.",
        .restores_shared = "Requests that joined a disk restore already in flight.",
        .restores_failed = "Disk restores aborted on every rank (missing, damaged, guarded).",
        .restores_cancelled = "Disk restores stopped: no request waited for them any more, or shutdown.",
        .restores_dropped = "Installed disk restores not adopted: no room, or memory already had them.",
        .restores_refused = "Disk restores not begun: the memory budget could not hold their destination, or the tier was at its capacity (once a request).",
        .restores_active = "Disk restore jobs in flight.",
        .restoring_bytes = "Memory the disk restores in flight reserved, charged to the prompt cache's budget.",
        .restore_blocks_done = "Blocks the restores in flight have copied so far.",
        .restore_blocks_total = "Blocks the restores in flight copy in all.",
        .restored_tokens = "Prompt tokens restored from disk.",
        .restore_waits = "Requests parked on a disk restore.",
        .restore_wait_timeouts = "Requests admitted without their disk restore after TF_SPILL_LOAD_WAIT_MS.",
        .restore_wait_ns = "Nanoseconds requests waited for disk restores (from their first wait, a deferral included).",
        .restore_wait_max_ns = "The longest wait for a disk restore, in nanoseconds.",
        .restores_deferred = "Requests whose disk restore was refused for capacity (restores or read slots in flight) and retried shortly instead of prefilling.",
        .restores_deferred_expired = "Of them, requests admitted on what memory holds without the restore starting: TF_SPILL_DEFER_MS with no restore ending, or TF_SPILL_LOAD_WAIT_MS in all.",
        .writes_enqueued = "Kept states enqueued for the disk tier at their keep.",
        .shared_promoted = "Disk points promoted to shared prefixes.",
        .flushes = "Clean-shutdown flushes of the disk tier.",
        .flush_timeouts = "Flushes that reached their deadline with points unpublished.",
        .flush_unpublished = "Points left unpublished at a flush deadline.",
        .parked = "Requests parked on a disk restore now (reading it, or deferred to ask again).",
        .chats_continued = "Streams that continued a conversation without resuming from its own turn state.",
        .chats_continued_prefill_tokens = "Prompt tokens those streams prefilled.",
        .cached_tokens = "Prompt tokens passes resumed from kept states.",
        .computed_tokens = "Prompt tokens passes computed.",
        .service_refused = "Parked-work services asked for inside a round, a prompt chunk or a CUDA graph capture, skipped.",
        .token_bytes = "Bytes of the kept states' host token arrays (their ids, last prompts, lineage) the memory budget counts (with the SSD tier on).",
    };
};

pub const Status = struct {
    /// Requests in prefill or decode, and those waiting for a lane.
    running: u32 = 0,
    waiting: u32 = 0,
    decode_tokens_per_second: f64 = 0,
    prefill_tokens_per_second: f64 = 0,
    /// Requests that gave a lane up to a later one; null when the engine does not count them.
    preemptions: ?u64 = null,
    /// CUDA graph captures that failed, their work run eagerly instead; null when the engine does not count them.
    graph_fallbacks: ?u64 = null,
    /// Decode rounds by the path the output head's logits took to the draws, in this order: the device's argmax, the
    /// tokens merged from the nodes' bests, the GPU's candidates, the candidates merged from the nodes' parts,
    /// candidates a flag sent on to whole rows, whole rows; null when the engine does not count them.
    head_rounds: ?[6]u64 = null,
    /// The drafter on a pruned vocabulary (DeepSeek's TF_DS_DRAFT_VOCAB): its passes and its guard's counts; null when
    /// the engine does not run one.
    draft_vocab: ?DraftVocab = null,
    warming: bool = false,
    /// Live streams written to the caller's buffer: tokens each holds.
    streams: usize = 0,
    /// Generated tokens held by live streams, excluding their prompts.
    generation_tokens: u64 = 0,
    /// Filling prompts' total tokens and those still to prefill (null: no prompt is going in).
    fill_prompt_tokens: ?u32 = null,
    fill_tokens_left: ?u32 = null,
    /// A prompt pass is running whose backend reports no `fill_at`: the totals stay null rather than guessed.
    fill_unknown: bool = false,
    /// Changes whenever the filling set does: a reader pairs its drain samples with one fill.
    fill_run: u64 = 0,
    /// The prompt cache's counters, when its tier or grouping is on.
    cache: ?CacheStats = null,
    /// The disk tier's state, when it was asked for (outside `cache`: a tier that is off still shows).
    spill: ?SpillState = null,
};

pub const SubmitError = error{ Closed, Busy, InvalidSpans, Stale }; // Stale: a learner's put-back has not reached the device

/// What a reply refused or ended for `Stale` says, a submit's error and a queued job's failure alike.
pub const stale_message = "a Living Weights delete could not be put back on the device: send the delete again to retry it, or restart the server";

/// A worked example for the learner: token ids, the answer from `start`; `decide`: its assistant header's last row.
pub const Example = struct { ids: []const u32, start: u32, decide: u32, subject: u32 = 0, value: u32 = 0 }; // subject: the row of its last token; value: the row before its answer's value (0: none)

/// One fact's lesson (Sliding Weights): answers, held-out answers, near misses, and prompts that must not move.
pub const LearnRequest = struct {
    train: []const Example = &.{},
    think: []const Example = &.{}, // the facts' answers with thinking on, learned beside `train` (one of each a step)
    held: []const Example = &.{},
    near: []const Example = &.{},
    keep: []const Example = &.{}, // answers of every kind this lesson's change must leave as they are
    asked: []const Example = &.{}, // more wordings of the fact's questions, for its router (prompts only)
    unasked: []const Example = &.{}, // questions it must not answer, for its router (prompts only)
    kept: []const Example = &.{}, // questions the last lesson took, with the answers they had (a reroute's)
    undo: bool = false, // instead take the last lesson's change back, as if it never ran
    steps: u32 = 60, // bounded steps this time at most
    more: bool = false, // more steps on the last lesson's rows, which are not captured again
    reroute: bool = false, // the last lesson's router fitted again with `unasked` among what it must not take
    drop: bool = false, // the last lesson taken out whole
    save: bool = false, // every kept lesson written beside the model's shards, each with its router
    retire: []const Neuron = &.{}, // instead zero these neurons of facts a newer fact replaces
    unretire: bool = false, // instead put back the neurons the last retire zeroed (the newer fact was not learned)
    forget: []const Neuron = &.{}, // instead zero these neurons' input weights (a fact deleted on request), kept to put back
    unforget: bool = false, // instead put back, bit for bit, what the last forget changed (a kept fact did not come back)
};

/// A neuron a pocket lesson placed (Sliding Weights): its layer, its index in that layer's pocket, the row it keys on.
pub const Neuron = struct { layer: u16, neuron: u16, key: NeuronKey };
pub const NeuronKey = enum { reply, subject, value };

/// A pocket lesson's stage, as it runs: its rows read, its neurons fitted, their outputs, its held-out answers judged.
pub const LearnPhase = enum { reading, fitting, values, checking };

/// Where a pocket lesson stands: its unit `at` of `of` (0: unknown), its outputs' loss, its neurons once placed.
pub const LearnProgress = struct { phase: LearnPhase, at: u32 = 0, of: u32 = 0, loss: f32 = 0, neurons: []const Neuron = &.{} };

/// A lesson's outcome in order, ending with `done` (a message says why it failed); slices live only during the call.
pub const LearnEvent = union(enum) {
    learned: struct { recalled: bool, steps: u32, loss: f32, neurons: []const Neuron = &.{} }, // neurons: a pocket's
    progress: LearnProgress,
    done: struct { message: []const u8 = "" },
};

/// Where a learn request's events go; called on the engine's thread, so it must return at once.
pub const LearnSink = struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, event: *const LearnEvent) void,
};

pub const LearnError = error{ Closed, Busy, Unsupported };

/// A family's learner, which LaneHost steps one bounded unit at a time while no stream decodes.
pub const Learner = struct {
    ctx: *anyopaque,
    /// Start a job; its events, `done` last, go to `sink` from later steps.
    begin: *const fn (ctx: *anyopaque, request: *const LearnRequest, sink: LearnSink) anyerror!void,
    /// One unit of the job: `changed` when weights moved (kept prompt states are then dropped), `done` when it ended.
    step: *const fn (ctx: *anyopaque) Step,
    /// End the open job now; it still sends `done`, with the reason.
    abort: *const fn (ctx: *anyopaque) void,

    /// stale: the device lacks a put-back the learner holds, so replies are refused until a later step clears it.
    pub const Step = struct { done: bool, changed: bool, stale: bool = false };
};

pub const Engine = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        info: *const fn (ctx: *anyopaque) Info,
        submit: *const fn (ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void,
        /// Ends a request between rounds; its ``finished`` (cancelled) still arrives. Unknown ids are ignored.
        cancel: *const fn (ctx: *anyopaque, id: Id) void,
        status: *const fn (ctx: *anyopaque, out: *Status, stream_tokens: []u32) void,
        /// Device memory in bytes, the peak since the last reset; null when the backend keeps no count.
        memory: *const fn (ctx: *anyopaque, reset_peak: bool) ?Memory,
        /// The engine's queue as a keepalive target, or null when the backend is not Metal.
        keepalive: ?*const fn (ctx: *anyopaque) ?keepalive.Target = null,
        /// One logit per label at the row after `prompt`, and the vocabulary logsumexp when `mass` (else 0).
        score: ?*const fn (ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 = null,
        /// Each branch's label logits at its last row after the decision's shared trunk.
        decide: ?*const fn (ctx: *anyopaque, d: *Decision) error{ Failed, Refused }!void = null,
        /// Queue `request` for the family's learner; it stays valid until `done`. Null: this engine does not learn.
        learn: ?*const fn (ctx: *anyopaque, request: *const LearnRequest, sink: LearnSink) LearnError!void = null,
        /// Recall's record for every later request's drafts and writes, before any request (null: no record read).
        recall: ?*const fn (ctx: *anyopaque, store: *lanes.recall_store.Store) void = null,
        /// Queue a question about kept states for the engine thread (it stays valid until `done`); false: refused.
        imprint: ?*const fn (ctx: *anyopaque, ask: *ImprintAsk) bool = null,
        /// A planned stop's first half: no more work, disk writes finished under a deadline, the engine thread joined.
        drain: ?*const fn (ctx: *anyopaque) void = null,
    };

    pub fn info(e: Engine) Info {
        return e.vtable.info(e.ctx);
    }
    pub fn submit(e: Engine, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        try @import("cache_modes.zig").validate(request.decode_spans);
        return e.vtable.submit(e.ctx, id, request, sink);
    }
    pub fn cancel(e: Engine, id: Id) void {
        e.vtable.cancel(e.ctx, id);
    }
    pub fn status(e: Engine, out: *Status, stream_tokens: []u32) void {
        e.vtable.status(e.ctx, out, stream_tokens);
    }
    pub fn memory(e: Engine, reset_peak: bool) ?Memory {
        return e.vtable.memory(e.ctx, reset_peak);
    }
    /// The engine's queue as a keepalive target, or null when the backend is not Metal.
    pub fn keepaliveTarget(e: Engine) ?keepalive.Target {
        const f = e.vtable.keepalive orelse return null;
        return f(e.ctx);
    }

    /// Hands the engine Recall's record; false when it drafts from none (its own copies only).
    pub fn attachRecall(e: Engine, store: *lanes.recall_store.Store) bool {
        const f = e.vtable.recall orelse return false;
        f(e.ctx, store);
        return true;
    }

    /// Queues `ask` for the engine thread; false when the engine answers none (its `done` never comes).
    pub fn imprintAsk(e: Engine, ask: *ImprintAsk) bool {
        const f = e.vtable.imprint orelse return false;
        return f(e.ctx, ask);
    }

    /// Finishes the engine's disk writes before the process ends; a no-op where it keeps none.
    pub fn drain(e: Engine) void {
        if (e.vtable.drain) |f| f(e.ctx);
    }

    /// The decision hook, or Unsupported when this engine does not score.
    pub fn score(e: Engine, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{ Failed, Unsupported }!f64 {
        const f = e.vtable.score orelse return error.Unsupported;
        return f(e.ctx, prompt, labels, logits, mass);
    }

    pub fn decide(e: Engine, d: *Decision) error{ Failed, Refused, Unsupported }!void {
        const f = e.vtable.decide orelse return error.Unsupported;
        return f(e.ctx, d);
    }

    /// The Sliding Weights hook, or Unsupported when this engine does not learn.
    pub fn learn(e: Engine, request: *const LearnRequest, sink: LearnSink) LearnError!void {
        const f = e.vtable.learn orelse return error.Unsupported;
        return f(e.ctx, request, sink);
    }
};

/// A backend's own driver for a lone greedy stream (the GPU round), which LaneHost runs while the stream is alone.
pub const Lone = struct {
    ctx: *anyopaque,
    sampled: bool = false, // it drives sampled streams too
    logprobs: bool = false, // it gives each token its logprob row too
    structures: bool = false, // it masks each window's rows by the stream's grammar too
    min_budget: u32 = 0, // thinking budgets above this and loop guards: it hands the stream over before a forced close (0: never)
    resumes: bool = false, // it takes the lane core's stream back between rounds (LoneHooks.resumed); else only opens
    /// Prefills (or resumes) `s` and decodes it until it finishes (false) or `hooks.yield` hands it to the lane core.
    run: *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: LoneHooks) anyerror!bool,
};

/// What a lone driver tells the host between rounds: tokens landed; and asks: hand the stream over now?
pub const LoneHooks = struct {
    ctx: *anyopaque,
    committed: *const fn (ctx: *anyopaque) void,
    yield: *const fn (ctx: *anyopaque) bool,
    resumed: bool = false, // the stream comes from the lane core between rounds: no prompt pass, its drafts the core's
};

/// The lane core served to the HTTP threads (lane_host.zig).
pub const LaneHost = @import("lane_host.zig").LaneHost;

/// The host's share and round-count settings (lane_host_share.zig): the family reads TF_DS_ROUNDS_BESIDE's text through it.
pub const share = @import("lane_host_share.zig");

/// Exact prompt reuse between requests, for any family (prompt_cache.zig).
pub const prompt_cache = @import("prompt_cache.zig");

/// The same for a paged cache over the backend's own KV pages (prompt_radix.zig).
pub const prompt_radix = @import("prompt_radix.zig");

/// Decoded prompt rows as canonical spans, for a cache keyed by the arithmetic that made each row (cache_modes.zig).
pub const cache_modes = @import("cache_modes.zig");

/// Learned prompt-cache states on disk (prompt_imprint.zig), and when a shared cut goes there (prompt_learn.zig).
pub const prompt_imprint = @import("prompt_imprint.zig");
pub const prompt_learn = @import("prompt_learn.zig");

/// A question about a conversation's kept states, in RAM and on disk (rows_ask.zig).
pub const ImprintAsk = @import("rows_ask.zig").Ask;
pub const ImprintOp = @import("rows_ask.zig").Op;
pub const ImprintDisk = @import("rows_ask.zig").Disk;

/// Recall's record of the ids earlier requests read and wrote, for drafts (not the prompt cache's learned states).
pub const recall_store = lanes.recall_store;

/// The idle keepalive's ticker and target contract.
pub const keepalive = @import("keepalive.zig");

/// Rewriting engine.cfg for a persisted change (cfg_write.zig).
pub const cfg_write = @import("cfg_write.zig");

/// The settings of class H that rank 0's host takes at its next loop turn (live_config.zig).
pub const live_config = @import("live_config.zig");

/// The KV pool sized from the free memory the operator leaves, and the ledger of a start (kv_pool.zig).
pub const kv_pool = @import("kv_pool.zig");

/// The memory each follower rank last reported to rank 0, for the admin surface (peer_mem.zig).
pub const peer_mem = @import("peer_mem.zig");

/// What the engine decided about its transports at the start (transport_facts.zig), for /status.
pub const transport = @import("transport_facts.zig");

/// A model folder's original files kept before --slide first teaches it, its hold and its rollback (originals.zig).
pub const originals = @import("originals.zig");

test {
    _ = @import("fill_order.zig");
    _ = @import("lane_host.zig");
    _ = @import("lane_host_fill_test.zig");
    _ = @import("lane_host_reuse_test.zig");
    _ = @import("lane_host_share.zig");
    _ = @import("lane_host_share_test.zig");
    _ = @import("prompt_cache.zig");
    _ = @import("prompt_cache_media_test.zig");
    _ = @import("spill_settings_test.zig");
    _ = @import("prompt_imprint.zig");
    _ = @import("keepalive.zig");
    _ = @import("kv_pool.zig");
    _ = @import("peer_mem.zig");
    _ = @import("kv_pool_test.zig");
    _ = @import("cfg_write.zig");
    _ = @import("cfg_write_test.zig");
    _ = @import("live_config.zig");
    _ = @import("live_config_test.zig");
    _ = @import("originals_test.zig");
}

test "a gated pass counts a pair's rows when the device has finished each pass, not when the host has issued them" {
    var gate: std.atomic.Value(u64) = .init(0);
    var p: lanes.stream.Progress = .{};
    p.begin(0);
    p.gated(&gate, 1);
    gate.store(1 << 32, .release); // armed for this fill: nothing finished
    // the host issued a whole pair (two 4,096-row passes) and returned: what prefillStep reports at once
    p.reach(8192);
    try std.testing.expectEqual(@as(u32, 0), p.computed());
    const issued: PromptPass = .{ .computed = p.computed(), .total = 8192, .resumed = 0, .seconds = 0.6 };
    try std.testing.expectEqual(@as(?f64, null), issued.rate()); // not 8,192 / 0.6 = 13.7 K tok/s
    // pass A's last layer is done
    gate.store((1 << 32) | 4096, .release);
    try std.testing.expectEqual(@as(u32, 4096), p.computed());
    const half: PromptPass = .{ .computed = p.computed(), .total = 8192, .resumed = 0, .seconds = 0.8 };
    try std.testing.expectApproxEqAbs(@as(f64, 5120), half.rate().?, 1e-9);
    // pass B's
    gate.store((1 << 32) | 8192, .release);
    try std.testing.expectEqual(@as(u32, 8192), p.computed());
    const done: PromptPass = .{ .computed = p.computed(), .total = 8192, .resumed = 0, .seconds = 1.1 };
    try std.testing.expectApproxEqAbs(@as(f64, 8192.0 / 1.1), done.rate().?, 1e-6);
    // a callback of an earlier fill (another generation) neither lifts nor lowers it; the device's rows never pass the host's
    gate.store((7 << 32) | 8192, .release);
    try std.testing.expectEqual(@as(u32, 0), p.computed());
    p.gated(&gate, 7);
    p.reach(4096);
    try std.testing.expectEqual(@as(u32, 4096), p.computed());
    // a pass resumed from a kept state counts from its resume point; begin() takes the gate off (a host pass without one)
    p.begin(2048);
    p.reach(4096);
    try std.testing.expectEqual(@as(u32, 2048), p.computed());
}
