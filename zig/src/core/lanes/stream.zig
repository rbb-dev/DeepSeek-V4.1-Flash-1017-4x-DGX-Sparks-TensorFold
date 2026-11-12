//! One stream (Python LaneStream) and the round loop's state for it (Python's dicts keyed by stream id).
const std = @import("std");
const grammar = @import("grammar.zig");
const shape = @import("shape.zig");
const plan_lanes = @import("plan_lanes.zig");
const Allocator = std.mem.Allocator;
const Proposer = @import("proposer.zig").Proposer;
const Reader = @import("recall.zig").Reader;
const Sampling = @import("sampling.zig").Sampling;
const State = @import("depth.zig").State;
const Row = @import("logprob.zig").Row;
const Novelty = @import("novelty.zig").Novelty;

pub const Reason = enum {
    none,
    stop,
    length,
    @"error",
    cancelled,

    pub fn name(r: Reason) []const u8 {
        return if (r == .none) "" else @tagName(r);
    }
};

pub const Mode = enum { pipe, drain, verify, exit };

/// One depth of the head's chains: rounds whose chain drafted that deep, those that reached it (kept every draft above
/// it), and those that kept its draft too (engine_api.DraftDepth).
pub const Depth = struct { drafted: u32 = 0, reached: u32 = 0, kept: u32 = 0 };

pub const loop_max_period: usize = 8;
pub const loop_min_run: usize = 256;
pub const loop_warm_in: usize = 64;

/// Return the shortest exact cycle at the end of a combined token prefix.
fn cycleAt(head: []const u32, tail: []const u32, n: usize) ?u32 {
    if (n < loop_min_run + 1) return null;
    for (1..loop_max_period + 1) |period| {
        if (n < loop_min_run + period) break;
        if (n - loop_min_run - period < loop_warm_in) break;
        var matches = true;
        for (1..loop_min_run + 1) |k| {
            const right = n - k;
            const left = right - period;
            const right_token = if (right < head.len) head[right] else tail[right - head.len];
            const left_token = if (left < head.len) head[left] else tail[left - head.len];
            if (right_token != left_token) {
                matches = false;
                break;
            }
        }
        if (matches) return @intCast(period);
    }
    return null;
}

/// Return the first token index in a candidate batch that would fire the guard.
pub fn loopCut(s: *const Stream, tokens: []const u32) ?usize {
    if (!s.guarding()) return null;
    const dry = if (s.novelty) |*n| n.firesAt(s.emitted(), tokens) else null;
    for (tokens, 0..) |token, i| {
        if (@as(i64, token) == s.think_end) return null;
        if (cycleAt(s.emitted(), tokens, s.emitted().len + i + 1) != null) return i;
        if (dry) |at| if (at == i) return i;
    }
    return null;
}

/// Return the detected period at the end of committed tokens, or null.
pub fn detectLoop(emitted: []const u32) ?u32 {
    return cycleAt(emitted, &.{}, emitted.len);
}

/// A stop-string check over the emitted tokens (the server's StopPolicy decodes their tail).
pub const StopCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque, emitted: []const u32) bool,
};

/// The host's answer to whether the stream's request was cancelled, asked between prompt chunks.
pub const CancelCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque) bool,
};

/// A kept prompt state the backend restores before its prompt pass (core/prompt_cache.zig), and where it keeps new ones.
pub const Reuse = struct {
    saved: ?*anyopaque = null, // the backend's copy of the state after `at` prompt tokens (null: start at 0)
    at: u32 = 0,
    marks: []const u32 = &.{}, // ascending: the pass cuts a chunk at each and calls `hook` there
    hook: ?MarkHook = null,
};

/// Called between prompt chunks, the pass standing at `at` (the GPU done with the chunk that ends there).
pub const MarkHook = struct {
    ptr: *anyopaque,
    at: *const fn (ptr: *anyopaque, s: *Stream, at: u32) void,
};

/// What follows a prompt's step in the host's plan, asked at the step's end: this prompt's next step, a decode round of
/// the live streams, another prompt's step, or nothing.
pub const Next = enum { same, round, other, idle };

/// The host's scheduling facts for a prompt's next step, asked at the step's start (passRows reads them alone):
/// `decoding`, another stream decodes; `filling`, another prompt than this one fills (its fill started and not
/// finished); `waiting`, a request other than this one is queued, parked, woken, or admitted and not yet started;
/// `step_rows`, the most prompt rows the step may run before the host's next round (under the default cadence the
/// family's resume rows while a stream decodes, maxInt(u32) while nothing decodes); `rounds_beside`, the rounds the host
/// runs after each long step while a stream decodes (TF_DS_ROUNDS_BESIDE; 0: unset, one round).
pub const Facts = struct { decoding: bool, filling: bool, waiting: bool, step_rows: u32, rounds_beside: u32 = 0 };

/// The most rows a prompt's step at `start` may pass, from the host's facts alone: `resume_rows` with no facts (a host
/// without them), below `pass_from`, while another request waits or, with `filling_pass` off, another prompt fills; else
/// `pass_rows` when nothing decodes or the host's step rows reach them; else the step rows in whole resume units (besideRows). With `filling_pass` on
/// another prompt filling does not cut the pass (prompts filling together take turns at whole passes); a waiting
/// request still does, and a decoding stream still leaves the pass to the host's step rows.
pub fn passRows(f: ?Facts, start: usize, pass_from: usize, pass_rows: u32, resume_rows: u32, filling_pass: bool) u32 {
    const x = f orelse return resume_rows;
    if (start < pass_from or x.waiting or (x.filling and !filling_pass)) return resume_rows;
    if (!x.decoding or x.step_rows >= pass_rows) return pass_rows;
    return besideRows(x.step_rows, pass_rows, resume_rows);
}

/// A step beside a decoding stream: the host's step rows in whole resume units, never above the pass rows and never below
/// one unit (a pass is a whole number of units; the step rows between two sizes run the lower).
pub fn besideRows(step_rows: u32, pass_rows: u32, resume_rows: u32) u32 {
    if (resume_rows == 0) return pass_rows;
    const units = step_rows / resume_rows;
    return @min(pass_rows, @max(1, units) * resume_rows);
}

/// Whether the step at `start` is a whole pass of the booted `pass_rows` by the host's facts alone, and may be the first of a pair:
/// the facts passRows reads (a host without them: no), and no stream decoding (a pair is one frame behind which the rounds wait)
/// unless the host owes the streams a count of rounds a step (`rounds_beside`, TF_DS_ROUNDS_BESIDE: 2N follow the pair) and its
/// step rows reach the pass rows (`pass_rows`, the booted ones).
/// Where `pass_rows` is the resume rows (TF_DS_PASS_ROWS=2048) a cut pass has the booted rows too, so the rows cannot tell a whole
/// pass from a cut one; the facts do. For pass rows above the resume rows it is passRows(...) == pass_rows and !decoding (a test).
/// Beside decoders without a count of rounds a pair is not formed, so "step rows" (Facts.step_rows) are a pass's, never a pair's.
pub fn wholePass(f: ?Facts, start: usize, pass_from: usize, filling_pass: bool, pass_rows: u32) bool {
    const x = f orelse return false;
    if (start < pass_from or x.waiting or (x.filling and !filling_pass)) return false;
    return !x.decoding or (x.rounds_beside != 0 and x.step_rows >= pass_rows);
}

/// Called between prompt chunks after the chunk's mark hook, the GPU's queue holding the chunk (the decode plan's Track
/// D, resumable prefill): the host's turn to run a round of the streams already decoding, so one prompt's pass stops
/// nobody's decode for its whole length. An error ends this prompt's pass.
pub const BetweenHook = struct {
    ptr: *anyopaque,
    run: *const fn (ptr: *anyopaque) anyerror!void,
    ready: *const fn (ptr: *anyopaque) bool, // whether `run` would run a round (another stream is live): `next == .round`
    /// The host's other turn at the same boundary, after the round part whether or not it ran: its parked work's
    /// service (the disk tier's restores, staged writes, deferred retries). Null: the host has none.
    tier: ?Service = null,
    /// What follows the step just issued (asked at its end). Null: the host has no plan to tell.
    next: ?*const fn (ptr: *anyopaque) Next = null,
    /// The host's facts for this prompt's next step (asked at its start). Null: none, and passRows gives the resume rows.
    facts: ?*const fn (ptr: *anyopaque) Facts = null,

    /// No round, no Engram read: called without joining the read-ahead; it cannot fail the pass. `ptr` is the hook's.
    pub const Service = struct {
        wanted: *const fn (ptr: *anyopaque) bool, // whether `run` has work now
        run: *const fn (ptr: *anyopaque) void,
    };
};

/// A backend that issues work ahead of the device (the host returns while the kernels still run) gates the count: `gate`
/// holds `generation << 32 | rows` the device has finished, and a reader counts no more than that.
    gen: std.atomic.Value(u32) = .init(0),
    gate: std.atomic.Value(?*const std.atomic.Value(u64)) = .init(null),
    /// The pass starts at `rows` (restored from a kept state); none computed yet. Takes any gate off.
        p.gate.store(null, .release);
    /// From here the rows count only once `finished` (generation `gen`) says the device is done with them.
    pub fn gated(p: *Progress, finished: *const std.atomic.Value(u64), gen: u32) void {
        p.gen.store(gen, .release);
        p.gate.store(finished, .release);
    }

        const issued = p.at.load(.acquire);
        const from = p.from.load(.monotonic);
        const g = p.gate.load(.acquire) orelse return .{ .from = from, .at = issued };
        const v = g.load(.acquire);
        // (another generation: a callback of an earlier fill, or the fill not armed yet: nothing finished)
        const done: u32 = if (@as(u32, @truncate(v >> 32)) == p.gen.load(.acquire)) @truncate(v) else 0;
        return .{ .from = from, .at = @min(issued, done) };
/// Drafts the backend holds for the stream's next round; a tree keeps its tokens and parents on the host.
pub const Held = struct {
    count: u32,
    tokens: ?[]u32 = null,
    parents: ?[]i32 = null,
};

/// An image in the prompt for a lane-vision family (its own image input, as the served lane renders images): the span
/// of `tokens` positions from `at` it fills (each position carries the image token id), and the payload its family's
/// LaneVision.prepare made, or with `rows` its span's embedding rows ready (a gate's input).
pub const LaneImage = struct {
    at: u32,
    tokens: u32,
    bytes: []const u8,
    rows: bool = false,
};

/// A lane-vision family's image input as a server renders it (engine_api.Info.lane_vision): each image part becomes
/// `placeholder` (one token: `token`) in the template's text, and `prepare` turns the image's bytes into its span's
/// length and the payload the backend reads (LaneImage.bytes), on the request's thread; on BadImage `problem` says why.
pub const LaneVision = struct {
    ctx: *anyopaque,
    placeholder: []const u8,
    token: u32,
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8, problem: *[]const u8) PrepareError!Prepared,

    pub const Prepared = struct { tokens: u32, payload: []const u8 };
    pub const PrepareError = error{ BadImage, OutOfMemory };
};

/// A stream's decode rounds by the plan its window ran (windows.Kind, as the round concluded): copied continuations
/// (`copy`) and the head's drafts (`head`), each the rounds, the plan's own drafts verified and those the target kept
/// (a grammar's drafts after them and suffix-match branches count in neither, as do forced, plain, grammar and fill
/// rounds).
pub const Plans = struct {
    copy: Count = .{},
    head: Count = .{},

    pub const Count = struct {
        rounds: u64 = 0,
        drafted: u64 = 0,
        accepted: u64 = 0,

        /// One round of this plan: its window's drafts and those kept.
        pub fn add(c: *Count, drafted: u64, accepted: u64) void {
            c.rounds += 1;
            c.drafted += drafted;
            c.accepted += accepted;
        }
    };
};

/// One image of a prompt: tower input (patches [gh * gw, width]) for the `tokens` placeholder rows from `at`.
pub const Image = struct { pixels: []const f32, gh: u32, gw: u32, at: u32, tokens: u32 };

pub const Spec = struct {
    id: []const u8,
    prompt: []const u32,
    max_new: u32,
    eos: []const u32 = &.{},
    sampling: ?Sampling = null,
    drafts: bool = true,
    proposer: ?Proposer = null,
    recall: ?*Reader = null, // copies and grafts from the record of earlier requests (lanes/recall.zig)
    stop_check: ?StopCheck = null,
    cancel_check: ?CancelCheck = null,
    yield_check: ?CancelCheck = null, // a background prompt pass: real work waits, so it keeps what it has and stops
    share_check: ?CancelCheck = null, // a stepped prompt pass: other work waits, so its step ends after its chunks
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: i64 = -1,
    think_open: ?bool = null, // null: open when a budget is set (the server's rule)
    loop_guard: bool = false,
    loop_windows: u32 = 0, // the guard's novelty signal: this many dry windows in a row close thinking (0: cycles only)
    chunks: []const u32 = &.{}, // where prefill chunks start after 0 (Python's PrefillPlan); empty: the backend's step
    history_len: u32 = 0, // prompt prefix lengths a backend may keep for a later turn: the rendered history,
    shared_prefixes: []const u32 = &.{}, // then shared system blocks
    reuse: Reuse = .{},
    decide: ?u32 = null, // the prompt row where living lessons decide whether they act on the reply
    untaught: bool = false, // no learned neuron acts on its rows: the model as shipped (Request.untaught)
    logprobs: ?u8 = null, // the target's log probabilities for each committed token, with this many best tokens
    prompt_logprobs: ?u8 = null, // and for each prompt token after the first, with ranks (the prompt pass starts at 0)
    grammar: ?*grammar.Constraint = null, // structured output: the reply stays in it (the creator frees it)
    decode_spans: []const [2]u32 = &.{}, // prompt rows an earlier reply decoded (Request.decode_spans), for backends that keep their bits
    images: []const Image = &.{}, // in prompt order; the backend's tower fills their rows
    key: []const u32 = &.{}, // the prompt as the cache matches it (image rows by their image); empty: the prompt
    lane_images: []const LaneImage = &.{}, // a lane-vision family's, in prompt order; the caller's until the stream is released
};

pub const Stream = struct {
    id: []const u8,
    prompt_len: usize,
    max_new: u32,
    eos: []const u32,
    sampling: ?Sampling,
    drafts: bool,
    proposer: ?Proposer,
    recall: ?*Reader,
    stop_check: ?StopCheck,
    cancel_check: ?CancelCheck,
    yield_check: ?CancelCheck,
    share_check: ?CancelCheck = null,
    think_budget: u32,
    think_close: []const u32,
    think_end: i64,
    think_open: bool,
    loop_guard: bool,
    loop_period: ?u32 = null,
    novelty: ?Novelty = null, // the guard's novelty signal (Spec.loop_windows)
    loop_novelty: ?f64 = null, // the dry window's share of new grams, when that signal closed the think block
    chunks: []const u32,
    history_len: u32,
    shared_prefixes: []const u32,
    logprobs: ?u8 = null,
    grammar: ?*grammar.Constraint = null, // follows each committed token; its masks keep the draws inside it
    problem: []const u8 = "", // why a stream finished with `error` (static text)
    rows: std.ArrayList(Row) = .empty, // with `logprobs`: one a token of emitted(), in order
    prompt_logprobs: ?u8 = null,
    prompt_rows: std.ArrayList(Row) = .empty, // with `prompt_logprobs`: one a prompt token after the first, in order
    reuse: Reuse = .{},
    decode_spans: []const [2]u32 = &.{},
    decide: ?u32 = null,
    untaught: bool = false,
    images: []const Image = &.{},
    between: ?BetweenHook = null, // the host's round between this stream's prompt chunks (a backend may call it)
    key: []const u32 = &.{},
    lane_images: []const LaneImage = &.{},
    cached: u32 = 0, // prompt tokens the backend restored from `reuse` (its prompt pass started there)
    fill_at: std.atomic.Value(u32) = .init(0), // prompt tokens the pass has prefilled; the engine thread writes
    reuse_failed: bool = false, // the backend's restore of `reuse` failed: it prefilled from 0
    settle_step: bool = false, // the lane host bills this prompt step's time to decoders: end it on the device
    step_pair: bool = false, // the backend's last prompt step ran as a pair of passes (the host owes the streams twice the rounds)
    context: std.ArrayList(u32) = .empty,
    pending: ?u32 = null,
    force: std.ArrayList(u32) = .empty,
    cache_len: u64 = 0,
    finished: bool = false,
    reason: Reason = .none,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    plans: Plans = .{}, // those rounds, drafts and kept drafts of copies and of the head's drafts (Plans)
    depths: [shape.max_depth]Depth = @splat(.{}), // the head's chains by depth (noteDepths)
    branch_rows: u64 = 0, // suffix-match branch rows verified (lanes/fill.zig)
    branch_accepted: u64 = 0, // of those, rows on the accepted path
    graft_hit: f64 = 1, // recent share of grafted rows kept; a read of the chain that grafted none counts as a miss
    recall_hit: f64 = 1, // the same for rows Recall's candidates grafted, kept apart
    odds: ?shape.Odds = null, // how often the target took the head's rank-r lane at each depth (head trees)
    lanes: ?shape.Shape = null, // the shape of the held tree drafts
    landed: [shape.max_depth][shape.ranks][2]u32 = @splat(@splat(.{ 0, 0 })), // tree lanes verified and taken
    picks: [plan_lanes.sizes.len + 1]u32 = @splat(0), // the planner's picks: each tree size, then chain + grafts
    graft_room: ?u32 = null, // grafted rows the planner left this round's window (null: no planner)
    held_levels: u32 = 0, // head levels drafted for the held window
    round_over: ?f64 = null, // the stream's round time beyond its window and head levels (ms, recent): chains
    tree_over: ?f64 = null, // and trees
    measured: [plan_lanes.sizes.len + 1]plan_lanes.Measured = @splat(.{}), // each pick's round times past its table price
    priced: ?struct { choice: usize, ms: f64 } = null, // the pick that shaped the held window, at its table price
    min_rows: u32 = 0,
    paused: bool = false,
    depth: ?State = null,
    mode: ?Mode = null,
    next: ?Held = null,
    inflight: ?u64 = null,
    served: ?u64 = null,
    granted: ?u32 = null,
    copy_width: ?u32 = null,

    pub fn init(gpa: Allocator, spec: Spec) !Stream {
        var s: Stream = .{
            .id = spec.id,
            .prompt_len = spec.prompt.len,
            .max_new = spec.max_new,
            .eos = spec.eos,
            .sampling = spec.sampling,
            .drafts = spec.drafts,
            .proposer = if (spec.drafts) spec.proposer else null,
            .recall = if (spec.drafts) spec.recall else null,
            .stop_check = spec.stop_check,
            .cancel_check = spec.cancel_check,
            .yield_check = spec.yield_check,
            .share_check = spec.share_check,
            .think_budget = spec.think_budget,
            .think_close = spec.think_close,
            .think_end = spec.think_end,
            .think_open = spec.think_open orelse (spec.think_budget > 0 or (spec.loop_guard and spec.think_end >= 0)),
            .loop_guard = spec.loop_guard,
            .novelty = if (spec.loop_guard and spec.loop_windows > 0) .{ .windows = spec.loop_windows } else null,
            .chunks = spec.chunks,
            .history_len = spec.history_len,
            .shared_prefixes = spec.shared_prefixes,
            .reuse = spec.reuse,
            .fill_at = .init(if (spec.reuse.saved != null) spec.reuse.at else 0), // a kept prefix is filled from the start
            .decide = spec.decide,
            .untaught = spec.untaught,
            .logprobs = spec.logprobs,
            .prompt_logprobs = spec.prompt_logprobs,
            .grammar = spec.grammar,
            .decode_spans = spec.decode_spans,
            .images = spec.images,
            .key = spec.key,
            .lane_images = spec.lane_images,
        };
        // under a grammar a forced close ends at the think end: the grammar takes the reply from the next token
        if (spec.grammar) |g| if (g.after) |end| if (std.mem.indexOfScalar(u32, s.think_close, end)) |at| {
            s.think_close = s.think_close[0 .. at + 1];
        };
        try s.context.appendSlice(gpa, spec.prompt);
        return s;
    }

    pub fn deinit(s: *Stream, gpa: Allocator) void {
        s.context.deinit(gpa);
        s.rows.deinit(gpa);
        s.prompt_rows.deinit(gpa);
        s.force.deinit(gpa);
        if (s.novelty) |*n| n.deinit(gpa);
        s.dropRoundState(gpa);
    }

    /// Forget the round loop's state (Python `_release_stream_state`).
    pub fn dropRoundState(s: *Stream, gpa: Allocator) void {
        if (s.depth) |*d| d.deinit(gpa);
        s.dropHeld(gpa);
        s.depth = null;
        s.mode = null;
        s.inflight = null;
        s.served = null;
        s.granted = null;
        s.copy_width = null;
    }

    pub fn dropHeld(s: *Stream, gpa: Allocator) void {
        if (s.next) |h| {
            if (h.tokens) |t| gpa.free(t);
            if (h.parents) |p| gpa.free(p);
        }
        s.next = null;
        if (s.lanes) |l| l.deinit(gpa);
        s.lanes = null;
    }

    pub fn prompt(s: *const Stream) []const u32 {
        return s.context.items[0..s.prompt_len];
    }

    pub fn emitted(s: *const Stream) []const u32 {
        return s.context.items[s.prompt_len..];
    }

    pub fn budgetLeft(s: *const Stream) i64 {
        return @as(i64, s.max_new) - @as(i64, @intCast(s.emitted().len));
    }

    /// The host's facts for this stream's next prompt step (null without a hook or its facts).
    pub fn hostFacts(s: *const Stream) ?Facts {
        const b = s.between orelse return null;
        const f = b.facts orelse return null;
        return f(b.ptr);
    }

    /// What the host runs after this stream's prompt step (null without a hook or its plan).
    pub fn hostNext(s: *const Stream) ?Next {
        const b = s.between orelse return null;
        const n = b.next orelse return null;
        return n(b.ptr);
    }

    /// Whether the request was cancelled (false with no hook); a prompt pass asks before each chunk.
    pub fn isCancelled(s: *const Stream) bool {
        const c = s.cancel_check orelse return false;
        return c.check(c.ptr);
    }

    /// Whether a background prompt pass should step aside now (false with no hook); asked before each chunk.
    pub fn yields(s: *const Stream) bool {
        const c = s.yield_check orelse return false;
        return c.check(c.ptr);
    }

    /// Whether a stepped prompt pass ends its step after its chunks (true with no hook: every step ends there);
    /// false while nothing else waits, so a pass alone runs on as a whole pass does. Asked after each chunk.
    pub fn shares(s: *const Stream) bool {
        const c = s.share_check orelse return true;
        return c.check(c.ptr);
    }

    fn budgetActive(s: *const Stream) bool {
        return s.think_open and s.think_budget > 0 and s.think_close.len > 0;
    }

    /// The index the thinking budget replaces with `think_close[0]`, or null if the block closed first.
    pub fn thinkCut(s: *const Stream, tokens: anytype) ?usize {
        if (!s.budgetActive()) return null;
        const done = s.emitted().len;
        for (tokens, 0..) |t, i| {
            if (done + i + 1 >= s.think_budget) return i;
            if (@as(i64, t) == s.think_end) return null;
        }
        return null;
    }

    /// The thinking budget's cut falls within the next `rows` tokens.
    pub fn budgetWithin(s: *const Stream, rows: usize) bool {
        return s.budgetActive() and s.emitted().len + rows >= s.think_budget;
    }

    /// Python `think_cut([-1]) == 0`: the next position is the budget's cut.
    pub fn cutsNext(s: *const Stream) bool {
        return s.budgetActive() and s.emitted().len + 1 >= s.think_budget;
    }

    /// Begin the thinking budget's close: its first token now, the rest forced after it.
    pub fn startClose(s: *Stream, gpa: Allocator) !u32 {
        s.think_open = false;
        s.force.clearRetainingCapacity();
        try s.force.appendSlice(gpa, s.think_close[1..]);
        return s.think_close[0];
    }

    /// Tokens a round may commit before the length limit or the thinking budget's cut.
    pub fn draftRoom(s: *const Stream) i64 {
        var room = s.budgetLeft();
        if (s.budgetActive()) room = @min(room, @as(i64, s.think_budget) - @as(i64, @intCast(s.emitted().len)));
        return room;
    }

    pub fn isEos(s: *const Stream, token: u32) bool {
        return std.mem.indexOfScalar(u32, s.eos, token) != null;
    }

    /// One head chain of `drafted` drafts, the target keeping its first `kept`, counted at each depth it drafted (a
    /// draft deeper than the counts counts at the last).
    pub fn noteDepths(s: *Stream, drafted: usize, kept: usize) void {
        for (0..drafted) |j| {
            const d = &s.depths[@min(j, s.depths.len - 1)];
            d.drafted += 1;
            d.reached += @intFromBool(kept >= j);
            d.kept += @intFromBool(kept > j);
        }
    }

    /// Each held tree lane whose parent row the round reached: the target took it or not (its depth and rank).
    pub fn observeLanes(s: *Stream, l: *const shape.Shape, path: []const u32) void {
        var on: [256]bool = @splat(false);
        for (path) |r| if (r < on.len) {
            on[r] = true;
        };
        for (l.parents, l.depths, l.ranks, 0..) |par, d, r, i| {
            const parent_row: usize = if (par < 0) 0 else 1 + @as(usize, @intCast(par));
            if (!on[parent_row]) continue;
            const took = on[1 + i];
            s.odds.?.observe(d, r, took);
            if (d < shape.max_depth and r < shape.ranks) {
                s.landed[d][r][0] += 1;
                s.landed[d][r][1] += @intFromBool(took);
            }
        }
    }

    /// Append tokens, and with `logprobs` their rows, until the stream finishes; how many landed.
    pub fn commit(s: *Stream, gpa: Allocator, tokens: []const u32, rows: []const Row) !usize {
        if (rows.len != if (s.logprobs != null) tokens.len else 0) return error.LogprobRowsMismatch;
        var landed: usize = 0;
        var ended = false; // a novelty window ended in this batch: only its first end may fire, as loopCut reads it
        for (tokens, 0..) |t, i| {
            if (s.finished) break;
            try s.context.append(gpa, t);
            if (rows.len > 0) try s.rows.append(gpa, rows[i]);
            landed += 1;
            if (@as(i64, t) == s.think_end) s.think_open = false;
            if (s.grammar) |g| {
                if (!try g.follow(t)) {
                    s.finished = true;
                    s.reason = .@"error";
                    s.problem = "the reply's grammar rejected a chosen token";
                    break;
                }
                if (g.finished() and !s.isEos(t)) { // the grammar's stop token ends the reply (eos ignored or not)
                    s.finished = true;
                    s.reason = .stop;
                    break;
                }
            }
            const watch = s.guarding();
            const period = if (watch) detectLoop(s.emitted()) else null;
            const dry = if (watch and period == null) try s.dryWindow(gpa, &ended) else null;
            if (s.isEos(t) or s.stopped()) {
                s.finished = true;
                s.reason = .stop;
            } else if (period != null or dry != null) {
                s.loop_period = period;
                s.loop_novelty = dry;
                if (s.emitted().len >= s.max_new) {
                    s.finished = true;
                    s.reason = .length;
                } else if (s.think_close.len == 0) {
                    s.finished = true;
                    s.reason = .stop;
                } else {
                    s.think_open = false;
                    s.force.clearRetainingCapacity();
                    try s.force.appendSlice(gpa, s.think_close);
                }
                break;
            } else if (s.emitted().len >= s.max_new) {
                s.finished = true;
                s.reason = .length;
            }
        }
        return landed;
    }

    fn stopped(s: *const Stream) bool {
        const c = s.stop_check orelse return false;
        return c.check(c.ptr, s.emitted());
    }

    /// The loop guard watches the open think block until one of its signals fires.
    fn guarding(s: *const Stream) bool {
        return s.loop_guard and s.think_open and s.think_end >= 0 and s.loop_period == null and s.loop_novelty == null;
    }

    /// The novelty signal after the newest token: the share of new grams of the window it ends, when that window fires
    /// and is the batch's first to end (`ended`).
    fn dryWindow(s: *Stream, gpa: Allocator, ended: *bool) !?f64 {
        if (s.novelty == null) return null;
        const n = &s.novelty.?;
        const new = try n.observe(gpa, s.emitted()) orelse return null;
        defer ended.* = true;
        return if (!ended.* and n.fires()) new else null;
    }

    pub fn popForce(s: *Stream) ?u32 {
        if (s.force.items.len == 0) return null;
        return s.force.orderedRemove(0);
    }
};

test "commit stops at eos and length" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9} });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }, &.{}));
    try std.testing.expectEqualSlices(u32, &.{ 5, 9 }, s.emitted());
    try std.testing.expect(s.finished and s.reason == .stop);
}

test "a head chain counts at each depth it drafted: reached while the drafts above it were kept, kept with its own" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9} });
    defer s.deinit(gpa);
    s.noteDepths(3, 1); // the second draft missed: the third was drafted, never reached
    s.noteDepths(2, 2); // both kept
    s.noteDepths(4, 0); // the first missed
    const want = [_]Depth{ .{ .drafted = 3, .reached = 3, .kept = 2 }, .{ .drafted = 3, .reached = 2, .kept = 1 }, .{ .drafted = 2, .reached = 0, .kept = 0 }, .{ .drafted = 1, .reached = 0, .kept = 0 } };
    try std.testing.expectEqualSlices(Depth, &want, s.depths[0..4]);
    try std.testing.expectEqual(Depth{}, s.depths[4]);
    // a chain deeper than the counts adds its deeper drafts at the last depth
    s.noteDepths(shape.max_depth + 2, shape.max_depth + 2);
    try std.testing.expectEqual(Depth{ .drafted = 3, .reached = 3, .kept = 3 }, s.depths[shape.max_depth - 1]);
}

test "a logprob stream lands a row with each token it lands, and refuses tokens without rows" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9}, .logprobs = 0 });
    defer s.deinit(gpa);
    try std.testing.expectError(error.LogprobRowsMismatch, s.commit(gpa, &.{5}, &.{}));
    const rows = [_]Row{ .{ .token = 5, .logprob = -0.5 }, .{ .token = 9, .logprob = -1 }, .{ .token = 7, .logprob = -2 } };
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }, &rows));
    try std.testing.expectEqual(@as(usize, 2), s.rows.items.len);
    try std.testing.expectEqual(@as(u32, 9), s.rows.items[1].token);
}

test "loop guard stops at the cap even when no cycle fires" {
    const gpa = std.testing.allocator;
    var tokens: [20]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i);
    var guarded = try Stream.init(gpa, .{ .id = "guarded", .prompt = &.{1}, .max_new = 10, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer guarded.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 10), try guarded.commit(gpa, &tokens, &.{}));
    try std.testing.expect(guarded.finished and guarded.reason == .length);
}

test "loop guard fires, closes thinking, and latches once" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..257) |_| try tokens.append(gpa, 7);
    var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 400, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items, &.{}));
    try std.testing.expectEqual(@as(u32, 1), s.loop_period.?);
    try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
    while (s.popForce()) |token| _ = try s.commit(gpa, &.{token}, &.{});
    try std.testing.expect(!s.finished and s.force.items.len == 0);
    _ = try s.commit(gpa, &.{8}, &.{});
    try std.testing.expect(!s.finished and s.loop_period.? == 1);
}


test "passRows gives resume rows below pass_from, while a request waits, while another prompt fills and with no hook, else pass rows when nothing decodes or step_rows reaches them" {
    const quiet: Facts = .{ .decoding = false, .filling = false, .waiting = false, .step_rows = std.math.maxInt(u32) };
    // no hook: the resume rows
    try std.testing.expectEqual(@as(u32, 2048), passRows(null, 16384, 8192, 4096, 2048, false));
    // nothing decodes, fills or waits: the pass rows, from pass_from on
    try std.testing.expectEqual(@as(u32, 4096), passRows(quiet, 16384, 8192, 4096, 2048, false));
    try std.testing.expectEqual(@as(u32, 4096), passRows(quiet, 8192, 8192, 4096, 2048, false));
    try std.testing.expectEqual(@as(u32, 2048), passRows(quiet, 6144, 8192, 4096, 2048, false));
    try std.testing.expectEqual(@as(u32, 4096), passRows(quiet, 0, 0, 4096, 2048, false));
    // another request waits, or another prompt fills
    var f = quiet;
    f.waiting = true;
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 16384, 8192, 4096, 2048, false));
    f = quiet;
    f.filling = true;
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 16384, 8192, 4096, 2048, false));
    // a stream decodes: the resume rows at the default cadence's step rows, the pass rows once a plan's step rows reach them
    f = quiet;
    f.decoding = true;
    f.step_rows = 2048;
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 16384, 8192, 4096, 2048, false));
    f.step_rows = 4096;
    try std.testing.expectEqual(@as(u32, 4096), passRows(f, 16384, 8192, 4096, 2048, false));
    f.step_rows = 6144;
    try std.testing.expectEqual(@as(u32, 4096), passRows(f, 16384, 8192, 4096, 2048, false));
    // pass rows equal to the resume rows: always the resume rows
    try std.testing.expectEqual(@as(u32, 2048), passRows(quiet, 16384, 8192, 2048, 2048, false));
}

test "passRows with TF_DS_PASS_FILLING: another prompt filling keeps the pass rows, while a waiting request or a decoding stream keeps the resume rows" {
    const quiet: Facts = .{ .decoding = false, .filling = false, .waiting = false, .step_rows = std.math.maxInt(u32) };
    // another prompt fills and nothing else: the baseline cuts to the resume rows; the switch keeps the pass rows
    var f = quiet;
    f.filling = true;
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 16384, 8192, 4096, 2048, false));
    try std.testing.expectEqual(@as(u32, 4096), passRows(f, 16384, 8192, 4096, 2048, true));
    // below pass_from, or with no facts: the resume rows either way
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 6144, 8192, 4096, 2048, true));
    try std.testing.expectEqual(@as(u32, 2048), passRows(null, 16384, 8192, 4096, 2048, true));
    // a request waits: the resume rows, as in the baseline, whatever the switch says
    f = quiet;
    f.waiting = true;
    f.filling = true;
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 16384, 8192, 4096, 2048, true));
    // a stream decodes: the switch follows the host's step rows as the plain rule does: the resume rows at the default
    // cadence's step rows, the pass rows once the host's step rows reach them
    f = quiet;
    f.decoding = true;
    f.step_rows = 2048;
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 16384, 8192, 4096, 2048, true));
    f.step_rows = 4096;
    try std.testing.expectEqual(@as(u32, 4096), passRows(f, 16384, 8192, 4096, 2048, false));
    try std.testing.expectEqual(@as(u32, 4096), passRows(f, 16384, 8192, 4096, 2048, true));
    // ... and a waiting request still cuts the pass, whatever the host's step rows are
    f.waiting = true;
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 16384, 8192, 4096, 2048, true));
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 16384, 8192, 4096, 2048, false));
    // nothing else: the pass rows
    try std.testing.expectEqual(@as(u32, 4096), passRows(quiet, 16384, 8192, 4096, 2048, true));
}

test "passRows over every combination: the switch changes a pass only for another prompt filling, and the host's step rows decide a decoding stream's pass alike with it" {
    const rows = [_]u32{ 0, 2048, 4096, std.math.maxInt(u32) };
    for ([_]bool{ false, true }) |decoding| for ([_]bool{ false, true }) |filling| for ([_]bool{ false, true }) |waiting| for (rows) |step_rows| {
        const f: Facts = .{ .decoding = decoding, .filling = filling, .waiting = waiting, .step_rows = step_rows };
        const plain = passRows(f, 16384, 8192, 4096, 2048, false);
        const switched = passRows(f, 16384, 8192, 4096, 2048, true);
        if (waiting) {
            try std.testing.expectEqual(@as(u32, 2048), plain);
            try std.testing.expectEqual(@as(u32, 2048), switched);
            continue;
        }
        const decoding_rows: u32 = if (step_rows >= 4096) 4096 else 2048;
        // a stream decoding: the host's step rows say, with the switch on or off; another prompt filling cuts only the plain rule
        if (decoding) try std.testing.expectEqual(decoding_rows, switched);
        if (decoding and !filling) try std.testing.expectEqual(decoding_rows, plain);
        if (!decoding) try std.testing.expectEqual(@as(u32, 4096), switched);
        if (!decoding and !filling) try std.testing.expectEqual(@as(u32, 4096), plain);
        if (!decoding and filling) try std.testing.expectEqual(@as(u32, 2048), plain);
        if (decoding and filling) try std.testing.expectEqual(@as(u32, 2048), plain);
    };
}

test "wholePass: the facts of a pass at the booted rows, from pass_from on, with nothing decoding; the rows alone cannot tell it from a cut pass" {
    const quiet: Facts = .{ .decoding = false, .filling = false, .waiting = false, .step_rows = std.math.maxInt(u32) };
    try std.testing.expect(wholePass(quiet, 8192, 8192, false, 8192));
    try std.testing.expect(wholePass(quiet, 0, 0, false, 8192));
    try std.testing.expect(!wholePass(quiet, 6144, 8192, false, 8192)); // below pass_from
    try std.testing.expect(!wholePass(null, 16384, 8192, false, 8192)); // a host without facts
    var f = quiet;
    f.waiting = true;
    try std.testing.expect(!wholePass(f, 16384, 8192, false, 8192));
    try std.testing.expect(!wholePass(f, 16384, 8192, true, 8192)); // a waiting request cuts the pass whatever the filling switch says
    f = quiet;
    f.filling = true;
    try std.testing.expect(!wholePass(f, 16384, 8192, false, 8192));
    try std.testing.expect(wholePass(f, 16384, 8192, true, 8192));
    f = quiet;
    f.decoding = true;
    f.step_rows = std.math.maxInt(u32); // (step rows that reach the pass: the pass is whole, a pair still is not without a count of rounds)
    try std.testing.expect(!wholePass(f, 16384, 8192, true, 8192));
    // a count of rounds (TF_DS_ROUNDS_BESIDE) makes it whole beside a decoding stream, when the step rows reach the booted pass rows
    f.rounds_beside = 2;
    try std.testing.expect(wholePass(f, 16384, 8192, true, 8192));
    f.step_rows = 4096;
    try std.testing.expect(!wholePass(f, 16384, 8192, true, 8192)); // (a step of 4,096 under passes of 8,192 is half a pass)
    try std.testing.expect(wholePass(f, 16384, 8192, true, 4096));
    f.waiting = true;
    try std.testing.expect(!wholePass(f, 16384, 8192, true, 4096));
    // the rule's cut at every pass size (2,048 included, where the rows are the same cut or whole): wholePass is the rule's whole
    // pass with nothing decoding, for pass rows above the resume rows, over every combination of facts
    for ([_]u32{ 4096, 6144, 8192, 10240, 12288, 14336, 16384 }) |p| for ([_]bool{ false, true }) |decoding| for ([_]bool{ false, true }) |filling| for ([_]bool{ false, true }) |waiting| for ([_]bool{ false, true }) |fp| for ([_]usize{ 0, 4096, 8192, 16384 }) |start| for ([_]u32{ 2048, 4096, 8192, std.math.maxInt(u32) }) |step_rows| {
        const g: Facts = .{ .decoding = decoding, .filling = filling, .waiting = waiting, .step_rows = step_rows };
        const rule_whole = passRows(g, start, 8192, p, 2048, fp) == p;
        try std.testing.expectEqual(rule_whole and !decoding, wholePass(g, start, 8192, fp, p));
        // with a count of rounds the rule's whole pass is the pass, decoding or not
        var h = g;
        h.rounds_beside = 3;
        try std.testing.expectEqual(rule_whole, wholePass(h, start, 8192, fp, p));
    };
}

test "a stream asks its host's hook for the facts and the plan, and has neither without one" {
    const H = struct {
        fn run(_: *anyopaque) anyerror!void {}
        fn ready(_: *anyopaque) bool {
            return false;
        }
        fn facts(_: *anyopaque) Facts {
            return .{ .decoding = true, .filling = false, .waiting = true, .step_rows = 2048 };
        }
        fn next(_: *anyopaque) Next {
            return .other;
        }
    };
    var s = try Stream.init(std.testing.allocator, .{ .id = "s", .prompt = &.{ 1, 2, 3 }, .max_new = 1, .drafts = false });
    defer s.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?Facts, null), s.hostFacts());
    try std.testing.expectEqual(@as(?Next, null), s.hostNext());
    var token: u8 = 0;
    s.between = .{ .ptr = &token, .run = H.run, .ready = H.ready };
    try std.testing.expectEqual(@as(?Facts, null), s.hostFacts());
    try std.testing.expectEqual(@as(?Next, null), s.hostNext());
    s.between = .{ .ptr = &token, .run = H.run, .ready = H.ready, .facts = H.facts, .next = H.next };
    try std.testing.expectEqual(@as(?Facts, H.facts(&token)), s.hostFacts());
    try std.testing.expectEqual(@as(?Next, .other), s.hostNext());
}

test "a step beside a decoding stream is the host's step rows in whole resume units, at most the pass rows, at least one unit" {
    const quiet: Facts = .{ .decoding = false, .filling = false, .waiting = false, .step_rows = std.math.maxInt(u32) };
    for ([_]u32{ 2048, 4096, 6144, 8192, 12288, 16384 }) |pass| for ([_]u32{ 2048, 4096, 6144, 8192, 16384, 20480 }) |beside| {
        var f = quiet;
        f.decoding = true;
        f.step_rows = beside;
        const want = @min(pass, beside);
        try std.testing.expectEqual(want, passRows(f, 16384, 8192, pass, 2048, false));
        try std.testing.expectEqual(want, passRows(f, 16384, 8192, pass, 2048, true));
        // nothing decoding: the pass, whatever the setting; a waiting request: one unit
        try std.testing.expectEqual(pass, passRows(quiet, 16384, 8192, pass, 2048, false));
        f.waiting = true;
        try std.testing.expectEqual(@as(u32, 2048), passRows(f, 16384, 8192, pass, 2048, false));
    };
    // a step between two sizes runs the lower; under one unit still one unit
    var f = quiet;
    f.decoding = true;
    f.step_rows = 5000;
    try std.testing.expectEqual(@as(u32, 4096), passRows(f, 0, 0, 8192, 2048, false));
    f.step_rows = 100;
    try std.testing.expectEqual(@as(u32, 2048), passRows(f, 0, 0, 8192, 2048, false));
    try std.testing.expectEqual(@as(u32, 6144), besideRows(6144, 8192, 2048));
}

const reference = @import("novelty.zig").reference;

/// Commit `tokens` `batch` at a time as the lane core's rounds do, cut where loopCut cuts, until the guard fires or the
/// stream finishes; how many landed. With rounds of 7 or 50 a window ends inside a round: only loopCut ends it there.
fn rounds(gpa: Allocator, s: *Stream, tokens: []const u32, batch: usize) !usize {
    var at: usize = 0;
    while (at < tokens.len and !s.finished and s.force.items.len == 0) {
        var next = tokens[at..@min(tokens.len, at + batch)];
        if (loopCut(s, next)) |cut| next = next[0 .. cut + 1];
        const landed = try s.commit(gpa, next, &.{});
        at += landed;
        if (landed < next.len) {
            try std.testing.expect(s.finished); // a guard that fires mid-batch fires where loopCut cut
            break;
        }
    }
    return at;
}

const Fire = struct { at: usize, share: f64 };

/// Where the novelty signal fires on `tokens` (the reply index of its last token) and the window's share of new grams.
fn novelFire(gpa: Allocator, tokens: []const u32, windows: u32, batch: usize) !?Fire {
    var s = try Stream.init(gpa, .{ .id = "novel", .prompt = &.{1}, .max_new = 1 << 20, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true, .loop_windows = windows });
    defer s.deinit(gpa);
    const landed = try rounds(gpa, &s, tokens, batch);
    try std.testing.expect(s.loop_period == null and !s.finished);
    const share = s.loop_novelty orelse {
        try std.testing.expectEqual(tokens.len, landed);
        return null;
    };
    try std.testing.expect(!s.think_open and std.mem.eql(u32, &.{ 90, 91, 92 }, s.force.items));
    return .{ .at = landed - 1, .share = share };
}

test "the novelty signal fires where the Python guard does, a token or a round at a time" {
    const gpa = std.testing.allocator;
    var a: std.ArrayList(u32) = .empty;
    defer a.deinit(gpa);
    try reference.fresh(gpa, &a, 0, 2500);
    try reference.loop13(gpa, &a, 20000);
    var b: std.ArrayList(u32) = .empty;
    defer b.deinit(gpa);
    try reference.fresh(gpa, &b, 0, 2048);
    try reference.loop13(gpa, &b, 5 * 1024);
    try reference.fresh(gpa, &b, 10000, 20 * 1024);
    var c: std.ArrayList(u32) = .empty; // the think block closes before the loop's fifth window ends
    defer c.deinit(gpa);
    try reference.fresh(gpa, &c, 0, 2500);
    try reference.loop13(gpa, &c, 3000);
    try c.append(gpa, 91);
    try reference.loop13(gpa, &c, 20000);
    var d: std.ArrayList(u32) = .empty;
    defer d.deinit(gpa);
    try reference.lcg(gpa, &d, 60 * 1024);
    const cases = [_]struct { tokens: []const u32, windows: u32, want: ?Fire }{
        .{ .tokens = a.items, .windows = 3, .want = .{ .at = 6143, .share = 0 } },
        .{ .tokens = a.items, .windows = 10, .want = .{ .at = 13311, .share = 0 } },
        .{ .tokens = b.items, .windows = 3, .want = .{ .at = 6143, .share = 0 } },
        .{ .tokens = b.items, .windows = 10, .want = null },
        .{ .tokens = c.items, .windows = 3, .want = null },
        .{ .tokens = c.items, .windows = 10, .want = null },
        .{ .tokens = d.items, .windows = 3, .want = .{ .at = 30719, .share = 5.0 / 1024.0 } },
        .{ .tokens = d.items, .windows = 10, .want = .{ .at = 37887, .share = 2.0 / 1024.0 } },
        .{ .tokens = a.items, .windows = 0, .want = null }, // the exact-cycle check alone: a loop of 13 is no cycle
    };
    for (cases) |case| for ([_]usize{ 1, 7, 50 }) |batch| {
        try std.testing.expectEqual(case.want, try novelFire(gpa, case.tokens, case.windows, batch));
    };
}

test "the novelty signal closes thinking, reports the window's share, and latches once" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    try reference.fresh(gpa, &tokens, 0, 1024);
    try reference.loop13(gpa, &tokens, 4 * 1024);
    const spec: Spec = .{ .id = "templated", .prompt = &.{1}, .max_new = 8000, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true, .loop_windows = 1 };
    var s = try Stream.init(gpa, spec);
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3 * 1024), try rounds(gpa, &s, tokens.items, 50));
    try std.testing.expect(s.loop_period == null and s.loop_novelty.? == 0);
    try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
    while (s.popForce()) |token| _ = try s.commit(gpa, &.{token}, &.{});
    try std.testing.expectEqual(tokens.items.len - 3 * 1024, try rounds(gpa, &s, tokens.items[3 * 1024 ..], 50));
    try std.testing.expect(!s.finished and s.loop_novelty.? == 0 and s.force.items.len == 0);

    // one batch: only its first window end may fire, as loopCut reads it, and that window is fresh
    var whole = try Stream.init(gpa, spec);
    defer whole.deinit(gpa);
    try std.testing.expectEqual(@as(?usize, null), loopCut(&whole, tokens.items));
    try std.testing.expectEqual(tokens.items.len, try whole.commit(gpa, tokens.items, &.{}));
    try std.testing.expect(whole.loop_novelty == null and whole.think_open);
}

test "shares: a stepped pass ends its step with no hook, runs on while its hook says nothing waits" {
    const Flag = struct {
        fn check(ptr: *anyopaque) bool {
            return @as(*bool, @ptrCast(@alignCast(ptr))).*;
        }
    };
    var waits = false;
    var s: Stream = undefined;
    s.share_check = null;
    try std.testing.expect(s.shares());
    s.share_check = .{ .ptr = &waits, .check = Flag.check };
    try std.testing.expect(!s.shares());
    waits = true;
    try std.testing.expect(s.shares());
}
