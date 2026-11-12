//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
//!
//! Pending durable restores and continuation telemetry port wojo's GLM 0110 (PR #124; the scheduler hooks) and 0111
//! (PR #128). A request may be parked before its prompt pass on a restore job reading a longer kept state from the
//! prompt cache's tier (shared by every request that wants the same point). Parked requests hold no lane, count as
//! waiting work and wake back to the front of the queue, where an ordinary lookup runs again. A restore ends only once
//! every rank holds its state, so the woken request's lookup resumes a state each rank has. With no tier nothing is
//! parked or serviced, and nothing else changes.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const share_mod = @import("lane_host_share.zig");
const live_config = @import("live_config.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Status = api.Status;
const Memory = api.Memory;
const Reason = api.Reason;
const Stats = api.Stats;
const SubmitError = api.SubmitError;

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(Id) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,
    engine_tid: std.atomic.Value(u64) = .init(0), // the engine thread's id once it runs (status asked from it must not wait for itself)
    /// A reader (status) wants the live counters, cache statistics and spill state brought up to date: the engine thread builds them (about
    /// a microsecond or two, and the cache statistics' ~80 counters copied twice) at its next turn only when asked, not every turn.
    tele_want: std.atomic.Value(bool) = .init(false),
    tele_done: std.Io.Condition = .init,
    tele_gen: u64 = 0, // refreshes made (under the lock)
    tele_builds: u64 = 0, // times the engine thread built the counters (a test reads it)
    decoded: std.ArrayList(Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    decoded_pending: usize = 0, // tokens sent since the last mark (engine thread)
    overlap_ran: bool = false, // the backend ran `overlapWork` in the round that just ended
    overlap: bool = false, // the backend runs `overlapWork` inside each round's forward wait (Backend.overlap): the round's tokens go out there
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_tokens: std.ArrayList(u32) = .empty,
    /// A Metal engine's keepalive target, set by the family that owns the queue; null keeps the ticker off.
    keepalive_target: ?api.keepalive.Target = null,
    live_generated: u64 = 0,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    memory: ?api.MemorySource = null, // the backend's memory counts; null: Engine.memory reports none
    graph_fallbacks: ?*const std.atomic.Value(u64) = null, // the backend's failed graph captures (read at any time)
    head_rounds: ?*const [6]std.atomic.Value(u64) = null, // the backend's decode rounds by the head's path (read at any time)
    draft_vocab: ?*const api.DraftVocabCounters = null, // the backend's drafter on a pruned vocabulary (read at any time)
    explain: ?api.Explain = null, // the backend's words for a request it refuses; null: the error's name
    // the request at the queue's front waits for room in the backend's pool (its prompt pass refused ContextFull while
    // other streams ran): admitted again once fewer than this many streams are active, or at full_retry (a backend
    // may refuse for a while to keep its kept states, waiting for one stream, and then place it)
    full_wait: ?usize = null,
    full_retry: i96 = 0,
    // after a long prompt pass (the whole pass runs inside its admission, no round meanwhile), the streams within
    // finish_left tokens of their max_tokens decode before the next admission, finish_ns at most: such a stream ends
    // and frees its extent (its kept state then resumes in place) instead of waiting out every queued pass, as the
    // Python lane's fills leave decoding its share
    finish_until: ?i96 = null,
    // Track D: a round run between an admitting job's prompt chunks failed (its error): once the prompt's pass has
    // unwound, every admitted job fails with it, as after any failed round
    between_failed: ?[]const u8 = null,
    fill_next: usize = 0, // next admitted prompt to advance by one chunk
    /// Prompt steps' share of wall time beside decoding streams (TENSORFOLD_PREFILL_SHARE; lane_host_share.zig): with
    /// a share below 1, the rounds that repay a long prompt's step run before its next one. 1: one step, one round.
    share: share_mod.Share = .{},
    /// The clock the share times its steps and rounds with; null: the host's. A test's.
    share_clock: ?ShareClock = null,
    fill_slots: std.ArrayList(api.fill_order.Slot) = .empty, // the fill order's reading of the admitted prompts (engine thread)
    /// TF_DS_PROMPT_TRACE=1: one log line per prompt step the host serves or holds back, with the order's reading of every
    /// admitted prompt, so a prompt that waited can be read off the engine log. Off: nothing is read or written.
    trace_fill: bool = false,
    /// The block of live settings this host takes from once a loop turn (null: none), the version it has taken, and the rounds beside
    /// the start gave (what `unset` returns to).
    live: ?*live_config.Block = null,
    live_seen: u64 = 0,
    /// What the start gave the rows a live change may replace (what `unset` returns to).
    boot: Boot = .{},
    /// What every rank was last told for each slot of class A (`unset`: nothing yet, the start's): a cfg frame goes out only when a
    /// slot's published value differs.
    live_sent: [live_config.slot_count]u64 = @splat(live_config.unset),
    // -- the service fence --------------------------------------------------------------------------------------------
    /// The backend's word that its device is between launches it may interleave with (dsv41: no CUDA graph capture
    /// on its stream); null: always.
    quiet: ?Quiet = null,
    filling_job: ?*Job = null, // engine thread: the job whose prompt runs now (its addStream, or one chunk of it)
    rounds_open: u32 = 0, // engine thread: inside the lane core's step (a round's launches)
    in_between: bool = false, // engine thread: inside Stream.between (the filling pass stands between two chunks)
    /// A service asked for where it must not run panics (debug builds), else is skipped and counted.
    fence_panic: bool = @import("builtin").mode == .debug,
    service_refused: u64 = 0,
    // -- parked requests (0110). The lists are the engine thread's; changes take the lock (status reads their lengths).
    ready: std.ArrayList(*Job) = .empty, // woken: admitted before the queue, in the order they woke
    loading: std.ArrayList(Load) = .empty, // each waits for a restore job (several may share one)
    turn: usize = 0, // which restore a service polls first (one budget, shared fairly)
    tier: TierPolicy = .{},
    host_counts: HostCounts = .{},
    /// What status reports under `cache` (published on the engine thread at each noteLive; null until configured).
    tele: ?api.CacheStats = null,
    /// Each round timed into times.round (configure: only with a tier bound, so a host without one reads no clock).
    time_rounds: bool = false,
    // -- restores the tier refused for capacity, asked again shortly (engine thread's; changes take the lock) ---------
    deferred: std.ArrayList(Defer) = .empty,
    last_restore_end: ?i96 = null, // when a restore last ended: every waiter's deferral clock counts from it
    // -- the disk tier's state for status (Status.spill) --------------------------------------------------------------
    spill_asked: bool = false, // the tier was asked for (Spill.asked)
    spill_off: ?api.SpillState = null, // off for the process from the start: the settings, the open, configure's refusal
    spill: ?api.SpillState = null, // what status reports (under the lock; refreshed on the engine thread at each noteLive)
    /// The host's calls into the store and its tier, timed with the host's clock (engine thread; published with the
    /// cache's counters): lookup (a park's probe), keep, service (each whole service), pump, poll (each restore's).
    times: pc.TierTimes = .{},

    const Mark = struct { at: i96, tokens: u64 };
    const full_retry_ns: i96 = std.time.ns_per_s;
    const long_pass_ns: i96 = std.time.ns_per_s;
    const finish_ns: i96 = 5 * std.time.ns_per_s;
    const finish_left: i64 = 256;
    const window_ns: i96 = 2 * std.time.ns_per_s;
    /// Prompt tokens a prompt step indexes for the copy proposer (SuffixLookup.warm): ~1 ms of this thread against a step of 100+ ms on the GPU.
    const warm_tokens: usize = 8192;
    /// The longest a status waits for the engine thread's refresh (ms): a long round, never the request path.
    const tele_wait_max_ms: u32 = 300;

    /// The restores one service polls at most (a fixed buffer; the tier holds TF_DS_SPILL_RESTORES in flight, at most
    /// 16 by the parser's range).
    const service_restores = 64;

    const Load = struct { job: *Job, restore: *pc.Restore, since: i96 };
    const Defer = struct { job: *Job, retry_at: i96 };

    /// A clock for the share's timing: awake nanoseconds.
    pub const ShareClock = struct { ctx: *anyopaque, now: *const fn (ctx: *anyopaque) i96 };

    /// The backend's device state for the service fence.
    pub const Quiet = struct { ctx: *anyopaque, check: *const fn (ctx: *anyopaque) bool };

    /// The failure a request cut by the stop ends with (not its client's cancel): an error the client sends again,
    /// never a reply that looks finished (HTTP 500, or a stream's error event).
    pub const stop_message = "the server is stopping: send the request again";

    /// How parked work is serviced: every number is a setting, read once by engine_api.Spill.fromEnv and taken as
    /// given by `configure`; these defaults are that parser's (a test holds them equal).
    pub const TierPolicy = struct {
        /// One elapsed-time budget per service for every restore together, then staged writes (TF_SPILL_LOAD_MS).
        service_ns: u64 = 40 * std.time.ns_per_ms,
        /// Staging's share of a service while streams decode (GLM pumped 4 blocks a round); idle, all that is left.
        pump_ns: u64 = 2 * std.time.ns_per_ms,
        /// How long a request waits for its restore in all, deferred and then parked on it, before it is admitted
        /// without it (TF_SPILL_LOAD_WAIT_MS).
        load_wait_ns: i96 = 30 * std.time.ns_per_s,
        /// A clean shutdown's bound for draining writes (TF_SPILL_FLUSH_MS).
        flush_ns: u64 = 15 * std.time.ns_per_s,
        /// A restore the tier refused for capacity (Store.deferrable) is asked again every defer_retry_ns, and when a
        /// restore in flight ends, for at most defer_ns with no restore ending (TF_SPILL_DEFER_MS; its clock restarts
        /// when one ends; 0: never, admitted on memory at once) and load_wait_ns in all, when it would save at least
        /// defer_min tokens over memory (0: the store's grid, or 1).
        defer_ns: i96 = 5 * std.time.ns_per_s,
        defer_retry_ns: i96 = 25 * std.time.ns_per_ms, // TF_SPILL_DEFER_RETRY_MS
        defer_min: u32 = 0, // TF_SPILL_DEFER_MIN
        /// How often an idle host polls work in flight: restores, staged writes, deferred retries (TF_SPILL_POLL_MS).
        poll_ns: u64 = 2 * std.time.ns_per_ms,
    };

    pub const Boot = struct {
        tier: TierPolicy = .{},
        share_p: f64 = 1,
        beside_rows: u32 = 0,
        fill: api.fill_order.Policy = .{},
    };

    pub const HostCounts = struct {
        restore_waits: u64 = 0, // requests parked on a restore
        restore_wait_timeouts: u64 = 0, // of them admitted without it at load_wait_ns
        restore_wait_ns: u64 = 0,
        restore_wait_max_ns: u64 = 0,
        cached_tokens: u64 = 0, // prompt tokens passes resumed (memory or restored)
        computed_tokens: u64 = 0, // and computed
        restores_deferred: u64 = 0, // requests whose restore was refused for capacity and asked again
        restores_deferred_expired: u64 = 0, // of them, admitted on memory (defer_ns with no restore ending, or load_wait_ns)
    };

    const Job = struct {
        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        delivered: usize = 0,
        started: bool = false,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        began: i96 = 0,
        rows_est: u64 = 0, // the rows its pass computes, as the cache stood when it was admitted (the fill order, before its pass starts)
        overtaken: u32 = 0, // prompt steps that went to a smaller prompt since this one's last step (the fill order's bound)
        stepped_at: i96 = 0, // when its last prompt step began, or when it was admitted (the fill order's maximum wait)
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up
        images: pc.Images = .none, // the images key of the prompt's prefixes (0 for a prompt without images)
        tried_disk: bool = false, // it asked the disk once: a later lookup takes what memory holds
        deferred_since: ?i96 = null, // its restore was refused for capacity since then, with no restore ending meanwhile
        waited_since: ?i96 = null, // it first waited for a restore then, deferred or parked (load_wait_ns from it)
        // while every lane was busy: it asked the disk ahead of its turn (once), parked on the restore it began, and
        // was woken when that restore ended
        read_ahead: enum { none, tried, parked, woken } = .none,
        join: pc.Join = .{}, // its conversation (0111)
        parent: ?u64 = null, // the durable point its pass resumes from, and where
        fork: u32 = 0,
        lineage: []pc.Fork = &.{}, // the computation its pass's rows belong to (gpa-owned; Store.passLineage)

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            if (!cacheable(job.request)) return;
            const h = job.host;
            const store = h.cache orelse return;
            const r = job.request;
            if (store.tier == null) {
                _ = store.keep(keyOf(r), at, s, r.chunks);
                return;
            }
            const pics = r.lane_images.len > 0; // (the tier records each picture's identity and the prompt's model ids)
            const t0 = h.clockNs();
            _ = store.keepWith(keyOf(r), at, s, r.chunks, .{ .group = job.join.group, .images = pics, .media = if (pics) tierMedia(r) else &.{}, .raw = if (pics) r.prompt else &.{}, .parent = job.parent, .fork = job.fork, .lineage = job.lineage });
            h.times.keep.observe(h.elapsedNs(t0));
        }

        /// The images key of the request's `[0, at)`: its pictures' spans and bytes, hashed. A kept state holds its
        /// image spans' placeholders as tokens, so the key is what tells one picture from another (0: none).
        fn imageKey(ctx: ?*const anyopaque, at: u32) u64 {
            const r: *const Request = @ptrCast(@alignCast(ctx orelse return 0));
            var h = std.hash.Wyhash.init(0x696d61676573); // "images"
            var any = false;
            for (r.lane_images) |img| {
                if (img.at >= at) continue; // the image starts past the kept prefix
                any = true;
                var fields = [3]u64{ img.at, img.tokens, std.hash.Wyhash.hash(0, img.bytes) };
                h.update(std.mem.asBytes(&fields));
            }
            return if (any) h.final() | 1 else 0; // odd: never 0, which means no image in the prefix
        }

        /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
        fn reported(job: *Job) void {
            const e = job.entry orelse return;
            job.entry = null;
            const store = job.host.cache orelse return;
            if (!job.started) return;
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            const ok = !job.stream.reuse_failed and job.stream.cached == e.at;
            if (!ok) { // it did not compute from that state's rows: a lineage of its own
                job.parent = null;
                job.fork = 0;
                if (job.lineage.len > 0) {
                    const fresh: []pc.Fork = store.passLineage(null) catch &.{};
                    job.host.gpa.free(job.lineage);
                    job.lineage = fresh;
                }
            }
            if (job.stream.reuse_failed) store.resumed(e, keyOf(job.request), false) else if (ok) store.resumed(e, keyOf(job.request), true);
        }
    };

    /// The pictures' identities the durable tier gets for a request: its `cache_media` only when they name exactly the
    /// spans the engine runs (`images`: the spans its restored rows and tower work are placed by), one for one; anything
    /// else gives none, and the tier then misses and keeps nothing for it (a disk hit must not resume pictures placed
    /// elsewhere than the point recorded them).
    pub fn tierMedia(r: *const Request) []const pc.Media {
        if (r.cache_media.len != r.lane_images.len) return &.{};
        for (r.cache_media, r.lane_images) |m, img| if (m.start != img.at or m.len != img.tokens) return &.{};
        return r.cache_media;
    }

    /// The ids the prompt cache keys a request by: its prompt, or with images the prompt with each picture's positions made
    /// from the picture's content (Request.cache_key).
    /// Whether the cache may serve or keep a request: without images always; with them only when the server keyed their
    /// positions by content (a kept state is found by tokens, and every picture's positions hold the same placeholder).
    fn cacheable(r: *const Request) bool {
        return r.lane_images.len == 0 or r.cache_key.len == r.prompt.len;
    }

    fn keyOf(r: *const Request) []const u32 {
        return if (r.cache_key.len == r.prompt.len) r.cache_key else r.prompt;
    }

    /// The tier's settings as engine_api.Spill.fromEnv parsed them, never clamped (before start, after the backend
    /// bound the store's tier or not). A value outside the parser's ranges, or the settings' off reason, turns the tier
    /// off for the process, named on status: the store's tier is unbound, so the host serves from memory as without
    /// one. Status carries the cache's counters when the tier was asked for, or the cache has a tier or grouping.
    pub fn configure(h: *LaneHost, sp: api.Spill) void {
        const ms = std.time.ns_per_ms;
        h.tier = .{
            .service_ns = @as(u64, sp.load_ms) * ms,
            .pump_ns = @as(u64, sp.pump_ms) * ms,
            .load_wait_ns = @as(i96, sp.load_wait_ms) * ms,
            .flush_ns = @as(u64, sp.flush_ms) * ms,
            .defer_ns = @as(i96, sp.defer_ms) * ms,
            .defer_retry_ns = @as(i96, sp.defer_retry_ms) * ms,
            .defer_min = sp.defer_min,
            .poll_ns = @as(u64, sp.poll_ms) * ms,
        };
        var buf: [120]u8 = undefined;
        const why = sp.off orelse refusal(sp, &buf);
        h.spill_asked = sp.asked;
        h.spill_off = if (why) |w| api.SpillState.offFor(w) else null;
        if (why != null) if (h.cache) |store| if (store.tier != null) {
            store.setTier(null, 0);
            store.groups = false; // (setTier turned grouping on for the tier's quotas: no tier, no grouping)
        };
        const store_on = if (h.cache) |s| s.tier != null or s.groups else false;
        h.time_rounds = if (h.cache) |s| s.tier != null else false;
        h.tele = if (sp.asked or store_on) .{} else null;
        h.spill = h.spillNow();
    }

    /// The host's scheduling settings outside the parser's ranges (engine_api.Spill.ints), or staging's share above the
    /// service's budget, in the parser's words; null: none. Only a Spill not made by fromEnv can meet one.
    fn refusal(sp: api.Spill, buf: []u8) ?[]const u8 {
        const policy = [_][]const u8{ "load_ms", "pump_ms", "load_wait_ms", "flush_ms", "defer_ms", "defer_retry_ms", "defer_min", "poll_ms" };
        inline for (api.Spill.ints) |n| {
            const used = comptime for (policy) |f| {
                if (std.mem.eql(u8, f, n.field)) break true;
            } else false;
            if (used) {
                const v = @field(sp, n.field);
                if (v < n.min or v > n.max) return std.fmt.bufPrint(buf, "{s}={d}: {s}, a whole number from {d} to {d}", .{ n.name, v, n.unit, n.min, n.max }) catch n.name;
            }
        }
        if (sp.pump_ms > sp.load_ms) return std.fmt.bufPrint(buf, "TF_SPILL_PUMP_MS={d}: more than TF_SPILL_LOAD_MS={d} (staging's share of a service cannot pass the service)", .{ sp.pump_ms, sp.load_ms }) catch "TF_SPILL_PUMP_MS: more than TF_SPILL_LOAD_MS";
        return null;
    }

    /// The disk tier's state for status (engine thread, or before start): null unless it was asked for; off for a
    /// refusal at the start; off with the tier's own run-time reason (Store.tierOff: a failed probe); on while a tier
    /// is bound; else null (not bound yet: the server shows "opening" from the settings).
    fn spillNow(h: *const LaneHost) ?api.SpillState {
        if (!h.spill_asked) return null;
        if (h.spill_off) |off| return off;
        const store = h.cache orelse return null;
        if (store.tier == null) return null;
        if (store.tierOff()) |why| return api.SpillState.offFor(why);
        return .{ .state = .on };
    }

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        var share = share_mod.Share.fromEnv();
        share.rounds = info_.rounds_beside;
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced, .share = share, .live = &live_config.block, .trace_fill = std.c.getenv("TF_DS_PROMPT_TRACE") != null };
    }

    pub fn start(h: *LaneHost) !void {
        h.captureBoot();
        if (share_mod.Share.refusal(h.share)) |why| {
            noteErr("{s}", .{why});
            return error.RoundsBesideWithShare;
        }
        if (h.core.fills()) {
            var line: [640]u8 = undefined;
            std.log.info("{s}", .{share_mod.startLine(&line, h.share.p, h.besideRows(), h.share.rounds, h.info_.resume_rows)});
        }
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what is left and joins the engine thread (whose last work runs before it returns),
    /// freeing nothing: the HTTP threads may still call the engine (submit is refused as Closed; cancel, status and
    /// memory answer normally) until stop frees the lists. A second call does nothing.
    pub fn quiesce(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
    }

    /// quiesce, then the lists freed: no HTTP thread may call the engine after it.
    pub fn stop(h: *LaneHost) void {
        h.quiesce();
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.fill_slots.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
        h.ready.deinit(h.gpa);
        h.loading.deinit(h.gpa);
        h.deferred.deinit(h.gpa);
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .keepalive = keepaliveFn } };
    }

    /// The family's queue as a keepalive target, when it set one.
    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        return self(ctx).keepalive_target;
    }

    fn self(ctx: *anyopaque) *LaneHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing or h.stale) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: Id) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?api.Memory {
        // the engine thread refreshes what the lock guards only for a reader: ask, and wait (a round or a turn at most) for the answer
        if (h.thread != null and !h.closing and h.engine_tid.load(.acquire) != 0 and h.engine_tid.load(.acquire) != std.Thread.getCurrentId()) {
            const gen = h.tele_gen;
            h.tele_want.store(true, .release);
            h.wake.signal(h.io);
            var waited: u32 = 0;
            while (h.tele_gen == gen and !h.closing and waited < tele_wait_max_ms) : (waited += 1) {
                h.tele_done.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(1), .clock = .awake } }) catch {};
            }
        }
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= now - window_ns) tokens += m.tokens;
        }
        const n = @min(stream_tokens.len, h.live_tokens.items.len);
        @memcpy(stream_tokens[0..n], h.live_tokens.items[0..n]);
            .waiting = @intCast(h.queued.items.len + h.waitingCount()),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .graph_fallbacks = if (h.graph_fallbacks) |c| c.load(.monotonic) else null,
            .head_rounds = if (h.head_rounds) |c| .{ c[0].load(.monotonic), c[1].load(.monotonic), c[2].load(.monotonic), c[3].load(.monotonic), c[4].load(.monotonic), c[5].load(.monotonic) } else null,
            .draft_vocab = if (h.draft_vocab) |c| c.read() else null,
            .streams = n,
            .generation_tokens = h.live_generated,
            .cache = h.tele,
            .spill = h.spill,
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// Track D: between this admitting job's prompt chunks, the other jobs' cancels taken (this one's stay queued for
    /// its own pass to see), then a round of the streams already live and their new tokens delivered, as the host's loop
    /// does after any round. A failed round ends this prompt's pass; admitOne then fails every admitted job.
    fn between(ctx: *anyopaque) anyerror!void {
        const job: *Job = @ptrCast(@alignCast(ctx));
        const h = job.host;
        h.in_between = true; // (the filling pass stands between two chunks: the service fence's one way in)
        defer h.in_between = false;
        h.takeCancelsExcept(job);
        if (h.core.activeCount() == 0) return;
        h.stepRound() catch |e| {
            h.between_failed = @errorName(e);
            return e;
        };
        var i: usize = 0;
        while (i < h.admitted.items.len) {
            const other = h.admitted.items[i];
            if (other == job or !other.started) {
                i += 1;
                continue;
            }
        }
        h.commitDecoded();
    }

    /// Whether a round between an admitting job's chunks would run (another stream is live): the backend asks first,
    /// so a prompt alone keeps its read-ahead.
    fn betweenReady(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        return job.host.core.activeCount() > 0;
    }

    /// The hook's plan for this job's step just issued, at the default cadence (one prompt step a turn in rotation, then
    /// one round while a stream decodes; `ready` is `.round`): a round while a stream decodes; else another prompt's
    /// step when another admitted prompt fills or waits for its first step, or a request is queued or woken with a lane
    /// free (the stepping host only); else this prompt's next step, or nothing once its pass has ended. With the fill
    /// order the plan is the order's pick: this prompt's next step while the order picks it (and that step is charged
    /// to the prompts it passes over), another's when it does not; a request waiting for a lane is admitted first, so
    /// the order sees it.
    fn betweenNext(ctx: *anyopaque) lanes.stream.Next {
        const job: *Job = @ptrCast(@alignCast(ctx));
        const h = job.host;
        if (h.core.activeCount() > 0) return .round;
        if (h.core.fills()) {
            h.lock();
            defer h.unlock();
            if (h.info_.fill.on) {
                if (h.queued.items.len + h.ready.items.len > 0 and h.admitted.items.len < h.info_.lanes) return .other;
                const me = for (h.admitted.items, 0..) |other, i| {
                    if (other == job) break i;
                } else null;
                // (no stream is active here, so the share owes nothing: every prompt may step)
                const next = h.pickFill(0, false) orelse return .idle;
                if (me == null or next != me.?) return .other;
                h.chargeFill(next);
                return .same;
            }
            for (h.admitted.items) |other| if (other != job and other.prefilled == null) return .other;
            if (h.queued.items.len + h.ready.items.len > 0 and h.admitted.items.len < h.info_.lanes) return .other;
        }
        return if (job.prefilled == null) .same else .idle;
    }

    /// The hook's facts for this job's next step (lanes.stream.Facts; passRows reads them alone): another stream decodes;
    /// another admitted prompt is past its first step and not prefilled (filling); a request is queued, parked, woken,
    /// or admitted and not yet started (waiting); the family's resume rows as the step rows while a stream decodes (one
    /// step between two rounds), else no bound. With the fill order an admitted prompt waiting its turn fills beside
    /// this one: it is not a request waiting for a lane, and its passes keep the pass rows as one filling does.
    fn betweenFacts(ctx: *anyopaque) lanes.stream.Facts {
        const job: *Job = @ptrCast(@alignCast(ctx));
        const h = job.host;
        const decoding = h.core.activeCount() > 0;
        const ordered = h.info_.fill.on;
        h.lock();
        defer h.unlock();
        var fills = false;
        var waits = h.queued.items.len > 0 or h.waitingCount() > 0;
        var unstarted = false;
        for (h.admitted.items) |other| {
            if (other == job) continue;
            if (!other.started) {
                // under the fill order an admitted prompt waiting its turn fills beside this one, not waits for a lane,
                // unless the order serves it before this one: then it waits as the rotation's arrival does
                if (ordered) unstarted = true else waits = true;
            } else if (other.prefilled == null) fills = true;
        }
        if (unstarted) {
            if (h.servesUnstartedFirst(job)) waits = true else fills = true;
        }
        return .{ .decoding = decoding, .filling = fills, .waiting = waits, .step_rows = if (decoding) h.besideRows() else std.math.maxInt(u32), .rounds_beside = h.share.rounds };
    }

    /// Whether the fill order's pick now is an admitted prompt that has not started its first step, not `job`: that
    /// prompt takes the next step, so `job`'s step must not run long ahead of it (the pass rule cuts a step to the resume
    /// rows while a request waits, and lets two passes form a pair, twice the rows, when none does).
    fn servesUnstartedFirst(h: *LaneHost, job: *const Job) bool {
        const next = h.pickFill(0, false) orelse return false;
        const other = h.admitted.items[next];
        return other != job and !other.started;
    }

    /// The rows a prompt step may run between two rounds while a stream decodes: the family's beside rows, else its resume rows.
    fn besideRows(h: *const LaneHost) u32 {
        return if (h.info_.beside_rows != 0) h.info_.beside_rows else h.info_.resume_rows;
    }

    /// Whether the tier part runs between two of an admitting job's chunks: parked work is in flight (restores,
    /// landings, writes to stage, requests parked or woken).
    fn betweenWanted(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        const h = job.host;
        return h.servicing() and h.pending();
    }

    /// The tier part's run: parked work's service between two chunks, after the prefill yield's round when one ran. No round and
    /// no Engram read (a backend calls it without joining its read-ahead), and it cannot fail the pass. The job reports
    /// its restored state first: a restore's room may evict the entry it resumed.
    fn betweenService(ctx: *anyopaque) void {
        const job: *Job = @ptrCast(@alignCast(ctx));
        const h = job.host;
        if (!h.servicing()) return;
        h.in_between = true; // (the filling pass stands between two chunks: the service fence's one way in)
        defer h.in_between = false;
        // the other requests' cancels first (this pass's own stay for its check): a parked or deferred request whose
        // client went stops its restore now, not after this pass (a lone pass runs no round part to take them)
        h.takeCancelsExcept(job);
        job.reported();
        h.service(h.tier.service_ns);
        h.noteLive();
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery.
    fn send(h: *LaneHost, job: *Job) void {
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            emit(job, .{ .tokens = emitted[job.delivered..] });
            h.noteDecoded(emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished and left the lists a status read walks.
    pub fn deliver(h: *LaneHost, job: *Job) bool {
        if (!job.started or job.prefilled == null) return false;
        h.send(job);
        if (!job.stream.finished) return false;
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        h.finish(job, reason, job.stream.problem);
        return true;
    }

    fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        if (job.started and job.prefilled != null) h.send(job); // (tokens the forward wait did not get to)
        job.reported();
        h.gpa.free(job.marks);
        h.gpa.free(job.lineage);
        const s = &job.stream;
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .plans = s.plans, .min_rows = s.min_rows, .loop_period = s.loop_period, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            if (s.grammar) |g| g.deinit(h.gpa);
            s.deinit(h.gpa);
            job.proposer.deinit();
        }
        h.gpa.destroy(job);
    }

    /// Counts tokens landed (engine thread, no lock); `commitDecoded` makes one mark of what a round's streams landed together.
    fn noteDecoded(h: *LaneHost, n: usize) void {
        h.decoded_pending += n;
    }

    /// The mark of the tokens landed since the last one: one clock read, one lock, and the marks older than the window dropped from the
    /// front (they are in time order), once a round however many streams landed tokens in it.
    fn commitDecoded(h: *LaneHost) void {
        if (h.decoded_pending == 0) return;
        const n = h.decoded_pending;
        h.decoded_pending = 0;
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        var old: usize = 0;
        while (old < h.decoded.items.len and h.decoded.items[old].at < now - window_ns) old += 1;
        if (old > 0) h.decoded.replaceRangeAssumeCapacity(0, old, &.{});
        h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
    }

    /// takeCancels for every job but `keep` (an admitting job in its prompt pass): under the one lock that takes the
    /// other ids, `keep`'s stay named (for its pass's own check and the loop after it), so a cancel of it arriving
    /// meanwhile can never free it here.
    fn takeCancelsExcept(h: *LaneHost, keep: *Job) void {
        h.mutex.lockUncancelable(h.io);
        var ids: std.ArrayList(Id) = .empty;
        var i: usize = 0;
        while (i < h.cancels.items.len) {
            const id = h.cancels.items[i];
            if (id == keep.id) {
                i += 1;
                continue;
            }
            ids.append(h.gpa, id) catch {
                i += 1;
                continue;
            };
            _ = h.cancels.orderedRemove(i);
        }
        h.mutex.unlock(h.io);
        h.cancelIds(ids.items);
        ids.deinit(h.gpa);
    }

    /// Cancels the jobs named since the last round: queued, parked or admitted.
    fn takeCancels(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        const ids = h.gpa.dupe(Id, h.cancels.items) catch &.{};
        h.cancels.clearRetainingCapacity();
        h.mutex.unlock(h.io);
        h.cancelIds(ids);
        h.gpa.free(ids);
    }

    /// The jobs these ids name, queued, parked or admitted, cancelled (their ids already taken from `cancels`).
    fn cancelIds(h: *LaneHost, ids: []const Id) void {
        h.mutex.lockUncancelable(h.io);
        const ending = endOf(h.closing); // (the stop's, or the clients' own cancels)
        var dropped: std.ArrayList(*Job) = .empty;
        var unloaded: std.ArrayList(*pc.Restore) = .empty; // restores a cancelled job waited for
        for (ids) |id| {
            for (h.queued.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.queued.orderedRemove(i)) catch {};
                break;
            };
            for (h.ready.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.ready.orderedRemove(i)) catch {};
                break;
            };
            for (h.loading.items, 0..) |l, i| if (l.job.id == id) {
                _ = h.loading.orderedRemove(i);
                dropped.append(h.gpa, l.job) catch {};
                unloaded.append(h.gpa, l.restore) catch {};
                break;
            };
            for (h.deferred.items, 0..) |d, i| if (d.job.id == id) {
                dropped.append(h.gpa, h.deferred.orderedRemove(i).job) catch {};
                break;
            };
        }
        h.mutex.unlock(h.io);
        // a restore no request waits for any more stops (its reads, DMA and reservations go), exactly once
        for (unloaded.items, 0..) |r, k| {
            if (std.mem.indexOfScalar(*pc.Restore, unloaded.items[0..k], r) != null) continue;
            const wanted = for (h.loading.items) |l| {
                if (l.restore == r) break true;
            } else false;
            if (!wanted) if (h.cache) |store| store.restoreCancel(r);
        }
        unloaded.deinit(h.gpa);
        for (dropped.items) |job| h.finish(job, ending.reason, ending.message);
        dropped.deinit(h.gpa);
        for (ids) |id| {
            for (h.admitted.items, 0..) |job, i| if (job.id == id) {
                h.finish(job, ending.reason, ending.message);
                break;
            };
        }
    }

    /// How a request ended by the host (not run to its end) finishes: cancelled for its client's cancel; a failure
    /// naming the stop while the host closes (stop_message: the client sends it again).
    fn endOf(closing: bool) struct { reason: Reason, message: []const u8 } {
        return if (closing) .{ .reason = .failed, .message = stop_message } else .{ .reason = .cancelled, .message = "" };
    }

    fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    /// Prefills the next woken or queued request into a free lane, or parks it on a restore (holding no lane); false
    /// when none waits or no lane is free, or the host is closing (closeAll cancels what waits, parked requests
    /// included: a stop in a prompt pass starts no other request's pass).
    fn admitOne(h: *LaneHost) bool {
        h.lock();
        if (h.closing or (h.queued.items.len == 0 and h.ready.items.len == 0)) {
            h.unlock();
            return false;
        }
        if (h.admitted.items.len >= h.info_.lanes) return h.readAhead();
        if (h.full_wait) |n| {
            const active = if (h.core.fills()) h.admitted.items.len else h.core.activeCount();
            if (active > 0 and active >= n and std.Io.Clock.awake.now(h.io).toNanoseconds() < h.full_retry) {
                h.unlock();
                return false;
            }
            h.full_wait = null;
        }
        if (h.finish_until) |t| {
            if (std.Io.Clock.awake.now(h.io).toNanoseconds() < t and h.nearEnd()) {
                h.unlock();
                return false;
            }
            h.finish_until = null;
        }
        const job = if (h.ready.items.len > 0) h.ready.orderedRemove(0) else h.queued.orderedRemove(0);
        if (h.parking()) {
            h.unlock();
            if (h.park(job, false)) return true; // waits for a restore, holding no lane
            h.lock();
        }
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        job.overtaken = 0;
        job.stepped_at = h.clockNs();
        h.unlock();
        if (h.info_.fill.on) job.rows_est = h.estimateRows(job);
        if (h.core.fills()) return true;
        if (!h.setup(job)) return true;
        const began = job.began;
        if (h.loneFits(job)) return h.runLone(job, began);
        // parked work's service between two chunks, after the prefill yield's round part (set after loneFits: a lone driver's
        // pass, Backend.opening, prefills through the same stream outside the fence)
        if (h.servicing()) job.stream.between.?.tier = .{ .wanted = betweenWanted, .run = betweenService };
        h.filling_job = job; // (the service fence: inside this pass only Stream.between may service)
        defer h.filling_job = null;
        h.core.addStream(&job.stream) catch |e| {
            if (h.between_failed) |msg| {
                // a round between this prompt's chunks failed: this job and every admitted one end with its error
                h.between_failed = null;
                _ = h.drop(job, msg);
                h.failAll(msg);
                return true;
            }
            if (e == error.Cancelled) return h.cancel(job);
            if (e == error.ContextFull and h.core.activeCount() > 0) return h.requeue(job);
            return h.drop(job, h.words(e));
        };
        h.prefilled(job, began);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        if (now - began >= long_pass_ns) {
            h.lock();
            h.finish_until = now + finish_ns;
            h.unlock();
        }
        return true;
    }

    /// Look up and consume a kept state in the same engine turn: another prompt may evict it at its next mark.
    fn setup(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        var reuse: lanes.stream.Reuse = .{};
        // a prompt with pictures is keyed by the server's ids for them (Request.cache_key), which tell one picture from
        // another, so the images key stays empty (.none): the pictures are hashed once, by the server
        job.images = .none;
        // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
        if (cacheable(r)) if (h.cache) |store| if (store.lookupKeyed(h.gpa, keyOf(r), r.history_len, r.shared_prefixes, r.chunks, job.images)) |l| {
            job.entry = l.entry;
            job.kept0 = store.counts.kept;
            job.marks = l.marks;
            reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
            if (store.groups) job.join = store.join(keyOf(r), l.entry, r.lane_images.len == 0 and r.drafts);
            if (store.tier != null) {
                if (l.entry) |e| {
                    job.parent = e.point;
                    job.fork = e.at;
                }
                // its rows' computation: the resumed state's forked at its position (written or not), else its own
                job.lineage = store.passLineage(l.entry) catch &.{};
            }
        } else |_| {};
        job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch {
            _ = h.drop(job, "the drafter could not start");
            return false;
        };
        job.stream = lanes.Stream.init(h.gpa, .{
            .id = "request",
            .prompt = r.prompt,
            .max_new = r.max_tokens,
            .eos = r.eos,
            .sampling = r.sampling,
            .drafts = r.drafts,
            .proposer = job.proposer.proposer(),
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .loop_guard = r.loop_guard,
            .chunks = r.chunks,
            .reuse = reuse,
            .lane_images = r.lane_images,
            .grammar = if (r.structure) |st| h.constraint(st) catch |e| {
                job.proposer.deinit();
                recalls.release(job);
                return h.drop(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
            } else null,
        }) catch {
            job.proposer.deinit();
            _ = h.drop(job, "out of memory");
            return false;
        };
        job.stream.between = .{ .ptr = job, .run = between, .ready = betweenReady, .next = betweenNext, .facts = betweenFacts };
        job.started = true;
        const began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        job.began = began;
        return true;
    }

    /// One prompt step per turn, the prompt the host's fill order picks (fill_order.zig; the rotation by default).
    /// With a share below 1 and decode time owed to the streams, only a short prompt steps, the one the order picks
    /// among the short ones (a long prompt's next step waits for the rounds that repay the last one), and a long
    /// prompt's step beside decoders is timed and billed.
    fn fillOne(h: *LaneHost) void {
        if (!h.core.fills()) return;
        const shared = h.share.active();
        const decoders: usize = if (shared) h.runnable() else 0;
        if (shared and decoders == 0) h.share.forgive(); // owed to streams that ended or paused: no one's
        var i = h.pickFill(h.fill_next, false) orelse {
            if (shared) h.share.forgive();
            return;
        };
        if (h.share.owes()) i = h.pickFill(h.fill_next, true) orelse return h.traceFill(null, decoders);
        const job = h.admitted.items[i];
        h.traceFill(i, decoders);
        h.chargeFill(i);
        h.fill_next = i + 1;
        const first = !job.started;
        // a long prompt's step beside decoders is the share's to bill (a short prompt's never is)
        const long = shared and decoders > 0 and h.fillRows(job) > h.share.short;
        const billed = long and h.share.p < 1; // (a count of rounds times nothing)
        job.stream.settle_step = billed; // timed on the host, so a step owed to decoders ends on the device; none left marked
        const t0: i96 = if (billed) h.shareNow() else 0;
        const step = blk: {
            h.filling_job = job; // (the service fence: no service inside a prompt's chunk)
            defer h.filling_job = null;
            break :blk h.core.fillStream(&job.stream, first);
        };
        const done = step catch |e| {
            if (e == error.Cancelled) {
                _ = h.cancel(job);
            } else if (e == error.ContextFull and h.admitted.items.len > 1) {
                _ = h.requeue(job);
            } else {
                _ = h.drop(job, h.words(e));
            }
            return;
        };
        if (billed) h.share.stepped(@intCast(@max(0, h.shareNow() - t0)), true);
        if (long) h.share.stepRounds(true, job.stream.step_pair);
        job.reported(); // the first step restored it; no dangling cache entry across another prompt's keep
        // the copy index of the prompt, a slice a step (the GPU runs the step just issued): the first round after the prompt finds it built
        if (!done and job.stream.drafts and job.stream.proposer != null) job.proposer.warm(job.stream.context.items, job.stream.prompt_len, warm_tokens) catch {};
        if (done) {
            h.prefilled(job, job.began);
        }
    }

    /// The TF_DS_PROMPT_TRACE line for a step served to admitted prompt `at`, or (null) held back because the share owes
    /// decoders time and no prompt is short. Each admitted prompt that has not prefilled shows as id:rows-left:overtaken
    /// (s: started, f: fills beside, w: not yet started), the picked one marked with an arrow.
    fn traceFill(h: *LaneHost, at: ?usize, decoders: usize) void {
        if (!h.trace_fill or @import("builtin").is_test) return;
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        for (h.admitted.items, 0..) |j, k| {
            if (j.prefilled != null) continue;
            w.print(" {s}{d}:{d}:{d}{s}", .{ if (at != null and at.? == k) ">" else "", j.id, if (j.started) rowsLeft(j) else j.rows_est, j.overtaken, if (j.started) "s" else "w" }) catch break;
        }
        std.log.info("prompt step {s}: share debt {d} us, decoders {d}, prompts{s}", .{ if (at == null) "held" else "served", @divTrunc(h.share.debt_ns, 1000), decoders, w.buffered() });
    }

    /// Streams the next round runs: unfinished and not paused.
    fn runnable(h: *const LaneHost) usize {
        var n: usize = 0;
        for (h.core.live.items) |s| n += @intFromBool(!s.finished and !s.paused);
        return n;
    }

    /// The prompt rows an admitted job's whole pass computes: its prompt less the rows a kept state restored. Before its
    /// first step that is the fill order's estimate of the kept state (the same rows its class is judged by), and the
    /// whole prompt when no order is on. A long prompt's last rows stay long: it is the prompt that is short or not, not
    /// what is left of it.
    fn fillRows(h: *const LaneHost, job: *const Job) u64 {
        if (!job.started) return if (h.info_.fill.on) job.rows_est else job.request.prompt.len;
        return @as(u64, job.stream.prompt_len) -| job.stream.cached;
    }

    /// The share's clock: awake nanoseconds on the host's, or a test's.
    fn shareNow(h: *const LaneHost) i96 {
        if (h.share_clock) |c| return c.now(c.ctx);
        return h.clockNs();
    }

    /// The rows a not-yet-started job's pass will compute: its prompt less the kept state the cache holds for it now
    /// (what its setup will find, barring a keep or an eviction by another pass in between).
    fn estimateRows(h: *LaneHost, job: *Job) u64 {
        const r = job.request;
        const hit: u64 = if (cacheable(r)) if (h.cache) |store| (if (store.findKeyed(keyOf(r), r.chunks, .none)) |e| e.at else 0) else 0 else 0;
        return r.prompt.len -| hit;
    }

    /// The rows a job has still to compute: its prompt less what its pass restored and has passed (the backend stores
    /// the rows it passed after each chunk), or its estimate before the pass starts.
    fn rowsLeft(job: *const Job) u64 {
        if (!job.started) return job.rows_est;
        const passed = @max(job.stream.cached, job.progress.read().at);
        return job.request.prompt.len -| passed;
    }

    /// The admitted index whose step is served next, or null when every admitted prompt has prefilled (the lock held, or
    /// the engine thread). `rotate_from` is where the rotation looks first. `short_only`: the share owes decoders time, so
    /// only a prompt the share counts as short may step (the order picks among those); null when there is none.
    fn pickFill(h: *LaneHost, rotate_from: usize, short_only: bool) ?usize {
        const jobs = h.admitted.items;
        const now = h.clockNs();
        h.fill_slots.clearRetainingCapacity();
        for (jobs) |job| h.fill_slots.append(h.gpa, .{
            .prefilled = job.prefilled != null,
            .eligible = !short_only or h.fillRows(job) <= h.share.short,
            .background = job.request.background,
            .rows = rowsLeft(job),
            .overtaken = job.overtaken,
            .waited_ns = @intCast(@max(0, @min(now - job.stepped_at, std.math.maxInt(i64)))),
        }) catch {
            // no room for the reading: the first prompt that has not prefilled (and is short, when only those may step)
            for (jobs, 0..) |j, i| if (j.prefilled == null and (!short_only or h.fillRows(j) <= h.share.short)) return i;
            return null;
        };
        return api.fill_order.pick(&h.info_.fill, h.fill_slots.items, rotate_from);
    }

    /// A step is served to admitted prompt `at`: its own count and wait start over and every prompt it passes over counts it
    /// when the order says so (fill_slots holds the reading pickFill took).
    fn chargeFill(h: *LaneHost, at: usize) void {
        const jobs = h.admitted.items;
        if (!h.info_.fill.on or h.fill_slots.items.len != jobs.len) return;
        const picked = h.fill_slots.items[at];
        const now = h.clockNs();
        for (jobs, h.fill_slots.items, 0..) |j, slot, i| {
            if (i == at) {
                j.overtaken = 0;
                j.stepped_at = now;
            } else if (api.fill_order.overtakes(&h.info_.fill, picked, slot)) j.overtaken +|= 1;
        }
    }

    fn filling(h: *const LaneHost) bool {
        if (!h.core.fills()) return false;
        for (h.admitted.items) |job| if (job.prefilled == null) return true;
        return false;
    }

    /// Whether an admitted stream decodes within finish_left tokens of its max_tokens (the lock held).
    fn nearEnd(h: *LaneHost) bool {
        for (h.admitted.items) |job| if (job.started and !job.stream.finished and job.stream.budgetLeft() <= finish_left) return true;
        return false;
    }

    /// A reply's grammar from the server's compiler (Info.structures engines get requests that carry one).
    fn constraint(h: *LaneHost, st: api.Structure) !*lanes.grammar.Constraint {
        const c = st.compiler orelse return error.NoGrammarCompiler;
        const k = try c.constraint(h.gpa, h.io, @fromBackingInt(@intCast(@backingInt(st.kind))), st.text, st.after);
        errdefer k.deinit(h.gpa);
        if (st.ban.len > 0) k.ban = try h.gpa.dupe(u32, st.ban);
        return k;
    }

    fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        h.unlock();
        job.reported();
        if (h.tele != null) {
            h.host_counts.cached_tokens += job.stream.cached;
            h.host_counts.computed_tokens += job.request.prompt.len -| job.stream.cached;
        }
        // a request that waited for a restore and resumed: its pass past the resumed tokens
        if (job.waited_since) |w| if (job.stream.cached > 0) {
            const tail: u64 = @intCast(@max(0, done - began));
            h.times.tail.observe(tail);
            if (!@import("builtin").is_test) std.log.info("lane host: a request waited {d} ms for its restore{s}, resumed {d} of its {d} prompt tokens and computed the rest in {d} ms", .{ @divTrunc(@max(0, began - w), std.time.ns_per_ms), if (job.read_ahead == .woken) " (read ahead while every lane was busy) and a lane" else "", job.stream.cached, job.request.prompt.len, tail / std.time.ns_per_ms });
        };
        if (h.cache) |store| {
            store.continued(job.join, job.stream.cached, job.request.prompt.len);
            store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
        }
        emit(job, .{ .prefilled = job.stream.cached });
    }

    /// An idle backend driver takes a lone drafted request, including sampling when supported.
    fn loneFits(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        const lone = h.lone orelse return false;
        if ((r.sampling != null and !lone.sampled) or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null) return false;
        h.lock();
        defer h.unlock();
        return h.admitted.items.len == 1 and h.queued.items.len == 0 and h.cancels.items.len == 0 and h.core.activeCount() == 0 and h.waitingCount() == 0;
    }

    /// Requests parked or woken (the lock held).
    fn waitingCount(h: *const LaneHost) usize {
        return h.ready.items.len + h.loading.items.len + h.deferred.items.len;
    }

    /// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
    fn runLone(h: *LaneHost, job: *Job, began: i96) bool {
        h.lone_job = job;
        job.delivered = 0;
        const lone = h.lone.?;
        const Hooks = struct {
            fn committed(ctx: *anyopaque) void {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                const j = host.lone_job.?;
                if (!j.prefill_sent) {
                    j.prefill_sent = true;
                    host.prefilled(j, j.began);
                }
                host.send(j);
                host.noteLive();
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.queued.items.len > 0 or host.cancels.items.len > 0 or host.closing or host.waitingCount() > 0;
            }
        };
        h.prefilled(job, began);
        if (h.cache) |store| _ = if (!job.request.untaught and store.rules.lookahead == 0 and keyed(job.request)) store.keepBorn(job.request.cacheKey(), job.request.bornLen(), @intCast(job.request.prompt.len), &job.stream, job.request.chunks, h.spans(job.request)) else false;
        be.release(&job.stream);
        h.remove(job);
        h.finish(job, .length, "");
        return true;
    }

    fn words(h: *const LaneHost, e: anyerror) []const u8 {
        const x = h.explain orelse return @errorName(e);
        return x.text(x.ctx, e) orelse @errorName(e);
    }

    fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        h.remove(job);
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// A request whose prompt pass the backend's pool cannot hold while other streams run (error.ContextFull, before
    /// anything of it was placed) goes back to the queue's front and waits until one of them ends, as the Python
    /// lane's window waits, instead of failing, or a second at most (the backend may place it by then). Returns
    /// false: nothing more is admitted now.
    fn requeue(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.core.discard(&job.stream);
        job.entry = null; // nothing was restored: the store keeps the entry for the next lookup
        h.gpa.free(job.marks);
        job.marks = &.{};
        if (job.stream.grammar) |g| g.deinit(h.gpa); // the next admission compiles its own
        job.stream.deinit(h.gpa);
        job.proposer.deinit();
        job.started = false;
        job.began = 0;
        job.rows_est = 0;
        job.overtaken = 0;
        job.stepped_at = 0;
        job.prefilled = null;
        job.delivered = 0;
        job.prefill_sent = false;
        job.join = .{};
        job.parent = null;
        job.fork = 0;
        h.gpa.free(job.lineage);
        job.lineage = &.{};
        // the pool's room changes by its next admission: it asks the disk again then (a restore refused for room now
        // may land then; TF_SPILL_LOAD_WAIT_MS still bounds its whole wait from waited_since)
        job.tried_disk = false;
        h.lock();
        h.queued.insert(h.gpa, 0, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.full_wait = if (h.core.fills()) h.admitted.items.len else h.core.activeCount();
        h.full_retry = std.Io.Clock.awake.now(h.io).toNanoseconds() + full_retry_ns;
        h.unlock();
        return false;
    }

    /// A job cancelled in its prompt pass, its lane already released (by its client, or by the stop).
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.lock();
        const ending = endOf(h.closing);
        h.unlock();
        h.remove(job);
        h.finish(job, ending.reason, ending.message);
        return true;
    }

    fn remove(h: *LaneHost, job: *Job) void {
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
    }

        if (!h.tele_want.swap(false, .acq_rel)) return; // (nobody asked since the last refresh)
        h.tele_builds += 1;
        const tele = h.telemetry();
        const spill = if (h.spill_asked) h.spillNow() else null;
        h.lock();
        defer h.unlock();
        h.live_tokens.clearRetainingCapacity();
        h.live_generated = 0;
        for (h.admitted.items) |job| if (job.started) {
            h.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
            h.live_generated += @intCast(job.stream.emitted().len);
        };
        if (tele) |t| h.tele = t;
        if (h.spill_asked) h.spill = spill;
        h.tele_gen +%= 1;
        h.tele_done.broadcast(h.io);
    }

    /// What the start gave, kept for `unset` to return to. Called by `start`, after the tier was configured.
    pub fn captureBoot(h: *LaneHost) void {
        h.boot = .{ .tier = h.tier, .share_p = h.share.p, .beside_rows = h.info_.beside_rows, .fill = h.info_.fill };
    }

    /// The host's once-a-turn look at the live block: one atomic load while nothing was published. A published value replaces what the
    /// host runs: the rounds beside a long step (what was owed at the old count is forgiven), the share, the rows a step runs beside a
    /// decoding stream, the tier's timings and the order prompt steps go in; `unset` returns a slot to what the start gave. A value
    /// the host cannot run is not taken (a count of rounds beside a share below 1; step rows that are not whole resume units; staging's
    /// share above the service's budget): the API refuses each before it publishes, and a host that is handed one anyway keeps what it had.
    pub fn liveTurn(h: *LaneHost) void {
        const blk = h.live orelse return;
        var snap: live_config.Snapshot = undefined;
        if (!blk.turnAll(&h.live_seen, &snap)) return;
        const S = live_config.Slot;
        const at = struct {
            fn get(sn: *const live_config.Snapshot, s: S) ?u64 {
                const v = sn.vals[@backingInt(s)];
                return if (v == live_config.unset) null else v;
            }
        }.get;
        // rounds beside
        {
            const want: u32 = if (at(&snap, .rounds_beside)) |v| (std.math.cast(u32, v) orelse h.share.rounds) else h.info_.rounds_beside;
            var next = h.share;
            next.rounds = want;
            if (want != h.share.rounds and share_mod.Share.refusal(next) == null) {
                h.share.rounds = want;
                h.share.due = 0;
                std.log.info("TF_DS_ROUNDS_BESIDE is now {d} (taken live at version {d})", .{ want, h.live_seen });
            }
        }
        // the share
        {
            const want: f64 = if (at(&snap, .prefill_share)) |v| @bitCast(v) else h.boot.share_p;
            var next = h.share;
            next.p = want;
            const ok = std.math.isFinite(want) and want >= share_mod.Share.min_share and want <= 1;
            if (ok and want != h.share.p and share_mod.Share.refusal(next) == null) {
                h.share.p = want;
                if (want >= 1) h.share.debt_ns = 0;
                std.log.info("TENSORFOLD_PREFILL_SHARE is now {d} (taken live at version {d})", .{ want, h.live_seen });
            }
        }
        // the rows a prompt step runs beside a decoding stream
        {
            const want: u32 = if (at(&snap, .pass_beside)) |v| (std.math.cast(u32, v) orelse h.info_.beside_rows) else h.boot.beside_rows;
            const unit = h.info_.resume_rows;
            const whole = want == 0 or (unit != 0 and want % unit == 0);
            if (whole and want != h.info_.beside_rows) {
                h.info_.beside_rows = want;
                std.log.info("TF_DS_PASS_BESIDE is now {d} rows (taken live at version {d})", .{ if (want != 0) want else unit, h.live_seen });
            }
        }
        // the tier's timings
        {
            var next = h.tier;
            const ms = std.time.ns_per_ms;
            const b = h.boot.tier;
            const U = struct {
                fn ns(v: ?u64, boot: u64) u64 {
                    return if (v) |x| x *| ms else boot;
                }
            };
            next.service_ns = U.ns(at(&snap, .spill_load_ms), b.service_ns);
            next.pump_ns = U.ns(at(&snap, .spill_pump_ms), b.pump_ns);
            next.poll_ns = U.ns(at(&snap, .spill_poll_ms), b.poll_ns);
            next.flush_ns = U.ns(at(&snap, .spill_flush_ms), b.flush_ns);
            next.load_wait_ns = if (at(&snap, .spill_load_wait_ms)) |x| @as(i96, x) * ms else b.load_wait_ns;
            next.defer_ns = if (at(&snap, .spill_defer_ms)) |x| @as(i96, x) * ms else b.defer_ns;
            next.defer_retry_ns = if (at(&snap, .spill_defer_retry_ms)) |x| @as(i96, x) * ms else b.defer_retry_ns;
            next.defer_min = if (at(&snap, .spill_defer_min)) |x| (std.math.cast(u32, x) orelse h.tier.defer_min) else b.defer_min;
            if (next.pump_ns <= next.service_ns and !std.meta.eql(next, h.tier)) {
                h.tier = next;
                std.log.info("the tier's timings changed live (version {d}): service {d} ms, staging {d} ms, poll {d} ms, load wait {d} ms, defer {d} ms / retry {d} ms / min {d}, flush {d} ms", .{ h.live_seen, next.service_ns / ms, next.pump_ns / ms, next.poll_ns / ms, @divTrunc(next.load_wait_ns, ms), @divTrunc(next.defer_ns, ms), @divTrunc(next.defer_retry_ns, ms), next.defer_min, next.flush_ns / ms });
            }
        }
        // the order prompt steps go in
        {
            const want = snap.fill orelse h.boot.fill;
            if (!std.meta.eql(want, h.info_.fill)) {
                h.info_.fill = want;
                var buf: [512]u8 = undefined;
                std.log.info("the prompt-step order is now live (version {d}): {s}", .{ h.live_seen, api.fill_order.describe(&want, &buf) });
            }
        }
        // the slots of class A: one cfg frame to every rank when one changed
        h.sendCfg(snap.vals);
    }

    /// The slots of class A whose published value is not the one the ranks were told: one cfg frame to every rank (Backend.live_cfg),
    /// sent from here, the loop top, between two frames and with no round open. Only on a change; a turn that took nothing new never gets
    /// here. A value no u32 holds is not sent (the API refuses it first).
    fn sendCfg(h: *LaneHost, vals: [live_config.slot_count]u64) void {
        var items: [live_config.slot_count]live_config.CfgItem = undefined;
        var n: usize = 0;
        for (std.enums.values(live_config.Slot)) |slot| {
            if (!live_config.isA(slot)) continue;
            const i = @backingInt(slot);
            if (vals[i] == h.live_sent[i]) continue;
            const value: u32 = if (vals[i] == live_config.unset) live_config.cfg_unset else std.math.cast(u32, vals[i]) orelse continue;
            items[n] = .{ .slot = @backingInt(slot), .value = value };
            n += 1;
        }
        if (n == 0) return;
        // told once: a failed send ends the engine through the link's watch, and a retry at every turn would only repeat the line
        for (items[0..n]) |it| h.live_sent[it.slot] = vals[it.slot];
        const sent = h.core.liveCfg(h.live_seen, items[0..n]) catch |err| {
            std.log.err("a live change of class A could not be sent to every rank ({s}); the ranks may disagree until a restart", .{@errorName(err)});
            return;
        };
        if (!sent) std.log.warn("a live change of class A was published and this backend has no frame for it: it applies at the next start", .{});
    }

    fn run(h: *LaneHost) void {
        h.engine_tid.store(std.Thread.getCurrentId(), .release);
        defer h.engine_tid.store(0, .release);
        h.overlap = h.core.setOverlap(.{ .ctx = h, .run = overlapWork });
        defer {
            if (h.overlap) _ = h.core.setOverlap(null);
            h.overlap = false;
        }
        while (true) {
            // tokens left for the next round's forward wait go out first when this turn does other work before its round (a prompt's
            // step, a restore's service, an admission, a cancel): they would wait that work's time otherwise
            if (h.overlap and h.turnWorks()) h.flushDeferred();
            h.commitDecoded(); // (a first token or a finish delivered outside a round)
            h.liveTurn();
            h.takeCancels();
            while (h.admitOne()) {}
            // restores, staged writes and deferred retries, between rounds, between prompt chunks and at idle (only with
            // work in flight)
            const busy = h.servicing() and h.pending();
            if (busy) h.service(h.tier.service_ns);
            h.fillOne();
            status.noteLive(h);
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len + h.waitingCount();
                h.unlock();
                if (left == 0) return h.shutdownTier();
                h.closeAll();
                continue;
            }
            if (h.core.activeCount() == 0) {
                if (!h.filling() and h.cancels.items.len == 0 and ((h.queued.items.len == 0 and h.ready.items.len == 0) or h.admitted.items.len >= h.info_.lanes)) {
                    // work in flight: poll it every TF_SPILL_POLL_MS; nothing: wait for a request (100 ms at most)
                    const wait: std.Io.Duration = if (busy) .fromNanoseconds(@intCast(h.tier.poll_ns)) else .fromMilliseconds(100);
                    h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = wait, .clock = .awake } }) catch {};
                }
                h.unlock();
                continue;
            }
            h.unlock();
            h.stepRound() catch |e| {
                h.failAll(@errorName(e));
                continue;
            };
            var i: usize = 0;
            const deferred = h.overlap and h.overlap_ran;
            h.overlap_ran = false;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                // a stream that goes on has its tokens sent in the next round's forward wait (overlapWork, which this round's backend
                // ran); one that finished, or a round that had no such wait, now
                if (deferred and !job.stream.finished and job.started) {
                    i += 1;
                    continue;
                }
            h.commitDecoded();
        }
    }

    /// The host's work inside a decode round's forward wait (Backend.overlap; the engine thread, between the round's launch and the wait for
    /// its head reads, every stream's context as the last round left it): the tokens the streams that go on landed in the round before, to
    /// their requests. Nothing here feeds the round or the next frame; only the streams that finished are left to the loop's tail.
    fn overlapWork(ctx: ?*anyopaque) void {
        const h: *LaneHost = @ptrCast(@alignCast(ctx.?));
        h.overlap_ran = true;
        h.flushDeferred();
    }

    /// The tokens the streams that go on landed, to their requests.
    fn flushDeferred(h: *LaneHost) void {
        for (h.admitted.items) |job| {
            if (!job.started or job.prefilled == null or job.stream.finished) continue;
            h.send(job);
        }
        h.commitDecoded();
    }

    /// Whether this turn does work of its own before its round: a prompt's step, parked work's service, an admission or a cancel.
    fn turnWorks(h: *LaneHost) bool {
        if (h.filling() or (h.servicing() and h.pending())) return true;
        h.lock();
        defer h.unlock();
        return h.queued.items.len > 0 or h.ready.items.len > 0 or h.cancels.items.len > 0 or h.closing;
    }

    /// A failed round ends every stream it held, with the backend's error.
    fn failAll(h: *LaneHost, message: []const u8) void {
        h.lock();
        const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
        h.admitted.clearRetainingCapacity();
        h.unlock();
        for (jobs) |job| {
            if (job.started and !job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .failed, message);
        }
        h.gpa.free(jobs);
    }

    fn closeAll(h: *LaneHost) void {
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.ready.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.loading.items) |l| h.cancels.append(h.gpa, l.job.id) catch {};
        for (h.deferred.items) |d| h.cancels.append(h.gpa, d.job.id) catch {};
        h.unlock();
        h.takeCancels();
    }

    // -- parked work (0110) ---------------------------------------------------------------------------------------

    /// Whether admission may park a request: the cache has a tier.
    fn parking(h: *const LaneHost) bool {
        const store = h.cache orelse return false;
        return store.tier != null;
    }

    /// Whether the engine thread services parked work (restores, staged writes, deferred retries).
    fn servicing(h: *const LaneHost) bool {
        return h.parking();
    }

    /// Work in flight the host must poll (engine thread): restores, writes to stage, requests parked or woken.
    fn pending(h: *const LaneHost) bool {
        if (h.loading.items.len > 0 or h.ready.items.len > 0 or h.deferred.items.len > 0) return true;
        const store = h.cache orelse return false;
        return store.busy();
    }

    /// Every lane busy (lock held; released): the next request in line that has not read ahead yet begins its disk
    /// restore now and waits for it holding no lane, in its place (woken requests are admitted before queued ones, so
    /// of the queued only the first: one behind it would overtake it once woken). A restore the tier cannot begin now is
    /// not deferred: the request asks the disk again at its turn. True when it parked (the next in line may read ahead).
    fn readAhead(h: *LaneHost) bool {
        if (!h.parking()) {
            h.unlock();
            return false;
        }
        var list = &h.ready;
        var i: usize = 0;
        while (i < list.items.len and !aheadable(list.items[i])) i += 1;
        if (i == list.items.len) {
            list = &h.queued;
            i = 0;
            if (list.items.len == 0 or !aheadable(list.items[0])) {
                h.unlock();
                return false;
            }
        }
        const job = list.orderedRemove(i);
        h.unlock();
        job.read_ahead = .tried;
        if (h.park(job, true)) {
            job.read_ahead = .parked;
            return true;
        }
        job.tried_disk = false; // (its turn asks the disk again)
        h.lock();
        list.insert(h.gpa, i, job) catch { // (a submit may have taken the room meanwhile)
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.unlock();
        return false;
    }

    /// Whether `job` may read ahead: it has not asked the disk yet, ahead or not.
    fn aheadable(job: *const Job) bool {
        return job.read_ahead == .none and !job.tried_disk;
    }

    /// Before `job`'s pass (engine thread, no lock): park it on a restore of a longer durable point (once). False:
    /// admit it now. `ahead` (every lane busy, readAhead): a restore the tier refuses for capacity is not deferred.
    fn park(h: *LaneHost, job: *Job, ahead: bool) bool {
        const store = h.cache orelse return false;
        const r = job.request;
        // (a woken job takes what memory holds now: asked before the probe, so it costs no second index scan)
        if (job.tried_disk or !cacheable(r)) return false;
        const key = keyOf(r);
        const now = h.clockNs();
        const probe = store.probeMedia(key, r.chunks, r.lane_images.len > 0, tierMedia(r));
        h.times.lookup.observe(h.elapsedNs(now));
        const p = probe.durable orelse return false;
        job.tried_disk = true;
        // its whole wait (deferred, then waiting for a lane) already reached TF_SPILL_LOAD_WAIT_MS: no restore begins
        // to be cancelled at the next service (its room evicted and its rows staled for nothing); memory serves it
        if (job.waited_since) |w| if (now - w >= h.tier.load_wait_ns) {
            if (!ahead) h.host_counts.restores_deferred_expired += 1; // (counted at its turn)
            return false;
        };
        h.lock();
        const room = h.loading.ensureUnusedCapacity(h.gpa, 1);
        h.unlock();
        room catch return false;
        // (the best memory state stays while the budget holds it beside the restore: refused, the pass resumes it; a
        // full tier refuses before any room is made)
        const rs = store.restoreBeginOver(p, key, .{ .have = probe.memory, .again = job.deferred_since != null }) catch |err| {
            if (ahead) return false;
            const left = store.find(key, r.chunks); // (the room may have evicted probe.memory: never read it now)
            if (h.deferRestore(job, p, left, err, now)) return true;
            if (left) |e| {
                note("reading {d} kept tokens from disk did not start ({s}); resuming the {d} memory holds", .{ p.at, @errorName(err), e.at });
            } else note("reading {d} kept tokens from disk did not start ({s}); prefilling", .{ p.at, @errorName(err) });
            return false;
        };
        const since = job.waited_since orelse now; // (LOAD_WAIT bounds the deferral and the restore together)
        h.lock();
        h.loading.appendAssumeCapacity(.{ .job = job, .restore = rs, .since = since });
        h.unlock();
        h.host_counts.restore_waits += 1;
        job.deferred_since = null;
        job.waited_since = since;
        return true;
    }

    /// A restore of `p` the tier or the budget refused with `err`: when that may change shortly (Store.deferrable)
    /// and the point saves at least defer_min tokens over `have` (the best memory state), `job` waits on the deferred
    /// list (holding no lane) and asks again every defer_retry_ns or when a restore in flight ends: defer_ns at most
    /// with no restore ending (any restore's end restarts its clock, on whichever list it waited then, since waiting
    /// may then pay) and load_wait_ns in all from its first wait; true then. False: admit it now on what memory holds.
    fn deferRestore(h: *LaneHost, job: *Job, p: pc.Point, have: ?*pc.Entry, err: anyerror, now: i96) bool {
        const store = h.cache orelse return false;
        if (h.tier.defer_ns <= 0 or !store.deferrable(err)) return false;
        const at: u32 = if (have) |e| e.at else 0;
        const min = if (h.tier.defer_min > 0) h.tier.defer_min else @max(store.rules.grid, 1);
        if (p.at <= at or p.at - at < min) return false;
        const first = job.deferred_since orelse now;
        const waited = job.waited_since orelse now;
        // its clock counts from its first refusal, or from the last restore that ended since, wherever it waited then
        const since = @max(first, h.last_restore_end orelse first);
        const quiet = now - since >= h.tier.defer_ns; // no restore ended for TF_SPILL_DEFER_MS
        if (quiet or now - waited >= h.tier.load_wait_ns) {
            h.host_counts.restores_deferred_expired += 1;
            if (quiet) {
                note("reading {d} kept tokens from disk was refused for capacity for {d} ms with no restore ending ({s}); admitted on what memory holds", .{ p.at, @divTrunc(now - since, std.time.ns_per_ms), @errorName(err) });
            } else note("reading {d} kept tokens from disk was refused for capacity ({s}) until its wait reached TF_SPILL_LOAD_WAIT_MS ({d} ms); admitted on what memory holds", .{ p.at, @errorName(err), @divTrunc(now - waited, std.time.ns_per_ms) });
            return false;
        }
        h.lock();
        defer h.unlock();
        h.deferred.append(h.gpa, .{ .job = job, .retry_at = now + h.tier.defer_retry_ns }) catch return false;
        if (job.deferred_since == null) h.host_counts.restores_deferred += 1;
        job.deferred_since = first;
        job.waited_since = waited;
        job.tried_disk = false; // (it asks for the restore again when it wakes)
        return true;
    }

    /// Deferred restores whose retry time came, or every one when a restore in flight ended (`ended`: its read slot
    /// is free now), go to the ready list to ask again.
    fn retryDeferred(h: *LaneHost, now: i96, ended: bool) void {
        if (h.deferred.items.len == 0) return;
        var moved = false;
        var i: usize = 0;
        while (i < h.deferred.items.len) {
            if (!ended and h.deferred.items[i].retry_at > now) {
                i += 1;
                continue;
            }
            h.lock();
            const d = h.deferred.orderedRemove(i);
            h.toReady(d.job); // (releases the lock)
            moved = true;
        }
        if (!moved) return;
        h.lock();
        h.wake.signal(h.io);
        h.unlock();
    }

    /// A woken job to the ready list (the lock held; released here). Out of memory fails it alone.
    fn toReady(h: *LaneHost, job: *Job) void {
        // one that read ahead waits for a lane next: at its turn it looks up memory and disk again, once (its restored
        // state may have left memory by then; while it is there the disk is not longer and nothing is read)
        if (job.read_ahead == .parked) {
            job.read_ahead = .woken;
            job.tried_disk = false;
        }
        h.ready.append(h.gpa, job) catch {
            h.unlock();
            return h.finish(job, .failed, "out of memory");
        };
        h.unlock();
    }

    /// Parked work's service (engine thread): restores progress under one elapsed-time budget shared fairly (each gets
    /// an equal share of what is left, a different one first each time); writes are staged with what remains. Restores
    /// that end wake every request waiting for them; deferred restores whose retry came ask again. A restore ends only
    /// once every rank holds it (the tier's landing runs inside its poll), so its waiters' lookups resume a state each
    /// rank has.
    fn service(h: *LaneHost, budget_ns: u64) void {
        const store = h.cache orelse return;
        if (!h.fenced("parked work's service")) return;
        const t0 = h.clockNs();
        const series = if (h.in_between) &h.times.service_chunk else &h.times.service; // (between chunks: its own)
        defer series.observe(h.elapsedNs(t0));
        h.expireLoads(t0);
        var ended = false;
        const n = store.restoring.items.len;
        if (n > 0) { // at most service_restores a service, a different first one each time: every restore is polled in turn
            var buf: [service_restores]*pc.Restore = undefined;
            const m = @min(n, buf.len);
            h.turn = (h.turn + 1) % n;
            for (0..m) |k| buf[k] = store.restoring.items[(h.turn + k) % n];
            for (buf[0..m], 0..) |r, k| {
                const spent: u64 = @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t0));
                if (spent >= budget_ns and k > 0) break;
                const share = (budget_ns -| spent) / (m - k);
                const p0 = h.clockNs();
                const st = store.restorePoll(r, share); // (a landing's pieces run inside it)
                h.times.poll.observe(h.elapsedNs(p0));
                if (st != .pending) {
                    h.restoreEnded(r, h.clockNs());
                    ended = true;
                }
            }
        }
        h.retryDeferred(std.Io.Clock.awake.now(h.io).toNanoseconds(), ended);
        const spent: u64 = @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t0));
        // nothing decodes and no prompt waits for its next chunk: staging takes what is left
        const idle = h.filling_job == null and !h.filling() and h.core.activeCount() == 0;
        const p0 = h.clockNs();
        store.pump(if (idle) budget_ns -| spent else @min(budget_ns -| spent, h.tier.pump_ns));
        h.times.pump.observe(h.elapsedNs(p0));
    }

    /// The host's clock (awake ns), and the nanoseconds since `t0` on it.
    fn clockNs(h: *const LaneHost) i96 {
        return std.Io.Clock.awake.now(h.io).toNanoseconds();
    }

    fn elapsedNs(h: *const LaneHost, t0: i96) u64 {
        return @intCast(@max(0, h.clockNs() - t0));
    }

    /// Requests parked on a restore longer than load_wait_ns are admitted without it (they look up what memory holds);
    /// a restore no request waits for any more stops, once.
    fn expireLoads(h: *LaneHost, now: i96) void {
        const store = h.cache orelse return;
        var any = false;
        var i: usize = 0;
        while (i < h.loading.items.len) {
            const l = h.loading.items[i];
            if (now - l.since < h.tier.load_wait_ns) {
                i += 1;
                continue;
            }
            any = true;
            h.lock();
            _ = h.loading.orderedRemove(i);
            const waited: u64 = @intCast(@max(0, now - l.since));
            h.host_counts.restore_wait_timeouts += 1;
            h.host_counts.restore_wait_ns += waited;
            h.times.wait.observe(waited);
            h.host_counts.restore_wait_max_ns = @max(h.host_counts.restore_wait_max_ns, waited);
            h.toReady(l.job); // (releases the lock)
            const wanted = for (h.loading.items) |o| {
                if (o.restore == l.restore) break true;
            } else false;
            if (!wanted) store.restoreCancel(l.restore);
        }
        if (!any) return;
        h.lock();
        h.wake.signal(h.io);
        h.unlock();
    }

    /// Restore `r` ended at `now` (adopted, dropped or failed; `r` is already freed, only its address is compared): its
    /// waiters wake in arrival order, to look up what memory holds now.
    fn restoreEnded(h: *LaneHost, r: *pc.Restore, now: i96) void {
        h.last_restore_end = now;
        while (true) {
            h.lock();
            const i = for (h.loading.items, 0..) |l, i| {
                if (l.restore == r) break i;
            } else {
                h.wake.signal(h.io);
                return h.unlock();
            };
            const l = h.loading.orderedRemove(i);
            const waited: u64 = @intCast(@max(0, now - l.since));
            h.host_counts.restore_wait_ns += waited;
            h.times.wait.observe(waited);
            h.host_counts.restore_wait_max_ns = @max(h.host_counts.restore_wait_max_ns, waited);
            h.toReady(l.job);
        }
    }

    /// Clean shutdown on the engine thread, nothing left to serve: restores stopped, writes drained within flush_ns.
    fn shutdownTier(h: *LaneHost) void {
        const store = h.cache orelse return;
        if (store.tier == null) return;
        _ = store.flush(h.tier.flush_ns);
        if (h.telemetry()) |t| {
            h.lock();
            h.tele = t;
            h.unlock();
        }
    }

    /// The cache's counters for status (engine thread); null while telemetry is off.
    fn telemetry(h: *LaneHost) ?api.CacheStats {
        if (h.tele == null) return null;
        var t: api.CacheStats = .{};
        if (h.cache) |store| {
            const c = store.counts;
            const s = store.stats;
            t.memory_hits = c.hits;
            t.memory_misses = c.misses;
            t.kept = c.kept;
            t.evicted = c.evicted;
            t.refused = c.refused;
            t.failed = c.failed;
            t.entries = store.entries.items.len;
            t.held_bytes = store.held;
            t.disk_lookups = s.disk_lookups;
            t.disk_hits = s.disk_hits;
            t.disk_hits_committed = s.disk_hits_committed;
            t.restores_begun = s.restores_begun;
            t.restores_shared = s.restores_shared;
            t.restores_failed = s.restores_failed;
            t.restores_cancelled = s.restores_cancelled;
            t.restores_dropped = s.restores_dropped;
            t.restores_refused = s.restores_refused;
            t.restores_active = store.restoring.items.len;
            t.restoring_bytes = store.restoring_bytes;
            for (store.restoring.items) |r| {
                t.restore_blocks_done += r.done;
                t.restore_blocks_total += r.total;
            }
            t.restored_tokens = s.restored_tokens;
            t.writes_enqueued = s.writes_enqueued;
            t.shared_promoted = s.shared_promoted;
            t.flushes = s.flushes;
            t.flush_timeouts = s.flush_timeouts;
            t.flush_unpublished = s.flush_unpublished;
            t.chats_continued = s.chats_continued;
            t.chats_continued_prefill_tokens = s.chats_continued_prefill_tokens;
            t.spill = store.tierCounts();
            if (comptime @hasField(api.CacheStats, "token_bytes")) { // (the token arrays the store's budget charges)
                if (comptime !@hasField(pc.Store, "token_held")) @compileError("CacheStats.token_bytes without the store's token_held");
                t.token_bytes = store.token_held;
            }
            // the calls the host times, and the tier's own work (a landing piece, a probe, the store lock)
            t.times = h.times;
            if (store.tierTimes()) |own| {
                inline for (pc.TierTimes.own) |name| @field(t.times, name) = @field(own, name);
            }
        }
        const hc = h.host_counts;
        t.restore_waits = hc.restore_waits;
        t.restore_wait_timeouts = hc.restore_wait_timeouts;
        t.restore_wait_ns = hc.restore_wait_ns;
        t.restore_wait_max_ns = hc.restore_wait_max_ns;
        t.restores_deferred = hc.restores_deferred;
        t.restores_deferred_expired = hc.restores_deferred_expired;
        t.cached_tokens = hc.cached_tokens;
        t.computed_tokens = hc.computed_tokens;
        t.service_refused = h.service_refused;
        t.parked = h.loading.items.len + h.deferred.items.len;
        return t;
    }

    // -- the service fence --------------------------------------------------------------------------------------

    /// One round of the lane core, counted open while it runs (the service fence).
    fn stepRound(h: *LaneHost) !void {
        h.rounds_open += 1;
        defer h.rounds_open -= 1;
        // with a share below 1 the round's time pays the debt a long prompt's step left
        const shared = h.share.p < 1; // (the round's time pays a debt of time; a count of rounds needs no clock)
        const r0: i96 = if (shared) h.shareNow() else 0;
        defer if (shared) h.share.rounded(@intCast(@max(0, h.shareNow() - r0))) else h.share.rounded(0);
        if (!h.time_rounds) return h.core.step();
        const t0 = h.clockNs();
        defer h.times.round.observe(h.elapsedNs(t0));
        return h.core.step();
    }

    /// Whether parked work may be serviced now: no round open (its launches, or a graph it captures, in flight), a
    /// prompt pass only from Stream.between (between two chunks, never between a chunk's layers), and the backend not
    /// capturing a CUDA graph (`quiet`).
    fn serviceSafe(h: *const LaneHost) bool {
        if (h.rounds_open > 0) return false;
        if (h.filling_job != null and !h.in_between) return false;
        if (h.quiet) |q| if (!q.check(q.ctx)) return false;
        return true;
    }

    /// serviceSafe, or the refusal: a panic in debug builds (an engine bug), else counted, logged once and skipped.
    fn fenced(h: *LaneHost, comptime what: []const u8) bool {
        if (h.serviceSafe()) return true;
        h.service_refused += 1;
        if (h.fence_panic) @panic("lane host: " ++ what ++ " asked for inside a round, a prompt chunk or a CUDA graph capture");
        if (h.service_refused == 1) noteErr(what ++ " asked for inside a round, a prompt chunk or a CUDA graph capture: skipped (an engine bug)", .{});
        return false;
    }
};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("lane host: " ++ fmt, args);
}

/// An error line (the fence's); tests stay quiet too.
fn noteErr(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.err("lane host: " ++ fmt, args);
}

test "a waiter's deferral clock restarts at the last restore that ended, whichever list it waited on then" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Fam = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return at;
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
            return error.Unused;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: pc.Saved) void {}
    };
    var fam: u8 = 0;
    var store = pc.Store.init(gpa, .{ .ptr = &fam, .vtable = &.{ .bytes = Fam.bytes, .save = Fam.save, .restore = Fam.restore, .drop = Fam.drop } }, .{}, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop(); // (never started: frees the lists)
    host.cache = &store;
    const ms = std.time.ns_per_ms;
    host.tier.defer_ns = 100 * ms;
    host.tier.defer_retry_ns = 10 * ms;
    host.tier.load_wait_ns = 10 * std.time.ns_per_s;
    host.tier.defer_min = 1;
    const prompt = [_]u32{ 1, 2, 3 };
    const req: Request = .{ .prompt = &prompt, .max_tokens = 1 };
    var job: LaneHost.Job = .{ .host = &host, .id = 1, .request = &req, .sink = undefined };
    const p: pc.Point = .{ .id = 7, .at = 64, .tokens = 64 };
    const t0: i96 = 1000 * ms;
    try std.testing.expect(host.deferRestore(&job, p, null, error.TooManyRestores, t0)); // refused for capacity: it waits
    host.retryDeferred(t0 + 20 * ms, false); // its retry came: on the ready list, until admission next takes it
    try std.testing.expectEqual(@as(usize, 1), host.ready.items.len);
    host.ready.clearRetainingCapacity();
    var other: pc.Restore = undefined;
    host.restoreEnded(&other, t0 + 60 * ms); // another restore ends meanwhile
    // asked again 150 ms after it first waited, 90 ms after that ending: it waits on (DEFER 100 ms)
    try std.testing.expect(host.deferRestore(&job, p, null, error.TooManyRestores, t0 + 150 * ms));
    host.deferred.clearRetainingCapacity();
    // DEFER past the last ending: it is admitted on what memory holds
    try std.testing.expect(!host.deferRestore(&job, p, null, error.TooManyRestores, t0 + 161 * ms));
}

test "a lane host serves the core's own tokens, in order, and cancels between rounds" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (box.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 24), box.tokens.items.len);
    var gone: Box = .{};
    defer gone.tokens.deinit(gpa);
    const long: Request = .{ .prompt = &prompt, .max_tokens = 100000 };
    try e.submit(2, &long, .{ .ctx = &gone, .event = Box.event });
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, gone.wait());

    const CancelPrefill = struct {
        engine: Engine,
        id: Id,
        at: usize,

        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == c.at) c.engine.cancel(c.id);
        }
    };
    const chunked_prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var chunked: Box = .{};
    defer chunked.tokens.deinit(gpa);
    var prefill_cancel = CancelPrefill{ .engine = e, .id = 3, .at = 2 };
    target.prefill_chunks = 10;
    target.prefill_count = 0;
    target.prefill_hook = CancelPrefill.call;
    target.prefill_hook_ctx = &prefill_cancel;
    const chunked_request: Request = .{ .prompt = &chunked_prompt, .max_tokens = 1 };
    try e.submit(3, &chunked_request, .{ .ctx = &chunked, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, chunked.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released

    // the lone driver's prompt pass (gpu_round.run starts with Backend.opening), cancelled the same way
    const Lone = struct {
        be: lanes.backend.Backend,

        fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            _ = try l.be.opening(gpa, s);
            return error.NotCancelled;
        }
    };
    var lone: Box = .{};
    defer lone.tokens.deinit(gpa);
    var lone_driver = Lone{ .be = target.backend() };
    host.lone = .{ .ctx = &lone_driver, .run = Lone.run };
    prefill_cancel.id = 4;
    target.prefill_count = 0;
    try e.submit(4, &chunked_request, .{ .ctx = &lone, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, lone.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test "a finished request's stats carry its rounds by plan: its copied continuations, verified and kept" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    // the fake's reply repeats 11, 12, 13 from its 8th token: the proposer copies it (no head here: every draft a copy)
    var target: lanes.fake.Fake = .{ .gpa = gpa, .cycle_after = 8, .probe_prompt = prompt.len };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        stats: ?Stats = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .finished => |f| b.stats = f.stats,
                else => {},
            }
        }
        fn wait(b: *@This()) Stats {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.stats;
                b.mutex.unlock(std.testing.io);
                if (d) |x| return x;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    var box: Box = .{};
    const request: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    try host.engine().submit(1, &request, .{ .ctx = &box, .event = Box.event });
    const st = box.wait();
    try std.testing.expect(st.plans.copy.rounds > 0);
    try std.testing.expectEqual(st.drafted, st.plans.copy.drafted);
    try std.testing.expectEqual(st.accepted, st.plans.copy.accepted);
    try std.testing.expectEqual(lanes.stream.Plans.Count{}, st.plans.head);
}

/// the prefill yield through the fake (`yield_rounds`: TF_DS_PREFILL_YIELD's stand-in): a stream decoding when another request's
/// six-chunk prompt pass begins, the decode's tokens noted at each chunk and its cancel sent at chunk 3. Returns what
/// the pass saw.
fn trackD(yield_rounds: bool) !struct { seen: [6]usize, done_at_5: ?Reason, a_end: Reason } {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: usize = 0,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn read(b: *@This()) struct { tokens: usize, done: ?Reason } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ .tokens = b.tokens, .done = b.done };
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                if (b.read().done) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    // A: a long decode, admitted and decoding before B arrives
    const prompt_a = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var a: Box = .{};
    const req_a: Request = .{ .prompt = &prompt_a, .max_tokens = 100000 };
    try e.submit(1, &req_a, .{ .ctx = &a, .event = Box.event });
    while (a.read().tokens == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    // B: a six-chunk prompt pass; at each of its chunks note A's tokens, and cancel A at chunk 3
    const Watch = struct {
        engine: Engine,
        a: *Box,
        seen: [6]usize = @splat(0),
        done_at_5: ?Reason = null,
        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            const r = w.a.read();
            if (chunk < w.seen.len) w.seen[chunk] = r.tokens;
            if (chunk == 3) w.engine.cancel(1);
            if (chunk == 5) w.done_at_5 = r.done;
        }
    };
    var watch = Watch{ .engine = e, .a = &a };
    target.yield_rounds = yield_rounds;
    target.prefill_chunks = 6;
    target.prefill_hook = Watch.call;
    target.prefill_hook_ctx = &watch;
    const prompt_b = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var b: Box = .{};
    const req_b: Request = .{ .prompt = &prompt_b, .max_tokens = 1 };
    try e.submit(2, &req_b, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b.wait());
    return .{ .seen = watch.seen, .done_at_5 = watch.done_at_5, .a_end = a.wait() };
}

test "the prefill yield: a decoding stream keeps getting tokens during another's prompt pass, and its cancel lands before that pass ends" {
    const r = try trackD(true);
    try std.testing.expect(r.seen[2] > r.seen[0]); // a round at each boundary delivered A's tokens
    try std.testing.expectEqual(@as(?Reason, Reason.cancelled), r.done_at_5); // its cancel was taken at boundary 3
    try std.testing.expectEqual(Reason.cancelled, r.a_end);
}

test "the prefill yield off: no round between another's prompt chunks, a cancel waits for the pass" {
    const r = try trackD(false);
    try std.testing.expectEqual(r.seen[0], r.seen[2]);
    try std.testing.expectEqual(r.seen[0], r.seen[5]);
    try std.testing.expectEqual(@as(?Reason, null), r.done_at_5);
    try std.testing.expectEqual(Reason.cancelled, r.a_end); // taken once the pass ended
}

test "the service fence: never inside a round, a prompt pass's chunk or a graph capture; skipped and counted when not panicking" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop(); // (never started: frees the lists)
    host.fence_panic = false;
    try std.testing.expect(host.fenced("service")); // the run loop between rounds, or idle
    host.rounds_open = 1; // inside the lane core's step: a round's launches in flight
    try std.testing.expect(!host.fenced("service"));
    host.rounds_open = 0;
    var job: LaneHost.Job = undefined;
    host.filling_job = &job; // a prompt pass running its chunk
    try std.testing.expect(!host.fenced("service"));
    host.in_between = true; // ... standing between two chunks (Stream.between)
    try std.testing.expect(host.fenced("service"));
    host.rounds_open = 1; // the prefill yield's round inside between
    try std.testing.expect(!host.fenced("service"));
    host.rounds_open = 0;
    host.filling_job = null;
    host.in_between = false;
    const Capture = struct {
        var on = true;
        fn quiet(_: *anyopaque) bool {
            return !on;
        }
    };
    host.quiet = .{ .ctx = &host, .check = Capture.quiet }; // the backend capturing a CUDA graph
    try std.testing.expect(!host.fenced("service"));
    Capture.on = false;
    try std.testing.expect(host.fenced("service"));
    try std.testing.expectEqual(@as(u64, 4), host.service_refused);
}

test "the prefill yield: every round runs inside the fence, the run loop's and the one between two chunks; a prompt chunk refuses a service" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    // the fake, noting what the fence says at every launch a round makes (queue, verify) and at each prompt chunk
    const Spy = struct {
        fake: lanes.fake.Fake,
        host: ?*LaneHost = null,
        loop_calls: usize = 0, // launches by the run loop's rounds (no pass filling)
        loop_open: usize = 0, // of them, with a round open (no service allowed)
        between_calls: usize = 0, // launches by the prefill yield's rounds (a pass standing between two chunks)
        between_open: usize = 0,
        chunks: usize = 0, // chunks begun by the filling pass
        chunk_refused: usize = 0, // of them, refusing a service
        mutex: std.Io.Mutex = .init,

        fn of(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        fn inner(ptr: *anyopaque) lanes.backend.Backend {
            return of(ptr).fake.backend();
        }
        fn backend(f: *@This()) lanes.backend.Backend {
            return .{ .ptr = f, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
        }
        /// A launch: a round's when no pass fills (the run loop) or the pass stands between chunks (the prefill yield).
        fn launch(f: *@This()) void {
            const h = f.host orelse return;
            f.mutex.lockUncancelable(std.testing.io);
            defer f.mutex.unlock(std.testing.io);
            if (h.filling_job == null) {
                f.loop_calls += 1;
                if (h.rounds_open > 0) f.loop_open += 1;
            } else if (h.in_between) {
                f.between_calls += 1;
                if (h.rounds_open > 0) f.between_open += 1;
            }
        }
        fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            return inner(ptr).prefill(s);
        }
        fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            return inner(ptr).first(s, position);
        }
        fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
            of(ptr).launch();
            return inner(ptr).queue(s, feed, position);
        }
        fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
            return inner(ptr).read(handle);
        }
        fn verify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
            of(ptr).launch();
            return inner(ptr).verify(windows, out);
        }
        fn keep(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
            return inner(ptr).keep(windows, paths);
        }
        fn draft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) anyerror!void {
            return inner(ptr).draft(requests);
        }
        fn release(ptr: *anyopaque, s: *lanes.Stream) void {
            inner(ptr).release(s);
        }
        fn chunk(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const f = of(ctx);
            const h = f.host orelse return;
            if (h.filling_job == null) return;
            f.chunks += 1;
            if (!h.fenced("a service in a prompt chunk")) f.chunk_refused += 1;
        }
        fn counts(f: *@This()) [2]usize {
            f.mutex.lockUncancelable(std.testing.io);
            defer f.mutex.unlock(std.testing.io);
            return .{ f.loop_calls, f.between_calls };
        }
    };
    var target: Spy = .{ .fake = .{ .gpa = gpa } };
    defer target.fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.fence_panic = false; // (the chunk's ask is refused on purpose: counted, not a panic)
    target.host = &host;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var a: Box = .{};
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 100000 }, .{ .ctx = &a, .event = Box.event });
    while (target.counts()[0] < 4) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {}; // A decoding
    target.fake.yield_rounds = true;
    target.fake.prefill_chunks = 4;
    target.fake.prefill_hook = Spy.chunk;
    target.fake.prefill_hook_ctx = &target;
    var b: Box = .{};
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 1 }, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b.wait());
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    target.mutex.lockUncancelable(std.testing.io);
    defer target.mutex.unlock(std.testing.io);
    try std.testing.expect(target.loop_calls >= 4);
    try std.testing.expectEqual(target.loop_calls, target.loop_open); // every run-loop round inside the fence
    try std.testing.expect(target.between_calls >= 3); // a round at each of B's three boundaries
    try std.testing.expectEqual(target.between_calls, target.between_open); // and each inside the fence
    try std.testing.expectEqual(@as(usize, 4), target.chunks);
    try std.testing.expectEqual(@as(usize, 4), target.chunk_refused); // inside a chunk: refused
    try std.testing.expectEqual(@as(u64, 4), host.service_refused); // counted, one each
}

test "under the fill order an admitted prompt the order serves first makes the stepping prompt's step wait, as a rotation's arrival does; one it serves later does not" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const long_prompt: [200]u32 = @splat(3);
    const short_prompt: [12]u32 = @splat(4);
    const big_prompt: [500]u32 = @splat(5);
    const rl: Request = .{ .prompt = &long_prompt, .max_tokens = 4, .drafts = false };
    const rs: Request = .{ .prompt = &short_prompt, .max_tokens = 4, .drafts = false };
    const rb: Request = .{ .prompt = &big_prompt, .max_tokens = 4, .drafts = false };
    for ([_]bool{ true, false }) |ordered| {
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4, .resume_rows = 4, .fill = if (ordered) api.fill_order.parse("serial", null, null).policy else .{} });
        defer host.stop();
        var me: LaneHost.Job = .{ .host = &host, .id = 1, .request = &rl, .sink = undefined, .started = true };
        me.stream = try lanes.Stream.init(gpa, .{ .id = "m", .prompt = &long_prompt, .max_new = 4, .eos = &.{96} });
        defer me.stream.deinit(gpa);
        var small: LaneHost.Job = .{ .host = &host, .id = 2, .request = &rs, .sink = undefined, .rows_est = 12 };
        var large: LaneHost.Job = .{ .host = &host, .id = 3, .request = &rb, .sink = undefined, .rows_est = 500 };
        // (admitted just now: none has waited its maximum, so the order ranks them by rows)
        const now = host.clockNs();
        me.stepped_at = now;
        small.stepped_at = now;
        large.stepped_at = now;
        try host.admitted.append(gpa, &me);
        try host.admitted.append(gpa, &small);
        // a smaller prompt waiting for its first step: the rotation and the order both make this step wait
        var f = LaneHost.betweenFacts(&me);
        try std.testing.expect(f.waiting);
        // a larger one: the rotation waits as before; the order serves this prompt first, so it fills beside it
        host.admitted.items[1] = &large;
        f = LaneHost.betweenFacts(&me);
        try std.testing.expectEqual(!ordered, f.waiting);
        try std.testing.expectEqual(ordered, f.filling);
    }
}

test "the share's decoders are the streams a round runs: a paused or finished stream is no decoder" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop(); // (never started: frees the lists)
    try std.testing.expectEqual(@as(usize, 0), host.runnable());
    var d = try lanes.Stream.init(gpa, .{ .id = "d", .prompt = &.{ 3, 1, 4, 1 }, .max_new = 8, .eos = &.{96} });
    defer d.deinit(gpa);
    try core.addStream(&d);
    defer core.backend.release(&d);
    try std.testing.expectEqual(@as(usize, 1), host.runnable());
    d.paused = true;
    try std.testing.expectEqual(@as(usize, 0), host.runnable());
    try std.testing.expectEqual(@as(usize, 1), core.activeCount()); // (paused ones are active, not decoders)
    d.paused = false;
    d.finished = true;
    try std.testing.expectEqual(@as(usize, 0), host.runnable());
}

test {
    _ = @import("lane_host_test.zig");
    _ = @import("lane_host_tier_test.zig");
}

}
