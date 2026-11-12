//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const share_mod = @import("lane_host_share.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Reason = api.Reason;
const Stats = api.Stats;

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
    filling: std.ArrayList(*Job) = .empty, // admitted jobs whose prompt is still going in, a chunk a round, oldest first
    fill_since: u32 = 0, // the rounds since a prompt pass's last step (Info.fill_rounds)
    cancels: std.ArrayList(Id) = .empty,
    closing: bool = false,
    stale: bool = false, // the learner's last step: a put-back has not reached the device, so replies are refused
    thread: ?std.Thread = null,
    decoded: std.ArrayList(status.Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_tokens: std.ArrayList(u32) = .empty,
    /// A Metal engine's keepalive target, set by the family that owns the queue; null keeps the ticker off.
    keepalive_target: ?api.keepalive.Target = null,
    live_generated: u64 = 0,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    recall: ?*lanes.recall_store.Store = null, // Recall's record: drafts from earlier requests (lane_host_recall.zig)
    decoded_rows: bool = false, // the backend prefills Request.decode_spans with decoded rows' arithmetic, cache or not
    memory: ?api.MemorySource = null, // the backend's memory counts; null: Engine.memory reports none
    graph_fallbacks: ?*const std.atomic.Value(u64) = null, // the backend's failed graph captures (read at any time)
    head_rounds: ?*const [6]std.atomic.Value(u64) = null, // the backend's decode rounds by the head's path (read at any time)
    draft_vocab: ?*const api.DraftVocabCounters = null, // the backend's drafter on a pruned vocabulary (read at any time)
    explain: ?api.Explain = null, // the backend's words for a request it refuses; null: the error's name
    learner: ?api.Learner = null, // the family's Sliding Weights learner; null: learn requests are refused
    lessons: std.ArrayList(Lesson) = .empty, // learn requests waiting for an idle engine
    lesson_open: bool = false, // the learner holds a begun job (engine thread only)
    admissions: u64 = 0, // jobs admitted so far: each job's `seq`
    memory_lanes: ?usize = null, // lanes the backend had memory for at its last LaneMemory, until a job ends
    warm_quiet: u64 = 0, // ns a warm pass waits for other work before its first chunk (0: none), set by its family
    disk: rows_host.Disk = .{}, // the Mac app's questions about kept states, and the disk writer's quiet
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
    /// What every rank was last told for each slot of class A (`unset`: nothing yet, the start's): a cfg frame goes out only when a
    /// slot's published value differs.
    live_sent: [live_config.slot_count]u64 = @splat(live_config.unset),

    const Lesson = struct { request: *const api.LearnRequest, sink: api.LearnSink };
    const full_retry_ns: i96 = std.time.ns_per_s;
    const long_pass_ns: i96 = std.time.ns_per_s;
    const finish_ns: i96 = 5 * std.time.ns_per_s;
    const finish_left: i64 = 256;

    pub const Job = struct {
    /// A clock for the share's timing: awake nanoseconds.
    pub const ShareClock = struct { ctx: *anyopaque, now: *const fn (ctx: *anyopaque) i96 };

        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        recall: recalls.Slot = .{}, // its reads of Recall's record (lane_host_recall.zig)
        delivered: usize = 0,
        started: bool = false,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        fill_began: bool = false, // its first prompt chunk ran
        began: i96 = 0,
        rows_est: u64 = 0, // the rows its pass computes, as the cache stood when it was admitted (the fill order, before its pass starts)
        overtaken: u32 = 0, // prompt steps that went to a smaller prompt since this one's last step (the fill order's bound)
        stepped_at: i96 = 0, // when its last prompt step began, or when it was admitted (the fill order's maximum wait)
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up
        images: pc.Images = .none, // the images key of the prompt's prefixes (0 for a prompt without images)

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            if (!cacheable(job.request)) return;
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
            if (!job.started) return store.release(e);
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            const ok = !job.stream.reuse_failed and job.stream.cached == e.at;
            if (job.stream.reuse_failed or ok) store.resumed(e, job.request.cacheKey(), ok) else store.release(e);
        }
    };

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

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        var share = share_mod.Share.fromEnv();
        share.rounds = info_.rounds_beside;
    }

    pub fn start(h: *LaneHost) !void {
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

    /// Stops admitting, cancels what is left and joins the engine thread.
    pub fn stop(h: *LaneHost) void {
        rows_host.drainThread(h);
        rows_host.close(h);
        const closed: api.LearnEvent = .{ .done = .{ .message = "the engine closed" } };
        for (h.lessons.items) |l| l.sink.event(l.sink.ctx, &closed);
        h.lessons.deinit(h.gpa);
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.fill_slots.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{
            .info = infoFn,
            .submit = submitFn,
            .cancel = cancelFn,
            .status = status.read,
            .memory = memoryFn,
            .keepalive = keepaliveFn,
            .learn = learn.submit,
            .recall = recalls.attach,
            .imprint = rows_host.ask,
            .drain = rows_host.drain,
        } };
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

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) api.SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing or h.stale) {
            h.gpa.destroy(job);
            return if (h.closing) error.Closed else error.Stale;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        while (!request.background and at > 0 and h.queued.items[at - 1].request.background) at -= 1;
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
            .graph_fallbacks = if (h.graph_fallbacks) |c| c.load(.monotonic) else null,
            .head_rounds = if (h.head_rounds) |c| .{ c[0].load(.monotonic), c[1].load(.monotonic), c[2].load(.monotonic), c[3].load(.monotonic), c[4].load(.monotonic), c[5].load(.monotonic) } else null,
            .draft_vocab = if (h.draft_vocab) |c| c.read() else null,
            .streams = n,
            .generation_tokens = h.live_generated,
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    pub fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// A background pass's yield hook: a foreground request waits or a stream decodes (read under the lock).
    fn yieldsTo(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.foregroundWaiting() or job.host.core.activeCount() > 0;
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

    }

    /// Hands a stream the tokens its rounds committed since the last delivery (their logprob rows first).
    pub fn send(h: *LaneHost, job: *Job) void {
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            if (job.stream.logprobs != null) emit(job, .{ .logprobs = job.stream.rows.items[job.delivered..emitted.len] });
            emit(job, .{ .tokens = emitted[job.delivered..] });
            status.noteDecoded(h, emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished and left the lists a status read walks.
    pub fn deliver(h: *LaneHost, job: *Job) bool {
        if (!job.started or job.prefilled == null) return false;
        h.send(job);
        if (!job.stream.finished) return false;
        h.remove(job);
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        h.finish(job, reason, job.stream.problem);
        return true;
    }

    pub fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        h.memory_lanes = null; // a lane may have freed: the next admission asks the backend for memory again
        job.reported();
        h.gpa.free(job.marks);
        const s = &job.stream;
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .plans = s.plans, .min_rows = s.min_rows, .loop_period = s.loop_period, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            if (s.grammar) |g| g.deinit(h.gpa);
            s.deinit(h.gpa);
            job.proposer.deinit();
            recalls.release(job);
        }
        h.gpa.destroy(job);
    }

    pub fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    pub fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    /// Prefills the next queued request into a free lane; false when none waits or no lane is free.
    fn admitOne(h: *LaneHost) bool {
        h.lock();
        // a stale device takes no reply: `learn.refuse` ends the queued ones instead
        if (h.stale or h.queued.items.len == 0 or h.admitted.items.len >= @min(h.info_.lanes, h.memory_lanes orelse h.info_.lanes)) {
            h.unlock();
            return false;
        }
        // a backend that holds one prompt pass at a time admits the next request once the pass ends: its lookup then
        // finds what that pass kept, and nothing keeps before its own pass, as with a pass run whole
        if (h.core.backend.vtable.prefill_one and h.filling.items.len > 0) {
            h.unlock();
            return false;
        }
        if (h.full_wait) |n| {
            const active = h.core.activeCount();
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
        const at = admit.pick(h) orelse { // a shared cache pool's head waits for room
            h.unlock();
            return false;
        };
        const job = h.queued.orderedRemove(at);
        job.seq = h.admissions;
        h.admissions += 1;
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
        if (r.background and r.max_tokens == 0 and h.warm_quiet > 0 and !h.quiet(job)) return h.cancel(job); // other work came first: it never ran
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
            .recall = recall,
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .yield_check = if (r.background and r.max_tokens == 0) .{ .ptr = job, .check = yieldsTo } else null,
            .share_check = .{ .ptr = job, .check = othersWait },
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .logprobs = r.logprobs,
            .prompt_logprobs = r.prompt_logprobs,
            .loop_guard = r.loop_guard,
            .loop_windows = r.loop_windows,
            .chunks = r.chunks,
            .history_len = r.history_len,
            .shared_prefixes = r.shared_prefixes,
            .reuse = reuse,
            .grammar = if (r.structure) |st| h.constraint(st) catch |e| {
                job.proposer.deinit();
                recalls.release(job);
                return h.drop(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
            } else null,
            .decode_spans = h.spans(r),
            .decide = r.decide,
            .untaught = r.untaught,
            .images = r.images,
            .key = r.cache_key,
            .lane_images = r.lane_images,
        }) catch {
            job.proposer.deinit();
            _ = h.drop(job, "out of memory");
            return false;
        };
        job.stream.between = .{ .ptr = job, .run = between, .ready = betweenReady, .next = betweenNext, .facts = betweenFacts };
        job.started = true;
        job.fill_on.store(true, .release); // status readers count the fill from here until the pass reports
        const began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        job.began = began;
        if (r.max_tokens == 0 and h.info_.warm_turns) return h.warmPass(job, began);
        if (lone.fits(h, job)) return lone.run(h, job, began, false);
        if (h.core.fills()) {
            h.filling.append(h.gpa, job) catch return h.drop(job, "out of memory");
            return true;
        }
        h.core.addStream(&job.stream) catch |e| return switch (e) {
            error.Cancelled => h.cancel(job),
            error.LaneMemory => h.requeue(job),
            error.ContextFull => if (h.core.activeCount() > 0) h.requeueFull(job) else h.drop(job, h.words(e)),
            else => h.drop(job, h.words(e)),
        };
        h.prefilled(job, began);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        if (now - began >= long_pass_ns) {
            h.lock();
            h.finish_until = now + finish_ns;
            h.unlock();
        }
        _ = h.deliver(job);
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

    /// Whether an admitted stream decodes within finish_left tokens of its max_tokens (the lock held).
    fn nearEnd(h: *LaneHost) bool {
        for (h.admitted.items) |job| if (job.started and !job.stream.finished and job.stream.budgetLeft() <= finish_left) return true;
        return false;
    }

    pub fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        job.fill_on.store(false, .release); // the pass reported: status readers stop counting this fill
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        h.unlock();
        rows_host.prefilled(h, job.request.prompt.len - job.stream.cached, done - began);
        job.reported();
        if (h.cache) |store| store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
        emit(job, .{ .prefilled = job.stream.cached });
        if (job.stream.prompt_logprobs != null) emit(job, .{ .prompt_logprobs = job.stream.prompt_rows.items });
        recalls.prompted(h, job);
    }

    /// A request's decoded prompt rows, for a backend that keeps their bits (`decoded_rows`); otherwise none.
    fn spans(h: *const LaneHost, r: *const Request) []const [2]u32 {
        return if (h.decoded_rows) r.decode_spans else &.{};
    }

    /// A request the prompt cache can key: a lane image's positions hold its placeholder, so its rows count only with
    /// a key that spells them (the server's, one word a position).
    fn keyed(r: *const Request) bool {
        return r.lane_images.len == 0 or r.cache_key.len == r.prompt.len;
    }

    /// A prompt-only pass on a `warm_turns` engine: the backend prefills, the cache keeps its end (a mark, if keyed on)
    fn warmPass(h: *LaneHost, job: *Job, began: i96) bool {
        const be = h.core.backend;
        be.prefill(&job.stream) catch |e| { // a yield ends it like a cancel: what it prefilled is kept
            if (e != error.Cancelled and e != error.Yielded) return h.drop(job, h.words(e));
            be.release(&job.stream);
            return h.cancel(job);
        };
        h.prefilled(job, began);
        if (h.cache) |store| _ = if (!job.request.untaught and store.rules.lookahead == 0 and keyed(job.request)) store.keepBorn(job.request.cacheKey(), job.request.bornLen(), @intCast(job.request.prompt.len), &job.stream, job.request.chunks, h.spans(job.request)) else false;
        be.release(&job.stream);
        h.remove(job);
        h.finish(job, .length, "");
        return true;
    }

    pub fn words(h: *const LaneHost, e: anyerror) []const u8 {
        const x = h.explain orelse return @errorName(e);
        return x.text(x.ctx, e) orelse @errorName(e);
    }

    pub fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        h.remove(job);
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// A request whose prompt pass the backend's pool cannot hold while other streams run (error.ContextFull, before
    /// anything of it was placed) goes back to the queue's front and waits until one of them ends, as the Python
    /// lane's window waits, instead of failing, or a second at most (the backend may place it by then). Returns
    /// false: nothing more is admitted now.
    fn requeueFull(h: *LaneHost, job: *Job) bool {
        h.core.discard(&job.stream);
        job.entry = null; // nothing was restored: the store keeps the entry for the next lookup
        h.gpa.free(job.marks);
        job.marks = &.{};
        if (job.stream.grammar) |g| g.deinit(h.gpa); // the next admission compiles its own
        job.stream.deinit(h.gpa);
        job.proposer.deinit();
        recalls.release(job);
        job.started = false;
        job.fill_began = false;
        job.began = 0;
        job.rows_est = 0;
        job.overtaken = 0;
        job.stepped_at = 0;
        job.prefilled = null;
        job.delivered = 0;
        job.prefill_sent = false;
        h.gpa.free(job.marks);
        job.marks = &.{};
        return false;
    }

    /// No memory for another lane yet: the job waits at the queue's head, and admission holds until a job ends.
    fn requeue(h: *LaneHost, job: *Job) bool {
        h.lock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            break;
        };
        const others = h.admitted.items.len;
        const waits = others > 0 and blk: { // with nothing else running no memory comes back: it would wait forever
            h.queued.insert(h.gpa, 0, job) catch break :blk false;
            break :blk true;
        };
        h.unlock();
        if (!waits) {
            h.finish(job, .failed, if (others > 0) "out of memory" else alone_words);
            return true;
        }
        job.fill_on.store(false, .release); // off the lists before the teardown a status reader could have walked into
        job.reported();
        if (job.stream.grammar) |g| g.deinit(h.gpa); // the next admission compiles its own
        job.stream.deinit(h.gpa);
        job.proposer.deinit();
        recalls.release(job);
        job.started = false;
        job.fill_began = false;
        job.entry = null; // looked up again at its next admission
        h.gpa.free(job.marks);
        job.marks = &.{};
        h.memory_lanes = others;
        return false;
    }

    const alone_words = "the engine has no memory for this request's caches with nothing else running: lower --context or --parallel";

    /// A job cancelled in its prompt pass, its lane already released.
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    pub fn remove(h: *LaneHost, job: *Job) void {
        fill.unfill(h, job);
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
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
    fn run(h: *LaneHost) void {
        while (true) {
            ends.takeCancels(h);
            rows_host.answer(h);
            learn.refuse(h);
            while (h.admitOne()) {}
            h.fillOne();
            status.noteLive(h);
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len;
                h.unlock();
                if (h.lesson_open) {
                    h.learner.?.abort(h.learner.?.ctx);
                    h.lesson_open = false;
                }
                if (left == 0) return rows_host.flush(h);
                ends.closeAll(h);
                continue;
            }
            if (h.core.activeCount() == 0) {
                if (h.filling.items.len > 0) {
                    h.unlock();
                    continue;
                }
                if (learn.ready(h)) {
                    h.unlock();
                    learn.step(h);
                    continue;
                }
                if (rows_host.idle(h)) continue;
                if (h.cancels.items.len == 0 and (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes))
                    h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
                h.unlock();
                continue;
            }
            h.unlock();
            h.core.step() catch |e| {
                ends.failAll(h, @errorName(e));
                continue;
            };
            h.fill_since +|= 1;
            var i: usize = 0;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                if (h.deliver(job)) continue; // it left the list: the next job slid into slot i
            if (job.started and !job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .failed, message);
        }
        h.gpa.free(jobs);
    }

    fn closeAll(h: *LaneHost) void {
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
                i += 1;
            }
            lone.reclaim(h);
        }
        // with a share below 1 the round's time pays the debt a long prompt's step left
        const shared = h.share.p < 1; // (the round's time pays a debt of time; a count of rounds needs no clock)
        const r0: i96 = if (shared) h.shareNow() else 0;
        defer if (shared) h.share.rounded(@intCast(@max(0, h.shareNow() - r0))) else h.share.rounded(0);
    }
};

/// A stream's head chains by depth as a reply's stats carry them (a depth past the API's counts adds to its last).
fn depthsOf(s: *const lanes.Stream) [api.draft_depths]api.DraftDepth {
    var out: [api.draft_depths]api.DraftDepth = @splat(.{});
    for (s.depths, 0..) |d, j| {
        const o = &out[@min(j, out.len - 1)];
        o.drafted += d.drafted;
        o.reached += d.reached;
        o.kept += d.kept;
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
}

}
