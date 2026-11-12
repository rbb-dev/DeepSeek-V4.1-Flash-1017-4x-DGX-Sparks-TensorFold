//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const status = @import("lane_host_status.zig");
const learn = @import("lane_host_learn.zig");
const lone = @import("lane_host_lone.zig");
const fill = @import("lane_host_fill.zig");
const admit = @import("lane_host_admit.zig");
const recalls = @import("lane_host_recall.zig");
const ends = @import("lane_host_end.zig");
const rows_host = @import("lane_host_rows.zig");
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

    const Lesson = struct { request: *const api.LearnRequest, sink: api.LearnSink };
    const full_retry_ns: i96 = std.time.ns_per_s;
    const long_pass_ns: i96 = std.time.ns_per_s;
    const finish_ns: i96 = 5 * std.time.ns_per_s;
    const finish_left: i64 = 256;

    pub const Job = struct {
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
        /// The engine thread says the prompt pass is going: status readers may count this job's fill atomics.
        fill_on: std.atomic.Value(bool) = .init(false),
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
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced };
    }

    pub fn start(h: *LaneHost) !void {
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
        h.filling.deinit(h.gpa);
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

    /// Whether work besides this job's prompt pass waits (a queued request, another pass, a decoding stream, a cancel):
    /// a stepped pass then ends its step after its chunks, else it runs on (asked on the engine thread, between chunks).
    fn othersWait(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        const h = job.host;
        h.lock();
        defer h.unlock();
        return h.queued.items.len > 0 or h.filling.items.len > 1 or h.core.activeCount() > 0 or h.cancels.items.len > 0 or h.closing;
    }

    /// A foreground request is queued (they queue ahead of every background one); the caller holds the lock.
    pub fn foregroundWaiting(h: *const LaneHost) bool {
        return h.queued.items.len > 0 and !h.queued.items[0].request.background;
    }

    /// Waits `warm_quiet` for other work (a request, a decoding stream, a cancel): true if none came.
    fn quiet(h: *LaneHost, job: *const Job) bool {
        const until: std.Io.Timeout = .{ .deadline = .fromNow(h.io, .{ .raw = .fromNanoseconds(h.warm_quiet), .clock = .awake }) };
        h.lock();
        defer h.unlock();
        while (!h.closing and !h.foregroundWaiting() and h.core.activeCount() == 0 and std.mem.indexOfScalar(Id, h.cancels.items, job.id) == null) {
            h.wake.waitTimeout(h.io, &h.mutex, until) catch |e| return e == error.Timeout;
        }
        return false;
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
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .depths = depthsOf(s), .min_rows = s.min_rows, .loop_period = s.loop_period, .loop_novelty = s.loop_novelty, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        const told = if (job.started) recalls.finished(h, job, reason, stats) else stats;
        emit(job, .{ .finished = .{ .reason = reason, .stats = told, .message = message } });
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
        h.unlock();
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

    /// The next filling job's next prompt chunk: the other streams' rounds run between chunks, not after the prompt.
    /// A pass that began waits out Info.fill_rounds of those rounds between two of its steps (none when none decodes).
    fn fillOne(h: *LaneHost) void {
        if (!h.core.fills()) return;
            return;
        };
        if (!done) return;
        _ = h.filling.orderedRemove(at);
        h.prefilled(job, job.began);
        _ = h.deliver(job);
        const job = h.admitted.items[i];
        h.fill_next = i + 1;
        const first = !job.started;
            if (e == error.Cancelled) {
                _ = h.cancel(job);
            } else if (e == error.ContextFull and h.admitted.items.len > 1) {
                _ = h.requeue(job);
            } else {
                _ = h.drop(job, h.words(e));
            }
            return;
        };
        job.reported(); // the first step restored it; no dangling cache entry across another prompt's keep
        if (done) {
            h.prefilled(job, job.began);
        }
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
    }
    return out;
}

test {
    _ = @import("lane_host_test.zig");
}

}
