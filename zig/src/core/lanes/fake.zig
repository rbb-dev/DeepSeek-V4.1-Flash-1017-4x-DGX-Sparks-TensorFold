//! A host-only test backend whose tokens depend on all fed history, so a rollback or position mistake changes them.
const std = @import("std");
const Allocator = std.mem.Allocator;
const be = @import("backend.zig");
const stream_mod = @import("stream.zig");
const Stream = stream_mod.Stream;
const Sampling = @import("sampling.zig").Sampling;
const keyBits = @import("sampling.zig").keyBits;
const accept = @import("accept.zig");
const Row = @import("logprob.zig").Row;
const grammar = @import("grammar.zig");

pub const vocab = 97;

/// The fake target's draw after `history` at `position` (keyed noise when sampled).
pub fn next(history: []const u32, s: ?Sampling, position: u64) u32 {
    var sum: u64 = 0;
    for (history) |t| sum +%= t;
    var token = (sum *% 7 +% history.len *% 3) % vocab;
    if (s) |st| token = (token + keyBits(st.seed, position, 0) % 5) % vocab;
    return @intCast(token);
}

/// The fake target's row after `history` for `pick`, a function of the history as logits are; best ids from the pick.
pub fn rowAt(history: []const u32, pick: u32, count: u8) Row {
    var sum: u64 = 0;
    for (history) |t| sum +%= t;
    const base = -@as(f32, @floatFromInt(sum % 50)) / 64.0;
    var r: Row = .{ .token = pick, .logprob = base, .count = count };
    for (0..count) |i| {
        r.ids[i] = @intCast((pick + i) % vocab);
        r.logprobs[i] = base - @as(f32, @floatFromInt(i));
    }
    return r;
}

const Lane = struct {
    history: std.ArrayList(u32) = .empty, // the stream's cache: prompt and fed tokens
    held: std.ArrayList(u32) = .empty, // drafts for the next round
    base: usize = 0, // the history length before the last verify
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    filled: usize = 0, // prompt chunks a stepped prefill has run
    filling: bool = false,
};

pub const Fake = struct {
    gpa: Allocator,
    lanes: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty,
    drawn: std.ArrayList(u32) = .empty, // handle -> token
    rounds: u64 = 0,
    cycle_after: ?usize = null,
    probe_prompt: usize = 0,
    pattern: []const u32 = &.{ 11, 12, 13 },
    // the reply's tokens at these indexes are 5 instead of the pattern's (a copy of the pattern fails there)
    holes: []const usize = &.{},
    answer_cycles: bool = false,
    prefill_chunks: usize = 0,
    prefill_count: usize = 0,
    prefill_hook: ?*const fn (ctx: *anyopaque, s: *Stream, chunk: usize) void = null,
    prefill_hook_ctx: ?*anyopaque = null,
    refuse_sampled: bool = false, // prefill refuses a sampled stream with error.SamplingRefused
    refuse_full: usize = 0, // prefill refuses this many times with error.ContextFull while another lane is live
    /// TF_DS_PREFILL_YIELD=1's stand-in: between two chunks the host's round part runs (Stream.between, when it says a
    /// round would run); its tier part runs at every boundary it is wanted, with or without this.
    yield_rounds: bool = false,
    /// The chunked backend's steps sized by the host's facts (stream.passRows over this grid) and a pass length without
    /// marks; each step logged in `steps` with its facts (read after prefill_hook) and, short of the prompt's end, the
    /// host's next. Null: four rows a step, nothing logged.
    pass_rule: ?PassRule = null,
    steps: std.ArrayList(StepSeen) = .empty,
    decodes: u64 = 0, // verify and queue calls (a round's work, pipelined or verified)
    /// The host's forward-wait work (Backend.VTable.overlap), run at the start of every verify as the real backend runs it inside the
    /// wait; `verify_probe` is called just before and just after it (a test reads what the host had delivered at each).
    overlap: ?be.Backend.Overlap = null,
    verify_probe: ?*const fn (ctx: *anyopaque, after_work: bool) void = null,
    verify_probe_ctx: ?*anyopaque = null,

    ranked: bool = false, // the head gives its held chain back (alternatives)
    memory_lanes: ?usize = null, // prefill past this many live lanes fails with error.LaneMemory (no room yet)
    fail_round: ?u64 = null, // this round's verify fails with error.GpuFailed, as a round the GPU fails does
    last_rows: usize = 0, // the last verify's rows, for test clocks
    last_sampled: bool = false, // and whether one of its streams drew keyed samples

    /// The fake's pass rule: its grid, the cap and pass_from, and whether another prompt filling keeps whole passes
    /// (TF_DS_PASS_FILLING's passage of stream.passRows).
    pub const PassRule = struct { resume_rows: u32, pass_rows: u32, pass_from: usize = 0, filling_pass: bool = false };
    /// One logged step: the stream's prompt length (which request), where it started, its rows, the host's facts at its
    /// start and next at its end, and the decode calls (verify or queue: a round's) served before it.
    pub const StepSeen = struct { prompt_len: usize, start: usize, rows: usize, facts: ?stream_mod.Facts, next: ?stream_mod.Next, decodes: u64 };

    pub fn deinit(x: *Fake) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| freeLane(x.gpa, l);
        x.lanes.deinit(x.gpa);
        x.drawn.deinit(x.gpa);
        x.steps.deinit(x.gpa);
    }

    /// A pass's rows without marks: a remainder under the resume rows alone, else whole resume units up to the cap.
    pub fn passLen(start: usize, len: usize, cap: u32, resume_rows: u32) usize {
        const rest = len - start;
        if (rest < resume_rows) return rest;
        return @min(cap, rest / resume_rows * resume_rows);
    }

    fn freeLane(gpa: Allocator, l: *Lane) void {
        l.history.deinit(gpa);
        l.held.deinit(gpa);
        l.rows.deinit(gpa);
    }

    pub fn backend(x: *Fake) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .cancel_prefill = cancelPrefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .overlap = setOverlap, .keep = keep, .draft = draft, .held = heldDrafts, .release = release } };
    }

    /// The same target, advancing four prompt tokens per host visit.
    pub fn chunkedBackend(x: *Fake) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .prefill_step = prefillStep, .cancel_prefill = cancelPrefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .overlap = setOverlap, .keep = keep, .draft = draft, .held = heldDrafts, .release = release } };
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream) anyerror!bool {
        const x = self(ptr);
        if (x.refuse_sampled and s.sampling != null) return error.SamplingRefused;
        if (s.isCancelled()) return error.Cancelled;
        if (x.refuse_full > 0 and x.lanes.count() > 0 and !x.lanes.contains(s)) {
            x.refuse_full -= 1;
            return error.ContextFull;
        }
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (!got.found_existing) {
            got.value_ptr.* = .{};
            s.cached = 0;
            if (s.reuse.saved) |saved| {
                const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
                try got.value_ptr.history.appendSlice(x.gpa, kept.items);
                s.cached = s.reuse.at;
            }
            if (s.progress) |p| p.begin(s.cached); // the rows a backend reports as its chunks finish (Stream.progress)
        }
        const h = &got.value_ptr.history;
        const begin = h.items.len;
        x.prefill_count += 1;
        if (x.prefill_hook) |hook| hook(x.prefill_hook_ctx.?, s, begin / 4);
        var rows: usize = 4;
        var facts: ?stream_mod.Facts = null;
        if (x.pass_rule) |r| {
            facts = s.hostFacts();
            rows = passLen(begin, s.prompt_len, stream_mod.passRows(facts, begin, r.pass_from, r.pass_rows, r.resume_rows, r.filling_pass), r.resume_rows);
        }
        const end = @min(begin + rows, s.prompt_len);
        for (s.prompt()[begin..end], begin + 1..) |t, at| {
            try h.append(x.gpa, t);
            if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(at));
        }
        if (s.progress) |p| p.reach(end);
        if (x.pass_rule != null) try x.steps.append(x.gpa, .{ .prompt_len = s.prompt_len, .start = begin, .rows = end - begin, .facts = facts, .next = if (end < s.prompt_len) s.hostNext() else null, .decodes = x.decodes });
        return end == s.prompt_len;
    }

    /// `stepped` as a backend that holds one prompt pass at a time (`prefill_one`).
    pub fn steppedOne(x: *Fake) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .features = features, .release = release, .first_row = firstRow, .reports_fill = true, .prefill_one = true } };
    }

    fn self(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }

    fn lane(x: *Fake, s: *Stream) *Lane {
        return x.lanes.getPtr(s).?;
    }

    fn draw(x: *Fake, l: *Lane, s: *Stream, position: u64) !u64 {
        if (position != l.history.items.len) return error.PositionMismatch;
        try x.drawn.append(x.gpa, x.targetNext(l.history.items, s.sampling, position));
        return x.drawn.items.len - 1;
    }

    fn targetNext(x: *Fake, history: []const u32, sampling: ?Sampling, position: u64) u32 {
        if (x.cycle_after) |start| if (history.len >= x.probe_prompt) {
            const reply = history[x.probe_prompt..];
            if (std.mem.indexOfScalar(u32, reply, 91)) |close| {
                if (reply.len == close + 1) return 92;
                const after = reply.len - close - 2;
                if (x.answer_cycles) {
                    const answer = [_]u32{ 21, 22, 23 };
                    return answer[after % 3];
                }
                const answer = [_]u32{ 40, 41, 42, 96 };
                return answer[@min(after, 3)];
            }
            if (reply.len >= start) {
                for (x.holes) |h| if (reply.len == h) return 5;
                return x.pattern[(reply.len - start) % x.pattern.len];
            }
        };
        return next(history, sampling, position);
    }

    fn value(x: *Fake, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| x.drawn.items[h],
            .value => |v| v,
        };
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        if (x.refuse_sampled and s.sampling != null) return error.SamplingRefused;
        if (x.refuse_full > 0 and x.lanes.count() > 0 and !x.lanes.contains(s)) {
            x.refuse_full -= 1;
            return error.ContextFull;
        }
        if (x.memory_lanes) |n| if (!x.lanes.contains(s) and x.lanes.count() >= n) return error.LaneMemory;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (got.found_existing) freeLane(x.gpa, got.value_ptr);
        got.value_ptr.* = .{};
        const chunks = if (x.prefill_chunks == 0) 1 else x.prefill_chunks;
        // the kept state's tokens stand in for the prompt's first `at`, in place before the first chunk as a real
        // backend restores it: the host may report the state at the first boundary and a keep there may free it
        s.cached = 0;
        if (s.reuse.saved) |saved| {
            const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
            try got.value_ptr.history.appendSlice(x.gpa, kept.items);
            s.cached = s.reuse.at;
        }
        // the rows a backend reports as its chunks finish (Stream.progress): from the kept state's end, evenly
        const resumed: usize = s.cached;
        if (s.progress) |p| p.begin(resumed);
        for (0..chunks) |chunk| {
            x.prefill_count += 1;
            s.fill_at.store(@intCast(s.prompt_len * (chunk + 1) / chunks), .monotonic);
            if (x.prefill_hook) |hook| hook(x.prefill_hook_ctx.?, s, chunk);
            if (s.isCancelled()) return error.Cancelled;
            if (s.progress) |p| p.reach(resumed + (s.prompt_len - resumed) * (chunk + 1) / chunks);
            // the host's turn between two chunks, as dsv41's prompt pass offers it: the prefill yield's round (TF_DS_PREFILL_YIELD's
            // stand-in, when a round would run), then the tier's service whether or not the round ran
            if (chunk + 1 < chunks) if (s.between) |b| {
                if (x.yield_rounds and b.ready(b.ptr)) try b.run(b.ptr);
                if (b.tier) |t| if (t.wanted(b.ptr)) t.run(b.ptr);
            };
        }
        const h = &x.lane(s).history; // (looked up again: the host's turns between chunks ran other streams' rounds)
        for (s.prompt()[h.items.len..], h.items.len + 1..) |t, at| {
            try h.append(x.gpa, t);
            if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(at));
        }
    }

    /// A copy of the stream's cache at `at` tokens (the prompt pass stands there), for core/prompt_cache.zig.
    pub fn save(x: *Fake, s: *Stream, at: u32) !*anyopaque {
        const l = x.lane(s);
        if (l.history.items.len != at) return error.NotAtMark;
        const kept = try x.gpa.create(std.ArrayList(u32));
        kept.* = .empty;
        try kept.appendSlice(x.gpa, l.history.items);
        return kept;
    }

    pub fn drop(x: *Fake, saved: *anyopaque) void {
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        kept.deinit(x.gpa);
        x.gpa.destroy(kept);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        return x.draw(x.lane(s), s, position);
    }

    fn firstMasked(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const x = self(ptr);
        const h = try x.draw(x.lane(s), s, position);
        x.drawn.items[h] = try allowed(x.drawn.items[h], mask);
        return h;
    }

    /// Structured output's draw: the first allowed token from the target's own (a function of the row and its bits).
    fn allowed(token: u32, mask: []const u32) !u32 {
        if (mask.len == 0) return token;
        for (0..vocab) |i| {
            const t: u32 = @intCast((token + i) % vocab);
            if (grammar.allows(mask, t)) return t;
        }
        return error.NoAllowedToken;
    }

    fn heldDrafts(ptr: *anyopaque, s: *Stream, out: []u32) anyerror!void {
        const h = self(ptr).lane(s).held.items;
        if (out.len > h.len) return error.NotHeld;
        @memcpy(out, h[0..out.len]);
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        x.decodes += 1;
        const l = x.lane(s);
        try l.history.append(x.gpa, x.value(feed));
        return x.draw(l, s, position);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).drawn.items[handle];
    }

    fn setOverlap(ptr: *anyopaque, work: ?be.Backend.Overlap) void {
        self(ptr).overlap = work;
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        x.rounds += 1;
        if (x.fail_round == x.rounds) return error.GpuFailed;
        x.last_rows = 0;
        x.last_sampled = false;
        x.decodes += 1;
        if (x.verify_probe) |probe| probe(x.verify_probe_ctx.?, false);
        if (x.overlap) |o| o.run(o.ctx);
        if (x.verify_probe) |probe| probe(x.verify_probe_ctx.?, true);
        for (windows, out) |w, o| {
            x.last_rows += w.rows();
            x.last_sampled = x.last_sampled or w.stream.sampling != null;
            const l = x.lane(w.stream);
            l.base = l.history.items.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(x.gpa, w.pending);
            try l.rows.appendSlice(x.gpa, l.held.items[0..w.held]);
            try l.rows.appendSlice(x.gpa, w.tokens);
            @memcpy(o.drafts, l.rows.items[1..]);
            const parents = try accept.rowParents(x.gpa, l.rows.items.len, w.parents);
            defer x.gpa.free(parents);
            for (0..l.rows.items.len) |r| {
                // each row reads the cache and its own path from the pending row
                l.history.shrinkRetainingCapacity(l.base);
                var path: std.ArrayList(u32) = .empty;
                defer path.deinit(x.gpa);
                var at: i32 = @intCast(r);
                while (at >= 0) : (at = parents[@intCast(at)]) try path.insert(x.gpa, 0, l.rows.items[@intCast(at)]);
                try l.history.appendSlice(x.gpa, path.items);
                if (w.positions[r] != l.history.items.len) return error.PositionMismatch;
                o.sampled[r] = x.targetNext(l.history.items, w.stream.sampling, w.positions[r]);
                if (w.masks.len > 0) {
                    const words = w.masks.len / l.rows.items.len;
                    o.sampled[r] = try allowed(o.sampled[r], w.masks[r * words ..][0..words]);
                }
                if (o.rows.len > 0) o.rows[r] = rowAt(l.history.items, o.sampled[r], w.stream.logprobs.?);
            }
            l.history.shrinkRetainingCapacity(l.base);
            try l.history.appendSlice(x.gpa, l.rows.items); // a chain keeps every row unless rolled back
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            const l = x.lane(w.stream);
            l.history.shrinkRetainingCapacity(l.base);
            for (path) |r| try l.history.append(x.gpa, l.rows.items[r]);
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        for (requests) |r| {
            const l = x.lane(r.stream);
            // the head reads the last verify's kept rows (a shared round drafts before its rollback)
            var guess: std.ArrayList(u32) = .empty;
            defer guess.deinit(x.gpa);
            if (r.rows) |path| {
                try guess.appendSlice(x.gpa, l.history.items[0..l.base]);
                for (path) |row| try guess.append(x.gpa, l.rows.items[row]);
            } else try guess.appendSlice(x.gpa, l.history.items);
            if (r.position != guess.items.len + 1) return error.PositionMismatch;
            const pending = if (r.first) |feed| x.value(feed) else r.follow[r.follow.len - 1];
            l.held.clearRetainingCapacity();
            try guess.append(x.gpa, pending);
            for (0..r.depth) |j| {
                const at = r.position + j;
                var t = x.targetNext(guess.items, r.stream.sampling, at);
                if ((at *% 2654435761 + j) % 5 == 0) t = (t + 1) % vocab; // a wrong guess now and then
                try l.held.append(x.gpa, t);
                try guess.append(x.gpa, t);
            }
        }
    }

    /// A row's "state" is its token (one u32 a row, whatever the taps): enough for a drafter to rebuild the history.
    fn features(ptr: *anyopaque, s: *Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        _ = taps;
        const l = self(ptr).lane(s);
        if (start + count > l.history.items.len) return error.RowsNotHeld;
        return .{ .buffer = @intFromPtr(l.history.items.ptr), .offset = start * 4, .rows = count, .row_bytes = 4, .space = .host, .dtype = .u32 };
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.lanes.fetchRemove(s)) |kv| {
            var l = kv.value;
            freeLane(x.gpa, &l);
        }
    }
};

/// Round times from a fixed cost model, so tests decide depth the same way every run.
pub const FixedClock = struct {
    calls: u64 = 0,

    pub fn clock(c: *FixedClock) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *FixedClock = @ptrCast(@alignCast(ptr));
        c.calls += 1;
        return 6.0 + @as(f64, @floatFromInt(c.calls % 7)) * 0.5;
    }
};

/// A BetweenHook that counts the host's turns between a fake prompt pass's chunks (both parts, and their asks).
const Turns = struct {
    ready: bool = true, // the round part would run a round (another stream is live)
    wanted: bool = true, // the tier part has work
    rounds: usize = 0,
    services: usize = 0,
    asked_ready: usize = 0,

    fn of(ptr: *anyopaque) *Turns {
        return @ptrCast(@alignCast(ptr));
    }
    fn readyFn(ptr: *anyopaque) bool {
        of(ptr).asked_ready += 1;
        return of(ptr).ready;
    }
    fn round(ptr: *anyopaque) anyerror!void {
        of(ptr).rounds += 1;
    }
    fn wantedFn(ptr: *anyopaque) bool {
        return of(ptr).wanted;
    }
    fn service(ptr: *anyopaque) void {
        of(ptr).services += 1;
    }
    fn hook(t: *Turns, with_tier: bool) @import("stream.zig").BetweenHook {
        return .{ .ptr = t, .run = round, .ready = readyFn, .tier = if (with_tier) .{ .wanted = wantedFn, .run = service } else null };
    }
};

test "the fake offers the host's turn between chunks: the round part with yield_rounds when a round would run, the tier part wherever it is wanted, neither after the last chunk" {
    const gpa = std.testing.allocator;
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const Case = struct { yield: bool, ready: bool, wanted: bool, tier: bool, chunks: usize, rounds: usize, services: usize };
    for ([_]Case{
        .{ .yield = false, .ready = true, .wanted = true, .tier = true, .chunks = 4, .rounds = 0, .services = 3 }, // the prefill yield off: the tier part still runs
        .{ .yield = true, .ready = true, .wanted = true, .tier = true, .chunks = 4, .rounds = 3, .services = 3 }, // both, at each of the 3 boundaries
        .{ .yield = true, .ready = false, .wanted = true, .tier = true, .chunks = 4, .rounds = 0, .services = 3 }, // no other stream: no round, the service still
        .{ .yield = true, .ready = true, .wanted = false, .tier = true, .chunks = 4, .rounds = 3, .services = 0 }, // nothing pending: no service
        .{ .yield = true, .ready = true, .wanted = true, .tier = false, .chunks = 4, .rounds = 3, .services = 0 }, // a host without a tier part
        .{ .yield = true, .ready = true, .wanted = true, .tier = true, .chunks = 1, .rounds = 0, .services = 0 }, // one chunk: no boundary
    }) |c| {
        var x: Fake = .{ .gpa = gpa, .prefill_chunks = c.chunks, .yield_rounds = c.yield };
        defer x.deinit();
        var turns: Turns = .{ .ready = c.ready, .wanted = c.wanted };
        var s = try Stream.init(gpa, .{ .id = "s", .prompt = &prompt, .max_new = 1 });
        defer s.deinit(gpa);
        s.between = turns.hook(c.tier);
        try x.backend().prefill(&s);
        x.backend().release(&s);
        try std.testing.expectEqual(c.rounds, turns.rounds);
        try std.testing.expectEqual(c.services, turns.services);
        if (!c.yield) try std.testing.expectEqual(@as(usize, 0), turns.asked_ready); // the prefill yield off asks nothing of the round part
    }
}

test "a kept state is in place before the fake's first chunk: a host that reports it (and frees it) at the first boundary still resumes it" {
    const gpa = std.testing.allocator;
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    // the host's turn between chunks: at the first boundary it reports the restored state (LaneHost.reported nulls
    // Stream.reuse.saved) and a keep's room evicts it (its copy freed)
    const Report = struct {
        fake: *Fake,
        s: *Stream,
        boundaries: usize = 0,
        fn of(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        fn no(_: *anyopaque) bool {
            return false;
        }
        fn noRound(_: *anyopaque) anyerror!void {}
        fn yes(_: *anyopaque) bool {
            return true;
        }
        fn service(ptr: *anyopaque) void {
            const r = of(ptr);
            r.boundaries += 1;
            if (r.s.reuse.saved) |saved| {
                r.s.reuse.saved = null;
                r.fake.drop(saved);
            }
        }
    };
    var x: Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer x.deinit();
    var s = try Stream.init(gpa, .{ .id = "s", .prompt = &prompt, .max_new = 1 });
    defer s.deinit(gpa);
    const kept = try gpa.create(std.ArrayList(u32)); // the fake's state of the prompt's first 4 tokens: that history
    kept.* = .empty;
    try kept.appendSlice(gpa, prompt[0..4]);
    s.reuse = .{ .saved = kept, .at = 4 };
    var report: Report = .{ .fake = &x, .s = &s };
    s.between = .{ .ptr = &report, .run = Report.noRound, .ready = Report.no, .tier = .{ .wanted = Report.yes, .run = Report.service } };
    try x.backend().prefill(&s);
    defer x.backend().release(&s);
    try std.testing.expectEqual(@as(usize, 2), report.boundaries);
    try std.testing.expect(s.reuse.saved == null); // reported, and gone
    try std.testing.expectEqual(@as(u32, 4), s.cached); // the pass resumed the kept state all the same
    try std.testing.expectEqualSlices(u32, &prompt, x.lane(&s).history.items); // its 4 tokens, then the rest
}

test "with no hook every pass is at most the resume rows" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 37, 40, 3 }) |len| {
        var x: Fake = .{ .gpa = gpa, .pass_rule = .{ .resume_rows = 4, .pass_rows = 8 } };
        defer x.deinit();
        const prompt = try gpa.alloc(u32, len);
        defer gpa.free(prompt);
        @memset(prompt, 5);
        var s = try Stream.init(gpa, .{ .id = "alone", .prompt = prompt, .max_new = 1, .drafts = false });
        defer s.deinit(gpa);
        const b = x.chunkedBackend();
        var steps: usize = 1;
        while (!try b.vtable.prefill_step.?(b.ptr, &s)) : (steps += 1) if (steps > len) return error.PassDidNotAdvance;
        var at: usize = 0;
        for (x.steps.items) |st| {
            try std.testing.expectEqual(at, st.start);
            try std.testing.expect(st.rows <= 4);
            try std.testing.expectEqual(@as(?stream_mod.Facts, null), st.facts);
            at += st.rows;
        }
        try std.testing.expectEqual(len, at);
    }
}
