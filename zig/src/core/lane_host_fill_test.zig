//! Late admission and per-stream prompt state through the served lane host.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const gpa = std.testing.allocator;

const Action = enum { none, cancel_long, cancel_short, refuse_short, full };

const Seen = struct {
    action: Action = .none,
    engine: api.Engine = undefined,
    short: *const api.Request,
    long_chunks: usize = 0,
    short_at: ?usize = null,
    submitted: bool = false,
    tokens: [2]std.ArrayList(u32) = @splat(.empty),
    reasons: [2]?api.Reason = .{ null, null },
    done: std.atomic.Value(u32) = .init(0),

    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const x: *Seen = @ptrCast(@alignCast(ctx));
        if (s.prompt_len != 32) return;
        x.long_chunks += 1;
        if (x.long_chunks == 1) switch (x.action) {
            .cancel_long => x.engine.cancel(1),
            .cancel_short => x.engine.cancel(2),
            else => {},
        };
        if (!x.submitted) {
            x.submitted = true;
            x.engine.submit(2, x.short, .{ .ctx = x, .event = event }) catch unreachable;
        }
    }

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const x: *Seen = @ptrCast(@alignCast(ctx));
        const i: usize = @intCast(id - 1);
        switch (e.*) {
            .tokens => |t| {
                if (id == 2 and x.short_at == null) x.short_at = x.long_chunks;
                x.tokens[i].appendSlice(gpa, t) catch unreachable;
            },
            .finished => |f| {
                x.reasons[i] = f.reason;
                _ = x.done.fetchAdd(1, .release);
            },
            else => {},
        }
    }

    fn wait(x: *Seen) !void {
        for (0..2000) |_| {
            if (x.done.load(.acquire) == 2) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.RequestsDidNotFinish;
    }

    fn deinit(x: *Seen) void {
        for (&x.tokens) |*t| t.deinit(gpa);
    }
};

fn reference(cfg: *const lanes.Config, prompt: []const u32) ![]u32 {
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer core.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "reference", .prompt = prompt, .max_new = 8, .drafts = false });
    defer s.deinit(gpa);
    try core.addStream(&s);
    while (core.activeCount() > 0) try core.step();
    return gpa.dupe(u32, s.emitted());
}

test "a late short request completes before another prompt finishes filling, with the same tokens as solo" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const long: [32]u32 = @splat(3);
    const short = [_]u32{ 1, 2, 3 };
    const short_request: api.Request = .{ .prompt = &short, .max_tokens = 8, .drafts = false };
    var seen: Seen = .{ .short = &short_request };
    defer seen.deinit();
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8, .prefill_hook = Seen.hook, .prefill_hook_ctx = &seen };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    seen.engine = host.engine();
    try host.start();
    defer host.stop();
    const request: api.Request = .{ .prompt = &long, .max_tokens = 8, .drafts = false };
    try seen.engine.submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
    try seen.wait();
    try std.testing.expect(seen.short_at.? < 8);
    try std.testing.expectEqual(api.Reason.length, seen.reasons[0].?);
    try std.testing.expectEqual(api.Reason.length, seen.reasons[1].?);
    const want_long = try reference(&cfg, &long);
    defer gpa.free(want_long);
    const want_short = try reference(&cfg, &short);
    defer gpa.free(want_short);
    try std.testing.expectEqualSlices(u32, want_long, seen.tokens[0].items);
    try std.testing.expectEqualSlices(u32, want_short, seen.tokens[1].items);
}

test "partial admission releases cancelled prompts, isolates refusal, and retries a full pool with another filler" {
    for ([_]Action{ .cancel_long, .cancel_short, .refuse_short, .full }) |action| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        const long: [32]u32 = @splat(3);
        const short = [_]u32{ 1, 2, 3 };
        const short_request: api.Request = .{
            .prompt = &short,
            .max_tokens = 8,
            .drafts = false,
            .sampling = if (action == .refuse_short) .{ .temperature = 1, .seed = 7 } else null,
        };
        var seen: Seen = .{ .short = &short_request, .action = action, .submitted = true };
        defer seen.deinit();
        var target: lanes.fake.Fake = .{
            .gpa = gpa,
            .prefill_hook = Seen.hook,
            .prefill_hook_ctx = &seen,
            .refuse_sampled = action == .refuse_short,
            .refuse_full = if (action == .full) 1 else 0,
        };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
        defer core.deinit();
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        seen.engine = host.engine();
        const request: api.Request = .{ .prompt = &long, .max_tokens = 8, .drafts = false };
        try seen.engine.submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
        try seen.engine.submit(2, &short_request, .{ .ctx = &seen, .event = Seen.event });
        try host.start();
        defer host.stop();
        try seen.wait();
        const want_long: api.Reason = if (action == .cancel_long) .cancelled else .length;
        const want_short: api.Reason = switch (action) {
            .cancel_short => .cancelled,
            .refuse_short => .failed,
            else => .length,
        };
        try std.testing.expectEqual(want_long, seen.reasons[0].?);
        try std.testing.expectEqual(want_short, seen.reasons[1].?);
        if (want_long == .length) {
            const want = try reference(&cfg, &long);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, seen.tokens[0].items);
        } else try std.testing.expectEqual(@as(usize, 0), seen.tokens[0].items.len);
        if (want_short == .length) {
            const want = try reference(&cfg, &short);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, seen.tokens[1].items);
        } else try std.testing.expectEqual(@as(usize, 0), seen.tokens[1].items.len);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}

// -- the scheduling hook's facts and plan (lanes.stream.Facts, Next) at the default cadence ------------------------

const Step = lanes.fake.Fake.StepSeen;

/// A run of the served host over the chunked fake with its pass rule (resume 4 rows, pass 8): the truth beside the
/// fake's log of every step (whether a stream decoded as it began, read in the prefill hook), and the step at which
/// each request was submitted (0: before the host started; `later` ones from the hook, when the stream of `on_len`
/// begins its `at`-th step). Every request has a prompt length of its own (`prompt_lens`).
const Cadence = struct {
    engine: api.Engine = undefined,
    core: *lanes.Engine = undefined,
    later: []const Later = &.{},
    fired: [4]bool = @splat(false),
    prompt_lens: [4]usize = @splat(0),
    submitted_at: [4]?usize = @splat(null),
    decoding: std.ArrayList(bool) = .empty,
    lens: std.ArrayList(usize) = .empty,
    reasons: [4]?api.Reason = @splat(null),
    tokens: [4]std.ArrayList(u32) = @splat(.empty), // the tokens each request's sink heard
    done: std.atomic.Value(u32) = .init(0),
    want: u32 = 0,
    /// The fake's pass rule keeps whole passes while only other prompts fill (TF_DS_PASS_FILLING's rule).
    filling_pass: bool = false,
    /// The host's fill order (TF_DS_PROMPT_ORDER and TF_DS_PROMPT_QOS); off: the rotation.
    fill: api.fill_order.Policy = .{},
    /// Milliseconds each prompt step takes (the hook sleeps), so the order's seconds rule has time to pass.
    step_ms: u32 = 0,

    const Later = struct { id: api.Id, request: *const api.Request, on_len: usize, at: usize };

    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const x: *Cadence = @ptrCast(@alignCast(ctx));
        const i = x.lens.items.len;
        if (x.step_ms > 0) std.Io.sleep(std.testing.io, .fromMilliseconds(x.step_ms), .awake) catch {};
        x.decoding.append(gpa, x.core.activeCount() > 0) catch unreachable;
        var before: usize = 0;
        for (x.lens.items) |l| before += @intFromBool(l == s.prompt_len);
        for (x.later, 0..) |l, k| {
            if (x.fired[k] or s.prompt_len != l.on_len or before != l.at) continue;
            x.fired[k] = true;
            x.submitted_at[@intCast(l.id - 1)] = i;
            x.engine.submit(l.id, l.request, .{ .ctx = x, .event = event }) catch unreachable;
        }
        x.lens.append(gpa, s.prompt_len) catch unreachable;
    }

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const x: *Cadence = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .tokens => |t| x.tokens[@intCast(id - 1)].appendSlice(gpa, t) catch unreachable,
            .finished => |f| {
                x.reasons[@intCast(id - 1)] = f.reason;
                _ = x.done.fetchAdd(1, .release);
            },
            else => {},
        }
    }

    fn wait(x: *Cadence) !void {
        for (0..5000) |_| {
            if (x.done.load(.acquire) == x.want) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.RequestsDidNotFinish;
    }

    fn deinit(x: *Cadence) void {
        x.decoding.deinit(gpa);
        x.lens.deinit(gpa);
        for (&x.tokens) |*t| t.deinit(gpa);
    }

    /// The `first` requests submitted before the host of `n_lanes` lanes starts, the `later` ones from the hook; the fake
    /// keeps its log for the caller.
    fn run(x: *Cadence, target: *lanes.fake.Fake, first: []const *const api.Request, n_lanes: u32) !void {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        target.pass_rule = .{ .resume_rows = 4, .pass_rows = 8, .filling_pass = x.filling_pass };
        target.prefill_hook = hook;
        target.prefill_hook_ctx = x;
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
        defer core.deinit();
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = n_lanes, .resume_rows = 4, .fill = x.fill });
        x.engine = host.engine();
        x.core = &core;
        x.want = @intCast(first.len + x.later.len);
        for (first, 1..) |r, id| {
            x.submitted_at[id - 1] = 0;
            try x.engine.submit(id, r, .{ .ctx = x, .event = event });
        }
        try host.start();
        defer host.stop();
        try x.wait();
    }

    /// The facts every step must have been given, from the log: a stream decoding as it began (the hook's reading);
    /// another request submitted by then and not yet at its first step (waiting); another one past its first step and
    /// short of its last (filling); the resume rows as step rows while a stream decodes, else no bound.
    fn expectFacts(x: *const Cadence, steps: []const Step) !void {
        try std.testing.expectEqual(steps.len, x.decoding.items.len);
        for (steps, 0..) |st, i| {
            var waiting = false;
            var filling = false;
            for (x.submitted_at, x.prompt_lens) |at, len| {
                const sub = at orelse continue;
                if (len == st.prompt_len) continue;
                const span = spanOf(steps, len);
                if (sub <= i and (span == null or span.?[0] > i)) {
                    // the fill order: an admitted prompt waiting its turn fills beside this one, it does not wait for a lane
                    if (x.fill.on) filling = true else waiting = true;
                }
                if (span) |sp| if (sp[0] < i and sp[1] > i) {
                    filling = true;
                };
            }
            const f = st.facts.?;
            try std.testing.expectEqual(x.decoding.items[i], f.decoding);
            try std.testing.expectEqual(waiting, f.waiting);
            try std.testing.expectEqual(filling, f.filling);
            try std.testing.expectEqual(@as(u32, if (f.decoding) 4 else std.math.maxInt(u32)), f.step_rows);
        }
    }
};

/// The first and last step indices of the stream of prompt length `len`.
fn spanOf(steps: []const Step, len: usize) ?[2]usize {
    var lo: ?usize = null;
    var hi: usize = 0;
    for (steps, 0..) |st, i| if (st.prompt_len == len) {
        if (lo == null) lo = i;
        hi = i;
    };
    return if (lo) |l| .{ l, hi } else null;
}

/// What followed step i in the log: a round (the fake served decode calls before the next step), this stream's next
/// step, or another stream's.
fn followed(steps: []const Step, i: usize) lanes.stream.Next {
    const j = i + 1;
    if (steps[j].decodes > steps[i].decodes) return .round;
    return if (steps[j].prompt_len == steps[i].prompt_len) .same else .other;
}

test "the hook answers what the host then does: each step's facts as it begins, its next at its end" {
    // a 32-token prompt alone, then a 16-token one arriving at its second step (they alternate), the shorter decoding
    // beside the rest of the longer, and a 4-token one arriving at its sixth step
    const p1: [32]u32 = @splat(3);
    const p2: [16]u32 = @splat(4);
    const p3: [4]u32 = @splat(5);
    const r1: api.Request = .{ .prompt = &p1, .max_tokens = 8, .drafts = false };
    const r2: api.Request = .{ .prompt = &p2, .max_tokens = 24, .drafts = false };
    const r3: api.Request = .{ .prompt = &p3, .max_tokens = 4, .drafts = false };
    var run: Cadence = .{ .later = &.{ .{ .id = 2, .request = &r2, .on_len = 32, .at = 1 }, .{ .id = 3, .request = &r3, .on_len = 32, .at = 5 } } };
    defer run.deinit();
    run.prompt_lens = .{ 32, 16, 4, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{&r1}, 3);
    for (run.reasons[0..3]) |r| try std.testing.expectEqual(api.Reason.length, r.?);
    const steps = target.steps.items;
    try run.expectFacts(steps);
    var seen = std.EnumSet(lanes.stream.Next){};
    for (steps, 0..) |st, i| {
        const n = st.next orelse continue; // (a prompt's last step is not asked)
        try std.testing.expectEqual(followed(steps, i), n);
        seen.insert(n);
    }
    // every answer of the default cadence came up: the prompt alone, rotation, a round while a stream decodes
    try std.testing.expect(seen.contains(.same) and seen.contains(.other) and seen.contains(.round));
}

test "a lone prompt with nothing decoding or waiting steps by the pass rows" {
    const p: [37]u32 = @splat(3);
    const r: api.Request = .{ .prompt = &p, .max_tokens = 4, .drafts = false };
    var run: Cadence = .{};
    defer run.deinit();
    run.prompt_lens = .{ 37, 0, 0, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{&r}, 2);
    try run.expectFacts(target.steps.items);
    // whole pass rows, then a whole resume unit, then the remainder under the resume rows alone
    const want = [_]usize{ 8, 8, 8, 8, 4, 1 };
    try std.testing.expectEqual(want.len, target.steps.items.len);
    for (target.steps.items, want) |st, w| try std.testing.expectEqual(w, st.rows);
}

test "one decoding stream beside a long prompt: every pass of the resume rows, and a round between every two" {
    // a 4-token prompt decodes 40 tokens; a 64-token prompt arrives at its step and fills beside it
    const pa: [4]u32 = @splat(5);
    const pb: [64]u32 = @splat(3);
    const ra: api.Request = .{ .prompt = &pa, .max_tokens = 40, .drafts = false };
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 4, .drafts = false };
    var run: Cadence = .{ .later = &.{.{ .id = 2, .request = &rb, .on_len = 4, .at = 0 }} };
    defer run.deinit();
    run.prompt_lens = .{ 4, 64, 0, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{&ra}, 2);
    const steps = target.steps.items;
    try run.expectFacts(steps);
    var long: usize = 0;
    var last: ?usize = null;
    for (steps, 0..) |st, i| {
        if (st.prompt_len != 64) continue;
        long += 1;
        try std.testing.expect(st.facts.?.decoding);
        try std.testing.expect(st.rows <= 4);
        if (last) |l| try std.testing.expect(steps[i].decodes > steps[l].decodes);
        last = i;
    }
    try std.testing.expectEqual(@as(usize, 16), long);
}

test "a second prompt filling beside a lone one: both step by the resume rows" {
    const pb: [32]u32 = @splat(3);
    const pc: [33]u32 = @splat(4);
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 4, .drafts = false };
    const rc: api.Request = .{ .prompt = &pc, .max_tokens = 4, .drafts = false };
    var run: Cadence = .{};
    defer run.deinit();
    run.prompt_lens = .{ 32, 33, 0, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{ &rb, &rc }, 2);
    const steps = target.steps.items;
    try run.expectFacts(steps);
    var beside: usize = 0;
    for (steps) |st| if (st.facts.?.filling or st.facts.?.waiting) {
        beside += 1;
        try std.testing.expect(st.rows <= 4);
    };
    try std.testing.expect(beside >= 15);
}

test "TF_DS_PASS_FILLING: two prompts filling together alternate whole passes, never while one waits or a stream decodes" {
    // both admitted before the host starts: the first prompt's first step waits for the second to start, then each
    // turn is a whole pass while the other only fills
    const pb: [32]u32 = @splat(3);
    const pc: [33]u32 = @splat(4);
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 4, .drafts = false };
    const rc: api.Request = .{ .prompt = &pc, .max_tokens = 4, .drafts = false };
    var run: Cadence = .{ .filling_pass = true };
    defer run.deinit();
    run.prompt_lens = .{ 32, 33, 0, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{ &rb, &rc }, 2);
    const steps = target.steps.items;
    try run.expectFacts(steps);
    // the first step waits for the other prompt's first step: the resume rows, as in the baseline
    try std.testing.expect(steps[0].facts.?.waiting);
    try std.testing.expectEqual(@as(usize, 4), steps[0].rows);
    var whole: usize = 0;
    var beside: usize = 0;
    for (steps) |st| {
        const f = st.facts.?;
        if (f.waiting or f.decoding) {
            // a request waiting for admission, or a stream decoding: the resume rows, whatever fills beside
            try std.testing.expect(st.rows <= 4);
        } else {
            if (f.filling) beside += 1;
            if (st.rows == 8) whole += 1;
            // the pass rule's remainder under the resume rows alone, as in the baseline
            try std.testing.expect(st.rows == 8 or st.rows <= 4);
        }
    }
    try std.testing.expect(whole >= 6);
    try std.testing.expect(beside >= 5);
    // the two prompts take turns at their whole passes: each of the first eight steps is the other prompt's
    try std.testing.expect(steps.len >= 8);
    for (1..8) |i| try std.testing.expect(steps[i].prompt_len != steps[i - 1].prompt_len);
}

test "TF_DS_PASS_FILLING: a third request waiting cuts both filling prompts back to the resume rows" {
    // the third request arrives at the first prompt's second step; until it has started, every other step waits
    const pb: [32]u32 = @splat(3);
    const pc: [33]u32 = @splat(4);
    const pd: [16]u32 = @splat(5);
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 4, .drafts = false };
    const rc: api.Request = .{ .prompt = &pc, .max_tokens = 4, .drafts = false };
    const rd: api.Request = .{ .prompt = &pd, .max_tokens = 4, .drafts = false };
    var run: Cadence = .{ .filling_pass = true, .later = &.{.{ .id = 3, .request = &rd, .on_len = 32, .at = 1 }} };
    defer run.deinit();
    run.prompt_lens = .{ 32, 33, 16, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{ &rb, &rc }, 3);
    const steps = target.steps.items;
    try run.expectFacts(steps);
    var waited: usize = 0;
    var whole: usize = 0;
    for (steps) |st| {
        const f = st.facts.?;
        if (f.waiting or f.decoding) {
            waited += 1;
            try std.testing.expect(st.rows <= 4);
        } else if (st.rows == 8) whole += 1;
    }
    try std.testing.expect(waited >= 1); // the arrival, before its first step
    try std.testing.expect(whole >= 6); // and the two long prompts' whole passes after it
}

test "TF_DS_PASS_FILLING: a decoding stream beside a long prompt keeps every pass at the resume rows" {
    // the same shape as the test above, with the switch's rule: its rounds keep the passes at the resume rows
    const pa: [4]u32 = @splat(5);
    const pb: [64]u32 = @splat(3);
    const ra: api.Request = .{ .prompt = &pa, .max_tokens = 40, .drafts = false };
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 4, .drafts = false };
    var run: Cadence = .{ .filling_pass = true, .later = &.{.{ .id = 2, .request = &rb, .on_len = 4, .at = 0 }} };
    defer run.deinit();
    run.prompt_lens = .{ 4, 64, 0, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{&ra}, 2);
    const steps = target.steps.items;
    try run.expectFacts(steps);
    var long: usize = 0;
    for (steps) |st| {
        if (st.prompt_len != 64) continue;
        long += 1;
        try std.testing.expect(st.facts.?.decoding);
        try std.testing.expect(st.rows <= 4);
    }
    try std.testing.expectEqual(@as(usize, 16), long);
}

fn firstStep(steps: []const Step, len: usize) ?usize {
    for (steps, 0..) |st, i| if (st.prompt_len == len) return i;
    return null;
}

fn lastStep(steps: []const Step, len: usize) ?usize {
    var r: ?usize = null;
    for (steps, 0..) |st, i| if (st.prompt_len == len) {
        r = i;
    };
    return r;
}

test "the serial order: two prompts admitted together fill one after the other, the smaller first" {
    // 33 rows against 32: the 32-row prompt has fewer rows left, so its four passes come first
    const pb: [33]u32 = @splat(3);
    const pc: [32]u32 = @splat(4);
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 4, .drafts = false };
    const rc: api.Request = .{ .prompt = &pc, .max_tokens = 4, .drafts = false };
    var run: Cadence = .{ .filling_pass = true, .fill = api.fill_order.parse("serial", null, null).policy };
    defer run.deinit();
    run.prompt_lens = .{ 33, 32, 0, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{ &rb, &rc }, 2);
    const steps = target.steps.items;
    try run.expectFacts(steps);
    const last32 = lastStep(steps, 32).?;
    try std.testing.expectEqual(@as(usize, 0), firstStep(steps, 32).?);
    try std.testing.expect(firstStep(steps, 33).? > last32);
    for (steps[0 .. last32 + 1]) |st| try std.testing.expectEqual(@as(usize, 32), st.prompt_len);
}

test "the serial order: a short prompt arriving during a long one's pass takes the next pass" {
    const pl: [64]u32 = @splat(3);
    const ps: [4]u32 = @splat(4);
    const rl: api.Request = .{ .prompt = &pl, .max_tokens = 4, .drafts = false };
    const rs: api.Request = .{ .prompt = &ps, .max_tokens = 4, .drafts = false };
    var run: Cadence = .{ .filling_pass = true, .fill = api.fill_order.parse("serial", null, null).policy, .later = &.{.{ .id = 2, .request = &rs, .on_len = 64, .at = 2 }} };
    defer run.deinit();
    run.prompt_lens = .{ 64, 4, 0, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{&rl}, 2);
    const steps = target.steps.items;
    const sub = run.submitted_at[1].?;
    // the long prompt had passed two steps when the short one came: its first step is the very next one served
    try std.testing.expectEqual(sub + 1, firstStep(steps, 4).?);
    try std.testing.expect(firstStep(steps, 4).? < lastStep(steps, 64).?);
}

test "a table bounds the overtaking: a larger prompt due takes a pass between smaller ones" {
    // class 1: up to 8 rows, never due; class 2: the rest, overtaken at most once. The large prompt (64 rows) fills beside
    // three small prompts: it is served after at most one small prompt's step each time it is passed over.
    const pl: [64]u32 = @splat(3);
    const p1: [8]u32 = @splat(4);
    const p2: [7]u32 = @splat(5);
    const p3: [6]u32 = @splat(6);
    const rl: api.Request = .{ .prompt = &pl, .max_tokens = 2, .drafts = false };
    const r1: api.Request = .{ .prompt = &p1, .max_tokens = 2, .drafts = false };
    const r2: api.Request = .{ .prompt = &p2, .max_tokens = 2, .drafts = false };
    const r3: api.Request = .{ .prompt = &p3, .max_tokens = 2, .drafts = false };
    var run: Cadence = .{ .filling_pass = true, .fill = api.fill_order.parse("serial", "8:1:0,*:2:1", null).policy };
    defer run.deinit();
    run.prompt_lens = .{ 64, 8, 7, 6 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{ &rl, &r1, &r2, &r3 }, 4);
    const steps = target.steps.items;
    // never two steps in a row of the small prompts before the large one's first (its bound is one overtake)
    const first_large = firstStep(steps, 64).?;
    try std.testing.expect(first_large <= 2);
    // and the large prompt is not left to the end: a small prompt's step still follows its first
    try std.testing.expect(lastStep(steps, 64).? > first_large);
}

test "a background prompt takes a step only while no foreground prompt fills" {
    const pb: [32]u32 = @splat(3);
    const pf: [64]u32 = @splat(4);
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 2, .drafts = false, .background = true };
    const rf: api.Request = .{ .prompt = &pf, .max_tokens = 2, .drafts = false };
    var run: Cadence = .{ .filling_pass = true, .fill = api.fill_order.parse("serial", null, null).policy };
    defer run.deinit();
    run.prompt_lens = .{ 32, 64, 0, 0 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    // the background prompt is the smaller and the first submitted
    try run.run(&target, &.{ &rb, &rf }, 2);
    const steps = target.steps.items;
    try run.expectFacts(steps);
    try std.testing.expect(firstStep(steps, 32).? > lastStep(steps, 64).?);
    try std.testing.expectEqual(@as(usize, 64), steps[0].prompt_len);
}

test "the prompt order leaves every reply's tokens as the rotation's" {
    const pb: [32]u32 = @splat(3);
    const pc: [33]u32 = @splat(4);
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 4, .drafts = false };
    const rc: api.Request = .{ .prompt = &pc, .max_tokens = 4, .drafts = false };
    var rot: Cadence = .{ .filling_pass = true };
    defer rot.deinit();
    rot.prompt_lens = .{ 32, 33, 0, 0 };
    var rot_target: lanes.fake.Fake = .{ .gpa = gpa };
    defer rot_target.deinit();
    try rot.run(&rot_target, &.{ &rb, &rc }, 2);
    var ord: Cadence = .{ .filling_pass = true, .fill = api.fill_order.parse("serial", "8:1:0,*:2:2", null).policy };
    defer ord.deinit();
    ord.prompt_lens = .{ 32, 33, 0, 0 };
    var ord_target: lanes.fake.Fake = .{ .gpa = gpa };
    defer ord_target.deinit();
    try ord.run(&ord_target, &.{ &rb, &rc }, 2);
    try std.testing.expectEqual(api.Reason.length, rot.reasons[0].?);
    try std.testing.expectEqual(api.Reason.length, ord.reasons[0].?);
    try std.testing.expectEqualSlices(u32, rot.tokens[0].items, ord.tokens[0].items);
    try std.testing.expectEqualSlices(u32, rot.tokens[1].items, ord.tokens[1].items);
    // the control: one prompt token changed, the same order, and the reply differs
    var other = pc;
    other[32] = 9;
    const rc2: api.Request = .{ .prompt = &other, .max_tokens = 4, .drafts = false };
    var ctl: Cadence = .{ .filling_pass = true, .fill = ord.fill };
    defer ctl.deinit();
    ctl.prompt_lens = .{ 32, 33, 0, 0 };
    var ctl_target: lanes.fake.Fake = .{ .gpa = gpa };
    defer ctl_target.deinit();
    try ctl.run(&ctl_target, &.{ &rb, &rc2 }, 2);
    try std.testing.expect(!std.mem.eql(u32, ord.tokens[1].items, ctl.tokens[1].items));
}

test "the seconds rule: a larger prompt passed over by a stream of small ones takes a step once it waited its maximum" {
    // class 1 up to 8 rows, class 2 the rest with no step bound; every step takes 600 ms and the maximum wait is 1 s. The
    // large prompt (16 rows, two passes) would otherwise wait for all three small prompts.
    const pl: [16]u32 = @splat(3);
    const p1: [4]u32 = @splat(4);
    const p2: [3]u32 = @splat(5);
    const p3: [2]u32 = @splat(6);
    const rl: api.Request = .{ .prompt = &pl, .max_tokens = 2, .drafts = false };
    const r1: api.Request = .{ .prompt = &p1, .max_tokens = 2, .drafts = false };
    const r2: api.Request = .{ .prompt = &p2, .max_tokens = 2, .drafts = false };
    const r3: api.Request = .{ .prompt = &p3, .max_tokens = 2, .drafts = false };
    var run: Cadence = .{ .filling_pass = true, .step_ms = 600, .fill = api.fill_order.parse("serial", "8:1:0,*:2:inf", "1").policy };
    defer run.deinit();
    run.prompt_lens = .{ 16, 4, 3, 2 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{ &rl, &r1, &r2, &r3 }, 4);
    const steps = target.steps.items;
    // two small steps (1.2 s), then the large prompt is past its maximum and goes before the third small prompt
    try std.testing.expectEqual(@as(usize, 2), steps[0].prompt_len);
    try std.testing.expectEqual(@as(usize, 3), steps[1].prompt_len);
    try std.testing.expectEqual(@as(usize, 16), steps[2].prompt_len);
}

test "the seconds rule: a background prompt takes a step once it waited its maximum, ahead of foreground prompts" {
    const pd: [1]u32 = @splat(3);
    const pa: [4]u32 = @splat(4);
    const pb: [3]u32 = @splat(5);
    const pc: [2]u32 = @splat(6);
    const rd: api.Request = .{ .prompt = &pd, .max_tokens = 2, .drafts = false, .background = true };
    const ra: api.Request = .{ .prompt = &pa, .max_tokens = 2, .drafts = false };
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 2, .drafts = false };
    const rc: api.Request = .{ .prompt = &pc, .max_tokens = 2, .drafts = false };
    // the foreground prompt of 4 rows arrives during the 3-row prompt's step, so it has waited less than the maximum
    // when the background prompt (waiting since the start) has waited more
    var run: Cadence = .{ .later = &.{.{ .id = 4, .request = &ra, .on_len = 3, .at = 0 }}, .filling_pass = true, .step_ms = 600, .fill = api.fill_order.parse("serial", null, "1").policy };
    defer run.deinit();
    run.prompt_lens = .{ 1, 2, 3, 4 };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try run.run(&target, &.{ &rd, &rc, &rb }, 4);
    const steps = target.steps.items;
    try std.testing.expectEqual(@as(usize, 2), steps[0].prompt_len);
    try std.testing.expectEqual(@as(usize, 3), steps[1].prompt_len);
    try std.testing.expectEqual(@as(usize, 1), steps[2].prompt_len); // the background prompt, due
    try std.testing.expectEqual(@as(usize, 4), steps[3].prompt_len);
}

const Warm = struct {
    indexed: [16]usize = @splat(0),
    chunks: usize = 0,
    done: std.atomic.Value(u32) = .init(0),

    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const w: *Warm = @ptrCast(@alignCast(ctx));
        const p: *lanes.SuffixLookup = @ptrCast(@alignCast(s.proposer.?.ptr));
        if (w.chunks < w.indexed.len) w.indexed[w.chunks] = p.indexed;
        w.chunks += 1;
    }

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const w: *Warm = @ptrCast(@alignCast(ctx));
        if (e.* == .finished) _ = w.done.fetchAdd(1, .release);
    }
};

test "a prompt's copy index is built a step at a time while the prompt fills, not by the first round after it" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1 + i % 5);
    var warm: Warm = .{};
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8, .prefill_hook = Warm.hook, .prefill_hook_ctx = &warm };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try host.engine().submit(1, &request, .{ .ctx = &warm, .event = Warm.event });
    for (0..2000) |_| {
        if (warm.done.load(.acquire) == 1) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    try std.testing.expectEqual(@as(u32, 1), warm.done.load(.acquire));
    try std.testing.expect(warm.chunks >= 3);
    // nothing indexed in the first chunk (the first step came before any warm); after each step the prompt's tokens are in
    try std.testing.expectEqual(@as(usize, 0), warm.indexed[0]);
    try std.testing.expectEqual(prompt.len, warm.indexed[warm.chunks - 1]);
}

const Overlapped = struct {
    tokens: std.atomic.Value(u32) = .init(0),
    before: u32 = 0,
    gained: u32 = 0, // tokens the host delivered inside the verify's forward wait, over every round
    rounds: u32 = 0,
    done: std.atomic.Value(u32) = .init(0),
    all: std.ArrayList(u32) = .empty,
    mutex: std.Io.Mutex = .init,

    fn probe(ctx: *anyopaque, after_work: bool) void {
        const o: *Overlapped = @ptrCast(@alignCast(ctx));
        const n = o.tokens.load(.acquire);
        if (!after_work) {
            o.before = n;
            o.rounds += 1;
        } else o.gained += n - o.before;
    }

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const o: *Overlapped = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .tokens => |t| {
                _ = o.tokens.fetchAdd(@intCast(t.len), .release);
                o.mutex.lockUncancelable(std.testing.io);
                defer o.mutex.unlock(std.testing.io);
                o.all.appendSlice(gpa, t) catch unreachable;
            },
            .finished => _ = o.done.fetchAdd(1, .release),
            else => {},
        }
    }
};

test "a decode round's tokens are delivered inside the next round's forward wait; a stream's last round at once, and every token arrives in order" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = false, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const prompt = [_]u32{ 1, 2, 3 };
    var o: Overlapped = .{};
    defer o.all.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .verify_probe = Overlapped.probe, .verify_probe_ctx = &o };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 12, .drafts = false };
    try host.engine().submit(1, &request, .{ .ctx = &o, .event = Overlapped.event });
    for (0..2000) |_| {
        if (o.done.load(.acquire) == 1) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    try std.testing.expectEqual(@as(u32, 1), o.done.load(.acquire));
    try std.testing.expect(o.rounds >= 3);
    try std.testing.expect(o.gained > 0); // the previous rounds' tokens came in the wait, not before it
    const want = try reference(&cfg, &prompt);
    defer gpa.free(want);
    try std.testing.expectEqualSlices(u32, want[0..@min(want.len, 12)], o.all.items[0..@min(want.len, 12)]);
    try std.testing.expectEqual(@as(usize, 12), o.all.items.len);
}

test "a prompt filling beside a decoding stream: the stream's tokens already landed are with its request when the prompt's next step runs" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = false, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const first = [_]u32{ 1, 2, 3 };
    var long: [32]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = @intCast(3 + i % 4);
    var o: Overlapped = .{};
    defer o.all.deinit(gpa);
    const Late = struct {
        o: *Overlapped,
        engine: api.Engine = undefined,
        short: *const api.Request,
        submitted: bool = false,
        at_step: ?u32 = null,
        steps: u32 = 0,
        fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const x: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len != 32) return;
            if (x.steps == 1) x.at_step = x.o.tokens.load(.acquire); // the second step of the long prompt: what the first stream's request has by now
            x.steps += 1;
        }
    };
    var late: Late = .{ .o = &o, .short = &.{ .prompt = &long, .max_tokens = 2, .drafts = false } };
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8, .prefill_hook = Late.hook, .prefill_hook_ctx = &late };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    late.engine = host.engine();
    try host.start();
    defer host.stop();
    const a: api.Request = .{ .prompt = &first, .max_tokens = 100000, .drafts = false };
    try late.engine.submit(1, &a, .{ .ctx = &o, .event = Overlapped.event });
    while (o.tokens.load(.acquire) < 6) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    const before = o.tokens.load(.acquire);
    try late.engine.submit(2, late.short, .{ .ctx = &o, .event = Overlapped.event });
    for (0..2000) |_| {
        if (late.at_step != null) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    late.engine.cancel(1);
    try std.testing.expect(late.at_step != null);
    try std.testing.expect(late.at_step.? >= before);
}

test "the decode rate takes one mark a round, however many streams landed tokens in it, and the marks add up to the tokens delivered" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = false, .hidden_rows = true, .streams_exact = true, .max_streams = 4 }, 8, 7);
    defer cfg.deinit(gpa);
    const p1 = [_]u32{ 1, 2, 3 };
    const p2 = [_]u32{ 4, 5, 6 };
    var o: Overlapped = .{};
    defer o.all.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const r1: api.Request = .{ .prompt = &p1, .max_tokens = 40, .drafts = false };
    const r2: api.Request = .{ .prompt = &p2, .max_tokens = 40, .drafts = false };
    try host.engine().submit(1, &r1, .{ .ctx = &o, .event = Overlapped.event });
    try host.engine().submit(2, &r2, .{ .ctx = &o, .event = Overlapped.event });
    for (0..4000) |_| {
        if (o.done.load(.acquire) == 2) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    try std.testing.expectEqual(@as(u32, 2), o.done.load(.acquire));
    // a round's tokens reach the requests before the host stores that round's mark: wait for the marks to catch up
    var total: usize = 0;
    for (0..2000) |_| {
        host.mutex.lockUncancelable(std.testing.io);
        total = 0;
        for (host.decoded.items) |m| total += m.tokens;
        host.mutex.unlock(std.testing.io);
        if (total >= 80) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    host.mutex.lockUncancelable(std.testing.io);
    defer host.mutex.unlock(std.testing.io);
    try std.testing.expectEqual(@as(usize, 80), total);
    try std.testing.expect(host.decoded.items.len <= target.rounds + 2);
}

test "the engine thread builds the status counters when status asks, not every turn: a request's turns build none, and each status gets a fresh answer" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = false, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const prompt = [_]u32{ 1, 2, 3 };
    var o: Overlapped = .{};
    defer o.all.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 30, .drafts = false };
    try host.engine().submit(1, &request, .{ .ctx = &o, .event = Overlapped.event });
    for (0..4000) |_| {
        if (o.done.load(.acquire) == 1) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    try std.testing.expectEqual(@as(u32, 1), o.done.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), host.tele_builds); // (30 rounds and the turns around them, none asked)
    var out: api.Status = undefined;
    var toks: [4]u32 = undefined;
    for (0..5) |_| host.engine().status(&out, &toks);
    try std.testing.expectEqual(@as(u64, 5), host.tele_builds);
    try std.testing.expectEqual(@as(u32, 0), out.running); // (the answer is the engine's now: the request ended)
}
