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
    done: std.atomic.Value(u32) = .init(0),
    want: u32 = 0,

    const Later = struct { id: api.Id, request: *const api.Request, on_len: usize, at: usize };

    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const x: *Cadence = @ptrCast(@alignCast(ctx));
        const i = x.lens.items.len;
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
    }

    /// The `first` requests submitted before the host of `n_lanes` lanes starts, the `later` ones from the hook; the fake
    /// keeps its log for the caller.
    fn run(x: *Cadence, target: *lanes.fake.Fake, first: []const *const api.Request, n_lanes: u32) !void {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        target.prefill_hook = hook;
        target.prefill_hook_ctx = x;
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
        defer core.deinit();
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
    }
    try std.testing.expectEqualSlices(Id, &.{ 2, 1 }, &order.prefilled);
}

test "status reads a filling prompt's tokens while its pass runs, and none once it ends" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const Seen = struct {
        engine: Engine = undefined,
        lefts: std.ArrayList(u32) = .empty,
        done: std.atomic.Value(bool) = .init(false),

        fn hook(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            var status: api.Status = .{};
            seen.engine.status(&status, &.{}); // the pass runs without the host's lock: this can read
            if (status.fill_tokens_left) |left| seen.lefts.append(gpa, left) catch {};
        }
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            if (e.* == .finished) seen.done.store(true, .release);
        }
    };
    var seen: Seen = .{};
    seen.engine = host.engine();
    target.prefill_hook = Seen.hook;
    target.prefill_hook_ctx = &seen;
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try seen.engine.submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
    while (!seen.done.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    defer seen.lefts.deinit(gpa);
    try std.testing.expect(seen.lefts.items.len >= 2); // each chunk of the pass sampled the status
    for (seen.lefts.items) |left| try std.testing.expect(left > 0 and left <= prompt.len);
    try std.testing.expect(seen.lefts.items[0] > seen.lefts.items[seen.lefts.items.len - 1]); // it counted down
    var status: api.Status = .{};
    seen.engine.status(&status, &.{});
    try std.testing.expect(status.fill_prompt_tokens == null and status.fill_tokens_left == null); // the pass is over
}

const Passes = struct {
    mutex: std.Io.Mutex = .init,
    prefilled: [2]Id = .{ 0, 0 },
    count: usize = 0,
    done: usize = 0,
    fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
        const o: *@This() = @ptrCast(@alignCast(ctx));
        o.mutex.lockUncancelable(std.testing.io);
        defer o.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => {
                o.prefilled[o.count] = id;
                o.count += 1;
            },
            .finished => o.done += 1,
            else => {},
        }
    }
    fn wait(o: *@This(), n: usize) void {
        while (true) {
            o.mutex.lockUncancelable(std.testing.io);
            const d = o.done;
            o.mutex.unlock(std.testing.io);
            if (d == n) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

test "a backend that holds one prompt pass at a time runs the pass it began to its end before the next comes in" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.steppedOne(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    var order: Passes = .{};
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const short = [_]u32{ 2, 7, 1, 8 };
    const background: Request = .{ .prompt = &long, .max_tokens = 4, .background = true };
    const foreground: Request = .{ .prompt = &short, .max_tokens = 4 };
    const e = host.engine();
    // the long background prompt's first chunk sends a short foreground request, which a stepping backend of
    // several passes fills first (above): here it comes in once the long pass ends, and no two passes fill at once
    const Arrive = struct {
        engine: api.Engine,
        request: *const Request,
        order: *Passes,
        target: *lanes.fake.Fake,
        sent: bool = false,
        most: usize = 0,
        fn call(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            var n: usize = 0;
            var it = a.target.lanes.valueIterator();
            while (it.next()) |l| n += @intFromBool(l.filling);
            a.most = @max(a.most, n);
            if (a.sent) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.order, .event = Passes.event }) catch {};
        }
    };
    var arrive: Arrive = .{ .engine = e, .request = &foreground, .order = &order, .target = &target };
    target.prefill_hook = Arrive.call;
    target.prefill_hook_ctx = &arrive;
    try e.submit(1, &background, .{ .ctx = &order, .event = Passes.event });
    try host.start();
    defer host.stop();
    order.wait(2);
    try std.testing.expectEqualSlices(Id, &.{ 1, 2 }, &order.prefilled);
    try std.testing.expectEqual(@as(usize, 1), arrive.most);
}

test "a pass that began waits out fill_rounds of the live streams' rounds between two of its steps" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.steppedOne(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .fill_rounds = 3 });
    var order: Passes = .{};
    const first_prompt = [_]u32{ 2, 7, 1, 8 };
    const second_prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const first: Request = .{ .prompt = &first_prompt, .max_tokens = 2000 };
    const second: Request = .{ .prompt = &second_prompt, .max_tokens = 4 };
    const e = host.engine();
    // the first pass's last chunk sends the second request: the first stream decodes while the second fills
    const Seen = struct {
        engine: api.Engine,
        request: *const Request,
        order: *Passes,
        target: *lanes.fake.Fake,
        core: *lanes.Engine,
        sent: bool = false,
        at: [4]u64 = @splat(0), // the core's rounds run before each step of the second pass (a lone stream's
        // pipelined round makes no verify call, so the fake's own count of rounds would miss them)
        steps: usize = 0,
        fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len == second_prompt.len) {
                if (a.steps < a.at.len) a.at[a.steps] = a.core.steps;
                a.steps += 1;
                return;
            }
            if (chunk + 1 < a.target.prefill_chunks or a.sent) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.order, .event = Passes.event }) catch {};
        }
    };
    var seen: Seen = .{ .engine = e, .request = &second, .order = &order, .target = &target, .core = &core };
    target.prefill_hook = Seen.call;
    target.prefill_hook_ctx = &seen;
    try e.submit(1, &first, .{ .ctx = &order, .event = Passes.event });
    try host.start();
    defer host.stop();
    order.wait(2);
    try std.testing.expectEqual(@as(usize, 4), seen.steps);
    for (seen.at[0..3], seen.at[1..4]) |a, b| try std.testing.expect(b - a >= 3); // the first stream's rounds between
    try std.testing.expectEqualSlices(Id, &.{ 1, 2 }, &order.prefilled);
}
