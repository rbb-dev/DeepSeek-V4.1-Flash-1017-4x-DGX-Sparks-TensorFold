//! The round loop on the fake target: drafted == one-token rounds, shared rounds == solo, greedy and sampled.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;
const grammar = @import("grammar.zig");
const win = @import("windows.zig");

const gpa = std.testing.allocator;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    sampling: ?Sampling = null,
    drafts: bool = true,
    think_budget: u32 = 0,
    logprobs: ?u8 = null,
    classes: ?Classes = null, // structured output under this test grammar
    eos: []const u32 = &.{96},
};

/// A test grammar: `length` tokens, the i-th of class i % 3 (any but 96 when `free`), then 96; it starts after `after`.
const Classes = struct {
    length: u32 = 12,
    after: ?u32 = null,
    free: bool = false,
    spell: bool = false, // the i-th token is want(i) alone, so the grammar drafts the rest of its text itself
    at: u32 = 0,
    done: bool = false,

    fn rules(c: *Classes) grammar.Rules {
        return .{ .ptr = c, .vtable = &.{ .accept = acceptFn, .rollback = rollbackFn, .terminated = terminatedFn, .fill = fillFn, .drafts = draftsFn, .free = freeFn } };
    }

    fn cast(ptr: *anyopaque) *Classes {
        return @ptrCast(@alignCast(ptr));
    }

    fn allows(c: *const Classes, t: u32) bool {
        if (c.done) return false;
        if (c.at == c.length) return t == 96;
        if (c.spell) return t == want(c.at);
        return t != 96 and (c.free or t % 3 == c.at % 3);
    }

    fn acceptFn(ptr: *anyopaque, t: u32) anyerror!bool {
        const c = cast(ptr);
        if (!c.allows(t)) return false;
        if (c.at == c.length) c.done = true else c.at += 1;
        return true;
    }

    fn rollbackFn(ptr: *anyopaque, n: usize) anyerror!void {
        const c = cast(ptr);
        for (0..n) |_| {
            if (c.done) c.done = false else c.at -= 1;
        }
    }

    fn terminatedFn(ptr: *anyopaque) bool {
        return cast(ptr).done;
    }

    fn fillFn(ptr: *anyopaque, words: []u32) anyerror!void {
        const c = cast(ptr);
        @memset(words, 0);
        for (0..fake.vocab) |t| if (c.allows(@intCast(t))) {
            words[t / 32] |= @as(u32, 1) << @intCast(t % 32);
        };
    }

    fn want(i: u32) u32 {
        return (5 + 7 * i) % 89;
    }

    fn draftsFn(ptr: *anyopaque, out: []u32) anyerror!usize {
        const c = cast(ptr);
        if (!c.spell or c.done) return 0;
        var n: usize = 0;
        while (n < out.len and c.at + n < c.length) : (n += 1) out[n] = want(c.at + @as(u32, @intCast(n)));
        return n;
    }

    fn freeFn(_: *anyopaque) void {}
};

fn model() !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

/// Every case's emitted tokens, the cases admitted together and stepped until done.
fn run(cases: []const Case) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    const classes = try gpa.alloc(Classes, cases.len);
    defer gpa.free(classes);
    const constraints = try gpa.alloc(grammar.Constraint, cases.len);
    defer gpa.free(constraints);
    for (cases, streams, proposers, classes, constraints) |c, *s, *p, *g, *k| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        if (c.classes) |given| {
            g.* = given;
            k.* = .init(g.rules(), (fake.vocab + 31) / 32, given.after);
        }
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = c.eos, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .logprobs = c.logprobs, .grammar = if (c.classes != null) k else null });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    for (cases, streams) |c, *s| if (c.logprobs) |k| try expectRows(c, s, k);
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

/// Each emitted token's row is the fake target's after the prompt and the tokens before it (forced: by `forToken`).
fn expectRows(c: Case, s: *const sm.Stream, k: u8) !void {
    try std.testing.expectEqual(s.emitted().len, s.rows.items.len);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, c.prompt);
    for (s.emitted(), s.rows.items) |t, got| {
        const want = fake.rowAt(history.items, fake.next(history.items, c.sampling, history.items.len), k).forToken(t);
        try std.testing.expectEqual(want.token, got.token);
        try std.testing.expectEqual(@as(u32, @bitCast(want.logprob)), @as(u32, @bitCast(got.logprob)));
        try std.testing.expectEqualSlices(u32, want.ids[0..k], got.ids[0..got.count]);
        try history.append(gpa, t);
    }
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const GuardCase = struct {
    prompt: []const u32,
    max_new: u32 = 400,
    drafts: bool = true,
    think_budget: u32 = 0,
    loop_guard: bool = true,
    loop_windows: u32 = 0,
    think_open: ?bool = null,
    cycle_after: usize = 70,
    pattern: []const u32 = &.{ 11, 12, 13 },
    answer_cycles: bool = false,
};

const GuardOut = struct {
    emitted: []u32,
    reason: sm.Reason,
    period: ?u32,
    novelty: ?f64,
    finished: bool,
};

fn runGuard(cases: []const GuardCase, step_limit: usize) ![]GuardOut {
    var cfg = try model();
    defer cfg.deinit(gpa);
    const c = cases[0];
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = c.cycle_after, .probe_prompt = c.prompt.len, .pattern = c.pattern, .answer_cycles = c.answer_cycles };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |item, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "guard", .prompt = item.prompt, .max_new = item.max_new, .eos = &.{96}, .drafts = item.drafts, .proposer = p.proposer(), .think_budget = item.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .think_open = item.think_open, .loop_guard = item.loop_guard, .loop_windows = item.loop_windows });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < step_limit) : (steps += 1) try engine.step();
    const out = try gpa.alloc(GuardOut, cases.len);
    for (out, streams) |*result, *s| result.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .period = s.loop_period, .novelty = s.loop_novelty, .finished = s.finished };
    return out;
}

fn freeGuard(runs: []GuardOut) void {
    for (runs) |run_result| gpa.free(run_result.emitted);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };

test "drafted rounds commit the one-token decode, greedy and sampled" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }});
        defer free(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }});
        defer free(plain);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        // and the fake target's own decode
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &p1);
        for (drafted[0]) |t| {
            try std.testing.expectEqual(fake.next(history.items, s, history.items.len), t);
            try history.append(gpa, t);
        }
    }
}

test "shared rounds commit what each stream commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } });
    defer free(together);
    const one = try run(&.{.{ .prompt = &p1 }});
    defer free(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30 }});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], together[0]);
    try std.testing.expectEqualSlices(u32, two[0], together[1]);
}

test "the thinking budget's forced close is the same drafted, plain and shared" {
    const drafted = try run(&.{.{ .prompt = &p2, .think_budget = 9 }});
    defer free(drafted);
    const plain = try run(&.{.{ .prompt = &p2, .think_budget = 9, .drafts = false }});
    defer free(plain);
    const shared = try run(&.{ .{ .prompt = &p2, .think_budget = 9 }, .{ .prompt = &p1, .drafts = false } });
    defer free(shared);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, plain[0], shared[0]);
    try std.testing.expectEqual(@as(u32, 90), drafted[0][8]);
}

test "logprob rows follow the committed tokens drafted, plain, shared, sampled and through a forced close" {
    const sampled: Sampling = .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 };
    const cases = [_][]const Case{
        &.{.{ .prompt = &p1, .logprobs = 3 }},
        &.{.{ .prompt = &p1, .logprobs = 3, .drafts = false }},
        &.{.{ .prompt = &p1, .logprobs = 0, .sampling = sampled }},
        &.{ .{ .prompt = &p1, .logprobs = 20 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30, .logprobs = 2 } },
        &.{.{ .prompt = &p2, .think_budget = 9, .logprobs = 4 }},
    };
    for (cases) |with| {
        var without: [2]Case = undefined;
        for (with, without[0..with.len]) |c, *w| w.* = .{ .prompt = c.prompt, .max_new = c.max_new, .sampling = c.sampling, .drafts = c.drafts, .think_budget = c.think_budget };
        const a = try run(with); // checks every row
        defer free(a);
        const b = try run(without[0..with.len]);
        defer free(b);
        for (a, b) |x, y| try std.testing.expectEqualSlices(u32, y, x);
    }
}

test "the loop guard matches drafted and plain, reports its period, and answers after close" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2 }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), drafted[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .stop);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, plain[0].emitted[329..]);

    const off = try runGuard(&.{.{ .prompt = &p2, .loop_guard = false, .think_open = true }}, 5000);
    defer freeGuard(off);
    try std.testing.expect(off[0].finished and off[0].reason == .length and off[0].emitted.len == 400);
}

test "the novelty guard matches drafted and plain, reports its share, and answers after close" {
    const pattern = [_]u32{ 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62 }; // a loop longer than any cycle
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false, .max_new = 4000, .loop_windows = 2, .pattern = &pattern }}, 10000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2, .max_new = 4000, .loop_windows = 2, .pattern = &pattern }}, 10000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expect(plain[0].period == null and plain[0].novelty.? == 0 and drafted[0].novelty.? == 0);
    try std.testing.expect(plain[0].finished and plain[0].reason == .stop);
    // the loop's second dry window ends at reply token 3 * 1024 - 1
    try std.testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, plain[0].emitted[3 * 1024 ..]);
}

test "a cycle in the answer does not refire or reclose" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), plain[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .length);
    var close_count: usize = 0;
    for (plain[0].emitted) |token| close_count += @intFromBool(token == 90);
    try std.testing.expectEqual(@as(usize, 1), close_count);
}

test "a prompt pass a chunk a round between other streams' rounds gives the whole pass's tokens" {
    const cases = [_]Case{ .{ .prompt = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4 } }, .{ .prompt = &.{ 9, 10, 11, 12, 13, 14, 15, 9, 10, 11 }, .max_new = 30, .sampling = .{ .temperature = 0.8, .seed = 7 } } };
    const whole = try run(&cases);
    defer free(whole);
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer engine.deinit();
    try std.testing.expect(engine.fills());
    var proposers: [2]SuffixLookup = undefined;
    var streams: [2]sm.Stream = undefined;
    for (cases, &streams, &proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    }
    defer for (&streams, &proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    while (!try engine.fillStream(&streams[0], target.prefill_count == 0)) {}
    var first = true;
    while (!try engine.fillStream(&streams[1], first)) {
        first = false;
        try engine.step(); // the first stream decodes between the second's chunks
    }
    try std.testing.expectEqual(@as(usize, 8), target.prefill_count);
    while (engine.activeCount() > 0) try engine.step();
    for (streams, whole) |s, w| try std.testing.expectEqualSlices(u32, w, s.emitted());
}

/// Every emitted token from the grammar's start on is one the test grammar allows, and it ends at its stop token.
fn expectInGrammar(emitted: []const u32, classes: Classes) !void {
    var g = classes;
    var active = g.after == null;
    for (emitted) |t| {
        if (!active) {
            active = t == g.after.?;
            continue;
        }
        try std.testing.expect(g.allows(t));
        _ = try Classes.acceptFn(&g, t);
    }
    try std.testing.expect(g.done);
}

test "structured output: drafted rounds commit the one-token decode, every token inside the grammar" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s, .classes = .{} }});
        defer free(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false, .classes = .{} }});
        defer free(plain);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        try expectInGrammar(drafted[0], .{});
    }
}

test "structured output: shared rounds with grammar and plain streams commit what each commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const cases = [_]Case{ .{ .prompt = &p1, .classes = .{ .length = 20 } }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 }, .{ .prompt = &p2, .sampling = sampled, .classes = .{ .length = 7 } } };
    const together = try run(&cases);
    defer free(together);
    for (cases, together) |c, got| {
        const alone = try run(&.{c});
        defer free(alone);
        try std.testing.expectEqualSlices(u32, alone[0], got);
        if (c.classes) |g| try expectInGrammar(got, g);
    }
}

test "structured output: with thinking the grammar starts after the think end, through a forced close" {
    const c: Case = .{ .prompt = &p2, .think_budget = 9, .classes = .{ .after = 91 } };
    const drafted = try run(&.{c});
    defer free(drafted);
    var plain_case = c;
    plain_case.drafts = false;
    const plain = try run(&.{plain_case});
    defer free(plain);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91 }, drafted[0][8..10]); // the close ends at the think end
    try expectInGrammar(drafted[0], .{ .after = 91 });
}

test "structured output: the grammar's stop token ends a reply that ignores eos" {
    const out = try run(&.{.{ .prompt = &p1, .eos = &.{}, .max_new = 200, .classes = .{ .length = 5 } }});
    defer free(out);
    try std.testing.expectEqual(@as(usize, 6), out[0].len);
    try expectInGrammar(out[0], .{ .length = 5 });
}

test "structured output: held head drafts come back as host tokens, so a grammar stream drafts every round" {
    for ([_]?Sampling{ null, .{ .seed = 3, .temperature = 0.9, .top_k = 0, .top_p = 0.9 } }) |sampling| {
        var cfg = try model();
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var p = try SuffixLookup.init(gpa, .{ .min_match = 1000 }); // no copies: every draft is the head's
        defer p.deinit();
        var g: Classes = .{ .length = 60, .free = true };
        var k: grammar.Constraint = .init(g.rules(), (fake.vocab + 31) / 32, null);
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &p1, .max_new = 80, .eos = &.{96}, .sampling = sampling, .proposer = p.proposer(), .think_close = &.{ 90, 91, 92 }, .think_end = 91, .grammar = &k });
        defer s.deinit(gpa);
        try engine.addStream(&s);
        while (engine.activeCount() > 0) try engine.step();
        try expectInGrammar(s.emitted(), .{ .length = 60, .free = true });
        // the fake head misses one draft in five: a grammar that takes them all keeps well over a token a round
        try std.testing.expect(s.drafted >= 2 * s.rounds and 2 * s.emitted().len > 3 * s.rounds);
    }
}

test "structured output: where the grammar spells its own text, its drafts fill the window and drafted == plain" {
    for ([_]?Sampling{ null, .{ .seed = 11, .temperature = 0.8, .top_k = 0, .top_p = 0.9 } }) |sampling| {
        const c: Case = .{ .prompt = &p1, .sampling = sampling, .max_new = 80, .classes = .{ .length = 50, .spell = true } };
        var plain_case = c;
        plain_case.drafts = false;
        const plain = try run(&.{plain_case});
        defer free(plain);
        var cfg = try model();
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        defer p.deinit();
        var g: Classes = c.classes.?;
        var k: grammar.Constraint = .init(g.rules(), (fake.vocab + 31) / 32, null);
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = sampling, .proposer = p.proposer(), .think_close = &.{ 90, 91, 92 }, .think_end = 91, .grammar = &k });
        defer s.deinit(gpa);
        try engine.addStream(&s);
        while (engine.activeCount() > 0) try engine.step();
        try std.testing.expectEqualSlices(u32, plain[0], s.emitted());
        try expectInGrammar(s.emitted(), c.classes.?);
        // the head's guesses are wrong here; the grammar's own drafts land every round, a window's worth at a time
        try std.testing.expect(s.emitted().len > 4 * s.rounds);
    }
}

test "the head's chains by depth add up to its drafts and kept drafts, each depth keeping no more than it reached" {
    for ([_]?Sampling{ null, .{ .seed = 4, .temperature = 0.8, .top_k = 0, .top_p = 0.9 } }) |sampling| {
        var cfg = try model();
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var p = try SuffixLookup.init(gpa, .{ .min_match = 1000 }); // no copies: every draft is the head's chain
        defer p.deinit();
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &p1, .max_new = 80, .eos = &.{96}, .sampling = sampling, .proposer = p.proposer() });
        defer s.deinit(gpa);
        try engine.addStream(&s);
        while (engine.activeCount() > 0) try engine.step();
        var drafted: u64 = 0;
        var kept: u64 = 0;
        for (s.depths, 0..) |d, j| {
            drafted += d.drafted;
            kept += d.kept;
            try std.testing.expect(d.kept <= d.reached and d.reached <= d.drafted);
            if (j > 0) try std.testing.expect(d.drafted <= s.depths[j - 1].drafted and d.reached <= s.depths[j - 1].kept);
        }
        try std.testing.expectEqual(s.drafted, drafted);
        try std.testing.expectEqual(s.accepted, kept);
        // every chain reaches its first draft; the fake head misses one draft in five, so some chain stops short
        try std.testing.expectEqual(s.depths[0].drafted, s.depths[0].reached);
        try std.testing.expect(s.depths[1].drafted > 0 and kept < drafted);
    }
}

test "a family that keeps copies to shared rounds proposes none for a lone stream" {
    for ([_]bool{ true, false }) |lone_copies| {
        var cfg = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 4, .lone_copies = lone_copies }, 16, 15);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
/// 13 from its 24th token, so the proposer finds copies there), copies as `copies` and `enter_match` say, the decision
/// log on.
fn runPlans(prompts: []const []const u32, copies: bool, enter_match: i64) ![]PlanOut {
    cfg.copies = copies;
    cfg.enter_match = enter_match;
    return runPlansOn(&cfg, prompts, null);
}

/// runPlans on `cfg`; `rows`: each copy round's window rows, by stream, appended.
fn runPlansOn(cfg: *Config, prompts: []const []const u32, rows: ?[]std.ArrayList(usize)) ![]PlanOut {
    var engine = Engine.init(gpa, cfg, target.backend(), clock.clock());
    for (out, streams, proposers, ids[0..prompts.len], 0..) |*o, *st, *p, id, k| {
            if (rows) |r| if (c == &o.logged.copy) try r[k].append(gpa, window.len);
        const runs = try runPlans(prompts, true, 8);
test "copies off (TF_DS_NO_COPIES): no copied continuation, the proposer never asked, the same tokens" {
    const on = try runPlans(&.{&p1}, true, 8);
    defer freePlans(on);
    const off = try runPlans(&.{&p1}, false, 8);
    defer freePlans(off);
    try std.testing.expect(on[0].plans.copy.rounds > 0);
    try std.testing.expectEqual(sm.Plans.Count{}, off[0].plans.copy);
    try std.testing.expectEqual(@as(i64, 0), off[0].proposals);
    try std.testing.expect(off[0].plans.head.rounds > on[0].plans.head.rounds);
    try std.testing.expectEqualSlices(u32, on[0].emitted, off[0].emitted);
}
test "a copy's entry match (TF_DS_COPY_MATCH): a match of the setting copies, one token short does not; a later entry, the same tokens" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    // 1 .. 8 after 90, then 1 .. 8 again: the last 3-gram's earlier match is 8 tokens long; 9, 1, 2, ... follow it
    const ctx = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8 };
    for ([_]i64{ 4, 8, 9 }, [_]bool{ true, true, false }) |match, copies| {
        var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        defer p.deinit();
        var st = try sm.Stream.init(gpa, .{ .id = "m", .prompt = &ctx, .max_new = 40, .proposer = p.proposer() });
        defer st.deinit(gpa);
        cfg.enter_match = match;
        const got = try win.copyProposal(&engine, &st);
        if (copies) try std.testing.expectEqualSlices(u32, &.{ 9, 1, 2, 3, 4, 5, 6, 7 }, got.tokens[0..8]) else try std.testing.expectEqual(@as(usize, 0), got.tokens.len);
        // copies off: none at any match
        cfg.copies = false;
        try std.testing.expectEqual(@as(usize, 0), (try win.copyProposal(&engine, &st)).tokens.len);
        cfg.copies = true;
    }
    // over a run: at 16 the stream copies later and in fewer rounds than at 8, its tokens the same
    const at8 = try runPlans(&.{&p1}, true, 8);
    defer freePlans(at8);
    const at16 = try runPlans(&.{&p1}, true, 16);
    defer freePlans(at16);
    try std.testing.expect(at16[0].plans.copy.rounds > 0 and at16[0].plans.copy.rounds < at8[0].plans.copy.rounds);
    try std.testing.expectEqualSlices(u32, at8[0].emitted, at16[0].emitted);
}

/// The served DeepSeek family's shape: windows of the pending row and up to five drafts, 48 rows a forward; `copy_rows`
/// a lone stream's copied window's rows at most (0: every window's six).
fn sixRows(copy_rows: u32) !Config {
    var costs: [6]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 20.0 + 4.0 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 6, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 5, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 48, .max_streams = 8, .draft_streams = true, .copy_rows = copy_rows }, 48, 47);
}

/// A reply that repeats 40 tokens from its 24th: once it has, each copy can run 37 tokens (a period of three, the
/// fake's own, bounds every copy at three).
const long_cycle = blk: {
    var x: [40]u32 = undefined;
    for (&x, 0..) |*t, i| t.* = 50 + i;
    break :blk x;
};

/// A wide-copy run (runPlans' streams on `cfg`, the reply repeating long_cycle): each stream's tokens and rounds, every
/// copy window's rows with whether its step had one live stream, and the copy widths the decision log's states held.
const WideOut = struct {
    emitted: [2][]u32 = .{ &.{}, &.{} },
    rounds: [2]u64 = .{ 0, 0 },
    copies: std.ArrayList([2]usize) = .empty, // (rows, alone) of each copy window
    widths: std.ArrayList(i64) = .empty,

    fn deinit(o: *WideOut) void {
        for (o.emitted) |x| gpa.free(x);
        o.copies.deinit(gpa);
        o.widths.deinit(gpa);
    }
};

fn runWide(cfg: *Config, prompts: []const []const u32) !WideOut {
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 24, .probe_prompt = prompts[0].len, .pattern = &long_cycle };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var log = ev.Log.init(gpa);
    defer log.deinit();
    engine.log = &log;
    const ids = [_][]const u8{ "a", "b" };
    var streams: [2]sm.Stream = undefined;
    var proposers: [2]SuffixLookup = undefined;
    for (prompts, streams[0..prompts.len], proposers[0..prompts.len], ids[0..prompts.len]) |prompt, *st, *p, id| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        st.* = try sm.Stream.init(gpa, .{ .id = id, .prompt = prompt, .max_new = 160, .eos = &.{}, .proposer = p.proposer() });
    }
    defer for (streams[0..prompts.len], proposers[0..prompts.len]) |*st, *p| {
        st.deinit(gpa);
        p.deinit();
    };
    for (streams[0..prompts.len]) |*st| try engine.addStream(st);
    while (engine.activeCount() > 0) try engine.step();
    var o: WideOut = .{};
    errdefer o.deinit();
    for (streams[0..prompts.len], 0..) |*st, k| {
        o.emitted[k] = try gpa.dupe(u32, st.emitted());
        o.rounds[k] = st.rounds;
    }
    // a step's rounds come before its state (engine.step): a copy's rows wait for the state that says whether it was alone
    var pending: std.ArrayList(usize) = .empty;
    defer pending.deinit(gpa);
    for (log.events.items) |fields| {
        const what = field(fields, "ev").?.str;
        if (std.mem.eql(u8, what, "round") and std.mem.eql(u8, field(fields, "kind").?.str, "copy")) {
            try pending.append(gpa, field(fields, "window").?.u32s.len);
        } else if (std.mem.eql(u8, what, "state")) {
            const alone = field(fields, "alone").?.bool;
            for (pending.items) |n| try o.copies.append(gpa, .{ n, @intFromBool(alone) });
            pending.clearRetainingCapacity();
            for (field(fields, "streams").?.list) |s| switch (field(s.obj, "copy_width").?) {
                .int => |x| try o.widths.append(gpa, x),
                else => {},
            };
        }
    }
    return o;
}

test "wide copies (TF_DS_COPY_ROWS): a lone stream's copies span up to copy_rows rows on the rungs, start at 8 drafts and double; shared rounds' stay six; the same tokens" {
    var c0 = try sixRows(0);
    defer c0.deinit(gpa);
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    // alone, unset: every copy at most six rows and the width every window's (5), as served
    var narrow = try runWide(&c0, &.{&p1});
    defer narrow.deinit();
    try std.testing.expect(narrow.copies.items.len > 0);
    for (narrow.copies.items) |x| try std.testing.expect(x[0] <= 6);
    for (narrow.widths.items) |w| try std.testing.expectEqual(@as(i64, 5), w);
    // alone, 17: copies past six rows, each on a rung (9, 13 or 17 rows); the width 8, then 16 while every copied token lands
    var wide = try runWide(&c17, &.{&p1});
    defer wide.deinit();
    var past: usize = 0;
    for (wide.copies.items) |x| {
        try std.testing.expect(x[0] <= 6 or x[0] == 9 or x[0] == 13 or x[0] == 17);
        past += @intFromBool(x[0] > 6);
    }
    try std.testing.expect(past > 0);
    var grown = false;
    for (wide.widths.items) |w| {
        try std.testing.expect(w == 8 or w == 16);
        grown = grown or w == 16;
    }
    try std.testing.expect(grown);
    // the same tokens in fewer rounds, and the same as with no copies at all (drafted == serial)
    try std.testing.expectEqualSlices(u32, narrow.emitted[0], wide.emitted[0]);
    try std.testing.expect(wide.rounds[0] < narrow.rounds[0]);
    c17.copies = false;
    var off = try runWide(&c17, &.{&p1});
    defer off.deinit();
    c17.copies = true;
    try std.testing.expectEqualSlices(u32, off.emitted[0], wide.emitted[0]);
    // two streams: a shared round's copies at most six rows; each stream's tokens as without the setting
    var two0 = try runWide(&c0, &.{ &p1, &p3 });
    defer two0.deinit();
    var two17 = try runWide(&c17, &.{ &p1, &p3 });
    defer two17.deinit();
    var shared: usize = 0;
    for (two17.copies.items) |x| if (x[1] == 0) {
        try std.testing.expect(x[0] <= 6);
        shared += 1;
    };
    try std.testing.expect(shared > 0);
    for (two0.emitted, two17.emitted) |a, b| try std.testing.expectEqualSlices(u32, a, b);
}

test "a wide copy's proposal: alone, wide_first drafts and then the stream's grown width; shared, every window's width" {
    var cfg = try sixRows(17);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    // 1 .. 30, 100, then 1 .. 20 again: the last 3-gram's earlier match is 20 tokens long and 31 tokens follow it
    var ctx: [51]u32 = undefined;
    for (ctx[0..30], 1..) |*t, i| t.* = @intCast(i);
    ctx[30] = 100;
    for (ctx[31..], 1..) |*t, i| t.* = @intCast(i);
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var st = try sm.Stream.init(gpa, .{ .id = "w", .prompt = &ctx, .max_new = 40, .proposer = p.proposer() });
    defer st.deinit(gpa);
    engine.alone = true;
    try std.testing.expectEqualSlices(u32, &.{ 21, 22, 23, 24, 25, 26, 27, 28 }, (try win.copyProposal(&engine, &st)).tokens);
    st.copy_width = 16;
    try std.testing.expectEqual(@as(usize, 16), (try win.copyProposal(&engine, &st)).tokens.len);
    st.copy_width = 13; // a width between rungs takes the rung under it
    try std.testing.expectEqual(@as(usize, 12), (try win.copyProposal(&engine, &st)).tokens.len);
    engine.alone = false;
    try std.testing.expectEqual(@as(usize, 5), (try win.copyProposal(&engine, &st)).tokens.len);
    // without the setting, alone: every window's five
    var narrow = try sixRows(0);
    defer narrow.deinit(gpa);
    var e0 = Engine.init(gpa, &narrow, target.backend(), clock.clock());
    defer e0.deinit();
    e0.alone = true;
    st.copy_width = null;
    try std.testing.expectEqual(@as(usize, 5), (try win.copyProposal(&e0, &st)).tokens.len);
}

test "a wide copy's rung: every window's width as it is, past it 8, 12 or 16 drafts under the most, or the most" {
    // most 16 (copy_rows 17): 6 and 7 back to every window's 5; 8 .. 11 -> 8; 12 .. 15 -> 12; 16 -> 16
    const want16 = [_]usize{ 0, 1, 2, 3, 4, 5, 5, 5, 8, 8, 8, 8, 12, 12, 12, 12, 16 };
    for (want16, 0..) |w, n| try std.testing.expectEqual(w, win.rung(n, 5, 16));
    // most 14 (copy_rows 15): the most is a rung of its own
    try std.testing.expectEqual(@as(usize, 12), win.rung(13, 5, 14));
    try std.testing.expectEqual(@as(usize, 14), win.rung(14, 5, 14));
    // most 6 (copy_rows 7): 6 alone
    try std.testing.expectEqual(@as(usize, 6), win.rung(6, 5, 6));
    // unset (most = every window's 5): nothing past 5 is ever proposed, and up to it nothing changes
    for (0..6) |n| try std.testing.expectEqual(n, win.rung(n, 5, 5));
}

test "a model's copy settings reach the round loop's: on and 8 unless it says otherwise" {
    var c = try Config.init(gpa, .{ .exact_width = 6 }, 6, 5);
    defer c.deinit(gpa);
    try std.testing.expect(c.copies);
    try std.testing.expectEqual(@as(i64, 8), c.enter_match);
    var d = try Config.init(gpa, .{ .exact_width = 6, .copies = false, .enter_match = 16 }, 6, 5);
    defer d.deinit(gpa);
    try std.testing.expect(!d.copies);
    try std.testing.expectEqual(@as(i64, 16), d.enter_match);
}

// -- finishes, cuts and guards inside wide copy windows (TF_DS_COPY_ROWS); each run against the same run, copies off --

/// A stop check on the reply's last three tokens, from `min_len` tokens on.
const Stop3 = struct {
    triple: [3]u32,
    min_len: usize = 0,
    fn check(ptr: *anyopaque, emitted: []const u32) bool {
        const c: *@This() = @ptrCast(@alignCast(ptr));
        if (emitted.len < 3 or emitted.len < c.min_len) return false;
        return std.mem.eql(u32, emitted[emitted.len - 3 ..], &c.triple);
    }
};

/// One lone stream's run on long_cycle with `holes` (the reply's token 5 there): its reply, reason and rounds, its copy
/// windows past six rows, whether its last round was one, and the windows that ran past the stream's room (a window's
/// rows at most max_new less the tokens before it).
const OneRun = struct { emitted: []u32, reason: sm.Reason, wide: usize, ended_wide: bool, past_room: usize };

const OneOpts = struct { max_new: u32 = 200, think_budget: u32 = 0, stop: ?*Stop3 = null, eos: []const u32 = &.{}, holes: []const usize = &.{} };

fn runOne(cfg: *Config, o: OneOpts) !OneRun {
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 24, .probe_prompt = p1.len, .pattern = &long_cycle, .holes = o.holes };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var log = ev.Log.init(gpa);
    defer log.deinit();
    engine.log = &log;
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var st = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &p1, .max_new = o.max_new, .eos = o.eos, .proposer = p.proposer(), .think_budget = o.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .stop_check = if (o.stop) |s| .{ .ptr = s, .check = Stop3.check } else null });
    defer st.deinit(gpa);
    try engine.addStream(&st);
    while (engine.activeCount() > 0) try engine.step();
    var out: OneRun = .{ .emitted = try gpa.dupe(u32, st.emitted()), .reason = st.reason, .wide = 0, .ended_wide = false, .past_room = 0 };
    // the tokens before each round: the reply's less what this round and the later ones committed
    var after: usize = 0;
    var i = log.events.items.len;
    while (i > 0) {
        i -= 1;
        const fields = log.events.items[i];
        if (!std.mem.eql(u8, field(fields, "ev").?.str, "round")) continue;
        after += field(fields, "got").?.u32s.len;
        const rows = field(fields, "window").?.u32s.len;
        const before = out.emitted.len - after;
        if (std.mem.eql(u8, field(fields, "kind").?.str, "copy") and rows > o.max_new - before) out.past_room += 1;
        if (!std.mem.eql(u8, field(fields, "kind").?.str, "copy") or rows <= 6) continue;
        out.wide += 1;
        if (after == field(fields, "got").?.u32s.len) out.ended_wide = true;
    }
    return out;
}

/// A run with copies and the same run with copies off: the same reply and reason; the wide windows and those ending it.
fn sameAsCopiesOff(cfg: *Config, o: OneOpts) ![2]usize {
    cfg.copies = true;
    const w = try runOne(cfg, o);
    defer gpa.free(w.emitted);
    cfg.copies = false;
    const s = try runOne(cfg, o);
    defer gpa.free(s.emitted);
    cfg.copies = true;
    try std.testing.expectEqualSlices(u32, s.emitted, w.emitted);
    try std.testing.expectEqual(s.reason, w.reason);
    try std.testing.expectEqual(@as(usize, 0), w.past_room);
    return .{ w.wide, @intFromBool(w.ended_wide) };
}

test "wide copies: max_new at every offset of a window, the same tokens and reason as copies off, no window past the stream's room" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var wide: usize = 0;
    var ended: usize = 0;
    var n: u32 = 40;
    while (n <= 150) : (n += 1) {
        const x = try sameAsCopiesOff(&c17, .{ .max_new = n });
        wide += x[0];
        ended += x[1];
    }
    try std.testing.expect(wide > 0 and ended > 0);
}

test "wide copies: the thinking budget's cut at every offset of a window, the same tokens and reason as copies off" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var wide: usize = 0;
    var n: u32 = 20;
    while (n <= 140) : (n += 1) wide += (try sameAsCopiesOff(&c17, .{ .think_budget = n }))[0];
    try std.testing.expect(wide > 0);
}

test "wide copies: a stop sequence at every offset of a window, the same tokens as copies off, ended by the stop" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var wide: usize = 0;
    var ended: usize = 0;
    var k: u32 = 0;
    while (k < 38) : (k += 1) for ([_]usize{ 80, 120, 160, 200 }) |min_len| {
        var stop: Stop3 = .{ .triple = .{ 50 + k, 50 + k + 1, 50 + k + 2 }, .min_len = min_len };
        const x = try sameAsCopiesOff(&c17, .{ .max_new = 400, .stop = &stop });
        wide += x[0];
        ended += x[1];
        const r = try runOne(&c17, .{ .max_new = 400, .stop = &stop });
        defer gpa.free(r.emitted);
        try std.testing.expect(r.reason == .stop);
    };
    try std.testing.expect(wide > 0 and ended > 0);
}

test "wide copies: an end-of-sequence token at every offset of a window, the same tokens as copies off, ended by it" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var wide: usize = 0;
    var ended: usize = 0;
    var h: usize = 70;
    while (h < 200) : (h += 1) {
        // the reply's token 5 at index h is the end of sequence: a copy of the pattern holds no 5, so the window's draft
        // there fails and its draw of the row ends the reply
        const x = try sameAsCopiesOff(&c17, .{ .max_new = 260, .eos = &.{5}, .holes = &.{h} });
        wide += x[0];
        ended += x[1];
        const r = try runOne(&c17, .{ .max_new = 260, .eos = &.{5}, .holes = &.{h} });
        defer gpa.free(r.emitted);
        try std.testing.expect(r.reason == .stop);
        try std.testing.expectEqual(h + 1, r.emitted.len);
    }
    try std.testing.expect(wide > 0 and ended > 0);
}

/// The kept counts of a lone stream's 17-row copy windows that did not land whole, and their count.
const Misses = struct { kept: [17]bool = @splat(false), n: usize = 0 };

test "wide copies: a deviation at every reply index (a miss at every offset of a 17-row window), the same tokens as copies off" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var seen: Misses = .{};
    var h: usize = 70;
    while (h < 200) : (h += 1) {
        _ = try sameAsCopiesOff(&c17, .{ .max_new = 260, .holes = &.{h} });
        var rounds: std.ArrayList(CopyRound) = .empty;
        defer rounds.deinit(gpa);
        try copyRounds(&c17, &.{h}, null, &rounds);
        for (rounds.items) |r| if (r.rows == 17 and r.kept < 16) {
            seen.kept[r.kept] = true;
            seen.n += 1;
        };
    }
    var distinct: usize = 0;
    for (seen.kept) |k| distinct += @intFromBool(k);
    try std.testing.expect(distinct >= 15);
}

const CopyRound = struct { rows: usize, kept: usize, alone: bool, stream: u8 };

/// A lone run's copy rounds (rows, kept, whether its step had one live stream) on long_cycle with `holes`; with
/// `second`, stream b (prompt p3, `second` tokens) joins at the first step.
fn copyRounds(cfg: *Config, holes: []const usize, second: ?u32, out: *std.ArrayList(CopyRound)) !void {
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 24, .probe_prompt = p1.len, .pattern = &long_cycle, .holes = holes };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var log = ev.Log.init(gpa);
    defer log.deinit();
    engine.log = &log;
    var pa = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pa.deinit();
    var pb = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pb.deinit();
    var sa = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &p1, .max_new = 300, .eos = &.{}, .proposer = pa.proposer() });
    defer sa.deinit(gpa);
    var sb = try sm.Stream.init(gpa, .{ .id = "b", .prompt = &p3, .max_new = second orelse 1, .eos = &.{}, .proposer = pb.proposer() });
    defer sb.deinit(gpa);
    try engine.addStream(&sa);
    if (second != null) try engine.addStream(&sb);
    while (engine.activeCount() > 0) try engine.step();
    // a step's rounds come before its state (engine.step): the rounds since the last state are that step's
    var step_from = out.items.len;
    for (log.events.items) |fields| {
        const what = field(fields, "ev").?.str;
        if (std.mem.eql(u8, what, "round") and std.mem.eql(u8, field(fields, "kind").?.str, "copy")) {
            try out.append(gpa, .{ .rows = field(fields, "window").?.u32s.len, .kept = field(fields, "path").?.u32s.len - 1, .alone = false, .stream = field(fields, "stream").?.str[0] });
        } else if (std.mem.eql(u8, what, "state")) {
            const alone = field(fields, "alone").?.bool;
            for (out.items[step_from..]) |*r| r.alone = alone;
            step_from = out.items.len;
        }
    }
}

test "wide copies: after a copy window that did not land whole, the stream's next copy opens at 8 drafts" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var checked: usize = 0;
    var kept: [17]bool = @splat(false);
    var h: usize = 100;
    while (h < 160) : (h += 1) {
        var rounds: std.ArrayList(CopyRound) = .empty;
        defer rounds.deinit(gpa);
        try copyRounds(&c17, &.{h}, null, &rounds);
        const r = rounds.items;
        for (r[0..r.len -| 1], r[1..], 0..) |a, b, i| {
            if (i == 0 or a.kept == a.rows - 1) continue; // the stream's first copy, or one that landed whole
            checked += 1;
            if (a.rows == 17) kept[a.kept] = true;
            if (b.rows > 6) try std.testing.expectEqual(@as(usize, 9), b.rows);
        }
    }
    try std.testing.expect(checked > 0 and kept[0] and kept[1] and kept[15]);
}

test "wide copies: a lone stream's first copy that lands whole doubles from 8 drafts, its next window 17 rows" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 0, .probe_prompt = p1.len, .pattern = &long_cycle };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &c17, target.backend(), clock.clock());
    defer engine.deinit();
    var log = ev.Log.init(gpa);
    defer log.deinit();
    engine.log = &log;
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var st = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &p1, .max_new = 200, .eos = &.{}, .proposer = p.proposer() });
    defer st.deinit(gpa);
    try engine.addStream(&st);
    while (engine.activeCount() > 0) try engine.step();
    var copies: std.ArrayList([2]usize) = .empty;
    defer copies.deinit(gpa);
    for (log.events.items) |fields| {
        if (!std.mem.eql(u8, field(fields, "ev").?.str, "round") or !std.mem.eql(u8, field(fields, "kind").?.str, "copy")) continue;
        try copies.append(gpa, .{ field(fields, "window").?.u32s.len, field(fields, "path").?.u32s.len - 1 });
    }
    // a reply of 40 distinct tokens repeating: its first copy is the stream's first, 8 drafts, and lands whole
    try std.testing.expect(copies.items.len >= 2);
    try std.testing.expectEqual([2]usize{ 9, 8 }, copies.items[0]);
    try std.testing.expectEqual(@as(usize, 17), copies.items[1][0]);
}

test "wide copies: a second stream from the start, the first's copy width does not move in the shared steps; alone again, its first copy opens at 8 drafts" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var rounds: std.ArrayList(CopyRound) = .empty;
    defer rounds.deinit(gpa);
    try copyRounds(&c17, &.{}, 120, &rounds);
    var shared_whole: usize = 0;
    var after_shared: ?CopyRound = null;
    for (rounds.items) |r| {
        if (r.stream != 'a') continue;
        if (!r.alone) {
            try std.testing.expect(r.rows <= 6);
            shared_whole += @intFromBool(r.kept == r.rows - 1);
        } else if (shared_whole > 0 and after_shared == null and r.rows > 6) after_shared = r;
    }
    // the first stream's copies landed whole in shared steps; its width (never set alone) is still the first
    try std.testing.expect(shared_whole > 0);
    try std.testing.expectEqual(@as(usize, 9), after_shared.?.rows);
}

test "wide copies: a second stream admitted while the first copies wide, each reply its own alone, no wide window in a shared step" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    const alone = try runOne(&c17, .{ .max_new = 300 });
    defer gpa.free(alone.emitted);
    var wide: usize = 0;
    for ([_]u32{ 10, 40, 90, 200 }) |b_len| {
        var at: usize = 3;
        while (at < 40) : (at += 1) {
            const t = try runAdmitted(&c17, at, b_len);
            defer gpa.free(t.a);
            try std.testing.expectEqualSlices(u32, alone.emitted, t.a);
            try std.testing.expectEqual(@as(usize, 0), t.shared_wide);
            wide += t.wide_a;
        }
    }
    try std.testing.expect(wide > 0);
}

/// Stream a alone on long_cycle (300 tokens); b (p3, `b_len` tokens) admitted before step `at`: a's reply, a's copy
/// windows past six rows, and the wide copy windows of any step with two live streams.
fn runAdmitted(cfg: *Config, at: usize, b_len: u32) !struct { a: []u32, wide_a: usize, shared_wide: usize } {
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 24, .probe_prompt = p1.len, .pattern = &long_cycle };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var log = ev.Log.init(gpa);
    defer log.deinit();
    engine.log = &log;
    var pa = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pa.deinit();
    var pb = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer pb.deinit();
    var sa = try sm.Stream.init(gpa, .{ .id = "a", .prompt = &p1, .max_new = 300, .eos = &.{}, .proposer = pa.proposer() });
    defer sa.deinit(gpa);
    var sb = try sm.Stream.init(gpa, .{ .id = "b", .prompt = &p1, .max_new = b_len, .eos = &.{}, .proposer = pb.proposer() });
    defer sb.deinit(gpa);
    try engine.addStream(&sa);
    var steps: usize = 0;
    while (engine.activeCount() > 0) : (steps += 1) {
        if (steps == at) try engine.addStream(&sb);
        try engine.step();
    }
    var wide_a: usize = 0;
    var shared_wide: usize = 0;
    var pending: std.ArrayList(usize) = .empty;
    defer pending.deinit(gpa);
    for (log.events.items) |fields| {
        const what = field(fields, "ev").?.str;
        if (std.mem.eql(u8, what, "round") and std.mem.eql(u8, field(fields, "kind").?.str, "copy")) {
            const n = field(fields, "window").?.u32s.len;
            try pending.append(gpa, n);
            if (n > 6 and field(fields, "stream").?.str[0] == 'a') wide_a += 1;
        } else if (std.mem.eql(u8, what, "state")) {
            if (!field(fields, "alone").?.bool) for (pending.items) |n| {
                shared_wide += @intFromBool(n > 6);
            };
            pending.clearRetainingCapacity();
        }
    }
    return .{ .a = try gpa.dupe(u32, sa.emitted()), .wide_a = wide_a, .shared_wide = shared_wide };
}

/// A lone run on periodic text of `period` tokens (60 ..) with the loop guard: reply, reason, the guard's period and
/// the copy windows past six rows.
fn runGuarded(cfg: *Config, period: usize) !struct { emitted: []u32, reason: sm.Reason, period: ?u32, wide: usize } {
    var pat: [16]u32 = undefined;
    for (pat[0..period], 0..) |*t, i| t.* = @intCast(60 + i);
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 70, .probe_prompt = p2.len, .pattern = pat[0..period] };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var log = ev.Log.init(gpa);
    defer log.deinit();
    engine.log = &log;
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var st = try sm.Stream.init(gpa, .{ .id = "g", .prompt = &p2, .max_new = 600, .eos = &.{96}, .proposer = p.proposer(), .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer st.deinit(gpa);
    try engine.addStream(&st);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < 5000) : (steps += 1) try engine.step();
    var wide: usize = 0;
    for (log.events.items) |fields| {
        if (!std.mem.eql(u8, field(fields, "ev").?.str, "round") or !std.mem.eql(u8, field(fields, "kind").?.str, "copy")) continue;
        wide += @intFromBool(field(fields, "window").?.u32s.len > 6);
    }
    return .{ .emitted = try gpa.dupe(u32, st.emitted()), .reason = st.reason, .period = st.loop_period, .wide = wide };
}

test "wide copies: the loop guard on periodic text of every period 3 .. 16, the same tokens, reason and period as copies off" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var wide: usize = 0;
    var fired: usize = 0;
    for (3..17) |period| {
        c17.copies = true;
        const w = try runGuarded(&c17, period);
        defer gpa.free(w.emitted);
        c17.copies = false;
        const o = try runGuarded(&c17, period);
        defer gpa.free(o.emitted);
        c17.copies = true;
        try std.testing.expectEqualSlices(u32, o.emitted, w.emitted);
        try std.testing.expectEqual(o.reason, w.reason);
        try std.testing.expectEqual(o.period, w.period);
        wide += w.wide;
        fired += @intFromBool(w.period != null);
    }
    try std.testing.expect(wide > 0 and fired > 0);
}

/// A grammar stream (Classes of `length`) on long_cycle: reply, reason and the windows past six rows.
fn runGrammar(cfg: *Config, length: u32) !struct { emitted: []u32, reason: sm.Reason, wide: usize } {
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 0, .probe_prompt = p1.len, .pattern = &long_cycle };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var log = ev.Log.init(gpa);
    defer log.deinit();
    engine.log = &log;
    var classes: Classes = .{ .length = length, .free = true };
    var constraint: grammar.Constraint = .init(classes.rules(), (fake.vocab + 31) / 32, null);
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var st = try sm.Stream.init(gpa, .{ .id = "g", .prompt = &p1, .max_new = 400, .eos = &.{96}, .proposer = p.proposer(), .grammar = &constraint });
    defer st.deinit(gpa);
    try engine.addStream(&st);
    while (engine.activeCount() > 0) try engine.step();
    var wide: usize = 0;
    for (log.events.items) |fields| {
        if (!std.mem.eql(u8, field(fields, "ev").?.str, "round")) continue;
        wide += @intFromBool(field(fields, "window").?.u32s.len > 6);
    }
    return .{ .emitted = try gpa.dupe(u32, st.emitted()), .reason = st.reason, .wide = wide };
}

test "wide copies: a grammar stream, every length of its text, the same tokens and reason as copies off" {
    var c17 = try sixRows(17);
    defer c17.deinit(gpa);
    var wide: usize = 0;
    var length: u32 = 90;
    while (length <= 200) : (length += 1) {
        c17.copies = true;
        const w = try runGrammar(&c17, length);
        defer gpa.free(w.emitted);
        c17.copies = false;
        const o = try runGrammar(&c17, length);
        defer gpa.free(o.emitted);
        c17.copies = true;
        try std.testing.expectEqualSlices(u32, o.emitted, w.emitted);
        try std.testing.expectEqual(o.reason, w.reason);
        wide += w.wide;
    }
    try std.testing.expect(wide > 0);
}

test "wide copies with the host's own draws (gpu_tokens false, as the served family): the same tokens as copies off and as unset" {
    var costs: [6]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 20.0 + 4.0 * @as(f64, @floatFromInt(w)) };
    var cfgs: [2]Config = undefined;
    for (&cfgs, [_]u32{ 0, 17 }) |*c, rows| c.* = try Config.init(gpa, .{ .exact_width = 6, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 5, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 48, .max_streams = 8, .draft_streams = true, .copy_rows = rows }, 48, 47);
    defer for (&cfgs) |*c| c.deinit(gpa);
    const x = try sameAsCopiesOff(&cfgs[1], .{ .max_new = 200 });
    try std.testing.expect(x[0] > 0);
    const unset = try runOne(&cfgs[0], .{ .max_new = 200 });
    defer gpa.free(unset.emitted);
    const wide = try runOne(&cfgs[1], .{ .max_new = 200 });
    defer gpa.free(wide.emitted);
    try std.testing.expectEqualSlices(u32, unset.emitted, wide.emitted);
}
