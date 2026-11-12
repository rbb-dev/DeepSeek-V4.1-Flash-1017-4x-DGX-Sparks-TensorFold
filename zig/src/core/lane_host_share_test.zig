//! The share of wall time for prompt steps beside decoding streams, through the served lane host over the chunked fake
//! (lane_host_share.zig): how many decode rounds follow a prompt step, in what order prompts go, what is owed to whom.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const Share = @import("lane_host_share.zig").Share;
const gpa = std.testing.allocator;

/// The share's clock, so a test chooses what a step and a round cost: every reading is one quantum later than the
/// last, and a prompt step costs half a quantum a row on top. A step read twice is then (1 + rows / 2) quanta and a
/// round one quantum, whatever the machine does meanwhile.
const StepClock = struct {
    t: i96 = 0,
    reads: usize = 0,

    const quantum: i96 = 1000;

    fn now(ctx: *anyopaque) i96 {
        const c: *StepClock = @ptrCast(@alignCast(ctx));
        c.reads += 1;
        c.t += quantum;
        return c.t;
    }

    fn rows(c: *StepClock, n: usize) void {
        c.t += @divTrunc(@as(i96, @intCast(n)) * quantum, 2);
    }
};

const Late = struct { id: api.Id, request: *const api.Request, on_len: usize, at: usize };

/// One run of the host: the requests, the share, the fake's pass rule (resume 4 rows, pass 8) and its log.
const Case = struct {
    engine: api.Engine = undefined,
    clock: StepClock = .{},
    share: Share = .{},
    fill: api.fill_order.Policy = .{}, // the prompt-step order (off: the rotation)
    late: []const Late = &.{},
    fired: [4]bool = @splat(false),
    seen: [4]usize = @splat(0), // steps begun by the request of prompt_lens[i]
    prompt_lens: [4]usize = @splat(0),
    done: std.atomic.Value(u32) = .init(0),
    want: u32 = 0,
    beside_rows: u32 = 0, // Info.beside_rows: the rows of a step beside a decoder (0: the resume rows, 4)
    filling_pass: bool = false, // the fake's pass rule keeps whole passes where TF_DS_PASS_FILLING does
    long_marked: usize = 0, // the 64-row prompt's steps begun with the step marked to end on its device, and not
    long_unmarked: usize = 0,
    other_marked: usize = 0, // any other prompt's marked steps
    pair_steps: bool = false, // the fake's long-prompt steps of a whole pass (8 rows) report themselves as a pair (Stream.step_pair)

    fn hook(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
        const x: *Case = @ptrCast(@alignCast(ctx));
        if (s.prompt_len == 64) {
            if (s.settle_step) x.long_marked += 1 else x.long_unmarked += 1;
        } else if (s.settle_step) x.other_marked += 1;
        const begin = chunk * 4;
        const cap = lanes.stream.passRows(s.hostFacts(), begin, 0, 8, 4, x.filling_pass);
        s.step_pair = x.pair_steps and s.prompt_len == 64 and cap == 8;
        x.clock.rows(lanes.fake.Fake.passLen(begin, s.prompt_len, cap, 4));
        for (x.late, 0..) |l, k| {
            if (x.fired[k] or s.prompt_len != l.on_len) continue;
            x.seen[k] += 1;
            if (x.seen[k] <= l.at) continue;
            x.fired[k] = true;
            x.engine.submit(l.id, l.request, .{ .ctx = x, .event = event }) catch unreachable;
        }
    }

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const x: *Case = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .finished => _ = x.done.fetchAdd(1, .release),
            else => {},
        }
    }

    fn wait(x: *Case) !void {
        for (0..5000) |_| {
            if (x.done.load(.acquire) == x.want) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.RequestsDidNotFinish;
    }

    fn run(x: *Case, target: *lanes.fake.Fake, first: []const *const api.Request, n_lanes: u32) !void {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        target.pass_rule = .{ .resume_rows = 4, .pass_rows = 8, .filling_pass = x.filling_pass };
        target.prefill_hook = hook;
        target.prefill_hook_ctx = x;
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
        defer core.deinit();
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = n_lanes, .resume_rows = 4, .beside_rows = x.beside_rows, .fill = x.fill });
        host.share = x.share;
        host.share_clock = .{ .ctx = &x.clock, .now = StepClock.now };
        x.engine = host.engine();
        x.want = @intCast(first.len + x.late.len);
        for (first, 1..) |r, id| try x.engine.submit(id, r, .{ .ctx = x, .event = event });
        try host.start();
        defer host.stop();
        try x.wait();
    }
};

/// The decode calls the fake served between each two consecutive steps of the prompt of `len` rows.
fn roundsBetween(steps: []const lanes.fake.Fake.StepSeen, len: usize, out: *std.ArrayList(u64)) !void {
    var last: ?u64 = null;
    for (steps) |st| {
        if (st.prompt_len != len) continue;
        if (last) |l| try out.append(gpa, st.decodes - l);
        last = st.decodes;
    }
}

const decoder: [4]u32 = @splat(5);
const long: [64]u32 = @splat(3);
const chat: [6]u32 = @splat(7);

test "unset share: one round after every long prompt step, and the share's clock is never read" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var case: Case = .{};
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try case.run(&target, &.{ &ra, &rb }, 2);
    var gaps: std.ArrayList(u64) = .empty;
    defer gaps.deinit(gpa);
    try roundsBetween(target.steps.items, 64, &gaps);
    try std.testing.expectEqual(@as(usize, 15), gaps.items.len); // 16 steps of 4 rows
    for (gaps.items) |g| try std.testing.expectEqual(gaps.items[0], g);
    try std.testing.expect(gaps.items[0] > 0);
    try std.testing.expectEqual(@as(usize, 0), case.clock.reads);
}

test "a share of 0.5: a step of three quanta is followed by three rounds, then the next step" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var one: Case = .{};
    var t1: lanes.fake.Fake = .{ .gpa = gpa };
    defer t1.deinit();
    try one.run(&t1, &.{ &ra, &rb }, 2);
    var base: std.ArrayList(u64) = .empty;
    defer base.deinit(gpa);
    try roundsBetween(t1.steps.items, 64, &base);
    const per_round = base.items[0]; // the fake's decode calls in one round

    var case: Case = .{ .share = .{ .p = 0.5, .short = 8 } };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try case.run(&target, &.{ &ra, &rb }, 2);
    var gaps: std.ArrayList(u64) = .empty;
    defer gaps.deinit(gpa);
    try roundsBetween(target.steps.items, 64, &gaps);
    try std.testing.expectEqual(@as(usize, 15), gaps.items.len);
    // a 4-row step reads (1 + 4 / 2) = 3 quanta, a round 1: 3 x (1 - 0.5) / 0.5 = 3 quanta owed = 3 rounds
    for (gaps.items) |g| try std.testing.expectEqual(3 * per_round, g);
}

test "a share of 0.25 owes three times a step: nine rounds after a three-quanta step" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var one: Case = .{};
    var t1: lanes.fake.Fake = .{ .gpa = gpa };
    defer t1.deinit();
    try one.run(&t1, &.{ &ra, &rb }, 2);
    var base: std.ArrayList(u64) = .empty;
    defer base.deinit(gpa);
    try roundsBetween(t1.steps.items, 64, &base);

    var case: Case = .{ .share = .{ .p = 0.25, .short = 8 } };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try case.run(&target, &.{ &ra, &rb }, 2);
    var gaps: std.ArrayList(u64) = .empty;
    defer gaps.deinit(gpa);
    try roundsBetween(target.steps.items, 64, &gaps);
    for (gaps.items) |g| try std.testing.expectEqual(9 * base.items[0], g);
}

test "a short prompt goes before a long prompt's next step, whatever is owed" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    const rc: api.Request = .{ .prompt = &chat, .max_tokens = 4, .drafts = false };
    var one: Case = .{};
    var t1: lanes.fake.Fake = .{ .gpa = gpa };
    defer t1.deinit();
    try one.run(&t1, &.{ &ra, &rb }, 3);
    var base: std.ArrayList(u64) = .empty;
    defer base.deinit(gpa);
    try roundsBetween(t1.steps.items, 64, &base);

    // the chat arrives at the long prompt's second step, with nine rounds owed to the decoder
    var case: Case = .{ .share = .{ .p = 0.25, .short = 8 }, .late = &.{.{ .id = 3, .request = &rc, .on_len = 64, .at = 2 }} };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try case.run(&target, &.{ &ra, &rb }, 3);
    var at: ?usize = null;
    var before: ?u64 = null;
    for (target.steps.items, 0..) |st, i| {
        if (st.prompt_len == 6 and at == null) at = i;
        if (st.prompt_len == 64 and at == null) before = st.decodes;
    }
    // the chat's first step: no more than the one round after the long step that was running, not the nine owed
    try std.testing.expect(at != null);
    try std.testing.expect(target.steps.items[at.?].decodes - before.? <= base.items[0]);
    // and the long prompt still pays: the gaps after it are the nine rounds again
    var gaps: std.ArrayList(u64) = .empty;
    defer gaps.deinit(gpa);
    try roundsBetween(target.steps.items, 64, &gaps);
    try std.testing.expectEqual(@as(u64, 9) * base.items[0], gaps.items[gaps.items.len - 1]);
}

test "the prompt order picks among the short prompts while decode time is owed: the fewest rows first, and the long prompt still waits" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    const c6: api.Request = .{ .prompt = &chat, .max_tokens = 4, .drafts = false };
    const c5: api.Request = .{ .prompt = chat[0..5], .max_tokens = 4, .drafts = false };
    // both chats arrive at the long prompt's second step, with rounds owed to the decoder
    const late: []const Late = &.{
        .{ .id = 3, .request = &c6, .on_len = 64, .at = 2 },
        .{ .id = 4, .request = &c5, .on_len = 64, .at = 2 },
    };
    // the rotation (no order): the chat that arrived first goes first
    var rot: Case = .{ .share = .{ .p = 0.25, .short = 8 }, .late = late };
    var t1: lanes.fake.Fake = .{ .gpa = gpa };
    defer t1.deinit();
    try rot.run(&t1, &.{ &ra, &rb }, 4);
    try std.testing.expect(firstStepOf(t1.steps.items, 6).? < firstStepOf(t1.steps.items, 5).?);
    // the serial order: the chat with fewer rows goes first, beside the same share
    var ord: Case = .{ .share = .{ .p = 0.25, .short = 8 }, .fill = api.fill_order.parse("serial", null, null).policy, .late = late };
    var t2: lanes.fake.Fake = .{ .gpa = gpa };
    defer t2.deinit();
    try ord.run(&t2, &.{ &ra, &rb }, 4);
    try std.testing.expect(firstStepOf(t2.steps.items, 5).? < firstStepOf(t2.steps.items, 6).?);
    // neither chat waited for the debt (no more rounds before its first step than the one after the long step running),
    // and the long prompt still pays: the nine rounds again after its steps
    var base: std.ArrayList(u64) = .empty;
    defer base.deinit(gpa);
    try roundsBetween(t1.steps.items, 64, &base);
    var gaps: std.ArrayList(u64) = .empty;
    defer gaps.deinit(gpa);
    try roundsBetween(t2.steps.items, 64, &gaps);
    try std.testing.expect(gaps.items.len > 0);
    try std.testing.expectEqual(base.items[base.items.len - 1], gaps.items[gaps.items.len - 1]);
    // outputs of a prompt step are the same in both runs: the same steps for the long prompt
    try std.testing.expectEqual(stepsOf(t1.steps.items, 64), stepsOf(t2.steps.items, 64));
}

fn firstStepOf(steps: []const lanes.fake.Fake.StepSeen, len: usize) ?usize {
    for (steps, 0..) |st, i| if (st.prompt_len == len) return i;
    return null;
}

fn stepsOf(steps: []const lanes.fake.Fake.StepSeen, len: usize) usize {
    var n: usize = 0;
    for (steps) |st| n += @intFromBool(st.prompt_len == len);
    return n;
}

test "debt owed to a decoder that ended is not owed to anyone: the prompt's steps run on" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 2, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var case: Case = .{ .share = .{ .p = 0.01, .short = 8 } };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try case.run(&target, &.{ &ra, &rb }, 2); // (a hang here is the debt kept: RequestsDidNotFinish)
    var rows: usize = 0;
    for (target.steps.items) |st| rows += if (st.prompt_len == 64) st.rows else 0;
    try std.testing.expectEqual(@as(usize, 64), rows); // every row of it went in
}

test "a long prompt's step owed to decoders is marked to end on its device; a share of 1, or nothing decoding, marks none" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var plain: Case = .{ .share = .{ .short = 8 } }; // p = 1
    var t1: lanes.fake.Fake = .{ .gpa = gpa };
    defer t1.deinit();
    try plain.run(&t1, &.{ &ra, &rb }, 2);
    try std.testing.expectEqual(@as(usize, 0), plain.long_marked);
    try std.testing.expect(plain.long_unmarked > 0);

    var shared: Case = .{ .share = .{ .p = 0.5, .short = 8 } };
    var t2: lanes.fake.Fake = .{ .gpa = gpa };
    defer t2.deinit();
    try shared.run(&t2, &.{ &ra, &rb }, 2);
    try std.testing.expectEqual(@as(usize, 16), shared.long_marked); // every step of it, beside the decoder
    try std.testing.expectEqual(@as(usize, 0), shared.long_unmarked);
    try std.testing.expectEqual(@as(usize, 0), shared.other_marked); // the decoder's own prompt, with nothing decoding

    // alone, with a share set: nothing decodes, nothing is owed, nothing is marked
    var alone: Case = .{ .share = .{ .p = 0.5, .short = 8 } };
    var t3: lanes.fake.Fake = .{ .gpa = gpa };
    defer t3.deinit();
    try alone.run(&t3, &.{&rb}, 1);
    try std.testing.expectEqual(@as(usize, 0), alone.long_marked);
}

/// The rows of the long prompt's steps begun beside the decoder, each with whether a request waited then.
fn besideRows(steps: []const lanes.fake.Fake.StepSeen, out: *std.ArrayList(usize)) !void {
    for (steps) |st| if (st.prompt_len == 64 and st.facts.?.decoding and !st.facts.?.waiting) try out.append(gpa, st.rows);
}

test "unset beside rows: a step beside a decoder is the resume rows, with or without TF_DS_PASS_FILLING's rule" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    for ([_]bool{ false, true }) |filling_pass| {
        var case: Case = .{ .filling_pass = filling_pass };
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        try case.run(&target, &.{ &ra, &rb }, 2);
        var rows: std.ArrayList(usize) = .empty;
        defer rows.deinit(gpa);
        try besideRows(target.steps.items, &rows);
        try std.testing.expectEqual(@as(usize, 16), rows.items.len);
        for (rows.items) |r| try std.testing.expectEqual(@as(usize, 4), r);
    }
}

test "beside rows of 8: a step beside a decoder is 8 rows, TF_DS_PASS_FILLING's rule included, and a waiting request cuts the next one to 4" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    const rc: api.Request = .{ .prompt = &chat, .max_tokens = 4, .drafts = false };
    for ([_]bool{ false, true }) |filling_pass| {
        var case: Case = .{ .filling_pass = filling_pass, .beside_rows = 8 };
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        try case.run(&target, &.{ &ra, &rb }, 2);
        var rows: std.ArrayList(usize) = .empty;
        defer rows.deinit(gpa);
        try besideRows(target.steps.items, &rows);
        try std.testing.expectEqual(@as(usize, 8), rows.items.len); // 64 rows in steps of 8
        for (rows.items) |r| try std.testing.expectEqual(@as(usize, 8), r);

        // a request arriving at the long prompt's second step waits, and the step begun while it waits is the resume rows
        var late: Case = .{ .filling_pass = filling_pass, .beside_rows = 8, .late = &.{.{ .id = 3, .request = &rc, .on_len = 64, .at = 1 }} };
        var t2: lanes.fake.Fake = .{ .gpa = gpa };
        defer t2.deinit();
        try late.run(&t2, &.{ &ra, &rb }, 3);
        var cut: usize = 0;
        var whole: usize = 0;
        for (t2.steps.items) |st| {
            if (st.prompt_len != 64 or !st.facts.?.decoding) continue;
            if (st.facts.?.waiting) {
                cut += 1;
                try std.testing.expect(st.rows <= 4);
            } else if (st.rows == 8) whole += 1;
        }
        try std.testing.expect(cut >= 1);
        try std.testing.expect(whole >= 4);
    }
}

test "a share of 0.5 after a step of 8 rows: five rounds, from the step's five quanta" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var one: Case = .{ .beside_rows = 8 };
    var t1: lanes.fake.Fake = .{ .gpa = gpa };
    defer t1.deinit();
    try one.run(&t1, &.{ &ra, &rb }, 2);
    var base: std.ArrayList(u64) = .empty;
    defer base.deinit(gpa);
    try roundsBetween(t1.steps.items, 64, &base);
    for (base.items) |g| try std.testing.expectEqual(base.items[0], g); // unset share: one round a step at any step size

    var case: Case = .{ .beside_rows = 8, .share = .{ .p = 0.5, .short = 8 } };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try case.run(&target, &.{ &ra, &rb }, 2);
    var gaps: std.ArrayList(u64) = .empty;
    defer gaps.deinit(gpa);
    try roundsBetween(target.steps.items, 64, &gaps);
    // 8 steps of 8 rows read (1 + 8 / 2) = 5 quanta each, a round 1: the rounds that repay a step. The last step before the
    // end has no short remainder (a long prompt's last rows stay long), so every gap is five rounds.
    try std.testing.expectEqual(@as(usize, 7), gaps.items.len);
    for (gaps.items) |g| try std.testing.expectEqual(5 * base.items[0], g);
}

test "TF_DS_ROUNDS_BESIDE=3: three rounds follow every long prompt step, whatever the step took, and the share's clock is never read" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var one: Case = .{};
    var t1: lanes.fake.Fake = .{ .gpa = gpa };
    defer t1.deinit();
    try one.run(&t1, &.{ &ra, &rb }, 2);
    var base: std.ArrayList(u64) = .empty;
    defer base.deinit(gpa);
    try roundsBetween(t1.steps.items, 64, &base);
    for ([_]u32{ 3, 1, 5 }) |n| for ([_]u32{ 0, 8 }) |beside| {
        var case: Case = .{ .share = .{ .rounds = n, .short = 8 }, .beside_rows = beside };
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        try case.run(&target, &.{ &ra, &rb }, 2);
        var gaps: std.ArrayList(u64) = .empty;
        defer gaps.deinit(gpa);
        try roundsBetween(target.steps.items, 64, &gaps);
        try std.testing.expect(gaps.items.len >= 7);
        for (gaps.items) |g| try std.testing.expectEqual(@as(u64, n) * base.items[0], g);
        try std.testing.expectEqual(@as(usize, 0), case.clock.reads);
        try std.testing.expectEqual(@as(usize, 0), case.long_marked); // no time is billed: no step ends on the device for it
    };
}

test "TF_DS_ROUNDS_BESIDE=2: a step that ran as a stagger pair owes twice the rounds, a single pass the plain count" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var one: Case = .{};
    var t1: lanes.fake.Fake = .{ .gpa = gpa };
    defer t1.deinit();
    try one.run(&t1, &.{ &ra, &rb }, 2);
    var base: std.ArrayList(u64) = .empty;
    defer base.deinit(gpa);
    try roundsBetween(t1.steps.items, 64, &base);
    var case: Case = .{ .share = .{ .rounds = 2, .short = 8 }, .beside_rows = 8, .pair_steps = true };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try case.run(&target, &.{ &ra, &rb }, 2);
    var gaps: std.ArrayList(u64) = .empty;
    defer gaps.deinit(gpa);
    try roundsBetween(target.steps.items, 64, &gaps);
    try std.testing.expect(gaps.items.len >= 6);
    for (gaps.items) |g| try std.testing.expectEqual(@as(u64, 4) * base.items[0], g);
}

test "TF_DS_ROUNDS_BESIDE: a short prompt goes before the owed rounds and does not add to them; a prompt alone is not slowed" {
    const ra: api.Request = .{ .prompt = &decoder, .max_tokens = 400, .drafts = false };
    const rb: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    const rc: api.Request = .{ .prompt = &chat, .max_tokens = 4, .drafts = false };
    var one: Case = .{};
    var t1: lanes.fake.Fake = .{ .gpa = gpa };
    defer t1.deinit();
    try one.run(&t1, &.{ &ra, &rb }, 3);
    var base: std.ArrayList(u64) = .empty;
    defer base.deinit(gpa);
    try roundsBetween(t1.steps.items, 64, &base);
    var case: Case = .{ .share = .{ .rounds = 9, .short = 8 }, .late = &.{.{ .id = 3, .request = &rc, .on_len = 64, .at = 2 }} };
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    try case.run(&target, &.{ &ra, &rb }, 3);
    var at: ?usize = null;
    var before: ?u64 = null;
    for (target.steps.items, 0..) |st, i| {
        if (st.prompt_len == 6 and at == null) at = i;
        if (st.prompt_len == 64 and at == null) before = st.decodes;
    }
    try std.testing.expect(at != null);
    try std.testing.expect(target.steps.items[at.?].decodes - before.? <= base.items[0]);
    var gaps: std.ArrayList(u64) = .empty;
    defer gaps.deinit(gpa);
    try roundsBetween(target.steps.items, 64, &gaps);
    try std.testing.expectEqual(@as(u64, 9) * base.items[0], gaps.items[gaps.items.len - 1]);
    // nothing decodes: the rounds owed to no one are not run, the prompt steps back to back
    var alone: Case = .{ .share = .{ .rounds = 9, .short = 8 } };
    var t3: lanes.fake.Fake = .{ .gpa = gpa };
    defer t3.deinit();
    try alone.run(&t3, &.{&rb}, 2);
    var g3: std.ArrayList(u64) = .empty;
    defer g3.deinit(gpa);
    try roundsBetween(t3.steps.items, 64, &g3);
    for (g3.items) |g| try std.testing.expectEqual(@as(u64, 0), g);
}

test "TF_DS_ROUNDS_BESIDE with a share below 1 refuses the host's start, naming both" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .resume_rows = 4 });
    host.share = .{ .p = 0.5, .rounds = 2 };
    try std.testing.expectError(error.RoundsBesideWithShare, host.start());
    try std.testing.expect(std.mem.indexOf(u8, Share.refusal(host.share).?, "TF_DS_ROUNDS_BESIDE") != null);
    try std.testing.expect(std.mem.indexOf(u8, Share.refusal(host.share).?, "TENSORFOLD_PREFILL_SHARE") != null);
    try std.testing.expect(Share.refusal(.{ .p = 1, .rounds = 2 }) == null);
    try std.testing.expect(Share.refusal(.{ .p = 0.5 }) == null);
}

test "the start line names the rounds and the step size" {
    var buf: [640]u8 = undefined;
    const line = @import("lane_host_share.zig").startLine(&buf, 1, 4096, 5, 2048);
    try std.testing.expect(std.mem.indexOf(u8, line, "TF_DS_ROUNDS_BESIDE=5") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "up to 4096 rows") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "5 Token Generation rounds") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "10 after a stagger pair") != null);
}

test "TF_DS_ROUNDS_BESIDE's text: whole numbers from 1; unset is none; anything else is no setting" {
    const roundsText = @import("lane_host_share.zig").roundsFromText;
    try std.testing.expectEqual(@as(?u32, 0), roundsText(null));
    for ([_][]const u8{ "1", "2", "5", "50", "007" }) |t| try std.testing.expect(roundsText(t).? >= 1);
    try std.testing.expectEqual(@as(?u32, 50), roundsText("50"));
    for ([_][]const u8{ "", "0", "-1", "+2", "2.0", "1.5", " 2", "2 ", "two", "4294967296", "1_0", "0x10" }) |t| try std.testing.expectEqual(@as(?u32, null), roundsText(t));
}

test "the host takes a live rounds-beside at its next turn, falls back to the start's on unset, and refuses a count beside a share" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .resume_rows = 4, .rounds_beside = 3 });
    const live = api.live_config;
    var blk: live.Block = .{};
    host.live = &blk;
    host.liveTurn();
    try std.testing.expectEqual(@as(u32, 3), host.share.rounds); // nothing published: the start's value stays
    blk.publish(.rounds_beside, 6, 1);
    host.share.due = 4;
    host.liveTurn();
    try std.testing.expectEqual(@as(u32, 6), host.share.rounds);
    try std.testing.expectEqual(@as(u32, 0), host.share.due); // what was owed at the old count is forgiven
    try std.testing.expectEqual(@as(u64, 1), blk.active_version.load(.acquire));
    blk.publish(.rounds_beside, live.unset, 2);
    host.liveTurn();
    try std.testing.expectEqual(@as(u32, 3), host.share.rounds); // unset: the start's own
    // a count beside a share below 1 is two controls: the host keeps what it had, and the version is still taken (the API refused it first)
    host.share.p = 0.5;
    host.share.rounds = 0;
    blk.publish(.rounds_beside, 4, 3);
    host.liveTurn();
    try std.testing.expectEqual(@as(u32, 0), host.share.rounds);
    // a value no count can be (above a u32) is not applied
    host.share.p = 1;
    blk.publish(.rounds_beside, 1 << 40, 4);
    host.liveTurn();
    try std.testing.expectEqual(@as(u32, 0), host.share.rounds);
}

var cfg_seen: [8]api.live_config.CfgItem = undefined;
var cfg_versions: [8]u64 = undefined;
var cfg_count: usize = 0;

fn recordCfg(_: *anyopaque, version: u64, items: []const api.live_config.CfgItem) anyerror!void {
    for (items) |it| {
        cfg_seen[cfg_count] = it;
        cfg_versions[cfg_count] = version;
        cfg_count += 1;
    }
}

test "the host sends a cfg frame for a slot of class A only when its value changed, once, with the start's value for unset, and never for a slot of class H" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var be = target.chunkedBackend();
    var vt = be.vtable.*;
    vt.live_cfg = recordCfg;
    be.vtable = &vt;
    var core = lanes.Engine.init(gpa, &cfg, be, clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .resume_rows = 4, .rounds_beside = 3 });
    const live = api.live_config;
    var blk: live.Block = .{};
    host.live = &blk;
    cfg_count = 0;
    const ka = @backingInt(live.Slot.keep_ahead);
    // nothing published: no frame; a class H slot published: no frame
    host.liveTurn();
    blk.publish(.rounds_beside, 6, 1);
    host.liveTurn();
    try std.testing.expectEqual(@as(usize, 0), cfg_count);
    // a class A slot: one item, at the version taken
    blk.publish(.keep_ahead, 1, 2);
    host.liveTurn();
    try std.testing.expectEqual(@as(usize, 1), cfg_count);
    try std.testing.expectEqual(@as(u8, ka), cfg_seen[0].slot);
    try std.testing.expectEqual(@as(u32, 1), cfg_seen[0].value);
    try std.testing.expectEqual(@as(u64, 2), cfg_versions[0]);
    // the same value again (a new version, nothing changed): no frame
    blk.publish(.keep_ahead, 1, 3);
    host.liveTurn();
    try std.testing.expectEqual(@as(usize, 1), cfg_count);
    // a change of a class H slot beside it sends nothing more; unset travels as the marker the ranks read as the start's
    blk.publish(.rounds_beside, 5, 4);
    host.liveTurn();
    try std.testing.expectEqual(@as(usize, 1), cfg_count);
    blk.publish(.keep_ahead, live.unset, 5);
    host.liveTurn();
    try std.testing.expectEqual(@as(usize, 2), cfg_count);
    try std.testing.expectEqual(live.cfg_unset, cfg_seen[1].value);
    // a value no u32 holds is not sent
    blk.publish(.keep_ahead, 1 << 40, 6);
    host.liveTurn();
    try std.testing.expectEqual(@as(usize, 2), cfg_count);
}
