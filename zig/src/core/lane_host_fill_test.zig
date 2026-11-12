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
            }
        }
    };
    var order: Order = .{};
    const short = [_]u32{ 2, 7, 1, 8 };
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const background: Request = .{ .prompt = &short, .max_tokens = 4, .background = true };
    const foreground: Request = .{ .prompt = &long, .max_tokens = 4 };
    const e = host.engine();
    // the background prompt's first chunk sends the foreground request: it is admitted while the other fills
    const Arrive = struct {
        engine: api.Engine,
        request: *const Request,
        order: *Order,
        sent: bool = false,
        fn call(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (a.sent) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.order, .event = Order.event }) catch {};
        }
    };
    var arrive: Arrive = .{ .engine = e, .request = &foreground, .order = &order };
    target.prefill_hook = Arrive.call;
    target.prefill_hook_ctx = &arrive;
    try e.submit(1, &background, .{ .ctx = &order, .event = Order.event });
    try host.start();
    defer host.stop();
    while (true) {
        order.mutex.lockUncancelable(std.testing.io);
        const d = order.done;
        order.mutex.unlock(std.testing.io);
        if (d == 2) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
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
