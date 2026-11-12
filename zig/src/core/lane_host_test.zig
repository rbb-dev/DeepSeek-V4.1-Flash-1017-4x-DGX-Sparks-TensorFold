//! Host memory reporting, isolated request refusal and the pool-full retry through LaneHost.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Memory = api.Memory;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "a lane host reports its backend's memory counts, and none without them" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expect(host.engine().memory(false) == null);
    const Counts = struct {
        resets: u32 = 0,
        fn read(ctx: ?*anyopaque, reset_peak: bool) ?Memory {
            const c: *@This() = @ptrCast(@alignCast(ctx.?));
            if (reset_peak) c.resets += 1;
            return .{ .active = 5, .peak = if (reset_peak) 5 else 9 };
        }
    };
    var counts: Counts = .{};
    host.memory = .{ .ctx = &counts, .read = Counts.read };
    try std.testing.expectEqual(@as(u64, 9), host.engine().memory(false).?.peak);
    try std.testing.expectEqual(@as(u64, 5), host.engine().memory(true).?.peak);
    try std.testing.expectEqual(@as(u32, 1), counts.resets);
}

test "a lane host reports its backend's failed graph captures, and none without a count" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    var status: api.Status = .{};
    var streams: [4]u32 = undefined;
    host.engine().status(&status, &streams);
    try std.testing.expect(status.graph_fallbacks == null);
    var failed: std.atomic.Value(u64) = .init(0);
    host.graph_fallbacks = &failed;
    _ = failed.fetchAdd(3, .monotonic);
    host.engine().status(&status, &streams);
    try std.testing.expectEqual(@as(?u64, 3), status.graph_fallbacks);
}

test "a lane host reports its backend's decode rounds by the head's path, and none without a count" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    var status: api.Status = .{};
    var streams: [4]u32 = undefined;
    host.engine().status(&status, &streams);
    try std.testing.expect(status.head_rounds == null);
    var rounds: [6]std.atomic.Value(u64) = .{ .init(5), .init(6), .init(7), .init(8), .init(9), .init(4) };
    host.head_rounds = &rounds;
    _ = rounds[2].fetchAdd(3, .monotonic);
    host.engine().status(&status, &streams);
    try std.testing.expectEqual(@as(?[6]u64, .{ 5, 6, 10, 8, 9, 4 }), status.head_rounds);
}

test "a request the backend refuses fails alone, in the backend's words" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_sampled = true };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Words = struct {
        fn text(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return if (err == error.SamplingRefused) "send temperature 0" else null;
        }
    };
    host.explain = .{ .text = Words.text };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        message: []const u8 = "",
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                },
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
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var plain: Box = .{};
    var sampled: Box = .{};
    const greedy: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    const keyed: Request = .{ .prompt = &prompt, .max_tokens = 64, .sampling = .{ .seed = 3, .temperature = 0.7, .top_k = 5 } };
    const e = host.engine();
    try e.submit(1, &greedy, .{ .ctx = &plain, .event = Box.event });
    try e.submit(2, &keyed, .{ .ctx = &sampled, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, sampled.wait());
    try std.testing.expectEqualStrings("send temperature 0", sampled.message);
    try std.testing.expectEqual(Reason.length, plain.wait());
    try std.testing.expectEqual(@as(usize, 64), plain.tokens);
}

test "a request refused for room while another stream runs is tried again within a second" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_full = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        tokens: usize = 0,
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
        fn state(b: *@This()) struct { ?Reason, usize } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ b.done, b.tokens };
        }
        // the reason the request ended with, or null when it has not ended within `ms`
        fn waitFor(b: *@This(), ms: u32) ?Reason {
            for (0..ms) |_| {
                if (b.state()[0]) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
            return b.state()[0];
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var long: Box = .{};
    var short: Box = .{};
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &long, .event = Box.event });
    while (long.state()[1] == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    // the second request's prompt pass is refused once for room (error.ContextFull) while the first stream runs on:
    // it goes back to the queue and is tried again within a second, not only when that stream ends
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 8 }, .{ .ctx = &short, .event = Box.event });
    try std.testing.expectEqual(@as(?Reason, Reason.length), short.waitFor(5000));
    try std.testing.expectEqual(@as(usize, 8), short.state()[1]);
    try std.testing.expectEqual(@as(usize, 0), target.refuse_full);
    try std.testing.expect(long.state()[0] == null);
    e.cancel(1);
    try std.testing.expect(long.waitFor(5000) != null);
}

test "after a long prompt pass, a stream near its max_tokens ends before the next request's pass" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const Slow = struct {
        // each prompt pass takes 1.1 s, a long pass for the host (its whole pass runs inside the admission)
        fn hook(_: *anyopaque, _: *lanes.stream.Stream, _: usize) void {
            std.Io.sleep(std.testing.io, .fromMilliseconds(1100), .awake) catch {};
        }
    };
    var unused: u8 = 0;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_hook = Slow.hook, .prefill_hook_ctx = &unused };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Seq = struct {
        mutex: std.Io.Mutex = .init,
        n: u32 = 0,
        a_done: ?u32 = null,
        b_first: ?u32 = null,
        b_done: bool = false,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const q: *@This() = @ptrCast(@alignCast(ctx));
            q.mutex.lockUncancelable(std.testing.io);
            defer q.mutex.unlock(std.testing.io);
            q.n += 1;
            switch (e.*) {
                .tokens => if (id == 2 and q.b_first == null) {
                    q.b_first = q.n;
                },
                .finished => if (id == 1) {
                    q.a_done = q.n;
                } else {
                    q.b_done = true;
                },
                else => {},
            }
        }
        fn done(q: *@This()) bool {
            q.mutex.lockUncancelable(std.testing.io);
            defer q.mutex.unlock(std.testing.io);
            return q.a_done != null and q.b_done;
        }
    };
    var seq: Seq = .{};
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &seq, .event = Seq.event });
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &seq, .event = Seq.event });
    for (0..10000) |_| {
        if (seq.done()) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(seq.done());
    // the first request (64 tokens: within finish_left) ended before the second's first token
    try std.testing.expect(seq.a_done.? < seq.b_first.?);
}

/// A request's sink for the stop tests: its tokens and how it ended, read from the test thread.
const Ended = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    tokens: usize = 0,
    message: [96]u8 = undefined,
    message_len: usize = 0,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| {
                b.done = f.reason;
                b.message_len = @min(f.message.len, b.message.len);
                @memcpy(b.message[0..b.message_len], f.message[0..b.message_len]);
            },
            else => {},
        }
    }
    fn said(b: *@This()) []const u8 {
        return b.message[0..b.message_len];
    }
    fn state(b: *@This()) struct { ?Reason, usize } {
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        return .{ b.done, b.tokens };
    }
};

test "quiesce ends every request and leaves the engine's calls safe; stop then frees what is left, once" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var running: Ended = .{};
    var queued: Ended = .{};
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &running, .event = Ended.event });
    while (running.state()[1] == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 8 }, .{ .ctx = &queued, .event = Ended.event }); // one lane: it waits
    host.quiesce();
    // ended by the stop, not by their clients: a failure naming it, never a reply that looks finished
    try std.testing.expectEqual(@as(?Reason, .failed), running.state()[0]);
    try std.testing.expectEqualStrings(api.LaneHost.stop_message, running.said());
    try std.testing.expectEqual(@as(?Reason, .failed), queued.state()[0]);
    try std.testing.expectEqualStrings(api.LaneHost.stop_message, queued.said());
    try std.testing.expectEqual(@as(usize, 0), queued.state()[1]);
    // what a connection thread still calls after it: a submit is refused, a cancel and a status return normally
    var late: Ended = .{};
    try std.testing.expectError(error.Closed, e.submit(3, &.{ .prompt = &prompt, .max_tokens = 8 }, .{ .ctx = &late, .event = Ended.event }));
    try std.testing.expectEqual(@as(?Reason, null), late.state()[0]);
    e.cancel(1);
    e.cancel(99);
    var status: api.Status = .{};
    var streams: [4]u32 = undefined;
    e.status(&status, &streams);
    try std.testing.expectEqual(@as(u32, 0), status.running);
    try std.testing.expectEqual(@as(u32, 0), status.waiting);
    try std.testing.expectEqual(@as(usize, 0), status.streams);
    host.quiesce(); // a second one does nothing
    stopped = true;
    host.stop(); // its lists freed once: the testing allocator finds no leak and no second free
}

test "a quiesce in a long prompt pass cancels it at its next chunk, and no queued request's pass starts" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    // a long prompt's pass: 512 chunks of 10 ms (5 s whole)
    const Pass = struct {
        mutex: std.Io.Mutex = .init,
        reached: usize = 0, // chunks begun
        first: ?*const lanes.Stream = null,
        others: usize = 0, // chunks begun by any other stream's pass
        fn hook(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const p: *@This() = @ptrCast(@alignCast(ctx));
            p.mutex.lockUncancelable(std.testing.io);
            if (p.first == null) p.first = s;
            if (p.first == s) p.reached = chunk + 1 else p.others += 1;
            p.mutex.unlock(std.testing.io);
            std.Io.sleep(std.testing.io, .fromMilliseconds(10), .awake) catch {};
        }
        fn state(p: *@This()) struct { usize, usize } {
            p.mutex.lockUncancelable(std.testing.io);
            defer p.mutex.unlock(std.testing.io);
            return .{ p.reached, p.others };
        }
    };
    var pass: Pass = .{};
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 512, .prefill_hook = Pass.hook, .prefill_hook_ctx = &pass };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var long: Ended = .{};
    var waiting: [3]Ended = @splat(.{});
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &long, .event = Ended.event });
    for (&waiting, 2..) |*w, id| try e.submit(id, &.{ .prompt = &prompt, .max_tokens = 8 }, .{ .ctx = w, .event = Ended.event });
    while (pass.state()[0] < 3) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    host.quiesce();
    const took_ms = @divTrunc(std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - began, std.time.ns_per_ms);
    try std.testing.expectEqual(@as(?Reason, .failed), long.state()[0]); // its pass cut by the stop
    try std.testing.expectEqualStrings(api.LaneHost.stop_message, long.said());
    for (&waiting) |*w| try std.testing.expectEqual(@as(?Reason, .failed), w.state()[0]);
    const seen = pass.state();
    try std.testing.expect(seen[0] < 16); // stopped at a chunk end, not after the 512
    try std.testing.expectEqual(@as(usize, 0), seen[1]); // the queued requests ended without a pass
    try std.testing.expect(took_ms < 1000); // a chunk or two, not the pass's 5 s
}

test "a lane host fills prompts a chunk a round, serves their tokens, and cancels between chunks" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
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
    const prompts = [2][]const u32{ &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, &.{ 2, 7, 1, 8, 2, 8 } };
    var requests: [2]Request = undefined;
    var boxes: [2]Box = .{ .{}, .{} };
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    const e = host.engine();
    for (prompts, &requests, &boxes, 1..) |prompt, *r, *b, id| {
        r.* = .{ .prompt = prompt, .max_tokens = 24 };
        try e.submit(id, r, .{ .ctx = b, .event = Box.event });
    }
    for (prompts, &boxes) |prompt, *b| {
        try std.testing.expectEqual(Reason.length, b.wait());
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, prompt);
        for (b.tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
        try std.testing.expectEqual(@as(usize, 24), b.tokens.items.len);
    }
    try std.testing.expectEqual(@as(usize, 8), target.prefill_count);

    const CancelPrefill = struct {
        engine: Engine,
        id: Id,
        at: usize,

        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == c.at) c.engine.cancel(c.id);
        }
    };
    var chunked: Box = .{};
    defer chunked.tokens.deinit(gpa);
    var prefill_cancel = CancelPrefill{ .engine = e, .id = 3, .at = 1 };
    target.prefill_chunks = 10;
    target.prefill_count = 0;
    target.prefill_hook = CancelPrefill.call;
    target.prefill_hook_ctx = &prefill_cancel;
    try e.submit(3, &requests[0], .{ .ctx = &chunked, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, chunked.wait());
    try std.testing.expectEqual(@as(usize, 0), chunked.tokens.items.len);
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test "a lane host hands the backend the request's history and shared prefix lengths with its stream" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    const Seen = struct {
        history: u32 = 0,
        shared: [2]u32 = .{ 0, 0 },
        done: std.atomic.Value(bool) = .init(false),
        fn prefill(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            seen.history = s.history_len;
            @memcpy(seen.shared[0..s.shared_prefixes.len], s.shared_prefixes);
        }
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const seen: *@This() = @ptrCast(@alignCast(ctx));
            if (e.* == .finished) seen.done.store(true, .release);
        }
    };
    var seen: Seen = .{};
    target.prefill_hook = Seen.prefill;
    target.prefill_hook_ctx = &seen;
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const shared = [_]u32{ 2, 4 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 2, .history_len = 6, .shared_prefixes = &shared };
    try host.engine().submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
    while (!seen.done.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try std.testing.expectEqual(@as(u32, 6), seen.history);
    try std.testing.expectEqualSlices(u32, &shared, &seen.shared);
}

test "the prefill yield: a round that fails between two chunks fails the filling request and every admitted one with its error" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    // the fake, whose rounds fail once `broken` is set (a device fault in the round the prefill yield runs between two chunks)
    const Failing = struct {
        fake: lanes.fake.Fake,
        broken: std.atomic.Value(bool) = .init(false),
        chunks_b: usize = 0, // the filling request's chunks begun

        fn of(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        fn inner(ptr: *anyopaque) lanes.backend.Backend {
            return of(ptr).fake.backend();
        }
        fn backend(f: *@This()) lanes.backend.Backend {
            return .{ .ptr = f, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
        }
        fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            return inner(ptr).prefill(s);
        }
        fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            return inner(ptr).first(s, position);
        }
        fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
            if (of(ptr).broken.load(.acquire)) return error.RoundFault;
            return inner(ptr).queue(s, feed, position);
        }
        fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
            return inner(ptr).read(handle);
        }
        fn verify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
            if (of(ptr).broken.load(.acquire)) return error.RoundFault;
            return inner(ptr).verify(windows, out);
        }
        fn keep(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
            return inner(ptr).keep(windows, paths);
        }
        fn draft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) anyerror!void {
            if (of(ptr).broken.load(.acquire)) return error.RoundFault;
            return inner(ptr).draft(requests);
        }
        fn release(ptr: *anyopaque, s: *lanes.Stream) void {
            inner(ptr).release(s);
        }
        // the filling request's chunk hook: its first chunk breaks the device, so the round after it fails
        fn chunk(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const f = of(ctx);
            f.chunks_b += 1;
            f.broken.store(true, .release);
        }
    };
    var target: Failing = .{ .fake = .{ .gpa = gpa } };
    defer target.fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: usize = 0,
        prefilled: bool = false,
        done: ?Reason = null,
        message: [32]u8 = undefined,
        message_len: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => b.prefilled = true,
                .logprobs, .prompt_logprobs => {},
                .tokens => |t| b.tokens += t.len,
                .finished => |f| {
                    b.done = f.reason;
                    b.message_len = @min(f.message.len, b.message.len);
                    @memcpy(b.message[0..b.message_len], f.message[0..b.message_len]);
                },
            }
        }
        fn state(b: *@This()) struct { ?Reason, usize, bool } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ b.done, b.tokens, b.prefilled };
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                if (b.state()[0]) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var a: Box = .{};
    var b: Box = .{};
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &a, .event = Box.event });
    while (a.state()[1] == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    // B: a four-chunk pass with the prefill yield on; its first chunk breaks the device, so the round between it and the next fails
    target.fake.yield_rounds = true;
    target.fake.prefill_chunks = 4;
    target.fake.prefill_hook = Failing.chunk;
    target.fake.prefill_hook_ctx = &target;
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 8 }, .{ .ctx = &b, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, b.wait());
    try std.testing.expectEqual(Reason.failed, a.wait());
    try std.testing.expectEqualStrings("RoundFault", b.message[0..b.message_len]);
    try std.testing.expectEqualStrings("RoundFault", a.message[0..a.message_len]);
    try std.testing.expect(!b.state()[2]); // its pass ended at that boundary: never prefilled, no first token
    try std.testing.expectEqual(@as(usize, 1), target.chunks_b); // and ran no chunk after it
}

test {
    _ = @import("lane_host_fail_test.zig");
    _ = @import("lane_host_lone_test.zig");
    _ = @import("lane_host_admit_test.zig");
    _ = @import("lane_host_recall_test.zig");
    _ = @import("lane_host_fill_test.zig");
    _ = @import("lane_host_status_test.zig");
    _ = @import("lane_host_stale_test.zig");
    _ = @import("lane_host_reuse_test.zig");
    _ = @import("lane_host_cache_lifetime_test.zig");
    _ = @import("lane_host_borrow_test.zig");
    _ = @import("lane_host_warm_test.zig");
    _ = @import("lane_host_rows_test.zig");
}

test "a request refused for room while another stream runs is tried again within a second" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_full = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        tokens: usize = 0,
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
        fn state(b: *@This()) struct { ?Reason, usize } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ b.done, b.tokens };
        }
        // the reason the request ended with, or null when it has not ended within `ms`
        fn waitFor(b: *@This(), ms: u32) ?Reason {
            for (0..ms) |_| {
                if (b.state()[0]) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var long: Box = .{};
    var short: Box = .{};
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &long, .event = Box.event });
    while (long.state()[1] == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    // the second request's prompt pass is refused once for room (error.ContextFull) while the first stream runs on:
    // it goes back to the queue and is tried again within a second, not only when that stream ends
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 8 }, .{ .ctx = &short, .event = Box.event });
    try std.testing.expectEqual(@as(?Reason, Reason.length), short.waitFor(5000));
    try std.testing.expectEqual(@as(usize, 8), short.state()[1]);
    try std.testing.expectEqual(@as(usize, 0), target.refuse_full);
    try std.testing.expect(long.state()[0] == null);
    e.cancel(1);
    try std.testing.expect(long.waitFor(5000) != null);
}

test "after a long prompt pass, a stream near its max_tokens ends before the next request's pass" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const Slow = struct {
        // each prompt pass takes 1.1 s, a long pass for the host (its whole pass runs inside the admission)
        fn hook(_: *anyopaque, _: *lanes.stream.Stream, _: usize) void {
            std.Io.sleep(std.testing.io, .fromMilliseconds(1100), .awake) catch {};
        }
    };
    var unused: u8 = 0;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_hook = Slow.hook, .prefill_hook_ctx = &unused };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Seq = struct {
        mutex: std.Io.Mutex = .init,
        n: u32 = 0,
        a_done: ?u32 = null,
        b_first: ?u32 = null,
        b_done: bool = false,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const q: *@This() = @ptrCast(@alignCast(ctx));
            q.mutex.lockUncancelable(std.testing.io);
            defer q.mutex.unlock(std.testing.io);
            q.n += 1;
            switch (e.*) {
                .tokens => if (id == 2 and q.b_first == null) {
                    q.b_first = q.n;
                },
                .finished => if (id == 1) {
                    q.a_done = q.n;
                } else {
                    q.b_done = true;
                },
                else => {},
            }
        }
        fn done(q: *@This()) bool {
            q.mutex.lockUncancelable(std.testing.io);
            defer q.mutex.unlock(std.testing.io);
            return q.a_done != null and q.b_done;
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const e = host.engine();
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &seq, .event = Seq.event });
    try e.submit(2, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &seq, .event = Seq.event });
    for (0..10000) |_| {
        if (seq.done()) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(seq.done());
    // the first request (64 tokens: within finish_left) ended before the second's first token
    try std.testing.expect(seq.a_done.? < seq.b_first.?);
}
