//! LaneHost with a durable tier (wojo's GLM 0110's scheduler hooks): requests parked on restores, their service
//! between rounds and at idle, cancellation and shutdown, through the fake lane backend and an in-memory tier with a
//! landing: a restore is ready only once every rank holds it.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const lh = @import("lane_host.zig");
const ft = @import("prompt_cache_fake_tier.zig");
const LaneHost = lh.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const gpa = std.testing.allocator;
const io = std.testing.io;

pub const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,

    pub fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .logprobs, .prompt_logprobs => {},
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
        }
    }
    pub fn sink(b: *Box) api.Sink {
        return .{ .ctx = b, .event = event };
    }
    pub fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
    pub fn deinit(b: *Box) void {
        b.tokens.deinit(gpa);
    }
    pub fn finished(b: *Box) ?Reason {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.done;
    }
    pub fn count(b: *Box) usize {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.tokens.items.len;
    }
    /// Whether the request's pass ended (prefilled) or any token came.
    pub fn started(b: *Box) bool {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.cached != null or b.tokens.items.len > 0;
    }
    /// The reply equals the fake target's over `prompt` (any slip in the restored history changes it).
    pub fn exact(b: *Box, prompt: []const u32) !void {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, prompt);
        for (b.tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
    }
};

/// The fake backend, a prompt cache over its states, an in-memory tier and the host, at stable addresses.
pub const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    store: pc.Store,
    tier: ft.FakeTier,
    host: LaneHost,
    stopped: bool = false,

    pub fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    pub fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    pub fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }
    pub fn pack(_: *anyopaque, a: std.mem.Allocator, saved: pc.Saved) anyerror![]u8 {
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        return a.dupe(u8, std.mem.sliceAsBytes(kept.items));
    }
    pub fn unpack(_: *anyopaque, b: []const u8) anyerror!pc.Saved {
        const kept = try gpa.create(std.ArrayList(u32));
        kept.* = .empty;
        const n = b.len / 4;
        try kept.resize(gpa, n);
        @memcpy(std.mem.sliceAsBytes(kept.items), b[0 .. n * 4]);
        return kept;
    }

    pub fn create(sp: api.Spill) !*Rig {
        return createWith(sp, null);
    }

    /// `pre` runs on the host before its thread starts (the robustness tests' policies).
    pub fn createWith(sp: api.Spill, pre: ?*const fn (*Rig) void) !*Rig {
        const r = try gpa.create(Rig);
        errdefer gpa.destroy(r);
        r.* = .{ .cfg = undefined, .target = .{ .gpa = gpa }, .core = undefined, .store = undefined, .tier = .{ .gpa = gpa, .codec = .{ .ctx = r, .pack = pack, .unpack = unpack } }, .host = undefined };
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        r.store = pc.Store.init(gpa, .{ .ptr = &r.target, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        r.store.setTier(r.tier.tier(), 0);
        r.host = LaneHost.init(gpa, io, &r.core, .{ .lanes = 2 });
        r.host.cache = &r.store;
        r.host.configure(sp);
        if (pre) |f| f(r);
        try r.host.start();
        return r;
    }

    pub fn stop(r: *Rig) void {
        if (r.stopped) return;
        r.stopped = true;
        r.host.stop();
    }

    pub fn destroy(r: *Rig) void {
        r.stop();
        r.store.deinit();
        r.tier.deinit();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
        gpa.destroy(r);
    }

    /// A published point of `key[0..at]` whose state is those tokens (the fake backend's state is its history).
    pub fn point(r: *Rig, key: []const u32, at: u32) !u64 {
        return r.tier.put(key[0..at], at, std.mem.sliceAsBytes(key[0..at]), 0, 0, false);
    }

    pub fn status(r: *Rig) api.Status {
        var st: api.Status = .{};
        r.host.engine().status(&st, &.{});
        return st;
    }

    pub fn stats(r: *Rig) api.CacheStats {
        return r.status().cache.?;
    }

    /// Waits (5 s at most, else fails) until the host's state is: nothing deferred or loading, and exactly one request on
    /// the ready list whose read-ahead was tried (refused): it waits for a lane and asks the disk no more before its turn.
    pub fn untilReadyAfterAhead(r: *Rig) !void {
        for (0..5000) |_| {
            r.host.mutex.lockUncancelable(r.host.io);
            const ok = r.host.deferred.items.len == 0 and r.host.loading.items.len == 0 and
                r.host.ready.items.len == 1 and r.host.ready.items[0].read_ahead == .tried;
            r.host.mutex.unlock(r.host.io);
            if (ok) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        std.debug.print("timed out waiting for: one request ready after its refused read-ahead, none deferred or loading\n", .{});
        return error.Timeout;
    }

    /// Waits (5 s at most) until the published counters satisfy `ok`.
    pub fn until(r: *Rig, comptime ok: fn (api.CacheStats) bool) !void {
        for (0..5000) |_| {
            if (ok(r.stats())) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

pub const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
pub const t2 = t1 ++ [_]u32{ 7, 7, 5, 3, 8, 8 };

test "a turn whose state left memory waits parked for its restore, resumes it exactly, and writes its own keeps forked from it" {
    const r = try Rig.create(.{ .load_ms = 5, .flush_ms = 1000 });
    defer r.destroy();
    r.tier.polls = 3;
    const id = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.writes_enqueued == 1;
        }
    }.ok);
    const st = r.status();
    const s = st.cache.?;
    try std.testing.expectEqual(@as(u64, 1), s.restore_waits);
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 8), s.restored_tokens);
    try std.testing.expectEqual(@as(u64, 1), s.memory_hits);
    try std.testing.expectEqual(@as(u64, 8), s.cached_tokens);
    try std.testing.expectEqual(@as(u64, 8), s.computed_tokens);
    try std.testing.expectEqual(@as(?u64, 8), st.prompt_tokens_computed); // (the 16 less the 8 restored)
    try std.testing.expectEqual(@as(u64, 0), s.parked);
    try std.testing.expect(s.spill != null);
    r.stop();
    try std.testing.expectEqual(@as(u64, 1), r.tier.flushed); // the clean shutdown drained the tier
    const kept = for (r.tier.points.items) |p| {
        if (p.at == 14) break p;
    } else return error.TestUnexpectedResult;
    try std.testing.expectEqual(id, kept.parent.?); // its lineage forks the restored point at the resume point
    try std.testing.expectEqual(@as(u32, 8), kept.fork);
    try std.testing.expect(kept.group != 0); // a conversation of its own (the restored state is no turn's own)
    try std.testing.expectEqual(@as(usize, 1), kept.lineage.len); // the point carried none: a lineage of its own
}

test "two requests on one point share its restore; it stops once, when the last of them is cancelled" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, a.sink());
    try e.submit(2, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 2;
        }
    }.ok);
    var st: api.Status = .{};
    e.status(&st, &.{});
    try std.testing.expectEqual(@as(u32, 2), st.waiting);
    try std.testing.expectEqual(@as(u32, 0), st.running);
    try std.testing.expectEqual(@as(u64, 1), st.cache.?.restores_begun);
    try std.testing.expectEqual(@as(u64, 1), st.cache.?.restores_shared);
    try std.testing.expectEqual(@as(u64, 2), st.cache.?.parked);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(@as(u64, 1), r.stats().restores_active); // still wanted by the other
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, b.wait());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_cancelled == 1 and s.restores_active == 0 and s.parked == 0;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(@as(u64, 1), r.tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), r.tier.misuse);
    try std.testing.expectEqual(@as(usize, 0), r.tier.jobs.items.len);
}

test "a restore that fails on any rank prefills from the start, exactly" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.fail_next = error.CrcMismatch;
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_failed);
    try std.testing.expectEqual(@as(u64, 0), s.disk_hits_committed);
}

test "shutdown with requests parked on a restore that never ends: they end failed (the stop's, sent again by their clients), the restore stops once, the tier flushes" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, a.sink());
    try r.host.engine().submit(2, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 2;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(Reason.failed, a.wait());
    try std.testing.expectEqual(Reason.failed, b.wait());
    try std.testing.expectEqual(@as(u64, 1), r.tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), r.tier.misuse);
    try std.testing.expectEqual(@as(u64, 1), r.tier.flushed);
}

test "without a tier nothing is parked and status carries no cache counters" {
    const gpa_ = gpa;
    var cfg = try lanes.Config.init(gpa_, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa_);
    var target: lanes.fake.Fake = .{ .gpa = gpa_ };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa_, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(gpa_, .{ .ptr = &target, .vtable = &.{ .bytes = Rig.bytes, .save = Rig.save, .restore = Rig.restore, .drop = Rig.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa_, io, &core, .{ .lanes = 2 });
    host.cache = &store;
    host.configure(.{});
    try host.start();
    defer host.stop();
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 3, .history_len = 14 };
    try host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    var st: api.Status = .{};
    host.engine().status(&st, &.{});
    try std.testing.expect(st.cache == null);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), store.entries.items[0].group);
}

test "a request parked on a restore past TF_SPILL_LOAD_WAIT_MS is admitted without it, and the restore stops" {
    const r = try Rig.create(.{ .load_ms = 5, .load_wait_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    try b.exact(&t2);
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_wait_timeouts == 1 and s.restores_cancelled == 1 and s.restores_active == 0 and s.restoring_bytes == 0;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(@as(u64, 1), r.tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), r.tier.misuse);
}

test "restores are serviced between decode rounds while another stream decodes" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    const along = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var a: Box = .{};
    defer a.deinit();
    const areq: Request = .{ .prompt = &along, .max_tokens = 100000 };
    try e.submit(1, &areq, a.sink());
    while (a.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const p0 = @atomicLoad(u64, &r.tier.polled, .monotonic);
    const n0 = a.count();
    var grew = false;
    for (0..5000) |_| {
        if (a.count() > n0 + 32 and @atomicLoad(u64, &r.tier.polled, .monotonic) > p0 + 8) {
            grew = true;
            break;
        }
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(grew); // polled round after round while the core is never idle
    try std.testing.expect(a.finished() == null);
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
}

const busy1 = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
const busy2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5 };
const busy = [_]Request{ .{ .prompt = &busy1, .max_tokens = 100000 }, .{ .prompt = &busy2, .max_tokens = 100000 } };

/// Two requests (ids 101, 102) decoding until cancelled: every lane of the rig busy.
fn bothLanes(e: api.Engine, a1: *Box, a2: *Box) !void {
    try e.submit(101, &busy[0], a1.sink());
    try e.submit(102, &busy[1], a2.sink());
    while (a1.count() == 0 or a2.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
}

test "every lane busy: the next request in line reads its state ahead of its turn and resumes it when a lane frees" {
    var a1: Box = .{};
    defer a1.deinit();
    var a2: Box = .{};
    defer a2.deinit();
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    try bothLanes(e, &a1, &a2);
    _ = try r.point(&t2, 8);
    r.tier.gate.store(false, .release);
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &req, b.sink());
    // its restore begins while both lanes still decode, and it holds no lane
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1 and s.parked == 1;
        }
    }.ok);
    try std.testing.expect(a1.finished() == null and a2.finished() == null and !b.started());
    r.tier.gate.store(true, .release);
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.disk_hits_committed == 1 and s.parked == 0;
        }
    }.ok);
    try std.testing.expect(!b.started()); // read, landed, and still waiting for a lane
    e.cancel(101);
    try std.testing.expectEqual(Reason.cancelled, a1.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // its pass resumed the 8 tokens read ahead
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restore_waits); // read once: its turn found the state in memory
    try std.testing.expectEqual(@as(u64, 1), r.tier.begins);
    e.cancel(102);
    try std.testing.expectEqual(Reason.cancelled, a2.wait());
}

test "every lane busy: a read-ahead the tier refuses for capacity is not deferred; the request reads at its turn as before" {
    var a1: Box = .{};
    defer a1.deinit();
    var a2: Box = .{};
    defer a2.deinit();
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    try bothLanes(e, &a1, &a2);
    _ = try r.point(&t2, 8);
    r.tier.refuse = 1; // the read-ahead's begin: refused for capacity
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &req, b.sink());
    for (0..5000) |_| {
        if (@atomicLoad(u64, &r.tier.refused, .monotonic) == 1) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    } else return error.Timeout;
    std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {}; // rounds go by: it is asked once, not again and again
    var s = r.stats();
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
    try std.testing.expectEqual(@as(u64, 1), @atomicLoad(u64, &r.tier.begins, .monotonic));
    try std.testing.expect(!b.started());
    e.cancel(101);
    try std.testing.expectEqual(Reason.cancelled, a1.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // its turn read it
    try b.exact(&t2);
    s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred);
    e.cancel(102);
    try std.testing.expectEqual(Reason.cancelled, a2.wait());
}

test "every lane busy: a queued request with nothing on disk keeps its place, and the one behind it does not read ahead past it" {
    var a1: Box = .{};
    defer a1.deinit();
    var a2: Box = .{};
    defer a2.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    try bothLanes(e, &a1, &a2);
    _ = try r.point(&t2, 8);
    const fresh = [_]u32{ 9, 9, 8, 9, 9, 7, 9, 9, 6 };
    const creq: Request = .{ .prompt = &fresh, .max_tokens = 6 };
    try e.submit(3, &creq, c.sink());
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.times.lookup.count >= 1; // the first in line asked the disk ahead (nothing there)
        }
    }.ok);
    std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {}; // rounds go by
    try std.testing.expectEqual(@as(u64, 0), r.stats().restore_waits); // the second in line waits behind it, unread
    try std.testing.expect(!b.started() and !c.started());
    e.cancel(101);
    try std.testing.expectEqual(Reason.cancelled, a1.wait());
    try std.testing.expectEqual(Reason.length, c.wait()); // the freed lane went to the first in line
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    e.cancel(102);
    try std.testing.expectEqual(Reason.cancelled, a2.wait());
}

test "every lane busy: a read-ahead whose landing failed is read again at its turn" {
    var a1: Box = .{};
    defer a1.deinit();
    var a2: Box = .{};
    defer a2.deinit();
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    try bothLanes(e, &a1, &a2);
    _ = try r.point(&t2, 8);
    r.tier.land_fail = error.SpillLandFailed; // the read-ahead's landing (the point stays on disk)
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_failed == 1 and s.parked == 0;
        }
    }.ok);
    try std.testing.expect(!b.started());
    e.cancel(101);
    try std.testing.expectEqual(Reason.cancelled, a1.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // its turn read the point again
    try b.exact(&t2);
    try std.testing.expectEqual(@as(u64, 2), r.stats().restore_waits);
    e.cancel(102);
    try std.testing.expectEqual(Reason.cancelled, a2.wait());
}

// -- the parked wait and the prompt pass on status; the landing -------------------------------------------------------

test "a parked request's wait is not its prompt pass: no pass shows while it waits, and its pass resumes the restored tokens" {
    var b: Box = .{}; // (before the rig: a failed check stops the host before the box goes)
    defer b.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const parked = r.status();
    try std.testing.expect(parked.prompt_pass == null); // a disk restore is no prompt pass
    try std.testing.expectEqual(@as(u32, 1), parked.waiting);
    try std.testing.expectEqual(@as(u32, 0), parked.running);
    try std.testing.expectEqual(@as(?u64, 0), parked.prompt_tokens_computed);
    // its pass, seen at its first chunk: it resumed the restored 8 and computes the other 8
    const Look = struct {
        engine: api.Engine,
        seen: ?api.PromptPass = null,
        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk != 0) return;
            var st: api.Status = .{};
            l.engine.status(&st, &.{});
            l.seen = st.prompt_pass;
        }
    };
    var look: Look = .{ .engine = e };
    r.target.prefill_hook = Look.call;
    r.target.prefill_hook_ctx = &look;
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    const pass = look.seen.?;
    try std.testing.expectEqual(@as(u64, 8), pass.resumed);
    try std.testing.expectEqual(@as(u64, 8), pass.total);
    try std.testing.expectEqual(@as(?u64, 8), r.status().prompt_tokens_computed);
}

test "a parked request is admitted only after its restore's last landing poll; a stream decoding meanwhile keeps getting tokens" {
    var a: Box = .{}; // (before the rig: a failed check stops the host before the boxes go)
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    const along = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    try e.submit(1, &.{ .prompt = &along, .max_tokens = 100000 }, a.sink());
    while (a.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    // at each landing poll (the engine thread, inside a service): B neither prefilled nor running, A's tokens so far
    const Watch = struct {
        a: *Box,
        b: *Box,
        host: *LaneHost,
        n: usize = 0,
        a_tokens: [3]usize = @splat(0),
        b_started: [3]bool = @splat(true),
        running: [3]usize = @splat(9),
        fn at(ctx: *anyopaque) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (w.n < 3) {
                w.a_tokens[w.n] = w.a.count();
                w.b_started[w.n] = w.b.started();
                w.running[w.n] = w.host.admitted.items.len; // (the engine thread's own list)
            }
            w.n += 1;
        }
    };
    var watch: Watch = .{ .a = &a, .b = &b, .host = &r.host };
    r.tier.polls = 1; // its read: one poll; its landing: three more, the last of them ready
    r.tier.land_polls = 3;
    r.tier.land_hook = Watch.at;
    r.tier.land_ctx = &watch;
    _ = try r.point(&t2, 8);
    try e.submit(2, &.{ .prompt = &t2, .max_tokens = 6, .history_len = 14 }, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    try std.testing.expectEqual(@as(usize, 3), watch.n);
    try std.testing.expectEqual(@as(u64, 3), r.tier.land_polled);
    for (0..3) |k| {
        try std.testing.expect(!watch.b_started[k]); // not admitted before the landing's last poll
        try std.testing.expectEqual(@as(usize, 1), watch.running[k]); // only A holds a lane
    }
    try std.testing.expect(watch.a_tokens[0] < watch.a_tokens[1] and watch.a_tokens[1] < watch.a_tokens[2]); // a round between landing polls
    try std.testing.expectEqual(@as(u64, 1), r.stats().disk_hits_committed);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
}

test "a landing that cannot start: the request prefills from the start, exactly, and the point stays for the next" {
    var a: Box = .{}; // (before the rig: a failed check stops the host before the boxes go)
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.land_polls = 2;
    r.tier.land_fail = error.SpillLandRefused; // the first restore's landing: no rank's slot free (the ready vote)
    _ = try r.point(&t2, 8);
    // history_len 0: the pass keeps nothing in memory, so the second request can only resume from disk
    const req: Request = .{ .prompt = &t2, .max_tokens = 6 };
    try r.host.engine().submit(1, &req, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 0), a.cached);
    try a.exact(&t2);
    try std.testing.expectEqual(@as(u64, 1), r.stats().restores_failed);
    try std.testing.expectEqual(@as(usize, 1), r.tier.points.items.len); // not forgotten: a landing's failure is no damage
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // restored this time
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 2), s.restore_waits);
    try std.testing.expectEqual(@as(u64, 1), s.spill.?.slice()[0].landings_skipped);
    try std.testing.expectEqual(@as(u64, 1), s.spill.?.slice()[0].landings);
}

// -- the tier part between two prompt chunks --------------------------------------------------------------------------

test "with the prefill yield off, restores progress between another request's prompt chunks, then their request resumes" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const Watch = struct {
        tier: *ft.FakeTier,
        seen: [6]u64 = @splat(0),
        fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len == 12 and chunk < w.seen.len) w.seen[chunk] = w.tier.polled; // (the engine thread's)
        }
    };
    var watch: Watch = .{ .tier = &r.tier };
    r.target.prefill_chunks = 6; // (yield_rounds off: the prefill yield's round part never runs)
    r.target.prefill_hook = Watch.call;
    r.target.prefill_hook_ctx = &watch;
    const other = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var o: Box = .{};
    defer o.deinit();
    const oreq: Request = .{ .prompt = &other, .max_tokens = 1 };
    try e.submit(2, &oreq, o.sink());
    try std.testing.expectEqual(Reason.length, o.wait());
    try std.testing.expect(watch.seen[5] > watch.seen[0]); // polled inside the other's pass, between its chunks
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
}

/// A pass's chunks seen from its prefill hook (the engine thread): a decoding stream's tokens, the tier's polls and the
/// store's hits at each chunk's start; only the pass whose prompt has `len` tokens.
const Chunks = struct {
    host: *LaneHost,
    tier: *ft.FakeTier,
    a: *Box,
    len: usize,
    n: usize = 0,
    a_tokens: [8]usize = @splat(0),
    polled: [8]u64 = @splat(0),
    hits: [8]u64 = @splat(0),

    fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        if (s.prompt_len != w.len or chunk >= w.a_tokens.len) return;
        w.a_tokens[chunk] = w.a.count();
        w.polled[chunk] = w.tier.polled;
        w.hits[chunk] = w.host.cache.?.counts.hits;
        w.n = chunk + 1;
    }
};

/// A request decoding (A) and one parked on a restore that never ends in the test (its read, then 1000 landing polls):
/// the setting of the prefill-yield tests below. Returns once both stand.
fn decodingAndParked(r: *Rig, a: *Box, c: *Box) !void {
    const e = r.host.engine();
    const along = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    try e.submit(1, &.{ .prompt = &along, .max_tokens = 100000 }, a.sink());
    while (a.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    r.tier.polls = 1;
    r.tier.land_polls = 1000;
    _ = try r.point(&t2, 8);
    try e.submit(3, &.{ .prompt = &t2, .max_tokens = 6 }, c.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
}

test "with the prefill yield on and another stream live, each boundary runs the round part, then the tier part; with nothing pending the tier part never runs" {
    var a: Box = .{}; // (before the rig: a failed check stops the host before the boxes go)
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    var d: Box = .{};
    defer d.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    try decodingAndParked(r, &a, &c);
    // at each landing poll the tier part makes inside B's pass: B's chunk so far and A's tokens then
    const Turns = struct {
        host: *LaneHost,
        a: *Box,
        chunks: *Chunks,
        n: usize = 0,
        chunk: [8]usize = @splat(0),
        a_tokens: [8]usize = @splat(0),
        fn at(ctx: *anyopaque) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (w.host.filling_job == null or w.n >= w.chunk.len) return; // (the run loop's services: not B's pass)
            w.chunk[w.n] = w.chunks.n - 1;
            w.a_tokens[w.n] = w.a.count();
            w.n += 1;
        }
    };
    const prompt_b = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5 };
    var chunks: Chunks = .{ .host = &r.host, .tier = &r.tier, .a = &a, .len = prompt_b.len };
    var turns: Turns = .{ .host = &r.host, .a = &a, .chunks = &chunks };
    r.tier.land_hook = Turns.at;
    r.tier.land_ctx = &turns;
    r.target.yield_rounds = true;
    r.target.prefill_chunks = 4;
    r.target.prefill_hook = Chunks.call;
    r.target.prefill_hook_ctx = &chunks;
    const e = r.host.engine();
    try e.submit(2, &.{ .prompt = &prompt_b, .max_tokens = 1 }, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(usize, 4), chunks.n);
    try std.testing.expectEqual(@as(usize, 3), turns.n); // one service a boundary, none after the last chunk
    for (0..3) |k| {
        try std.testing.expectEqual(k, turns.chunk[k]); // the service after chunk k
        try std.testing.expect(turns.a_tokens[k] > chunks.a_tokens[k]); // after that boundary's round: A got tokens first
    }
    // C cancelled: nothing pending, so a pass with the prefill yield on runs its rounds and no service at all
    e.cancel(3);
    try std.testing.expectEqual(Reason.cancelled, c.wait());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.parked == 0 and s.restores_active == 0;
        }
    }.ok);
    const Pumps = struct {
        tier: *ft.FakeTier,
        len: usize,
        at: [4]u64 = @splat(0),
        fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len == w.len and chunk < w.at.len) w.at[chunk] = w.tier.pumped;
        }
    };
    var pumps: Pumps = .{ .tier = &r.tier, .len = prompt_b.len };
    r.target.prefill_hook = Pumps.call;
    r.target.prefill_hook_ctx = &pumps;
    const a0 = a.count();
    try e.submit(4, &.{ .prompt = &prompt_b, .max_tokens = 1 }, d.sink());
    try std.testing.expectEqual(Reason.length, d.wait());
    try std.testing.expectEqual(pumps.at[0], pumps.at[3]); // no service inside the pass
    try std.testing.expect(a.count() > a0);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
}

test "a picture prompt keeps the prefill yield and the tier part between its chunks" {
    var a: Box = .{}; // (before the rig: a failed check stops the host before the boxes go)
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    try decodingAndParked(r, &a, &c);
    // a picture at [2, 5): the placeholder (99) in the prompt, keyed by the picture's content
    const prompt_b = [_]u32{ 3, 1, 99, 99, 99, 5, 9, 2, 6, 5, 7, 7, 4, 4 };
    var key = prompt_b;
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("a picture", &d, .{});
    for (0..3) |j| key[2 + j] = pc.pictureId(&d, @intCast(j));
    const img = [_]api.LaneImage{.{ .at = 2, .tokens = 3, .bytes = "" }};
    const media = [_]pc.Media{.{ .start = 2, .len = 3, .digest = d }};
    var chunks: Chunks = .{ .host = &r.host, .tier = &r.tier, .a = &a, .len = prompt_b.len };
    r.target.yield_rounds = true;
    r.target.prefill_chunks = 4;
    r.target.prefill_hook = Chunks.call;
    r.target.prefill_hook_ctx = &chunks;
    const e = r.host.engine();
    try e.submit(2, &.{ .prompt = &prompt_b, .lane_images = &img, .cache_key = &key, .cache_media = &media, .max_tokens = 1 }, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(usize, 4), chunks.n);
    for (1..4) |k| {
        try std.testing.expect(chunks.a_tokens[k] > chunks.a_tokens[k - 1]); // the prefill yield's round at each boundary
        try std.testing.expect(chunks.polled[k] > chunks.polled[k - 1]); // and the tier part's service
    }
    e.cancel(1);
    e.cancel(3);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(Reason.cancelled, c.wait());
}

test "a resumed multi-chunk pass with tier work at every boundary reports its restore at the first boundary and keeps the restored prefix" {
    var a: Box = .{}; // (before the rig: a failed check stops the host before the boxes go)
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    var b1: Box = .{};
    defer b1.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    // B's conversation, turn 1: its state at 8 stays in memory
    const turn1 = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    try e.submit(5, &.{ .prompt = &turn1, .max_tokens = 2, .history_len = 8 }, b1.sink());
    try std.testing.expectEqual(Reason.length, b1.wait());
    try decodingAndParked(r, &a, &c);
    const turn2 = turn1 ++ [_]u32{ 2, 3, 4, 5 };
    var chunks: Chunks = .{ .host = &r.host, .tier = &r.tier, .a = &a, .len = turn2.len };
    r.target.yield_rounds = true;
    r.target.prefill_chunks = 4;
    r.target.prefill_hook = Chunks.call;
    r.target.prefill_hook_ctx = &chunks;
    try e.submit(2, &.{ .prompt = &turn2, .max_tokens = 6 }, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // the restored prefix, though reported at the first boundary
    try b.exact(&turn2);
    try std.testing.expectEqual(chunks.hits[0] + 1, chunks.hits[1]); // reported by the tier part after chunk 0
    for (1..4) |k| try std.testing.expect(chunks.polled[k] > chunks.polled[k - 1]); // tier work at every boundary
    e.cancel(1);
    e.cancel(3);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(Reason.cancelled, c.wait());
}

test "no service is refused across a prefill-yield pass with tier work at every boundary (the fence panics on one in tests)" {
    var a: Box = .{}; // (before the rig: a failed check stops the host before the boxes go)
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    try std.testing.expect(r.host.fence_panic); // (debug: a refused service would end the test)
    try decodingAndParked(r, &a, &c);
    const prompt_b = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5 };
    var chunks: Chunks = .{ .host = &r.host, .tier = &r.tier, .a = &a, .len = prompt_b.len };
    r.target.yield_rounds = true;
    r.target.prefill_chunks = 6;
    r.target.prefill_hook = Chunks.call;
    r.target.prefill_hook_ctx = &chunks;
    const e = r.host.engine();
    try e.submit(2, &.{ .prompt = &prompt_b, .max_tokens = 1 }, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expect(chunks.polled[5] >= chunks.polled[0] + 5); // a service at each of the 5 boundaries
    try std.testing.expectEqual(@as(u64, 0), r.stats().service_refused);
    e.cancel(1);
    e.cancel(3);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(Reason.cancelled, c.wait());
}

// -- restores refused for capacity ------------------------------------------------------------------------------------

test "a restore the tier refuses for capacity is asked again shortly, then resumes exactly" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.refuse = 2; // two refusals (TooManyRestores: restores in flight), then the begin goes through
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred); // once a request, however often it asked
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred_expired);
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 2), r.tier.refused);
    try std.testing.expectEqual(@as(u64, 3), r.tier.begins);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
}

test "a full tier refuses before any room is made: the request asks again, the refusal counted once" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    try shortThenPoint(r);
    r.tier.full = 3; // the restores in flight at the tier's cap for three asks
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), s.restores_refused); // once a request, not once an ask
    try std.testing.expectEqual(@as(u64, 1), r.tier.begins); // never asked to begin while full
    try std.testing.expectEqual(@as(u64, 0), s.evicted);
}

/// A memory state of t2 at 4 (a short turn's keep), then t2's durable point at 8.
fn shortThenPoint(r: *Rig) !void {
    var a: Box = .{};
    defer a.deinit();
    const short: Request = .{ .prompt = t2[0..6], .max_tokens = 1, .history_len = 4 };
    try r.host.engine().submit(1, &short, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    _ = try r.point(&t2, 8);
}

test "a restore still refused past its deferral bound resumes the shorter memory state, never a prefill from 0" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.host.tier.defer_ns = 40 * std.time.ns_per_ms;
    try shortThenPoint(r);
    r.tier.refuse = 1 << 20; // at capacity throughout
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 4), b.cached); // the memory state, kept while the restore was refused
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred_expired);
    try std.testing.expect(r.tier.refused >= 2); // asked again while deferred
    try std.testing.expectEqual(@as(u64, 0), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
}

test "with the deferral off a refused restore resumes what memory holds at once" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.host.tier.defer_ns = 0;
    try shortThenPoint(r);
    r.tier.refuse = 1 << 20;
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 4), b.cached);
    try b.exact(&t2);
    try std.testing.expectEqual(@as(u64, 0), r.stats().restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), r.tier.refused);
}

test "a refusal that waiting cannot cure is not deferred; a deferred request cancelled leaves every list once" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.refuse = 1;
    r.tier.refuse_err = error.PointMissing; // not a capacity refusal: admitted at once, prefilled
    _ = try r.point(&t2, 8);
    var a: Box = .{};
    defer a.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 0), a.cached);
    try std.testing.expectEqual(@as(u64, 0), r.stats().restores_deferred);
    // at capacity for long: the request waits deferred (no lane, counted waiting), and its cancel ends it there
    r.host.tier.defer_ns = 10 * std.time.ns_per_s;
    r.tier.refuse = 1 << 20;
    r.tier.refuse_err = error.TooManyRestores;
    const other = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6 };
    _ = try r.point(&other, 8);
    var b: Box = .{};
    defer b.deinit();
    const req_b: Request = .{ .prompt = &other, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &req_b, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_deferred == 1 and s.parked == 1;
        }
    }.ok);
    var st: api.Status = .{};
    e.status(&st, &.{});
    try std.testing.expectEqual(@as(u32, 1), st.waiting);
    try std.testing.expectEqual(@as(u32, 0), st.running);
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, b.wait());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.parked == 0;
        }
    }.ok);
    try std.testing.expectEqual(@as(usize, 0), r.tier.jobs.items.len); // nothing was ever begun for it
}

test "shutdown with a deferred request ends it once, failed (the stop's)" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.host.tier.defer_ns = 10 * std.time.ns_per_s;
    r.tier.refuse = 1 << 20;
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_deferred == 1;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(@as(?Reason, Reason.failed), b.finished());
}

// -- TF_SPILL_DEFER_MS, restarted when a restore ends, and TF_SPILL_LOAD_WAIT_MS, a request's whole wait --------------

/// `n` conversations' prompts of 12 tokens, each starting with its own token (none is a prefix of another's).
fn sessions(comptime n: usize) [n][12]u32 {
    var out: [n][12]u32 = undefined;
    for (&out, 0..) |*p, i| {
        p[0] = @intCast(100 + i);
        for (p[1..], 1..) |*t, j| t.* = @intCast((i * 31 + j * 7) % 89 + 1);
    }
    return out;
}

test "a burst of 16 sessions with two restores at a time: every request is restored, none expires while restores keep ending" {
    var boxes: [16]Box = @splat(.{}); // (before the rig: a failed check stops the host before the boxes go)
    defer for (&boxes) |*b| b.deinit();
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    // DEFER shorter than the burst (8 rounds of two restores of 20 ms or more: 160 ms at least), longer than a restore
    r.host.tier.defer_ns = 100 * std.time.ns_per_ms;
    r.tier.max_jobs = 2;
    r.tier.min_ns = 20 * std.time.ns_per_ms;
    var prompts = sessions(16);
    for (&prompts) |*p| _ = try r.point(p, 8);
    var reqs: [16]Request = undefined;
    for (&reqs, &prompts) |*q, *p| q.* = .{ .prompt = p, .max_tokens = 6 };
    for (&reqs, &boxes, 0..) |*q, *b, i| try r.host.engine().submit(@intCast(i + 1), q, b.sink());
    for (&boxes, &prompts) |*b, *p| {
        try std.testing.expectEqual(Reason.length, b.wait());
        try std.testing.expectEqual(@as(?u32, 8), b.cached);
        try b.exact(p);
    }
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 16), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred_expired);
    try std.testing.expect(s.restores_deferred >= 8); // most of the burst waited deferred
}

test "a request refused throughout while other restores keep ending is admitted on memory at TF_SPILL_LOAD_WAIT_MS, not kept deferred" {
    var x: Box = .{}; // (before the rig: a failed check stops the host before the boxes go)
    defer x.deinit();
    var others: [12]Box = @splat(.{});
    defer for (&others) |*b| b.deinit();
    const r = try Rig.create(.{ .load_ms = 5, .load_wait_ms = 300 });
    defer r.destroy();
    r.host.tier.defer_ns = 10 * std.time.ns_per_s; // each restore that ends restarts it: never reached here
    r.tier.max_jobs = 1; // the others' restores one at a time, 20 ms or more each: an ending every 20 ms or so
    r.tier.min_ns = 20 * std.time.ns_per_ms;
    try shortThenPoint(r); // X's memory state at 4 and its point at 8 (the last one put), which is never restored
    r.tier.refuse_point = r.tier.next_id - 1;
    var prompts = sessions(12);
    for (&prompts) |*p| _ = try r.point(p, 8);
    var reqs: [12]Request = undefined;
    for (&reqs, &prompts) |*q, *p| q.* = .{ .prompt = p, .max_tokens = 6 };
    const e = r.host.engine();
    for (&reqs, &others, 0..) |*q, *b, i| try e.submit(@intCast(i + 10), q, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits >= 1;
        }
    }.ok);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &req, x.sink());
    try std.testing.expectEqual(Reason.length, x.wait());
    const waited_ms = @divTrunc(std.Io.Clock.awake.now(io).toNanoseconds() - t0, std.time.ns_per_ms);
    try std.testing.expectEqual(@as(?u32, 4), x.cached); // what memory held
    try x.exact(&t2);
    try std.testing.expect(waited_ms >= 300); // it waited its whole LOAD_WAIT ...
    try std.testing.expect(waited_ms < 300 + 1500); // ... and no more (DEFER, restarted by every ending, is 10 s)
    try std.testing.expect(r.stats().restores_deferred_expired >= 1);
    for (&others) |*b| try std.testing.expectEqual(Reason.length, b.wait());
}

// -- the host takes the settings as given (from the environment, one parser) ------------------------------------------

/// A host over the fake with a prompt cache and no tier (the tier asked for but never bound, or not asked).
const Bare = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    store: pc.Store,
    host: LaneHost,

    fn create() !*Bare {
        const b = try gpa.create(Bare);
        errdefer gpa.destroy(b);
        b.* = .{ .cfg = undefined, .target = .{ .gpa = gpa }, .core = undefined, .store = undefined, .host = undefined };
        b.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        b.core = lanes.Engine.init(gpa, &b.cfg, b.target.backend(), b.clock.clock());
        b.store = pc.Store.init(gpa, .{ .ptr = &b.target, .vtable = &.{ .bytes = Rig.bytes, .save = Rig.save, .restore = Rig.restore, .drop = Rig.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        b.host = LaneHost.init(gpa, io, &b.core, .{ .lanes = 2 });
        b.host.cache = &b.store;
        return b;
    }

    fn destroy(b: *Bare) void {
        b.host.stop();
        b.store.deinit();
        b.core.deinit();
        b.target.deinit();
        b.cfg.deinit(gpa);
        gpa.destroy(b);
    }
};

test "configure takes every scheduling number as the settings give it, none clamped, and its defaults are the parser's" {
    const b = try Bare.create();
    defer b.destroy();
    const ms = std.time.ns_per_ms;
    b.host.configure(.{ .load_ms = 30, .pump_ms = 3, .load_wait_ms = 1000, .defer_ms = 0, .flush_ms = 20_000, .poll_ms = 5, .defer_retry_ms = 10, .defer_min = 2048 });
    const p = b.host.tier;
    try std.testing.expectEqual(@as(u64, 30 * ms), p.service_ns);
    try std.testing.expectEqual(@as(u64, 3 * ms), p.pump_ns);
    try std.testing.expectEqual(@as(i96, 1000 * ms), p.load_wait_ns);
    try std.testing.expectEqual(@as(i96, 0), p.defer_ns);
    try std.testing.expectEqual(@as(u64, 20_000 * ms), p.flush_ns);
    try std.testing.expectEqual(@as(u64, 5 * ms), p.poll_ns);
    try std.testing.expectEqual(@as(i96, 10 * ms), p.defer_retry_ns);
    try std.testing.expectEqual(@as(u32, 2048), p.defer_min);
    // the policy's defaults are the parser's: configure of the parser's defaults changes nothing
    b.host.configure(.{});
    try std.testing.expectEqual(LaneHost.TierPolicy{}, b.host.tier);
}

test "refused, never clamped: a service budget of 0, staging's share above it or a wait of 0 turn the tier off, named, and the host serves from memory" {
    const Case = struct { sp: api.Spill, why: []const u8 };
    for ([_]Case{
        .{ .sp = .{ .asked = true, .load_ms = 0, .pump_ms = 0 }, .why = "TF_SPILL_LOAD_MS=0: " },
        .{ .sp = .{ .asked = true, .load_ms = 4, .pump_ms = 5 }, .why = "TF_SPILL_PUMP_MS=5: more than TF_SPILL_LOAD_MS=4" },
        .{ .sp = .{ .asked = true, .load_wait_ms = 0 }, .why = "TF_SPILL_LOAD_WAIT_MS=0: " },
    }) |c| {
        var b: Box = .{}; // (before the rig: a failed check stops the host before the box goes)
        defer b.deinit();
        const r = try Rig.create(c.sp);
        defer r.destroy();
        _ = try r.point(&t2, 8);
        try r.host.engine().submit(1, &.{ .prompt = &t2, .max_tokens = 6, .history_len = 14 }, b.sink());
        try std.testing.expectEqual(Reason.length, b.wait());
        try std.testing.expectEqual(@as(?u32, 0), b.cached); // no restore: the disk tier is off
        try b.exact(&t2);
        const st = r.status();
        const sp = st.spill orelse return error.TestNoSpillState;
        try std.testing.expectEqual(api.SpillState.State.off, sp.state);
        std.testing.expect(std.mem.startsWith(u8, sp.reason(), c.why)) catch |err| {
            std.debug.print("reason \"{s}\", wanted \"{s}...\"\n", .{ sp.reason(), c.why });
            return err;
        };
        try std.testing.expectEqual(@as(u64, 0), st.cache.?.restore_waits);
        try std.testing.expectEqual(@as(u64, 0), r.tier.begins);
        try std.testing.expect(r.store.tier == null and !r.store.groups); // the store as with no tier (set before start)
        try std.testing.expectEqual(@as(u64, 0), st.cache.?.chats_continued);
    }
}

test "the idle host polls work in flight every TF_SPILL_POLL_MS" {
    var took: [2]i96 = undefined;
    for ([_]u32{ 50, 1 }, &took) |poll_ms, *t| {
        var b: Box = .{}; // (before the rig: a failed check stops the host before the box goes)
        defer b.deinit();
        const r = try Rig.create(.{ .load_ms = 5, .poll_ms = poll_ms });
        defer r.destroy();
        r.tier.polls = 3; // three services, two idle waits between them
        _ = try r.point(&t2, 8);
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        try r.host.engine().submit(1, &.{ .prompt = &t2, .max_tokens = 1, .history_len = 14 }, b.sink());
        try std.testing.expectEqual(Reason.length, b.wait());
        t.* = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
        try std.testing.expectEqual(@as(?u32, 8), b.cached);
    }
    try std.testing.expect(took[0] >= 2 * 50 * std.time.ns_per_ms); // two waits of 50 ms at least
    try std.testing.expect(took[1] < took[0]); // and at 1 ms less than that
}

test "with the tier asked for but off at the start, status carries the cache counters and the tier's state, off with its reason" {
    const b = try Bare.create();
    defer b.destroy();
    b.host.configure(.{ .asked = true, .off = "the store did not open in TF_DS_SPILL_OPEN_MS=300000" });
    try b.host.start();
    var x: Box = .{};
    defer x.deinit();
    try b.host.engine().submit(1, &.{ .prompt = &t2, .max_tokens = 3, .history_len = 14 }, x.sink());
    try std.testing.expectEqual(Reason.length, x.wait());
    var st: api.Status = .{};
    b.host.engine().status(&st, &.{});
    const sp = st.spill orelse return error.TestNoSpillState;
    try std.testing.expectEqual(api.SpillState.State.off, sp.state);
    try std.testing.expectEqualStrings("the store did not open in TF_DS_SPILL_OPEN_MS=300000", sp.reason());
    const c = st.cache orelse return error.TestNoCacheStats; // the memory counters all the same
    try std.testing.expectEqual(@as(u64, 1), c.memory_misses);
    try std.testing.expectEqual(@as(u64, 1), c.kept);
    try std.testing.expectEqual(@as(u64, 0), c.disk_lookups);
}

test "not asked: no tier state and no cache counters (TF_DS_SPILL_GIB unset or 0 leaves status unchanged)" {
    const b = try Bare.create();
    defer b.destroy();
    b.host.configure(.{});
    try b.host.start();
    var x: Box = .{};
    defer x.deinit();
    try b.host.engine().submit(1, &.{ .prompt = &t2, .max_tokens = 3, .history_len = 14 }, x.sink());
    try std.testing.expectEqual(Reason.length, x.wait());
    var st: api.Status = .{};
    b.host.engine().status(&st, &.{});
    try std.testing.expect(st.spill == null);
    try std.testing.expect(st.cache == null);
}

test "the tier's state on status: on while it serves, off with the tier's own reason once it turns itself off" {
    var b: Box = .{}; // (before the rig: a failed check stops the host before the boxes go)
    defer b.deinit();
    var x: Box = .{};
    defer x.deinit();
    const r = try Rig.create(.{ .asked = true, .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    try e.submit(1, &.{ .prompt = &t1, .max_tokens = 2 }, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(api.SpillState.State.on, (r.status().spill orelse return error.TestNoSpillState).state);
    {
        r.tier.mu.lockUncancelable(io);
        defer r.tier.mu.unlock(io);
        r.tier.off_why = "probe: rank 2's kept bytes differ from rank 0's";
    }
    try e.submit(2, &.{ .prompt = &t2, .max_tokens = 2 }, x.sink()); // (a request: the engine thread looks again)
    try std.testing.expectEqual(Reason.length, x.wait());
    const sp = r.status().spill orelse return error.TestNoSpillState;
    try std.testing.expectEqual(api.SpillState.State.off, sp.state);
    try std.testing.expectEqualStrings("probe: rank 2's kept bytes differ from rank 0's", sp.reason());
}

// -- the tier's call timings ------------------------------------------------------------------------------------------

test "with a tier every host call into the store is timed, the tier's own times pass through, and without a tier nothing is" {
    var b: Box = .{}; // (before the rig: a failed check stops the host before the box goes)
    defer b.deinit();
    const r = try Rig.createWith(.{ .load_ms = 5 }, struct {
        fn pre(rig: *Rig) void { // the tier's own (before the host's thread reads them)
            rig.tier.times.land_piece.observe(5000);
            rig.tier.times.lock_wait.observe(3000);
            rig.tier.times.block_read.observe(4000);
            rig.tier.times.read_wall.observe(6000);
            rig.tier.times.restore_land.observe(9000);
            rig.tier.times.lookup.observe(1); // (a call the host times itself: never merged with the host's)
        }
    }.pre);
    defer r.destroy();
    r.tier.polls = 3;
    _ = try r.point(&t2, 8);
    try r.host.engine().submit(1, &.{ .prompt = &t2, .max_tokens = 6, .history_len = 14 }, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.writes_enqueued == 1 and s.times.keep.count == 1;
        }
    }.ok);
    const t = r.stats().times;
    try std.testing.expectEqual(@as(u64, 1), t.lookup.count); // its one park's probe
    try std.testing.expectEqual(@as(u64, 1), t.keep.count); // its one mark (14)
    try std.testing.expectEqual(@as(u64, 3), t.poll.count); // the restore's three polls
    try std.testing.expect(t.service.count >= 3 and t.pump.count >= 3); // a service each, a pump in each
    try std.testing.expect(t.service.max_ns >= t.poll.max_ns); // (a service holds its polls)
    try std.testing.expectEqual(@as(u64, 1), t.land_piece.count); // the tier's own, as it gave them
    try std.testing.expectEqual(@as(u64, 5000), t.land_piece.max_ns);
    try std.testing.expectEqual(@as(u64, 1), t.lock_wait.count);
    try std.testing.expectEqual(@as(u64, 0), t.probe.count);
    try std.testing.expectEqual(@as(u64, 4000), t.block_read.max_ns); // the store's and the restore's, passed through
    try std.testing.expectEqual(@as(u64, 1), t.read_wall.count);
    try std.testing.expectEqual(@as(u64, 9000), t.restore_land.max_ns);
    try std.testing.expectEqual(@as(u64, 1), t.wait.count); // its wait for the restore, park to wake
    try std.testing.expectEqual(@as(u64, 1), t.tail.count); // its resumed pass (the 8 tokens past the restored 8)

    // the tier asked for but never bound: the counters show, and no call was timed
    const bare = try Bare.create();
    defer bare.destroy();
    bare.host.configure(.{ .asked = true, .off = "the store did not open in TF_DS_SPILL_OPEN_MS=300000" });
    try bare.host.start();
    var x: Box = .{};
    defer x.deinit();
    try bare.host.engine().submit(1, &.{ .prompt = &t2, .max_tokens = 3, .history_len = 14 }, x.sink());
    try std.testing.expectEqual(Reason.length, x.wait());
    var st: api.Status = .{};
    bare.host.engine().status(&st, &.{});
    const none = (st.cache orelse return error.TestNoCacheStats).times;
    inline for (@typeInfo(pc.TierTimes).@"struct".field_names) |name| try std.testing.expectEqual(@as(u64, 0), @field(none, name).count);
}

test "with a tier each round of the lane core is timed; without one none is" {
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.createWith(.{ .load_ms = 5 }, null);
    defer r.destroy();
    try r.host.engine().submit(1, &.{ .prompt = &t2, .max_tokens = 6, .history_len = 14 }, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.times.round.count >= 5; // a round a generated token at least (6 tokens, the first from the pass)
        }
    }.ok);
    const t = r.stats().times;
    try std.testing.expect(t.round.max_ns > 0 and t.round.sum_ns >= t.round.max_ns);
    // the tier asked for but never bound: no round timed (nothing to compare it with)
    const bare = try Bare.create();
    defer bare.destroy();
    bare.host.configure(.{ .asked = true, .off = "the store did not open in TF_DS_SPILL_OPEN_MS=300000" });
    try bare.host.start();
    var x: Box = .{};
    defer x.deinit();
    try bare.host.engine().submit(1, &.{ .prompt = &t2, .max_tokens = 3, .history_len = 14 }, x.sink());
    try std.testing.expectEqual(Reason.length, x.wait());
    var st: api.Status = .{};
    bare.host.engine().status(&st, &.{});
    try std.testing.expectEqual(@as(u64, 0), (st.cache orelse return error.TestNoCacheStats).times.round.count);
}

test "a parked request cancelled during another request's lone prompt pass ends at the next chunk boundary, its restore stopped, not after the pass" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release); // (its restore never ends in the test)
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const e = r.host.engine();
    try e.submit(1, &.{ .prompt = &t2, .max_tokens = 6, .history_len = 14 }, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const Watch = struct {
        engine: api.Engine,
        parked: *Box,
        tier: *ft.FakeTier,
        ended_by: ?usize = null, // the chunk at whose start the parked request had ended
        cancels_by: u64 = 0,
        fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len != 12) return;
            if (chunk == 1) w.engine.cancel(1); // its client goes while the other's pass runs alone
            if (w.ended_by == null and chunk > 1 and w.parked.finished() != null) {
                w.ended_by = chunk;
                w.cancels_by = w.tier.cancels;
            }
        }
    };
    var watch: Watch = .{ .engine = e, .parked = &b, .tier = &r.tier };
    r.target.prefill_chunks = 6; // (no other stream live: the prefill yield's round part never runs, only the tier part)
    r.target.prefill_hook = Watch.call;
    r.target.prefill_hook_ctx = &watch;
    const other = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var o: Box = .{};
    defer o.deinit();
    try e.submit(2, &.{ .prompt = &other, .max_tokens = 1 }, o.sink());
    try std.testing.expectEqual(Reason.length, o.wait());
    try std.testing.expectEqual(Reason.cancelled, b.wait());
    try std.testing.expect(watch.ended_by != null and watch.ended_by.? <= 3); // the next boundary or the one after
    try std.testing.expectEqual(@as(u64, 1), watch.cancels_by); // its restore stopped then too
}

test "a restore refused for room in the pool (the family's ContextFull) is asked again once room is back, and counted" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    var a: Box = .{};
    defer a.deinit();
    const along = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    try e.submit(1, &.{ .prompt = &along, .max_tokens = 100000 }, a.sink()); // A holds the pool
    while (a.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    _ = try r.point(&t2, 8);
    r.tier.refuse = 1;
    r.tier.refuse_err = error.ContextFull; // no room for the restore's destination while A runs
    r.target.refuse_full = 1; // nor for B's own pass: it goes back to the queue
    var b: Box = .{};
    defer b.deinit();
    try e.submit(2, &.{ .prompt = &t2, .max_tokens = 6, .history_len = 14 }, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_refused >= 1;
        }
    }.ok);
    e.cancel(1); // A goes: room is back
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // restored, not prefilled from 0
    try b.exact(&t2);
}

test "a deferred request whose TF_SPILL_LOAD_WAIT_MS passed while both lanes were held is admitted on memory without beginning its restore" {
    var x: Box = .{};
    defer x.deinit();
    var a1: Box = .{};
    defer a1.deinit();
    var a2: Box = .{};
    defer a2.deinit();
    const r = try Rig.create(.{ .load_ms = 5, .load_wait_ms = 300 });
    defer r.destroy();
    r.host.tier.defer_ns = 5 * std.time.ns_per_s;
    r.host.tier.defer_retry_ns = std.time.ns_per_ms;
    _ = try r.point(&t2, 8);
    r.tier.refuse = 1 << 20; // refused for capacity while the lanes fill: X waits deferred
    const e = r.host.engine();
    try e.submit(2, &.{ .prompt = &t2, .max_tokens = 6, .history_len = 14 }, x.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_deferred == 1;
        }
    }.ok);
    const p1 = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    const p2 = [_]u32{ 9, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    try e.submit(1, &.{ .prompt = &p1, .max_tokens = 100000 }, a1.sink());
    try e.submit(3, &.{ .prompt = &p2, .max_tokens = 100000 }, a2.sink());
    for (0..5000) |i| { // (5 s at most: a lane that never decodes fails the test, it does not hang it)
        if (a1.count() > 0 and a2.count() > 0) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        if (i == 4999) {
            std.debug.print("timed out waiting for: both long requests to hold a lane and decode\n", .{});
            return error.Timeout;
        }
    }
    // Both lanes are held, but X may still be on the deferred list: its next retry (1 ms) wakes it and its read-ahead is
    // refused too (`ahead`: not deferred again; it waits ready for a lane, and reads ahead no more). The tier gives room
    // back only once X is in that state: given earlier, the read-ahead BEGINS X's restore (X's wait is far inside its
    // 300 ms: restores_begun 1, expired 0, the engine right and the test not at its premise). It failed 18 of 20 on
    // aarch64, where the window between the lanes filling and the room returning is shorter than the retry.
    try r.untilReadyAfterAhead();
    r.tier.refuse = 0; // the tier has room again, but X waits for a lane now (ready, no lane free)
    std.Io.sleep(io, .fromMilliseconds(400), .awake) catch {}; // past its TF_SPILL_LOAD_WAIT_MS from its first wait
    const before = r.stats();
    e.cancel(1); // a lane frees
    try std.testing.expectEqual(Reason.cancelled, a1.wait());
    try std.testing.expectEqual(Reason.length, x.wait());
    const after = r.stats();
    try std.testing.expectEqual(before.restores_begun, after.restores_begun); // no restore begun past its wait ...
    try std.testing.expectEqual(@as(u64, 0), after.restores_cancelled); // ... and none cancelled at once for it
    try std.testing.expect(after.restores_deferred_expired >= 1);
    e.cancel(3);
    try std.testing.expectEqual(Reason.cancelled, a2.wait());
}

test "the tier's services between a prompt's chunks are timed apart from the run loop's" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release); // (its restore never ends: tier work at every boundary)
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const e = r.host.engine();
    try e.submit(1, &.{ .prompt = &t2, .max_tokens = 6, .history_len = 14 }, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1 and s.times.service.count > 0;
        }
    }.ok);
    const loop_before = r.stats().times.service.count;
    r.target.prefill_chunks = 6; // another request's pass, alone: the tier part between its chunks
    const other = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var o: Box = .{};
    defer o.deinit();
    try e.submit(2, &.{ .prompt = &other, .max_tokens = 1 }, o.sink());
    try std.testing.expectEqual(Reason.length, o.wait());
    const t = r.stats().times;
    try std.testing.expect(t.service_chunk.count >= 4); // one at each of its boundaries with work
    try std.testing.expect(t.service.count >= loop_before); // the run loop's own, never mixed in
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
}

/// The rig's lane core on the fake's chunked backend (before the host starts): prompts taken a step of four tokens at
/// a time, as with TF_DS_PREFILL_YIELD=1.
fn chunked(r: *Rig) void {
    r.core.deinit();
    r.core = lanes.Engine.init(gpa, &r.cfg, r.target.chunkedBackend(), r.clock.clock());
}

test "prompts taken a chunk at a time: a request arriving during a long prompt's pass starts its restore at once and resumes before that pass ends" {
    const r = try Rig.createWith(.{ .load_ms = 5 }, chunked);
    defer r.destroy();
    r.tier.polls = 2;
    r.tier.land_polls = 2;
    _ = try r.point(&t2, 8);
    var long: [64]u32 = undefined; // sixteen steps
    for (&long, 0..) |*t, i| t.* = @intCast(100 + i % 7);
    const Watch = struct {
        engine: api.Engine,
        req: *const Request,
        b: *Box,
        submitted: bool = false,
        b_started_at: ?usize = null, // the long prompt's step at whose start B's pass had ended
        fn call(ctx: *anyopaque, s: *lanes.Stream, step: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len != 64) return;
            if (step == 1 and !w.submitted) {
                w.submitted = true;
                w.engine.submit(2, w.req, w.b.sink()) catch unreachable;
            }
            if (w.b_started_at == null and w.b.started()) w.b_started_at = step;
        }
    };
    var b: Box = .{};
    defer b.deinit();
    const breq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    var watch: Watch = .{ .engine = r.host.engine(), .req = &breq, .b = &b };
    r.target.prefill_hook = Watch.call;
    r.target.prefill_hook_ctx = &watch;
    var a: Box = .{};
    defer a.deinit();
    const areq: Request = .{ .prompt = &long, .max_tokens = 2 };
    try r.host.engine().submit(1, &areq, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // its pass resumed the 8 tokens read back from disk
    try b.exact(&t2);
    try a.exact(&long);
    // its restore ran between the long prompt's steps and its own pass ended before that prompt's last step
    try std.testing.expect(watch.b_started_at.? < 15);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restore_waits);
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
}
