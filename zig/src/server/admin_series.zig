//! One sample of the engine's status a second, kept in a ring, so the admin page draws its charts from the engine's own
//! record and a reload loses nothing. The sampler thread is the only periodic caller of the engine's status for the admin
//! surface; a request for a series reads the ring and never the engine.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const admin_requests = @import("admin_requests.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// Samples kept per series (an hour at one a second).
pub const capacity = 3600;

const base_names = [_][]const u8{ "decode_tps", "prefill_tps", "running", "waiting", "generation_tokens", "memory_active" };

/// Seats whose KV share is kept a second (the first slots of the requests table): series seat_kv_0 .. seat_kv_15.
pub const seat_rows = 16;

pub const names = blk: {
    var out: [base_names.len + seat_rows][]const u8 = undefined;
    for (base_names, 0..) |n, i| out[i] = n;
    for (0..seat_rows) |k| out[base_names.len + k] = std.fmt.comptimePrint("seat_kv_{d}", .{k});
    break :blk out;
};

pub const Sampler = struct {
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    /// NaN: the value was not available at that second.
    ring: [names.len][]f32,
    /// Samples taken since the thread started; the newest sits at (taken - 1) % capacity.
    taken: u64 = 0,
    /// The first sample's unix time in ms (0 until one is taken), and the newest sample's.
    since_ms: i64 = 0,
    last_ms: i64 = 0,
    stop_flag: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// The requests table the seats' KV share is read from (null: the seat series stay empty).
    requests: ?*admin_requests.Registry = null,
    /// The seats' KV tokens for the next `sample` (set by the sampler thread just before it; NaN = not tracked).
    seat_kv: [seat_rows]f32 = @splat(std.math.nan(f32)),

    pub fn create(gpa: Allocator) !*Sampler {
        const s = try gpa.create(Sampler);
        errdefer gpa.destroy(s);
        s.* = .{ .gpa = gpa, .ring = undefined };
        var made: usize = 0;
        errdefer for (s.ring[0..made]) |r| gpa.free(r);
        for (&s.ring) |*r| {
            r.* = try gpa.alloc(f32, capacity);
            @memset(r.*, std.math.nan(f32));
            made += 1;
        }
        return s;
    }

    pub fn destroy(s: *Sampler) void {
        s.stop();
        for (s.ring) |r| s.gpa.free(r);
        s.gpa.destroy(s);
    }

    /// Takes one sample: `st` and `mem` as the engine gave them at unix time `now_ms`.
    pub fn sample(s: *Sampler, io: std.Io, now_ms: i64, st: api.Status, mem: ?api.Memory) void {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const at: usize = @intCast(s.taken % capacity);
        s.ring[0][at] = @floatCast(st.decode_tokens_per_second);
        s.ring[1][at] = @floatCast(st.prefill_tokens_per_second);
        s.ring[2][at] = @floatFromInt(st.running);
        s.ring[3][at] = @floatFromInt(st.waiting);
        s.ring[4][at] = @floatFromInt(st.generation_tokens);
        s.ring[5][at] = if (mem) |m| @floatFromInt(m.active) else std.math.nan(f32);
        for (0..seat_rows) |k| s.ring[base_names.len + k][at] = s.seat_kv[k];
        if (s.taken == 0) s.since_ms = now_ms;
        s.last_ms = now_ms;
        s.taken += 1;
    }

    /// Starts the thread that takes a sample every second until `stop`.
    pub fn start(s: *Sampler, io: std.Io, engine: api.Engine) void {
        s.thread = std.Thread.spawn(.{}, run, .{ s, io, engine }) catch null;
    }

    fn run(s: *Sampler, io: std.Io, engine: api.Engine) void {
        while (!s.stop_flag.load(.acquire)) {
            var st: api.Status = .{};
            engine.status(&st, &.{});
            const mem = engine.memory(false);
            if (s.requests) |r| r.seatKv(io, &s.seat_kv);
            s.sample(io, @intCast(@divTrunc(std.Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_ms)), st, mem);
            var waited: u32 = 0;
            while (waited < 10 and !s.stop_flag.load(.acquire)) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
        }
    }

    pub fn stop(s: *Sampler) void {
        s.stop_flag.store(true, .release);
        if (s.thread) |t| t.join();
        s.thread = null;
    }

    /// `points` are the mean and `peaks` the highest sample of each bucket, oldest first; `last_ms` is the unix time of the newest
    /// sample, which ends the last bucket (the page counts the buckets back from it).
    pub const Window = struct { points: []Value, peaks: []Value, since_ms: i64, last_ms: i64, step_s: u32 };

    /// The last `seconds` of series `name` in fixed buckets of `step` samples: bucket k holds the samples numbered k*step to
    /// k*step+step-1 since the sampler began, whatever the window's end is, so a bucket's mean and peak do not change as the
    /// window moves on (only the newest bucket, still filling, and the oldest, cut by the window, are partial). Null where no
    /// sample of a bucket was available. Null when `name` is no series.
    pub fn window(s: *Sampler, io: std.Io, a: Allocator, name: []const u8, seconds: u32, step: u32) !?Window {
        const idx = for (names, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) break i;
        } else return null;
        const stride: usize = @max(1, step);
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const taken: usize = @intCast(s.taken);
        const kept: usize = @min(taken, capacity); // the samples the ring still holds: numbers taken-kept .. taken-1
        const want = @min(@as(usize, @max(1, seconds)), kept);
        if (want == 0) return .{ .points = try a.alloc(Value, 0), .peaks = try a.alloc(Value, 0), .since_ms = s.since_ms, .last_ms = s.last_ms, .step_s = @intCast(stride) };
        const first_bucket = (taken - want) / stride;
        const last_bucket = (taken - 1) / stride;
        const buckets = last_bucket - first_bucket + 1;
        const points = try a.alloc(Value, buckets);
        const peaks = try a.alloc(Value, buckets);
        for (points, peaks, 0..) |*mean, *peak, b| {
            const lo = @max((first_bucket + b) * stride, taken - kept);
            const hi = @min((first_bucket + b) * stride + stride, taken); // exclusive
            var sum: f64 = 0;
            var top: f64 = 0;
            var n: usize = 0;
            var k = lo;
            while (k < hi) : (k += 1) {
                const v = s.ring[idx][k % capacity];
                if (std.math.isNan(v)) continue;
                sum += v;
                top = if (n == 0) v else @max(top, v);
                n += 1;
            }
            mean.* = if (n == 0) .null else .{ .float = sum / @as(f64, @floatFromInt(n)) };
            peak.* = if (n == 0) .null else .{ .float = top };
        }
        return .{ .points = points, .peaks = peaks, .since_ms = s.since_ms, .last_ms = s.last_ms, .step_s = @intCast(stride) };
    }
};

test "samples land in the ring and a window averages each step, oldest first" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s = try Sampler.create(std.testing.allocator);
    defer s.destroy();
    var t: i64 = 1_000;
    var i: u32 = 0;
    while (i < 6) : (i += 1) {
        s.sample(io, t, .{ .decode_tokens_per_second = @floatFromInt(i * 10), .running = i }, if (i < 3) null else .{ .active = 100 * i, .cache = 0, .peak = 0 });
        t += 1000;
    }
    const w = (try s.window(io, arena.allocator(), "decode_tps", 600, 1)).?;
    try std.testing.expectEqual(@as(usize, 6), w.points.len);
    try std.testing.expectEqual(@as(i64, 1000), w.since_ms);
    try std.testing.expectEqual(@as(f64, 0), w.points[0].float);
    try std.testing.expectEqual(@as(f64, 50), w.points[5].float);
    const two = (try s.window(io, arena.allocator(), "decode_tps", 600, 2)).?;
    try std.testing.expectEqual(@as(usize, 3), two.points.len);
    try std.testing.expectEqual(@as(f64, 5), two.points[0].float); // mean of 0 and 10
    try std.testing.expectEqual(@as(f64, 45), two.points[2].float); // mean of 40 and 50
    const mem = (try s.window(io, arena.allocator(), "memory_active", 600, 1)).?;
    try std.testing.expect(mem.points[0] == .null and mem.points[2] == .null);
    try std.testing.expectEqual(@as(f64, 300), mem.points[3].float);
    const last3 = (try s.window(io, arena.allocator(), "running", 3, 1)).?;
    try std.testing.expectEqual(@as(usize, 3), last3.points.len);
    try std.testing.expectEqual(@as(f64, 3), last3.points[0].float);
    try std.testing.expect((try s.window(io, arena.allocator(), "nothing", 600, 1)) == null);
}

test "the ring wraps: only the newest hour is kept" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s = try Sampler.create(std.testing.allocator);
    defer s.destroy();
    var i: u32 = 0;
    while (i < capacity + 10) : (i += 1) s.sample(io, i, .{ .running = i }, null);
    const w = (try s.window(io, arena.allocator(), "running", 100000, 1)).?;
    try std.testing.expectEqual(@as(usize, capacity), w.points.len);
    try std.testing.expectEqual(@as(f64, 10), w.points[0].float);
    try std.testing.expectEqual(@as(f64, capacity + 9), w.points[capacity - 1].float);
}

test "a bucket keeps its mean and peak as the window moves on: buckets are fixed to the sample count, not to the window's end" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s = try Sampler.create(std.testing.allocator);
    defer s.destroy();
    // a spike of 90 at sample 7, otherwise 10; buckets of 4 samples: [0-3] [4-7] [8-11]
    var i: u32 = 0;
    while (i < 10) : (i += 1) s.sample(io, 1_000 + @as(i64, i) * 1000, .{ .decode_tokens_per_second = if (i == 7) 90 else 10 }, null);
    const before = (try s.window(io, arena.allocator(), "decode_tps", 8, 4)).?;
    // the 8 last samples are 2..9: buckets 0 (samples 0-3), 1 (4-7), 2 (8, 9)
    try std.testing.expectEqual(@as(usize, 3), before.points.len);
    try std.testing.expectEqual(@as(f64, 90), before.peaks[1].float);
    try std.testing.expectEqual(@as(f64, 30), before.points[1].float); // (10+10+10+90)/4
    try std.testing.expectEqual(@as(i64, 10_000), before.last_ms);
    // one more sample: the window moves on by one, the middle bucket is the same
    s.sample(io, 11_000, .{ .decode_tokens_per_second = 10 }, null);
    const after = (try s.window(io, arena.allocator(), "decode_tps", 8, 4)).?;
    // window samples 3..10: buckets 0 [3 of 0-3], 1 [4-7], 2 [8-10]
    try std.testing.expectEqual(@as(usize, 3), after.points.len);
    try std.testing.expectEqual(before.points[1].float, after.points[1].float);
    try std.testing.expectEqual(before.peaks[1].float, after.peaks[1].float); // RED with buckets counted back from the newest sample: the spike falls into [3-6] / [7-10] and the peaks differ
    // no sample yet: no points, and no division by zero
    const empty = try Sampler.create(std.testing.allocator);
    defer empty.destroy();
    try std.testing.expectEqual(@as(usize, 0), (try empty.window(io, arena.allocator(), "decode_tps", 60, 5)).?.points.len);
}
