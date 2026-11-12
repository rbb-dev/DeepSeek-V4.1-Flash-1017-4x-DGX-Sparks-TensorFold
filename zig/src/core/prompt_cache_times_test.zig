//! The tier's call timings: `Timing`'s log2 buckets and quantiles, and the store passing the tier's own times (landing
//! pieces, probes, the store lock) through to the lane host.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const FakeTier = @import("prompt_cache_fake_tier.zig").FakeTier;
const Timing = pc.Timing;
const gpa = std.testing.allocator;

test "a timing puts each value in the first bucket whose bound holds it, and keeps the count, sum and max" {
    var t: Timing = .{};
    t.observe(0);
    t.observe(1024); // the first bound: 2^10 ns
    t.observe(1025);
    t.observe(2048); // a bound itself: its own bucket, not the next
    t.observe(3 << 20);
    t.observe(1 << 40); // past the last bound: the overflow bucket
    try std.testing.expectEqual(@as(u64, 2), t.buckets[0]);
    try std.testing.expectEqual(@as(u64, 2), t.buckets[1]);
    try std.testing.expectEqual(@as(u64, 1), t.buckets[12]); // 3 MiB of ns: above 2^21, at most 2^22
    try std.testing.expectEqual(@as(u64, 1), t.buckets[Timing.edges]);
    try std.testing.expectEqual(@as(u64, 6), t.count);
    try std.testing.expectEqual(@as(u64, 0 + 1024 + 1025 + 2048 + (3 << 20) + (1 << 40)), t.sum_ns);
    try std.testing.expectEqual(@as(u64, 1 << 40), t.max_ns);
    try std.testing.expectEqual(@as(u64, 1024), Timing.bound(0));
    try std.testing.expectEqual(@as(u64, 1 << 36), Timing.bound(Timing.edges - 1));
    var u: Timing = .{};
    u.observe(7);
    t.merge(u);
    try std.testing.expectEqual(@as(u64, 7), t.count);
    try std.testing.expectEqual(@as(u64, 3), t.buckets[0]);
}

test "a timing's quantile is the bound of the bucket holding the true quantile: never below it, under twice it, at most the max" {
    var prng: std.Random.DefaultPrng = .init(8);
    const r = prng.random();
    var case: usize = 0;
    while (case < 200) : (case += 1) {
        var t: Timing = .{};
        var v: [300]u64 = undefined;
        const n = 1 + r.uintLessThan(usize, v.len);
        for (v[0..n]) |*x| {
            const shift: u6 = r.uintLessThan(u6, 37); // up to the last bound, 2^36 ns (the overflow bucket is below)
            x.* = r.uintLessThan(u64, (@as(u64, 1) << shift) + 1);
            t.observe(x.*);
        }
        std.mem.sort(u64, v[0..n], {}, std.sort.asc(u64));
        for ([_]f64{ 0.5, 0.9, 0.99, 1.0 }) |q| {
            const rank: usize = @max(1, @as(usize, @intFromFloat(@ceil(q * @as(f64, @floatFromInt(n))))));
            const truth = v[rank - 1];
            const e = t.quantile(q);
            try std.testing.expect(e >= truth);
            try std.testing.expect(e <= t.max_ns);
            try std.testing.expect(e <= @max(2 * truth, Timing.bound(0)));
        }
    }
    const empty: Timing = .{};
    try std.testing.expectEqual(@as(u64, 0), empty.quantile(0.5));
    var long: Timing = .{}; // past the last bound only the max is known
    long.observe(1 << 37);
    long.observe(1 << 40);
    try std.testing.expectEqual(@as(u64, 1 << 40), long.quantile(0.5));
}

/// A family the store never asks anything of here.
const Idle = struct {
    fn snapshots(f: *Idle) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
        return error.Unused;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.Unused;
    }
    fn drop(_: *anyopaque, _: pc.Saved) void {}
    fn pack(_: *anyopaque, _: std.mem.Allocator, _: pc.Saved) anyerror![]u8 {
        return error.Unused;
    }
    fn unpack(_: *anyopaque, _: []const u8) anyerror!pc.Saved {
        return error.Unused;
    }
};

test "the store passes the tier's own times through (landing pieces, probes, the store lock); with no tier, none" {
    var fam: Idle = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = .{ .ctx = &fam, .pack = Idle.pack, .unpack = Idle.unpack } };
    defer tier.deinit();
    var s = pc.Store.init(gpa, fam.snapshots(), .{}, 1 << 20);
    defer s.deinit();
    try std.testing.expect(s.tierTimes() == null);
    tier.times.land_piece.observe(5000);
    tier.times.probe.observe(1 << 20);
    tier.times.lock_wait.observe(3000);
    s.setTier(tier.tier(), 1);
    const t = s.tierTimes().?;
    try std.testing.expectEqual(@as(u64, 1), t.land_piece.count);
    try std.testing.expectEqual(@as(u64, 5000), t.land_piece.max_ns);
    try std.testing.expectEqual(@as(u64, 1), t.probe.count);
    try std.testing.expectEqual(@as(u64, 1), t.lock_wait.count);
    try std.testing.expectEqual(@as(u64, 0), t.lookup.count); // the lane host's own, not the tier's
    s.setTier(null, 0);
}
