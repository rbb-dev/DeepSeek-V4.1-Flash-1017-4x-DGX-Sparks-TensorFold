//! Store.charge_tokens with a tier: a restore in flight charges its key's ids, an adopted state its two arrays.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const FakeTier = @import("prompt_cache_fake_tier.zig").FakeTier;
const gpa = std.testing.allocator;

/// A family whose state is its position and a sum (the bytes are 100 + at, as the core tests' Fake).
const Sum = struct {
    const State = struct { at: u32, sum: u64 };
    fn snapshots(f: *Sum) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 100 + @as(u64, at);
    }
    fn save(_: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const st = try gpa.create(State);
        st.* = .{ .at = at, .sum = 0 };
        return st;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(_: *anyopaque, saved: pc.Saved) void {
        gpa.destroy(@as(*State, @ptrCast(@alignCast(saved))));
    }
    fn codec(f: *Sum) @import("prompt_cache_fake_tier.zig").Codec {
        return .{ .ctx = f, .pack = pack, .unpack = unpack };
    }
    fn pack(_: *anyopaque, a: std.mem.Allocator, saved: pc.Saved) anyerror![]u8 {
        return a.dupe(u8, std.mem.asBytes(@as(*State, @ptrCast(@alignCast(saved)))));
    }
    fn unpack(_: *anyopaque, b: []const u8) anyerror!pc.Saved {
        const st = try gpa.create(State);
        @memcpy(std.mem.asBytes(st), b);
        return st;
    }
};

test "token arrays with the tier: a restore in flight charges its key, the adopted state both arrays, inside the budget" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 2 };
    defer tier.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .grid = 16 }, 1000);
    defer s.deinit();
    s.charge_tokens = true;
    s.setTier(tier.tier(), 1);
    var key: [48]u32 = undefined;
    for (&key, 0..) |*x, i| x.* = @intCast(7 + i);
    const st: Sum.State = .{ .at = 32, .sum = 0 };
    _ = try tier.put(key[0..32], 32, std.mem.asBytes(&st), 0, 0, false);
    const pr = s.probe(&key, &.{});
    const point = pr.durable orelse return error.NoPoint;
    const r = try s.restoreBegin(point, &key);
    try std.testing.expectEqual(@as(u64, 100 + 32 + 4 * 32), s.restoring_bytes); // pointBytes + its key's ids
    try std.testing.expect(s.held + s.restoring_bytes <= s.budget);
    var k: usize = 0;
    while (k < 100) : (k += 1) switch (s.restorePoll(r, std.time.ns_per_ms)) {
        .pending => continue,
        .adopted => break,
        .dropped, .failed => return error.NotAdopted,
    };
    try std.testing.expectEqual(@as(u64, 0), s.restoring_bytes);
    try std.testing.expectEqual(@as(u64, 100 + 32 + 8 * 32), s.held); // pointBytes + tokens and last (the key twice)
    try std.testing.expectEqual(@as(u64, 8 * 32), s.token_held);
    try std.testing.expect(s.held <= s.budget);
}
