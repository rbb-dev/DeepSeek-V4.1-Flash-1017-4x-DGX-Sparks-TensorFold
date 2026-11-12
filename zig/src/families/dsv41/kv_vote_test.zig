//! The KV pool's vote over the host star: four ranks each offer their part, every rank reads the same table and decides the same
//! pool, and a start that cannot hold the context ends on every rank with the same words.
const std = @import("std");
const kv = @import("engine_api").kv_pool;
const star = @import("star.zig");

const gib: u64 = 1 << 30;
const ctx: u64 = 1_048_576;

const Lin = struct {
    pub fn bytes(_: Lin, positions: usize) usize {
        return @intCast(gib + 890 * positions);
    }
};

/// Rank g's part with `free_gib` free and the same settings as the others.
fn offer(g: u32, free_gib: u64, target_gib: u64) kv.Part {
    const lin: Lin = .{};
    var p = kv.Part.init(g);
    p.free_bytes = free_gib * gib;
    p.total_bytes = 121 * gib;
    p.target_bytes = target_gib * gib;
    p.growth_bytes = 6 * gib;
    p.floor = ctx;
    p.asked = ctx;
    p.fit = if (target_gib != 0) kv.fit(p.free_bytes, p.target_bytes, p.growth_bytes, lin) else 0;
    p.floor_bytes = lin.bytes(ctx);
    p.b_lo = lin.bytes(kv.pool_step);
    p.b_hi = lin.bytes(17 * kv.pool_step);
    p.bytes_per_token = 890;
    p.n = 1;
    p.entries[0] = .{ .feature = @backingInt(kv.Feature.weights), .device = (80 - @as(u64, g)) * gib, .host = 0 };
    return p;
}

fn vote(s: star.Star, mine: kv.Part, out: *kv.Decision, table: *[4]kv.Part) !void {
    try s.allGather(std.mem.asBytes(&mine), std.mem.sliceAsBytes(table[0..4]));
    out.* = kv.decide(table[0..4]);
}

test "the four ranks read one table and take one pool: the smallest any fits" {
    try star.runRanks(struct {
        fn f(s: star.Star, g: u32) !void {
            const free = [4]u64{ 60, 50, 55, 58 };
            var d: kv.Decision = undefined;
            var table: [4]kv.Part = undefined;
            try vote(s, offer(g, free[g], 8), &d, &table);
            try std.testing.expectEqual(kv.Verdict.auto, d.verdict);
            try std.testing.expectEqual(@as(u32, 1), d.limiting_rank);
            try std.testing.expectEqual(table[1].fit, d.pool);
            for (table, 0..) |p, r| try std.testing.expectEqual(@as(u32, @intCast(r)), p.rank);
            // a second vote over the same links: the frames stay paired
            var d2: kv.Decision = undefined;
            try vote(s, offer(g, free[g], 0), &d2, &table);
            try std.testing.expectEqual(kv.Verdict.off, d2.verdict);
        }
    }.f);
}

test "a pool the context does not fit in ends the start on every rank with the same words" {
    try star.runRanks(struct {
        fn f(s: star.Star, g: u32) !void {
            const free = [4]u64{ 60, 60, 12, 60 };
            var d: kv.Decision = undefined;
            var table: [4]kv.Part = undefined;
            try vote(s, offer(g, free[g], 8), &d, &table);
            try std.testing.expectEqual(kv.Verdict.refused, d.verdict);
            try std.testing.expectEqual(@as(u32, 2), d.limiting_rank);
            var buf: [1600]u8 = undefined;
            const text = kv.refusalText(&buf, &table, d);
            try std.testing.expect(std.mem.indexOf(u8, text, "rank 2") != null and std.mem.indexOf(u8, text, "GiB short") != null);
            // the words are the same on every rank: a hash of them is gathered back to the ranks and compared
            var mine: [8]u8 = undefined;
            std.mem.writeInt(u64, &mine, std.hash.Wyhash.hash(0, text), .little);
            var all: [32]u8 = undefined;
            try s.allGather(&mine, &all);
            for (1..4) |r| try std.testing.expectEqualSlices(u8, all[0..8], all[r * 8 ..][0..8]);
        }
    }.f);
}

test "a rank given other settings is named by every rank before any pool is taken" {
    try star.runRanks(struct {
        fn f(s: star.Star, g: u32) !void {
            var d: kv.Decision = undefined;
            var table: [4]kv.Part = undefined;
            try vote(s, offer(g, 60, if (g == 3) 4 else 8), &d, &table);
            try std.testing.expectEqual(kv.Verdict.refused, d.verdict);
            try std.testing.expectEqual(kv.Refusal.settings_differ, d.refusal.?);
            try std.testing.expectEqual(@as(u32, 3), d.limiting_rank);
        }
    }.f);
}
