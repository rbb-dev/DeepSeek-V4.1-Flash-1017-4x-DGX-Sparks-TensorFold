//! The KV pool sized from the free memory the operator leaves: what each rank fits, the vote's min, the refusals' words, the
//! ledger's sum, and the re-fit a dry run makes.
const std = @import("std");
const kv = @import("kv_pool.zig");

const gib: u64 = 1 << 30;

/// A pool's bytes as a Spark's caches take them: a fixed part (the slots' windows) and 890 bytes a position.
const Linear = struct {
    fixed: u64 = 1 * gib,
    per: u64 = 890,
    pub fn bytes(l: Linear, positions: usize) usize {
        return @intCast(l.fixed + l.per * positions);
    }
};

fn part(rank: u32, free_gib: u64, target_gib: u64, growth_gib: u64, floor: u64, lin: Linear) kv.Part {
    var p = kv.Part.init(rank);
    p.free_bytes = free_gib * gib;
    p.total_bytes = 121 * gib;
    p.target_bytes = target_gib * gib;
    p.growth_bytes = growth_gib * gib;
    p.floor = floor;
    p.asked = floor;
    p.fit = kv.fit(p.free_bytes, p.target_bytes, p.growth_bytes, lin);
    p.floor_bytes = lin.bytes(@intCast(floor));
    p.b_lo = lin.bytes(kv.pool_step);
    p.b_hi = lin.bytes(17 * kv.pool_step);
    p.bytes_per_token = lin.per;
    return p;
}

const ctx: u64 = 1_048_576;

test "fit: the largest multiple of the step whose bytes fit what is free less the target, the growth and the arena's slack" {
    const lin: Linear = .{};
    // 40 GiB free, 8 kept, 6 for what follows, 0.25 for the arena's block: 25.75 GiB for the pool
    const avail = 40 * gib - 8 * gib - 6 * gib - kv.arena_slack_bytes;
    const f = kv.fit(40 * gib, 8 * gib, 6 * gib, lin);
    try std.testing.expect(f % kv.pool_step == 0);
    try std.testing.expect(lin.bytes(f) <= avail);
    try std.testing.expect(lin.bytes(f + kv.pool_step) > avail); // the next step does not fit
    // nothing fits: less free than the target, the growth and the slack together
    try std.testing.expectEqual(@as(usize, 0), kv.fit(10 * gib, 8 * gib, 6 * gib, lin));
    // a pool of one step is the least fit reports
    try std.testing.expectEqual(@as(usize, kv.pool_step), kv.fit(lin.bytes(kv.pool_step) + 6 * gib + kv.arena_slack_bytes + 8 * gib, 8 * gib, 6 * gib, lin));
    // more free is never fewer positions
    var prev: usize = 0;
    var g: u64 = 20;
    while (g <= 100) : (g += 5) {
        const now = kv.fit(g * gib, 8 * gib, 6 * gib, lin);
        try std.testing.expect(now >= prev);
        prev = now;
    }
    // no pool passes what a position index holds
    try std.testing.expect(kv.fit(1 << 50, 0, 0, lin) <= kv.max_pool_positions);
}

test "with no free target the vote asks nothing: the pool is the one the settings named, on every rank" {
    const lin: Linear = .{};
    var t: [4]kv.Part = undefined;
    for (&t, 0..) |*p, r| p.* = part(@intCast(r), 30, 0, 6, ctx, lin);
    const d = kv.decide(&t);
    try std.testing.expectEqual(kv.Verdict.off, d.verdict);
    try std.testing.expectEqual(@as(usize, ctx), d.pool);
}

test "with a free target the pool is the smallest any rank fits, and the rank that fits least is named" {
    const lin: Linear = .{};
    var t = [_]kv.Part{ part(0, 60, 8, 6, ctx, lin), part(1, 50, 8, 6, ctx, lin), part(2, 55, 8, 6, ctx, lin), part(3, 58, 8, 6, ctx, lin) };
    const d = kv.decide(&t);
    try std.testing.expectEqual(kv.Verdict.auto, d.verdict);
    try std.testing.expectEqual(@as(u32, 1), d.limiting_rank);
    try std.testing.expectEqual(t[1].fit, d.pool);
    try std.testing.expect(d.pool > ctx); // past the context: the extra holds more kept conversations
    try std.testing.expect(d.pool % kv.pool_step == 0);
    for (t) |p| try std.testing.expect(d.pool <= p.fit);
    // exactly the context fits: the pool is the context, never below it
    var tight = [_]kv.Part{ part(0, 60, 8, 6, ctx, lin), part(1, 60, 8, 6, ctx, lin), part(2, 60, 8, 6, ctx, lin), part(3, 60, 8, 6, ctx, lin) };
    tight[2].free_bytes = lin.bytes(ctx) + 8 * gib + 6 * gib + kv.arena_slack_bytes;
    tight[2].fit = kv.fit(tight[2].free_bytes, tight[2].target_bytes, tight[2].growth_bytes, lin);
    const dt = kv.decide(&tight);
    try std.testing.expectEqual(kv.Verdict.auto, dt.verdict);
    try std.testing.expectEqual(@as(usize, ctx), dt.pool);
}

test "a pool the context does not fit in is refused, naming the shortfall, the rank and the ledger's largest lines" {
    const lin: Linear = .{};
    var t = [_]kv.Part{ part(0, 60, 8, 6, ctx, lin), part(1, 60, 8, 6, ctx, lin), part(2, 12, 8, 6, ctx, lin), part(3, 60, 8, 6, ctx, lin) };
    // rank 2's ledger: three lines, the weights the largest
    t[2].n = 3;
    t[2].entries[0] = .{ .feature = @backingInt(kv.Feature.transport), .device = 3 * gib, .host = 0 };
    t[2].entries[1] = .{ .feature = @backingInt(kv.Feature.weights), .device = 80 * gib, .host = 0 };
    t[2].entries[2] = .{ .feature = @backingInt(kv.Feature.chunk), .device = 5 * gib, .host = 0 };
    const d = kv.decide(&t);
    try std.testing.expectEqual(kv.Verdict.refused, d.verdict);
    try std.testing.expectEqual(kv.Refusal.below_context, d.refusal.?);
    try std.testing.expectEqual(@as(u32, 2), d.limiting_rank);
    var buf: [1200]u8 = undefined;
    const text = kv.refusalText(&buf, &t, d);
    try std.testing.expect(std.mem.indexOf(u8, text, "rank 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "1,048,576") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "GiB short") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "TF_DS_KV_FREE_TARGET_GIB") != null);
    const w = std.mem.indexOf(u8, text, kv.featureInfo(.weights).label).?;
    const tr = std.mem.indexOf(u8, text, kv.featureInfo(.transport).label).?;
    try std.testing.expect(w < tr); // largest first
    // how short: the context's bytes and everything kept back, less what rank 2 has free
    const need = lin.bytes(ctx) + 8 * gib + 6 * gib + kv.arena_slack_bytes;
    var short: [40]u8 = undefined;
    const want = std.fmt.bufPrint(&short, "{d:.2} GiB short", .{@as(f64, @floatFromInt(need - 12 * gib)) / @as(f64, gib)}) catch unreachable;
    try std.testing.expect(std.mem.indexOf(u8, text, want) != null);
}

test "ranks that were given different settings do not vote: the refusal names the setting and the flag a follower takes" {
    const lin: Linear = .{};
    var t = [_]kv.Part{ part(0, 60, 8, 6, ctx, lin), part(1, 60, 8, 6, ctx, lin), part(2, 60, 4, 6, ctx, lin), part(3, 60, 8, 6, ctx, lin) };
    const d = kv.decide(&t);
    try std.testing.expectEqual(kv.Verdict.refused, d.verdict);
    try std.testing.expectEqual(kv.Refusal.settings_differ, d.refusal.?);
    var buf: [800]u8 = undefined;
    const text = kv.refusalText(&buf, &t, d);
    try std.testing.expect(std.mem.indexOf(u8, text, "rank 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "--kv-free-target-gib") != null);
    // another growth, another floor
    var g = [_]kv.Part{ part(0, 60, 8, 6, ctx, lin), part(1, 60, 8, 7, ctx, lin), part(2, 60, 8, 6, ctx, lin), part(3, 60, 8, 6, ctx, lin) };
    for (&g) |*p| p.growth_set = 1; // an operator's number on every rank: the ranks must agree on it
    try std.testing.expectEqual(kv.Refusal.settings_differ, kv.decide(&g).refusal.?);
    // the defaults are per role (rank 0 holds the server): different growth without an operator's number is no disagreement
    var role = [_]kv.Part{ part(0, 60, 8, 9, ctx, lin), part(1, 60, 8, 7, ctx, lin), part(2, 60, 8, 7, ctx, lin), part(3, 60, 8, 7, ctx, lin) };
    try std.testing.expectEqual(kv.Verdict.auto, kv.decide(&role).verdict);
    // one rank given the number and the others not is a disagreement
    role[2].growth_set = 1;
    try std.testing.expectEqual(kv.Refusal.settings_differ, kv.decide(&role).refusal.?);
    var f = [_]kv.Part{ part(0, 60, 8, 6, ctx, lin), part(1, 60, 8, 6, ctx, lin), part(2, 60, 8, 6, ctx / 2, lin), part(3, 60, 8, 6, ctx, lin) };
    try std.testing.expectEqual(kv.Refusal.settings_differ, kv.decide(&f).refusal.?);
    // in the off mode a follower's own pool may be larger (the recipe's rule: it holds rank 0's); only the free target decides here
    var off = [_]kv.Part{ part(0, 60, 0, 6, ctx, lin), part(1, 60, 0, 6, ctx, lin), part(2, 60, 0, 6, ctx, lin), part(3, 60, 0, 6, ctx, lin) };
    off[1].asked = 2 * ctx;
    try std.testing.expectEqual(kv.Verdict.off, kv.decide(&off).verdict);
    try std.testing.expectEqual(@as(usize, ctx), kv.decide(&off).pool); // rank 0's
}

test "refit: what a different free target would give each rank, from the numbers the start voted on" {
    const lin: Linear = .{};
    var t = [_]kv.Part{ part(0, 60, 8, 6, ctx, lin), part(1, 50, 8, 6, ctx, lin), part(2, 55, 8, 6, ctx, lin), part(3, 58, 8, 6, ctx, lin) };
    var out: [4]usize = undefined;
    const same = kv.refit(&t, 8 * gib, &out);
    try std.testing.expectEqual(kv.decide(&t).pool, same.pool); // the same target gives the pool the start chose
    const more = kv.refit(&t, 12 * gib, &out);
    try std.testing.expect(more.pool < same.pool);
    for (out, t) |o, p| try std.testing.expect(o == 0 or o <= p.fit); // a bigger target never gives a rank more
    const exact = [_]usize{ kv.fit(t[0].free_bytes, 12 * gib, t[0].growth_bytes, lin), kv.fit(t[1].free_bytes, 12 * gib, t[1].growth_bytes, lin), kv.fit(t[2].free_bytes, 12 * gib, t[2].growth_bytes, lin), kv.fit(t[3].free_bytes, 12 * gib, t[3].growth_bytes, lin) };
    for (out, exact) |o, e| try std.testing.expect(@abs(@as(i64, @intCast(o)) - @as(i64, @intCast(e))) <= kv.pool_step); // within a step of the exact fit
    // a target no rank can keep and still hold the context
    const none = kv.refit(&t, 100 * gib, &out);
    try std.testing.expectEqual(kv.Verdict.refused, none.verdict);
}

test "the ledger's lines and the remainder add up to what the device holds; the part carries them" {
    var l: kv.Ledger = .{};
    l.record(@src(), .weights, 80 * gib, 0);
    l.record(@src(), .transport, 3 * gib, 1 << 20);
    l.record(@src(), .chunk, 5 * gib, 0);
    l.close(@src(), 100 * gib); // 100 GiB are in use: the rest is everything this ledger does not name
    try std.testing.expectEqual(@as(u64, 100 * gib), l.deviceTotal());
    try std.testing.expectEqual(@as(u32, 4), l.n);
    try std.testing.expectEqual(kv.Feature.other, @as(kv.Feature, @fromBackingInt(@intCast(l.entries[3].feature))));
    try std.testing.expectEqual(@as(u64, 12 * gib), l.entries[3].device);
    // the start named more than the device holds (rounding in a block): the remainder is zero, never negative
    var over: kv.Ledger = .{};
    over.record(@src(), .weights, 10 * gib, 0);
    over.close(@src(), 9 * gib);
    try std.testing.expectEqual(@as(u64, 10 * gib), over.deviceTotal());
    var p = kv.Part.init(2);
    l.fill(&p);
    try std.testing.expectEqual(@as(u32, 4), p.n);
    try std.testing.expectEqual(@as(u64, 3 * gib), p.entries[1].device);
    // every site is the call's file and line, fixed at build
    try std.testing.expect(std.mem.endsWith(u8, l.sites[0].file, "kv_pool_test.zig"));
    try std.testing.expect(l.sites[0].line > 0 and l.sites[1].line == l.sites[0].line + 1);
    // a full ledger drops the line it cannot hold and says so
    var full: kv.Ledger = .{};
    for (0..kv.max_entries + 3) |_| full.record(@src(), .chunk, 1, 0);
    try std.testing.expectEqual(@as(u32, kv.max_entries), full.n);
    try std.testing.expectEqual(@as(u32, 3), full.dropped);
}

test "a part is one fixed frame the star carries" {
    try std.testing.expect(@sizeOf(kv.Part) <= 16 * 1024);
    try std.testing.expectEqual(@as(usize, 0), @sizeOf(kv.Part) % 8);
    var p = kv.Part.init(3);
    try std.testing.expectEqual(kv.Part.tag, p.magic);
    try std.testing.expectEqual(@as(u32, 3), p.rank);
    p.magic = 0;
    var t = [_]kv.Part{ kv.Part.init(0), p };
    try std.testing.expectEqual(kv.Refusal.bad_part, kv.decide(&t).refusal.?); // a frame that is not a part ends the start, named
}

test "every feature has a label, and the options a feature answers to name rows the settings table has" {
    const settings = @import("settings.zig");
    for (std.enums.values(kv.Feature)) |f| {
        const info = kv.featureInfo(f);
        try std.testing.expect(info.label.len > 0);
        if (info.option.len > 0) try std.testing.expect(settings.byEnv(info.option) != null);
    }
}

test "the free target is a positive number of GiB, the growth a whole number of MiB; anything else is refused by name" {
    try std.testing.expectEqual(@as(u64, 8 * gib), try kv.parseTargetGib("8"));
    try std.testing.expectEqual(@as(u64, gib + gib / 2), try kv.parseTargetGib("1.5"));
    for ([_][]const u8{ "", "0", "-1", "x", "inf", "nan", "2000", "8 GiB" }) |bad| try std.testing.expectError(error.BadKvTarget, kv.parseTargetGib(bad));
    try std.testing.expectEqual(@as(u64, 6144 << 20), try kv.parseGrowthMib("6144"));
    try std.testing.expectEqual(@as(u64, 0), try kv.parseGrowthMib("0")); // none kept: the operator's word
    for ([_][]const u8{ "", "-1", "1.5", "x", "99999999999" }) |bad| try std.testing.expectError(error.BadKvGrowth, kv.parseGrowthMib(bad));
}

test "a follower given a free target and no --pool holds the context, never its default: the pool it starts from is the floor the vote lifts" {
    try std.testing.expectEqual(@as(usize, ctx), kv.startPool(1 << 18, ctx, 8 * gib)); // the default pool 262,144 < the context
    try std.testing.expectEqual(@as(usize, 2 * ctx), kv.startPool(2 * ctx, ctx, 8 * gib)); // a pool named stays (the vote's floor is the context)
    try std.testing.expectEqual(@as(usize, 1 << 18), kv.startPool(1 << 18, ctx, 0)); // no target: exactly as given, as before
    try std.testing.expectEqual(@as(usize, 1 << 18), kv.startPool(1 << 18, 0, 8 * gib)); // no context asked: nothing to hold
}

test "the growth left for what follows the pool is per role, MEASURED: rank 0 holds the server and the admin threads, a follower less" {
    try std.testing.expect(kv.defaultGrowth(0) > kv.defaultGrowth(1));
    try std.testing.expectEqual(kv.defaultGrowth(1), kv.defaultGrowth(3));
    // what the Sparks showed at a 14 GiB target (2026-11-09, release 3df841574): free after the start 10.9 on rank 0, 12.6 on a worker,
    // so the allowance covers the 6 GiB assumed plus 3.1 and 1.3 GiB more
    try std.testing.expect(kv.defaultGrowth(0) >= 9 * gib + gib / 10);
    try std.testing.expect(kv.defaultGrowth(1) >= 7 * gib + 3 * gib / 10);
    try std.testing.expect(std.mem.indexOf(u8, kv.growth_measured, "2026-11-09") != null);
    // the operator's number wins on every rank, and a rank is told whose number it used
    try std.testing.expectEqual(@as(u64, 5 * gib), kv.growthFor(2, true, 5 * gib));
    try std.testing.expectEqual(kv.defaultGrowth(2), kv.growthFor(2, false, 0));
    try std.testing.expectEqual(kv.defaultGrowth(0), kv.growthFor(0, false, 123));
}

// ---- the live resize's decision ----------------------------------------------------------------------------------

fn rpart(rank: u32, cur: u64, fit_pos: u64, in_use: u64) kv.ResizePart {
    return .{ .magic = kv.ResizePart.tag, .rank = rank, .vmm = 1, .restoring = 0, .cur = cur, .fit = fit_pos, .ceiling = 1 << 30, .in_use = in_use, .free_bytes = 0, .mapped_bytes = 0, .target_bytes = 8 << 30 };
}

test "the growth grows with the pass: the pool, not the free memory, pays for larger prompt passes" {
    // at the measured pass and below: the role's growth, unchanged
    try std.testing.expectEqual(kv.defaultGrowth(0), kv.growthForPass(0, false, 0, 4096));
    try std.testing.expectEqual(kv.defaultGrowth(2), kv.growthForPass(2, false, 0, 2048));
    // 8,192-row passes: one more 4,096 share, by role
    try std.testing.expectEqual(kv.defaultGrowth(0) + kv.growth_per_pass4096_rank0_bytes, kv.growthForPass(0, false, 0, 8192));
    try std.testing.expectEqual(kv.defaultGrowth(1) + kv.growth_per_pass4096_follower_bytes, kv.growthForPass(1, false, 0, 8192));
    // a part of a share counts whole: 6,144 rows take one share
    try std.testing.expectEqual(kv.defaultGrowth(3) + kv.growth_per_pass4096_follower_bytes, kv.growthForPass(3, false, 0, 6144));
    // the measured shares cover the shortfall (2.58 GiB on rank 0, 2.36 on a worker)
    try std.testing.expect(kv.growth_per_pass4096_rank0_bytes >= 2 * gib + 58 * gib / 100);
    try std.testing.expect(kv.growth_per_pass4096_follower_bytes >= 2 * gib + 36 * gib / 100);
    // the operator's number is the whole growth, whatever the pass
    try std.testing.expectEqual(@as(u64, 5 * gib), kv.growthForPass(0, true, 5 * gib, 8192));
}

const rctx: usize = 1_048_576;

test "live resize: every rank decides the same from the same table, the smallest fit wins" {
    const step = kv.pool_step;
    const cur: u64 = 20 * step;
    var t = [4]kv.ResizePart{ rpart(0, cur, 40 * step, 0), rpart(1, cur, 33 * step, 0), rpart(2, cur, 52 * step, 0), rpart(3, cur, 41 * step, 0) };
    const d = kv.decideResize(&t, rctx);
    try std.testing.expectEqual(kv.ResizeVerdict.grow, d.verdict);
    try std.testing.expectEqual(@as(usize, 33 * step), d.positions);
    try std.testing.expectEqual(@as(u32, 1), d.limiting_rank);
    // a rank's ceiling bounds the grow (its reservation or the tier's structures)
    t[2].ceiling = 30 * step;
    try std.testing.expectEqual(@as(usize, 30 * step), kv.decideResize(&t, rctx).positions);
}

test "live resize: a shrink goes as far as memory asks, stops at the extents in use and says so, and never goes under the context" {
    const step = kv.pool_step;
    const cur: u64 = 100 * step;
    // memory asks for 60 steps, nothing is in use: shrink
    var t = [4]kv.ResizePart{ rpart(0, cur, 60 * step, 0), rpart(1, cur, 70 * step, 0), rpart(2, cur, 70 * step, 0), rpart(3, cur, 70 * step, 0) };
    var d = kv.decideResize(&t, rctx);
    try std.testing.expectEqual(kv.ResizeVerdict.shrink, d.verdict);
    try std.testing.expectEqual(@as(usize, 60 * step), d.positions);
    // a stream holds an extent ending at 75 steps + 1 position: the pool stops at 76 steps (the tail is not free)
    t[0].in_use = 75 * step + 1;
    d = kv.decideResize(&t, rctx);
    try std.testing.expectEqual(kv.ResizeVerdict.shrink_partial, d.verdict);
    try std.testing.expectEqual(@as(usize, 76 * step), d.positions);
    try std.testing.expectEqual(@as(usize, 60 * step), d.wanted);
    // an extent in use up to the pool's end: nothing can be given back
    t[0].in_use = cur;
    d = kv.decideResize(&t, rctx);
    try std.testing.expectEqual(kv.ResizeVerdict.blocked_by_use, d.verdict);
    try std.testing.expectEqual(@as(usize, 100 * step), d.positions);
    // less memory than the context needs: refused by name, the pool stays
    t[0].in_use = 0;
    t[3].fit = rctx - step;
    d = kv.decideResize(&t, rctx);
    try std.testing.expectEqual(kv.ResizeVerdict.refused_context, d.verdict);
    try std.testing.expectEqual(@as(usize, 100 * step), d.positions);
    try std.testing.expectEqual(@as(u32, 3), d.limiting_rank);
    // the context is also the shrink's floor when extents are small
    t[3].fit = rctx;
    try std.testing.expectEqual(@as(usize, rctx), kv.decideResize(&t, rctx).positions);
}

test "live resize: a disk restore loading blocks every shrink, however free the tail, and never a grow" {
    const step = kv.pool_step;
    const cur: u64 = 100 * step;
    var t = [4]kv.ResizePart{ rpart(0, cur, 60 * step, 0), rpart(1, cur, 70 * step, 0), rpart(2, cur, 70 * step, 0), rpart(3, cur, 70 * step, 0) };
    t[0].restoring = 1;
    var d = kv.decideResize(&t, rctx);
    try std.testing.expectEqual(kv.ResizeVerdict.blocked_by_use, d.verdict);
    try std.testing.expectEqual(@as(usize, 100 * step), d.positions);
    t[0].restoring = 0;
    try std.testing.expectEqual(kv.ResizeVerdict.shrink, kv.decideResize(&t, rctx).verdict); // the same table without the restore shrinks
    for (&t) |*p| p.fit = 130 * step;
    t[0].restoring = 2;
    d = kv.decideResize(&t, rctx);
    try std.testing.expectEqual(kv.ResizeVerdict.grow, d.verdict); // a grow only adds rows past every extent
    try std.testing.expectEqual(@as(usize, 130 * step), d.positions);
}

test "live resize: no change, and parts that do not describe one pool are refused" {
    const step = kv.pool_step;
    const cur: u64 = 20 * step;
    var t = [4]kv.ResizePart{ rpart(0, cur, 20 * step + 5, 0), rpart(1, cur, 25 * step, 0), rpart(2, cur, 25 * step, 0), rpart(3, cur, 25 * step, 0) };
    try std.testing.expectEqual(kv.ResizeVerdict.same, kv.decideResize(&t, rctx).verdict); // fits 20 steps and a few positions: the step floor
    t[1].cur = 21 * step; // a rank with another pool: the ranks have diverged
    try std.testing.expectEqual(kv.ResizeVerdict.refused_disagree, kv.decideResize(&t, rctx).verdict);
    t[1].cur = cur;
    t[2].target_bytes += 1; // another target
    try std.testing.expectEqual(kv.ResizeVerdict.refused_disagree, kv.decideResize(&t, rctx).verdict);
    t[2].target_bytes = t[0].target_bytes;
    t[3].magic = 0; // not a part of this kind
    try std.testing.expectEqual(kv.ResizeVerdict.refused_disagree, kv.decideResize(&t, rctx).verdict);
    t[3].magic = kv.ResizePart.tag;
    t[1].vmm = 0; // a rank whose pool is not in virtual memory: it took part, and nothing moves, by name
    const nv = kv.decideResize(&t, rctx);
    try std.testing.expectEqual(kv.ResizeVerdict.refused_no_vmm, nv.verdict);
    try std.testing.expectEqual(@as(u32, 1), nv.limiting_rank);
    try std.testing.expectEqual(@as(usize, cur), nv.positions);
}

test "live resize: a rank that ran out of memory is named, the first by rank, and an error elsewhere is its own verdict" {
    const step = kv.pool_step;
    var t = [4]kv.ResizePart{ rpart(0, 20 * step, 30 * step, 0), rpart(1, 20 * step, 30 * step, 0), rpart(2, 20 * step, 30 * step, 0), rpart(3, 20 * step, 30 * step, 0) };
    const d = kv.decideResize(&t, rctx);
    try std.testing.expectEqual(kv.ResizeVerdict.grow, d.verdict);
    var o = kv.resizeOutcome(d, &.{ 0, 0, 0, 0 });
    try std.testing.expectEqual(kv.ResizeVerdict.grow, o.verdict); // all ranks moved: the decision stands
    o = kv.resizeOutcome(d, &.{ 0, 0, kv.ballot_memory, kv.ballot_memory });
    try std.testing.expectEqual(kv.ResizeVerdict.refused_memory, o.verdict);
    try std.testing.expectEqual(@as(u32, 2), o.rank);
    o = kv.resizeOutcome(d, &.{ 0, kv.ballot_error, kv.ballot_memory, 0 }); // an out-of-memory rank outranks another failure
    try std.testing.expectEqual(kv.ResizeVerdict.refused_memory, o.verdict);
    try std.testing.expectEqual(@as(u32, 2), o.rank);
    o = kv.resizeOutcome(d, &.{ 0, 0, 0, kv.ballot_error });
    try std.testing.expectEqual(kv.ResizeVerdict.refused_error, o.verdict);
    try std.testing.expectEqual(@as(u32, 3), o.rank);
}

test "live resize: fitIn is fit without the start's allowances" {
    const Lin = struct {
        pub fn bytes(_: @This(), positions: usize) usize {
            return 1_000_000 + 890 * positions;
        }
    };
    const avail: u64 = 50 << 30;
    const n = kv.fitIn(avail, Lin{});
    try std.testing.expect(n % kv.pool_step == 0);
    try std.testing.expect((Lin{}).bytes(n) <= avail and (Lin{}).bytes(n + kv.pool_step) > avail);
    try std.testing.expectEqual(n, kv.fit(avail + (3 << 30) + kv.arena_slack_bytes, 1 << 30, 2 << 30, Lin{})); // the start's rule adds its keeps
    try std.testing.expectEqual(@as(usize, 0), kv.fitIn(10, Lin{}));
}
