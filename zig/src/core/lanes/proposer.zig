//! Copy proposals from a stream's own context (Python SuffixLookupProposer) behind a small proposer interface.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// What the round loop asks of a proposer: a copied continuation, its evidence, and how it fared.
pub const Proposer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        propose: *const fn (ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32,
        last_match: *const fn (ptr: *anyopaque) i64,
        observe: ?*const fn (ptr: *anyopaque, proposed: i64, accepted: i64) void = null,
    };

    /// Tokens valid until the next `propose`.
    pub fn propose(p: Proposer, context: []const u32, max_draft: i64) ![]const u32 {
        return p.vtable.propose(p.ptr, context, max_draft);
    }

    pub fn lastMatch(p: Proposer) i64 {
        return p.vtable.last_match(p.ptr);
    }

    pub fn observe(p: Proposer, proposed: i64, accepted: i64) void {
        if (p.vtable.observe) |f| f(p.ptr, proposed, accepted);
    }
};

/// What the copy proposer costs the engine thread (it runs there, between rounds, with every GPU idle): the calls, the candidate
/// positions each scan walked, the time of a call and of the index's growth. One engine thread writes it; `line` reads and clears it.
pub const Meter = struct {
    calls: u64 = 0,
    scans: u64 = 0, // calls that walked the candidate positions
    cached: u64 = 0, // calls that took a scan's answer from the call before (same context)
    positions: u64 = 0,
    positions_max: u64 = 0,
    hist: [hist_n]u64 = @splat(0), // the positions of a scan, by log2 (0, 1, 2-3, 4-7, ...)
    ticks: u64 = 0,
    ticks_max: u64 = 0,
    extended: u64 = 0, // tokens the index took in a propose
    extend_ticks: u64 = 0,
    extend_ticks_max: u64 = 0,
    warmed: u64 = 0, // tokens the index took between prompt chunks (warm)
    warm_ticks: u64 = 0,
    warm_ticks_max: u64 = 0,

    pub const hist_n = 14;

    pub fn bucket(n: u64) usize {
        return if (n == 0) 0 else @min(hist_n - 1, @as(usize, 64 - @clz(n)));
    }

    fn scanned(m: *Meter, n: u64) void {
        m.scans += 1;
        m.positions += n;
        m.positions_max = @max(m.positions_max, n);
        m.hist[bucket(n)] += 1;
    }

    /// The span's line when it has calls, else null; clears the span.
    pub fn line(m: *Meter, buf: []u8) ?[]const u8 {
        if (m.calls == 0 and m.warmed == 0) return null;
        const us = struct {
            fn of(t: u64) f64 {
                return @as(f64, @floatFromInt(t)) * 1e6 / @as(f64, @floatFromInt(@max(1, tickRate())));
            }
        }.of;
        var h: [hist_n * 6]u8 = undefined;
        var w: std.Io.Writer = .fixed(&h);
        for (m.hist, 0..) |c, i| w.print("{s}{d}", .{ if (i == 0) "" else ",", c }) catch break;
        const calls: f64 = @floatFromInt(@max(1, m.calls));
        const out = std.fmt.bufPrint(buf, "copy proposer: {d} calls ({d} walked the positions, {d} reused the scan before), positions a walk mean {d:.1} max {d} (log2 buckets {s}), call mean {d:.2} us max {d:.1} us, index grew {d} tokens in a call ({d:.1} ms, longest {d:.1} us), warmed {d} tokens between chunks ({d:.1} ms, longest {d:.1} us)", .{
            m.calls,                   m.scans,              m.cached,   @as(f64, @floatFromInt(m.positions)) / @as(f64, @floatFromInt(@max(1, m.scans))), m.positions_max,        w.buffered(),
            us(m.ticks) / calls,       us(m.ticks_max),      m.extended, us(m.extend_ticks) / 1000.0,                                                      us(m.extend_ticks_max), m.warmed,
            us(m.warm_ticks) / 1000.0, us(m.warm_ticks_max),
        }) catch buf[0..0];
        m.* = .{};
        return out;
    }
};

/// The engine thread's copy-proposer meter (the proposers all run on it).
pub var meter: Meter = .{};

/// A clock for the meter: aarch64's virtual counter (a register read), else the monotonic clock in ns.
fn ticks() u64 {
    if (@import("builtin").cpu.arch == .aarch64) return asm volatile ("mrs %[v], cntvct_el0"
        : [v] "=r" (-> u64),
    );
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn tickRate() u64 {
    if (@import("builtin").cpu.arch == .aarch64) return asm volatile ("mrs %[v], cntfrq_el0"
        : [v] "=r" (-> u64),
    );
    return std.time.ns_per_s;
}

pub const max_ngram = 8;
const Key = [max_ngram]u32;

pub const Params = struct {
    ngram: usize = 3,
    min_match: i64 = 6,
    max_extension: i64 = 64,
    silence_rounds: i64 = 16,
    window: usize = 4,
    confident_match: i64 = 24,
    /// A call at the context the call before saw takes that call's scan (the answer depends on the context alone; every other effect of
    /// the call is made again). False: every call walks the positions (the equivalence test's reference).
    scan_memo: bool = true,
};

/// Continuations backed by enough matching context; silent for a while after a run of rejected proposals.
pub const SuffixLookup = struct {
    gpa: Allocator,
    params: Params,
    index: std.AutoHashMapUnmanaged(Key, std.ArrayList(u32)) = .empty,
    indexed: usize = 0,
    last_key: Key = undefined,
    recent: std.ArrayList(i64) = .empty,
    silent_for: i64 = 0,
    out: std.ArrayList(u32) = .empty,
    proposals: i64 = 0,
    proposed_tokens: i64 = 0,
    judged_tokens: i64 = 0,
    accepted_tokens: i64 = 0,
    silenced_rounds: i64 = 0,
    last_match: i64 = 0,
    last_confident: bool = false,
    memo: ?Scan = null, // the last scan's answer (`scan`)
    gen: u32 = 0, // counts the index's rebuilds: a memo of another index is not taken

    pub fn init(gpa: Allocator, params: Params) !SuffixLookup {
        if (params.ngram < 1 or params.ngram > max_ngram) return error.BadNgram;
        if (params.min_match < @as(i64, @intCast(params.ngram))) return error.MinMatchBelowNgram;
        var p = params;
        p.window = @max(1, p.window);
        return .{ .gpa = gpa, .params = p };
    }

    pub fn deinit(s: *SuffixLookup) void {
        s.clearIndex();
        s.index.deinit(s.gpa);
        s.recent.deinit(s.gpa);
        s.out.deinit(s.gpa);
    }

    pub fn proposer(s: *SuffixLookup) Proposer {
        return .{ .ptr = s, .vtable = &vtable };
    }

    pub const vtable: Proposer.VTable = .{ .propose = proposeFn, .last_match = lastMatchFn, .observe = observeFn };

    fn proposeFn(ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        return s.propose(context, max_draft);
    }

    fn lastMatchFn(ptr: *anyopaque) i64 {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        return s.last_match;
    }

    fn observeFn(ptr: *anyopaque, proposed: i64, accepted: i64) void {
        const s: *SuffixLookup = @ptrCast(@alignCast(ptr));
        s.observe(proposed, accepted) catch {};
    }

    fn clearIndex(s: *SuffixLookup) void {
        var it = s.index.valueIterator();
        while (it.next()) |list| list.deinit(s.gpa);
        s.index.clearRetainingCapacity();
        s.gen +%= 1;
        s.memo = null;
    }

    fn keyAt(s: *const SuffixLookup, context: []const u32, end: usize) Key {
        var key: Key = @splat(0xFFFF_FFFF);
        const n = s.params.ngram;
        @memcpy(key[0..n], context[end - n .. end]);
        return key;
    }

    /// The index over the context's first `end` tokens (from where it stands).
    fn extendTo(s: *SuffixLookup, context: []const u32, end: usize) !void {
        const n = s.params.ngram;
        var position = @max(s.indexed, n - 1);
        while (position < end) : (position += 1) {
            const got = try s.index.getOrPut(s.gpa, s.keyAt(context, position + 1));
            if (!got.found_existing) got.value_ptr.* = .empty;
            try got.value_ptr.append(s.gpa, @intCast(position));
        }
        s.indexed = end;
    }

    /// A prompt's tokens indexed ahead of its first proposal, `max_tokens` at a time (the host calls it once per prompt step, while the
    /// GPU runs the step): the index is the same one `propose` would build from the whole prompt at once (a function of the tokens in
    /// order), so no proposal changes, and the first round of a long prompt's stream does not build it with every GPU idle. Only
    /// `context[0..upto]` is indexed (the prompt; the tokens after it are the stream's own, taken by `propose`).
    pub fn warm(s: *SuffixLookup, context: []const u32, upto: usize, max_tokens: usize) !void {
        const n = s.params.ngram;
        const limit = @min(upto, context.len);
        if (limit < n + 1 or max_tokens == 0) return;
        if (s.indexed > context.len or (s.indexed != 0 and !std.mem.eql(u32, &s.keyAt(context, s.indexed), &s.last_key))) {
            s.clearIndex();
            s.indexed = 0;
        }
        if (s.indexed >= limit) return;
        const t0 = ticks();
        // a long prompt's map is sized once, here, not grown by doubling (each growth rehashes every key in one go: tens of ms at a million)
        if (s.indexed == 0 and limit >= warm_presize_from) try s.index.ensureTotalCapacity(s.gpa, @intCast(limit - n + 1));
        const from = s.indexed;
        try s.extendTo(context, @min(limit, @max(s.indexed, n - 1) + max_tokens));
        s.last_key = s.keyAt(context, s.indexed);
        const dt = ticks() -% t0;
        meter.warmed += s.indexed - from;
        meter.warm_ticks += dt;
        meter.warm_ticks_max = @max(meter.warm_ticks_max, dt);
    }

    /// Prompts of this many tokens or more get their map sized for every position at the first warm.
    const warm_presize_from = 4096;

    fn extend(s: *SuffixLookup, context: []const u32) !void {
        const from = s.indexed;
        const t0 = ticks();
        try s.extendTo(context, context.len);
        if (context.len > from) {
            const dt = ticks() -% t0;
            meter.extended += context.len - from;
            meter.extend_ticks += dt;
            meter.extend_ticks_max = @max(meter.extend_ticks_max, dt);
        }
    }

    fn matchLength(s: *const SuffixLookup, context: []const u32, end: usize) i64 {
        const len: i64 = @intCast(context.len);
        const e: i64 = @intCast(end);
        var length: i64 = 0;
        const limit = @min(s.params.max_extension, e);
        while (length < limit and context[@intCast(e - 1 - length)] == context[@intCast(len - 1 - length)]) {
            length += 1;
            if (len - 1 - length < 0) break;
        }
        return length;
    }

    /// Python `propose`: the copied continuation (owned by the proposer until the next call), or empty.
    pub fn propose(s: *SuffixLookup, context: []const u32, max_draft: i64) ![]const u32 {
        const t0 = ticks();
        defer {
            const dt = ticks() -% t0;
            meter.calls += 1;
            meter.ticks += dt;
            meter.ticks_max = @max(meter.ticks_max, dt);
        }
        return s.proposeAt(context, max_draft);
    }

    fn proposeAt(s: *SuffixLookup, context: []const u32, max_draft: i64) ![]const u32 {
        const n = s.params.ngram;
        s.out.clearRetainingCapacity();
        if (max_draft <= 0 or context.len < n + 1) return s.out.items;
        if (s.silent_for > 0) {
            s.silent_for -= 1;
            s.silenced_rounds += 1;
            return s.out.items;
        }
        // the context changed underneath the index (a new request): rebuild it
        const changed = s.indexed > context.len or (s.indexed != 0 and !std.mem.eql(u32, &s.keyAt(context, s.indexed), &s.last_key));
        if (changed) {
            s.clearIndex();
            s.indexed = 0;
        }
        try s.extend(context);
        s.last_key = s.keyAt(context, s.indexed);
        const m = try s.scan(context);
        if (!m.found) return s.out.items;
        const best_end = m.best_end;
        const best_len = m.best_len;
        s.last_confident = false;
        s.last_match = best_len;
        if (best_end < 0 or best_len < s.params.min_match) return s.out.items;
        s.last_confident = best_len >= s.params.confident_match;
        const from: usize = @intCast(best_end);
        const to = @min(context.len, from + @as(usize, @intCast(max_draft)));
        try s.out.appendSlice(s.gpa, context[from..to]);
        if (s.out.items.len > 0) {
            s.proposals += 1;
            s.proposed_tokens += @intCast(s.out.items.len);
        }
        return s.out.items;
    }

    /// What the candidate scan found at one context: no earlier place of the last n-gram (`found` false), else the longest match's end and length.
    const Scan = struct { len: usize, tok: u32, gen: u32, found: bool, best_end: i64, best_len: i64 };

    /// The scan of the candidate positions of the context's last n-gram (the index is extended to the context). Its answer is a function of the
    /// context alone, and the stream asks three times at one context in a round (the depth rule twice, the next window's plan once), so a
    /// call at the context the call before saw takes that answer; everything else a call does (the silence count, the counters, the copy) is
    /// made again by the caller, call by call.
    fn scan(s: *SuffixLookup, context: []const u32) !Scan {
        if (s.params.scan_memo) if (s.memo) |m| {
            if (m.len == context.len and m.gen == s.gen and m.tok == context[context.len - 1]) {
                meter.cached += 1;
                return m;
            }
        };
        var out: Scan = .{ .len = context.len, .tok = context[context.len - 1], .gen = s.gen, .found = false, .best_end = -1, .best_len = 0 };
        const positions: []const u32 = if (s.index.get(s.keyAt(context, context.len))) |l| l.items else &.{};
        if (positions.len > 0) {
            out.found = true;
            meter.scanned(positions.len);
            var i = positions.len;
            while (i > 0) {
                i -= 1;
                const end = positions[i] + 1;
                if (end >= context.len) continue;
                const length = s.matchLength(context, end);
                if (length > out.best_len) {
                    out.best_len = length;
                    out.best_end = end;
                    if (length >= s.params.max_extension) break;
                }
            }
        }
        s.memo = out;
        return out;
    }

    pub fn observe(s: *SuffixLookup, proposed: i64, accepted: i64) !void {
        s.judged_tokens += proposed;
        s.accepted_tokens += accepted;
        if (proposed <= 0) return;
        try s.recent.append(s.gpa, accepted);
        if (s.recent.items.len > s.params.window) {
            const drop = s.recent.items.len - s.params.window;
            std.mem.copyForwards(i64, s.recent.items, s.recent.items[drop..]);
            s.recent.shrinkRetainingCapacity(s.params.window);
        }
        if (s.recent.items.len >= s.params.window and std.mem.max(i64, s.recent.items) == 0) {
            s.silent_for = s.params.silence_rounds;
            s.recent.clearRetainingCapacity();
        }
    }
};

test "suffix lookup copies the continuation after a repeated n-gram" {
    const gpa = std.testing.allocator;
    var s = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2 });
    defer s.deinit();
    const ctx = [_]u32{ 1, 2, 3, 4, 5, 9, 1, 2, 3 };
    const got = try s.propose(&ctx, 3);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, got);
    try std.testing.expectEqual(@as(i64, 3), s.last_match);
}

test "the meter counts a call, the positions its scan walked and the tokens the index grew by, and its line clears the span" {
    const gpa = std.testing.allocator;
    meter = .{};
    var s = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2 });
    defer s.deinit();
    const ctx = [_]u32{ 1, 2, 3, 4, 5, 9, 1, 2, 3, 7, 1, 2, 3 };
    _ = try s.propose(&ctx, 3);
    try std.testing.expectEqual(@as(u64, 1), meter.calls);
    try std.testing.expectEqual(@as(u64, 1), meter.scans);
    try std.testing.expectEqual(@as(u64, 3), meter.positions); // "2 3" ends at three places, the context's own end among them
    try std.testing.expectEqual(@as(u64, ctx.len), meter.extended);
    try std.testing.expectEqual(@as(u64, 1), meter.hist[Meter.bucket(3)]);
    var buf: [768]u8 = undefined;
    const text = meter.line(&buf) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, text, "1 calls (1 walked the positions") != null);
    try std.testing.expect(meter.line(&buf) == null);
    try std.testing.expectEqual(@as(usize, 0), Meter.bucket(0));
    try std.testing.expectEqual(@as(usize, 1), Meter.bucket(1));
    try std.testing.expectEqual(@as(usize, 2), Meter.bucket(3));
    try std.testing.expectEqual(@as(usize, Meter.hist_n - 1), Meter.bucket(1 << 40));
}

/// A context grown the way a stream's is: tokens of a small alphabet (so trigrams repeat and scans have candidates), a few at a time.
fn grownContext(gpa: Allocator, seed: u64, rounds: usize) !std.ArrayList(u32) {
    var out: std.ArrayList(u32) = .empty;
    var x = seed;
    for (0..rounds * 3) |_| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        try out.append(gpa, @intCast((x >> 33) % 5));
    }
    return out;
}

test "three calls at one context walk the candidate positions once" {
    const gpa = std.testing.allocator;
    meter = .{};
    var s = try SuffixLookup.init(gpa, .{ .ngram = 2, .min_match = 2 });
    defer s.deinit();
    const ctx = [_]u32{ 1, 2, 3, 4, 5, 9, 1, 2, 3, 7, 1, 2, 3 };
    const a = try gpa.dupe(u32, try s.propose(&ctx, 3));
    defer gpa.free(a);
    const b = try gpa.dupe(u32, try s.propose(&ctx, 3));
    defer gpa.free(b);
    const c = try gpa.dupe(u32, try s.propose(&ctx, 2));
    defer gpa.free(c);
    try std.testing.expectEqual(@as(u64, 3), meter.calls);
    try std.testing.expectEqual(@as(u64, 1), meter.scans);
    try std.testing.expectEqual(@as(u64, 2), meter.cached);
    try std.testing.expectEqualSlices(u32, a, b);
    try std.testing.expectEqualSlices(u32, a[0..2], c);
}

test "the scan memo changes nothing a call does: outputs and every counter equal a proposer that walks every time, over grown contexts, repeats, silence and a new request" {
    const gpa = std.testing.allocator;
    for ([_]u64{ 1, 7, 99 }) |seed| {
        var ctx = try grownContext(gpa, seed, 120);
        defer ctx.deinit(gpa);
        var with = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 4 });
        defer with.deinit();
        var without = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 4, .scan_memo = false });
        defer without.deinit();
        var len: usize = 8;
        var round: usize = 0;
        while (len <= ctx.items.len) : (round += 1) {
            const view = ctx.items[0..len];
            if (round == 60) { // a new request on the same objects: the context is another one
                var other = try grownContext(gpa, seed + 1000, 40);
                defer other.deinit(gpa);
                for ([_]*SuffixLookup{ &with, &without }) |p| _ = try p.propose(other.items, 4);
            }
            for ([_]i64{ 5, 3, 5, 0, 5 }) |max_draft| {
                const x = try gpa.dupe(u32, try with.propose(view, max_draft));
                defer gpa.free(x);
                const y = try without.propose(view, max_draft);
                try std.testing.expectEqualSlices(u32, y, x);
                try std.testing.expectEqual(without.last_match, with.last_match);
                try std.testing.expectEqual(without.last_confident, with.last_confident);
                try std.testing.expectEqual(without.silent_for, with.silent_for);
                try std.testing.expectEqual(without.proposals, with.proposals);
                try std.testing.expectEqual(without.proposed_tokens, with.proposed_tokens);
                try std.testing.expectEqual(without.silenced_rounds, with.silenced_rounds);
            }
            // a round's verdict: mostly rejected, so the silence engages
            const accepted: i64 = if (round % 9 == 0) 2 else 0;
            try with.observe(3, accepted);
            try without.observe(3, accepted);
            len += 1 + round % 3;
        }
        try std.testing.expect(without.silenced_rounds > 0); // (the silence was exercised)
        try std.testing.expect(without.proposals > 0);
    }
}

test "an index warmed in slices ahead of the first proposal gives the proposals, and every counter, of one built by the proposal" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 1, 7, 100, 4096 }) |slice| {
        var ctx = try grownContext(gpa, 5, 2400); // 7,200 tokens: past the size the map is sized for at once
        defer ctx.deinit(gpa);
        const prompt = 7000;
        var a = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 4 });
        defer a.deinit();
        var b = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 4 });
        defer b.deinit();
        var warmed: usize = 0;
        while (b.indexed < prompt) {
            const before = b.indexed;
            try b.warm(ctx.items, prompt, slice);
            try std.testing.expect(b.indexed > before);
            try std.testing.expect(b.indexed - before <= slice + 3); // (the first slice also takes the n-gram's lead-in)
            warmed += 1;
        }
        try std.testing.expectEqual(@as(usize, prompt), b.indexed);
        try b.warm(ctx.items, prompt, slice); // nothing left: a no-op
        try std.testing.expectEqual(@as(usize, prompt), b.indexed);
        // the decode's contexts from the prompt on: the first proposal and every one after it
        meter = .{};
        var len: usize = prompt;
        while (len <= ctx.items.len) : (len += 1) {
            const view = ctx.items[0..len];
            for ([_]i64{ 5, 5 }) |md| {
                const x = try gpa.dupe(u32, try b.propose(view, md));
                defer gpa.free(x);
                const y = try a.propose(view, md);
                try std.testing.expectEqualSlices(u32, y, x);
                try std.testing.expectEqual(a.last_match, b.last_match);
                try std.testing.expectEqual(a.proposals, b.proposals);
            }
            try a.observe(3, 0);
            try b.observe(3, 0);
        }
        try std.testing.expectEqual(@as(u32, 0), b.gen); // (the warmed index was never found changed and rebuilt)
    }
}

test "a warm after the context was rewritten starts the index again" {
    const gpa = std.testing.allocator;
    var one = try grownContext(gpa, 3, 40);
    defer one.deinit(gpa);
    var two = try grownContext(gpa, 4, 40);
    defer two.deinit(gpa);
    var a = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 4 });
    defer a.deinit();
    try a.warm(one.items, one.items.len, 1000);
    try std.testing.expectEqual(one.items.len, a.indexed);
    try a.warm(two.items, two.items.len, 1000);
    try std.testing.expectEqual(two.items.len, a.indexed);
    var fresh = try SuffixLookup.init(gpa, .{ .ngram = 3, .min_match = 4 });
    defer fresh.deinit();
    const x = try gpa.dupe(u32, try a.propose(two.items, 5));
    defer gpa.free(x);
    try std.testing.expectEqualSlices(u32, try fresh.propose(two.items, 5), x);
}
