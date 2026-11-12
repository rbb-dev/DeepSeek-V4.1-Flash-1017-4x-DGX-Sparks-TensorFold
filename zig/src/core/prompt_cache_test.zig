//! The prompt cache's tests: a fake family whose state is a running sum, so a resumed pass equals a fresh one exactly when it should.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const imprint = @import("prompt_imprint.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Snapshots = pc.Snapshots;
const Saved = pc.Saved;
const Store = pc.Store;
const Plan = pc.Plan;
const Images = pc.Images;
const Allocator = std.mem.Allocator;

/// Rank 1's bytes: it applies a request's drops before that request's keeps; drops made during a pass wait for the next request.
const Peer = struct {
    held: u64 = 0,
    pending: u64 = 0,
    max: u64 = 0,
    in_pass: bool = false,

    fn request(p: *Peer) void {
        p.held -= p.pending;
        p.pending = 0;
    }
};

/// A family over host memory for tests: its live state is a position and a running sum of the prompt's tokens.
const Fake = struct {
    gpa: Allocator,
    at: u32 = 0,
    sum: u64 = 0,
    fail_save: bool = false,
    fail_restore: bool = false,
    live: usize = 0,
    spare_bytes: u64 = 0,
    peer: ?*Peer = null, // speed-up mode's rank 1: it keeps the same states and drops them only when a request names them
    refuse_at: ?u32 = null, // the family's own rule (`usable`): a state at this position does not restore

    const State = struct { at: u32, sum: u64 };

    fn snapshots(f: *Fake) Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn } };
    }
    /// With the family's own usable rule: a state at `refuse_at` does not restore.
    fn checked(f: *Fake) Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .usable = usableFn } };
    }
    fn usableFn(ptr: *anyopaque, saved: Saved, _: u32) bool {
        const st: *State = @ptrCast(@alignCast(saved));
        return of(ptr).refuse_at != st.at;
    }
    /// With learned states on disk: a state's position and sum in one file.
    fn learned(f: *Fake) Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .write = writeFn, .read = readFn, .forget = forgetFn } };
    }
    fn forgetFn(_: *anyopaque, dir: [:0]const u8, key: u64) void {
        var path: [512]u8 = undefined;
        _ = std.c.unlink(file(&path, dir, key) catch return);
    }
    fn file(buf: []u8, dir: []const u8, key: u64) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0);
    }
    fn writeFn(_: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64) anyerror!void {
        var path: [512]u8 = undefined;
        const fd = std.c.open(try file(&path, dir, key), .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.WriteFailed;
        defer _ = std.c.close(fd);
        try imprint.writeAll(fd, std.mem.asBytes(@as(*State, @ptrCast(@alignCast(saved)))));
    }
    fn readFn(ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!Saved {
        const f = of(ptr);
        var path: [512]u8 = undefined;
        const fd = std.c.open(try file(&path, dir, key), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.ReadFailed;
        defer _ = std.c.close(fd);
        const st = try f.gpa.create(State);
        errdefer f.gpa.destroy(st);
        if (!imprint.readAt(fd, std.mem.asBytes(st), 0) or st.at != at) return error.ReadFailed;
        f.live += 1;
        return st;
    }
    /// A pool-like family: a kept state holds 10 bytes under `bytes`, and a dropped one's storage stays spare.
    fn pooled(f: *Fake) Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = savePooledFn, .restore = restoreFn, .drop = dropSpareFn, .charged = chargedFn, .spare = spareFn, .trim = trimFn, .reuses = reusesFn } };
    }
    fn reusesFn(ptr: *anyopaque, at: u32) bool {
        return of(ptr).spare_bytes >= 90 + at;
    }
    fn savePooledFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved {
        const st = try saveFn(ptr, owner, at);
        const f = of(ptr);
        if (f.spare_bytes >= 90 + at) f.spare_bytes -= 90 + at; // the spare storage took it
        return st;
    }
    fn chargedFn(_: *anyopaque, saved: Saved) u64 {
        const st: *State = @ptrCast(@alignCast(saved));
        return 90 + st.at;
    }
    fn dropSpareFn(ptr: *anyopaque, saved: Saved) void {
        const st: *State = @ptrCast(@alignCast(saved));
        of(ptr).spare_bytes += 90 + st.at;
        dropFn(ptr, saved);
    }
    fn spareFn(ptr: *anyopaque) u64 {
        return of(ptr).spare_bytes;
    }
    fn trimFn(ptr: *anyopaque, room_: u64) void {
        const f = of(ptr);
        f.spare_bytes = @min(f.spare_bytes, room_);
    }
    fn of(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytesFn(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!Saved {
        const f = of(ptr);
        if (f.fail_save) return error.CopyFailed;
        if (at != f.at) return error.NotAtMark;
        const st = try f.gpa.create(State);
        st.* = .{ .at = f.at, .sum = f.sum };
        f.live += 1;
        if (f.peer) |p| {
            p.held += 100 + at;
            p.max = @max(p.max, p.held);
        }
        return st;
    }
    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: Saved) anyerror!void {
        const f = of(ptr);
        if (f.fail_restore) return error.CopyFailed;
        const st: *State = @ptrCast(@alignCast(saved));
        f.at, f.sum = .{ st.at, st.sum };
    }
    fn dropFn(ptr: *anyopaque, saved: Saved) void {
        const f = of(ptr);
        const st: *State = @ptrCast(@alignCast(saved));
        if (f.peer) |p| { // dropped before the pass: named in this request; during it: in the next one
            if (p.in_pass) p.pending += 100 + st.at else p.held -= 100 + st.at;
        }
        f.gpa.destroy(st);
        f.live -= 1;
    }

    /// A prompt pass from `plan.from`, keeping at each mark; returns the sum a fresh pass would give.
    fn pass(f: *Fake, s: *Store, prompt: []const u32, plan: Plan) u64 {
        return f.passKeyed(s, prompt, plan.from, plan.marks, .none);
    }

    /// `pass` for a prompt with images: the states it keeps are keyed by them too.
    fn passKeyed(f: *Fake, s: *Store, prompt: []const u32, from: u32, marks: []const u32, images: Images) u64 {
        if (from == 0) f.* = .{ .gpa = f.gpa, .fail_save = f.fail_save, .fail_restore = f.fail_restore, .live = f.live, .spare_bytes = f.spare_bytes, .peer = f.peer };
        if (f.peer) |p| p.in_pass = true;
        defer if (f.peer) |p| {
            p.in_pass = false;
        };
        var mi: usize = 0;
        for (prompt[from..]) |t| {
            f.sum = f.sum *% 31 +% t;
            f.at += 1;
            if (mi < marks.len and marks[mi] == f.at) {
                _ = s.keepKeyed(prompt, f.at, null, &.{}, images);
                mi += 1;
            }
        }
        return f.sum;
    }
};

/// Where a lookup's pass starts: the resumed state's mark, else 0.
fn startsAt(l: pc.Lookup) u32 {
    return if (l.entry) |e| e.at else 0;
}

/// A prompt's one image: its span starts at `at`, and its bytes are told apart by `tag`.
const OneImage = struct {
    at: u32,
    tag: u64,

    fn images(o: *const OneImage) Images {
        return .{ .ctx = @ptrCast(o), .hash = key };
    }

    fn key(ctx: ?*const anyopaque, at: u32) u64 {
        const o: *const OneImage = @ptrCast(@alignCast(ctx orelse return 0));
        return if (o.at < at) o.tag | 1 else 0; // a state's prefix holds the images that start before its mark
    }
};

fn fresh(prompt: []const u32) u64 {
    var sum: u64 = 0;
    for (prompt) |t| sum = sum *% 31 +% t;
    return sum;
}

test "a growing conversation resumes each turn where the last one's history ended, and equals a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 9, 9 }; // history 5, then a two-token generation prompt
    var p = try s.begin(a, &t1, 5, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqualSlices(u32, &.{5}, p.marks);
    try std.testing.expectEqual(fresh(&t1), f.pass(&s, &t1, p));
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 9, 7, 7, 6, 6, 9, 9 }; // the reply and a tool result, history 10
    p = try s.begin(a, &t2, 10, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 5), p.from);
    try std.testing.expectEqualSlices(u32, &.{10}, p.marks);
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, p));
    const edited = [_]u32{ 1, 2, 3, 8, 5, 9, 7, 7, 6, 6, 9, 9 }; // an earlier turn edited: nothing resumes past it
    p = try s.begin(a, &edited, 10, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(fresh(&edited), f.pass(&s, &edited, p));
    try std.testing.expectEqual(@as(u64, 1), s.counts.hits);
    try std.testing.expectEqual(@as(u64, 2), s.counts.misses);
}

test "an entry keys its lookahead tokens: a prompt that differs right after the state does not resume it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 9 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null)); // keeps [1 2 3 4] + 5
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 6, 9 }, &.{}) == null);
    try std.testing.expectEqual(@as(u32, 4), s.find(&.{ 1, 2, 3, 4, 5, 6 }, &.{}).?.at);
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4 }, &.{}) == null); // the lookahead token must be in the prompt
}

test "a state within the window of a prompt's end is passed over for an earlier one, and stays for the conversation that kept it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .window = 3, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const long = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    _ = f.pass(&s, &long, .{ .marks = &.{ 4, 8 } });
    const short = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 20, 21 }; // shares 8 tokens and ends 2 past them, inside the tail
    var p = try s.begin(a, &short, 9, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 4), p.from);
    try std.testing.expectEqual(fresh(&short), f.pass(&s, &short, p));
    const next = long ++ [_]u32{ 13, 14 }; // the long conversation's next turn resumes its own state at 8
    p = try s.begin(a, &next, 12, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 8), p.from);
}

test "the native eligibility rule: on the grid, at most len - window from the end, its lookahead inside the prompt" {
    try std.testing.expect(!pc.Rules.eligible(.{ .grid = 8, .window = 4 }, 47, 64));
    try std.testing.expect(pc.Rules.eligible(.{ .grid = 8, .window = 4 }, 48, 52));
    try std.testing.expect(!pc.Rules.eligible(.{ .grid = 8, .window = 4 }, 48, 51));
    try std.testing.expectEqual(@as(?u32, 48), pc.Rules.maxAt(.{ .grid = 8, .window = 4 }, 52));
    try std.testing.expectEqual(@as(?u32, null), pc.Rules.maxAt(.{ .grid = 8, .window = 4 }, 9));
    try std.testing.expectEqual(@as(?u32, 9), pc.Rules.maxAt(.{}, 10));
    try std.testing.expectEqual(@as(u32, 16), pc.Rules.floorGrid(.{ .grid = 8 }, 21));
    try std.testing.expectEqual(@as(u32, 21), pc.Rules.floorGrid(.{}, 21));
}

test "with no grid and no family rule, find returns exactly the state the tail rule found (2,000 seeded cases)" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0x9011);
    const r = prng.random();
    var case: usize = 0;
    while (case < 2000) : (case += 1) {
        const window = r.uintAtMost(u32, 64);
        const lookahead = r.uintAtMost(u32, 1);
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.snapshots(), .{ .window = window, .lookahead = lookahead, .min_prompt = 0 }, 1 << 24);
        defer s.deinit();
        var base: [3][200]u32 = undefined;
        var lens: [3]usize = undefined;
        for (&base, &lens, 0..) |*b, *len, c| { // three conversations: the second and third share a random prefix of the first
            len.* = 2 + r.uintLessThan(usize, 199);
            const shared = if (c == 0) 0 else r.uintAtMost(usize, @min(len.*, lens[0]));
            for (b[0..len.*], 0..) |*t, i| t.* = if (i < shared) base[0][i] else r.uintAtMost(u32, 3);
            var marks: [200]u32 = undefined;
            var m: usize = 0;
            var at: u32 = 1;
            while (at <= len.*) : (at += 1) {
                if (r.uintLessThan(u32, 8) == 0) {
                    marks[m] = at;
                    m += 1;
                }
            }
            _ = f.pass(&s, b[0..len.*], .{ .marks = marks[0..m] });
        }
        var query: [270]u32 = undefined; // a random prefix of one conversation, then random tokens
        const from = r.uintLessThan(usize, 3);
        const k = r.uintAtMost(usize, lens[from]);
        @memcpy(query[0..k], base[from][0..k]);
        const extra = r.uintAtMost(usize, 70);
        for (query[k .. k + extra]) |*t| t.* = r.uintAtMost(u32, 3);
        const prompt = query[0 .. k + extra];
        if (prompt.len == 0) continue;
        var want: ?*pc.Entry = null; // find's predicate, written out
        for (s.entries.items) |e| {
            if (e.tokens.len > prompt.len or e.at >= prompt.len or @as(usize, e.at) + window > prompt.len or e.at == 0) continue;
            if (want != null and e.at <= want.?.at) continue;
            if (!std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens)) continue;
            want = e;
        }
        try std.testing.expectEqual(want, s.find(prompt, &.{}));
    }
}

test "a state the family's own rule refuses is passed over for a shorter one; without the rule it is found" {
    const gpa = std.testing.allocator;
    var f: Fake = .{ .gpa = gpa };
    const long = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var with = Store.init(gpa, f.checked(), .{ .min_prompt = 0 }, 1 << 20);
    defer with.deinit();
    _ = f.pass(&with, &long, .{ .marks = &.{ 4, 8 } });
    var without = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer without.deinit();
    _ = f.pass(&without, &long, .{ .marks = &.{ 4, 8 } });
    f.refuse_at = 8;
    try std.testing.expectEqual(@as(u32, 4), with.find(&long, &.{}).?.at);
    try std.testing.expectEqual(@as(u32, 8), without.find(&long, &.{}).?.at);
}

test "planned families resume and keep only at the request's chunk starts" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const p = try s.begin(a, &t1, 7, &.{}, &.{ 4, 6 }, null);
    try std.testing.expectEqualSlices(u32, &.{6}, p.marks); // history 7 floored to the start at 6
    _ = f.pass(&s, &t1, p);
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 5, 6, 7, 9 }, &.{4}) == null); // 6 is not one of this prompt's starts
    try std.testing.expectEqual(@as(u32, 6), s.find(&.{ 1, 2, 3, 4, 5, 6, 7, 9 }, &.{ 4, 6 }).?.at);
}

test "eviction frees a conversation's superseded state first, then the oldest; a state past the budget is refused" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 330); // fake bytes: 100 + at
    defer s.deinit();
    const other = [_]u32{ 5, 5, 5, 5 };
    _ = f.pass(&s, &other, try s.begin(a, &other, 3, &.{}, &.{}, null)); // 103 bytes
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null)); // 104: 207 held
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = f.pass(&s, &t2, try s.begin(a, &t2, 6, &.{}, &.{}, null)); // 106 more: 313
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    const t3 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    _ = f.pass(&s, &t3, try s.begin(a, &t3, 8, &.{}, &.{}, null)); // 108: t1's state (extended by later turns) goes first
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.find(&.{ 5, 5, 5, 5 }, &.{}) != null);
    try std.testing.expectEqual(@as(u32, 8), s.find(&t3, &.{}).?.at);
    const big: [300]u32 = @splat(7);
    _ = f.pass(&s, &big, try s.begin(a, &big, 299, &.{}, &.{}, null)); // 399 > 330: refused, nothing evicted
    try std.testing.expectEqual(@as(u64, 1), s.counts.refused);
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.held <= s.budget);
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "a failed copy keeps nothing and a failed restore prefills from the start" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa, .fail_save = true };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null));
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    f.fail_save = false;
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    f.fail_restore = true;
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    const p = try s.begin(a, &t2, 6, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len); // the entry that failed is gone
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, p));
}

test "marks: the stable prefix with the last prompt and shared blocks, past the resume point, before the end" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_gap = 2, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const prev = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 0, 0 };
    try std.testing.expectEqualSlices(u32, &.{ 3, 7, 10 }, try s.marks(a, &prompt, 0, 10, &.{ 3, 12, 0 }, &.{}, &prev));
    try std.testing.expectEqualSlices(u32, &.{10}, try s.marks(a, &prompt, 7, 10, &.{3}, &.{}, &prev));
    try std.testing.expectEqualSlices(u32, &.{}, try s.marks(a, &prompt, 0, 0, &.{}, &.{}, &.{})); // a raw prompt keeps nothing
    try std.testing.expectEqualSlices(u32, &.{10}, try s.marks(a, &prompt, 0, 10, &.{9}, &.{}, &.{})); // a block next to the history
}

test "prompts under min_prompt keep nothing (their extra prompt call would cost more than reuse saves), longer ones keep as before" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 8 }, 1 << 20);
    defer s.deinit();
    const short = [_]u32{ 1, 2, 3, 4, 5, 9, 9 }; // 7 tokens, history 5
    const p = try s.begin(a, &short, 5, &.{}, &.{}, null);
    try std.testing.expectEqualSlices(u32, &.{}, p.marks);
    try std.testing.expectEqual(fresh(&short), f.pass(&s, &short, p));
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    const long = [_]u32{ 1, 2, 3, 4, 5, 9, 7, 7, 9, 9 }; // 10 tokens: the next turn keeps its history, resuming nothing
    const q = try s.begin(a, &long, 8, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), q.from);
    try std.testing.expectEqualSlices(u32, &.{8}, q.marks);
    try std.testing.expectEqual(fresh(&long), f.pass(&s, &long, q));
    try std.testing.expectEqual(@as(u32, 8), s.find(&.{ 1, 2, 3, 4, 5, 9, 7, 7, 9, 9, 4 }, &.{}).?.at);
}

test "kept states charge their real storage, a save takes spare storage first, and spare stays inside the budget" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.pooled(), .{ .min_prompt = 0 }, 330); // new storage 100 + at, a kept state 90 + at
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null));
    try std.testing.expectEqual(@as(u64, 94), s.held);
    f.spare_bytes = 120; // a buffer readied while idle
    try std.testing.expectEqual(@as(u64, 116), s.room());
    const other = [_]u32{ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
    _ = f.pass(&s, &other, try s.begin(a, &other, 10, &.{}, &.{}, null)); // takes the readied buffer: nothing evicted
    try std.testing.expectEqual(@as(u64, 194), s.held);
    try std.testing.expectEqual(@as(u64, 20), s.spare());
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    var t2: [102]u32 = undefined;
    for (&t2, 1..) |*t, i| t.* = @intCast(i);
    _ = f.pass(&s, &t2, try s.begin(a, &t2, 101, &.{}, &.{}, null)); // resumes t1's; 201 new: the spare shrinks, then `other` goes
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expectEqual(@as(u64, 94 + 191), s.held);
    try std.testing.expect(s.held + s.spare() <= s.budget);
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "a shared system cut outlives its own conversation's turns: a second conversation resumes it after cold prompts fill the budget" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 600); // bytes 100 + at
    defer s.deinit();
    var c1: [30]u32 = @splat(9);
    var c2: [30]u32 = @splat(8);
    for ([_][]u32{ &c1, &c2 }) |c| { // cold prompts first
        _ = f.pass(&s, c, try s.begin(a, c, 29, &.{}, &.{}, null));
        try std.testing.expect(s.held <= s.budget);
    }
    var conv: [45]u32 = undefined;
    for (&conv, 1..) |*t, i| t.* = @intCast(i); // a 20-token system block, then the turns
    for ([_]u32{ 25, 35, 45 }, [_]u32{ 24, 34, 44 }) |len, history| {
        _ = f.pass(&s, conv[0..len], try s.begin(a, conv[0..len], history, &.{20}, &.{}, null));
        try std.testing.expect(s.held <= s.budget);
    }
    var c3: [60]u32 = @splat(7);
    _ = f.pass(&s, &c3, try s.begin(a, &c3, 59, &.{}, &.{}, null));
    try std.testing.expect(s.held <= s.budget);
    var other: [25]u32 = undefined;
    @memcpy(other[0..20], conv[0..20]);
    for (other[20..], 0..) |*t, i| t.* = @intCast(90 + i);
    const plan = try s.begin(a, &other, 24, &.{20}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 20), plan.from); // the cut survived the first conversation's turns
    try std.testing.expectEqual(fresh(&other), f.pass(&s, &other, plan));
    try std.testing.expect(s.held <= s.budget);
}

test "a pass makes room for every state it keeps before it starts, so a peer told of the evictions with the request stays inside the budget" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var peer: Peer = .{};
    var f: Fake = .{ .gpa = gpa, .peer = &peer };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 600); // bytes 100 + at
    defer s.deinit();
    var conv: [200]u32 = undefined;
    for (&conv, 1..) |*t, i| t.* = @intCast(i);
    var len: u32 = 30;
    while (len <= 200) : (len += 17) { // a conversation's turns: each keeps its history and the stable prefix, two states a pass
        peer.request(); // the drops the last pass made reach the peer with this request
        const plan = try s.begin(a, conv[0..len], len - 3, &.{20}, &.{}, null); // its own evictions too, before the peer keeps anything
        _ = f.pass(&s, conv[0..len], plan);
        try std.testing.expect(s.held <= s.budget);
    }
    try std.testing.expect(s.counts.evicted > 0);
    try std.testing.expect(peer.max <= s.budget); // evicting at keep time instead, the peer holds the old states until the next request
}

test "a state a peer cannot resume is forgotten: the next prompt misses instead of asking again" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null));
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const plan = try s.begin(a, &t2, 6, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 4), plan.from);
    s.forget(&t2, plan.from); // the peer's answer: it lacks that state
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    try std.testing.expectEqual(@as(?*pc.Entry, null), s.find(&t2, &.{}));
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "a learned harness state outlives its store: a fresh session on a new one resumes it from disk, equal to a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-test-{d}", .{std.c.getpid()});
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const harness = [_]u32{ 7, 7, 7, 7, 7, 7 }; // shared by every session: a state at 5 reads it all (lookahead 1)
    const key = imprint.Imprint.keyOf(&harness);
    defer rmTree(root);
    {
        var im = try imprint.Imprint.open(gpa, root, 3, 1 << 30);
        defer im.deinit();
        {
            var f: Fake = .{ .gpa = gpa };
            var s = Store.init(gpa, f.learned(), rules, 1 << 20);
            defer s.deinit();
            s.imprint = &im;
            const first = harness ++ [_]u32{ 1, 2, 9 };
            try std.testing.expectEqual(fresh(&first), f.pass(&s, &first, try s.begin(a, &first, 8, &.{5}, &.{}, null)));
            try std.testing.expect(im.has(key));
        }
        var again = try imprint.Imprint.open(gpa, root, 3, 1 << 30); // a new server reads the index back
        defer again.deinit();
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &again;
        const second = harness ++ [_]u32{ 3, 4, 9 };
        const p = try s.begin(a, &second, 8, &.{5}, &.{}, null);
        try std.testing.expectEqual(@as(u32, 5), p.from);
        try std.testing.expectEqual(fresh(&second), f.pass(&s, &second, p));
    }
}

test "past the learned-state cap the least recently used state is forgotten, and a file that no longer reads is too" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-cap-{d}", .{std.c.getpid()});
    defer rmTree(root);
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const one = [_]u32{ 7, 7, 7, 7, 7, 7 };
    const two = [_]u32{ 8, 8, 8, 8, 8, 8 };
    var im = try imprint.Imprint.open(gpa, root, 3, 150); // a state at 5 is 105 bytes: one fits
    defer im.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.learned(), rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const p1 = one ++ [_]u32{ 1, 9 };
    try std.testing.expectEqual(fresh(&p1), f.pass(&s, &p1, try s.begin(a, &p1, 7, &.{5}, &.{}, null)));
    const p2 = two ++ [_]u32{ 1, 9 };
    try std.testing.expectEqual(fresh(&p2), f.pass(&s, &p2, try s.begin(a, &p2, 7, &.{5}, &.{}, null)));
    try std.testing.expect(im.has(imprint.Imprint.keyOf(&two)) and !im.has(imprint.Imprint.keyOf(&one)));
    var path: [512]u8 = undefined;
    try std.testing.expect(std.c.unlink(try Fake.file(&path, im.dir, imprint.Imprint.keyOf(&one))) != 0); // its file went too
    _ = std.c.unlink(try Fake.file(&path, im.dir, imprint.Imprint.keyOf(&two))); // a file lost behind the store's back
    var g: Fake = .{ .gpa = gpa };
    var t = Store.init(gpa, g.learned(), rules, 1 << 20);
    defer t.deinit();
    t.imprint = &im;
    const p3 = two ++ [_]u32{ 2, 9 };
    const plan = try t.begin(a, &p3, 7, &.{5}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), plan.from); // the read failed: a fresh pass, and the state is forgotten
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(&two)));
    try std.testing.expectEqual(fresh(&p3), g.pass(&t, &p3, plan));
    try std.testing.expect(im.has(imprint.Imprint.keyOf(&two))); // learned again by that pass
}

test "a kept state keys the prompt's images: another picture at the same tokens resumes nothing" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    // an image's positions hold its placeholder token, so two different pictures give one prompt's tokens
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 7, 7 };
    const first: OneImage = .{ .at = 2, .tag = 0x1111 };
    const other: OneImage = .{ .at = 2, .tag = 0x2222 };
    const later: OneImage = .{ .at = 5, .tag = 0x3333 }; // starts at the mark: outside the state's prefix
    const plan = try s.lookupKeyed(a, &prompt, 5, &.{}, &.{}, first.images());
    try std.testing.expectEqual(@as(?*pc.Entry, null), plan.entry);
    try std.testing.expectEqual(fresh(&prompt), f.passKeyed(&s, &prompt, startsAt(plan), plan.marks, first.images()));
    // the same picture resumes at the history mark, and the pass past it equals a fresh one
    const again = try s.lookupKeyed(a, &prompt, 5, &.{}, &.{}, first.images());
    try std.testing.expectEqual(@as(u32, 5), again.entry.?.at);
    try Fake.restoreFn(&f, null, again.entry.?.saved); // the backend restores before its first chunk
    s.resumed(again.entry.?, &prompt, true);
    try std.testing.expectEqual(fresh(&prompt), f.passKeyed(&s, &prompt, startsAt(again), again.marks, first.images()));
    // another picture at that span resumes nothing, and keeps its state beside the first
    const diff = try s.lookupKeyed(a, &prompt, 5, &.{}, &.{}, other.images());
    try std.testing.expectEqual(@as(?*pc.Entry, null), diff.entry);
    try std.testing.expectEqual(fresh(&prompt), f.passKeyed(&s, &prompt, startsAt(diff), diff.marks, other.images()));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    // an image that starts at the mark is outside the prefix: its key is 0, so no picture's state resumes...
    try std.testing.expectEqual(@as(?*pc.Entry, null), (try s.lookupKeyed(a, &prompt, 5, &.{}, &.{}, later.images())).entry);
    // ...and a prompt without images keeps, and then resumes, its own state beside both pictures'
    const plain = try s.lookup(a, &prompt, 5, &.{}, &.{});
    try std.testing.expectEqual(@as(?*pc.Entry, null), plain.entry);
    try std.testing.expectEqual(fresh(&prompt), f.passKeyed(&s, &prompt, startsAt(plain), plain.marks, .none));
    try std.testing.expectEqual(@as(u32, 5), (try s.lookup(a, &prompt, 5, &.{}, &.{})).entry.?.at);
    try std.testing.expectEqual(@as(u32, 5), (try s.lookupKeyed(a, &prompt, 5, &.{}, &.{}, later.images())).entry.?.at);
    try std.testing.expectEqual(@as(u32, 5), (try s.lookupKeyed(a, &prompt, 5, &.{}, &.{}, other.images())).entry.?.at);
}

test "every tier counter has its /metrics help, and every help names a counter (v3's landing, probe and store counters too)" {
    const T = pc.TierCounts;
    inline for (@typeInfo(T).@"struct".field_names) |name| try std.testing.expect(@field(T.help, name).len > 0);
    inline for (@typeInfo(@TypeOf(T.help)).@"struct".field_names) |name| try std.testing.expect(@hasField(T, name));
    inline for (.{ "direct_refused", "recover_deferred", "orphans_removed", "builds_pruned", "missing_errors", "landings", "landings_skipped", "landings_failed", "landing_pieces", "landing_bytes", "probe_runs", "probe_failed" }) |name|
        try std.testing.expect(@hasField(T, name));
    inline for (@typeInfo(@TypeOf(T.help)).@"struct".field_names) |name| try std.testing.expect(std.mem.indexOf(u8, @field(T.help, name), "--spill") == null); // settings are environment names
}

// -- the token arrays a kept state holds on the host, charged to the budget (charge_tokens) --------------------------

fn ids(comptime n: usize, base: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*x, i| x.* = base + @as(u32, @intCast(i));
    return out;
}

test "token arrays: a keep at 16 of a 40-token prompt holds its state and both arrays (340 bytes); off, its state (116)" {
    const gpa = std.testing.allocator;
    for ([_]bool{ true, false }) |charge| {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
        defer s.deinit();
        s.charge_tokens = charge;
        const p = ids(40, 1);
        _ = f.pass(&s, &p, .{ .marks = &.{16} });
        try std.testing.expectEqual(@as(u64, if (charge) 116 + 16 * 4 + 40 * 4 else 116), s.held);
        try std.testing.expectEqual(@as(u64, if (charge) 16 * 4 + 40 * 4 else 0), s.token_held);
    }
}

test "token arrays: a second keep the arrays push past the budget evicts the first; off, both stay" {
    const gpa = std.testing.allocator;
    for ([_]bool{ true, false }) |charge| {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 500);
        defer s.deinit();
        s.charge_tokens = charge;
        _ = f.pass(&s, &ids(40, 1), .{ .marks = &.{16} });
        _ = f.pass(&s, &ids(40, 1000), .{ .marks = &.{16} });
        try std.testing.expectEqual(@as(usize, if (charge) 1 else 2), s.entries.items.len);
        try std.testing.expect(s.held <= s.budget);
        if (charge) try std.testing.expectEqual(@as(u32, 1000), s.entries.items[0].tokens[0]); // the newer one stays
    }
}

test "token arrays: a mark whose state and arrays pass the budget is refused and counted; off, it is kept" {
    const gpa = std.testing.allocator;
    for ([_]bool{ true, false }) |charge| {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 300);
        defer s.deinit();
        s.charge_tokens = charge;
        _ = f.pass(&s, &ids(40, 1), .{ .marks = &.{16} });
        try std.testing.expectEqual(@as(usize, if (charge) 0 else 1), s.entries.items.len);
        try std.testing.expectEqual(@as(u64, if (charge) 1 else 0), s.counts.refused);
    }
}

test "token arrays: a resume by a longer prompt charges its longer last prompt, evicting another state, never itself" {
    const gpa = std.testing.allocator;
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    s.charge_tokens = true;
    const first = ids(40, 1);
    _ = f.pass(&s, &first, .{ .marks = &.{16} }); // 340
    _ = f.pass(&s, &ids(40, 500), .{ .marks = &.{16} }); // 340 more: 680
    var long: [120]u32 = undefined; // the first conversation goes on: 120 tokens resume its state at 16
    @memcpy(long[0..40], &first);
    for (long[40..], 40..) |*x, i| x.* = @intCast(9000 + i);
    const before = s.held;
    // the pass restored the state itself, then tells the store it resumed
    const e = s.find(&long, &.{}) orelse return error.NotFound;
    try std.testing.expectEqual(@as(u32, 16), e.at);
    s.resumed(e, &long, true);
    // a last prompt of 120 tokens: 320 bytes more; 1000 - 680 = 320 fits exactly, nothing evicted
    try std.testing.expectEqual(before + 4 * (120 - 40), s.held);
    try std.testing.expect(s.held <= s.budget);
    // a still longer resume: the other state goes for it, never the resumed one
    var longer: [180]u32 = undefined;
    @memcpy(longer[0..120], &long);
    for (longer[120..], 120..) |*x, i| x.* = @intCast(9000 + i);
    s.resumed(s.find(&longer, &.{}).?, &longer, true);
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(u32, 1), s.entries.items[0].tokens[0]);
    try std.testing.expect(s.held <= s.budget);
}
