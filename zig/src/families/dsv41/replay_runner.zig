//! The kernels replay's runner of a launch class over windows of rows: the baseline one launch over a window against each slab's launch of
//! the plan (slab_plan.zig), with the controls the comparison rests on. Before two runs are compared, the null control asks whether
//! each run wrote its outputs and whether the reference reads the window's inputs at all: a comparison of two runs that did nothing, or
//! of two runs over the same stale inputs, is equal and proves nothing. A function of the class's methods and a stream to synchronize,
//! so the host tests drive it with a class in host memory.
//!
//! A class has, all public (called on every class; a missing one is a compile error):
//!   prep(d)            the window's inputs from their sources (no-op where the launches read the sources)
//!   reset()            the outputs to the bytes every run starts from (poisoned, or an in-place input restored)
//!   whole(d)           the baseline one launch over the window's d rows
//!   slab(d, k, s, m)   slab k's launch at rows [s, s + m)
//!   after(d)           the launch that follows the slabs' (no-op where there is none)
//!   read(d)            the outputs' first d rows, page_allocator memory
//!   perturb(d, on)     one source value of the window's last row changed (on) or put back (off)
//!   strided            a bool constant: the slabs read blocks at the plan's stride, and `short` (a field) shortens it

const std = @import("std");
const slab_plan = @import("slab_plan.zig");

/// The byte an output is filled with before a launch.
pub const poison_byte: u8 = 0xa5;

/// Whether a run wrote its outputs: at least half of `out`'s bytes differ from the bytes the run started from (`start`), and they are
/// not all zero.
pub fn wrote(start: []const u8, out: []const u8) bool {
    std.debug.assert(start.len == out.len);
    if (out.len == 0) return false;
    var changed: usize = 0;
    var zero: usize = 0;
    for (start, out) |a, b| {
        changed += @intFromBool(a != b);
        zero += @intFromBool(b == 0);
    }
    return changed * 2 >= out.len and zero != out.len;
}

/// wrote() of a run that started from poison bytes.
pub fn wrotePoisoned(out: []const u8) bool {
    if (out.len == 0) return false;
    var changed: usize = 0;
    var zero: usize = 0;
    for (out) |b| {
        changed += @intFromBool(b != poison_byte);
        zero += @intFromBool(b == 0);
    }
    return changed * 2 >= out.len and zero != out.len;
}

pub fn differs(x: []const u8, y: []const u8) usize {
    var n: usize = 0;
    for (x, y) |p, q| n += @intFromBool(p != q);
    return n;
}

/// The control's short slab of a window of `n` rows: the last slab with more than one row (launched one row short); in a window of one
/// row, its slab (launched with none: no launch).
pub fn shortSlab(n: usize) ?usize {
    var short: ?usize = null;
    var last: ?usize = null;
    for (slab_plan.slabs(n), 0..) |sb, k| {
        if (sb.n > 1) short = k;
        if (sb.n > 0) last = k;
    }
    return short orelse last;
}

/// Why a window's null control failed.
pub const NullFail = enum { reference_unwritten, slabs_unwritten, reference_deaf };

pub fn nullText(f: NullFail) []const u8 {
    return switch (f) {
        .reference_unwritten => "the baseline one launch did not write its outputs",
        .slabs_unwritten => "the slabs' launches did not write their outputs",
        .reference_deaf => "the baseline one launch does not read the window's inputs (a changed input value left it as it was)",
    };
}

pub const Result = struct {
    windows: usize = 0,
    /// The bytes that differ between the slabs' outputs and the reference's in the windows whose null control held; a window whose null
    /// control failed counts every byte of its outputs (its comparison is not made).
    bad: usize = 0,
    /// Windows whose control (shortSlab's slab one row short) differed from the reference.
    cut_ok: usize = 0,
    /// Windows whose stride control (the blocks read q - 1 rows apart) differed, for a strided class.
    stride_ok: usize = 0,
    /// Windows whose null control held.
    null_ok: usize = 0,
    first_null: ?struct { d: usize, why: NullFail } = null,
};

/// Where a run failed: the window, the slab (k, start, rows) when a slab's launch did, and the step.
pub const Failure = struct { d: usize = 0, slab: ?[3]usize = null, step: []const u8 = "" };

fn at(fail: *Failure, d: usize, slab: ?[3]usize, step: []const u8, e: anyerror) anyerror {
    fail.* = .{ .d = d, .slab = slab, .step = step };
    return e;
}

/// One run of class `c` over a window of `d` rows: reset, then the baseline one launch (`slabs` false) or each slab's launch (`cut`: the
/// control's short slab one row short) and the launch after the slabs; the stream synchronized; the outputs read back.
pub fn once(s: anytype, c: anytype, d: usize, slabs: bool, cut: bool, fail: *Failure) ![]u8 {
    c.reset() catch |e| return at(fail, d, null, "reset", e);
    if (!slabs) {
        c.whole(d) catch |e| return at(fail, d, null, "the window's one launch", e);
    } else {
        const short = if (cut) shortSlab(d) else null;
        for (slab_plan.slabs(d), 0..) |sb, k| if (sb.n > 0) {
            const m = if (short != null and k == short.?) sb.n - 1 else sb.n;
            if (m > 0) c.slab(d, k, sb.start, m) catch |e| return at(fail, d, .{ k, sb.start, m }, "the slab's launch", e);
        };
        c.after(d) catch |e| return at(fail, d, null, "the launch after the slabs", e);
    }
    s.synchronize() catch |e| return at(fail, d, null, "synchronize", e);
    return c.read(d) catch |e| return at(fail, d, null, "read back", e);
}

fn free(x: []u8) void {
    std.heap.page_allocator.free(x);
}

/// Class `c` at every window of `ds`. A window: its inputs prepared; the bytes a run starts from; the baseline one launch; the slabs'; the
/// null control (both runs wrote their outputs, and the reference changes when one input value of the window's last row changes, its
/// inputs prepared again); then, only if it holds, the comparison; the control (a slab one row short) and, for a strided class, the
/// stride control.
pub fn runClass(s: anytype, c: anytype, ds: []const usize, fail: *Failure) !Result {
    var res: Result = .{};
    for (ds) |d| {
        res.windows += 1;
        c.prep(d) catch |e| return at(fail, d, null, "the window's inputs", e);
        c.reset() catch |e| return at(fail, d, null, "reset", e);
        s.synchronize() catch |e| return at(fail, d, null, "synchronize", e);
        const start = c.read(d) catch |e| return at(fail, d, null, "read the start bytes", e);
        defer free(start);
        const ref = try once(s, c, d, false, false, fail);
        defer free(ref);
        const got = try once(s, c, d, true, false, fail);
        defer free(got);
        // the null control, before the comparison
        var why: ?NullFail = null;
        if (!wrote(start, ref)) why = .reference_unwritten else if (!wrote(start, got)) why = .slabs_unwritten;
        if (why == null) {
            s.synchronize() catch |e| return at(fail, d, null, "synchronize", e);
            c.perturb(d, true) catch |e| return at(fail, d, null, "change an input value", e);
            c.prep(d) catch |e| return at(fail, d, null, "the window's changed inputs", e);
            const moved = try once(s, c, d, false, false, fail);
            defer free(moved);
            c.perturb(d, false) catch |e| return at(fail, d, null, "put the input value back", e);
            c.prep(d) catch |e| return at(fail, d, null, "the window's inputs again", e);
            if (differs(ref, moved) == 0) why = .reference_deaf;
        }
        if (why) |w| {
            res.bad += ref.len;
            if (res.first_null == null) res.first_null = .{ .d = d, .why = w };
        } else {
            res.null_ok += 1;
            res.bad += differs(ref, got);
        }
        const cut = try once(s, c, d, true, true, fail);
        res.cut_ok += @intFromBool(differs(ref, cut) > 0);
        free(cut);
        if (@TypeOf(c.*).strided) {
            c.short = 1;
            defer c.short = 0;
            const st = try once(s, c, d, true, false, fail);
            res.stride_ok += @intFromBool(differs(ref, st) > 0);
            free(st);
        }
    }
    return res;
}

// -- host tests: a class in host memory whose launches can be made to do what the replay's did ---------------------------------------

const NoStream = struct {
    pub fn synchronize(_: NoStream) !void {}
};

/// A window's rows as u16 values: its sources (`src`), the inputs its launches read (`staged`, prep's copy of the sources), the slabs'
/// intermediate (`mid`) and the outputs (`out`). the baseline one launch: out = f(staged); the slabs: mid = staged at their rows, then the launch
/// after them: out = f(mid). The switches make it misbehave the ways a replay can.
const Fake = struct {
    pub const strided = false;
    src: [128]u16 = undefined,
    staged: [128]u16 = @splat(0x1234),
    mid: [128]u16 = @splat(0),
    out: [128]u16 = @splat(0),
    saved: u16 = 0,
    /// prep never copies the sources (be5eb85fe's DecMix / DecPost: the planes and blocks never written)
    no_prep: bool = false,
    /// the launch after the slabs never runs (be5eb85fe's DecExperts: gateUpCodes never launched)
    no_after: bool = false,
    /// the baseline one launch writes nothing
    dead_whole: bool = false,
    /// the slabs' launch takes the row after its own (a real difference the comparison must see)
    off_by_one: bool = false,

    fn f(x: u16) u16 {
        return x *% 40503 +% 12345;
    }
    pub fn prep(c: *Fake, d: usize) !void {
        _ = d;
        if (!c.no_prep) c.staged = c.src;
    }
    pub fn reset(c: *Fake) !void {
        @memset(std.mem.asBytes(&c.out), poison_byte);
        @memset(std.mem.asBytes(&c.mid), poison_byte);
    }
    pub fn whole(c: *Fake, d: usize) !void {
        if (c.dead_whole) return;
        for (0..d) |i| c.out[i] = f(c.staged[i]);
    }
    pub fn slab(c: *Fake, d: usize, k: usize, s: usize, m: usize) !void {
        _ = k;
        for (s..s + m) |i| c.mid[i] = c.staged[if (c.off_by_one) @min(i + 1, d - 1) else i];
    }
    pub fn after(c: *Fake, d: usize) !void {
        if (c.no_after) return;
        for (0..d) |i| c.out[i] = f(c.mid[i]);
    }
    pub fn read(c: *Fake, d: usize) ![]u8 {
        return std.heap.page_allocator.dupe(u8, std.mem.sliceAsBytes(c.out[0..d]));
    }
    pub fn perturb(c: *Fake, d: usize, on: bool) !void {
        if (on) {
            c.saved = c.src[d - 1];
            c.src[d - 1] ^= 0x4000;
        } else c.src[d - 1] = c.saved;
    }
};

fn fake() Fake {
    var c: Fake = .{};
    var prng = std.Random.DefaultPrng.init(7);
    for (&c.src) |*x| x.* = prng.random().int(u16);
    return c;
}

const all_windows = blk: {
    var ds: [128]usize = undefined;
    for (&ds, 1..) |*x, i| x.* = i;
    break :blk ds;
};

test "a class that runs as it must: every window's slabs equal the baseline launch, the null control and the control hold in every window" {
    var c = fake();
    var fail: Failure = .{};
    const r = try runClass(NoStream{}, &c, &all_windows, &fail);
    try std.testing.expectEqual(@as(usize, 0), r.bad);
    try std.testing.expectEqual(@as(usize, 128), r.null_ok);
    try std.testing.expectEqual(@as(usize, 128), r.cut_ok);
    try std.testing.expect(r.first_null == null);
}

test "RED on the replay of be5eb85fe: inputs never prepared compare equal, and only the null control sees that the baseline launch reads nothing" {
    // the comparison alone: both runs read the same stale inputs and agree
    var old = fake();
    old.no_prep = true;
    try old.prep(5);
    var f0: Failure = .{};
    const a = try once(NoStream{}, &old, 5, false, false, &f0);
    defer free(a);
    const b = try once(NoStream{}, &old, 5, true, false, &f0);
    defer free(b);
    try std.testing.expectEqual(@as(usize, 0), differs(a, b));
    // the null control names it, in the first window, and the bytes line cannot read "as it must"
    var c = fake();
    c.no_prep = true;
    var fail: Failure = .{};
    const r = try runClass(NoStream{}, &c, &all_windows, &fail);
    try std.testing.expectEqual(NullFail.reference_deaf, r.first_null.?.why);
    try std.testing.expectEqual(@as(usize, 1), r.first_null.?.d);
    try std.testing.expectEqual(@as(usize, 0), r.null_ok);
    try std.testing.expect(r.bad != 0);
}

test "RED on the replay of be5eb85fe: the launch after the slabs never run leaves the slabs' outputs poisoned, named by the null control" {
    var c = fake();
    c.no_after = true;
    var fail: Failure = .{};
    const r = try runClass(NoStream{}, &c, &all_windows, &fail);
    try std.testing.expectEqual(NullFail.slabs_unwritten, r.first_null.?.why);
    try std.testing.expectEqual(@as(usize, 0), r.null_ok);
}

test "a reference that writes nothing is named, not compared" {
    var c = fake();
    c.dead_whole = true;
    var fail: Failure = .{};
    const r = try runClass(NoStream{}, &c, &all_windows, &fail);
    try std.testing.expectEqual(NullFail.reference_unwritten, r.first_null.?.why);
}

test "control: a real difference passes the null control and shows in the comparison" {
    var c = fake();
    c.off_by_one = true;
    var fail: Failure = .{};
    const r = try runClass(NoStream{}, &c, &all_windows, &fail);
    try std.testing.expectEqual(@as(usize, 128), r.null_ok);
    try std.testing.expect(r.bad != 0);
}

test "wrote: poison, zeros and too few changed bytes are not written; data is" {
    var p: [64]u8 = @splat(poison_byte);
    try std.testing.expect(!wrotePoisoned(&p));
    var z: [64]u8 = @splat(0);
    try std.testing.expect(!wrotePoisoned(&z));
    // zeros from poison changed every byte, but are all zero: not written
    try std.testing.expect(!wrote(&p, &z));
    var half = p;
    for (half[0..32]) |*x| x.* = 1;
    try std.testing.expect(wrotePoisoned(&half));
    var less = p;
    for (less[0..31]) |*x| x.* = 1;
    try std.testing.expect(!wrotePoisoned(&less));
    var start: [64]u8 = undefined;
    for (&start, 0..) |*x, i| x.* = @intCast(i);
    try std.testing.expect(!wrote(&start, &start));
}

test "RED on be5eb85fe's dispatch itself: a private prep and after behind @hasDecl never run, and the null control's checks fail" {
    // be5eb85fe's classes: prep and after private, dispatched by @hasDecl (which does not see them)
    const Old = struct {
        inner: Fake,
        fn prep(c: *@This(), d: usize) !void {
            try c.inner.prep(d);
        }
        fn after(c: *@This(), d: usize) !void {
            try c.inner.after(d);
        }
    };
    var o: Old = .{ .inner = fake() };
    const d = 9;
    if (@hasDecl(Old, "prep")) try o.prep(d);
    try std.testing.expect(!@hasDecl(Old, "prep") and !@hasDecl(Old, "after"));
    try o.inner.reset();
    const start = try o.inner.read(d);
    defer free(start);
    try o.inner.whole(d);
    const ref = try o.inner.read(d);
    defer free(ref);
    try o.inner.reset();
    for (slab_plan.slabs(d), 0..) |sb, k| if (sb.n > 0) try o.inner.slab(d, k, sb.start, sb.n);
    if (@hasDecl(Old, "after")) try o.after(d);
    const got = try o.inner.read(d);
    defer free(got);
    // the slabs' output is still the poison: not written
    try std.testing.expect(wrote(start, ref));
    try std.testing.expect(!wrote(start, got));
    // and the reference does not move with its source: the inputs were never prepared
    try o.inner.perturb(d, true);
    if (@hasDecl(Old, "prep")) try o.prep(d);
    try o.inner.reset();
    try o.inner.whole(d);
    const moved = try o.inner.read(d);
    defer free(moved);
    try std.testing.expectEqual(@as(usize, 0), differs(ref, moved));
}
