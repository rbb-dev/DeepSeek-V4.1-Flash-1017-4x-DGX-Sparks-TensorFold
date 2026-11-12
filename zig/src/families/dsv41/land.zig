//! The landing: a state rank 0 restored from its store (its kept buffer and its extent's compressed rows) given to
//! every other rank before the request is admitted. Rank 0 sends snap_land frames on the ordered command stream; every
//! rank votes once at the open and takes every piece's broadcast, so no rank is left out of a collective. Followers
//! write the rows at rank 0's base in their own pools. Host logic only: lanes.zig binds the Hooks to the engine.
const std = @import("std");

/// A run of device memory: a landing moves its spans' bytes in order, as one stream.
pub const Span = struct { addr: u64, len: usize };

/// A compressed plane as copyExtent copies it (Model.SpillRefPlane): its address, row bytes and ratio.
pub const Plane = struct { plane: u64, row: usize, ratio: usize };

/// The most spans a landing holds: the kept buffer and four planes a layer (spill_state.max_layers).
pub const max_spans = 1 + 4 * 64;

/// The most landing buffers a rank keeps (TF_DS_SPILL_LAND_SLOTS's range).
pub const max_slots = 16;

/// A landing's spans in order: the kept buffer (snapCopy's layout, `small_bytes`), then each plane's rows of the
/// extent [base, base + at) in copyExtent's order (at / ratio rows from row base / ratio).
pub fn plan(buf: u64, small_bytes: usize, planes: []const Plane, base: usize, at: usize, out: []Span) error{ BadView, TooManySpans }![]Span {
    if (out.len < 1 + planes.len) return error.TooManySpans;
    out[0] = .{ .addr = buf, .len = small_bytes };
    for (planes, out[1 .. 1 + planes.len]) |p, *o| {
        if (p.ratio == 0 or base % p.ratio != 0 or at % p.ratio != 0) return error.BadView;
        o.* = .{ .addr = p.plane + base / p.ratio * p.row, .len = at / p.ratio * p.row };
    }
    return out[0 .. 1 + planes.len];
}

pub fn total(spans: []const Span) usize {
    var n: usize = 0;
    for (spans) |s| n += s.len;
    return n;
}

/// The sub-spans of `spans` that hold bytes [off, off + len) of their concatenation, in order.
pub fn cut(spans: []const Span, off: usize, len: usize, out: []Span) error{ OutOfRange, TooManySpans }![]Span {
    if (off + len > total(spans)) return error.OutOfRange;
    var k: usize = 0;
    var base: usize = 0;
    for (spans) |s| {
        defer base += s.len;
        const lo = @max(off, base);
        const hi = @min(off + len, base + s.len);
        if (lo >= hi) continue;
        if (k == out.len) return error.TooManySpans;
        out[k] = .{ .addr = s.addr + (lo - base), .len = hi - lo };
        k += 1;
    }
    return out[0..k];
}

/// The snap_land frame's operations, sent by rank 0 to every follower in the command stream's order.
pub const Frame = union(enum) {
    /// Take a landing buffer for `id`, check this rank's stream is `total` bytes, then vote.
    open: struct { id: u64, base: u64, at: u64, total: u64 },
    /// Bytes [off, off + len) of the landing's stream, broadcast from rank 0.
    piece: struct { id: u64, off: u64, len: u64 },
    /// The landing buffer becomes kept state `id` (snap_restore and snap_drop name it from here).
    adopt: struct { id: u64 },
    /// The landing ends without adopting: the buffer goes back to the pool.
    abort: struct { id: u64 },
    /// Every rank hashes kept state `id` and the extent [base, base + at) (probe.zig: `k` sampled rows a plane from
    /// `seed`), and the digests are all-gathered. Not a landing: lanes.zig runs it.
    probe: struct { id: u64, base: u64, at: u64, seed: u64, k: u32 },
};

/// What a landing needs of its rank. Followers use `vote` and `bcast` only.
pub const Hooks = struct {
    ctx: *anyopaque,
    /// Rank 0: one frame to every follower, after every frame sent before it.
    frame: *const fn (ctx: *anyopaque, f: Frame) anyerror!void,
    /// Every rank: true when every rank voted ready (a one-byte all-reduce; it waits for this rank's stream).
    vote: *const fn (ctx: *anyopaque, ok: bool) anyerror!bool,
    /// Every rank: rank 0's bytes of these spans into each follower's (one group of broadcasts, enqueued in order).
    bcast: *const fn (ctx: *anyopaque, spans: []const Span) anyerror!void,
    /// Rank 0: its stream has no work queued (an open's vote waits for the stream: only at an idle service point).
    idle: *const fn (ctx: *anyopaque) bool,
    /// Rank 0: mark the piece just enqueued (an event), and whether that piece has completed.
    mark: *const fn (ctx: *anyopaque) anyerror!void,
    marked: *const fn (ctx: *anyopaque) bool,
    /// Rank 0: wait for the marked piece (an abort, before rank 0's buffer can be freed).
    wait: *const fn (ctx: *anyopaque) void,
};

/// A landing step's outcome for the tier's poll.
pub const Step = enum {
    /// Not done: ask again at the next service point.
    pending,
    /// Every rank holds the state (the followers adopted it under rank 0's id).
    landed,
    /// A rank could not take it (no free buffer, another size): every rank skipped, rank 0 discards its destination.
    refused,
    /// A frame or a collective failed (an engine fault), or no landing was in flight.
    failed,
};

/// Rank 0's side: one landing at a time, driven by `step` at service points.
pub const Lander = struct {
    hooks: Hooks,
    /// Bytes a piece (TF_DS_SPILL_LAND_PIECE_MIB).
    piece: usize,
    counts: Counts = .{},
    state: enum { none, planned, open } = .none,
    id: u64 = 0,
    base: u64 = 0,
    at: u64 = 0,
    total: usize = 0,
    sent: usize = 0,
    n: usize = 0,
    spans_buf: [max_spans]Span = undefined,
    cut_buf: [max_spans]Span = undefined,

    pub const Counts = struct {
        landings: u64 = 0,
        skipped: u64 = 0,
        aborted: u64 = 0,
        failed: u64 = 0,
        pieces: u64 = 0,
        bytes: u64 = 0,
    };

    /// A landing of state `id`: rank 0's kept buffer `buf` and the extent rows [base, base + at) of `planes`. Nothing
    /// is sent until `step`.
    pub fn begin(l: *Lander, id: u64, buf: u64, small_bytes: usize, planes: []const Plane, base: usize, at: usize) !void {
        if (l.state != .none) return error.LandingInFlight;
        if (l.piece == 0) return error.BadPieceSize;
        const s = try plan(buf, small_bytes, planes, base, at, &l.spans_buf);
        l.state = .planned;
        l.id = id;
        l.base = base;
        l.at = at;
        l.total = total(s);
        l.n = s.len;
        l.sent = 0;
    }

    pub fn inFlight(l: *const Lander) bool {
        return l.state != .none;
    }

    /// One step: the open and its vote (only when rank 0's stream is idle), then a piece when the one before it has
    /// completed, then the adopt once the last one has.
    pub fn step(l: *Lander) Step {
        const h = l.hooks;
        if (l.state == .none) return .failed;
        if (l.state == .planned) {
            if (!h.idle(h.ctx)) return .pending;
            h.frame(h.ctx, .{ .open = .{ .id = l.id, .base = l.base, .at = l.at, .total = l.total } }) catch return l.fail();
            const ok = h.vote(h.ctx, true) catch return l.fail();
            if (!ok) {
                l.state = .none;
                l.counts.skipped += 1;
                return .refused;
            }
            l.state = .open;
        }
        if (!h.marked(h.ctx)) return .pending; // one piece in flight at most
        if (l.sent < l.total) {
            const len = @min(l.piece, l.total - l.sent);
            const sub = cut(l.spans_buf[0..l.n], l.sent, len, &l.cut_buf) catch return l.fail();
            h.frame(h.ctx, .{ .piece = .{ .id = l.id, .off = l.sent, .len = len } }) catch return l.fail();
            h.bcast(h.ctx, sub) catch return l.fail();
            h.mark(h.ctx) catch return l.fail();
            l.sent += len;
            l.counts.pieces += 1;
            l.counts.bytes += len;
            return .pending;
        }
        h.frame(h.ctx, .{ .adopt = .{ .id = l.id } }) catch return l.fail();
        l.state = .none;
        l.counts.landings += 1;
        return .landed;
    }

    /// Stop the landing in flight (its request cancelled, a shutdown): the followers give their buffers back. Waits
    /// for the piece in flight, so rank 0's buffer may be freed after it returns.
    pub fn abort(l: *Lander) void {
        switch (l.state) {
            .none => return,
            .planned => {},
            .open => {
                l.hooks.wait(l.hooks.ctx);
                l.hooks.frame(l.hooks.ctx, .{ .abort = .{ .id = l.id } }) catch {
                    l.counts.failed += 1;
                };
            },
        }
        l.state = .none;
        l.counts.aborted += 1;
    }

    fn fail(l: *Lander) Step {
        l.state = .none;
        l.counts.failed += 1;
        return .failed;
    }
};

/// A follower's side: its pool of landing buffers (each `small_bytes`, allocated at its start, so a landing that
/// passed the vote never allocates) and the landing in flight. `Buf` has `ptr: u64` (cuda.DeviceBuffer).
pub fn Follower(comptime Buf: type) type {
    return struct {
        const Self = @This();

        /// How the pool makes and frees its buffers.
        pub const Mem = struct {
            ctx: *anyopaque,
            alloc: *const fn (ctx: *anyopaque) anyerror!Buf,
            free: *const fn (ctx: *anyopaque, b: Buf) void,
        };
        /// A landed state for the caller's kept map.
        pub const Adopted = struct { id: u64, buf: Buf };

        mem: Mem,
        small_bytes: usize,
        planes: []const Plane,
        slots: []?Buf,
        active: ?struct { id: u64, buf: Buf, slot: usize } = null,
        /// Buffers the pool could not make (refill): it holds fewer, and an open finding none votes not ready.
        refill_failed: u64 = 0,
        n: usize = 0,
        spans_buf: [max_spans]Span = undefined,
        cut_buf: [max_spans]Span = undefined,

        /// `count` buffers of `small_bytes` (a failed allocation is counted and the pool holds fewer); `planes` is this
        /// rank's own, in copyExtent's order, and outlives the follower.
        pub fn init(gpa: std.mem.Allocator, mem: Mem, small_bytes: usize, planes: []const Plane, count: usize) !Self {
            const slots = try gpa.alloc(?Buf, count);
            @memset(slots, null);
            var f: Self = .{ .mem = mem, .small_bytes = small_bytes, .planes = planes, .slots = slots };
            f.refill();
            return f;
        }

        pub fn deinit(f: *Self, gpa: std.mem.Allocator) void {
            for (f.slots) |s| if (s) |b| f.mem.free(f.mem.ctx, b);
            if (f.active) |a| f.mem.free(f.mem.ctx, a.buf);
            gpa.free(f.slots);
            f.* = undefined;
        }

        /// Free buffers in the pool.
        pub fn free(f: *const Self) usize {
            var n: usize = 0;
            for (f.slots) |s| n += @intFromBool(s != null);
            return n;
        }

        /// Make the buffers the pool lacks (after an adopt took one; never inside a collective).
        pub fn refill(f: *Self) void {
            for (f.slots) |*s| if (s.* == null and !f.slotActive(s)) {
                s.* = f.mem.alloc(f.mem.ctx) catch {
                    f.refill_failed += 1;
                    return;
                };
            };
        }

        fn slotActive(f: *const Self, s: *const ?Buf) bool {
            const a = f.active orelse return false;
            return &f.slots[a.slot] == s;
        }

        /// One snap_land frame: an open votes (always: a follower with no buffer votes not ready), a piece takes its
        /// broadcast (always, once the vote passed), an adopt hands the buffer to the caller, an abort takes it back.
        pub fn onFrame(f: *Self, h: Hooks, fr: Frame) !?Adopted {
            switch (fr) {
                .open => |o| {
                    if (f.active != null) return error.LandingInFlight;
                    var ready = false;
                    for (f.slots, 0..) |s, i| if (s) |b| {
                        if (plan(b.ptr, f.small_bytes, f.planes, o.base, o.at, &f.spans_buf)) |sp| {
                            f.n = sp.len;
                            ready = total(sp) == o.total;
                        } else |_| {}
                        if (ready) {
                            f.slots[i] = null;
                            f.active = .{ .id = o.id, .buf = b, .slot = i };
                        }
                        break;
                    };
                    if (!try h.vote(h.ctx, ready)) f.giveBack();
                    return null;
                },
                .piece => |p| {
                    const a = f.active orelse return error.LandingNotOpen;
                    if (a.id != p.id) return error.LandingNotOpen;
                    // the vote checked this rank's total, so a piece always maps; one that does not fails loudly
                    const sub = cut(f.spans_buf[0..f.n], p.off, p.len, &f.cut_buf) catch return error.LandingPieceOutOfRange;
                    try h.bcast(h.ctx, sub);
                    return null;
                },
                .adopt => |x| {
                    const a = f.active orelse return error.LandingNotOpen;
                    if (a.id != x.id) return error.LandingNotOpen;
                    f.active = null;
                    return .{ .id = a.id, .buf = a.buf };
                },
                .abort => |x| {
                    const a = f.active orelse return error.LandingNotOpen;
                    if (a.id != x.id) return error.LandingNotOpen;
                    f.giveBack();
                    return null;
                },
                .probe => return error.ProbeIsNotALanding, // (lanes.zig takes it before this)
            }
        }

        fn giveBack(f: *Self) void {
            const a = f.active orelse return;
            f.slots[a.slot] = a.buf;
            f.active = null;
        }
    };
}
