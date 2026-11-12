//! The bounded issue lead of the prompt passes (TF_DS_ISSUE_LEAD=N): the host issues a layer only once the GPU has started the
//! layer `N` before it. Without it the host runs a whole pass ahead of the GPU, the launch queue fills, and the driver holds the
//! engine thread inside cuLaunchKernel (a busy-wait: one core a rank at 100 % for the whole prompt). With it the wait is on the
//! start event of the oldest layer queued, an event made to sleep (Event.initBlocking), and no launch call waits for room.
//!
//! An event recorded at the start of each layer says when the GPU got there; a ring of them serves, the gate counts the layers
//! issued and the layers started. The gate only ever delays an issue: the order of the launches, the tickets a transport takes
//! and the frames a rank sends are the ungated pass's. It waits only for the GPU to reach a layer whose every launch and send
//! the host has already made, so what it waits for never depends on this host's later work: across ranks, layer L's issue waits
//! for the start of layer L - N, which needs every rank's issue of the layers before L - N (the exchanges are per layer), and
//! those are strictly earlier in each rank's own order. No decode round, drafter pass or frame runs inside the gate.
const std = @import("std");
const roci_pair = @import("roci_pair.zig");

/// Events in the ring: more than the largest lead, so an event is never recorded again before the gate has counted it started.
pub const ring = 32;

/// The largest lead a setting may name.
pub const max_lead = ring - 1;

/// TF_DS_ISSUE_LEAD's text: unset, empty or "0" is off, 1 to 31 the lead in layers; anything else is no setting.
pub const Setting = union(enum) { off, on: u32, invalid };

pub fn setting(text: ?[]const u8) Setting {
    const v = std.mem.trim(u8, text orelse return .off, " \t\r\n");
    if (v.len == 0) return .off;
    const n = std.fmt.parseInt(u32, v, 10) catch return .invalid;
    if (n == 0) return .off;
    return if (n <= max_lead) .{ .on = n } else .invalid;
}

/// How the host waits for the oldest layer's start: `poll` sleeps a slice, queries the event and looks at the transports between
/// slices (a host that sleeps in one blocking wait notices nothing that fails meanwhile); `block` is one blocking-sync wait.
pub const Wait = enum { poll, block, backoff };

/// The longest nap `backoff` grows to, in microseconds: a layer of a 4,096-row pass runs ~14.6 ms, so the start of the layer the gate waits
/// for is seen at most this late, and the transport check and the Engram worker's slots are served at least this often.
pub const backoff_cap_us: u32 = 1000;

/// The nap after `cur` in a `backoff` wait: double, never past `cap`, and never shorter than the setting's own slice (a slice above
/// the cap is kept).
pub fn backoffNext(cur: u32, cap: u32) u32 {
    if (cur >= cap) return cur;
    return @min(cur *| 2, cap);
}

/// TF_DS_ISSUE_LEAD_WAIT's text: unset or empty is poll; "poll", "block" and "backoff"; anything else is no setting (null).
pub fn waitSetting(text: ?[]const u8) ?Wait {
    const v = std.mem.trim(u8, text orelse return .poll, " \t\r\n");
    if (v.len == 0 or std.mem.eql(u8, v, "poll")) return .poll;
    if (std.mem.eql(u8, v, "block")) return .block;
    if (std.mem.eql(u8, v, "backoff")) return .backoff;
    return null;
}

/// The default slice of a polled wait, in microseconds, and the most a setting may name.
pub const slice_default_us: u32 = 100;
pub const slice_max_us: u32 = 100_000;

/// TF_DS_ISSUE_LEAD_SLICE_US's text: unset or empty is 100; 1 to 100000 microseconds; anything else is no setting (null).
pub fn sliceSetting(text: ?[]const u8) ?u32 {
    const v = std.mem.trim(u8, text orelse return slice_default_us, " \t\r\n");
    if (v.len == 0) return slice_default_us;
    const n = std.fmt.parseInt(u32, v, 10) catch return null;
    return if (n >= 1 and n <= slice_max_us) n else null;
}

/// TF_DS_ISSUE_LEAD_WARN_MS's text: unset or empty is 2000; 0 is never; 1 to 3600000 milliseconds; anything else no setting (null).
pub const warn_default_ms: u32 = 2000;

pub fn warnSetting(text: ?[]const u8) ?u32 {
    const v = std.mem.trim(u8, text orelse return warn_default_ms, " \t\r\n");
    if (v.len == 0) return warn_default_ms;
    const n = std.fmt.parseInt(u32, v, 10) catch return null;
    return if (n <= 3_600_000) n else null;
}

/// When a long wait says so: once it has lasted `warn_ns`, then at most every `every_ns` (never fatal: the line names the place, the
/// transport's watchdogs decide what is a failure). `warn_ns` 0 never says anything.
pub const Deadline = struct {
    warn_ns: u64,
    every_ns: u64 = 5 * std.time.ns_per_s,
    last: ?u64 = null,

    pub fn due(d: *Deadline, waited_ns: u64, now_ns: u64) bool {
        if (d.warn_ns == 0 or waited_ns < d.warn_ns) return false;
        if (d.last) |l| if (now_ns -| l < d.every_ns) return false;
        d.last = now_ns;
        return true;
    }
};

/// What one sleep cost: the nanoseconds the host slept, and the slices of a polled wait (0 for a blocking one).
pub const Slept = struct { ns: u64, slices: u64 };

/// What the gate did since the last `take`: the layers it issued, the times it had to sleep, for how long and in how many slices, and
/// the layers it was entered at with nothing queued behind the layer the GPU was running (the GPU may have gone idle there: the
/// lead is too small, or the host too slow, at those layers).
/// `bare_first` counts those entries that were a pass's first layer (the queue the previous pass left, drained while the host
/// prepared this one: the boundary, not the lead), `bare` those inside a pass.
pub const Stats = struct { layers: u64 = 0, sleeps: u64 = 0, sleep_ns: u64 = 0, slices: u64 = 0, bare: u64 = 0, bare_first: u64 = 0 };

/// The ring of start events, made once and freed once: `make` frees what a former `make` left and, if one creation fails, frees the
/// ones made and leaves the ring empty; `free` is safe on an empty ring and on a ring already freed. `ctx` has `create() !E` and
/// `destroy(*E) void`.
pub fn Events(comptime E: type) type {
    return struct {
        ev: [ring]?E = @splat(null),

        pub fn make(s: *@This(), ctx: anytype) !void {
            s.free(ctx);
            errdefer s.free(ctx);
            for (&s.ev) |*e| e.* = try ctx.create();
        }

        pub fn free(s: *@This(), ctx: anytype) void {
            for (&s.ev) |*e| {
                if (e.*) |*x| ctx.destroy(x);
                e.* = null;
            }
        }
    };
}

pub const Gate = struct {
    n: u32, // the most layers issued and not yet seen started
    issued: u64 = 0,
    lo: u64 = 0, // every layer before it has been seen started
    pending: [ring]bool = @splat(false), // by ring slot: issued and not yet seen started
    npend: u32 = 0,
    stats: Stats = .{},
    first: bool = false, // the next entry is a pass's first layer

    pub fn init(n: u32) Gate {
        std.debug.assert(n >= 1 and n <= max_lead);
        return .{ .n = n };
    }

    /// Layers issued whose start the GPU has not been seen to reach. Counted over every pending slot, not as an oldest-first
    /// prefix: two lanes start their layers out of issue order (B of one layer before A of the next), and a prefix would call a layer
    /// queued that has started, leaving a lane short of its lead.
    pub fn ahead(g: *const Gate) u64 {
        return g.npend;
    }

    /// Layers seen started.
    pub fn startedCount(g: *const Gate) u64 {
        return g.issued - g.npend;
    }

    /// Whether the next layer may be issued now: fewer than `n` are queued, and the ring slot it takes is free (its layer from `ring`
    /// layers back was seen started: recording the slot's event again cannot touch a pending one).
    pub fn mayIssue(g: *const Gate) bool {
        return g.npend < g.n and !g.pending[@intCast(g.issued % ring)];
    }

    /// The ring slot of the next layer's event; the layer counts as issued. Only after `mayIssue`.
    pub fn issue(g: *Gate) usize {
        std.debug.assert(g.mayIssue());
        const slot: usize = @intCast(g.issued % ring);
        g.pending[slot] = true;
        g.npend += 1;
        g.issued += 1;
        g.stats.layers += 1;
        return slot;
    }

    /// The absolute number of the layer a pending slot holds.
    pub fn layerOf(g: *const Gate, slot: usize) u64 {
        std.debug.assert(g.pending[slot]);
        const back: u64 = (g.issued - 1 - slot % ring + ring) % ring;
        return g.issued - 1 - back;
    }

    /// The slot of the oldest layer not seen started, or null when every one has been.
    pub fn oldest(g: *const Gate) ?usize {
        var i = g.lo;
        while (i < g.issued) : (i += 1) {
            const slot: usize = @intCast(i % ring);
            if (g.pending[slot]) return slot;
        }
        return null;
    }

    /// Counts the layers the GPU has started, in any order: every pending slot `ctx.done(slot)` says has fired.
    pub fn poll(g: *Gate, ctx: anytype) !void {
        var i = g.lo;
        while (i < g.issued) : (i += 1) {
            const slot: usize = @intCast(i % ring);
            if (g.pending[slot] and try ctx.done(slot)) {
                g.pending[slot] = false;
                g.npend -= 1;
            }
        }
        while (g.lo < g.issued and !g.pending[@intCast(g.lo % ring)]) g.lo += 1;
    }

    /// The next entry is the first layer of a pass or pair.
    pub fn beginPass(g: *Gate) void {
        g.first = true;
    }

    /// The gate's work before an issue, with `ctx.done(slot)` (the slot's event has fired) and `ctx.sleep(g)` (the host sleeps
    /// until the gate may issue; it returns a `Slept`): the started layers counted, then, while the lead is at its bound,
    /// a sleep on the oldest layer's event. Returns the slot to record this layer's start event in.
    pub fn enter(g: *Gate, ctx: anytype) !usize {
        try g.poll(ctx);
        if (g.issued > 0 and g.ahead() == 0) {
            if (g.first) g.stats.bare_first += 1 else g.stats.bare += 1;
        }
        g.first = false;
        while (!g.mayIssue()) {
            g.stats.sleeps += 1;
            const slept: Slept = try ctx.sleep(g);
            g.stats.sleep_ns += slept.ns;
            g.stats.slices += slept.slices;
            try g.poll(ctx);
        }
        return g.issue();
    }

    /// The counters since the last call.
    pub fn take(g: *Gate) Stats {
        const s = g.stats;
        g.stats = .{};
        return s;
    }
};

/// The polled wait: until the gate may issue, a nap of a slice, then, each slice, a look at the transports (`ctx.check()`, which fails
/// the wait if one has), the host's own service of what only it can do while the GPU runs (`ctx.service()`: the Engram worker's pinned
/// slots go back to it only when the engine thread polls them), and a count of the layers started. Returns the slices slept.
pub fn pollWait(g: *Gate, ctx: anytype) !u64 {
    var slices: u64 = 0;
    while (!g.mayIssue()) {
        try ctx.nap();
        slices += 1;
        try ctx.check();
        try ctx.service();
        try g.poll(ctx);
    }
    return slices;
}

// -- tests -----------------------------------------------------------------------------------------------------------

/// A GPU that starts a layer when it is issued and the stream is free, runs it `layer_us`, and the host's clock: what the gate's
/// `done` and `sleep` read.
const SimGpu = struct {
    now: u64 = 0,
    starts: [8192]u64 = undefined,
    ends: [8192]u64 = undefined,
    n: usize = 0,
    slept: u64 = 0,

    // (the gate's slot is issued % ring: the sim keeps the layer numbers the slots stand for)
    pub fn done(s: *SimGpu, g: *const Gate, slot: usize) bool {
        return s.starts[@intCast(g.layerOf(slot))] <= s.now;
    }
};

const SimCtx = struct {
    gpu: *SimGpu,
    g: *Gate,
    pub fn done(c: SimCtx, slot: usize) !bool {
        return c.gpu.done(c.g, slot);
    }
    pub fn sleep(c: SimCtx, g: *Gate) !Slept {
        _ = g;
        // (a polled wait wakes at the first layer to start, in any order)
        var t: u64 = std.math.maxInt(u64);
        var i = c.g.lo;
        while (i < c.g.issued) : (i += 1) {
            if (c.g.pending[@intCast(i % ring)]) t = @min(t, c.gpu.starts[@intCast(i)]);
        }
        const d = if (t > c.gpu.now) t - c.gpu.now else 0;
        c.gpu.now += d;
        c.gpu.slept += d;
        return .{ .ns = d, .slices = 1 + d / 100_000 };
    }
};

test "the setting: unset, empty and 0 are off, 1 to 31 the lead, anything else no setting" {
    try std.testing.expectEqual(Setting.off, setting(null));
    try std.testing.expectEqual(Setting.off, setting(""));
    try std.testing.expectEqual(Setting.off, setting(" "));
    try std.testing.expectEqual(Setting.off, setting("0"));
    try std.testing.expectEqual(Setting{ .on = 1 }, setting("1"));
    try std.testing.expectEqual(Setting{ .on = 8 }, setting(" 8\n"));
    try std.testing.expectEqual(Setting{ .on = 31 }, setting("31"));
    for ([_][]const u8{ "32", "-1", "x", "1.5", "10 layers", "0x10" }) |t| try std.testing.expectEqual(Setting.invalid, setting(t));
}

test "the lead is bounded: at most n layers queued, the GPU never idles, the host sleeps instead of issuing" {
    // 12 passes of 43 layers at 20 ms a layer; the host issues a layer in 60 us
    inline for (.{ 1, 2, 10, 31 }) |n| {
        var g = Gate.init(n);
        var gpu: SimGpu = .{};
        var max_ahead: u64 = 0;
        var idle: u64 = 0;
        const layers = 43 * 12;
        while (gpu.n < layers) {
            const before = g.issued;
            _ = try g.enter(SimCtx{ .gpu = &gpu, .g = &g });
            try std.testing.expectEqual(before + 1, g.issued);
            const start = if (gpu.n == 0) gpu.now else @max(gpu.now, gpu.ends[gpu.n - 1]);
            if (gpu.n > 0 and gpu.now > gpu.ends[gpu.n - 1]) idle += gpu.now - gpu.ends[gpu.n - 1];
            gpu.starts[gpu.n] = start;
            gpu.ends[gpu.n] = start + 20_000;
            gpu.n += 1;
            gpu.now += 60;
            max_ahead = @max(max_ahead, g.ahead());
        }
        try std.testing.expect(max_ahead <= n);
        try std.testing.expectEqual(@as(u64, 0), idle);
        const s = g.take();
        try std.testing.expectEqual(@as(u64, layers), s.layers);
        try std.testing.expect(s.sleeps > 0 and s.sleep_ns > 0 and s.slices >= s.sleeps); // the host slept where the queue would have filled
        try std.testing.expectEqual(@as(u64, 0), g.take().layers); // (take resets)
    }
}

test "a pass shorter than the lead never sleeps, and the bound is the pass" {
    var g = Gate.init(10);
    var gpu: SimGpu = .{};
    for (0..5) |_| {
        _ = try g.enter(SimCtx{ .gpu = &gpu, .g = &g });
        gpu.starts[gpu.n] = (gpu.n + 1) * 20_000; // (a busy GPU: none of the five has started)
        gpu.ends[gpu.n] = (gpu.n + 2) * 20_000;
        gpu.n += 1;
    }
    try std.testing.expectEqual(@as(u64, 5), g.ahead());
    const s = g.take();
    try std.testing.expectEqual(@as(u64, 0), s.sleeps);
    try std.testing.expectEqual(@as(u64, 5), s.layers);
}

test "the ring wraps: a slot is recorded again only after its previous layer is seen started" {
    var g = Gate.init(max_lead);
    var gpu: SimGpu = .{};
    for (0..1000) |_| {
        const slot = try g.enter(SimCtx{ .gpu = &gpu, .g = &g });
        // the layer this slot held `ring` layers back has been seen started, or enter would still be asleep
        try std.testing.expectEqual(@as(usize, @intCast((g.issued - 1) % ring)), slot);
        try std.testing.expectEqual(g.issued - 1, g.layerOf(slot));
        const start = if (gpu.n == 0) gpu.now else @max(gpu.now, gpu.ends[gpu.n - 1]);
        gpu.starts[gpu.n] = start;
        gpu.ends[gpu.n] = start + 5_000;
        gpu.n += 1;
        gpu.now += 50;
        try std.testing.expect(g.ahead() <= max_lead);
    }
}

test "a layer is counted started in any order: a later event firing first lowers the count, the oldest slot stays the oldest" {
    var g = Gate.init(4);
    var flags: [ring]bool = @splat(false);
    const Ctx = struct {
        d: *[ring]bool,
        pub fn done(c: @This(), slot: usize) !bool {
            return c.d[slot];
        }
        pub fn sleep(c: @This(), gate: *Gate) !Slept {
            c.d[gate.oldest().?] = true;
            return .{ .ns = 7, .slices = 3 };
        }
    };
    const ctx = Ctx{ .d = &flags };
    for (0..4) |k| try std.testing.expectEqual(k, try g.enter(ctx));
    flags[1] = true; // layer 1 started, layer 0 not
    try g.poll(ctx);
    try std.testing.expectEqual(@as(u64, 3), g.ahead());
    try std.testing.expectEqual(@as(?usize, 0), g.oldest());
    // so a fifth layer is admitted at once: three are queued, not four
    try std.testing.expectEqual(@as(usize, 4), try g.enter(ctx));
    try std.testing.expectEqual(@as(u64, 0), g.stats.sleeps);
    // the next finds four queued (layers 0, 2, 3, 4): the gate sleeps on the oldest, layer 0's event, then admits it
    try std.testing.expectEqual(@as(usize, 5), try g.enter(ctx));
    try std.testing.expectEqual(@as(u64, 1), g.stats.sleeps);
    try std.testing.expectEqual(@as(u64, 7), g.stats.sleep_ns);
    try std.testing.expectEqual(@as(u64, 3), g.stats.slices);
}

test "a polled wait looks at the transports and gives the Engram worker's slots back every slice, before it counts the layers" {
    var g = Gate.init(2);
    const Ctx = struct {
        g: *Gate,
        slice: u32 = 0,
        log: [32]u8 = undefined,
        n: usize = 0,
        returned: u32 = 0, // Engram slots handed back while the host slept
        fn note(c: *@This(), ch: u8) void {
            c.log[c.n] = ch;
            c.n += 1;
        }
        pub fn nap(c: *@This()) !void {
            c.note('n');
        }
        pub fn check(c: *@This()) !void {
            c.note('c');
        }
        pub fn service(c: *@This()) !void {
            c.note('s');
            c.returned += 1; // (engramPoll: a copy done is a slot back for the worker)
        }
        pub fn done(c: *@This(), slot: usize) !bool {
            _ = slot;
            c.note('d');
            c.slice += 1;
            return c.slice >= 3 * 2; // the oldest fires in the third slice (two pending slots are asked each time)
        }
    };
    var ctx: Ctx = .{ .g = &g };
    _ = g.issue();
    _ = g.issue();
    try std.testing.expect(!g.mayIssue());
    const slices = try pollWait(&g, &ctx);
    try std.testing.expect(slices >= 3);
    try std.testing.expectEqual(@as(u32, @intCast(slices)), ctx.returned); // a slot returned in every slice of the wait
    // each slice in this order: nap, check, service, then the count of started layers
    var k: usize = 0;
    var seen: u64 = 0;
    while (k + 3 < ctx.n) : (k += 1) {
        if (ctx.log[k] == 'n') {
            try std.testing.expectEqualStrings("ncs", ctx.log[k .. k + 3]);
            seen += 1;
        }
    }
    try std.testing.expect(seen >= 1);
    // an error from the transport check ends the wait at once, before the service and the count
    const Bad = struct {
        pub fn nap(_: *@This()) !void {}
        pub fn check(_: *@This()) !void {
            return error.RociFailed;
        }
        pub fn service(_: *@This()) !void {
            return error.ServiceRan;
        }
        pub fn done(_: *@This(), _: usize) !bool {
            return false;
        }
    };
    var bad: Bad = .{};
    var g2 = Gate.init(1);
    _ = g2.issue();
    try std.testing.expectError(error.RociFailed, pollWait(&g2, &bad));
}

test "the gate counts a layer entered with nothing queued behind the running one" {
    var g = Gate.init(3);
    var gpu: SimGpu = .{};
    const ctx = SimCtx{ .gpu = &gpu, .g = &g };
    _ = try g.enter(ctx); // the first layer: nothing issued before, not counted
    gpu.starts[0] = 0;
    gpu.ends[0] = 100;
    gpu.n = 1;
    gpu.now = 50; // layer 0 has started
    _ = try g.enter(ctx);
    try std.testing.expectEqual(@as(u64, 1), g.stats.bare);
}

/// Drives `pairs` consecutive pairs of `half` layers (roci_pair.pairSchedule, gap `gap`) through the gate on two streams in order,
/// with the pair's cache-event waits as dependencies, and returns the layer order the host issued in. The GPU's start of a layer is
/// known when it is issued (everything it waits for was issued earlier): a host that sleeps on the oldest start cannot wait for
/// work it has not issued.
fn runPairs(comptime half: usize, gap: usize, pairs: usize, n: u32, issued_out: []roci_pair.Step, max_ahead: *u64, lead_wrong: *u32) !usize {
    var g = Gate.init(n);
    var gpu: SimGpu = .{};
    var free = [_]u64{ 0, 0 }; // each stream's end
    var ev_a: [half + 1]u64 = @splat(0); // this pair's A events (end of the layer, boundary last)
    var ev_b: [half + 1]u64 = @splat(0);
    var prev_b: [half + 1]u64 = @splat(0); // the previous pair's B events
    var out: usize = 0;
    var buf: [roci_pair.maxSteps(roci_pair.max_half)]roci_pair.Step = undefined;
    for (0..pairs) |p| {
        const first = p == 0;
        const steps = roci_pair.pairSchedule(half, gap, first, &buf);
        var dep = [_]u64{ 0, 0 };
        for (steps) |st| switch (st) {
            .wait => |w| dep[@backingInt(w.pass)] = if (w.pass == .b) ev_a[w.li] else prev_b[w.li],
            .layer => |l| {
                const si = @backingInt(l.pass);
                _ = try g.enter(SimCtx{ .gpu = &gpu, .g = &g });
                // the gate's count of layers queued is the true one: issued layers (before this) the GPU has not started by now. A prefix
                // count would call a layer queued that started out of issue order (B of one layer before A of the next)
                var truth: u64 = 0;
                for (0..gpu.n) |i| {
                    if (gpu.starts[i] > gpu.now) truth += 1;
                }
                if (g.ahead() != truth + 1) lead_wrong.* += 1; // (+1: this layer, issued by enter)
                max_ahead.* = @max(max_ahead.*, g.ahead());
                // the lead event is recorded after the stream's wait: the GPU "starts" the layer when the wait is met
                const start = @max(free[si], dep[si]);
                gpu.starts[gpu.n] = start;
                gpu.ends[gpu.n] = start + 20_000;
                gpu.n += 1;
                free[si] = start + 20_000;
                if (l.pass == .a) ev_a[l.li] = free[si] else ev_b[l.li] = free[si];
                dep[si] = 0;
                gpu.now += 60;
                issued_out[out] = st;
                out += 1;
            },
            .boundary => |q| {
                const si = @backingInt(q);
                const start = @max(free[si], dep[si]);
                free[si] = start + 1_000;
                if (q == .a) ev_a[half] = free[si] else ev_b[half] = free[si];
                dep[si] = 0;
            },
        };
        prev_b = ev_b;
    }
    return out;
}

test "two lanes through the gate: the layers are issued in the schedule's order, the bound holds on the pair, and the host never waits on work it has not issued" {
    const half = 20;
    inline for (.{ 1, 2, 10, 31 }) |n| {
        var got: [4 * 2 * half]roci_pair.Step = undefined;
        var max_ahead: u64 = 0;
        var lead_wrong: u32 = 0;
        const count = try runPairs(half, 2, 4, n, &got, &max_ahead, &lead_wrong);
        try std.testing.expectEqual(@as(u32, 0), lead_wrong); // the lead is never below what the bound says
        try std.testing.expectEqual(@as(usize, 4 * 2 * half), count);
        try std.testing.expect(max_ahead <= n);
        // the layer steps of the ungated schedule, pair after pair, are exactly what was issued: the gate reorders nothing
        var want: [4 * 2 * half]roci_pair.Step = undefined;
        var w: usize = 0;
        var buf: [roci_pair.maxSteps(roci_pair.max_half)]roci_pair.Step = undefined;
        for (0..4) |p| for (roci_pair.pairSchedule(half, 2, p == 0, &buf)) |st| switch (st) {
            .layer => {
                want[w] = st;
                w += 1;
            },
            else => {},
        };
        try std.testing.expectEqual(w, count);
        for (got[0..count], want[0..w]) |a, b| {
            try std.testing.expectEqual(@backingInt(a.layer.pass), @backingInt(b.layer.pass));
            try std.testing.expectEqual(a.layer.li, b.layer.li);
        }
        // each pass's layers come in order within a pair (the ticket order of a lane is its layers' order)
        var next = [_]usize{ 0, 0 };
        for (got[0..count]) |st| {
            const p = @backingInt(st.layer.pass);
            try std.testing.expectEqual(next[p], st.layer.li);
            next[p] += 1;
            if (next[p] == half) next[p] = 0;
        }
    }
}

test "the wait mode and the slice: unset is poll and 100 us, block is the other mode, anything else no setting" {
    try std.testing.expectEqual(@as(?Wait, .poll), waitSetting(null));
    try std.testing.expectEqual(@as(?Wait, .poll), waitSetting(""));
    try std.testing.expectEqual(@as(?Wait, .poll), waitSetting("poll"));
    try std.testing.expectEqual(@as(?Wait, .block), waitSetting(" block\n"));
    for ([_][]const u8{ "spin", "Block", "1", "poll,block" }) |t| try std.testing.expectEqual(@as(?Wait, null), waitSetting(t));
    try std.testing.expectEqual(@as(?u32, 100), sliceSetting(null));
    try std.testing.expectEqual(@as(?u32, 100), sliceSetting(" "));
    try std.testing.expectEqual(@as(?u32, 1), sliceSetting("1"));
    try std.testing.expectEqual(@as(?u32, 250), sliceSetting("250"));
    try std.testing.expectEqual(@as(?u32, 100_000), sliceSetting("100000"));
    for ([_][]const u8{ "0", "100001", "-5", "x", "1.5", "100 us" }) |t| try std.testing.expectEqual(@as(?u32, null), sliceSetting(t));
}

test "chunks of any length, one that stops at the boundary included, count layers absolutely: a wait is only ever for the record of the layer it names" {
    // A ring slot remembers the absolute number of the layer recorded in it last. The gate asks the slot of a pending layer: the record
    // there must be that layer itself, never an earlier chunk's layer, and never a later one that reused the slot.
    inline for (.{ 1, 2, 5, 31 }) |n| {
        var g = Gate.init(n);
        var gpu: SimGpu = .{};
        var gen: [ring]u64 = @splat(std.math.maxInt(u64)); // (never recorded)
        const Ctx = struct {
            gpu: *SimGpu,
            g: *Gate,
            gen: *[ring]u64,
            wrong: *u32,
            pub fn done(c: @This(), slot: usize) !bool {
                if (c.gen[slot] != c.g.layerOf(slot)) c.wrong.* += 1;
                return c.gpu.done(c.g, slot);
            }
            pub fn sleep(c: @This(), gate: *Gate) !Slept {
                return try (SimCtx{ .gpu = c.gpu, .g = c.g }).sleep(gate);
            }
        };
        var wrong: u32 = 0;
        // chunks of 43 layers, 21 (stopped at the boundary: the rest of its layers never issued), 3 (shorter than n), 43, 7, 1
        for ([_]usize{ 43, 21, 3, 43, 7, 1 }) |len| for (0..len) |_| {
            const slot = try g.enter(Ctx{ .gpu = &gpu, .g = &g, .gen = &gen, .wrong = &wrong });
            gen[slot] = g.issued - 1;
            const start = if (gpu.n == 0) gpu.now else @max(gpu.now, gpu.ends[gpu.n - 1]);
            gpu.starts[gpu.n] = start;
            gpu.ends[gpu.n] = start + 20_000;
            gpu.n += 1;
            gpu.now += 60;
        };
        try std.testing.expectEqual(@as(u32, 0), wrong);
        try std.testing.expectEqual(@as(u64, 43 + 21 + 3 + 43 + 7 + 1), g.issued);
        try std.testing.expect(g.ahead() <= n);
    }
}

test "the events are made once and freed once: a second make frees the first set, a failed make frees what it made, a free after either is harmless" {
    const Ev = struct { id: u32 };
    const Ctx = struct {
        made: u32 = 0,
        destroyed: u32 = 0,
        fail_at: u32 = 1 << 30, // the creation (counted from 0 in the current make) that fails
        calls: u32 = 0,
        live: i32 = 0,
        pub fn create(c: *@This()) !Ev {
            if (c.calls == c.fail_at) return error.OutOfMemory;
            c.calls += 1;
            c.made += 1;
            c.live += 1;
            return .{ .id = c.made };
        }
        pub fn destroy(c: *@This(), e: *Ev) void {
            _ = e;
            c.destroyed += 1;
            c.live -= 1;
        }
    };
    var c: Ctx = .{};
    var ring_ev: Events(Ev) = .{};
    try ring_ev.make(&c);
    try std.testing.expectEqual(@as(u32, ring), c.made);
    try std.testing.expectEqual(@as(i32, ring), c.live);
    // open run again without a close: the first set goes, no leak
    c.calls = 0;
    try ring_ev.make(&c);
    try std.testing.expectEqual(@as(u32, 2 * ring), c.made);
    try std.testing.expectEqual(@as(i32, ring), c.live);
    // close twice: the events go once
    ring_ev.free(&c);
    ring_ev.free(&c);
    try std.testing.expectEqual(@as(i32, 0), c.live);
    try std.testing.expectEqual(c.made, c.destroyed);
    // a failed open leaves nothing for the close after it to destroy again
    c.calls = 0;
    c.fail_at = 7;
    try std.testing.expectError(error.OutOfMemory, ring_ev.make(&c));
    try std.testing.expectEqual(@as(i32, 0), c.live);
    for (ring_ev.ev) |e| try std.testing.expect(e == null);
    const before = c.destroyed;
    ring_ev.free(&c);
    try std.testing.expectEqual(before, c.destroyed);
}

test "an entry at a pass's first layer with nothing queued is the boundary's, not a bubble inside the pass" {
    var g = Gate.init(3);
    var gpu: SimGpu = .{};
    const ctx = SimCtx{ .gpu = &gpu, .g = &g };
    _ = try g.enter(ctx);
    gpu.starts[0] = 0;
    gpu.ends[0] = 100;
    gpu.n = 1;
    gpu.now = 50;
    g.beginPass(); // the next pass: its first layer finds the queue drained
    _ = try g.enter(ctx);
    try std.testing.expectEqual(@as(u64, 1), g.stats.bare_first);
    try std.testing.expectEqual(@as(u64, 0), g.stats.bare);
    gpu.starts[1] = 60;
    gpu.ends[1] = 160;
    gpu.n = 2;
    gpu.now = 70; // inside the pass: layer 1 has started, nothing queued behind it
    _ = try g.enter(ctx);
    try std.testing.expectEqual(@as(u64, 1), g.stats.bare_first);
    try std.testing.expectEqual(@as(u64, 1), g.stats.bare);
}

test "the host's work between two passes drains a short lead: the GPU idles by the gap less the layers left queued" {
    // passes of 43 layers at 20 ms; the host spends 140 ms (the next pass's rows, hashes and reads) between two calls
    const layer_us: u64 = 20_000;
    const gap_us: u64 = 140_000;
    var idle_at = [_]u64{ 0, 0, 0, 0 };
    inline for (.{ 2, 4, 8, 31 }, 0..) |n, k| {
        var g = Gate.init(n);
        var gpu: SimGpu = .{};
        var idle: u64 = 0;
        for (0..3) |pass| {
            if (pass > 0) gpu.now += gap_us;
            g.beginPass();
            for (0..43) |_| {
                _ = try g.enter(SimCtx{ .gpu = &gpu, .g = &g });
                const start = if (gpu.n == 0) gpu.now else @max(gpu.now, gpu.ends[gpu.n - 1]);
                if (gpu.n > 0 and gpu.now > gpu.ends[gpu.n - 1]) idle += gpu.now - gpu.ends[gpu.n - 1];
                gpu.starts[gpu.n] = start;
                gpu.ends[gpu.n] = start + layer_us;
                gpu.n += 1;
                gpu.now += 60;
            }
        }
        idle_at[k] = idle / 2; // (two gaps)
    }
    // the last issue waited for layer L - N to start, so the GPU then holds that layer and the N queued: (N + 1) x 20 ms of work, and the gap beyond it is idle
    try std.testing.expect(idle_at[0] >= 80_000 - 2_000 and idle_at[0] <= 100_000); // N = 2: ~80 ms
    try std.testing.expect(idle_at[1] >= 40_000 - 2_000 and idle_at[1] <= 60_000); // N = 4: ~40 ms
    try std.testing.expectEqual(@as(u64, 0), idle_at[2]); // N = 8: 180 ms of work covers the gap
    try std.testing.expectEqual(@as(u64, 0), idle_at[3]);
    try std.testing.expect(idle_at[0] > idle_at[1] and idle_at[1] > idle_at[2]);
}

test "the deadline log: nothing before the warning, then one line, then at most one every 5 s; 0 never" {
    var d: Deadline = .{ .warn_ns = 2 * std.time.ns_per_s };
    const s1 = std.time.ns_per_s;
    try std.testing.expect(!d.due(1 * s1, 1 * s1));
    try std.testing.expect(d.due(2 * s1, 2 * s1)); // the first
    try std.testing.expect(!d.due(3 * s1, 3 * s1));
    try std.testing.expect(!d.due(6 * s1, 6 * s1)); // 4 s after the first
    try std.testing.expect(d.due(7 * s1, 7 * s1)); // 5 s after
    try std.testing.expect(!d.due(9 * s1, 9 * s1));
    var off: Deadline = .{ .warn_ns = 0 };
    try std.testing.expect(!off.due(100 * s1, 100 * s1));
    try std.testing.expectEqual(@as(?u32, 2000), warnSetting(null));
    try std.testing.expectEqual(@as(?u32, 0), warnSetting("0"));
    try std.testing.expectEqual(@as(?u32, 500), warnSetting(" 500 "));
    for ([_][]const u8{ "-1", "x", "3600001", "2 s" }) |t| try std.testing.expectEqual(@as(?u32, null), warnSetting(t));
}

test "TF_DS_ISSUE_LEAD_WAIT=backoff is a mode; the slice doubles per empty wake up to the cap and never past it" {
    try std.testing.expectEqual(Wait.backoff, waitSetting("backoff").?);
    try std.testing.expectEqual(Wait.backoff, waitSetting(" backoff\n").?);
    try std.testing.expectEqual(@as(?Wait, null), waitSetting("backoffs"));
    try std.testing.expectEqual(@as(u32, 200), backoffNext(100, backoff_cap_us));
    try std.testing.expectEqual(backoff_cap_us, backoffNext(backoff_cap_us - 1, backoff_cap_us));
    try std.testing.expectEqual(backoff_cap_us, backoffNext(backoff_cap_us, backoff_cap_us));
    // a slice set above the cap is kept: the cap never shortens a nap the setting asked for
    try std.testing.expectEqual(@as(u32, 5000), backoffNext(5000, backoff_cap_us));
    // wakes needed to cover a 14.6 ms layer from a 100 us first slice: far fewer than 146 fixed slices
    var cur: u32 = 100;
    var slept: u64 = 0;
    var wakes: u32 = 0;
    while (slept < 14_600) : (wakes += 1) {
        slept += cur;
        cur = backoffNext(cur, backoff_cap_us);
    }
    try std.testing.expect(wakes <= 20);
}
