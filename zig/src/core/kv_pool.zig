//! The KV pool sized from the free memory the operator leaves, and the ledger of what a start takes.
//!
//! Every rank offers its numbers just before the pool is allocated (the last big allocation that is a setting): the device
//! memory free then, the free target, the room kept for what the engine takes after the pool, the positions its free memory
//! fits, and the ledger of what the start took so far. The ranks' parts go round the host star (star.zig; never the device
//! communicator) and every rank reads the same table, so every rank decides the same pool: the smallest any rank fits, never
//! below the context (a request is never cut: the context is never lowered). With no free target the vote asks nothing and
//! the pool is the one the settings name, as it always was.
//!
//! What is measured here is what the driver reports free (cuMemGetInfo); the growth after the pool is a measured allowance by role
//! (`growthFor`) unless the operator gives one, and the start's own line says how much the start took after the pool.
const std = @import("std");

pub const gib: u64 = 1 << 30;
/// A pool's granularity: a multiple of 65,536 positions (the refusals of a pool name the same steps).
pub const pool_step: usize = 65536;
/// The device arena grows in 256 MiB blocks, so a pool may take that much more than its caches' bytes.
pub const arena_slack_bytes: u64 = 256 << 20;
/// What the engine takes after the pool and while it serves, by role, MEASURED on the four Sparks at TF_DS_KV_FREE_TARGET_GIB=14 (release
/// 3df841574, 2026-11-09): with 6 GiB assumed, the free memory at the start's end was 10.9 GiB on rank 0 and 12.6 to 12.7 on the
/// workers (and the same after a cold 305K-token prompt), so the allowance is the 6 GiB plus the shortfalls, 3.1 and 1.3 GiB, rounded
/// up to a quarter of a GiB. Rank 0 holds the server, the admin threads and the sampler. One measurement of one run: a start that
/// measures another says so in its line (the growth this start took after the pool is logged beside it).
pub const growth_rank0_bytes: u64 = 9 * gib + gib / 4;
pub const growth_follower_bytes: u64 = 7 * gib + gib / 2;
pub const growth_measured = "MEASURED 2026-11-09 on the four Sparks at a 14 GiB target (release 3df841574): 10.9 GiB free at the end on rank 0, 12.6 on a worker, against 6 GiB assumed";

pub fn defaultGrowth(rank: u32) u64 {
    return if (rank == 0) growth_rank0_bytes else growth_follower_bytes;
}

/// The growth a rank leaves: the operator's number (TF_DS_KV_GROWTH_MIB, the same on every rank) when given, else the role's measured one.
pub fn growthFor(rank: u32, given: bool, bytes: u64) u64 {
    return if (given) bytes else defaultGrowth(rank);
}

/// The rows a prompt pass had when the role's growth above was measured.
pub const growth_measured_pass_rows: usize = 4096;
/// What a rank takes after the pool for every further 4,096 rows of a pass (TF_DS_PASS_ROWS above 4,096: the prompt-pass buffers
/// allocated after the vote, the Engram rows' staging among them), by role, MEASURED on the four Sparks at TF_DS_KV_FREE_TARGET_GIB=12
/// (a release with 8,192-row passes): 8,192-row passes + GEMM halves left 2.58 GiB less free on rank 0 and 2.36 GiB on a worker than
/// 4,096-row passes did, beside the pool; rounded up to a quarter of a GiB. Without it an 8,192-row start keeps ~10.4 GiB free on
/// rank 0 at a 12 GiB target: the pool, not the free memory, pays for the larger passes.
pub const growth_per_pass4096_rank0_bytes: u64 = 2 * gib + 3 * gib / 4;
pub const growth_per_pass4096_follower_bytes: u64 = 2 * gib + gib / 2;

/// The growth a rank leaves for passes of `pass_rows`: the operator's number when given (it already names the whole growth), else the
/// role's measured growth, plus the role's measured share for every 4,096 rows above the measured pass (rounded up a whole 4,096).
pub fn growthForPass(rank: u32, given: bool, bytes: u64, pass_rows: usize) u64 {
    if (given) return bytes;
    const base = defaultGrowth(rank);
    if (pass_rows <= growth_measured_pass_rows) return base;
    const extra_units: u64 = (pass_rows - growth_measured_pass_rows + growth_measured_pass_rows - 1) / growth_measured_pass_rows;
    const per: u64 = if (rank == 0) growth_per_pass4096_rank0_bytes else growth_per_pass4096_follower_bytes;
    return base +| per *| extra_units;
}
/// No pool holds more positions than an index of them (u32), less one step.
pub const max_pool_positions: usize = (1 << 32) - pool_step;
pub const max_entries = 24;

/// TF_DS_KV_FREE_TARGET_GIB (a follower's --kv-free-target-gib): the free memory to keep on each Spark, GiB, as bytes.
pub fn parseTargetGib(text: []const u8) error{BadKvTarget}!u64 {
    const x = std.fmt.parseFloat(f64, text) catch return error.BadKvTarget;
    if (!std.math.isFinite(x) or x <= 0 or x > 1024) return error.BadKvTarget;
    return @intFromFloat(x * @as(f64, gib));
}

/// TF_DS_KV_GROWTH_MIB (--kv-growth-mib): what the engine is left to take after the pool, MiB, as bytes (0 allowed).
pub fn parseGrowthMib(text: []const u8) error{BadKvGrowth}!u64 {
    const x = std.fmt.parseInt(u64, text, 10) catch return error.BadKvGrowth;
    if (x > (1 << 20)) return error.BadKvGrowth;
    return x << 20;
}

/// The pool a rank starts from when it was not given one: with a free target the vote sizes the pool, so a rank without --pool (the
/// options' default of 262,144 positions) starts from the context, the floor the vote never goes under; without a target the pool is
/// as given.
pub fn startPool(pool: usize, context: usize, target_bytes: u64) usize {
    return if (target_bytes != 0 and context != 0) @max(pool, context) else pool;
}

/// What a ledger line is for.
pub const Feature = enum(u8) { weights, engram, transport, chunk, kv_pool, window_rings, round_buffers, drafter, ssd_staging, other };

pub const FeatureInfo = struct { label: []const u8, option: []const u8 };

/// The words a feature goes by, and the setting that sizes it ("": none).
pub fn featureInfo(f: Feature) FeatureInfo {
    return switch (f) {
        .weights => .{ .label = "the weights", .option = "" },
        .engram => .{ .label = "the Engram tables", .option = "" },
        .transport => .{ .label = "the transport's arenas", .option = "" },
        .chunk => .{ .label = "a pass's chunk and node", .option = "TF_DS_PASS_ROWS" },
        .kv_pool => .{ .label = "the KV pool", .option = "TF_DS_WINDOW" },
        .window_rings => .{ .label = "the window rings", .option = "TF_DS_STREAMS" },
        .round_buffers => .{ .label = "the round's buffers", .option = "TF_DS_ROUND_ROWS" },
        .drafter => .{ .label = "the drafter", .option = "" },
        .ssd_staging => .{ .label = "the SSD tier's staging", .option = "TF_DS_SPILL_STAGING_MIB" },
        .other => .{ .label = "everything else (the CUDA context, the kernels, the graphs)", .option = "" },
    };
}

/// One ledger line as a rank sends it: the feature and the bytes it took, on the device and on the host.
pub const Entry = extern struct { feature: u8 = 0, pad: [7]u8 = @splat(0), device: u64 = 0, host: u64 = 0 };

/// One rank's offer to the vote: a fixed frame (the star carries equal parts).
pub const Part = extern struct {
    magic: u32,
    rank: u32,
    /// The device memory free just before the pool, and the device's total.
    free_bytes: u64,
    total_bytes: u64,
    /// The free target and the growth kept for what follows the pool (0 target: the vote asks nothing).
    target_bytes: u64,
    growth_bytes: u64,
    /// The context: the pool never holds fewer positions; and the pool the settings named (what the pool is without a target).
    floor: u64,
    asked: u64,
    /// The positions this rank's free memory fits (a multiple of `pool_step`; 0: none).
    fit: u64,
    /// The pool's bytes at the floor, and at one step and at seventeen steps (the line a dry run re-fits with).
    floor_bytes: u64,
    b_lo: u64,
    b_hi: u64,
    bytes_per_token: u64,
    n: u32,
    /// 1 when the operator gave the growth (the same on every rank); 0: this rank's role default.
    growth_set: u32,
    entries: [max_entries]Entry,

    pub const tag: u32 = 0x4b565031; // "KVP1"

    pub fn init(rank: u32) Part {
        var p = std.mem.zeroes(Part);
        p.magic = tag;
        p.rank = rank;
        return p;
    }
};

comptime {
    if (@sizeOf(Part) > 16 * 1024) @compileError("a part must fit one star frame");
}

/// The positions (a multiple of `pool_step`) a rank with `free` bytes free can give the pool while it keeps `target` free and
/// leaves `growth` for what follows and a block of arena slack; `b.bytes(positions)` is the pool's bytes. 0: not even a step.
pub fn fit(free: u64, target: u64, growth: u64, b: anytype) usize {
    const keep = std.math.add(u64, std.math.add(u64, target, growth) catch return 0, arena_slack_bytes) catch return 0;
    if (free <= keep) return 0;
    return fitIn(free - keep, b);
}

/// The positions (a multiple of `pool_step`; 0: not even a step) whose pool bytes `b.bytes(positions)` fit `avail`.
pub fn fitIn(avail: u64, b: anytype) usize {
    if (b.bytes(pool_step) > avail) return 0;
    var good: usize = 1;
    var bad: usize = max_pool_positions / pool_step + 1;
    if (b.bytes(max_pool_positions) <= avail) return max_pool_positions;
    while (bad - good > 1) {
        const mid = good + (bad - good) / 2;
        if (b.bytes(mid * pool_step) <= avail) good = mid else bad = mid;
    }
    return good * pool_step;
}

/// One rank's offer for a live resize of the pool (TF_DS_KV_FREE_TARGET_GIB written while the engine runs, with TF_DS_KV_VMM=1): the
/// positions its memory holds with the new target kept free, the most its reservation and structures take, and, from rank 0 alone
/// (the only rank that knows which extents streams and kept states hold), the end of the highest extent in use.
pub const ResizePart = extern struct {
    magic: u32,
    rank: u32,
    /// 1 when this rank's pool is in virtual memory (TF_DS_KV_VMM=1); a rank without it still takes part so no rank waits for it.
    vmm: u32,
    /// Rank 0 alone: the disk restores still loading into the pool (their extents are held, and nothing is given back while one runs).
    restoring: u32,
    cur: u64,
    fit: u64,
    ceiling: u64,
    in_use: u64,
    free_bytes: u64,
    mapped_bytes: u64,
    target_bytes: u64,

    pub const tag: u32 = 0x4b565232; // "KVR2"
};

pub const ResizeVerdict = enum(u8) {
    same, // the pool already is what the ranks fit
    grow,
    shrink,
    shrink_partial, // an extent in use at the tail: shrunk as far as it is free
    blocked_by_use, // the memory asks for a smaller pool and an extent in use holds the whole tail, or a disk restore is loading: nothing changes
    refused_context, // fewer positions fit than the context: the context is never lowered
    refused_disagree, // the ranks' parts do not describe the same pool and target
    refused_memory, // a rank ran out of memory mapping the change (device or host): every rank undid it; `rank` in the outcome names the first
    refused_error, // a rank failed to map the change for another reason: every rank undid it; the outcome names the first
    refused_no_vmm, // a rank's pool is not in virtual memory (TF_DS_KV_VMM=1 was not given to every rank): nothing moves
};

/// What one rank reports after it tried to move the pool (a byte a rank, gathered over the star).
pub const ballot_ok: u8 = 0;
pub const ballot_memory: u8 = 1;
pub const ballot_error: u8 = 2;

pub const ResizeOutcome = struct { verdict: ResizeVerdict, rank: u32 };

/// The decision as it stands once every rank's ballot is in: a rank that ran out of memory makes the outcome `refused_memory` naming it
/// (the first by rank), another failure `refused_error`; otherwise the decision's own verdict, and the rank that limited the fit.
pub fn resizeOutcome(d: ResizeDecision, ballots: []const u8) ResizeOutcome {
    var err_rank: ?u32 = null;
    for (ballots, 0..) |b, r| {
        if (b == ballot_memory) return .{ .verdict = .refused_memory, .rank = @intCast(r) };
        if (b != ballot_ok and err_rank == null) err_rank = @intCast(r);
    }
    if (err_rank) |r| return .{ .verdict = .refused_error, .rank = r };
    return .{ .verdict = d.verdict, .rank = d.limiting_rank };
}

pub const ResizeDecision = struct {
    verdict: ResizeVerdict,
    /// The pool every rank moves to (the current one when nothing changes or the resize is refused).
    positions: usize,
    /// What the ranks' memory and reservations hold, before the extents in use are considered.
    wanted: usize,
    limiting_rank: u32,
};

/// The same decision on every rank from the same table: the smallest fit any rank has, within every rank's ceiling, never below the
/// context; a shrink never goes under the extents rank 0 reports in use (steps of `pool_step`), and says so (`shrink_partial`).
pub fn decideResize(parts: []const ResizePart, context: usize) ResizeDecision {
    const cur: usize = @intCast(parts[0].cur);
    var d: ResizeDecision = .{ .verdict = .refused_disagree, .positions = cur, .wanted = cur, .limiting_rank = 0 };
    var want: u64 = std.math.maxInt(u64);
    var ceil: u64 = std.math.maxInt(u64);
    for (parts) |p| {
        if (p.magic != ResizePart.tag or p.cur != parts[0].cur or p.target_bytes != parts[0].target_bytes) return d;
        if (p.vmm == 0) {
            d.verdict = .refused_no_vmm;
            d.limiting_rank = p.rank;
            return d;
        }
        if (p.fit < want) {
            want = p.fit;
            d.limiting_rank = p.rank;
        }
        ceil = @min(ceil, p.ceiling);
    }
    const w: usize = @intCast(@min(want, ceil) / pool_step * pool_step);
    d.wanted = w;
    if (w < context) {
        d.verdict = .refused_context;
        return d;
    }
    if (w == cur) {
        d.verdict = .same;
        return d;
    }
    if (w > cur) {
        d.verdict = .grow;
        d.positions = w;
        return d;
    }
    if (parts[0].restoring != 0) {
        d.verdict = .blocked_by_use; // a restore in flight: its unpack lands in the pool, so no tail is given back until it is installed
        return d;
    }
    const floor = @max(context, std.mem.alignForward(usize, @intCast(parts[0].in_use), pool_step));
    const to = @max(w, floor);
    if (to >= cur) {
        d.verdict = .blocked_by_use;
        return d;
    }
    d.positions = to;
    d.verdict = if (to == w) .shrink else .shrink_partial;
    return d;
}

pub const Verdict = enum { off, auto, refused };
pub const Refusal = enum { settings_differ, below_context, bad_part };

pub const Decision = struct {
    verdict: Verdict,
    /// The pool, in positions (0 when refused).
    pool: usize = 0,
    /// The rank that fits least (auto), has the largest shortfall (below_context) or differs (settings_differ, bad_part).
    limiting_rank: u32 = 0,
    refusal: ?Refusal = null,
};

fn refused(why: Refusal, rank: u32) Decision {
    return .{ .verdict = .refused, .refusal = why, .limiting_rank = rank };
}

/// What each rank must have free for the pool at its floor: the pool's bytes there, and everything kept back.
fn need(p: Part) u64 {
    return p.floor_bytes +| p.target_bytes +| p.growth_bytes +| arena_slack_bytes;
}

/// The rank whose free memory is the furthest short of its need (the first of equals).
fn mostShort(table: []const Part) u32 {
    var worst: u32 = 0;
    var gap: u64 = 0;
    for (table, 0..) |p, i| {
        const n = need(p);
        const g = if (n > p.free_bytes) n - p.free_bytes else 0;
        if (g > gap) {
            gap = g;
            worst = @intCast(i);
        }
    }
    return worst;
}

/// The pool every rank takes from the ranks' parts: with no free target the pool the settings named (rank 0's); with one the
/// smallest any rank fits, never below the context.
pub fn decide(table: []const Part) Decision {
    for (table, 0..) |p, i| if (p.magic != Part.tag or p.rank != i) return refused(.bad_part, @intCast(i));
    const lead = table[0];
    if (lead.target_bytes == 0) {
        for (table[1..], 1..) |p, i| if (p.target_bytes != 0) return refused(.settings_differ, @intCast(i));
        return .{ .verdict = .off, .pool = @intCast(lead.asked) };
    }
    for (table[1..], 1..) |p, i| {
        // the growth differs by role unless the operator gave one: then every rank must have the same
        const growth_differs = p.growth_set != lead.growth_set or (lead.growth_set != 0 and p.growth_bytes != lead.growth_bytes);
        if (p.target_bytes != lead.target_bytes or growth_differs or p.floor != lead.floor) return refused(.settings_differ, @intCast(i));
    }
    var fits: [8]usize = undefined;
    for (table, 0..) |p, i| fits[i] = @intCast(p.fit);
    return choose(table, fits[0..table.len]);
}

fn choose(table: []const Part, fits: []const usize) Decision {
    var least: u32 = 0;
    for (fits, 0..) |f, i| if (f < fits[least]) {
        least = @intCast(i);
    };
    const floor: usize = @intCast(table[0].floor);
    if (fits[least] >= floor) return .{ .verdict = .auto, .pool = @min(fits[least], max_pool_positions), .limiting_rank = least };
    // the floor is not a multiple of the step: it still fits when every rank's bytes at the floor do
    for (table) |p| if (need(p) > p.free_bytes) return refused(.below_context, mostShort(table));
    return .{ .verdict = .auto, .pool = floor, .limiting_rank = least };
}

/// The same vote with another free target: what each rank would fit (`out`, one a rank, from the line through the pool's bytes at
/// one step and at seventeen: an ESTIMATE, within a step of the exact fit) and the pool that gives. A target of 0 is the off mode.
pub fn refit(table: []const Part, target_bytes: u64, out: []usize) Decision {
    if (target_bytes == 0) {
        for (out) |*o| o.* = 0;
        return .{ .verdict = .off, .pool = @intCast(table[0].asked) };
    }
    for (table, out[0..table.len]) |p, *o| {
        const keep = p.growth_bytes +| target_bytes +| arena_slack_bytes;
        if (p.free_bytes <= keep or p.b_hi <= p.b_lo) {
            o.* = 0;
            continue;
        }
        const avail: f64 = @floatFromInt(p.free_bytes - keep);
        const slope: f64 = @as(f64, @floatFromInt(p.b_hi - p.b_lo)) / @as(f64, 16 * pool_step);
        const base: f64 = @as(f64, @floatFromInt(p.b_lo)) - slope * @as(f64, pool_step);
        const positions = (avail - base) / slope;
        if (positions < pool_step) {
            o.* = 0;
            continue;
        }
        const steps: usize = @intFromFloat(@floor(positions / pool_step));
        o.* = @min(steps * pool_step, max_pool_positions);
    }
    var shifted: [8]Part = undefined;
    for (table, 0..) |p, i| {
        shifted[i] = p;
        shifted[i].target_bytes = target_bytes;
    }
    return choose(shifted[0..table.len], out[0..table.len]);
}

fn commas(buf: []u8, n: u64) []const u8 {
    var tmp: [24]u8 = undefined;
    const digits = std.fmt.bufPrint(&tmp, "{d}", .{n}) catch return "?";
    var w: usize = 0;
    for (digits, 0..) |c, i| {
        if (i > 0 and (digits.len - i) % 3 == 0 and w < buf.len) {
            buf[w] = ',';
            w += 1;
        }
        if (w < buf.len) {
            buf[w] = c;
            w += 1;
        }
    }
    return buf[0..w];
}

fn inGib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / @as(f64, gib);
}

/// The words that end a start for the pool: what differs, or how short the worst rank is and what its start has taken.
pub fn refusalText(buf: []u8, table: []const Part, d: Decision) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    const r = d.limiting_rank;
    switch (d.refusal orelse return "") {
        .bad_part => w.print("rank {d}'s answer to the KV pool's vote is not one: run every rank from the same image", .{r}) catch {},
        .settings_differ => {
            const lead = table[0];
            const p = table[r];
            w.print("the ranks were given different KV pool settings: rank {d} has a free target of {d:.2} GiB, growth {d:.2} GiB and context {d}, rank 0 {d:.2} GiB, {d:.2} GiB and {d}: give every rank the same TF_DS_KV_FREE_TARGET_GIB (a follower: --kv-free-target-gib), the same TF_DS_KV_GROWTH_MIB (--kv-growth-mib) or none on every rank, and the same context", .{ r, inGib(p.target_bytes), inGib(p.growth_bytes), p.floor, inGib(lead.target_bytes), inGib(lead.growth_bytes), lead.floor }) catch {};
        },
        .below_context => {
            const p = table[r];
            var nb: [32]u8 = undefined;
            const want = need(p);
            w.print("not enough free memory for the KV pool at the context: {s} positions take {d:.2} GiB, and rank {d} has {d:.2} GiB free before the pool, {d:.2} GiB to keep free (TF_DS_KV_FREE_TARGET_GIB), {d:.2} GiB for what the engine takes after the pool (TF_DS_KV_GROWTH_MIB) and {d:.2} GiB for the arena's block: {d:.2} GiB short. ", .{ commas(&nb, p.floor), inGib(p.floor_bytes), r, inGib(p.free_bytes), inGib(p.target_bytes), inGib(p.growth_bytes), inGib(arena_slack_bytes), inGib(want -| p.free_bytes) }) catch {};
            // its largest lines first
            var order: [max_entries]u32 = undefined;
            const n = @min(p.n, max_entries);
            for (0..n) |i| order[i] = @intCast(i);
            std.mem.sort(u32, order[0..n], &p, struct {
                fn big(q: *const Part, a: u32, b: u32) bool {
                    return q.entries[a].device > q.entries[b].device;
                }
            }.big);
            w.print("What rank {d}'s start has taken, largest first: ", .{r}) catch {};
            for (order[0..@min(n, 6)], 0..) |idx, k| {
                const e = p.entries[idx];
                const feat: Feature = if (e.feature <= @backingInt(Feature.other)) @fromBackingInt(@intCast(e.feature)) else .other;
                w.print("{s}{s} {d:.2} GiB", .{ if (k > 0) ", " else "", featureInfo(feat).label, inGib(e.device) }) catch {};
            }
            w.print(". Lower the free target, a smaller TF_DS_PASS_ROWS, or drop a feature named above.", .{}) catch {};
        },
    }
    return w.buffered();
}

/// Where a line was recorded: the call's file and line, fixed at build.
pub const Site = struct { file: []const u8 = "", line: u32 = 0 };

/// What one rank's start took, line by line, as it took it.
pub const Ledger = struct {
    entries: [max_entries]Entry = @splat(.{}),
    sites: [max_entries]Site = @splat(.{}),
    n: u32 = 0,
    /// Lines that did not fit the ledger.
    dropped: u32 = 0,

    pub fn record(l: *Ledger, src: std.builtin.SourceLocation, f: Feature, device: u64, host: u64) void {
        if (l.n >= max_entries) {
            l.dropped += 1;
            return;
        }
        l.entries[l.n] = .{ .feature = @backingInt(f), .device = device, .host = host };
        l.sites[l.n] = .{ .file = src.file, .line = src.line };
        l.n += 1;
    }

    pub fn deviceTotal(l: *const Ledger) u64 {
        var t: u64 = 0;
        for (l.entries[0..l.n]) |e| t += e.device;
        return t;
    }

    /// The device holds `used` bytes: whatever the lines do not name is the remainder, "everything else" (never negative: lines
    /// that name more than the device holds, by a block's rounding, stand as they are).
    pub fn close(l: *Ledger, src: std.builtin.SourceLocation, used: u64) void {
        const named = l.deviceTotal();
        if (used > named) l.record(src, .other, used - named, 0);
    }

    /// The lines into a vote part.
    pub fn fill(l: *const Ledger, p: *Part) void {
        p.n = l.n;
        p.entries = l.entries;
    }
};

/// What the start decided and measured, for the admin surface to read: written once while the engine starts, read after.
pub const Report = struct {
    world: u32 = 0,
    table: [4]Part = std.mem.zeroes([4]Part),
    decision: Decision = .{ .verdict = .off },
    /// This rank's own ledger, with the lines recorded after the vote, and the sites of its lines.
    mine: Ledger = .{},
    /// Every rank's whole ledger, gathered at the start's end (after the pool and everything that follows it).
    final: [4]Part = std.mem.zeroes([4]Part),
    have_final: bool = false,
    /// The pool this rank allocated, its streams, and what the caches' layout says that takes.
    pool_active: usize = 0,
    streams: u32 = 0,
    pool_predicted: u64 = 0,
    /// The device bytes the pool's allocation took here (measured), and what the engine took after it up to the start's end.
    pool_taken: u64 = 0,
    after_taken: u64 = 0,
    free_ready: u64 = 0,
    /// The last live resize (rank 0): the cfg version that asked, what it decided, from and to (0 until one ran).
    resize_version: u64 = 0,
    resize_verdict: ResizeVerdict = .same,
    resize_from: usize = 0,
    resize_wanted: usize = 0,
    /// The rank the outcome names: the one that ran out of memory or failed, else the rank whose memory limited the fit.
    resize_rank: u32 = 0,
};

pub var report: Report = .{};
pub var ready: std.atomic.Value(bool) = .init(false);
