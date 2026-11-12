//! The round loop's constants from the model's attributes and the engine's settings (Python _family_setup).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;
const alloc = @import("allocate.zig");
const Constants = @import("depth.zig").Constants;

pub const Cost = alloc.Cost;

/// The class default acceptance by depth (Python DraftDepth.depth_prior).
pub const default_prior = [_]f64{ 0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };

/// What a backend reports about its loaded model: the attributes the Python engine reads at setup.
pub const Model = struct {
    solo_head_overhead_ms: ?f64 = null, // fixed overhead for a lone head window; null keeps the live estimate
    copy_proposals: bool = true, // suffix-lookup copy windows; false opts a backend out
    exact_width: u32 = 1, // the widest window that reproduces one-token steps bit for bit
    first_copy_rows: u32 = 0, // a lone stream's first copy window (0: the exact width)
    gpu_tokens: bool = false, // one-token rounds may run a step ahead (pipelined)
    mtp: bool = false, // a draft head is loaded
    speculate: bool = false, // the head can draft from a verify's rows
    speculate_early: bool = true, // draft every row's first draft behind the verify
    draft_prior: []const f64 = &.{}, // the head's acceptance by depth until a stream has its own
    depth_rate: f64 = 0, // the newest round's weight in a depth's acceptance estimate (0: the core's)
    plain_guard: bool = false, // plain rounds compete with drafted depths
    drafts: u32 = 1, // head drafts a round at most
    window_costs: []const Cost = &.{}, // forward ms by exact window width (timed at load)
    mtp_step_ms: f64 = 0.0, // one chained head step (timed at load)
    streams_exact: bool = true, // shared forwards keep each stream's bits
    hidden_rows: bool = false, // the backend runs several streams' windows in one forward
    batch_rows: u32 = 0, // a shared forward's rows at most (0: 32)
    max_streams: u32 = 0, // a shared forward's streams at most (0: 32)
    shared_costs: []const Cost = &.{}, // shared forwards' ms by total rows (timed at load)
    draft_probabilities: bool = false, // the head gives each drafted node's chance of landing
    draft_streams: bool = false, // the head drafts every stream of a shared round in one batch
    head_trees: bool = false, // the head drafts trees of lanes (DraftRequest.lanes) and holds them
    lane_costs: []const Cost = &.{}, // a lone stream's window ms by rows, up to its widest tree (timed at load)
    copies: bool = true, // windows may take a stream's copied continuations (windows.copyProposal)
    enter_match: i64 = 8, // and the matching tokens a copy needs
    copy_rows: u32 = 0, // a lone stream's copied window's rows at most, past the exact width (0: the exact width)
    lone_copies: bool = true, // a lone stream proposes copies (false: only streams in shared rounds do)
};

/// A wide copy's first width in drafts (Model.copy_rows set): a lone stream's copies start there and double to their
/// most while every copied token lands.
pub const wide_first: u32 = 8;

/// A lone stream's first copy width past every window's (copy_rows set): wide_first drafts, at most `alone_max`.
pub fn aloneFirst(alone_max: u32) u32 {
    return @min(wide_first, alone_max);
}

pub const Config = struct {
    solo_head_overhead_ms: ?f64 = null, // fixed overhead for a lone head window; null keeps the live estimate
    copy_proposals: bool = true, // suffix-lookup copy windows; false opts a backend out
    family_width: u32,
    max_copy: u32,
    first_copy: u32,
    base_width: u32,
    family_mtp: bool,
    speculate_early: bool,
    pipelined: bool,
    plain_guard: bool,
    most_drafts: u32,
    family_streams: bool,
    batch_rows: u32,
    batch_streams: u32,
    node_probabilities: bool,
    fill_lanes: u32 = 0, // rows a window may grow to with suffix-match branches (0: off; lanes/fill.zig)
    fill_match: u32 = 8, // context tokens a suffix-match branch must match (shorter ones are coincidental phrases)
    fill_floor: f64 = 0.35, // the share of grafted rows a stream must keep to go on reading its chain back every round
    draft_streams: bool,
    head_trees: bool,
    head_lanes: u32 = 0, // a lone stream's lanes at most, the planner picking each round's (0: the depth rule's chain)
    head_depth: u32 = 16, // and their depth at most
    own_prices: bool = true, // the planner prices each pick from the stream's own measured rounds (seeded from the table)
    depth_prior: []f64,
    family_costs: Table,
    shared_costs: Table,
    lane_costs: Table, // window ms by rows to 64 (lanes/plan_lanes.zig)
    mtp_step_ms: f64,
    copies: bool = true, // windows take copied continuations (false: never; Model.copies)
    enter_match: i64 = 8, // matching tokens a copy needs (rejects coincidental indentation; Model.enter_match)
    copy_rate: f64 = 0.94, // a copied token's chance of landing in allocation
    // a lone stream's copy widths in drafts, first and most (Model.copy_rows past the exact width: wide_first and
    // copy_rows - 1; else first_copy and max_copy, every window's)
    alone_first: u32 = 0,
    alone_max: u32 = 0,
    lone_copies: bool = true, // Model.lone_copies
    k: Constants = .{},

    pub fn init(gpa: Allocator, m: Model, max_rows: u32, max_draft: u32) !Config {
        if (max_rows < 1) return error.BadMaxRows;
        if (m.solo_head_overhead_ms) |cost| {
            if (!std.math.isFinite(cost) or cost < 0.0) return error.BadSoloHeadOverhead;
        }
        const width: u32 = @max(1, @min(@max(m.exact_width, 1), max_rows));
        const max_copy = width - 1;
        const first_rows: i64 = if (m.first_copy_rows != 0) m.first_copy_rows else width;
        const first_copy: u32 = @intCast(@max(1, @min(@as(i64, max_copy), first_rows - 1)));
        const prior = if (m.draft_prior.len > 0) m.draft_prior else &default_prior;
        const streams = width >= 2 and m.streams_exact and m.hidden_rows;
        const batch_rows: u32 = if (m.batch_rows != 0) m.batch_rows else 32;
        var family_costs: Table = .{};
        errdefer family_costs.deinit(gpa);
        for (m.window_costs) |c| {
            if (c.width >= 1 and c.width <= width) try family_costs.put(gpa, c.width, c.ms);
        }
        // shared forwards: window costs then timed shared rows (a later table overrides), extended to batch_rows
        var timed: std.ArrayList(Cost) = .empty;
        defer timed.deinit(gpa);
        for ([_][]const Cost{ m.window_costs, m.shared_costs }) |table| {
            for (table) |c| {
                if (c.width < 1 or c.width > batch_rows) continue;
                for (timed.items) |*t| {
                    if (t.width == c.width) {
                        t.ms = c.ms;
                        break;
                    }
                } else try timed.append(gpa, c);
            }
        }
        const depth_prior = try gpa.dupe(f64, prior);
        errdefer gpa.free(depth_prior);
        const wide = m.copy_rows > width and max_copy >= 1;
        const alone_max: u32 = if (wide) @min(m.copy_rows, max_rows) - 1 else max_copy;
        return .{
            .alone_first = if (wide) aloneFirst(alone_max) else first_copy,
            .alone_max = alone_max,
            .lone_copies = m.lone_copies,
            .family_width = width,
            .max_copy = max_copy,
            .first_copy = first_copy,
            .base_width = @min(width, first_copy + 1),
            .family_mtp = m.mtp and width >= 2 and m.speculate,
            .speculate_early = m.speculate_early,
            .pipelined = m.gpu_tokens,
            .plain_guard = m.plain_guard,
            .most_drafts = @min(m.drafts, @min(width - 1, max_draft)),
            .family_streams = streams,
            .batch_rows = batch_rows,
            .batch_streams = if (streams) (if (m.max_streams != 0) m.max_streams else 32) else 1,
            .node_probabilities = m.draft_probabilities,
            .draft_streams = m.draft_streams,
            .head_trees = m.head_trees,
            .depth_prior = depth_prior,
            .family_costs = family_costs,
            .shared_costs = try alloc.extendCosts(gpa, timed.items, batch_rows),
            .lane_costs = try alloc.extendCosts(gpa, m.lane_costs, 64),
            .mtp_step_ms = m.mtp_step_ms,
            .copies = m.copies,
            .enter_match = m.enter_match,
            .k = .{ .depth_rate = if (m.depth_rate > 0) m.depth_rate else (Constants{}).depth_rate },
            .solo_head_overhead_ms = m.solo_head_overhead_ms,
            .copy_proposals = m.copy_proposals,
        };
    }

    pub fn deinit(c: *Config, gpa: Allocator) void {
        gpa.free(c.depth_prior);
        c.family_costs.deinit(gpa);
        c.shared_costs.deinit(gpa);
        c.lane_costs.deinit(gpa);
    }
};

test "setup derives widths like Python" {
    const gpa = std.testing.allocator;
    var c = try Config.init(gpa, .{ .exact_width = 64, .first_copy_rows = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .drafts = 4, .hidden_rows = true, .batch_rows = 128, .max_streams = 64 }, 64, 63);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 15), c.first_copy);
    try std.testing.expectEqual(@as(u32, 16), c.base_width);
    try std.testing.expectEqual(@as(u32, 4), c.most_drafts);
    try std.testing.expect(c.family_mtp and c.family_streams);
    try std.testing.expectEqual(@as(f64, 0.15), c.k.depth_rate);
    var steady = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 4, .depth_rate = 0.08 }, 16, 15);
    defer steady.deinit(gpa);
    try std.testing.expectEqual(@as(f64, 0.08), steady.k.depth_rate);
}

test "solo allowance is opt-in and preserved by setup" {
    const a = std.testing.allocator;
    var original = try Config.init(a, .{}, 8, 7);
    defer original.deinit(a);
    try std.testing.expect(original.solo_head_overhead_ms == null);
    var selected = try Config.init(a, .{ .solo_head_overhead_ms = 3.5 }, 8, 7);
    defer selected.deinit(a);
    try std.testing.expectEqual(@as(?f64, 3.5), selected.solo_head_overhead_ms);
}
test "solo allowance rejects invalid costs before allocating" {
    for ([_]f64{ -1.0, std.math.inf(f64), std.math.nan(f64) }) |cost| {
        try std.testing.expectError(error.BadSoloHeadOverhead, Config.init(std.testing.allocator, .{ .solo_head_overhead_ms = cost }, 8, 7));
    }
}
test "copy windows stay on unless a backend opts out" {
    const a = std.testing.allocator;
    var original = try Config.init(a, .{}, 8, 7);
    defer original.deinit(a);
    try std.testing.expect(original.copy_proposals);
    var off = try Config.init(a, .{ .copy_proposals = false }, 8, 7);
    defer off.deinit(a);
    try std.testing.expect(!off.copy_proposals);
}

test "a lone stream's copy widths: every window's without copy_rows; wide_first up to copy_rows - 1 with it, the other widths kept" {
    const gpa = std.testing.allocator;
    const m: Model = .{ .exact_width = 6, .gpu_tokens = false, .mtp = true, .speculate = true, .drafts = 5, .hidden_rows = true };
    var c0 = try Config.init(gpa, m, 48, 47);
    defer c0.deinit(gpa);
    try std.testing.expectEqual(c0.first_copy, c0.alone_first);
    try std.testing.expectEqual(c0.max_copy, c0.alone_max);
    for ([_][3]u32{ .{ 17, 8, 16 }, .{ 13, 8, 12 }, .{ 7, 6, 6 }, .{ 9, 8, 8 } }) |x| {
        var wm = m;
        wm.copy_rows = x[0];
        var c = try Config.init(gpa, wm, 48, 47);
        defer c.deinit(gpa);
        try std.testing.expectEqual(x[1], c.alone_first);
        try std.testing.expectEqual(x[2], c.alone_max);
        // every other width as without it
        try std.testing.expectEqual(c0.first_copy, c.first_copy);
        try std.testing.expectEqual(c0.max_copy, c.max_copy);
        try std.testing.expectEqual(c0.base_width, c.base_width);
        try std.testing.expectEqual(c0.family_width, c.family_width);
        try std.testing.expectEqual(c0.most_drafts, c.most_drafts);
    }
    // the rows the host allows bound it; copy_rows at or under the exact width is no wide copy
    for ([_][3]u32{ .{ 17, 12, 11 }, .{ 6, 48, 5 }, .{ 4, 48, 5 } }) |x| {
        var wm = m;
        wm.copy_rows = x[0];
        var c = try Config.init(gpa, wm, x[1], x[1] - 1);
        defer c.deinit(gpa);
        try std.testing.expectEqual(x[2], c.alone_max);
    }
}
