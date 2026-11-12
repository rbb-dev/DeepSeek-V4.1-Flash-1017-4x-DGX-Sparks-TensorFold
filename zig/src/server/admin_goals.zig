//! The goals controller's surface (build step 6, alert-only): the goal rows as the engine runs them, and the report GET /goals
//! and the status carry. The controller itself is core/goals.zig; it is fed by the series sampler, never by the engine's host.
const std = @import("std");
const api = @import("engine_api");
const settings = api.settings;
const goals = api.goals;
const admin_state = @import("admin_state.zig");
const json = @import("json");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const id_mode = 926;
pub const id_decode = 927;
pub const id_prefill = 928;
pub const id_cap = 929;
const id_rounds = 122;

fn startText(o: *const settings.Option) ?[]const u8 {
    const res = settings.resolved[settings.indexOf(o)];
    if (res.source != .default and !o.secret) return res.value;
    return if (o.default.len == 0) null else o.default;
}

/// A row's value as the engine runs it: a live change (the controller takes it at once), else the start's; null: none.
fn textOf(id: u32) ?[]const u8 {
    const o = settings.byId(id) orelse return null;
    if (admin_state.liveValue(o)) |t| return if (std.mem.eql(u8, t, "unset")) startText(o) else t;
    return startText(o);
}

fn numOf(id: u32, none: f64) f64 {
    const t = textOf(id) orelse return none;
    return @floatFromInt(std.fmt.parseInt(u32, t, 10) catch return none);
}

/// The goal rows as they stand now.
pub fn config() goals.Config {
    var c: goals.Config = .{};
    c.mode = goals.modeOf(textOf(id_mode) orelse "off");
    c.decode_user_tps = numOf(id_decode, 0);
    c.prefill_tps = numOf(id_prefill, 0);
    c.n_cap = @intFromFloat(numOf(id_cap, 256));
    return c;
}

/// The rounds beside a prompt step the engine runs, when the row says (null: the share's own rule decides, or not set).
pub fn nNow() ?u32 {
    const t = textOf(id_rounds) orelse return null;
    return std.fmt.parseInt(u32, t, 10) catch null;
}

fn optF(x: ?f64) Value {
    return if (x) |v| .{ .float = v } else .null;
}

fn optInt(a: Allocator, x: ?u32) !Value {
    return if (x) |v| try json.intValue(a, v) else .null;
}

/// What the report says in words, built by the engine (the page shows it, never a verdict of its own). Every model figure is
/// ESTIMATED from the priors in core/goals.zig.
pub fn reachability(a: Allocator, cfg: goals.Config, r: goals.Report) ![]const u8 {
    const have_n = r.n_now != null;
    return switch (r.state) {
        .off => "goals are off",
        .idle => switch (r.reason) {
            .no_goal => "the mode names a goal that has no value yet",
            else => "nothing to judge in the last seconds (no user generating for a decode goal, no prompt filling for a prefill goal)",
        },
        .meeting => try std.fmt.allocPrint(a, "met: {d:.0} % of the last {d} judged seconds reached the goal; the rounds beside a prompt step that would just hold it (ESTIMATED) are {d}", .{ (r.met_fraction orelse 0) * 100, r.judged_s, r.n_rec orelse 0 }),
        .@"unreachable" => switch (cfg.mode) {
            .decode_first => try std.fmt.allocPrint(a, "unreachable: {d:.0} tok/s a user is above what one round loop can give ({d:.0} alone, ESTIMATED from the record)", .{ cfg.decode_user_tps, r.decode_user_max }),
            else => try std.fmt.allocPrint(a, "unreachable: {d:.0} tok/s of prompt is above what a prompt step alone does ({d:.0}, ESTIMATED from the record)", .{ cfg.prefill_tps, r.prefill_max }),
        },
        .short => switch (r.reason) {
            .over_cap => try std.fmt.allocPrint(a, "short: the goal needs about {d:.0} rounds beside a prompt step (ESTIMATED), above the cap of {d}", .{ r.n_needed orelse 0, cfg.n_cap }),
            .starves_prefill => try std.fmt.allocPrint(a, "short: the goal needs about {d:.0} rounds (ESTIMATED), which would take the prompt below its {d:.0} tok/s floor; the most the floor allows is {d}", .{ r.n_needed orelse 0, cfg.prefill_floor_tps, r.n_rec orelse 0 }),
            .starves_decode => try std.fmt.allocPrint(a, "short: holding the prompt at {d:.0} tok/s leaves a decoding user under {d:.0} tok/s (ESTIMATED)", .{ cfg.prefill_tps, cfg.decode_floor_tps }),
            else => if (have_n)
                try std.fmt.allocPrint(a, "short now: only {d:.0} % of the last {d} judged seconds reached the goal; the engine runs {d} rounds beside a prompt step and about {d} would hold it (ESTIMATED); alert only, nothing was changed", .{ (r.met_fraction orelse 0) * 100, r.judged_s, r.n_now.?, r.n_rec orelse 0 })
            else
                try std.fmt.allocPrint(a, "short now: only {d:.0} % of the last {d} judged seconds reached the goal; about {d} rounds beside a prompt step would hold it (ESTIMATED); alert only, nothing was changed", .{ (r.met_fraction orelse 0) * 100, r.judged_s, r.n_rec orelse 0 }),
        },
    };
}

/// The status block: the state with its numbers in short.
pub fn statusBlock(a: Allocator, cfg: goals.Config, r: goals.Report) !Value {
    const o = try json.newObject(a);
    try o.put(a, "state", .{ .string = @tagName(r.state) });
    try o.put(a, "n", try optInt(a, r.n_now));
    const m = try json.newObject(a);
    try m.put(a, "decode_user_tps", optF(r.decode_user_tps));
    try m.put(a, "prefill_tps", optF(r.prefill_tps));
    try m.put(a, "window_s", try json.intValue(a, r.window_s));
    try o.put(a, "measured", .{ .object = m });
    const ach = try json.newObject(a);
    try ach.put(a, "decode_user_max", .{ .float = r.decode_user_max });
    try ach.put(a, "prefill_max", .{ .float = r.prefill_max });
    try o.put(a, "achievable", .{ .object = ach });
    try o.put(a, "mode", .{ .string = @tagName(cfg.mode) });
    return .{ .object = o };
}

/// GET /goals.
pub fn value(a: Allocator, cfg: goals.Config, r: goals.Report) !Value {
    const o = try json.newObject(a);
    try o.put(a, "mode", .{ .string = @tagName(cfg.mode) });
    var rows: std.ArrayList(Value) = .empty;
    if (cfg.decode_user_tps > 0) {
        const g = try json.newObject(a);
        try g.put(a, "name", .{ .string = "decode" });
        try g.put(a, "target", .{ .float = cfg.decode_user_tps });
        try g.put(a, "unit", .{ .string = "tok/s" });
        try g.put(a, "basis", .{ .string = "user" });
        try rows.append(a, .{ .object = g });
    }
    if (cfg.prefill_tps > 0) {
        const g = try json.newObject(a);
        try g.put(a, "name", .{ .string = "prefill" });
        try g.put(a, "target", .{ .float = cfg.prefill_tps });
        try g.put(a, "unit", .{ .string = "tok/s" });
        try g.put(a, "basis", .{ .string = "aggregate" });
        try rows.append(a, .{ .object = g });
    }
    try o.put(a, "goals", .{ .array = rows.items });
    try o.put(a, "state", .{ .string = @tagName(r.state) });
    try o.put(a, "reason", .{ .string = @tagName(r.reason) });
    try o.put(a, "n", try optInt(a, r.n_now));
    try o.put(a, "n_wanted", try optInt(a, r.n_rec));
    try o.put(a, "n_needed", optF(r.n_needed));
    try o.put(a, "n_cap", try json.intValue(a, cfg.n_cap));
    const m = try json.newObject(a);
    try m.put(a, "decode_user_tps", optF(r.decode_user_tps));
    try m.put(a, "prefill_tps", optF(r.prefill_tps));
    try m.put(a, "users", .{ .float = r.users });
    try m.put(a, "window_s", try json.intValue(a, r.window_s));
    try m.put(a, "judged_s", try json.intValue(a, r.judged_s));
    try o.put(a, "measured", .{ .object = m });
    try o.put(a, "met_fraction", optF(r.met_fraction));
    const ach = try json.newObject(a);
    try ach.put(a, "decode_user_max", .{ .float = r.decode_user_max });
    try ach.put(a, "prefill_max", .{ .float = r.prefill_max });
    try ach.put(a, "basis", .{ .string = "ESTIMATED: the model's priors from the record (4,096-row step 0.63 s, round 0.038 s with one user, 2.5 tokens a round on prose), not measured on this build" });
    try o.put(a, "achievable", .{ .object = ach });
    const floors = try json.newObject(a);
    try floors.put(a, "prefill_tps", .{ .float = cfg.prefill_floor_tps });
    try floors.put(a, "decode_user_tps", .{ .float = cfg.decode_floor_tps });
    try o.put(a, "floors", .{ .object = floors });
    try o.put(a, "reachability", .{ .string = try reachability(a, cfg, r) });
    try o.put(a, "alert_only", .{ .bool = true });
    return .{ .object = o };
}
