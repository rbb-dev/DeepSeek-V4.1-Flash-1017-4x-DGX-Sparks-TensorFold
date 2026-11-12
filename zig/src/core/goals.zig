//! The goals controller, alert-only (build step 6 of the dynamic-config design; FINAL-DESIGN section 4, TARGETS section 5).
//!
//! A goal is "keep decode at least G tok/s for each user" or "keep prefill at least P tok/s". Prompt steps and decode rounds
//! share rank 0's host loop, so both goals sit on one dial: N, the rounds that follow each long prompt step. With a step's wall
//! time t_s and a round's t_r, a user sees tau * N / (t_s + N * t_r) tok/s and the prompt runs rows / (t_s + N * t_r) tok/s:
//! decode rises with N, prefill falls with it.
//!
//! This controller changes nothing. It reads the sampled status the admin surface already takes once a second (never the
//! engine's host loop), keeps the last `window` seconds, and says whether the goal is met, what the goal needs (the N the model
//! solves for, the highest N the other side's floor allows), and whether it can be met at all. Nothing here is on a round's path.
//!
//! The model's constants are PRIORS from the repo's record (TARGETS section 1: a 4,096-row step 0.63 s, a round 0.038 s with one
//! decoding user and 0.049 s with four, 2.5 tokens a round on prose; ESTIMATED on one build, within the rig's +-4-5 % noise floor
//! on the three points it was checked against). The measured rates decide `meeting`; the model only says what N would.
const std = @import("std");

pub const Mode = enum(u8) { off, decode_first, prefill_first };

/// The mode a setting's text names (an unknown text is off).
pub fn modeOf(text: []const u8) Mode {
    return std.meta.stringToEnum(Mode, text) orelse .off;
}

/// idle: nothing in the window to judge (no decoding user, or no prompt filling). unreachable: the plant cannot do it at any N.
/// short: the plant can, but not now (the engine runs a lower N than the goal needs) or only by starving the other side.
pub const State = enum { off, idle, meeting, short, @"unreachable" };

pub const Reason = enum { none, no_goal, ceiling, over_cap, starves_prefill, starves_decode, below_goal_now };

pub const Config = struct {
    mode: Mode = .off,
    /// Decode goal, tok/s as the user sees them (drafts included), per decoding user; 0: none.
    decode_user_tps: f64 = 0,
    /// Prefill goal, tok/s over all prompts filling; 0: none.
    prefill_tps: f64 = 0,
    n_cap: u32 = 256,
    /// A decode goal never takes the prompt below this (the protection floor, TARGETS section 7, D73's 300).
    prefill_floor_tps: f64 = 300,
    /// A prefill goal never takes a decoding user below this.
    decode_floor_tps: f64 = 5,
};

pub const Plant = struct {
    t_s: f64 = 0.63,
    t_r_1: f64 = 0.038,
    t_r_4: f64 = 0.049,
    rows: f64 = 4096,
    tau: f64 = 2.5,

    /// A round's wall time with `u` users decoding (interpolated between the one-user and four-user records, never below the first).
    pub fn tRound(p: Plant, u: f64) f64 {
        return @max(p.t_r_1, p.t_r_1 + (p.t_r_4 - p.t_r_1) * (u - 1) / 3);
    }
    /// One user's tok/s beside a prompt at N rounds a step.
    pub fn decodeBeside(p: Plant, n: f64, u: f64) f64 {
        return p.tau * n / (p.t_s + n * p.tRound(u));
    }
    /// One user's tok/s with no prompt: the round's own ceiling.
    pub fn decodeAlone(p: Plant, u: f64) f64 {
        return p.tau / p.tRound(u);
    }
    pub fn prefillBeside(p: Plant, n: f64, u: f64) f64 {
        return p.rows / (p.t_s + n * p.tRound(u));
    }
    /// The N a decode goal needs; null when the goal is at or above the ceiling.
    pub fn nForDecode(p: Plant, g: f64, u: f64) ?f64 {
        const r = g / p.tau;
        const tr = p.tRound(u);
        if (r * tr >= 1) return null;
        return r * p.t_s / (1 - r * tr);
    }
    /// The largest N whose prompt rate still reaches `pf`; null when even N = 0 falls short (pf above the step's own speed).
    pub fn nForPrefill(p: Plant, pf: f64, u: f64) ?f64 {
        const n = (p.rows / pf - p.t_s) / p.tRound(u);
        if (n < 0) return null;
        return n;
    }
};

pub const Sample = struct {
    decode_tps: f64 = 0,
    prefill_tps: f64 = 0,
    /// Requests in prefill or decode (the status's `running`).
    running: u32 = 0,
    /// A prompt pass is in flight.
    filling: bool = false,

    /// The users generating: the running requests less the prompt filling beside them.
    fn users(s: Sample) u32 {
        const fill: u32 = if (s.filling) 1 else 0;
        return if (s.running > fill) s.running - fill else 0;
    }
};

pub const window = 10;
/// A sample meets a goal within the rig's noise floor (+-4-5 %): at 95 % of it.
pub const deadband = 0.95;
/// The share of a window's judged seconds that must meet the goal for `meeting`.
pub const met_needed = 0.8;

pub const Report = struct {
    state: State = .off,
    reason: Reason = .none,
    /// The N the engine runs (the caller's; null: not known) and the N the goal wants (null: no goal, or no N helps).
    n_now: ?u32 = null,
    n_rec: ?u32 = null,
    /// The N the model solves for before the other side's floor and the cap (null: no N does).
    n_needed: ?f64 = null,
    decode_user_tps: ?f64 = null,
    prefill_tps: ?f64 = null,
    users: f64 = 0,
    /// Seconds in the window and those the goal could be judged on, and the share of the latter that met it.
    window_s: u32 = 0,
    judged_s: u32 = 0,
    met_fraction: ?f64 = null,
    /// What the plant can do at its best (ESTIMATED from the priors): one user's decode alone, the prompt alone.
    decode_user_max: f64 = 0,
    prefill_max: f64 = 0,
};

pub const Controller = struct {
    plant: Plant = .{},
    ring: [window]Sample = @splat(.{}),
    taken: u64 = 0,

    pub fn observe(c: *Controller, s: Sample) void {
        c.ring[@intCast(c.taken % window)] = s;
        c.taken += 1;
    }

    pub fn clear(c: *Controller) void {
        c.taken = 0;
    }

    pub fn report(c: *const Controller, cfg: Config, n_now: ?u32) Report {
        const have: usize = @intCast(@min(c.taken, window));
        var r: Report = .{ .n_now = n_now, .window_s = @intCast(have) };
        // the users and the measured rates over the window
        var user_sum: f64 = 0;
        var user_n: f64 = 0;
        var dec_sum: f64 = 0;
        var dec_n: f64 = 0;
        var pre_sum: f64 = 0;
        var pre_n: f64 = 0;
        for (c.ring[0..have]) |s| {
            const u = s.users();
            if (u >= 1) {
                user_sum += @floatFromInt(u);
                user_n += 1;
                dec_sum += s.decode_tps / @as(f64, @floatFromInt(u));
                dec_n += 1;
            }
            if (s.filling) {
                pre_sum += s.prefill_tps;
                pre_n += 1;
            }
        }
        const u_est: f64 = if (user_n > 0) user_sum / user_n else 1;
        r.users = if (user_n > 0) u_est else 0;
        r.decode_user_tps = if (dec_n > 0) dec_sum / dec_n else null;
        r.prefill_tps = if (pre_n > 0) pre_sum / pre_n else null;
        r.decode_user_max = c.plant.decodeAlone(u_est);
        r.prefill_max = c.plant.prefillBeside(0, u_est);
        if (cfg.mode == .off) return r;
        const cap: f64 = @floatFromInt(@max(1, cfg.n_cap));
        switch (cfg.mode) {
            .off => unreachable,
            .decode_first => {
                if (cfg.decode_user_tps <= 0) return finish(r, .idle, .no_goal);
                // the highest N the prompt floor allows
                const n_floor = c.plant.nForPrefill(cfg.prefill_floor_tps, u_est) orelse 1;
                const n_allowed = @max(1, @min(cap, @floor(n_floor)));
                r.n_needed = c.plant.nForDecode(cfg.decode_user_tps, u_est);
                if (r.n_needed == null) {
                    r.n_rec = @intFromFloat(n_allowed);
                    return finish(r, .@"unreachable", .ceiling);
                }
                if (r.n_needed.? > n_allowed) {
                    r.n_rec = @intFromFloat(n_allowed);
                    return finish(r, .short, if (r.n_needed.? > cap) .over_cap else .starves_prefill);
                }
                r.n_rec = @intFromFloat(@max(1, @ceil(r.n_needed.?)));
                return judge(c, &r, cfg.decode_user_tps, true, user_n);
            },
            .prefill_first => {
                if (cfg.prefill_tps <= 0) return finish(r, .idle, .no_goal);
                const n_max = c.plant.nForPrefill(cfg.prefill_tps, u_est);
                if (n_max == null or n_max.? < 1) {
                    r.n_rec = 1;
                    return finish(r, .@"unreachable", .ceiling);
                }
                r.n_needed = n_max;
                const n = @max(1, @min(cap, @floor(n_max.?)));
                r.n_rec = @intFromFloat(n);
                if (user_n > 0 and c.plant.decodeBeside(n, u_est) < cfg.decode_floor_tps) return finish(r, .short, .starves_decode);
                return judge(c, &r, cfg.prefill_tps, false, pre_n);
            },
        }
    }

    /// The goal is reachable at n_rec: the measured seconds decide whether it is met now.
    fn judge(c: *const Controller, r: *Report, goal: f64, decode: bool, judged_possible: f64) Report {
        _ = judged_possible;
        const have: usize = @intCast(@min(c.taken, window));
        var judged: u32 = 0;
        var met: u32 = 0;
        for (c.ring[0..have]) |s| {
            const u = s.users();
            if (decode) {
                if (u < 1) continue;
                judged += 1;
                if (s.decode_tps / @as(f64, @floatFromInt(u)) >= goal * deadband) met += 1;
            } else {
                if (!s.filling) continue;
                judged += 1;
                if (s.prefill_tps >= goal * deadband) met += 1;
            }
        }
        r.judged_s = judged;
        if (judged == 0) return finish(r.*, .idle, .none);
        const f = @as(f64, @floatFromInt(met)) / @as(f64, @floatFromInt(judged));
        r.met_fraction = f;
        return finish(r.*, if (f >= met_needed) .meeting else .short, if (f >= met_needed) .none else .below_goal_now);
    }

    fn finish(r: Report, state: State, reason: Reason) Report {
        var out = r;
        out.state = state;
        out.reason = reason;
        return out;
    }
};

// ---- host tests on a fake plant

/// A fake plant: what the engine would sample each second at N rounds a step with `users` decoding and a prompt filling,
/// the model's own rates with a small deterministic wobble (+-4 %, the rig's noise floor) so no test rests on exact numbers.
const FakePlant = struct {
    p: Plant = .{},
    n: f64,
    users: u32,
    filling: bool = true,
    tick: u32 = 0,

    fn wobble(f: *FakePlant) f64 {
        f.tick += 1;
        const k: f64 = @floatFromInt(f.tick % 5);
        return 0.96 + 0.02 * k; // 0.96 .. 1.04
    }

    fn sample(f: *FakePlant) Sample {
        const w = f.wobble();
        const u: f64 = @floatFromInt(f.users);
        const dec = if (f.users == 0) 0 else u * (if (f.filling) f.p.decodeBeside(f.n, u) else f.p.decodeAlone(u)) * w;
        return .{
            .decode_tps = dec,
            .prefill_tps = if (f.filling) f.p.prefillBeside(f.n, @max(1, u)) * w else 0,
            .running = f.users + @as(u32, if (f.filling) 1 else 0),
            .filling = f.filling,
        };
    }

    fn run(f: *FakePlant, c: *Controller, seconds: u32) void {
        var i: u32 = 0;
        while (i < seconds) : (i += 1) c.observe(f.sample());
    }
};

const t = std.testing;

test "model: the numbers it was checked against (TARGETS section 1, 1 user and 4 users at N=5)" {
    const p: Plant = .{};
    try t.expectApproxEqRel(@as(f64, 5000), p.prefillBeside(5, 1), 0.01);
    try t.expectApproxEqRel(@as(f64, 6130), p.prefillBeside(1, 1), 0.01);
    try t.expectApproxEqRel(@as(f64, 4680), p.prefillBeside(5, 4), 0.01);
    try t.expectApproxEqRel(@as(f64, 6.1 * 2.5), p.decodeBeside(5, 1), 0.02);
}

test "goals off: nothing is judged and nothing recommended" {
    var f: FakePlant = .{ .n = 5, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const r = c.report(.{ .mode = .off, .decode_user_tps = 40 }, 5);
    try t.expectEqual(State.off, r.state);
    try t.expectEqual(@as(?u32, null), r.n_rec);
}

test "decode goal 16 at N=5: met by the plant as it is (T9's point)" {
    var f: FakePlant = .{ .n = 5, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const r = c.report(.{ .mode = .decode_first, .decode_user_tps = 15 }, 5);
    try t.expectEqual(State.meeting, r.state);
    try t.expect(r.met_fraction.? >= met_needed);
}

test "decode goal 40 at N=5: short now; the recommended N, applied to the plant, meets it (closed loop through the fake)" {
    var f: FakePlant = .{ .n = 5, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const cfg: Config = .{ .mode = .decode_first, .decode_user_tps = 40 };
    const r = c.report(cfg, 5);
    try t.expectEqual(State.short, r.state);
    try t.expectEqual(Reason.below_goal_now, r.reason);
    try t.expect(r.n_rec.? > 5);
    // the model says ~26 for 40 tok/s prose (TARGETS section 3)
    try t.expect(r.n_rec.? >= 24 and r.n_rec.? <= 30);
    f.n = @floatFromInt(r.n_rec.?);
    c.clear();
    f.run(&c, 10);
    const after = c.report(cfg, r.n_rec);
    try t.expectEqual(State.meeting, after.state);
    // the recommendation is stable once met: it does not climb (a goal never asks for more than it states)
    try t.expectEqual(r.n_rec, after.n_rec);
}

test "decode goal 100 on prose is unreachable: alone one stream does about 66" {
    var f: FakePlant = .{ .n = 5, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const r = c.report(.{ .mode = .decode_first, .decode_user_tps = 100 }, 5);
    try t.expectEqual(State.@"unreachable", r.state);
    try t.expectEqual(Reason.ceiling, r.reason);
    try t.expect(r.decode_user_max < 100 and r.decode_user_max > 60);
    // the recommendation stays inside the prompt floor and the cap
    try t.expect(r.n_rec.? >= 1 and r.n_rec.? <= 256);
}

test "decode goal 60 is reachable above the 300 floor (N ~170, the prompt ~560); a 1,000 floor makes it short, held at the floor's N" {
    var f: FakePlant = .{ .n = 5, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const open = c.report(.{ .mode = .decode_first, .decode_user_tps = 60 }, 5);
    try t.expectEqual(State.short, open.state);
    try t.expectEqual(Reason.below_goal_now, open.reason);
    try t.expect(open.n_rec.? > 150 and open.n_rec.? < 190);
    try t.expect((Plant{}).prefillBeside(@floatFromInt(open.n_rec.?), 1) >= 300);
    const r = c.report(.{ .mode = .decode_first, .decode_user_tps = 60, .prefill_floor_tps = 1000 }, 5);
    try t.expectEqual(State.short, r.state);
    try t.expectEqual(Reason.starves_prefill, r.reason);
    try t.expect((Plant{}).prefillBeside(@floatFromInt(r.n_rec.?), 1) >= 1000);
}

test "decode goal over the cap is short with the cap's reason" {
    var f: FakePlant = .{ .n = 5, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const r = c.report(.{ .mode = .decode_first, .decode_user_tps = 60, .n_cap = 100, .prefill_floor_tps = 1 }, 5);
    try t.expectEqual(State.short, r.state);
    try t.expectEqual(Reason.over_cap, r.reason);
    try t.expectEqual(@as(?u32, 100), r.n_rec);
}

test "prefill goal 2,000 needs N near 37 with one user; applied to the plant it is met" {
    var f: FakePlant = .{ .n = 5, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const cfg: Config = .{ .mode = .prefill_first, .prefill_tps = 2000 };
    const r = c.report(cfg, 5);
    // at N=5 the prompt runs ~5,000: the goal is met with room, and the largest N that holds it is ~37
    try t.expectEqual(State.meeting, r.state);
    try t.expect(r.n_rec.? >= 34 and r.n_rec.? <= 40);
    f.n = @floatFromInt(r.n_rec.?);
    c.clear();
    f.run(&c, 10);
    try t.expectEqual(State.meeting, c.report(cfg, r.n_rec).state);
}

test "prefill goal 5,000 at N=20 is short now; the largest N that holds it is about 5" {
    var f: FakePlant = .{ .n = 20, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const r = c.report(.{ .mode = .prefill_first, .prefill_tps = 5000 }, 20);
    try t.expectEqual(State.short, r.state);
    try t.expectEqual(Reason.below_goal_now, r.reason);
    try t.expect(r.n_rec.? >= 4 and r.n_rec.? <= 6);
}

test "prefill goal 7,000 is unreachable: alone the step does about 6,500" {
    var f: FakePlant = .{ .n = 5, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const r = c.report(.{ .mode = .prefill_first, .prefill_tps = 7000 }, 5);
    try t.expectEqual(State.@"unreachable", r.state);
    try t.expectEqual(Reason.ceiling, r.reason);
}

test "prefill goal 6,000 starves the decoder it sits beside: short, and it says so" {
    var f: FakePlant = .{ .n = 1, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 10);
    const r = c.report(.{ .mode = .prefill_first, .prefill_tps = 6000 }, 1);
    try t.expectEqual(State.short, r.state);
    try t.expectEqual(Reason.starves_decode, r.reason);
}

test "nothing to judge is idle, not meeting: no decoding user under a decode goal, no prompt under a prefill goal" {
    var f: FakePlant = .{ .n = 5, .users = 0 };
    var c: Controller = .{};
    f.run(&c, 10);
    try t.expectEqual(State.idle, c.report(.{ .mode = .decode_first, .decode_user_tps = 16 }, 5).state);
    var g: FakePlant = .{ .n = 5, .users = 2, .filling = false };
    var d: Controller = .{};
    g.run(&d, 10);
    try t.expectEqual(State.idle, d.report(.{ .mode = .prefill_first, .prefill_tps = 2000 }, 5).state);
    // a mode with no value for its goal is idle with its reason
    const none = d.report(.{ .mode = .prefill_first }, 5);
    try t.expectEqual(State.idle, none.state);
    try t.expectEqual(Reason.no_goal, none.reason);
}

test "the verdict does not flap on the rig's noise: ten windows of +-4 % give one state and one recommendation" {
    var f: FakePlant = .{ .n = 26, .users = 1 };
    var c: Controller = .{};
    const cfg: Config = .{ .mode = .decode_first, .decode_user_tps = 40 };
    var first: ?Report = null;
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        f.run(&c, 1);
        if (c.taken < 3) continue;
        const r = c.report(cfg, 26);
        if (first == null) first = r;
        try t.expectEqual(first.?.n_rec, r.n_rec);
        try t.expectEqual(State.meeting, r.state);
    }
}

test "more users make the same goal cost more N (a round gets longer with users)" {
    var one: FakePlant = .{ .n = 5, .users = 1 };
    var four: FakePlant = .{ .n = 5, .users = 4 };
    var c1: Controller = .{};
    var c4: Controller = .{};
    one.run(&c1, 10);
    four.run(&c4, 10);
    const cfg: Config = .{ .mode = .decode_first, .decode_user_tps = 30 };
    try t.expect(c4.report(cfg, 5).n_rec.? > c1.report(cfg, 5).n_rec.?);
}

test "a short window judges what it has" {
    var f: FakePlant = .{ .n = 5, .users = 1 };
    var c: Controller = .{};
    f.run(&c, 2);
    const r = c.report(.{ .mode = .decode_first, .decode_user_tps = 15 }, 5);
    try t.expectEqual(@as(u32, 2), r.window_s);
    try t.expectEqual(@as(u32, 2), r.judged_s);
}
