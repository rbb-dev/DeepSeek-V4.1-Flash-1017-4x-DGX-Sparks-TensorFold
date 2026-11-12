//! A share of wall time for prompt steps beside decoding streams (TENSORFOLD_PREFILL_SHARE).
//!
//! The host alternates one prompt step with one decode round. Decoders wait at most one step but get one round per
//! step. With a share p below 1 a long prompt's step that took t owes the decoders t (1 - p) / p of decode time, and
//! the host runs rounds, not prompt steps, until the rounds have paid it. p = 1 (the default, unset) owes nothing:
//! the baseline alternation, one step and one round.
//!
//! TF_DS_ROUNDS_BESIDE=N is the other control, by count: N Token Generation rounds follow every long prompt's step while
//! a stream generates, 2N after a step that ran as a stagger pair. It owes rounds, not time, so it times nothing.
//!
//! Ported from ashhart/TensorFold (lane_host_fill.zig: "lane host: a share of wall time for prompt steps beside
//! decoding streams", Jürgen Schmied; the share's floor, the debt's bound and the decoders that ended or paused).
const std = @import("std");

pub const Share = struct {
    /// The prompt steps' share of wall time while streams decode, in [min_share, 1].
    p: f64 = 1,
    /// Decode time owed to the streams before the next long prompt's step.
    debt_ns: i64 = 0,
    /// A prompt with at most this many rows left never waits for the debt: a short chat goes before a long prompt's
    /// next step. A field so a host test can use small prompts.
    short: u64 = short_rows,

    /// TF_DS_ROUNDS_BESIDE (0: unset): the rounds a long prompt's step is followed by while a stream generates.
    rounds: u32 = 0,
    /// Rounds still owed to the streams from the last long step (rounds, or twice that after a pair).
    due: u32 = 0,

    pub const short_rows = 512;
    pub const min_share = 0.01; // below it a step's debt is minutes of decoding, and is no share

    /// The share from TENSORFOLD_PREFILL_SHARE; unset or unreadable is 1.
    pub fn fromEnv() Share {
        const text = std.c.getenv("TENSORFOLD_PREFILL_SHARE") orelse return .{};
        return fromText(std.mem.span(text));
    }

    /// A share from its text: anything but a number in [min_share, 1] is the default, 1.
    pub fn fromText(text: []const u8) Share {
        const p = std.fmt.parseFloat(f64, text) catch return .{};
        return .{ .p = if (p >= min_share and p <= 1) p else 1 };
    }

    /// A long prompt's step that took `ns` while `decoding` streams were live: the decode time it owes them.
    pub fn stepped(s: *Share, ns: i64, decoding: bool) void {
        if (s.p >= 1 or !decoding) return;
        const owed = @as(f64, @floatFromInt(ns)) * (1 - s.p) / s.p;
        s.debt_ns +|= @intFromFloat(@min(owed, @as(f64, @floatFromInt(std.math.maxInt(i64) / 2))));
    }

    /// A long prompt's step beside `decoding` streams under TF_DS_ROUNDS_BESIDE: the rounds it owes them, twice as many
    /// after a pair of passes (`pair`, the backend's report: Stream.step_pair).
    pub fn stepRounds(s: *Share, decoding: bool, pair: bool) void {
        if (s.rounds == 0 or !decoding) return;
        s.due = if (pair) s.rounds *| 2 else s.rounds;
    }

    /// A decode round of `ns` pays the debt down, and is one of the rounds owed by count.
    pub fn rounded(s: *Share, ns: i64) void {
        s.debt_ns = @max(0, s.debt_ns - ns);
        s.due -|= 1;
    }

    /// Whether decode time (or rounds) are owed before a long prompt's next step.
    pub fn owes(s: *const Share) bool {
        return (s.p < 1 and s.debt_ns > 0) or s.due > 0;
    }

    /// Whether the host keeps any account of streams that decode: a share below 1 or a round count.
    pub fn active(s: *const Share) bool {
        return s.p < 1 or s.rounds > 0;
    }

    /// Nothing is owed any more (the streams it was owed to ended or paused).
    pub fn forgive(s: *Share) void {
        s.debt_ns = 0;
        s.due = 0;
    }

    /// Why the two controls cannot both be set, or null: one control, not two fighting.
    pub fn refusal(s: Share) ?[]const u8 {
        if (s.rounds != 0 and s.p < 1) return "TF_DS_ROUNDS_BESIDE and TENSORFOLD_PREFILL_SHARE below 1 are both set: they are two controls of the same rounds; set one (TF_DS_ROUNDS_BESIDE counts rounds a step, the share repays time)";
        return null;
    }
};

/// TF_DS_ROUNDS_BESIDE's setting: unset is 0 (none); a whole number of 1 or more (digits only) is the rounds; anything
/// else is null, no setting (the start refuses it).
pub fn roundsFromText(text: ?[]const u8) ?u32 {
    const t = text orelse return 0;
    if (t.len == 0) return null;
    for (t) |c| if (c < '0' or c > '9') return null;
    const v = std.fmt.parseInt(u32, t, 10) catch return null;
    return if (v >= 1) v else null;
}

/// The start line (rank 0's host): the share, the rows of a prompt step while a stream decodes, and the rounds that
/// follow a step. `rows` 0: the family sets none. `rounds` above 0: TF_DS_ROUNDS_BESIDE's line.
pub fn startLine(buf: []u8, p: f64, rows: u32, rounds: u32, unit: u32) []const u8 {
    var rows_text: [160]u8 = undefined;
    const rows_part = if (rows == 0) "the family's own rows" else if (unit == 0) (std.fmt.bufPrint(&rows_text, "up to {d} rows", .{rows}) catch "") else (std.fmt.bufPrint(&rows_text, "up to {d} rows, in whole {d}-row units and never above the pass rows", .{ rows, unit }) catch "");
    if (rounds != 0) return std.fmt.bufPrint(buf, "Prompt Processing beside Token Generation: TF_DS_ROUNDS_BESIDE={d}; while a stream generates, a prompt step runs {s} (TF_DS_PASS_BESIDE); {d} Token Generation rounds follow every prompt step, {d} after a stagger pair (a prompt of at most {d} rows steps ahead of them)", .{ rounds, rows_part, rounds, rounds *| 2, Share.short_rows }) catch buf[0..0];
    if (p >= 1) return std.fmt.bufPrint(buf, "Prompt Processing beside Token Generation: share 1 (TENSORFOLD_PREFILL_SHARE unset or 1); while a stream generates, a prompt step runs {s} (TF_DS_PASS_BESIDE); one Token Generation round follows each step", .{rows_part}) catch buf[0..0];
    return std.fmt.bufPrint(buf, "Prompt Processing beside Token Generation: share {d} (TENSORFOLD_PREFILL_SHARE); while a stream generates, a prompt step runs {s} (TF_DS_PASS_BESIDE); Token Generation rounds, at least one, follow each step until their time is {d:.2} times the step's", .{ p, rows_part, (1 - p) / p }) catch buf[0..0];
}

test "the start line names the share, the step rows and the rounds' rule" {
    var buf: [512]u8 = undefined;
    const plain = startLine(&buf, 1, 2048, 0, 2048);
    try std.testing.expect(std.mem.indexOf(u8, plain, "share 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, plain, "up to 2048 rows, in whole 2048-row units and never above the pass rows") != null);
    try std.testing.expect(std.mem.indexOf(u8, plain, "one Token Generation round follows each step") != null);
    const half = startLine(&buf, 0.5, 4096, 0, 2048);
    try std.testing.expect(std.mem.indexOf(u8, half, "share 0.5") != null);
    try std.testing.expect(std.mem.indexOf(u8, half, "up to 4096 rows") != null);
    try std.testing.expect(std.mem.indexOf(u8, half, "1.00 times the step's") != null);
    const quarter = startLine(&buf, 0.25, 0, 0, 2048);
    try std.testing.expect(std.mem.indexOf(u8, quarter, "the family's own rows") != null);
    try std.testing.expect(std.mem.indexOf(u8, quarter, "3.00 times") != null);
}

test "a share of 0.5 owes the decoders a step's time; 1 owes nothing; rounds pay it down" {
    var s: Share = .{ .p = 0.5 };
    s.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.stepped(1000, false);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.rounded(400);
    s.rounded(800);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
    var one: Share = .{};
    one.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 0), one.debt_ns);
    var q: Share = .{ .p = 0.25 };
    q.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 3000), q.debt_ns);
}

test "a share below the floor, NaN and inf fall back to 1; the floor and 1 stand" {
    for ([_][]const u8{ "1e-12", "0.0099", "0", "-0.5", "nan", "inf", "-inf", "2", "x", "" }) |text| {
        try std.testing.expectEqual(@as(f64, 1), Share.fromText(text).p);
    }
    try std.testing.expectEqual(@as(f64, 0.01), Share.fromText("0.01").p);
    try std.testing.expectEqual(@as(f64, 1), Share.fromText("1").p);
    try std.testing.expectEqual(@as(f64, 0.5), Share.fromText("0.5").p);
}

test "a step's debt saturates instead of overflowing" {
    var s: Share = .{ .p = Share.min_share };
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expect(s.debt_ns > 0);
    s.stepped(std.math.maxInt(i64), true);
    try std.testing.expectEqual(@as(i64, std.math.maxInt(i64)), s.debt_ns);
    s.rounded(1000);
    try std.testing.expect(s.debt_ns < std.math.maxInt(i64));
}

test "the share's debt is the upstream formula: t (1 - p) / p of decode time for a step of t" {
    for ([_]f64{ 0.01, 0.1, 0.25, 0.5, 0.75, 0.9 }) |p| {
        var s: Share = .{ .p = p };
        s.stepped(2_000_000, true);
        const want: i64 = @intFromFloat(2_000_000.0 * (1 - p) / p);
        try std.testing.expectEqual(want, s.debt_ns);
        try std.testing.expect(s.owes());
    }
    var one: Share = .{};
    try std.testing.expect(!one.owes());
}

test "TENSORFOLD_PREFILL_SHARE unset is a share of 1: nothing owed, the host's one step and one round" {
    const env = struct { // the process environment, as the host reads it
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    const name = "TENSORFOLD_PREFILL_SHARE";
    if (std.c.getenv(name) != null) return error.SkipZigTest; // the caller's own setting: left alone
    defer _ = env.unsetenv(name);
    try std.testing.expectEqual(@as(f64, 1), Share.fromEnv().p);
    _ = env.setenv(name, "0.5", 1);
    try std.testing.expectEqual(@as(f64, 0.5), Share.fromEnv().p);
    _ = env.setenv(name, "", 1);
    try std.testing.expectEqual(@as(f64, 1), Share.fromEnv().p);
    _ = env.unsetenv(name);
    try std.testing.expectEqual(@as(f64, 1), Share.fromEnv().p);
}
