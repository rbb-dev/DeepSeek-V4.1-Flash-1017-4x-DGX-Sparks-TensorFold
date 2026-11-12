//! roci_wc_classes' verdict on a class's null control (the same cudaMalloc set timed twice): which of "holds", "too short to time", "noisy",
//! "unresolved" the replay prints. A class line times page types and compares no bytes, so no verdict fails the replay: a null control past
//! every bar says the timing is unresolved, never that a kernel is wrong. Pure, so the host tests run it (the replay's own exe has no tests).
const std = @import("std");

pub const short_ms: f32 = 0.05; // under this a launch is too short to time
pub const noisy_pct: f32 = 5.0; // a kernel timed fairly may read this far from its control and be called noisy
pub const short_noisy_ms: f32 = 0.1; // a kernel under this (a few tens of us) reads up to short_noisy_pct from its control and is noisy, not wrong
pub const short_noisy_pct: f32 = 25.0;

pub const Verdict = enum { holds, too_short, noisy, noisy_short, unresolved };

/// `null_pct`: the null control's difference to the first set in percent; `ms0`: the first set's ms a launch.
pub fn verdict(null_pct: f32, ms0: f32) Verdict {
    const a = @abs(null_pct);
    if (a <= 2.0) return .holds;
    if (ms0 < short_ms) return .too_short;
    if (a <= noisy_pct) return .noisy;
    if (ms0 < short_noisy_ms and a <= short_noisy_pct) return .noisy_short;
    return .unresolved;
}

test "the null control's verdict: within 2 % holds, a launch under 50 us is too short to time, 2 to 5 % is noisy, a kernel under 100 us up to 25 % is noisy, anything else is unresolved" {
    try std.testing.expectEqual(Verdict.holds, verdict(1.9, 0.5));
    try std.testing.expectEqual(Verdict.holds, verdict(-2.0, 0.01));
    try std.testing.expectEqual(Verdict.too_short, verdict(30.0, 0.04));
    try std.testing.expectEqual(Verdict.noisy, verdict(4.5, 0.0533)); // testa-replay-33 line 19: add2, null 4.5 %
    try std.testing.expectEqual(Verdict.noisy, verdict(-4.9, 3.0));
    // testa-replay-34 line 61: add2 0.0618 against 0.0532 = -13.9 %: this failed the replay in 2 of 3 runs
    try std.testing.expectEqual(Verdict.noisy_short, verdict((0.0532 / 0.0618 - 1.0) * 100.0, 0.0618));
    // a long kernel that disagrees with itself is unresolved, and so is a short one beyond 25 % (T15's replay: rmsnorm into qr,
    // 0.0549 ms against its null 0.0401 = -27.1 %, failed a byte proof whose every byte line held)
    try std.testing.expectEqual(Verdict.unresolved, verdict(8.0, 3.0));
    try std.testing.expectEqual(Verdict.unresolved, verdict(8.0, 0.2));
    try std.testing.expectEqual(Verdict.unresolved, verdict(40.0, 0.06));
    try std.testing.expectEqual(Verdict.unresolved, verdict((0.0401 / 0.0549 - 1.0) * 100.0, 0.0549));
}
