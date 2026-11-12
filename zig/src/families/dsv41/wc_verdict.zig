const std = @import("std");

pub const short_ms: f32 = 0.05; // under this a launch is too short to time
pub const noisy_pct: f32 = 5.0; // a kernel timed fairly may read this far from its control and be called noisy
pub const short_noisy_ms: f32 = 0.1; // a kernel under this (a few tens of us) reads up to short_noisy_pct from its control and is noisy, not wrong
pub const short_noisy_pct: f32 = 25.0;


/// `null_pct`: the null control's difference to the first set in percent; `ms0`: the first set's ms a launch.
pub fn verdict(null_pct: f32, ms0: f32) Verdict {
    const a = @abs(null_pct);
    if (a <= 2.0) return .holds;
    if (ms0 < short_ms) return .too_short;
    if (a <= noisy_pct) return .noisy;
    if (ms0 < short_noisy_ms and a <= short_noisy_pct) return .noisy_short;
}

    try std.testing.expectEqual(Verdict.holds, verdict(1.9, 0.5));
    try std.testing.expectEqual(Verdict.holds, verdict(-2.0, 0.01));
    try std.testing.expectEqual(Verdict.too_short, verdict(30.0, 0.04));
    try std.testing.expectEqual(Verdict.noisy, verdict(4.5, 0.0533)); // testa-replay-33 line 19: add2, null 4.5 %
    try std.testing.expectEqual(Verdict.noisy, verdict(-4.9, 3.0));
    // testa-replay-34 line 61: add2 0.0618 against 0.0532 = -13.9 %: this failed the replay in 2 of 3 runs
    try std.testing.expectEqual(Verdict.noisy_short, verdict((0.0532 / 0.0618 - 1.0) * 100.0, 0.0618));
}
