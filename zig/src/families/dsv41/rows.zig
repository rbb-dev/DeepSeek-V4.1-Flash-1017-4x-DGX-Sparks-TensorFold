//! A stream's rows in the lane pool: what a prompt plus its reply takes, and so the most a request may use. The lanes
//! check `streamRows(...) <= ctx_cap` when they place a stream (lanes.zig prefillFn); the server is told
//! `usableRows(...)` as its context window and admits a request only when prompt + reply tokens fit it (server/chat.zig
//! prepare). The two must agree: before this the server was told the whole window, clamped a request that sent no
//! max_tokens to exactly that, and the lanes then refused it (round_rows + 2 rows short, every time).
const std = @import("std");

/// Rows a stream takes: its prompt, its reply budget, the rows a decode round adds (drafts of every stream together,
/// `round_rows`, and 2), in whole prompt chunks of `chunk` rows.
pub fn streamRows(prompt: usize, max_new: usize, round_rows: usize, chunk: usize) usize {
    return std.mem.alignForward(usize, prompt + max_new + round_rows + 2, chunk);
}

/// The most prompt plus reply tokens a stream holds in a `ctx_cap`-row window: the server's context window.
pub fn usableRows(ctx_cap: usize, round_rows: usize, chunk: usize) usize {
    const whole = std.mem.alignBackward(usize, ctx_cap, chunk); // streamRows rounds up to whole chunks
    return if (whole > round_rows + 2) whole - round_rows - 2 else 0;
}

test "the window the server is told is exactly what a stream holds" {
    const chunk = 2048;
    for ([_]usize{ 1_048_576, 1_048_575, 262_144, 8192 }) |cap| for ([_]usize{ 1, 48, 64 }) |rows| {
        const u = usableRows(cap, rows, chunk);
        for ([_]usize{ 1, 1577, u / 2, u - 1 }) |prompt| {
            try std.testing.expect(streamRows(prompt, u - prompt, rows, chunk) <= cap); // the whole room fits
            try std.testing.expect(streamRows(prompt, u - prompt + 1, rows, chunk) > cap); // one token more does not
        }
    };
}

test "the running engine: a 1,048,576-token window with 48-row rounds holds 1,048,526" {
    try std.testing.expectEqual(@as(usize, 1_048_526), usableRows(1_048_576, 48, 2048));
    // a client's request: 1,577 prompt tokens, no max_tokens; the server gives it every token left
    try std.testing.expect(streamRows(1577, usableRows(1_048_576, 48, 2048) - 1577, 48, 2048) <= 1_048_576);
}
