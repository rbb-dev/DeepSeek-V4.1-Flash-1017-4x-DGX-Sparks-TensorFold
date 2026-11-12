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
    if (std.mem.eql(u8, v, "backoff")) return .backoff;

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
