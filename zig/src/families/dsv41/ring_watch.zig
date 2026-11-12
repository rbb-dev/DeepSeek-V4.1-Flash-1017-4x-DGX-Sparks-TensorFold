
/// Whether the watchdog opens a sequence at its post. Under TF_DS_RING_WORD the receiving proxy sees no arrival (the sender's
/// NIC writes the generation words straight into memory), so nothing would ever close a sequence and an idle ring would fail
/// ten seconds after its last post; there the collecting kernel's own deadline and the abort word are the bound.
pub fn armed(arrivals_seen: bool) bool {
    return arrivals_seen;
}

test "a ring that waits on generation words opens no watch: the proxy sees no arrival to close it" {
    try std.testing.expect(armed(true) and !armed(false));
    const src = @embedFile("rdma.zig");
    const code = src[0..std.mem.indexOf(u8, src, "\ntest \"").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, code, "watch.posted(slot, post_t[slot])"));
    try std.testing.expect(std.mem.indexOf(u8, code, "if (ring_watch.armed(!r.roci_live)) watch.posted(slot, post_t[slot]);") != null);
}
