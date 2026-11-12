//! A decode ring's progress watchdog: a sequence this rank posted whose peers' parts do not all arrive within `dead` ticks fails
//! the ring by name, so a round that waits for a peer which never joined it (a collect kernel polling flags nobody writes) ends
//! with a line that says where this rank's host was, never silently. The proxy owns it (one thread): no atomics.
const std = @import("std");

pub const max_slots = 32;

pub const Watch = struct {
    dead: u64, // ticks without progress while a sequence is open
    open: u32 = 0, // sequences posted whose peers' parts are not all published
    last: u64 = 0, // when a sequence was last posted or published
    early: [max_slots]bool = @splat(false), // slot's sequence completed before this rank posted it

    /// This rank posted the sequence in `slot`. A sequence whose parts were all there before the post is not open.
    pub fn posted(w: *Watch, slot: usize, now: u64) void {
        w.last = now;
        if (w.early[slot]) {
            w.early[slot] = false;
            return;
        }
        w.open += 1;
    }

    /// The last peer's part of the sequence in `slot` was published; `was_posted`: this rank had posted it.
    pub fn completed(w: *Watch, slot: usize, now: u64, was_posted: bool) void {
        w.last = now;
        if (was_posted) {
            if (w.open > 0) w.open -= 1;
        } else w.early[slot] = true;
    }

    /// Whether a sequence has been open for `dead` ticks with nothing posted or published since.
    pub fn expired(w: *const Watch, now: u64) bool {
        return w.open > 0 and now -% w.last > w.dead;
    }
};

test "a sequence open past the limit with no progress expires; progress, a late post of an early sequence and an idle ring do not" {
    var w: Watch = .{ .dead = 100 };
    try std.testing.expect(!w.expired(10_000)); // nothing open: idle
    w.posted(1, 1000);
    try std.testing.expect(!w.expired(1050));
    try std.testing.expect(w.expired(1101)); // the peers never came
    w.completed(1, 1101, true);
    try std.testing.expect(!w.expired(5000));
    // two open, one completes: the clock restarts from that progress
    w.posted(2, 6000);
    w.posted(3, 6010);
    w.completed(2, 6090, true);
    try std.testing.expect(!w.expired(6150));
    try std.testing.expect(w.expired(6191)); // sequence 3 still open
    // peers ahead of this rank: the sequence completes before the post and is never open
    var v: Watch = .{ .dead = 100 };
    v.completed(5, 100, false);
    v.posted(5, 120);
    try std.testing.expectEqual(@as(u32, 0), v.open);
    try std.testing.expect(!v.expired(100_000));
    // a counter that wrapped
    var u: Watch = .{ .dead = 100, .last = std.math.maxInt(u64) - 10 };
    u.open = 1;
    try std.testing.expect(!u.expired(50));
    try std.testing.expect(u.expired(200));
}

test "the proxy posts, publishes and checks the watch, and a ring that expires fails by name with its limit" {
    const src = @embedFile("rdma.zig");
    for ([_][]const u8{
        "var watch: ring_watch.Watch = .{ .dead = dead_ms * 1000 * tpu };",
        "watch.posted(slot, post_t[slot]);",
        "watch.completed(sl, instr.ticks(), posted >= q);",
        "if (watch.open > 0 and watch.expired(instr.ticks())) {",
        "a sequence posted by this rank has had no peer part for {d} ms",
        "where.report(&pb)",
        "r.fail(msg);",
        "\"TF_DS_RING_DEAD_MS\"",
    }) |needle| try std.testing.expect(std.mem.indexOf(u8, src, needle) != null);
}

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
