//! The live block: what rank 0's host reads once a turn, written by the admin API's threads.
const std = @import("std");
const live = @import("live_config.zig");

test "a published value reaches the host at its next turn, once, and the active version follows it" {
    var b: live.Block = .{};
    var seen: u64 = 0;
    var vals: [live.slot_count]u64 = undefined;
    try std.testing.expect(!b.turn(&seen, &vals)); // nothing published: no change, and no copy
    try std.testing.expectEqual(@as(u64, 0), b.active_version.load(.acquire));
    b.publish(.rounds_beside, 5, 7);
    try std.testing.expectEqual(@as(u64, 0), b.active_version.load(.acquire)); // the host has not turned yet
    try std.testing.expect(b.turn(&seen, &vals));
    try std.testing.expectEqual(@as(u64, 5), vals[@backingInt(live.Slot.rounds_beside)]);
    try std.testing.expectEqual(@as(u64, 7), b.active_version.load(.acquire));
    try std.testing.expect(!b.turn(&seen, &vals)); // the same version: nothing more
    b.publish(.rounds_beside, live.unset, 8);
    try std.testing.expect(b.turn(&seen, &vals));
    try std.testing.expectEqual(live.unset, vals[@backingInt(live.Slot.rounds_beside)]);
}

test "a wired row is named by its id, and no other row is" {
    try std.testing.expectEqual(live.Slot.rounds_beside, live.slotOf(122).?);
    try std.testing.expectEqual(live.Slot.log_polls, live.slotOf(4).?);
    try std.testing.expect(live.slotOf(110) == null);
    try std.testing.expect(live.slotOf(504) == null);
    const settings = @import("settings.zig");
    try std.testing.expectEqual(live.Slot.keep_ahead, live.slotOf(412).?);
    // a wired row is one of class H, or of class A exactly where the slot travels in a cfg frame
    for (live.wired) |w| try std.testing.expectEqual(if (live.isA(w.slot)) settings.Apply.A else settings.Apply.H, settings.byId(w.id).?.apply);
}

test "a row of class A is live only where the engine says it can take it" {
    live.able.store(0, .release);
    try std.testing.expect(live.takes(122)); // class H: its hook is enough
    try std.testing.expect(!live.takes(412));
    try std.testing.expect(!live.takes(101)); // no hook (nor a fill row: takes is the slots' alone)
    live.setAble(.keep_ahead, true);
    try std.testing.expect(live.takes(412));
    live.setAble(.keep_ahead, false);
    try std.testing.expect(!live.takes(412));
}

test "the cfg frame's bytes: a round trip, the bounds, one send() whole" {
    var buf: [live.CfgFrame.head_len + 5 * live.CfgFrame.max_items]u8 = undefined;
    const items = [_]live.CfgItem{ .{ .slot = 2, .value = 1 }, .{ .slot = 7, .value = live.cfg_unset } };
    const b = try live.CfgFrame.encode(&buf, 0x0102030405060708, 0, &items);
    try std.testing.expectEqual(@as(usize, live.CfgFrame.head_len + 10), b.len);
    const d = try live.CfgFrame.decode(b);
    try std.testing.expectEqual(@as(u64, 0x0102030405060708), d.version);
    try std.testing.expectEqual(@as(u8, 0), d.flags);
    try std.testing.expectEqual(@as(u8, 2), d.n);
    try std.testing.expectEqual(items[0], live.CfgFrame.item(d, 0));
    try std.testing.expectEqual(items[1], live.CfgFrame.item(d, 1));
    // a cut or padded frame is refused, a frame of the most items still fits the link's one send() with its header (kind + step = 9 bytes)
    try std.testing.expectError(error.BadFrame, live.CfgFrame.decode(b[0 .. b.len - 1]));
    try std.testing.expectError(error.BadFrame, live.CfgFrame.decode(b[0..5]));
    try std.testing.expect(9 + live.CfgFrame.head_len + 5 * live.CfgFrame.max_items <= 2048);
    var many: [live.CfgFrame.max_items + 1]live.CfgItem = @splat(.{ .slot = 0, .value = 0 });
    try std.testing.expectError(error.CfgTooLong, live.CfgFrame.encode(&buf, 1, 0, &many));
    try std.testing.expectError(error.CfgTooLong, live.CfgFrame.encode(buf[0..11], 1, 0, &items));
}

test "a change storm: the host never sees a torn pair of values, and its version only rises" {
    var b: live.Block = .{};
    var stop = std.atomic.Value(bool).init(false);
    const Writer = struct {
        // Bursts of publications, every slot always equal, one after another with no pause (the reader overlaps a write: a torn
        // pair is possible there), then a rest. A writer that never left the odd word would starve every seqlock reader, which is
        // no property of the block; the real writers are an admin API's threads, and the rests are where the reader gets through.
        fn run(blk: *live.Block, flag: *std.atomic.Value(bool)) void {
            var v: u64 = 1;
            while (!flag.load(.acquire)) {
                for (0..200) |_| {
                    blk.publishAll(@splat(v), v);
                    v += 1;
                }
                for (0..3000) |_| std.atomic.spinLoopHint();
            }
        }
    };
    const t = try std.Thread.spawn(.{}, Writer.run, .{ &b, &stop });
    var seen: u64 = 0;
    var vals: [live.slot_count]u64 = undefined;
    var last: u64 = 0;
    var turns: usize = 0;
    var changes: usize = 0;
    var failure: ?anyerror = null;
    while (changes < 60_000 and turns < 400_000_000) : (turns += 1) {
        if (b.turn(&seen, &vals)) {
            changes += 1;
            for (vals[1..]) |x| if (x != vals[0]) {
                failure = error.TornPair; // the slots are one publication
            };
            if (seen < last) failure = error.VersionWentBack;
            last = seen;
            if (failure != null) break;
        }
    }
    stop.store(true, .release); // (stopped before the checks: a failed expect must not leave the writer running)
    t.join();
    if (failure) |f| {
        std.debug.print("storm: {s} after {d} changes (slot 0 = {d}, version {d})\n", .{ @errorName(f), changes, vals[0], seen });
        return f;
    }
    try std.testing.expect(changes >= 60_000); // a storm that the reader got through, not a single change
}

test "a turn with nothing changed is one atomic load: the version word is all it reads" {
    var b: live.Block = .{};
    b.publish(.log_polls, 1, 3);
    var seen: u64 = 3;
    var vals: [live.slot_count]u64 = @splat(99);
    try std.testing.expect(!b.turn(&seen, &vals));
    try std.testing.expectEqual(@as(u64, 99), vals[0]); // untouched
}

test "the rows with a hook: the host's step rows, the share, the tier's timings and the fill order are wired, each a row of class H" {
    const settings = @import("settings.zig");
    for ([_]u16{ 100, 106, 119, 120, 121, 122, 4, 446, 447, 448, 449, 450, 451, 452, 453 }) |id| {
        try std.testing.expect(live.isWired(id));
        try std.testing.expectEqual(settings.Apply.H, settings.byId(id).?.apply);
    }
    try std.testing.expect(!live.isWired(110)); // pass rows: a start's
    try std.testing.expect(!live.isWired(440)); // the store's quota: not the host's
    try std.testing.expectEqual(live.Slot.prefill_share, live.slotOf(100).?);
    try std.testing.expect(live.slotOf(120) == null and live.isFillRow(120) and live.isFillRow(121) and live.isFillRow(119));
}

test "a fill policy published with the values reaches the host whole, with the version it belongs to" {
    const fill_order = @import("fill_order.zig");
    var b: live.Block = .{};
    var seen: u64 = 0;
    var snap: live.Snapshot = undefined;
    try std.testing.expect(!b.turnAll(&seen, &snap));
    const turns = fill_order.parse("turns", null, "12").policy;
    b.publishFill(turns, 4);
    try std.testing.expect(b.turnAll(&seen, &snap));
    try std.testing.expect(snap.fill != null and snap.fill.?.max_wait_s == 12 and snap.fill.?.order == .turns);
    try std.testing.expectEqual(@as(u64, 4), b.active_version.load(.acquire));
    b.publishFill(null, 5); // back to the start's
    try std.testing.expect(b.turnAll(&seen, &snap));
    try std.testing.expect(snap.fill == null);
}
