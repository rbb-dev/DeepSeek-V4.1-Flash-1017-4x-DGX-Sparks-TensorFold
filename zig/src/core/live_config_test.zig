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
    try std.testing.expectEqual(@as(usize, live.CfgFrame.head_len + 10), b.len);
    const d = try live.CfgFrame.decode(b);
    try std.testing.expectEqual(@as(u64, 0x0102030405060708), d.version);
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