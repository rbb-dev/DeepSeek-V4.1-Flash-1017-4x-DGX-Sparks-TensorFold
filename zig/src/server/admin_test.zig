    try std.testing.expect(std.mem.indexOf(u8, st.body, "\"writable_classes\": [\"H\", \"A\", \"R\"]") != null);

test "set: keep-ahead (class A) is live only where the engine can take it, then it is published for the host to send to every rank" {
    var rig: Rig = .{ .arena = undefined, .tmp = undefined };
    try rig.open(null, admin_keys_file);
    defer rig.close();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try Writable.open(&rig, "schema 1\n");
    defer w.restore();
    defer live.able.store(0, .release);
    // an engine that did not verify it at its start: the row is a setting for the next start; a live-only change is refused by name
    live.able.store(0, .release);
    const no = try setBody(&rig, "{\"set\": {\"tf_ds_engram_keep_ahead\": 0}, \"persist\": false, \"wait_ms\": 0}");
    defer no.free();
    try std.testing.expectEqual(@as(u16, 400), no.status);
    try std.testing.expect(std.mem.indexOf(u8, no.body, "invalid_apply_mode") != null);
    var seen: u64 = 0;
    var vals: [live.slot_count]u64 = undefined;
    try std.testing.expect(!live.block.turn(&seen, &vals)); // nothing was published
    // one that did: live, one publication in the class A slot, the host's turn takes it (and sends the frame: core's test)
    live.setAble(.keep_ahead, true);
    const yes = try setBody(&rig, "{\"set\": {\"tf_ds_engram_keep_ahead\": 0}, \"persist\": false, \"wait_ms\": 0}");
    defer yes.free();
    try std.testing.expectEqual(@as(u16, 200), yes.status);
    const doc = (try json.parse(a, yes.body)).ok;
    const ch = doc.get("changes").?.array[0];
    try std.testing.expectEqualStrings("live", ch.get("apply_mode").?.string);
    try std.testing.expectEqualStrings("A", ch.get("apply_class").?.string);
    try std.testing.expect(live.block.turn(&seen, &vals));
    try std.testing.expectEqual(@as(u64, 0), vals[@backingInt(live.Slot.keep_ahead)]); // 0: a change from the default-on 1 (speed_defaults, T6)
    // the engine's host has taken the version: the row is active (by construction: rank 0 handed the frame to the link)
    const cfg = try request(rig.srv, "GET", base("config"), bearer_read);
    defer cfg.free();
    const cd = (try json.parse(a, cfg.body)).ok;
    var found = false;
    for (cd.get("values").?.array) |v| if (std.mem.eql(u8, v.get("name").?.string, "tf_ds_engram_keep_ahead")) {
        found = true;
        try std.testing.expectEqualStrings("0", v.get("value").?.string); // the value set above (0, a change from the default-on 1)
        try std.testing.expectEqualStrings("active", v.get("state").?.string);
    };
    try std.testing.expect(found);
    // unset returns it to the start's: the slot carries the marker
    const un = try setBody(&rig, "{\"set\": {\"tf_ds_engram_keep_ahead\": null}, \"persist\": false, \"wait_ms\": 0}");
    defer un.free();
    try std.testing.expectEqual(@as(u16, 200), un.status);
    try std.testing.expect(live.block.turn(&seen, &vals));
    try std.testing.expectEqual(live.unset, vals[@backingInt(live.Slot.keep_ahead)]);
}