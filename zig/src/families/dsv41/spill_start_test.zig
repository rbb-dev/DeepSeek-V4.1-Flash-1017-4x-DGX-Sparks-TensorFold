    try std.testing.expectEqualStrings("packed=1 owned=0 instream=0 aio=1", c.engram_switches);
    try std.testing.expectEqualStrings("packed=1 owned=0 instream=0 aio=0", odd.engram_switches);

    var buf: [1024]u8 = undefined;
    var off = math;
    off.roci = false;
    off.roci_collect = false;
    const text = try start.mathText(&buf, off);
    try std.testing.expect(std.mem.indexOf(u8, text, "roci") == null);
    try std.testing.expect(std.mem.endsWith(u8, text, "mma_sf_exact=1"));
    var buf2: [1024]u8 = undefined;
    const on = try start.mathText(&buf2, math);
    try std.testing.expectEqualStrings(text, on[0..text.len]);
}