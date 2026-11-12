    try std.testing.expect(std.mem.indexOf(u8, src, "(if (waiting) pw.f else r.k.publish)") != null);
    try std.testing.expect(std.mem.indexOf(u8, sw, "Kernels.pick(r.k.stage_wait, r.k.stage_wait_x)") != null);

test "the exchange log layout the kernel writes is the layout rdma.zig's xlogDecode reads" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    const p = body(src, "dsv41_rdma_collect_peers_roci_x(") orelse return error.TestUnexpectedResult;
    try std.testing.expect(has(p, "e[0] = (q << 3) | (uint64_t)p;"));
    try std.testing.expect(has(p, "fault + 8 + ((q * 8 + (uint64_t)p) & 65535) * 4"));
    try std.testing.expect(has(p, "e[2] = t_enter;"));
    try std.testing.expect(has(p, "((q * 8 + (uint64_t)p) & 65535) * 4 + 3] = now_ns();"));
    try std.testing.expectEqual(@as(usize, 8192), @import("rdma.zig").xlog_records);
    // the receive window is 8,192 sequences x 8 peer slots: the kernel's mask is the record count less one, and the doorbell log sits behind it
    try std.testing.expectEqual(@as(usize, 65536), @import("rdma.zig").xlog_recv_records);
    try std.testing.expectEqual(@as(usize, 8192), @import("rdma.zig").xlog_recv_records / 8);
}

test "the expert-count log layout the kernel writes is the layout mx.zig's nactDecode reads" {
    const gpa = std.testing.allocator;
    const src = std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/cuda/dsv41/native.cu", gpa, .limited(1 << 25)) catch return error.SkipZigTest;
    defer gpa.free(src);
    const at = std.mem.indexOf(u8, src, "tf_dsn_nact_log_kernel(") orelse return error.TestUnexpectedResult;
    const k = src[at .. at + 1400];
    try std.testing.expect(has(k, "ring + 8 + (i & 4095) * 4"));
    try std.testing.expect(has(k, "r[0] = i + 1;"));
    try std.testing.expect(has(k, "r[1] = ((unsigned long long)rows << 32) | (unsigned)n;"));
    try std.testing.expect(has(k, "r[2] = ((unsigned long long)sh << 32) | (unsigned)owned;"));
    try std.testing.expectEqual(@as(usize, 4096), @import("mx.zig").nact_records);
}

test "the doorbell log the card-WAIT kernels write is the layout rdma.zig's xlogPubDecode reads" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    for ([_][]const u8{ "dsv41_rdma_stage_wait_x(", "dsv41_rdma_publish_wait_x(" }) |h| {
        const k = body(src, h) orelse return error.TestUnexpectedResult;
        try std.testing.expect(has(k, "xlog_pub[(q & 8191) * 2] = q;"));
        try std.testing.expect(has(k, "xlog_pub[(q & 8191) * 2 + 1] = now_ns();"));
        try std.testing.expect(has(k, "if (xlog_pub) probe = nullptr;"));
    }
}

test "with TF_DS_ROCI_XLOG off the exchange runs the instructions it ran before the log: the plain kernels hold no logging, the _x kernels are the plain ones plus their xlog lines" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    for ([_][]const u8{ "dsv41_rdma_stage_wait", "dsv41_rdma_publish_wait", "dsv41_rdma_collect_peers_roci" }) |name| {
        var hb: [96]u8 = undefined;
        const plain = body(src, std.fmt.bufPrint(&hb, "{s}(", .{name}) catch unreachable) orelse return error.TestUnexpectedResult;
        var xb: [96]u8 = undefined;
        const x = body(src, std.fmt.bufPrint(&xb, "{s}_x(", .{name}) catch unreachable) orelse return error.TestUnexpectedResult;
        try std.testing.expect(!has(plain, "xlog"));
        // the _x body without the lines that mention the log is the plain body (its name apart)
        var stripped: std.ArrayList(u8) = .empty;
        defer stripped.deinit(gpa);
        var it = std.mem.splitScalar(u8, x, '\n');
        while (it.next()) |line| {
            if (std.ascii.findIgnoreCase(line, "xlog") != null) continue;
            try stripped.appendSlice(gpa, line);
            try stripped.append(gpa, '\n');
        }
        var expect: std.ArrayList(u8) = .empty;
        defer expect.deinit(gpa);
        try expect.appendSlice(gpa, plain);
        try expect.append(gpa, '\n');
        const renamed = try std.mem.replaceOwned(u8, gpa, stripped.items, std.fmt.bufPrint(&xb, "{s}_x(", .{name}) catch unreachable, std.fmt.bufPrint(&hb, "{s}(", .{name}) catch unreachable);
        defer gpa.free(renamed);
        try std.testing.expectEqualStrings(expect.items, renamed);
    }
}
