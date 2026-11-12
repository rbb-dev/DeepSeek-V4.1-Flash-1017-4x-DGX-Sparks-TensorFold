    return getWith(srv, path, "");
}

/// GET `path` with `headers` (each line ending CRLF), through the key gate and the routes, as a connection runs it.
fn getWith(srv: *server.Server, path: []const u8, headers: []const u8) !Reply {
    var head_buf: [256]u8 = undefined;
    const head = try std.fmt.bufPrint(&head_buf, "GET {s} HTTP/1.1\r\nHost: t\r\n{s}\r\n", .{ path, headers });
    if (srv.authenticate(&conn, arena.allocator())) routes.dispatch(srv, &conn, arena.allocator());

test "keys on: /health gives a key holder its whole answer; without a valid key it says ok and nothing more" {
    var problem: []const u8 = "";
    var keys_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer keys_arena.deinit();
    var keyed = try auth.Store.init(keys_arena.allocator(), &.{"k1"}, "", null, false, &problem);
    var text: NoText = .{};
    var e: Engine = .{};
    const srv = try testServer(&e, &text, &keyed);
    defer srv.deinit();
    const holder = try getWith(srv, "/health", "Authorization: Bearer k1\r\n");
    defer std.testing.allocator.free(holder.body);
    try std.testing.expectEqual(@as(u16, 200), holder.status);
    try std.testing.expect(std.mem.indexOf(u8, holder.body, "\"model\": \"m\"") != null);
    for ([_][]const u8{ "", "Authorization: Bearer nope\r\n", "x-api-key: nope\r\n" }) |h| {
        const r = try getWith(srv, "/health", h);
        defer std.testing.allocator.free(r.body);
        try std.testing.expectEqual(@as(u16, 200), r.status);
        try std.testing.expectEqualStrings("{\"status\": \"ok\"}", r.body);
    }
}

test "a GET's body is drained: the next request on a kept connection is read as itself" {
    var text: NoText = .{};
    var e: Engine = .{};
    const srv = try testServer(&e, &text, null);
    defer srv.deinit();
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    // the body is a request line of its own: read as the next request, it would be served as one
    const two = "GET /health HTTP/1.1\r\nHost: t\r\nContent-Length: 9\r\n\r\nGET /nope" ++
        "GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n";
    if (std.c.write(pair[0], two.ptr, two.len) != @as(isize, two.len)) return error.Closed;
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqual(.ready, try conn.readRequest(arena.allocator(), false));
    routes.dispatch(srv, &conn, arena.allocator());
    try std.testing.expectEqual(.ready, try conn.readRequest(arena.allocator(), false));
    try std.testing.expectEqualStrings("GET", conn.method);
    try std.testing.expectEqualStrings("/v1/models", conn.path);
}
