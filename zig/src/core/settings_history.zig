    changes: []const struct { id: u32 = 0, to: ?[]const u8 = null, persisted: bool = true } = &.{},
            if (c.id == 0 or c.id > std.math.maxInt(u16) or !c.persisted) continue;
/// One change of a write row: what it was, what it is, whether engine.cfg holds it, and its state.

/// A row the admin API writes for a change it accepted.
pub const WriteRow = struct {
    at_ms: i64,
    label: []const u8,
    scope: []const u8,
    peer: []const u8,
    request_id: []const u8,
    endpoint: []const u8,
    apply: []const u8,
    outcome: []const u8,
    reason: []const u8,
    release: []const u8,
    /// The hash of the engine.cfg this write left (null: it did not write the file).
    cfg_sha256: ?[]const u8,
    changes: []const WriteChange,
    reverts_version: ?u64 = null,
};

/// Appends a write row at the next version and raises the version; the caller holds the writers' lock. The row reaches the file
/// (synced) before this returns; a failure leaves the version where it was.
pub fn appendWrite(a: std.mem.Allocator, io: std.Io, p: []const u8, r: WriteRow) !struct { version: u64, handed: bool } {
    _ = io;
    const v = version.load(.acquire) + 1;
    var row: std.ArrayList(u8) = .empty;
    try row.print(a, "{{\"version\":{d},\"at_ms\":{d},\"label\":", .{ v, r.at_ms });
    try quote(a, &row, r.label);
    try row.appendSlice(a, ",\"scope\":");
    try quote(a, &row, r.scope);
    try row.appendSlice(a, ",\"peer\":");
    try quote(a, &row, r.peer);
    try row.appendSlice(a, ",\"request_id\":");
    try quote(a, &row, r.request_id);
    try row.appendSlice(a, ",\"endpoint\":");
    try quote(a, &row, r.endpoint);
    try row.appendSlice(a, ",\"apply\":");
    try quote(a, &row, r.apply);
    try row.appendSlice(a, ",\"outcome\":");
    try quote(a, &row, r.outcome);
    try row.appendSlice(a, ",\"reason\":");
    try quote(a, &row, r.reason);
    try row.print(a, ",\"schema_version\":{d},\"release\":", .{settings.schema_version});
    try quote(a, &row, r.release);
    if (r.cfg_sha256) |h| try row.print(a, ",\"cfg_sha256\":\"{s}\"", .{h});
    if (r.reverts_version) |rv| try row.print(a, ",\"reverts_version\":{d}", .{rv});
    try row.appendSlice(a, ",\"changes\":[");
    var n: usize = 0;
    for (r.changes) |c| {
        if (n > 0) try row.append(a, ',');
        n += 1;
        const o = settings.byId(c.id);
        try row.print(a, "{{\"id\":{d},\"name\":", .{c.id});
        try quote(a, &row, if (o) |x| x.name else "");
        try row.appendSlice(a, ",\"from\":");
        try optQuote(a, &row, c.from);
        try row.appendSlice(a, ",\"to\":");
        try optQuote(a, &row, c.to);
        try row.print(a, ",\"persisted\":{s},\"source_before\":\"{s}\",\"state_after\":", .{ if (c.persisted) "true" else "false", if (c.from != null) "cfg" else "default" });
        try quote(a, &row, c.state_after);
        try row.print(a, ",\"out_changes\":{s}}}", .{if (o != null and o.?.out_changes) "true" else "false"});
    }
    try row.appendSlice(a, "],\"lease\":null,\"active_after_ms\":0}\n");
    const handed = try appendLine(a, p, row.items);
    version.store(v, .release);
    return .{ .version = v, .handed = handed };
}
