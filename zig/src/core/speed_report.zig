/// kind 0: the switch cannot run ("off"); kind 1: it runs, in another form than asked ("instead: ...", e.g. a narrower kit variant).
const Note = struct { env: [64]u8 = undefined, env_len: u8 = 0, why: [200]u8 = undefined, why_len: u8 = 0, kind: u8 = 0 };
    const instead = std.mem.startsWith(u8, rest, "instead: ");
    const off = instead or std.mem.startsWith(u8, rest, "off") or std.mem.indexOf(u8, head, ": off") != null;
    n.kind = if (instead) 1 else 0;
    // the reason without the leading "off: " / "instead: "
    if (instead) why = why[9..];
    return notedKind(env, 0);
}

/// What the last "instead: " line of `env` said (it runs, in another form than asked), or null.
pub fn notedInstead(env: []const u8) ?[]const u8 {
    return notedKind(env, 1);
}

fn notedKind(env: []const u8, kind: u8) ?[]const u8 {
        if (n.kind == kind and std.mem.eql(u8, n.env[0..n.env_len], env)) return n.why[0..n.why_len];
            var also = false;
            for (d.also_on) |o| also = also or std.mem.eql(u8, v, o);
            if (std.mem.eql(u8, v, d.value) or also) {
                const how = if (speed_defaults.applied[i]) "on by default" else "set by the operator";
                if (notedInstead(d.env)) |ins|
                    put(r, d.what, d.env, true, false, d.cost, "{s}={s}, {s}; {s}", .{ d.env, v, how, ins })
                else
                    put(r, d.what, d.env, true, false, d.cost, "{s}={s}, {s}", .{ d.env, v, how });
test "the report: TF_DS_MERGE_D is listed as wanted and not running, with the kit's reason, when the loaded kit lacks the twins" {
    const keep_applied = pinApplied(false);
    defer unpin(keep_applied);
    reset();
    noteLine("TF_DS_MERGE_D: off: the loaded kit's extra set has no _sparse_attn_merge_d: kit a02466e or later");
    var r: Report = undefined;
    build(&r, .{}, fakeLookup);
    const m = find(&r, "TF_DS_MERGE_D").?;
    try std.testing.expect(!m.on and !m.asked); // a WARN: nobody turned it off
    try expectHas(m.reason, "has no _sparse_attn_merge_d: kit a02466e or later");
    try std.testing.expect(m.cost.len > 0);
    var w = std.Io.Writer.fixed(&buf);
    try r.writeBlock(&w);
    try expectHas(w.buffered(), "WARN   OFF decode attention merge in column blocks");
    // the same switch with its kit: on at its default; set to 0 by the operator: off and asked
    reset();
    build(&r, .{}, struct {
        fn f(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DS_MERGE_D")) "1" else null;
        }
    }.f);
    try std.testing.expect(find(&r, "TF_DS_MERGE_D").?.on);
    build(&r, .{}, struct {
        fn f(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DS_MERGE_D")) "0" else null;
        }
    }.f);
    const off = find(&r, "TF_DS_MERGE_D").?;
    try std.testing.expect(!off.on and off.asked);
}

/// `hay` holds `needle`, else both are printed (the test's failure then says what the report wrote).
fn expectHas(hay: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, hay, needle) != null) return;
    std.debug.print("\nexpected to contain: {s}\nactual:              {s}\n", .{ needle, hay });
    return error.TestUnexpectedResult;
}

/// speed_defaults.applied is process-wide state other tests set: a test that reads the "on by default" / "set by the operator" words
/// pins the MERGE_D entry itself (and restores it with the `defer` its caller writes: `const keep = pinApplied(x); defer unpin(keep);`).
fn mergeIndex() usize {
    for (speed_defaults.table, 0..) |d, i| if (std.mem.eql(u8, d.env, "TF_DS_MERGE_D")) return i;
    unreachable;
}
fn pinApplied(v: bool) bool {
    const keep = speed_defaults.applied[mergeIndex()];
    speed_defaults.applied[mergeIndex()] = v;
    return keep;
}
fn unpin(keep: bool) void {
    speed_defaults.applied[mergeIndex()] = keep;
}

fn mergeLookup(comptime v: ?[]const u8) *const fn ([]const u8) ?[]const u8 {
    return struct {
        fn f(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DS_MERGE_D")) v else null;
        }
    }.f;
}

test "the report: TF_DS_MERGE_D is ON at every width (1, 2, 3), ON with the width it runs when the kit lacks the asked one, OFF with the kit's reason when the kit has no width" {
    var r: Report = undefined;
    const keep_applied = pinApplied(false);
    defer unpin(keep_applied);
    // every value that runs it is ON (3 is the default, 1 = 128 columns, 2 = 64); 0 is the operator's choice
    reset();
    inline for (.{ "1", "2", "3" }) |v| {
        build(&r, .{}, mergeLookup(v));
        const f = find(&r, "TF_DS_MERGE_D").?;
        try std.testing.expect(f.on and !f.asked);
        try std.testing.expect(std.mem.indexOf(u8, f.reason, "not ") == null); // no "instead" text without the kit's line
    }
    build(&r, .{}, mergeLookup("0"));
    try std.testing.expect(!find(&r, "TF_DS_MERGE_D").?.on and find(&r, "TF_DS_MERGE_D").?.asked);
    // a kit with 128 (and 64) but not 16: the start logs "instead": still ON, and the reason says which width runs instead of which
    reset();
    noteLine("TF_DS_MERGE_D: instead: runs 128 columns a program, not 16: the loaded kit's extra set has no _sparse_attn_merge_d at 16 columns (it needs the kit regenerated with that width)");
    try std.testing.expect(noted("TF_DS_MERGE_D") == null); // not an off line
    build(&r, .{}, mergeLookup("3"));
    const inst = find(&r, "TF_DS_MERGE_D").?;
    try std.testing.expect(inst.on and !inst.asked);
    try expectHas(inst.reason, "runs 128 columns a program, not 16");
    var w = std.Io.Writer.fixed(&buf);
    try r.writeBlock(&w);
    // (writeBlock: "INFO   ON  {name}: {reason}", and the name is the table's `what`, its parenthesis included)
    try expectHas(w.buffered(), "INFO   ON  decode attention merge in column blocks (the kernel kit's extra set): TF_DS_MERGE_D=3, set by the operator; runs 128 columns a program, not 16");
    // the same line when the engine applied the default itself (applied[] is what tells the two apart)
    _ = pinApplied(true);
    build(&r, .{}, mergeLookup("3"));
    var w2 = std.Io.Writer.fixed(&buf);
    try r.writeBlock(&w2);
    try expectHas(w2.buffered(), "INFO   ON  decode attention merge in column blocks (the kernel kit's extra set): TF_DS_MERGE_D=3, on by default; runs 128 columns a program, not 16");
    _ = pinApplied(false);
    // a kit with no merge_d at any width (or none at all): OFF, unasked, with the kit's reason, whatever the value
    reset();
    noteLine("TF_DS_MERGE_D: off: the loaded kit's extra set has no _sparse_attn_merge_d at 128, 64 or 16 columns a program");
    build(&r, .{}, mergeLookup("3"));
    const none = find(&r, "TF_DS_MERGE_D").?;
    try std.testing.expect(!none.on and !none.asked);
    try expectHas(none.reason, "at 128, 64 or 16 columns a program");
    try std.testing.expect(notedInstead("TF_DS_MERGE_D") == null);
    // an "instead" line of an earlier start does not outlive reset()
    reset();
    build(&r, .{}, mergeLookup("3"));
    try std.testing.expect(find(&r, "TF_DS_MERGE_D").?.on);
}
