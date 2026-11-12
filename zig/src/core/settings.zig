    /// Never shown, listed or set through the admin surface: it answers a name that does not exist (a key file's path, the
    /// debug rows, the options that only a test binary reads).
    hidden: bool,

/// A name that no longer exists. Naming it, in the environment or in engine.cfg, ends the start by name: an old
/// configuration is caught, never ignored.
pub const Retired = struct {
    env: []const u8,
    /// `.set`: any value refuses. `.on`: a value other than 0 refuses (0 is the one state that remains).
    when: When,
    /// What replaced it, in the words after "no longer exists: ".
    why: []const u8,

    pub const When = enum { set, on };
};
pub const retired = table.retired;
/// The retired name `env`, or null.
pub fn retiredByKey(key: []const u8) ?*const Retired {
    for (&retired) |*r| if (std.mem.eql(u8, r.env, key)) return r;
    return null;
}

/// The refusal for retired name `r` set to `value`, in `buf`; null when the value is the one that remains (`.on` and 0).
pub fn retiredRefusal(r: *const Retired, value: []const u8, buf: []u8) ?[]const u8 {
    if (r.when == .on and std.mem.eql(u8, value, "0")) return null;
    return say(buf, "{s} no longer exists: {s}", .{ r.env, r.why });
}

/// The refusal for the first retired name `lookup` (anything with `get(name) ?[]const u8`: the process environment in the
/// engine) finds set, in `buf`; null when none is.
pub fn firstRetired(lookup: anytype, buf: []u8) ?[]const u8 {
    for (&retired) |*r| if (lookup.get(r.env)) |v| if (retiredRefusal(r, v, buf)) |why| return why;
    return null;
}

        if (retiredByKey(l.key)) |r| if (!std.mem.eql(u8, l.value, "unset")) if (retiredRefusal(r, l.value, &buf)) |why| return fail(gpa, problem, "{s}: line {d}: {s}", .{ path, l.n, why });