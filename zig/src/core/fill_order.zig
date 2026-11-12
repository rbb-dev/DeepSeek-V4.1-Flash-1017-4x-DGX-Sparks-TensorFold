//! The order prompt steps are served in while several prompts fill together.
//!
//! A step is one prompt pass (the smallest unit the host can reorder). The choice reads, per admitted prompt, the rows it
//! still has to compute after the prompt cache and the disk tier gave back what they held, whether it is background work,
//! and how many steps went to a smaller prompt while it waited:
//!
//!   1. a prompt that is due goes next: it has been passed over as often as its class allows (foreground only), or it has
//!      waited its class's maximum wait in seconds since its last step (every prompt, background included); the one
//!      that has waited longest first;
//!   2. else foreground before background;
//!   3. else the lowest class (the smallest uncached rows);
//!   4. within a class: `serial` the prompt with the fewest rows left (it keeps the steps until it ends), or `turns` the
//!      rotation.
//!
//! Beside decoding streams the host's share of wall time (lane_host_share.zig) comes first: while decode time is owed,
//! only a short prompt may step, and the order above picks among the short ones (`Slot.eligible`). A long prompt held
//! back that way keeps counting its waits and overtakes and is due the moment the rounds have repaid the debt; the share
//! is a rule about wall time and the order is a rule about who goes next, so the order never overrides the share.
//!
//! The classes are a table of uncached-row bounds, each with the number of times a smaller prompt may overtake it.
//! Without a table there is one class that every other prompt's step counts against, bounded at `default_overtakes`.
//! Both bounds apply at every tier: the step count bounds overtaking at light load, the seconds bound whatever the load
//! (a stream of small prompts can pass over a prompt for as long as it lasts). Background prompts are bounded by seconds
//! only: they take a step ahead of their time only while no foreground prompt fills.
const std = @import("std");

/// How the prompts of one class take steps.
pub const Order = enum { serial, turns };

/// Times another prompt's step may pass over a prompt when no table is given.
pub const default_overtakes: u32 = 8;

/// The longest a prompt of any class waits for its next step when no setting says otherwise. A step is 0.4-0.7 s, so 30 s
/// is some fifty to seventy steps: past what the step bound allows at any load where prompts compete, short of the
/// minutes a client waits before it gives up.
pub const default_max_wait_s: u32 = 30;

/// A class: prompts whose uncached rows are at most `rows` (and above the class before it). `id` ranks classes (lower
/// is served first). `overtakes`: the times a step may go to a prompt of a lower class before this one is due (null:
/// without limit: the seconds bound still applies). `max_wait_s`: this class's longest wait for a step (null: the policy's).
pub const Class = struct { rows: u64, id: u8, overtakes: ?u32, max_wait_s: ?u32 = null };

pub const max_classes = 8;

pub const Policy = struct {
    /// false: the rotation, one step each in turn, as a host without a fill order runs.
    on: bool = false,
    order: Order = .serial,
    /// true (no table): every step that goes elsewhere counts as an overtake. false (a table): only a step that goes to a
    /// prompt of a lower class.
    count_all: bool = true,
    len: u8 = 1,
    /// The longest any prompt waits for its next step (TF_DS_PROMPT_MAX_WAIT_S), unless its class says otherwise.
    max_wait_s: u32 = default_max_wait_s,
    classes: [max_classes]Class = @splat(.{ .rows = std.math.maxInt(u64), .id = 1, .overtakes = default_overtakes }),

    /// The class index of a prompt with `rows` left to compute.
    pub fn classOf(p: *const Policy, rows: u64) usize {
        for (p.classes[0..p.len], 0..) |c, i| if (rows <= c.rows) return i;
        return p.len - 1;
    }

    /// The longest wait for a step, in nanoseconds, of a prompt with `rows` left.
    pub fn maxWaitNs(p: *const Policy, rows: u64) i64 {
        return @as(i64, p.classes[p.classOf(rows)].max_wait_s orelse p.max_wait_s) * std.time.ns_per_s;
    }

    fn idOf(p: *const Policy, rows: u64) u8 {
        return p.classes[p.classOf(rows)].id;
    }
};

/// One admitted prompt as the order reads it.
pub const Slot = struct {
    prefilled: bool,
    /// false: this prompt may not take the next step whatever its rank (the host's share owes decoders time and the
    /// prompt is not short); the order picks among the others.
    eligible: bool = true,
    background: bool = false,
    /// Rows still to compute: the prompt less what the cache restored and what the pass has passed.
    rows: u64,
    /// Steps that went to a smaller prompt (or to any prompt, without a table) since this one last took a step.
    overtaken: u32 = 0,
    /// Nanoseconds since this prompt's last step began (since admission when it had none).
    waited_ns: i64 = 0,

    /// Whether this prompt cannot take the next step: it has prefilled, or the host holds it back.
    fn closed(s: Slot) bool {
        return s.prefilled or !s.eligible;
    }
};

/// Whether a prompt is due: waited its class's maximum wait, or (foreground) been overtaken as often as its class allows.
fn isDue(p: *const Policy, s: Slot) bool {
    if (s.waited_ns >= p.maxWaitNs(s.rows)) return true;
    if (s.background) return false;
    const limit = p.classes[p.classOf(s.rows)].overtakes orelse return false;
    return s.overtaken > 0 and s.overtaken >= limit;
}

/// The admitted index whose step is next, or null when every prompt has prefilled. `rotate_from` is where the rotation
/// starts looking (the index after the last prompt served).
pub fn pick(p: *const Policy, slots: []const Slot, rotate_from: usize) ?usize {
    const n = slots.len;
    if (n == 0) return null;
    if (!p.on) return rotation(slots, rotate_from, null, p);
    // 1. the due prompt that has waited longest, a lower class first on a tie, the earlier admitted after that
    var due: ?usize = null;
    for (slots, 0..) |s, i| {
        if (s.closed() or !isDue(p, s)) continue;
        const better = due == null or s.waited_ns > slots[due.?].waited_ns or
            (s.waited_ns == slots[due.?].waited_ns and p.idOf(s.rows) < p.idOf(slots[due.?].rows));
        if (better) due = i;
    }
    if (due) |d| return d;
    // 2. foreground before background
    var foreground = false;
    for (slots) |s| if (!s.closed() and !s.background) {
        foreground = true;
        break;
    };
    // 3. the lowest class among them
    var lowest: ?u8 = null;
    for (slots) |s| {
        if (s.closed() or s.background == foreground) continue;
        const id = p.idOf(s.rows);
        if (lowest == null or id < lowest.?) lowest = id;
    }
    const id = lowest orelse return null;
    // 4. within the class
    switch (p.order) {
        .turns => return rotation(slots, rotate_from, .{ .id = id, .foreground = foreground }, p),
        .serial => {
            var best: ?usize = null;
            for (slots, 0..) |s, i| {
                if (s.closed() or s.background == foreground or p.idOf(s.rows) != id) continue;
                if (best == null or s.rows < slots[best.?].rows) best = i;
            }
            return best;
        },
    }
}

const Within = struct { id: u8, foreground: bool };

/// The first prompt from `from` (cyclically) that may take a step (not prefilled, eligible) and, with `within`, is in that class and tier.
fn rotation(slots: []const Slot, from: usize, within: ?Within, p: *const Policy) ?usize {
    const n = slots.len;
    for (0..n) |off| {
        const i = (from + off) % n;
        const s = slots[i];
        if (s.closed()) continue;
        if (within) |w| if (s.background == w.foreground or p.idOf(s.rows) != w.id) continue;
        return i;
    }
    return null;
}

/// Whether a step served to `picked` counts as an overtake of the waiting prompt `other`.
pub fn overtakes(p: *const Policy, picked: Slot, other: Slot) bool {
    if (!p.on or other.prefilled) return false;
    if (p.count_all) return true;
    return p.idOf(picked.rows) < p.idOf(other.rows);
}

// -- the settings ------------------------------------------------------------------------------------------------

/// What the settings said: the policy to run, and for each setting whether its text was understood (an unreadable text
/// runs the default, and the caller says so once, with `qos_why` for a table).
pub const Settings = struct {
    policy: Policy = .{},
    order_ok: bool = true,
    qos_ok: bool = true,
    wait_ok: bool = true,
    /// Why the table was refused (a static text), or empty.
    qos_why: []const u8 = "",
};

/// TF_DS_PROMPT_ORDER (`serial` or `turns`), TF_DS_PROMPT_QOS (`size:class:overtakes[:max_wait_s],...`, see `parseTable`)
/// and TF_DS_PROMPT_MAX_WAIT_S (whole seconds, at least 1). All unset: the rotation. Any of the first two set: the order is on.
pub fn parse(order_text: ?[]const u8, qos_text: ?[]const u8, wait_text: ?[]const u8) Settings {
    var s: Settings = .{};
    if (wait_text) |t| {
        if (std.fmt.parseInt(u32, std.mem.trim(u8, t, " \t"), 10)) |n| {
            if (n >= 1 and n <= max_wait_limit_s) s.policy.max_wait_s = n else s.wait_ok = false;
        } else |_| s.wait_ok = false;
    }
    if (order_text) |t| {
        const v = std.mem.trim(u8, t, " \t");
        if (std.mem.eql(u8, v, "serial")) {
            s.policy.on = true;
            s.policy.order = .serial;
        } else if (std.mem.eql(u8, v, "turns")) {
            s.policy.on = true;
            s.policy.order = .turns;
        } else s.order_ok = false;
    }
    if (qos_text) |t| {
        switch (parseTable(t)) {
            .ok => |table| {
                s.policy.on = true;
                s.policy.count_all = false;
                s.policy.len = table.len;
                s.policy.classes = table.classes;
            },
            .bad => |why| {
                s.qos_ok = false;
                s.qos_why = why;
            },
        }
    }
    return s;
}

/// The largest maximum wait a setting may name (a day).
pub const max_wait_limit_s: u32 = 86_400;

/// A row count: digits, then an optional `k` (x 1,024) or `m` (x 1,048,576), or `*` for no bound.
fn parseRows(text: []const u8) ?u64 {
    const t = std.mem.trim(u8, text, " \t");
    if (std.mem.eql(u8, t, "*")) return std.math.maxInt(u64);
    if (t.len == 0) return null;
    const unit: u64 = switch (t[t.len - 1]) {
        'k', 'K' => 1024,
        'm', 'M' => 1024 * 1024,
        else => 1,
    };
    const digits = if (unit == 1) t else t[0 .. t.len - 1];
    const n = std.fmt.parseInt(u64, digits, 10) catch return null;
    return std.math.mul(u64, n, unit) catch null;
}

const Table = struct { len: u8, classes: [max_classes]Class };
const TableResult = union(enum) { ok: Table, bad: []const u8 };

/// `size:class:overtakes[:max_wait_s]` entries separated by commas, smallest first: a prompt with at most `size` uncached
/// rows is in `class` (lower is served first; equal numbers merge) and may be overtaken `overtakes` times by prompts of a
/// lower class (`inf` switches the step bound off; the seconds bound still applies). `max_wait_s` (whole seconds, at least
/// 1) overrides TF_DS_PROMPT_MAX_WAIT_S for the class. The last entry's size is `*`, or its bound is taken as `*`. Sizes
/// strictly ascend, class numbers do not fall. A table that is none says why.
fn parseTable(text: []const u8) TableResult {
    var t: Table = .{ .len = 0, .classes = undefined };
    var it = std.mem.splitScalar(u8, text, ',');
    var last_rows: u64 = 0;
    var last_id: u8 = 0;
    while (it.next()) |entry| {
        if (t.len == max_classes) return .{ .bad = "more than 8 classes" };
        var f = std.mem.splitScalar(u8, entry, ':');
        const size = parseRows(f.next() orelse return .{ .bad = "an empty entry" }) orelse return .{ .bad = "a size is not a row count (digits, k, m or *)" };
        const id_text = std.mem.trim(u8, f.next() orelse return .{ .bad = "an entry needs size:class:overtakes" }, " \t");
        const over_text = std.mem.trim(u8, f.next() orelse return .{ .bad = "an entry needs size:class:overtakes" }, " \t");
        var wait: ?u32 = null;
        if (f.next()) |w| {
            const n = std.fmt.parseInt(u32, std.mem.trim(u8, w, " \t"), 10) catch return .{ .bad = "max_wait_s is not whole seconds" };
            if (n < 1 or n > max_wait_limit_s) return .{ .bad = "max_wait_s is outside 1 to 86400" };
            wait = n;
        }
        if (f.next() != null) return .{ .bad = "an entry has more than four fields" };
        const id = std.fmt.parseInt(u8, id_text, 10) catch return .{ .bad = "a class is not a number" };
        const over: ?u32 = if (std.mem.eql(u8, over_text, "inf")) null else (std.fmt.parseInt(u32, over_text, 10) catch return .{ .bad = "overtakes is not a number or inf" });
        if (size == 0) return .{ .bad = "a size is zero" };
        if (t.len > 0 and size <= last_rows) return .{ .bad = "sizes do not ascend" };
        if (id < last_id) return .{ .bad = "class numbers fall" };
        t.classes[t.len] = .{ .rows = size, .id = id, .overtakes = over, .max_wait_s = wait };
        t.len += 1;
        last_rows = size;
        last_id = id;
    }
    if (t.len == 0) return .{ .bad = "no entries" };
    t.classes[t.len - 1].rows = std.math.maxInt(u64);
    return .{ .ok = t };
}

/// The start line for a policy that is on, naming what it will do.
pub fn describe(p: *const Policy, buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("prompt order: {s}", .{@tagName(p.order)}) catch return buf[0..0];
    if (p.count_all) {
        w.print(", one class, a prompt passed over by {d} steps takes the next", .{p.classes[0].overtakes.?}) catch return w.buffered();
    } else {
        w.writeAll(", classes by uncached rows:") catch return w.buffered();
        var from: u64 = 0;
        for (p.classes[0..p.len]) |c| {
            if (c.rows == std.math.maxInt(u64)) {
                w.print(" >{d}:class {d}", .{ from, c.id }) catch return w.buffered();
            } else {
                w.print(" <={d}:class {d}", .{ c.rows, c.id }) catch return w.buffered();
            }
            if (c.overtakes) |o| w.print(" (overtaken {d}x", .{o}) catch return w.buffered() else w.writeAll(" (no step bound") catch return w.buffered();
            w.print(", waits {d} s)", .{c.max_wait_s orelse p.max_wait_s}) catch return w.buffered();
            from = c.rows;
        }
    }
    w.print("; any prompt that waited {d} s for a step takes the next; background prompts take a step ahead of their time only while no foreground prompt fills; while TENSORFOLD_PREFILL_SHARE owes decoders time only a short prompt steps, the order picking among those", .{p.max_wait_s}) catch {};
    return w.buffered();
}

// -- tests -------------------------------------------------------------------------------------------------------

const tt = std.testing;

fn slot(rows: u64) Slot {
    return .{ .prefilled = false, .rows = rows };
}

test "off: the rotation from where it stopped, skipping prompts that have prefilled" {
    const p: Policy = .{};
    const s = [_]Slot{ slot(10), slot(5), .{ .prefilled = true, .rows = 0 } };
    try tt.expectEqual(@as(?usize, 0), pick(&p, &s, 0));
    try tt.expectEqual(@as(?usize, 1), pick(&p, &s, 1));
    try tt.expectEqual(@as(?usize, 0), pick(&p, &s, 2));
    const done = [_]Slot{.{ .prefilled = true, .rows = 0 }};
    try tt.expectEqual(@as(?usize, null), pick(&p, &done, 0));
    try tt.expectEqual(@as(?usize, null), pick(&p, &.{}, 0));
    // background is not a rank in the rotation
    const bg = [_]Slot{ .{ .prefilled = false, .background = true, .rows = 1 }, slot(9) };
    try tt.expectEqual(@as(?usize, 0), pick(&p, &bg, 0));
}

test "no table: the fewest rows left goes first, and a prompt passed over eight times takes the next step" {
    const p = parse("serial", null, null).policy;
    try tt.expect(p.on and p.count_all);
    var s = [_]Slot{ slot(100_000), slot(300), slot(5_000) };
    try tt.expectEqual(@as(?usize, 1), pick(&p, &s, 0));
    s[0].overtaken = 7;
    try tt.expectEqual(@as(?usize, 1), pick(&p, &s, 0));
    s[0].overtaken = 8;
    try tt.expectEqual(@as(?usize, 0), pick(&p, &s, 0));
    // two due: the one that has waited longer
    s[2].overtaken = 10;
    s[2].waited_ns = 5;
    try tt.expectEqual(@as(?usize, 2), pick(&p, &s, 0));
    s[0].waited_ns = 9;
    try tt.expectEqual(@as(?usize, 0), pick(&p, &s, 0));
}

test "no table: every step that goes elsewhere is an overtake" {
    const p = parse("serial", null, null).policy;
    try tt.expect(overtakes(&p, slot(10), slot(10)));
    try tt.expect(!overtakes(&p, slot(10), .{ .prefilled = true, .rows = 0 }));
    const off: Policy = .{};
    try tt.expect(!overtakes(&off, slot(10), slot(10)));
}

test "foreground before background whatever the sizes, and background is never due" {
    const p = parse("serial", null, null).policy;
    var s = [_]Slot{ .{ .prefilled = false, .background = true, .rows = 10, .overtaken = 1000 }, slot(900_000) };
    try tt.expectEqual(@as(?usize, 1), pick(&p, &s, 0));
    s[1].prefilled = true;
    try tt.expectEqual(@as(?usize, 0), pick(&p, &s, 0));
    // among background prompts the same order: the fewest rows
    const bg = [_]Slot{ .{ .prefilled = false, .background = true, .rows = 500 }, .{ .prefilled = false, .background = true, .rows = 20 } };
    try tt.expectEqual(@as(?usize, 1), pick(&p, &bg, 0));
}

test "a table: classes by uncached rows, each with its own bound" {
    const r = parse("serial", "4k:1:0,16k:2:2,64k:3:4,256k:4:8,*:5:64", null);
    try tt.expect(r.order_ok and r.qos_ok);
    const p = r.policy;
    try tt.expect(p.on and !p.count_all and p.len == 5);
    try tt.expectEqual(@as(usize, 0), p.classOf(4096));
    try tt.expectEqual(@as(usize, 1), p.classOf(4097));
    try tt.expectEqual(@as(usize, 4), p.classOf(10_000_000));
    // the small one first, the 1M-row prompt last
    var s = [_]Slot{ slot(1_000_000), slot(3_000), slot(200_000) };
    try tt.expectEqual(@as(?usize, 1), pick(&p, &s, 0));
    // class 4 (200K rows) is due after eight overtakes; the 1M class after 64
    s[0].overtaken = 63;
    s[2].overtaken = 7;
    try tt.expectEqual(@as(?usize, 1), pick(&p, &s, 0));
    s[2].overtaken = 8;
    try tt.expectEqual(@as(?usize, 2), pick(&p, &s, 0));
    // only a step to a lower class counts
    try tt.expect(overtakes(&p, slot(3_000), slot(200_000)));
    try tt.expect(!overtakes(&p, slot(200_000), slot(3_000)));
    try tt.expect(!overtakes(&p, slot(150_000), slot(200_000))); // the same class
}

test "a prompt promoted by its shrinking rows is judged by its new class" {
    const p = parse("serial", "4k:1:0,*:2:inf", null).policy;
    var s = [_]Slot{ slot(50_000), slot(3_000) };
    s[0].overtaken = 5;
    try tt.expectEqual(@as(?usize, 1), pick(&p, &s, 0)); // class 2 never due
    s[0].rows = 2_000; // class 1 now: its bound is 0, and it was overtaken
    s[1].rows = 3_000;
    try tt.expectEqual(@as(?usize, 0), pick(&p, &s, 0));
}

test "a prompt the host holds back is no candidate, due or not; the order picks among the rest" {
    const p = parse("serial", "4k:1:0,*:2:inf", "30").policy;
    const s30 = 30 * std.time.ns_per_s;
    // the long prompt is due by seconds and by steps, but not eligible: the short prompts go in class order
    var s = [_]Slot{
        .{ .prefilled = false, .eligible = false, .rows = 500_000, .overtaken = 99, .waited_ns = s30 },
        slot(400),
        slot(200),
    };
    try tt.expectEqual(@as(?usize, 2), pick(&p, &s, 0));
    s[2].prefilled = true;
    try tt.expectEqual(@as(?usize, 1), pick(&p, &s, 0));
    // none eligible: nothing steps, and the rotation (order off) holds back the same way
    s[1].eligible = false;
    try tt.expectEqual(@as(?usize, null), pick(&p, &s, 0));
    const off: Policy = .{};
    try tt.expectEqual(@as(?usize, null), pick(&off, &s, 0));
    s[1].eligible = true;
    try tt.expectEqual(@as(?usize, 1), pick(&off, &s, 0));
    // eligible again: the long prompt is due at once
    s[0].eligible = true;
    s[1].prefilled = true;
    try tt.expectEqual(@as(?usize, 0), pick(&p, &s, 0));
}

test "within a class: serial keeps the fewest rows, turns rotates" {
    const serial = parse("serial", "*:1:inf", null).policy;
    const turns = parse("turns", "*:1:inf", null).policy;
    const s = [_]Slot{ slot(900), slot(100), slot(500) };
    try tt.expectEqual(@as(?usize, 1), pick(&serial, &s, 0));
    try tt.expectEqual(@as(?usize, 1), pick(&serial, &s, 2));
    try tt.expectEqual(@as(?usize, 0), pick(&turns, &s, 0));
    try tt.expectEqual(@as(?usize, 1), pick(&turns, &s, 1));
    try tt.expectEqual(@as(?usize, 2), pick(&turns, &s, 2));
    // the rotation stays inside the lowest class
    const mixed = [_]Slot{ slot(900_000), slot(100), slot(200) };
    const tab = parse("turns", "1k:1:inf,*:2:inf", null).policy;
    try tt.expectEqual(@as(?usize, 1), pick(&tab, &mixed, 0));
    try tt.expectEqual(@as(?usize, 2), pick(&tab, &mixed, 2));
    try tt.expectEqual(@as(?usize, 1), pick(&tab, &mixed, 3));
}

test "the settings: unset is the rotation, readable text is on, unreadable text is said and ignored" {
    try tt.expect(!parse(null, null, null).policy.on);
    try tt.expect(parse(null, "*:1:3", null).policy.on);
    try tt.expectEqual(Order.turns, parse("turns", null, null).policy.order);
    const bad = parse("fifo", null, null);
    try tt.expect(!bad.order_ok and !bad.policy.on);
    try tt.expect(!parse(null, null, "0").wait_ok and !parse(null, null, "x").wait_ok and !parse(null, null, "86401").wait_ok);
    try tt.expectEqual(@as(u32, 45), parse(null, null, "45").policy.max_wait_s);
    try tt.expectEqual(@as(u32, 30), parse(null, null, null).policy.max_wait_s);
    for ([_][]const u8{ "", "4k", "4k:1", "4k:1:x", "4k:1:2:3:4", "0:1:1", "x:1:1", "8k:1:1,4k:2:1", "4k:2:1,8k:1:1", "1:1:1,2:1:1,3:1:1,4:1:1,5:1:1,6:1:1,7:1:1,8:1:1,9:1:1" }) |text| {
        const r = parse(null, text, null);
        try tt.expect(!r.qos_ok and !r.policy.on);
    }
    // sizes with units, equal class numbers, the last bound widened to everything
    const r = parse(null, "1k:1:0,2k:1:0,1m:2:5", null);
    try tt.expect(r.qos_ok and r.policy.len == 3);
    try tt.expectEqual(@as(u64, 1024), r.policy.classes[0].rows);
    try tt.expectEqual(@as(u64, 2048), r.policy.classes[1].rows);
    try tt.expectEqual(std.math.maxInt(u64), r.policy.classes[2].rows);
    try tt.expectEqual(@as(?u32, 5), r.policy.classes[2].overtakes);
}

test "describe names the order and the classes" {
    var buf: [512]u8 = undefined;
    const one = describe(&parse("serial", null, null).policy, &buf);
    try tt.expect(std.mem.indexOf(u8, one, "serial") != null and std.mem.indexOf(u8, one, "8 steps") != null);
    const many = describe(&parse("turns", "4k:1:0,*:2:inf", null).policy, &buf);
    try tt.expect(std.mem.indexOf(u8, many, "<=4096:class 1 (overtaken 0x, waits 30 s)") != null);
    try tt.expect(std.mem.indexOf(u8, many, ">4096:class 2 (no step bound, waits 30 s)") != null);
    try tt.expect(std.mem.indexOf(u8, many, "waited 30 s") != null);
}

test "the 3- and 4-field table forms parse, and a bad field is refused with its reason" {
    const r = parse(null, "4k:1:0,16k:2:2:5, *:3:inf:120", "20");
    try tt.expect(r.qos_ok and r.policy.len == 3 and r.policy.max_wait_s == 20);
    try tt.expectEqual(@as(?u32, null), r.policy.classes[0].max_wait_s);
    try tt.expectEqual(@as(?u32, 5), r.policy.classes[1].max_wait_s);
    try tt.expectEqual(@as(?u32, 120), r.policy.classes[2].max_wait_s);
    try tt.expectEqual(@as(?u32, null), r.policy.classes[2].overtakes);
    // the class's own wait, else the global one
    try tt.expectEqual(@as(i64, 20) * std.time.ns_per_s, r.policy.maxWaitNs(100));
    try tt.expectEqual(@as(i64, 5) * std.time.ns_per_s, r.policy.maxWaitNs(10_000));
    try tt.expectEqual(@as(i64, 120) * std.time.ns_per_s, r.policy.maxWaitNs(1_000_000));
    const cases = [_]struct { text: []const u8, why: []const u8 }{
        .{ .text = "4k:1:0:0,*:2:1", .why = "max_wait_s is outside" },
        .{ .text = "4k:1:0:x,*:2:1", .why = "max_wait_s is not whole seconds" },
        .{ .text = "4k:1:0:inf,*:2:1", .why = "max_wait_s is not whole seconds" },
        .{ .text = "4k:1:0:5:6", .why = "more than four fields" },
        .{ .text = "4k:1", .why = "needs size:class:overtakes" },
        .{ .text = "8k:1:1,4k:2:1", .why = "sizes do not ascend" },
        .{ .text = "4k:2:1,8k:1:1", .why = "class numbers fall" },
        .{ .text = "4k:a:1", .why = "class is not a number" },
        .{ .text = "4k:1:y", .why = "overtakes is not a number" },
    };
    for (cases) |c| {
        const b = parse(null, c.text, null);
        try tt.expect(!b.qos_ok and !b.policy.on);
        try tt.expect(std.mem.indexOf(u8, b.qos_why, c.why) != null);
    }
}

test "the seconds rule: a prompt that waited its maximum is due whatever the step counts say" {
    const p = parse("serial", "4k:1:0,*:2:inf", "30").policy;
    const s30 = 30 * std.time.ns_per_s;
    var s = [_]Slot{ .{ .prefilled = false, .rows = 500_000, .waited_ns = s30 - 1 }, slot(3_000) };
    try tt.expectEqual(@as(?usize, 1), pick(&p, &s, 0)); // under the wait, no step bound: the small one
    s[0].waited_ns = s30;
    try tt.expectEqual(@as(?usize, 0), pick(&p, &s, 0));
    // a background prompt ages by seconds too, and ahead of foreground ones
    var bg = [_]Slot{ .{ .prefilled = false, .background = true, .rows = 900, .waited_ns = s30 - 1, .overtaken = 99 }, slot(50) };
    try tt.expectEqual(@as(?usize, 1), pick(&p, &bg, 0));
    bg[0].waited_ns = s30;
    try tt.expectEqual(@as(?usize, 0), pick(&p, &bg, 0));
    // the earliest-waiting due prompt first, whatever its class; a tie goes to the lower class
    var due = [_]Slot{
        .{ .prefilled = false, .rows = 500_000, .waited_ns = s30 + 5 },
        .{ .prefilled = false, .background = true, .rows = 10_000, .waited_ns = s30 + 9 },
        .{ .prefilled = false, .rows = 100, .waited_ns = s30 + 9 },
    };
    try tt.expectEqual(@as(?usize, 2), pick(&p, &due, 0));
    due[2].waited_ns = s30;
    try tt.expectEqual(@as(?usize, 1), pick(&p, &due, 0));
    // a class's own wait overrides the global one
    const q = parse("serial", "4k:1:0,*:2:inf:5", "30").policy;
    const big = [_]Slot{ .{ .prefilled = false, .rows = 500_000, .waited_ns = 5 * std.time.ns_per_s }, slot(3_000) };
    try tt.expectEqual(@as(?usize, 0), pick(&q, &big, 0));
}
