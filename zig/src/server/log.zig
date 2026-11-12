//! Server output: ``[tensorfold]`` lines on stdout, and the live status line they clear first.
const std = @import("std");
const Conn = @import("http_conn.zig").Conn;
const afterChars = @import("reply_text.zig").afterChars;

var global_io: ?std.Io = null;
var mutex: std.Io.Mutex = .init;
var shown = false; // the live line is on screen
var line_start = true; // the newest write ended its line, so a redraw cannot split it
var quiet = false;

pub fn init(io_: std.Io, silent: bool) void {
    global_io = io_;
    quiet = silent;
}

pub fn io() std.Io {
    return global_io orelse std.Io.Threaded.global_single_threaded.io();
}

const clear = "\r\x1b[2K";

fn writeOut(bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const rc = std.posix.system.write(1, bytes[off..].ptr, bytes.len - off);
        if (std.posix.errno(rc) != .SUCCESS or rc <= 0) return;
        off += @intCast(rc);
    }
}

/// Tests: where lines go (a test's stdout is the build runner's channel, so without it they go nowhere).
pub var capture: ?*std.Io.Writer = null;

/// One ``[tensorfold] ...`` line.
pub fn line(comptime fmt: []const u8, args: anytype) void {
    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.print("[tensorfold] " ++ fmt ++ "\n", args) catch {
        buf[buf.len - 1] = '\n';
        w.end = buf.len;
    };
    emit(w.buffered(), null);
}

/// Where a formatted line goes: the ring first (what the admin surface reads, whatever the console does), then the
/// console. `ring_text` is what the ring keeps when it must differ from the console's line (null: the same).
fn emit(text: []const u8, ring_text: ?[]const u8) void {
    ring.push(text: {
        const t = ring_text orelse text;
        const body = if (std.mem.startsWith(u8, t, "[tensorfold] ")) t["[tensorfold] ".len..] else t;
        break :text std.mem.trimEnd(u8, body, "\n");
    });
    if (@import("builtin").is_test) {
        if (capture) |w| w.writeAll(text) catch {};
        return;
    }
    if (quiet) return;
    raw(text);
}

/// The ring of the newest lines the admin surface's /logs reads: `capacity` lines, each cut at `max_line` bytes.
pub const ring = struct {
    pub const capacity = 2000;
    pub const max_line = 512;
    pub const Level = enum { info, warn, @"error" };
    pub const Slot = struct { seq: u64 = 0, at_ms: i64 = 0, level: Level = .info, len: u16 = 0, text: [max_line]u8 = undefined };

    var lock: std.Io.Mutex = .init;
    var slots: ?[]Slot = null;
    var next_seq: u64 = 1;

    /// Drops every line and starts the numbering again (tests).
    pub fn reset() void {
        const m = io();
        lock.lockUncancelable(m);
        defer lock.unlock(m);
        if (slots) |s| for (s) |*x| {
            x.* = .{};
        };
        next_seq = 1;
    }

    /// The level a line reads as: error words anywhere, warning and refusal lines at the start. A guess from the text,
    /// since the lines carry no level of their own.
    pub fn levelOf(text: []const u8) Level {
        if (std.ascii.findIgnoreCase(text, "error") != null or std.ascii.findIgnoreCase(text, "failed") != null) return .@"error";
        for ([_][]const u8{ "warning", "refused", "ended " }) |p| if (std.ascii.startsWithIgnoreCase(text, p)) return .warn;
        return .info;
    }

    pub fn push(text: []const u8) void {
        const m = io();
        const now_ms: i64 = @intCast(@divTrunc(std.Io.Clock.real.now(m).toNanoseconds(), std.time.ns_per_ms));
        lock.lockUncancelable(m);
        defer lock.unlock(m);
        if (slots == null) {
            slots = std.heap.page_allocator.alloc(Slot, capacity) catch return;
            for (slots.?) |*x| x.* = .{};
        }
        // a cut never splits a character
        var n = @min(text.len, max_line);
        while (n > 0 and n < text.len and (text[n] & 0xC0) == 0x80) n -= 1;
        const slot = &slots.?[@intCast((next_seq - 1) % capacity)];
        slot.* = .{ .seq = next_seq, .at_ms = now_ms, .level = levelOf(text[0..n]), .len = @intCast(n) };
        @memcpy(slot.text[0..n], text[0..n]);
        next_seq += 1;
    }

    pub const Page = struct { lines: []Slot, next_seq: u64, dropped: u64 };

    /// The lines after `since` (0: from the oldest kept), at most `limit`, those at `min` or above that hold `contains`,
    /// oldest first, copied into `a`. `dropped` counts lines after `since` that the ring has overwritten.
    pub fn read(a: std.mem.Allocator, since: u64, limit: usize, min: Level, contains: []const u8) !Page {
        const m = io();
        lock.lockUncancelable(m);
        defer lock.unlock(m);
        const s = slots orelse return .{ .lines = &.{}, .next_seq = next_seq, .dropped = 0 };
        const newest = next_seq - 1;
        const oldest = if (newest > capacity) newest - capacity + 1 else 1;
        const first = @max(since + 1, oldest);
        var out: std.ArrayList(Slot) = .empty;
        var seq = first;
        var last_read = since;
        while (seq <= newest and out.items.len < limit) : (seq += 1) {
            last_read = seq;
            const x = s[@intCast((seq - 1) % capacity)];
            if (@backingInt(x.level) < @backingInt(min)) continue;
            if (contains.len > 0 and std.mem.indexOf(u8, x.text[0..x.len], contains) == null) continue;
            try out.append(a, x);
        }
        return .{ .lines = out.items, .next_seq = if (seq > newest) newest + 1 else last_read + 1, .dropped = if (first > since + 1) first - since - 1 else 0 };
    }
};

/// Text as given, the live line cleared first.
pub fn raw(text: []const u8) void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (shown) {
        writeOut(clear);
        shown = false;
    }
    writeOut(text);
    if (text.len > 0) line_start = text[text.len - 1] == '\n';
}

/// Redraws the status line unless a log line is half written.
pub fn status(text: []const u8, width: usize) void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (!line_start) return;
    writeOut(clear);
    writeOut(text[0..@min(text.len, @max(20, width) - 1)]);
    shown = true;
}

pub fn clearStatus() void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (shown) writeOut(clear);
    shown = false;
}

pub fn keySuffix(c: *const Conn) []const u8 {
    if (c.log_label) |label| return label_suffix(label);
    if (!c.auth_enabled) return "";
    return if (c.key_label) |label| label_suffix(label) else " key=unauthenticated";
}

threadlocal var suffix_buf: [80]u8 = undefined;

fn label_suffix(label: []const u8) []const u8 {
    return std.fmt.bufPrint(&suffix_buf, " key={s}", .{label}) catch " key=?";
}

/// The access line ``send_response`` prints: ``ip "GET / HTTP/1.1" 200 -``.
pub fn request(c: *const Conn, code: u16) void {
    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.print("[tensorfold] {s} \"{s}\" {d} -{s}\n", .{ c.peer, c.requestline, code, keySuffix(c) }) catch {
        buf[buf.len - 1] = '\n';
        w.end = buf.len;
    };
    // the ring's copy has no query string: it can hold a key or a prompt, and the admin surface shows the ring
    var cut: [1024]u8 = undefined;
    emit(w.buffered(), withoutQuery(&cut, w.buffered()));
}

/// `text` with the first query string (from a `?` to the next space) removed, in `buf`; `text` itself when it has none or
/// does not fit.
fn withoutQuery(buf: []u8, text: []const u8) []const u8 {
    const q = std.mem.indexOfScalar(u8, text, '?') orelse return text;
    const end = std.mem.indexOfScalarPos(u8, text, q, ' ') orelse text.len;
    if (q + (text.len - end) > buf.len) return text[0..q];
    @memcpy(buf[0..q], text[0..q]);
    @memcpy(buf[q..][0 .. text.len - end], text[end..]);
    return buf[0 .. q + text.len - end];
}

/// ``refused <id>: <message>``, the message cut to 300 characters: why a request got an error reply.
pub fn refused(buf: []u8, id: []const u8, message: []const u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("refused {s}: {s}", .{ id, message[0 .. message.len - afterChars(message, 300).len] }) catch {};
    return w.buffered();
}

/// ``ended <id> reason=...``: a request that ends without its reply, because its client left (``why`` null) or it failed.
pub fn ended(buf: []u8, id: []const u8, why: ?[]const u8, prompt: usize, tokens: usize, after: f64) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("ended {s} reason=", .{id}) catch {};
    if (why) |e| w.print("error ({s})", .{e}) catch {} else w.writeAll("client-left") catch {};
    w.print(" prompt={d} tokens={d} after={d:.2}s", .{ prompt, tokens, after }) catch {};
    return w.buffered();
}

test "a request's refused and ended lines" {
    var buf: [1400]u8 = undefined;
    const accents: [301][2]u8 = @splat(.{ 0xc3, 0xa9 }); // two bytes a character, so a byte cut would split one
    const cut = refused(&buf, "cmpl-1", std.mem.sliceAsBytes(&accents));
    try std.testing.expectEqualStrings("refused cmpl-1: ", cut[0..16]);
    try std.testing.expectEqualStrings(std.mem.sliceAsBytes(accents[0..300]), cut[16..]);
    try std.testing.expectEqualStrings("refused cmpl-1: short", refused(&buf, "cmpl-1", "short"));
    try std.testing.expectEqualStrings("ended chatcmpl-1 reason=client-left prompt=12 tokens=3 after=0.50s", ended(&buf, "chatcmpl-1", null, 12, 3, 0.5));
    try std.testing.expectEqualStrings("ended chatcmpl-1 reason=error (OutOfMemory) prompt=12 tokens=0 after=1.25s", ended(&buf, "chatcmpl-1", "OutOfMemory", 12, 0, 1.25));
}
