//! The Mac app's questions about a conversation's kept states (/v1/imprint), answered on the engine thread.
const std = @import("std");
const pc = @import("prompt_cache.zig");

/// Where a prompt's next request would resume.
pub const Resume = struct {
    tokens: u32 = 0, // the prompt's tokens
    ram: u32 = 0, // tokens a kept state in RAM covers
    disk: u32 = 0, // tokens a state on disk covers
    disk_bytes: u64 = 0,
    from: enum { ram, disk, none } = .none,
    seconds: f64 = 0, // the estimated wait: the disk read, then prefilling what is left
};

pub const Op = enum { where, load, forget };

/// Which of a request's keys the host's cache lookups pass: its chunk starts, its decoded spans.
pub const Keys = struct { starts: bool = true, spans: bool = true };

/// One question: a rendered conversation (its tokens, chunk starts and decoded spans), and the answer once `done`.
pub const Ask = struct {
    op: Op,
    prompt: []const u32 = &.{},
    starts: []const u32 = &.{},
    spans: []const [2]u32 = &.{},
    out: Resume = .{},
    loaded: u32 = 0, // load: the tokens a disk read brought into RAM (0: none was needed, or it did not pay)
    load_seconds: f64 = 0,
    freed: u64 = 0, // forget: bytes removed from disk
    disk: bool = false, // the engine keeps conversation states on disk
    done: std.atomic.Value(bool) = .init(false),

    /// The engine thread's answer from its kept states (`store` null: it keeps none), keyed as its lookups are. This engine
    /// keeps no conversation states on disk through this route (TensorFold's --learn-conversations is not in the merge): the
    /// answer is what its RAM states cover, and `load` and `forget` change nothing.
    pub fn answer(a: *Ask, store: ?*pc.Store, keys: Keys) void {
        defer a.done.store(true, .release);
        const s = store orelse return;
        a.disk = false;
        const starts: []const u32 = if (keys.starts) a.starts else &.{};
        a.out = .{ .tokens = @intCast(a.prompt.len) };
        if (s.find(a.prompt, starts)) |e| {
            a.out.ram = e.at;
            a.out.from = .ram;
        }
    }

    /// The answer within `ms` milliseconds (false: none came).
    pub fn wait(a: *const Ask, io: std.Io, ms: u32) bool {
        var waited: u32 = 0;
        while (!a.done.load(.acquire)) : (waited += 2) {
            if (waited >= ms) return false;
            std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
        }
        return true;
    }
};

/// A host's side of disk states: the asks waiting for its engine thread, and when the engine last went quiet.
pub const Disk = struct {
    asks: std.ArrayList(*Ask) = .empty,
    seen: u64 = 0, // the host's admissions when the writer last looked: a new one restarts the quiet
    since: ?i96 = null,

    /// The quiet after the last admission before the first write: a client's next turn often comes sooner.
    pub const quiet_ns: i96 = 150 * std.time.ns_per_ms;

    /// Queue `a` (the host's lock held).
    pub fn push(d: *Disk, gpa: std.mem.Allocator, a: *Ask) bool {
        d.asks.append(gpa, a) catch return false;
        return true;
    }

    /// The queued asks, taken (the host's lock held); the caller frees the list.
    pub fn take(d: *Disk, gpa: std.mem.Allocator) []*Ask {
        if (d.asks.items.len == 0) return &.{};
        const list = gpa.dupe(*Ask, d.asks.items) catch return &.{};
        d.asks.clearRetainingCapacity();
        return list;
    }

    /// Whether the writer may run now: no admission since `quiet_ns` ago (`admissions`: the host's count so far).
    pub fn quiet(d: *Disk, admissions: u64, now: i96) bool {
        if (d.seen != admissions or d.since == null) {
            d.seen = admissions;
            d.since = now;
            return false;
        }
        return now - d.since.? >= quiet_ns;
    }

    /// Asks left when the engine stops, answered as having no states; the list freed.
    pub fn close(d: *Disk, gpa: std.mem.Allocator) void {
        for (d.asks.items) |a| a.answer(null, .{});
        d.asks.deinit(gpa);
    }
};
