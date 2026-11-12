//! Rank 0's store open on its own thread: opening a full store scans every manifest, and the engine's start must not
//! wait on a disk. The engine thread waits at most TF_DS_SPILL_OPEN_MS; past it the tier is off for the process, named,
//! and the opener is abandoned, never joined: when the open ends, its own thread closes what opened and frees itself.
//! The tier never refuses the start.
const std = @import("std");

/// How a bounded wait for the open ended without a store.
pub const Outcome = union(enum) { timeout, failed: anyerror };

/// The tier's off reason for an open that gave no store.
pub fn reason(o: Outcome, open_ms: u32, buf: []u8) []const u8 {
    return switch (o) {
        .timeout => std.fmt.bufPrint(buf, "the store did not open within TF_DS_SPILL_OPEN_MS={d}", .{open_ms}) catch "the store did not open in time",
        .failed => |e| std.fmt.bufPrint(buf, "the store did not open: {s}", .{@errorName(e)}) catch "the store did not open",
    };
}

/// One open of a `Side` (anything the caller's `open` makes and `close` closes), on its own thread.
pub fn Opener(comptime Side: type) type {
    return struct {
        const Self = @This();

        pub const Result = union(enum) { opened: *Side, failed: anyerror, timeout };

        gpa: std.mem.Allocator,
        io: std.Io,
        mutex: std.Io.Mutex = .init,
        cond: std.Io.Condition = .init,
        state: enum { running, opened, failed, abandoned } = .running,
        side: ?*Side = null,
        err: anyerror = error.Unexpected,
        thread: std.Thread = undefined,
        ctx: *anyopaque,
        open: *const fn (ctx: *anyopaque) anyerror!*Side,
        close: *const fn (ctx: *anyopaque, side: *Side) void,

        /// The open started on its own thread; `ctx` outlives it (an abandoned opener still uses it when it ends).
        pub fn start(gpa: std.mem.Allocator, io: std.Io, ctx: *anyopaque, open: *const fn (ctx: *anyopaque) anyerror!*Side, close: *const fn (ctx: *anyopaque, side: *Side) void) !*Self {
            const self = try gpa.create(Self);
            self.* = .{ .gpa = gpa, .io = io, .ctx = ctx, .open = open, .close = close };
            self.thread = std.Thread.spawn(.{}, run, .{self}) catch |e| {
                gpa.destroy(self);
                return e;
            };
            return self;
        }

        fn run(self: *Self) void {
            const r = self.open(self.ctx);
            self.mutex.lockUncancelable(self.io);
            if (self.state == .abandoned) { // nobody waits: close what opened, free this, end
                self.mutex.unlock(self.io);
                if (r) |side| self.close(self.ctx, side) else |_| {}
                self.gpa.destroy(self);
                return;
            }
            if (r) |side| {
                self.side = side;
                self.state = .opened;
            } else |e| {
                self.err = e;
                self.state = .failed;
            }
            self.cond.broadcast(self.io);
            self.mutex.unlock(self.io);
        }

        /// Waits for the open at most `timeout_ns`. Opened or failed: the thread joined and this freed. Past the
        /// bound: abandoned (the thread detached; it closes what it opens and frees this itself).
        pub fn wait(self: *Self, timeout_ns: u64) Result {
            const deadline = std.Io.Clock.awake.now(self.io).toNanoseconds() + @as(i96, timeout_ns);
            self.mutex.lockUncancelable(self.io);
            while (self.state == .running) {
                const left = deadline - std.Io.Clock.awake.now(self.io).toNanoseconds();
                if (left <= 0) {
                    self.state = .abandoned;
                    self.mutex.unlock(self.io);
                    self.thread.detach();
                    return .timeout;
                }
                self.cond.waitTimeout(self.io, &self.mutex, .{ .duration = .{ .raw = .fromNanoseconds(left), .clock = .awake } }) catch {};
            }
            const r: Result = if (self.state == .opened) .{ .opened = self.side.? } else .{ .failed = self.err };
            self.mutex.unlock(self.io);
            self.thread.join();
            self.gpa.destroy(self);
            return r;
        }
    };
}
