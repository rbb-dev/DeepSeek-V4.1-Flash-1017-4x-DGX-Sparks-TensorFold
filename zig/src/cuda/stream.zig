//! Streams and events: ordering, waits and GPU-side timing.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const context = @import("context.zig");

pub const Stream = struct {
    d: *const Driver,
    handle: abi.Stream,

    /// A non-blocking stream never waits on the legacy default stream (what every engine stream should be).
    pub fn init(d: *const Driver, non_blocking: bool) Error!Stream {
        var s: abi.Stream = null;
        try d.check(d.api.cuStreamCreate(&s, if (non_blocking) abi.stream_non_blocking else 0), "cuStreamCreate");
        return .{ .d = d, .handle = s };
    }

    /// The context's stream priorities: `least` (the default, 0) and `greatest` (numerically lower: the most urgent).
    pub const PriorityRange = struct { least: c_int, greatest: c_int };

    pub fn priorityRange(d: *const Driver) Error!PriorityRange {
        var least: c_int = 0;
        var greatest: c_int = 0;
        try d.check(d.api.cuCtxGetStreamPriorityRange(&least, &greatest), "cuCtxGetStreamPriorityRange");
        return .{ .least = least, .greatest = greatest };
    }

    /// A non-blocking stream of the given priority (a value of priorityRange's ends; the driver clamps others).
    pub fn initPriority(d: *const Driver, prio: c_int) Error!Stream {
        var s: abi.Stream = null;
        try d.check(d.api.cuStreamCreateWithPriority(&s, abi.stream_non_blocking, prio), "cuStreamCreateWithPriority");
        return .{ .d = d, .handle = s };
    }

    pub fn priority(self: Stream) Error!c_int {
        var p: c_int = 0;
        try self.d.check(self.d.api.cuStreamGetPriority(self.handle, &p), "cuStreamGetPriority");
        return p;
    }

    pub fn deinit(self: *Stream) void {
        _ = self.d.api.cuStreamDestroy_v2(self.handle);
        self.* = undefined;
    }

    pub fn synchronize(self: Stream) Error!void {
        try self.d.check(self.d.api.cuStreamSynchronize(self.handle), "cuStreamSynchronize");
    }

    /// True once every queued item has finished.
    pub fn done(self: Stream) Error!bool {
        self.d.check(self.d.api.cuStreamQuery(self.handle), "cuStreamQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    /// A write of the 64-bit `value` to the mapped word `addr` (a device address of host memory), after the work queued
    /// before it; the driver fences the earlier work's stores first. It holds no SM.
    pub fn writeValue64(self: Stream, addr: abi.DevicePtr, value: u64) Error!void {
        try self.d.check(self.d.api.cuStreamWriteValue64_v2(self.handle, addr, value, 0), "cuStreamWriteValue64");
    }

    /// Waits for every (word, value) pair in one batch: all at least their values.
    pub fn waitValues64(self: Stream, words: []const abi.DevicePtr, values: []const u64) Error!void {
        std.debug.assert(words.len == values.len and words.len <= 8);
        var ops: [8]abi.BatchMemOp = undefined;
        for (words, values, 0..) |w, v, i| ops[i] = .{ .operation = abi.stream_mem_op_wait_value_64, .address = w, .value64 = v, .flags = abi.stream_wait_value_geq };
        try self.d.check(self.d.api.cuStreamBatchMemOp_v2(self.handle, @intCast(words.len), &ops, 0), "cuStreamBatchMemOp");
    }

    pub fn wait(self: Stream, event: Event) Error!void {
        try self.d.check(self.d.api.cuStreamWaitEvent(self.handle, event.handle, 0), "cuStreamWaitEvent");
    }

    /// `f(data)` on a driver thread once the work queued before it is done; the work queued after it waits for its
    /// return. `f` makes no CUDA call.
    pub fn hostFunc(self: Stream, f: abi.HostFn, data: ?*anyopaque) Error!void {
        try self.d.check(self.d.api.cuLaunchHostFunc(self.handle, f, data), "cuLaunchHostFunc");
    }
};

pub const Event = struct {
    d: *const Driver,
    handle: abi.Event,

    /// Timing events cost a little more to record; ordering-only events skip the timestamp.
    pub fn init(d: *const Driver, timing: bool) Error!Event {
        var e: abi.Event = null;
        // TF_DS_CUDA_EVENTS=blocking adds CU_EVENT_BLOCKING_SYNC (context.event_flags): a wait on the event sleeps
        const base: c_uint = if (timing) 0 else abi.event_disable_timing;
        try d.check(d.api.cuEventCreate(&e, base | context.event_flags), "cuEventCreate");
        return .{ .d = d, .handle = e };
    }

        return .{ .d = d, .handle = e };
    }

    pub fn deinit(self: *Event) void {
        _ = self.d.api.cuEventDestroy_v2(self.handle);
        self.* = undefined;
    }

    pub fn record(self: Event, stream: Stream) Error!void {
        try self.d.check(self.d.api.cuEventRecord(self.handle, stream.handle), "cuEventRecord");
    }

    pub fn synchronize(self: Event) Error!void {
        try self.d.check(self.d.api.cuEventSynchronize(self.handle), "cuEventSynchronize");
    }

    pub fn done(self: Event) Error!bool {
        self.d.check(self.d.api.cuEventQuery(self.handle), "cuEventQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    /// Milliseconds between two recorded timing events (about 0.5 us resolution).
    pub fn elapsedMs(start: Event, end: Event) Error!f32 {
        var ms: f32 = 0;
        try start.d.check(start.d.api.cuEventElapsedTime(&ms, start.handle, end.handle), "cuEventElapsedTime");
        return ms;
    }
};

test "TF_DS_CUDA_EVENTS: every event is made with the flags Context.init read, timing or not" {
    const Fake = struct {
        var flags: c_uint = 0xff;
        fn create(e: *abi.Event, f: c_uint) callconv(.c) abi.Result {
            e.* = null;
            flags = f;
            return abi.success;
        }
    };
    var driver: Driver = undefined;
    driver.api = undefined; // (only cuEventCreate is read)
    driver.api.cuEventCreate = &Fake.create;
    const saved = context.event_flags;
    defer context.event_flags = saved;

    context.event_flags = 0;
    _ = try Event.init(&driver, false);
    try std.testing.expectEqual(abi.event_disable_timing, Fake.flags);
    _ = try Event.init(&driver, true);
    try std.testing.expectEqual(@as(c_uint, 0), Fake.flags);

    context.event_flags = abi.event_blocking_sync;
    _ = try Event.init(&driver, false);
    try std.testing.expectEqual(abi.event_disable_timing | abi.event_blocking_sync, Fake.flags);
    _ = try Event.init(&driver, true);
    try std.testing.expectEqual(abi.event_blocking_sync, Fake.flags);
}
