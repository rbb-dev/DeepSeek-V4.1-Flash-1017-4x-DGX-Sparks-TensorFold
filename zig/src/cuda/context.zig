//! One device's primary context, the one CUDA's runtime and PyTorch share, made current on the calling thread.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
pub fn schedFlags(text: ?[]const u8) Error!?c_uint {
    if (v.len == 0) return null;
    const names = [_]struct { []const u8, c_uint }{ .{ "auto", abi.ctx_sched_auto }, .{ "spin", abi.ctx_sched_spin }, .{ "yield", abi.ctx_sched_yield }, .{ "blocking", abi.ctx_sched_blocking_sync } };
    for (names) |n| if (std.mem.eql(u8, v, n[0])) return n[1];
    return error.Invalid;
}

fn schedName(flags: c_uint) []const u8 {
    return switch (flags & abi.ctx_sched_mask) {
        abi.ctx_sched_spin => "spin",
        abi.ctx_sched_yield => "yield",
        abi.ctx_sched_blocking_sync => "blocking sync",
    };
}

/// TF_DS_CUDA_EVENTS: how a wait on one of the engine's events waits. Unset, empty or white space: the driver's default
/// (cuda.h: a cuEventSynchronize on an event made without CU_EVENT_BLOCKING_SYNC busy-waits, whatever the context's
/// scheduling flags). `blocking`, surrounding white space ignored: every event Context.init's process makes afterwards
/// carries CU_EVENT_BLOCKING_SYNC, so a thread that waits on one sleeps until the GPU completes it. Any other text is
/// not taken: the rank says so and keeps the default (how a wait waits changes no result).
pub fn eventFlags(text: ?[]const u8) error{Invalid}!c_uint {
    const v = std.mem.trim(u8, text orelse return 0, " \t\r\n");
    if (v.len == 0) return 0;
    if (std.mem.eql(u8, v, "blocking")) return abi.event_blocking_sync;
    return error.Invalid;
}

/// Reads TF_DS_CUDA_EVENTS into the flags Event.init adds; the start line when it is on, a warning when it is not taken.
fn eventsApply(text: ?[]const u8) c_uint {
    const f = eventFlags(text) catch {
        std.log.warn("TF_DS_CUDA_EVENTS={s}: blocking or unset; the engine's events keep the driver's default wait", .{text.?});
        return 0;
    };
    if (f != 0) std.log.info("TF_DS_CUDA_EVENTS={s}: a wait on the engine's events sleeps until the GPU completes them (CU_EVENT_BLOCKING_SYNC)", .{std.mem.trim(u8, text.?, " \t\r\n")});
    return f;
}

/// The flags every Event.init adds to its own (TF_DS_CUDA_EVENTS as Context.init read it).
pub var event_flags: c_uint = 0;

pub const Context = struct {
    d: *const Driver,
    device: abi.Device,
    handle: abi.Context,

    pub fn init(d: *const Driver, ordinal: c_int) Error!Context {
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        event_flags = eventsApply(if (std.c.getenv("TF_DS_CUDA_EVENTS")) |v| std.mem.span(v) else null);
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        errdefer _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
        try d.check(d.api.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        return .{ .d = d, .device = dev, .handle = ctx };
    }

    /// Waits for all work, then drops this retain; resources made in it must be released first.
    pub fn deinit(self: *Context) void {
        _ = self.d.api.cuCtxSynchronize();
        _ = self.d.api.cuDevicePrimaryCtxRelease_v2(self.device);
        self.* = undefined;
    }

    /// Makes the context current on another thread before it calls the driver.
    pub fn makeCurrent(self: *const Context) Error!void {
        try self.d.check(self.d.api.cuCtxSetCurrent(self.handle), "cuCtxSetCurrent");
    }

    pub fn synchronize(self: *const Context) Error!void {
        try self.d.check(self.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
    }

    pub fn attribute(self: *const Context, a: abi.DeviceAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.cuDeviceGetAttribute(&v, a, self.device), "cuDeviceGetAttribute");
        return v;
    }

    /// Compute capability as 10 * major + minor (GB10: 121).
    pub fn capability(self: *const Context) Error!u32 {
        const major = try self.attribute(.compute_capability_major);
        const minor = try self.attribute(.compute_capability_minor);
        return @intCast(10 * major + minor);
    }

    pub fn name(self: *const Context, buf: []u8) Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        try self.d.check(self.d.api.cuDeviceGetName(buf.ptr, @intCast(buf.len), self.device), "cuDeviceGetName");
        return std.mem.sliceTo(buf, 0);
    }

    pub const MemInfo = struct { free: usize, total: usize };

    pub fn memInfo(self: *const Context) Error!MemInfo {
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try self.d.check(self.d.api.cuMemGetInfo_v2(&m.free, &m.total), "cuMemGetInfo");
        return m;
    }
};

    try std.testing.expectEqual(@as(?c_uint, null), try schedFlags(null));
    try std.testing.expectEqual(@as(?c_uint, null), try schedFlags(""));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_auto), try schedFlags("auto"));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_spin), try schedFlags("spin"));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_yield), try schedFlags("yield"));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_blocking_sync), try schedFlags("blocking"));
    try std.testing.expectError(error.Invalid, schedFlags("block"));
    try std.testing.expectError(error.Invalid, schedFlags("4"));
}

test "TF_DS_CUDA_EVENTS: unset or white space adds nothing, blocking adds CU_EVENT_BLOCKING_SYNC, anything else refused" {
    try std.testing.expectEqual(@as(c_uint, 0), try eventFlags(null));
    try std.testing.expectEqual(@as(c_uint, 0), try eventFlags(""));
    try std.testing.expectEqual(@as(c_uint, 0), try eventFlags(" \t\r\n"));
    try std.testing.expectEqual(abi.event_blocking_sync, try eventFlags("blocking"));
    try std.testing.expectEqual(abi.event_blocking_sync, try eventFlags(" blocking\n"));
    try std.testing.expectError(error.Invalid, eventFlags("block"));
    try std.testing.expectError(error.Invalid, eventFlags("spin"));
    try std.testing.expectError(error.Invalid, eventFlags("1"));
}
