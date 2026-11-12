//! One device's primary context, the one CUDA's runtime and PyTorch share, made current on the calling thread.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const health = @import("context_health.zig");

/// TF_DS_CUDA_SCHED: how the primary context's host threads wait in a synchronize call. Unset, empty or white space:
/// the driver's default, and no flag call is made. `auto`, `spin`, `yield`, `blocking` (the thread sleeps until the GPU
/// signals), surrounding white space ignored. Any other text is refused at the ranks' settings exchange. Every tool
/// that calls Context.init (the server, the followers, the CLI, the replay mains and the benches) inherits this
/// setting from its environment.
pub fn schedFlags(text: ?[]const u8) Error!?c_uint {
    const v = std.mem.trim(u8, text orelse return null, " \t\r\n");
    if (v.len == 0) return null;
    const names = [_]struct { []const u8, c_uint }{ .{ "auto", abi.ctx_sched_auto }, .{ "spin", abi.ctx_sched_spin }, .{ "yield", abi.ctx_sched_yield }, .{ "blocking", abi.ctx_sched_blocking_sync } };
    for (names) |n| if (std.mem.eql(u8, v, n[0])) return n[1];
    return error.Invalid;
}

/// The settings exchange's codes for TF_DS_CUDA_SCHED: the request, and what the readback showed.
pub const sched_unset: u32 = 0;
pub const sched_auto: u32 = 1;
pub const sched_spin: u32 = 2;
pub const sched_yield: u32 = 3;
pub const sched_blocking: u32 = 4;
pub const sched_invalid: u32 = 5; // not one of auto, spin, yield, blocking
pub const sched_active_mismatch: u32 = 6; // the primary context was already active with other flags
pub const sched_after_mismatch: u32 = 7; // the set did not take effect

pub const sched_state_none: u32 = 0;
pub const sched_state_in_effect: u32 = 1;
pub const sched_state_active_mismatch: u32 = 2;
pub const sched_state_after_mismatch: u32 = 3;

/// TF_DS_CUDA_SCHED as Context.init read it: what it asked for, and what the primary context showed for it.
pub const Sched = struct {
    asked: u32 = sched_unset,
    state: u32 = sched_state_none,
};

/// The last Context.init's TF_DS_CUDA_SCHED, which startSettings carries to the ranks' settings exchange.
pub var sched: Sched = .{};

/// The code startSettings carries: the request, or the refusal the readback showed.
pub fn schedCode() u32 {
    return schedCodeOf(sched);
}

fn schedCodeOf(s: Sched) u32 {
    return switch (s.state) {
        sched_state_active_mismatch => sched_active_mismatch,
        sched_state_after_mismatch => sched_after_mismatch,
        else => s.asked,
    };
}

/// The settings exchange's words for a code.
pub fn schedWord(code: u32) []const u8 {
    return switch (code) {
        sched_unset => "unset",
        sched_auto => "auto",
        sched_spin => "spin",
        sched_yield => "yield",
        sched_blocking => "blocking",
        sched_invalid => "text that is not one of auto, spin, yield, blocking",
        sched_active_mismatch => "a request the already-active primary context did not match",
        sched_after_mismatch => "a request the primary context did not take",
        else => "a value this build has no name for",
    };
}

fn schedCodeOfFlags(flags: c_uint) u32 {
    return switch (flags & abi.ctx_sched_mask) {
        abi.ctx_sched_auto => sched_auto,
        abi.ctx_sched_spin => sched_spin,
        abi.ctx_sched_yield => sched_yield,
        abi.ctx_sched_blocking_sync => sched_blocking,
        else => sched_invalid, // not reachable through schedFlags
    };
}

fn schedName(flags: c_uint) []const u8 {
    return switch (flags & abi.ctx_sched_mask) {
        abi.ctx_sched_auto => "auto",
        abi.ctx_sched_spin => "spin",
        abi.ctx_sched_yield => "yield",
        abi.ctx_sched_blocking_sync => "blocking sync",
        else => "an unnamed combination",
    };
}

/// Reads TF_DS_CUDA_SCHED and applies it to the primary context's flags before the retain. The state is read first,
/// because an already-active context keeps its live scheduling; a request that differs is refused at the ranks'
/// settings exchange. After a set, the flags are read back and compared with the request. An invalid text is not
/// fatal here: the rank still connects and the exchange refuses it on every rank.
pub fn schedApply(d: *const Driver, dev: abi.Device, text: ?[]const u8) Error!Sched {
    var s: Sched = .{};
    const want = schedFlags(text) catch {
        std.log.warn("TF_DS_CUDA_SCHED={s}: one of auto, spin, yield, blocking", .{text.?});
        s.asked = sched_invalid;
        return s;
    };
    if (want) |w| {
        s.asked = schedCodeOfFlags(w);
        var flags: c_uint = 0;
        var active: c_int = 0;
        try d.check(d.api.cuDevicePrimaryCtxGetState(dev, &flags, &active), "cuDevicePrimaryCtxGetState");
        if (active != 0 and (flags & abi.ctx_sched_mask) != w) {
            s.state = sched_state_active_mismatch;
            std.log.warn("TF_DS_CUDA_SCHED={s}: the primary context is already active with {s} (flags 0x{x})", .{ text.?, schedName(flags), flags });
            return s;
        }
        if (active == 0) try d.check(d.api.cuDevicePrimaryCtxSetFlags_v2(dev, w), "cuDevicePrimaryCtxSetFlags");
        var after: c_uint = 0;
        var after_active: c_int = 0;
        try d.check(d.api.cuDevicePrimaryCtxGetState(dev, &after, &after_active), "cuDevicePrimaryCtxGetState");
        if ((after & abi.ctx_sched_mask) != w) {
            s.state = sched_state_after_mismatch;
            std.log.warn("TF_DS_CUDA_SCHED={s}: the primary context reports {s} (flags 0x{x}) after the request", .{ text.?, schedName(after), after });
        } else {
            s.state = sched_state_in_effect;
            std.log.info("TF_DS_CUDA_SCHED={s}: the primary context's host threads wait by {s} (flags 0x{x})", .{ text.?, schedName(after), after });
        }
    }
    return s;
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
        const text: ?[]const u8 = if (std.c.getenv("TF_DS_CUDA_SCHED")) |v| std.mem.span(v) else null;
        sched = try schedApply(d, dev, text);
        event_flags = eventsApply(if (std.c.getenv("TF_DS_CUDA_EVENTS")) |v| std.mem.span(v) else null);
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        errdefer _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
        try d.check(d.api.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        errdefer _ = d.api.cuCtxSetCurrent(null);
        try health.retain(d);
        return .{ .d = d, .device = dev, .handle = ctx };
    }

    /// Waits for all work, then drops this retain; resources made in it must be released first.
    pub fn deinit(self: *Context) void {
        _ = self.d.api.cuCtxSynchronize();
        _ = self.d.api.cuDevicePrimaryCtxRelease_v2(self.device);
        health.release(self.d);
        self.* = undefined;
    }

    /// The sticky fault a call on this context reported (0: none): it serves nothing more. No driver call, so any
    /// thread may ask.
    pub fn fault(self: *const Context) abi.Result {
        return health.fault(self.d);
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

test "TF_DS_CUDA_SCHED: unset or white space makes no call, each name its flag, anything else refused" {
    try std.testing.expectEqual(@as(?c_uint, null), try schedFlags(null));
    try std.testing.expectEqual(@as(?c_uint, null), try schedFlags(""));
    try std.testing.expectEqual(@as(?c_uint, null), try schedFlags(" \t\r\n"));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_auto), try schedFlags("auto"));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_spin), try schedFlags("spin"));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_yield), try schedFlags("yield"));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_blocking_sync), try schedFlags("blocking"));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_blocking_sync), try schedFlags(" blocking "));
    try std.testing.expectEqual(@as(?c_uint, abi.ctx_sched_spin), try schedFlags("\tspin\n"));
    try std.testing.expectError(error.Invalid, schedFlags("block"));
    try std.testing.expectError(error.Invalid, schedFlags("4"));
    try std.testing.expectError(error.Invalid, schedFlags("spin yield"));
}

test "TF_DS_CUDA_SCHED's names: auto for 0, each flag's name, an unnamed combination for anything else" {
    try std.testing.expectEqualStrings("auto", schedName(abi.ctx_sched_auto));
    try std.testing.expectEqualStrings("spin", schedName(abi.ctx_sched_spin));
    try std.testing.expectEqualStrings("yield", schedName(abi.ctx_sched_yield));
    try std.testing.expectEqualStrings("blocking sync", schedName(abi.ctx_sched_blocking_sync));
    try std.testing.expectEqualStrings("an unnamed combination", schedName(0x03));
    try std.testing.expectEqualStrings("an unnamed combination", schedName(0x07));
}

test "TF_DS_CUDA_SCHED: the primary context's state is read before its flags are set, each branch" {
    const Fake = struct {
        var calls: [16][]const u8 = undefined;
        var n: usize = 0;
        var flags: c_uint = 0;
        var active: c_int = 0;
        var set_takes: bool = true;
        fn reset(f: c_uint, a: c_int, takes: bool) void {
            n = 0;
            flags = f;
            active = a;
            set_takes = takes;
        }
        fn rec(name: []const u8) void {
            calls[n] = name;
            n += 1;
        }
        fn deviceGet(dev: *abi.Device, ordinal: c_int) callconv(.c) abi.Result {
            _ = ordinal;
            dev.* = 0;
            rec("cuDeviceGet");
            return abi.success;
        }
        fn getState(dev: abi.Device, f: *c_uint, a: *c_int) callconv(.c) abi.Result {
            _ = dev;
            f.* = flags;
            a.* = active;
            rec("cuDevicePrimaryCtxGetState");
            return abi.success;
        }
        fn setFlags(dev: abi.Device, f: c_uint) callconv(.c) abi.Result {
            _ = dev;
            if (set_takes) flags = f;
            rec("cuDevicePrimaryCtxSetFlags");
            return abi.success;
        }
        fn retain(ctx: *abi.Context, dev: abi.Device) callconv(.c) abi.Result {
            _ = dev;
            ctx.* = @ptrFromInt(1);
            rec("cuDevicePrimaryCtxRetain");
            return abi.success;
        }
        fn release(dev: abi.Device) callconv(.c) abi.Result {
            _ = dev;
            rec("cuDevicePrimaryCtxRelease");
            return abi.success;
        }
        fn setCurrent(ctx: abi.Context) callconv(.c) abi.Result {
            _ = ctx;
            rec("cuCtxSetCurrent");
            return abi.success;
        }
        fn synchronize() callconv(.c) abi.Result {
            rec("cuCtxSynchronize");
            return abi.success;
        }
    };
    var driver: Driver = undefined;
    driver.api = undefined; // (only the entry points schedApply calls are read)
    driver.api.cuDeviceGet = &Fake.deviceGet;
    driver.api.cuDevicePrimaryCtxGetState = &Fake.getState;
    driver.api.cuDevicePrimaryCtxSetFlags_v2 = &Fake.setFlags;
    driver.api.cuDevicePrimaryCtxRetain = &Fake.retain;
    driver.api.cuDevicePrimaryCtxRelease_v2 = &Fake.release;
    driver.api.cuCtxSetCurrent = &Fake.setCurrent;
    driver.api.cuCtxSynchronize = &Fake.synchronize;

    // a fresh context: the state is read, the flags set, the state read again, all before the retain
    Fake.reset(0, 0, true);
    const fresh = try schedApply(&driver, 0, "blocking");
    try std.testing.expectEqual(sched_blocking, schedCodeOf(fresh));
    try std.testing.expectEqual(sched_state_in_effect, fresh.state);
    try std.testing.expectEqualSlices([]const u8, &.{ "cuDevicePrimaryCtxGetState", "cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxGetState" }, Fake.calls[0..Fake.n]);

    // an active context with the same flags: no SetFlags call
    Fake.reset(abi.ctx_sched_blocking_sync, 1, true);
    const same = try schedApply(&driver, 0, "blocking");
    try std.testing.expectEqual(sched_blocking, schedCodeOf(same));
    try std.testing.expectEqual(sched_state_in_effect, same.state);
    try std.testing.expectEqualSlices([]const u8, &.{ "cuDevicePrimaryCtxGetState", "cuDevicePrimaryCtxGetState" }, Fake.calls[0..Fake.n]);

    // an active context with different flags: refused, no SetFlags call
    Fake.reset(abi.ctx_sched_spin, 1, true);
    const active_mismatch = try schedApply(&driver, 0, "blocking");
    try std.testing.expectEqual(sched_active_mismatch, schedCodeOf(active_mismatch));
    try std.testing.expectEqual(sched_state_active_mismatch, active_mismatch.state);
    try std.testing.expectEqualSlices([]const u8, &.{"cuDevicePrimaryCtxGetState"}, Fake.calls[0..Fake.n]);

    // a set the driver did not take: refused
    Fake.reset(0, 0, false);
    const after_mismatch = try schedApply(&driver, 0, "blocking");
    try std.testing.expectEqual(sched_after_mismatch, schedCodeOf(after_mismatch));
    try std.testing.expectEqual(sched_state_after_mismatch, after_mismatch.state);
    try std.testing.expectEqualSlices([]const u8, &.{ "cuDevicePrimaryCtxGetState", "cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxGetState" }, Fake.calls[0..Fake.n]);

    // an invalid text: no driver call, the exchange refuses it
    Fake.reset(0, 0, true);
    const invalid = try schedApply(&driver, 0, "block");
    try std.testing.expectEqual(sched_invalid, schedCodeOf(invalid));
    try std.testing.expectEqual(@as(usize, 0), Fake.n);

    // unset: no driver call, the driver's default
    Fake.reset(0, 0, true);
    const unset = try schedApply(&driver, 0, null);
    try std.testing.expectEqual(sched_unset, schedCodeOf(unset));
    try std.testing.expectEqual(@as(usize, 0), Fake.n);
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
