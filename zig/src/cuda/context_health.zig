//! Sticky device faults: CUDA reports these execution faults as leaving the context unusable until it is destroyed, so
//! every later call on it fails. A driver's context holds an entry while it lives (one context a driver here); the
//! first sticky result any call on it reports stays there, read without a driver call (HTTP threads read it).
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;

const Entry = struct { driver: usize = 0, refs: usize = 0, fault: abi.Result = 0 };
var entries: [16]Entry = @splat(.{});
var mutex: std.atomic.Mutex = .unlocked;

fn lock() void {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
}

/// The asynchronous execution faults CUDA documents as needing the context destroyed before further work.
pub fn sticky(code: abi.Result) bool {
    return switch (code) {
        214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719 => true,
        else => false,
    };
}

/// A context of `d` comes to life; a second retain neither adds an entry nor clears a fault.
pub fn retain(d: *const Driver) error{OutOfMemory}!void {
    lock();
    defer mutex.unlock();
    for (&entries) |*e| if (e.refs > 0 and e.driver == @intFromPtr(d)) {
        e.refs += 1;
        return;
    };
    for (&entries) |*e| if (e.refs == 0) {
        e.* = .{ .driver = @intFromPtr(d), .refs = 1 };
        return;
    };
    return error.OutOfMemory;
}

/// A context of `d` goes; the last one's entry goes with it.
pub fn release(d: *const Driver) void {
    lock();
    defer mutex.unlock();
    for (&entries) |*e| if (e.refs > 0 and e.driver == @intFromPtr(d)) {
        e.refs -= 1;
        if (e.refs == 0) e.* = .{};
        return;
    };
}

/// A call on `d`'s context returned `code`: a sticky fault is kept (the first one).
pub fn observe(d: *const Driver, code: abi.Result) void {
    if (!sticky(code)) return;
    lock();
    defer mutex.unlock();
    for (&entries) |*e| if (e.refs > 0 and e.driver == @intFromPtr(d)) {
        if (e.fault == 0) e.fault = code;
        return;
    };
}

/// The sticky fault `d`'s context reported (0: none, or no context).
pub fn fault(d: *const Driver) abi.Result {
    lock();
    defer mutex.unlock();
    for (entries) |e| if (e.refs > 0 and e.driver == @intFromPtr(d)) return e.fault;
    return 0;
}

test "a sticky fault stays on its driver's context through success and another retain; other drivers and other codes stay healthy" {
    var d: Driver = undefined;
    var other: Driver = undefined;
    try retain(&d);
    defer release(&d);
    try retain(&other);
    defer release(&other);
    try std.testing.expectError(error.CudaFailed, d.result(800));
    try std.testing.expectError(error.OutOfDeviceMemory, d.result(2));
    try std.testing.expectEqual(@as(abi.Result, 0), fault(&d));
    try std.testing.expectError(error.CudaContextLost, d.result(700));
    try d.result(0); // a later success does not repair the context
    try std.testing.expectError(error.CudaContextLost, d.result(719));
    try retain(&d);
    try std.testing.expectEqual(@as(abi.Result, 700), fault(&d)); // the first one kept
    release(&d);
    try std.testing.expectEqual(@as(abi.Result, 700), fault(&d));
    try std.testing.expectEqual(@as(abi.Result, 0), fault(&other));
    for ([_]abi.Result{ 214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719 }) |code| try std.testing.expect(sticky(code));
    for ([_]abi.Result{ 0, 1, 2, 600, 701, 800, 801, 999 }) |code| try std.testing.expect(!sticky(code));
}

test "a released context's fault goes with it; a fault with no context is not kept" {
    var d: Driver = undefined;
    try retain(&d);
    _ = d.result(700) catch {};
    release(&d);
    try std.testing.expectEqual(@as(abi.Result, 0), fault(&d));
    _ = d.result(700) catch {};
    try retain(&d);
    defer release(&d);
    try std.testing.expectEqual(@as(abi.Result, 0), fault(&d));
}

test "every entry held: another context is refused before it can escape" {
    var ds: [entries.len + 1]Driver = undefined;
    var held: usize = 0;
    defer for (ds[0..held]) |*d| release(d);
    for (ds[0..entries.len]) |*d| {
        try retain(d);
        held += 1;
    }
    try std.testing.expectError(error.OutOfMemory, retain(&ds[entries.len]));
}
