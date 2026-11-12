//! One read of several device regions into pinned host memory: the copies queued behind the work that produced them, then one wait.
//! (The drafter's pass read three regions with three synchronous copies after a stream wait, each costing a blocking sync's wake of
//! ~67 us on rank 0: MEASURED, deccap-r0; queued as asynchronous copies into page-locked memory they cost that wake once.)
const std = @import("std");

/// A device region and where its bytes go.
pub const Region = struct { dst: [*]u8, src: u64, bytes: usize };

/// Queues every region's copy on the stream in order, then waits for the stream once: afterwards every `dst` holds its bytes. `io` is
/// the stream: `copyAsync(dst, src, bytes)` queues a device-to-host copy behind what is already queued, `synchronize()` returns when
/// all of it has run.
pub fn readAll(io: anytype, regions: []const Region) !void {
    for (regions) |r| try io.copyAsync(r.dst, r.src, r.bytes);
    try io.synchronize();
}

/// The model the tests run on: device memory, a stream whose copies run only at synchronize (as a queued copy runs when the GPU gets to
/// it, not when the host calls), and a count of waits.
const Model = struct {
    device: []const u8,
    queue: [8]Region = undefined,
    queued: usize = 0,
    syncs: usize = 0,
    base: u64 = 0x1000,

    pub fn copyAsync(m: *Model, dst: [*]u8, src: u64, bytes: usize) !void {
        if (m.queued == m.queue.len) return error.QueueFull;
        m.queue[m.queued] = .{ .dst = dst, .src = src, .bytes = bytes };
        m.queued += 1;
    }

    pub fn synchronize(m: *Model) !void {
        m.syncs += 1;
        for (m.queue[0..m.queued]) |r| @memcpy(r.dst[0..r.bytes], m.device[r.src - m.base ..][0..r.bytes]);
        m.queued = 0;
    }
};

test "readAll: nothing is on the host until the one wait, then every region is, with exactly one wait" {
    var dev: [96]u8 = undefined;
    for (&dev, 0..) |*b, i| b.* = @truncate(i * 7 + 3);
    var m: Model = .{ .device = &dev };
    var a: [16]u8 = @splat(0xEE);
    var b: [32]u8 = @splat(0xEE);
    var c: [8]u8 = @splat(0xEE);
    const regions = [_]Region{
        .{ .dst = &a, .src = m.base + 0, .bytes = a.len },
        .{ .dst = &b, .src = m.base + 40, .bytes = b.len },
        .{ .dst = &c, .src = m.base + 88, .bytes = c.len },
    };
    // a reader that looked early would see the 0xEE fill: the copies are only queued
    for (regions) |r| try m.copyAsync(r.dst, r.src, r.bytes);
    try std.testing.expectEqual(@as(u8, 0xEE), a[0]);
    m.queued = 0;
    try readAll(&m, &regions);
    try std.testing.expectEqual(@as(usize, 1), m.syncs);
    try std.testing.expectEqualSlices(u8, dev[0..16], &a);
    try std.testing.expectEqualSlices(u8, dev[40..72], &b);
    try std.testing.expectEqualSlices(u8, dev[88..96], &c);
}

test "readAll: a failed copy fails the read before any wait" {
    var dev: [8]u8 = @splat(1);
    var m: Model = .{ .device = &dev };
    var sink: [8]u8 = undefined;
    var regions: [9]Region = undefined;
    for (&regions) |*r| r.* = .{ .dst = &sink, .src = m.base, .bytes = 1 };
    try std.testing.expectError(error.QueueFull, readAll(&m, &regions));
    try std.testing.expectEqual(@as(usize, 0), m.syncs);
}
