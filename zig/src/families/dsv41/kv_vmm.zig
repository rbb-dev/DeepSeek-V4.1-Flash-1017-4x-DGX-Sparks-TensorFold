//! The KV pool in CUDA virtual memory (TF_DS_KV_VMM=1): every cache plane of the pool owns a stretch of one address
//! reservation sized for the largest pool the Spark could ever hold, and only the front of each stretch is backed by
//! device memory. A plane's address is its place in the reservation, so it never moves, and a CUDA graph that baked it
//! in stays valid whatever is mapped behind it (the probe, measured on the first Spark, driver 580.178.04: a graph captured at
//! 1 GiB replays after a grow to 4 GiB with no recapture; reserve 64 GiB 0.013 ms, grow ~32 ms/GiB, shrink ~9.5 ms/GiB).
//!
//! The bookkeeping is `Pool(B)` over a backend `B`: the driver's calls in service (`CudaBackend`), a fake in the tests.
//! Mappings are made in segments of `seg` bytes (`seg_default`) and unmapped whole (cuMemUnmap takes exactly the range a cuMemMap
//! made), so a plane shrinks in segments from its tail.

const std = @import("std");
const cuda = @import("cuda");

pub const Error = error{ OutOfAddressSpace, TooManyPlanes, PlaneTooLarge, BadPlane } || std.mem.Allocator.Error || cuda.Error;

/// The default mapping segment: a plane's front is mapped in pieces of this many bytes (rounded to the granularity), and a shrink
/// gives back whole pieces, so it keeps at most one piece's worth more of a plane than asked. 8 MiB: 170-250 planes keep under
/// ~1.4 GiB of slack at worst after a shrink, and a 16 GiB pool is ~2,000 mappings (setAccess runs once over a whole grow).
pub const seg_default: u64 = 8 << 20;

/// What the pool asks of the driver. `granularity` is the unit every mapped size and every plane's offset is a multiple of.
/// `map` backs [va, va + bytes) and returns the allocation's handle; `access` makes a mapped range (one or several maps) usable by
/// the device; `unmap` takes exactly one mapped range back.
pub fn isBackend(comptime B: type) void {
    _ = @as(*const fn (*B, u64) Error!u64, &B.reserve);
    _ = @as(*const fn (*B, u64, u64) void, &B.release);
    _ = @as(*const fn (*B, u64, u64) Error!u64, &B.map);
    _ = @as(*const fn (*B, u64, u64) Error!void, &B.access);
    _ = @as(*const fn (*B, u64, u64, u64) void, &B.unmap);
}

pub fn Pool(comptime B: type) type {
    comptime isBackend(B);
    return struct {
        const Self = @This();

        pub const Plane = struct { off: u64, max: u64, mapped: u64 };
        const Segment = struct { plane: u32, va: u64, bytes: u64, handle: u64 };

        gpa: std.mem.Allocator,
        b: B,
        gran: u64,
        seg: u64,
        base: u64,
        va_bytes: u64,
        carved: u64 = 0,
        planes: std.ArrayList(Plane) = .empty,
        segs: std.ArrayList(Segment) = .empty,
        /// Bytes asked of addPlane (before rounding to the granularity), for the start line.
        asked: u64 = 0,
        mapped: u64 = 0,

        /// Reserves `va_bytes` of address space (rounded up to the granularity). Nothing is backed yet.
        pub fn open(gpa: std.mem.Allocator, b: B, gran: u64, seg_bytes: u64, va_bytes: u64) Error!Self {
            var s: Self = .{ .gpa = gpa, .b = b, .gran = gran, .seg = @max(gran, seg_bytes / gran * gran), .base = 0, .va_bytes = std.mem.alignForward(u64, va_bytes, gran) };
            s.base = try s.b.reserve(s.va_bytes);
            return s;
        }

        /// Unmaps everything, frees the reservation, and gives the bytes back to the process's device count.
        pub fn deinit(s: *Self) void {
            var i = s.segs.items.len;
            while (i > 0) {
                i -= 1;
                const g = s.segs.items[i];
                s.b.unmap(g.va, g.bytes, g.handle);
            }
            cuda.noteDeviceFreed(@intCast(s.mapped));
            s.mapped = 0;
            s.segs.deinit(s.gpa);
            s.planes.deinit(s.gpa);
            s.b.release(s.base, s.va_bytes);
            s.base = 0;
        }

        /// The address space a plane of up to `max` bytes takes.
        pub fn stride(s: *const Self, max: u64) u64 {
            return std.mem.alignForward(u64, @max(max, 1), s.gran);
        }

        /// A new plane: a stretch for up to `max` bytes with its first `bytes` backed (rounded up to the granularity); its
        /// address. Planes are carved in order.
        pub fn addPlane(s: *Self, max: u64, bytes: u64) Error!u64 {
            if (bytes > max) return error.PlaneTooLarge;
            const st = s.stride(max);
            if (s.carved + st > s.va_bytes) return error.OutOfAddressSpace;
            const i: u32 = @intCast(s.planes.items.len);
            try s.planes.append(s.gpa, .{ .off = s.carved, .max = max, .mapped = 0 });
            errdefer _ = s.planes.pop();
            s.carved += st;
            errdefer s.carved -= st;
            s.asked += bytes;
            errdefer s.asked -= bytes;
            try s.grow(i, bytes);
            return s.base + s.planes.items[i].off;
        }

        pub fn addressOf(s: *const Self, plane: u32) u64 {
            return s.base + s.planes.items[plane].off;
        }

        /// Backs plane `i` up to `bytes` (rounded up to the granularity). A failure leaves what was mapped before the call.
        pub fn grow(s: *Self, i: u32, bytes: u64) Error!void {
            if (i >= s.planes.items.len) return error.BadPlane;
            const p = &s.planes.items[i];
            if (bytes > p.max) return error.PlaneTooLarge;
            const want = std.mem.alignForward(u64, bytes, s.gran);
            const first = s.segs.items.len;
            var at = p.mapped;
            const from = at;
            errdefer s.undo(first, i);
            while (at < want) {
                const n = @min(s.seg, want - at);
                const va = s.base + p.off + at;
                const h = try s.b.map(va, n);
                s.segs.append(s.gpa, .{ .plane = i, .va = va, .bytes = n, .handle = h }) catch {
                    s.b.unmap(va, n, h);
                    return error.OutOfMemory;
                };
                at += n;
                p.mapped = at;
                s.mapped += n;
                cuda.noteDeviceHeld(@intCast(n));
            }
            if (want > from) try s.b.access(s.base + p.off + from, want - from);
        }

        fn undo(s: *Self, first: usize, i: u32) void {
            while (s.segs.items.len > first) {
                const g = s.segs.pop().?;
                s.b.unmap(g.va, g.bytes, g.handle);
                s.planes.items[i].mapped -= g.bytes;
                s.mapped -= g.bytes;
                cuda.noteDeviceFreed(@intCast(g.bytes));
            }
        }

        /// Gives back plane `i`'s tail: whole segments while at least `bytes` (rounded up) stay mapped. Returns the bytes
        /// the plane keeps mapped, which may exceed `bytes` by up to a segment.
        pub fn shrink(s: *Self, i: u32, bytes: u64) Error!u64 {
            if (i >= s.planes.items.len) return error.BadPlane;
            const p = &s.planes.items[i];
            const keep = std.mem.alignForward(u64, bytes, s.gran);
            var k = s.segs.items.len;
            while (k > 0 and p.mapped > keep) {
                k -= 1;
                const g = s.segs.items[k];
                if (g.plane != i) continue;
                if (p.mapped - g.bytes < keep) break;
                s.b.unmap(g.va, g.bytes, g.handle);
                _ = s.segs.orderedRemove(k);
                p.mapped -= g.bytes;
                s.mapped -= g.bytes;
                cuda.noteDeviceFreed(@intCast(g.bytes));
            }
            return p.mapped;
        }
    };
}

/// The driver's virtual memory calls for one device's own memory.
pub const CudaBackend = struct {
    d: *const cuda.Driver,
    device: cuda.abi.Device,

    fn prop(b: *const CudaBackend) cuda.abi.MemAllocationProp {
        return .{ .type = cuda.abi.mem_allocation_pinned, .location = .{ .type = cuda.abi.mem_location_device, .id = b.device } };
    }

    /// The granularity every mapping's size and offset is a multiple of (CU_MEM_ALLOC_GRANULARITY_RECOMMENDED).
    pub fn granularity(b: *const CudaBackend) cuda.Error!u64 {
        var g: usize = 0;
        const pr = b.prop();
        try b.d.check(b.d.api.cuMemGetAllocationGranularity(&g, &pr, cuda.abi.mem_granularity_recommended), "cuMemGetAllocationGranularity");
        return g;
    }

    pub fn reserve(b: *CudaBackend, bytes: u64) Error!u64 {
        var p: cuda.abi.DevicePtr = 0;
        try b.d.check(b.d.api.cuMemAddressReserve(&p, @intCast(bytes), 0, 0, 0), "cuMemAddressReserve");
        return p;
    }

    pub fn release(b: *CudaBackend, base: u64, bytes: u64) void {
        _ = b.d.api.cuMemAddressFree(base, @intCast(bytes));
    }

    pub fn map(b: *CudaBackend, va: u64, bytes: u64) Error!u64 {
        const pr = b.prop();
        var h: cuda.abi.MemHandle = 0;
        try b.d.check(b.d.api.cuMemCreate(&h, @intCast(bytes), &pr, 0), "cuMemCreate");
        errdefer _ = b.d.api.cuMemRelease(h);
        try b.d.check(b.d.api.cuMemMap(va, @intCast(bytes), 0, h, 0), "cuMemMap");
        return h;
    }

    /// The device's read and write access to [va, va + bytes): one call over every mapping the range holds.
    pub fn access(b: *CudaBackend, va: u64, bytes: u64) Error!void {
        const pr = b.prop();
        const acc = [1]cuda.abi.MemAccessDesc{.{ .location = pr.location, .flags = cuda.abi.mem_access_read_write }};
        try b.d.check(b.d.api.cuMemSetAccess(va, @intCast(bytes), &acc, 1), "cuMemSetAccess");
    }

    pub fn unmap(b: *CudaBackend, va: u64, bytes: u64, handle: u64) void {
        _ = b.d.api.cuMemUnmap(va, @intCast(bytes));
        _ = b.d.api.cuMemRelease(handle);
    }
};

pub const Driver = Pool(CudaBackend);

/// The most positions a pool of this layout could hold in `device_total` bytes: the reservation's size. `bytesAt(positions)` is
/// the layout's bytes (prompt.Caches.poolBytes), which grows with the positions; the answer is a multiple of `step`.
pub fn maxPositions(bytesAt: anytype, device_total: u64, step: usize) usize {
    var lo: usize = 0; // fits
    var hi: usize = step; // find one that does not
    while (bytesAt.bytes(hi) <= device_total) {
        lo = hi;
        hi *= 2;
        if (hi >= (1 << 33)) return lo;
    }
    while (hi - lo > step) {
        const mid = lo + (hi - lo) / 2 / step * step;
        if (mid == lo) break;
        if (bytesAt.bytes(mid) <= device_total) lo = mid else hi = mid;
    }
    return lo;
}

// ---- tests -------------------------------------------------------------------------------------------------------

/// A backend with no device: it records live mappings so a test can check what is mapped where.
pub const Fake = struct {
    const Map = struct { va: u64, bytes: u64, handle: u64 };
    live: std.ArrayList(Map) = .empty,
    next_handle: u64 = 1,
    reserved: u64 = 0,
    released: bool = false,
    fail_after: usize = std.math.maxInt(usize), // the Nth map call fails (OutOfDeviceMemory)
    calls: usize = 0,
    access_calls: usize = 0,
    fail_access: bool = false,
    gpa: std.mem.Allocator,

    pub fn reserve(f: *Fake, bytes: u64) Error!u64 {
        f.reserved = bytes;
        return 0x1000_0000_0000;
    }
    pub fn release(f: *Fake, _: u64, _: u64) void {
        f.released = true;
    }
    pub fn map(f: *Fake, va: u64, bytes: u64) Error!u64 {
        f.calls += 1;
        if (f.calls > f.fail_after) return error.OutOfDeviceMemory;
        const h = f.next_handle;
        f.next_handle += 1;
        f.live.append(f.gpa, .{ .va = va, .bytes = bytes, .handle = h }) catch return error.OutOfMemory;
        return h;
    }
    pub fn access(f: *Fake, _: u64, _: u64) Error!void {
        f.access_calls += 1;
        if (f.fail_access) return error.OutOfDeviceMemory;
    }
    pub fn unmap(f: *Fake, va: u64, bytes: u64, handle: u64) void {
        for (f.live.items, 0..) |m, i| if (m.va == va and m.bytes == bytes and m.handle == handle) {
            _ = f.live.orderedRemove(i);
            return;
        };
        @panic("unmap of a range that was not mapped as one piece");
    }
    /// Whether every byte of [va, va + len) is backed by a live mapping: what a kernel reading that range needs.
    pub fn isMapped(f: *const Fake, va: u64, len: u64) bool {
        var at = va;
        while (at < va + len) {
            var next: ?u64 = null;
            for (f.live.items) |m| if (m.va <= at and at < m.va + m.bytes) {
                next = m.va + m.bytes;
                break;
            };
            at = next orelse return false;
        }
        return true;
    }
    pub fn total(f: *const Fake) u64 {
        var t: u64 = 0;
        for (f.live.items) |m| t += m.bytes;
        return t;
    }
};

const mib: u64 = 1 << 20;

test "a plane's address is its place in the reservation, and only its front is backed" {
    const gpa = std.testing.allocator;
    var p = try Pool(Fake).open(gpa, .{ .gpa = gpa }, 2 * mib, 128 * mib, 1000 * mib); // (the pool holds the backend: p.b)
    defer p.b.live.deinit(gpa);
    try std.testing.expectEqual(@as(u64, 1000 * mib), p.va_bytes);
    const a = try p.addPlane(300 * mib, 5 * mib + 1); // backed 6 MiB: rounded up to the granularity
    const b = try p.addPlane(100 * mib, 100 * mib);
    try std.testing.expectEqual(p.base, a);
    try std.testing.expectEqual(p.base + 300 * mib, b); // the second plane starts after the first's whole stretch
    try std.testing.expectEqual(@as(u64, 106 * mib), p.mapped);
    try std.testing.expectEqual(@as(u64, 105 * mib + 1), p.asked);
    try std.testing.expectEqual(@as(u64, 106 * mib), p.b.total());
    p.deinit();
    try std.testing.expect(p.b.released);
    try std.testing.expectEqual(@as(usize, 0), p.b.live.items.len);
}

test "the reservation refuses a plane that does not fit, and a plane larger than its stretch" {
    const gpa = std.testing.allocator;
    var p = try Pool(Fake).open(gpa, .{ .gpa = gpa }, 2 * mib, 128 * mib, 100 * mib);
    defer p.b.live.deinit(gpa);
    defer p.deinit();
    _ = try p.addPlane(60 * mib, 10 * mib);
    try std.testing.expectError(error.OutOfAddressSpace, p.addPlane(60 * mib, 10 * mib));
    try std.testing.expectError(error.PlaneTooLarge, p.addPlane(10 * mib, 11 * mib));
    try std.testing.expectEqual(@as(usize, 1), p.planes.items.len); // a refused plane leaves nothing behind
    try std.testing.expectEqual(@as(u64, 10 * mib), p.mapped);
    try std.testing.expectError(error.PlaneTooLarge, p.grow(0, 61 * mib));
    try std.testing.expectError(error.BadPlane, p.grow(3, mib));
}

test "growing maps the new tail in segments and keeps the front's mappings and addresses" {
    const gpa = std.testing.allocator;
    var p = try Pool(Fake).open(gpa, .{ .gpa = gpa }, 2 * mib, 128 * mib, 2048 * mib);
    defer p.b.live.deinit(gpa);
    defer p.deinit();
    const a = try p.addPlane(1000 * mib, 100 * mib);
    try std.testing.expectEqual(@as(usize, 1), p.b.live.items.len);
    try p.grow(0, 500 * mib); // 400 MiB more: three segments of at most 128 MiB (128 + 128 + 128 + 16)
    try std.testing.expectEqual(@as(usize, 5), p.b.live.items.len);
    try std.testing.expectEqual(@as(u64, 500 * mib), p.mapped);
    try std.testing.expectEqual(a, p.addressOf(0));
    try std.testing.expectEqual(@as(u64, 100 * mib), p.b.live.items[0].bytes); // the front was not touched
    try p.grow(0, 200 * mib); // a smaller ask than is mapped changes nothing
    try std.testing.expectEqual(@as(u64, 500 * mib), p.mapped);
}

test "a grow that fails part way leaves the plane as it was" {
    const gpa = std.testing.allocator;
    var p = try Pool(Fake).open(gpa, .{ .gpa = gpa, .fail_after = 3 }, 2 * mib, 128 * mib, 2048 * mib);
    defer p.b.live.deinit(gpa);
    defer p.deinit();
    _ = try p.addPlane(1000 * mib, 100 * mib); // map call 1
    try std.testing.expectError(error.OutOfDeviceMemory, p.grow(0, 600 * mib)); // calls 2 and 3 map, 4 fails
    try std.testing.expectEqual(@as(u64, 100 * mib), p.mapped);
    try std.testing.expectEqual(@as(u64, 100 * mib), p.planes.items[0].mapped);
    try std.testing.expectEqual(@as(u64, 100 * mib), p.b.total()); // what the failed call mapped was unmapped again
}

test "a grow whose access call fails is undone whole" {
    const gpa = std.testing.allocator;
    var p = try Pool(Fake).open(gpa, .{ .gpa = gpa }, 2 * mib, 128 * mib, 2048 * mib);
    defer p.b.live.deinit(gpa);
    defer p.deinit();
    _ = try p.addPlane(1000 * mib, 100 * mib);
    p.b.fail_access = true;
    try std.testing.expectError(error.OutOfDeviceMemory, p.grow(0, 600 * mib));
    try std.testing.expectEqual(@as(u64, 100 * mib), p.mapped);
    try std.testing.expectEqual(@as(u64, 100 * mib), p.b.total());
    p.b.fail_access = false;
    try p.grow(0, 600 * mib); // and it can be asked again
    try std.testing.expectEqual(@as(u64, 600 * mib), p.mapped);
}

test "a pool opened with small segments gives back all but under one segment of a plane's tail" {
    const gpa = std.testing.allocator;
    var p = try Pool(Fake).open(gpa, .{ .gpa = gpa }, 2 * mib, 8 * mib, 1000 * mib);
    defer p.b.live.deinit(gpa);
    defer p.deinit();
    _ = try p.addPlane(500 * mib, 300 * mib); // 38 segments of 8 MiB (the last 4)
    try std.testing.expectEqual(@as(usize, 38), p.b.live.items.len);
    const kept = try p.shrink(0, 70 * mib + 1); // asked: 72 MiB rounded up to the granularity
    try std.testing.expect(kept >= 72 * mib and kept < 72 * mib + 8 * mib);
    try std.testing.expectEqual(@as(u64, 72 * mib), kept); // (72 is a multiple of the 8 MiB segment: nothing extra kept)
    try std.testing.expectEqual(kept, p.b.total());
    const kept2 = try p.shrink(0, 65 * mib); // asked 66 MiB: the segment ending at 72 holds 64-72, so 72 stays
    try std.testing.expectEqual(@as(u64, 72 * mib), kept2);
}

test "a shrink gives back whole tail segments, never below what is asked, and never another plane's" {
    const gpa = std.testing.allocator;
    var p = try Pool(Fake).open(gpa, .{ .gpa = gpa }, 2 * mib, 128 * mib, 4096 * mib);
    defer p.b.live.deinit(gpa);
    defer p.deinit();
    _ = try p.addPlane(1000 * mib, 100 * mib);
    _ = try p.addPlane(1000 * mib, 300 * mib); // segments 128, 128, 44
    try p.grow(0, 356 * mib); // plane 0's segments now 100, 128, 128; plane 1's sit between them in the list
    const before = p.mapped;
    // keep 250 MiB: the 128 MiB tail goes (356 -> 228 would drop below 250, so it stays); 356 - 128 = 228 < 250
    try std.testing.expectEqual(@as(u64, 356 * mib), try p.shrink(0, 250 * mib));
    try std.testing.expectEqual(before, p.mapped);
    // keep 228 MiB: the last segment goes
    try std.testing.expectEqual(@as(u64, 228 * mib), try p.shrink(0, 228 * mib));
    try std.testing.expectEqual(before - 128 * mib, p.mapped);
    try std.testing.expectEqual(@as(u64, 300 * mib), p.planes.items[1].mapped); // the other plane is whole
    try std.testing.expectEqual(@as(u64, 100 * mib), try p.shrink(0, 100 * mib)); // down to its first segment's end
    try std.testing.expectEqual(@as(u64, 0), try p.shrink(0, 0)); // and, asked for nothing, to nothing
    try std.testing.expectEqual(p.mapped, p.b.total());
    // and it can grow again, at the same address (plane 1's front stayed)
    const at = p.addressOf(0);
    try p.grow(0, 600 * mib);
    try std.testing.expectEqual(at, p.addressOf(0));
    try std.testing.expectEqual(@as(u64, 600 * mib), p.planes.items[0].mapped);
}

test "the largest pool a device could hold is found in steps of the pool step" {
    const Lin = struct {
        pub fn bytes(_: @This(), positions: usize) usize {
            return 1_000_000 + 890 * positions;
        }
    };
    const step: usize = 65536;
    const total: u64 = 100 << 30;
    const n = maxPositions(Lin{}, total, step);
    try std.testing.expect(n % step == 0);
    try std.testing.expect((Lin{}).bytes(n) <= total);
    try std.testing.expect((Lin{}).bytes(n + step) > total);
    try std.testing.expectEqual(@as(usize, 0), maxPositions(Lin{}, 10, step)); // nothing fits
}

test "the driver structures match cuda.h's sizes" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(cuda.abi.MemAllocationProp));
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(cuda.abi.MemAccessDesc));
}
