//! Engram rows kept in host memory: a set-associative cache of packed records by row id, shared by every reader
//! thread. It holds the bytes a read returned, so a hit is the row a read would give.
const std = @import("std");

/// How initFaulted faulted a cache's records in: 2 MiB pages asked for (MADV_HUGEPAGE taken), the kernel's populate
/// (else a write to each page, or none: `why`).
pub const Fault = struct { huge: bool = false, populated: bool = false, written: bool = false, why: ?[]const u8 = null };

/// AnonHugePages of /proc/self/smaps_rollup's text, in KiB (null: no such line).
pub fn anonHugeKiB(text: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |l| {
        if (!std.mem.startsWith(u8, l, "AnonHugePages:")) continue;
        var words = std.mem.tokenizeScalar(u8, l["AnonHugePages:".len..], ' ');
        return std.fmt.parseInt(u64, words.next() orelse return null, 10) catch null;
    }
    return null;
}

/// MADV_POPULATE_WRITE (Linux 5.14; not in Zig's MADV list).
const madv_populate_write = 23;

pub const Cache = struct {
    rec: usize,
    ways: usize,
    sets: usize,
    keys: []u64, // a way's row id + 1 (0: empty)
    data: []u8, // [sets * ways] records
    next: []u8, // each set's next way to fill
    locks: []std.atomic.Value(u32), // one a set
    hits: std.atomic.Value(u64) = .init(0),
    misses: std.atomic.Value(u64) = .init(0),
    mapped: bool = false, // `data` is a mapping of its own (initFaulted), unmapped by deinit

    /// About `bytes` of records of `rec` bytes, `ways` a set (bytes too few for one set: error.CacheTooSmall).
    pub fn init(gpa: std.mem.Allocator, bytes: usize, rec: usize, ways: usize) !Cache {
        if (rec == 0 or ways == 0 or ways > 255) return error.BadCache;
        const sets = bytes / rec / ways;
        if (sets == 0) return error.CacheTooSmall;
        const data = try gpa.alloc(u8, sets * ways * rec);
        errdefer gpa.free(data);
        return parts(gpa, sets, ways, rec, data);
    }

    /// The cache of `sets` sets over `data` (sets * ways records of rec bytes): its keys, fills and locks.
    fn parts(gpa: std.mem.Allocator, sets: usize, ways: usize, rec: usize, data: []u8) !Cache {
        const keys = try gpa.alloc(u64, sets * ways);
        errdefer gpa.free(keys);
        const next = try gpa.alloc(u8, sets);
        errdefer gpa.free(next);
        const locks = try gpa.alloc(std.atomic.Value(u32), sets);
        @memset(keys, 0);
        @memset(next, 0);
        @memset(locks, .init(0));
        return .{ .rec = rec, .ways = ways, .sets = sets, .keys = keys, .data = data, .next = next, .locks = locks };
    }

    /// init with the records in an anonymous mapping of their own, faulted in now (TF_DS_ENGRAM_PREFAULT; `got`: how):
    /// madvised for 2 MiB pages, which the kernel gives where it can, then populated writable (MADV_POPULATE_WRITE;
    /// a kernel without it: a write of 0 to each page). No record is read before it is put, so the cache answers as
    /// init's; its first puts take no page fault.
    pub fn initFaulted(gpa: std.mem.Allocator, bytes: usize, rec: usize, ways: usize, got: *Fault) !Cache {
        return initPopulated(gpa, bytes, rec, ways, got, madv_populate_write);
    }

    /// initFaulted with `populate` the populate advice (a kernel that refuses it: the writes).
    fn initPopulated(gpa: std.mem.Allocator, bytes: usize, rec: usize, ways: usize, got: *Fault, populate: u32) !Cache {
        if (rec == 0 or ways == 0 or ways > 255) return error.BadCache;
        const sets = bytes / rec / ways;
        if (sets == 0) return error.CacheTooSmall;
        const map = try std.posix.mmap(null, sets * ways * rec, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        errdefer std.posix.munmap(map);
        got.* = .{};
        got.huge = if (std.posix.madvise(map.ptr, map.len, std.os.linux.MADV.HUGEPAGE)) |_| true else |_| false;
        if (std.posix.madvise(map.ptr, map.len, populate)) |_| {
            got.populated = true;
        } else |e| switch (e) {
            // a kernel without it: a write to each page (the value a fresh page holds)
            error.InvalidSyscall => {
                var at: usize = 0;
                while (at < map.len) : (at += std.heap.page_size_min) map[at] = 0;
                got.written = true;
            },
            // out of memory or another refusal: the pages fault in as the cache fills, as without
            else => got.why = @errorName(e),
        }
        var c = try parts(gpa, sets, ways, rec, map);
        c.mapped = true;
        return c;
    }

    pub fn deinit(c: *Cache, gpa: std.mem.Allocator) void {
        gpa.free(c.keys);
        if (c.mapped) std.posix.munmap(@alignCast(c.data)) else gpa.free(c.data);
        gpa.free(c.next);
        gpa.free(c.locks);
        c.* = undefined;
    }

    fn lock(c: *Cache, s: usize) void {
        while (c.locks[s].cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn unlock(c: *Cache, s: usize) void {
        c.locks[s].store(0, .release);
    }

    /// Row `id`'s record into `out` (rec bytes) when held.
    pub fn get(c: *Cache, id: u64, out: []u8) bool {
        const s: usize = @intCast(id % c.sets);
        c.lock(s);
        defer c.unlock(s);
        for (0..c.ways) |w| {
            const k = s * c.ways + w;
            if (c.keys[k] == id + 1) {
                @memcpy(out[0..c.rec], c.data[k * c.rec ..][0..c.rec]);
                _ = c.hits.fetchAdd(1, .monotonic);
                return true;
            }
        }
        _ = c.misses.fetchAdd(1, .monotonic);
        return false;
    }

    /// Row `id`'s record (rec bytes) kept, over the set's oldest fill (a row held already stays as it is).
    pub fn put(c: *Cache, id: u64, row: []const u8) void {
        const s: usize = @intCast(id % c.sets);
        c.lock(s);
        defer c.unlock(s);
        for (0..c.ways) |i| if (c.keys[s * c.ways + i] == id + 1) return;
        const w: usize = c.next[s] % c.ways;
        c.next[s] = @intCast((w + 1) % c.ways);
        const k = s * c.ways + w;
        @memcpy(c.data[k * c.rec ..][0..c.rec], row[0..c.rec]);
        c.keys[k] = id + 1;
    }
};

test "a cached record is the one put, and a full set gives up its oldest fill" {
    const a = std.testing.allocator;
    var c = try Cache.init(a, 2 * 2 * 4, 4, 2); // two sets of two ways
    defer c.deinit(a);
    var out: [4]u8 = undefined;
    try std.testing.expect(!c.get(6, &out));
    c.put(6, &.{ 1, 2, 3, 4 });
    try std.testing.expect(c.get(6, &out));
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &out);
    c.put(6, &.{ 9, 9, 9, 9 }); // held already: unchanged
    try std.testing.expect(c.get(6, &out));
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &out);
    // set 0 holds 6; 8 and 10 fill it again, the first over 6
    c.put(8, &.{ 5, 5, 5, 5 });
    c.put(10, &.{ 7, 7, 7, 7 });
    try std.testing.expect(!c.get(6, &out));
    try std.testing.expect(c.get(8, &out) and c.get(10, &out));
    try std.testing.expectEqual(@as(u64, 4), c.hits.load(.monotonic));
    try std.testing.expectError(error.CacheTooSmall, Cache.init(a, 7, 4, 2));
}

test "a faulted cache: its records a mapping of their own, every page resident before the first put, answers as init's" {
    const a = std.testing.allocator;
    var got: Fault = .{};
    var c = try Cache.initFaulted(a, 1 << 20, 4, 2, &got); // 131,072 sets of two 4-byte ways: 256 pages
    defer c.deinit(a);
    try std.testing.expect(c.mapped);
    try std.testing.expect(@intFromPtr(c.data.ptr) % std.heap.page_size_min == 0);
    try std.testing.expect(got.populated and got.huge); // (a kernel with MADV_POPULATE_WRITE and transparent huge pages)
    try expectResidentZeros(c.data);
    // every byte as a fresh page's, and the cache's answers init's (the first test's sequence)
    var out: [4]u8 = undefined;
    try std.testing.expect(!c.get(6, &out));
    c.put(6, &.{ 1, 2, 3, 4 });
    try std.testing.expect(c.get(6, &out));
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &out);
    c.put(131072 + 6, &.{ 5, 5, 5, 5 });
    c.put(2 * 131072 + 6, &.{ 7, 7, 7, 7 }); // the set's third row: over its oldest fill, 6
    try std.testing.expect(!c.get(6, &out));
    try std.testing.expectEqual(@as(u64, 1), c.hits.load(.monotonic));
}

test "smaps_rollup's AnonHugePages in KiB" {
    const text = "55a0-7ffd ---p 00000000 00:00 0 [rollup]\nRss:             2311208 kB\nAnonymous:       2121376 kB\nAnonHugePages:   2097152 kB\nShmemPmdMapped:        0 kB\n";
    try std.testing.expectEqual(@as(?u64, 2097152), anonHugeKiB(text));
    try std.testing.expectEqual(@as(?u64, 0), anonHugeKiB("AnonHugePages:         0 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), anonHugeKiB("Rss: 12 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), anonHugeKiB("AnonHugePages: x kB\n"));
}

/// Every page of `data` resident, every byte 0.
fn expectResidentZeros(data: []u8) !void {
    const pages = (data.len + std.heap.page_size_min - 1) / std.heap.page_size_min;
    const vec = try std.testing.allocator.alloc(u8, pages);
    defer std.testing.allocator.free(vec);
    try std.posix.mincore(@alignCast(data.ptr), data.len, vec.ptr);
    for (vec) |v| try std.testing.expect(v & 1 == 1);
    try std.testing.expect(std.mem.allEqual(u8, data, 0));
}

test "a kernel that refuses the populate advice: a write of 0 to each page faults the records in" {
    const a = std.testing.allocator;
    var got: Fault = .{};
    var c = try Cache.initPopulated(a, 1 << 20, 4, 2, &got, 9999); // an advice no kernel has: EINVAL
    defer c.deinit(a);
    try std.testing.expect(c.mapped and !got.populated and got.written and got.why == null);
    try expectResidentZeros(c.data);
}

test "a round's lookups and the pass worker's fills race on one cache and never see a torn record" {
    const a = std.testing.allocator;
    const rec = 48;
    var c = try Cache.init(a, 64 * rec * 4, rec, 4);
    defer c.deinit(a);
    const Fill = struct {
        fn row(id: u64, out: *[rec]u8) void {
            for (out, 0..) |*b, i| b.* = @truncate(id *% 31 +% i);
        }
        fn writer(cache: *Cache, base: u64) void {
            var out: [rec]u8 = undefined;
            for (0..20000) |i| {
                const id = base + i % 997;
                row(id, &out);
                cache.put(id, &out);
            }
        }
        fn reader(cache: *Cache, torn: *std.atomic.Value(usize)) void {
            var out: [rec]u8 = undefined;
            var want: [rec]u8 = undefined;
            for (0..20000) |i| {
                const id: u64 = i % 997;
                if (cache.get(id, &out)) {
                    row(id, &want);
                    if (!std.mem.eql(u8, &out, &want)) _ = torn.fetchAdd(1, .monotonic);
                }
            }
        }
    };
    var torn: std.atomic.Value(usize) = .init(0);
    const w = try std.Thread.spawn(.{}, Fill.writer, .{ &c, @as(u64, 0) });
    const r1 = try std.Thread.spawn(.{}, Fill.reader, .{ &c, &torn });
    const r2 = try std.Thread.spawn(.{}, Fill.reader, .{ &c, &torn });
    w.join();
    r1.join();
    r2.join();
    try std.testing.expectEqual(@as(usize, 0), torn.load(.monotonic));
    try std.testing.expect(c.hits.load(.monotonic) > 0);
}
