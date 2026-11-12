//! What rank 0 tells the other ranks' stores in a keep and in a restore, as bytes. A keep's body is everything the store's
//! keep call takes that is rank 0's decision: the extent and the kept state it reads, the ids, the lineage's names, the
//! origin, the pictures. A restore's body names the destination rank 0 reserved. A rank rebuilds the same store arguments
//! from them, so every store names the same point by the same key and the same blocks by the same names.
const std = @import("std");
const core = @import("core");
const format = core.spill.format;

pub const SubOp = enum(u8) {
    /// A spill_sync.Frame follows.
    proto = 1,
    /// Kept state `snap`'s write stages what it still reads from token `from` on, or is cancelled (Side.guard).
    guard = 2,
};

pub const KeepBody = struct {
    snap: u64,
    base: u64,
    at: u32,
    shared: bool,
    group: u64,
    origin: ?format.Origin,
    lineage_root: [16]u8,
    lineage_part: ?u32,
    forks: []const format.Fork,
    keyed: []const u32,
    raw: ?[]const u32,
    media: []const format.Media,

    /// The bytes of the body appended to `out`.
    pub fn encode(b: KeepBody, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        try put(gpa, out, u64, b.snap);
        try put(gpa, out, u64, b.base);
        try put(gpa, out, u32, b.at);
        try put(gpa, out, u8, @intFromBool(b.shared));
        try put(gpa, out, u64, b.group);
        try put(gpa, out, u8, @intFromBool(b.origin != null));
        if (b.origin) |o| {
            try out.appendSlice(gpa, &o.key);
            try put(gpa, out, u32, o.at);
        }
        try out.appendSlice(gpa, &b.lineage_root);
        try put(gpa, out, u8, @intFromBool(b.lineage_part != null));
        try put(gpa, out, u32, b.lineage_part orelse 0);
        try put(gpa, out, u32, @intCast(b.forks.len));
        for (b.forks) |f| {
            try out.appendSlice(gpa, &f.name);
            try put(gpa, out, u32, f.at);
        }
        try put(gpa, out, u32, @intCast(b.keyed.len));
        for (b.keyed) |x| try put(gpa, out, u32, x);
        try put(gpa, out, u8, @intFromBool(b.raw != null));
        if (b.raw) |r| {
            try put(gpa, out, u32, @intCast(r.len));
            for (r) |x| try put(gpa, out, u32, x);
        }
        try put(gpa, out, u32, @intCast(b.media.len));
        for (b.media) |m| {
            try put(gpa, out, u32, m.start);
            try put(gpa, out, u32, m.len);
            try out.appendSlice(gpa, &m.digest);
        }
    }

    /// A body read back; its slices live in `arena`. Anything short or with a count past the bytes left is BadBody.
    pub fn decode(arena: std.mem.Allocator, bytes: []const u8) !KeepBody {
        var r: Rd = .{ .b = bytes };
        var b: KeepBody = undefined;
        b.snap = try r.get(u64);
        b.base = try r.get(u64);
        b.at = try r.get(u32);
        b.shared = (try r.get(u8)) != 0;
        b.group = try r.get(u64);
        b.origin = null;
        if ((try r.get(u8)) != 0) b.origin = .{ .key = (try r.take(32))[0..32].*, .at = try r.get(u32) };
        b.lineage_root = (try r.take(16))[0..16].*;
        const has_part = (try r.get(u8)) != 0;
        const part = try r.get(u32);
        b.lineage_part = if (has_part) part else null;
        const nf = try r.count(20);
        const forks = try arena.alloc(format.Fork, nf);
        for (forks) |*f| f.* = .{ .name = (try r.take(16))[0..16].*, .at = try r.get(u32) };
        b.forks = forks;
        const nk = try r.count(4);
        const keyed = try arena.alloc(u32, nk);
        for (keyed) |*x| x.* = try r.get(u32);
        b.keyed = keyed;
        b.raw = null;
        if ((try r.get(u8)) != 0) {
            const nr = try r.count(4);
            const raw = try arena.alloc(u32, nr);
            for (raw) |*x| x.* = try r.get(u32);
            b.raw = raw;
        }
        const nm = try r.count(40);
        const media = try arena.alloc(format.Media, nm);
        for (media) |*m| m.* = .{ .start = try r.get(u32), .len = try r.get(u32), .digest = (try r.take(32))[0..32].* };
        b.media = media;
        if (r.at != bytes.len) return error.BadBody;
        return b;
    }
};

/// The destination rank 0 reserved for a restore: the Snap it will be kept as, where its rows go, and how many.
pub const RestoreBody = struct {
    snap: u64,
    base: u64,
    at: u32,

    pub const len = 20;

    pub fn encode(b: RestoreBody, out: *[len]u8) void {
        std.mem.writeInt(u64, out[0..8], b.snap, .little);
        std.mem.writeInt(u64, out[8..16], b.base, .little);
        std.mem.writeInt(u32, out[16..20], b.at, .little);
    }

    pub fn decode(bytes: []const u8) !RestoreBody {
        if (bytes.len != len) return error.BadBody;
        return .{ .snap = std.mem.readInt(u64, bytes[0..8], .little), .base = std.mem.readInt(u64, bytes[8..16], .little), .at = std.mem.readInt(u32, bytes[16..20], .little) };
    }
};

/// A guard sub-op's bytes after the SubOp byte.
pub const Guard = struct {
    snap: u64,
    from: u32,

    pub const len = 12;

    pub fn encode(g: Guard, out: *[len]u8) void {
        std.mem.writeInt(u64, out[0..8], g.snap, .little);
        std.mem.writeInt(u32, out[8..12], g.from, .little);
    }

    pub fn decode(bytes: []const u8) !Guard {
        if (bytes.len != len) return error.BadBody;
        return .{ .snap = std.mem.readInt(u64, bytes[0..8], .little), .from = std.mem.readInt(u32, bytes[8..12], .little) };
    }
};

fn put(gpa: std.mem.Allocator, out: *std.ArrayList(u8), comptime T: type, v: T) !void {
    var b: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &b, v, .little);
    try out.appendSlice(gpa, &b);
}

const Rd = struct {
    b: []const u8,
    at: usize = 0,

    fn take(r: *Rd, n: usize) ![]const u8 {
        if (n > r.b.len - r.at) return error.BadBody;
        defer r.at += n;
        return r.b[r.at..][0..n];
    }

    fn get(r: *Rd, comptime T: type) !T {
        return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    /// A u32 count of items of `each` bytes that the rest of the body can hold.
    fn count(r: *Rd, each: usize) !usize {
        const n: usize = try r.get(u32);
        if (n > (r.b.len - r.at) / each) return error.BadBody;
        return n;
    }
};

test "a keep's body reads back as written, pictures and origin included" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const forks = [_]format.Fork{ .{ .name = @splat(3), .at = 2048 }, .{ .name = @splat(4), .at = 4096 } };
    const media = [_]format.Media{.{ .start = 7, .len = 3, .digest = @splat(9) }};
    const b: KeepBody = .{ .snap = 11, .base = 4096, .at = 2048, .shared = true, .group = 5, .origin = .{ .key = @splat(1), .at = 1024 }, .lineage_root = @splat(2), .lineage_part = 6, .forks = &forks, .keyed = &.{ 1, 2, 3, 0x8000_0001 }, .raw = &.{ 4, 5 }, .media = &media };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try b.encode(gpa, &out);
    const g = try KeepBody.decode(arena_state.allocator(), out.items);
    try std.testing.expectEqual(b.snap, g.snap);
    try std.testing.expectEqual(b.base, g.base);
    try std.testing.expectEqual(b.at, g.at);
    try std.testing.expect(g.shared);
    try std.testing.expectEqual(b.group, g.group);
    try std.testing.expectEqual(b.origin.?.at, g.origin.?.at);
    try std.testing.expectEqualSlices(u8, &b.lineage_root, &g.lineage_root);
    try std.testing.expectEqual(@as(?u32, 6), g.lineage_part);
    try std.testing.expectEqual(@as(usize, 2), g.forks.len);
    try std.testing.expectEqual(@as(u32, 4096), g.forks[1].at);
    try std.testing.expectEqualSlices(u32, b.keyed, g.keyed);
    try std.testing.expectEqualSlices(u32, b.raw.?, g.raw.?);
    try std.testing.expectEqual(@as(u32, 7), g.media[0].start);
    // a body with nothing optional
    var out2: std.ArrayList(u8) = .empty;
    defer out2.deinit(gpa);
    const plain: KeepBody = .{ .snap = 1, .base = 0, .at = 2048, .shared = false, .group = 0, .origin = null, .lineage_root = @splat(0), .lineage_part = null, .forks = &.{}, .keyed = &.{ 1, 2 }, .raw = null, .media = &.{} };
    try plain.encode(gpa, &out2);
    const h = try KeepBody.decode(arena_state.allocator(), out2.items);
    try std.testing.expect(h.origin == null and h.raw == null and h.lineage_part == null and h.forks.len == 0 and h.media.len == 0);
}

test "control: a body that is short, long or counts more than it holds is refused, never read past its end" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const b: KeepBody = .{ .snap = 1, .base = 0, .at = 2048, .shared = false, .group = 0, .origin = null, .lineage_root = @splat(0), .lineage_part = null, .forks = &.{}, .keyed = &.{ 1, 2 }, .raw = null, .media = &.{} };
    try b.encode(gpa, &out);
    try std.testing.expectError(error.BadBody, KeepBody.decode(arena_state.allocator(), out.items[0 .. out.items.len - 1]));
    try out.append(gpa, 0);
    try std.testing.expectError(error.BadBody, KeepBody.decode(arena_state.allocator(), out.items));
    // the ids' count rewritten to claim far more than the body holds
    out.items.len -= 1;
    const at = 8 + 8 + 4 + 1 + 8 + 1 + 16 + 1 + 4 + 4;
    std.mem.writeInt(u32, out.items[at..][0..4], 0xffff_fff0, .little);
    try std.testing.expectError(error.BadBody, KeepBody.decode(arena_state.allocator(), out.items));
    try std.testing.expectError(error.BadBody, RestoreBody.decode(&.{1}));
    try std.testing.expectError(error.BadBody, Guard.decode(&.{1}));
}
