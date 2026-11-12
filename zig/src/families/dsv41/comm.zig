//! The ranks' host collectives: the start-up agreements, the queue-pair tables, the graph captures' votes and the SSD tier's
//! digests, over the engine's own links (star.zig, a star from rank 0). Every exchange between the Sparks while serving is
//! Rocinante's (rocinante.zig) or the decode rings'; there is no other transport.
const std = @import("std");
const cuda = @import("cuda");
const star_mod = @import("star.zig");

/// Where the ranks' host collectives go: `star` is the engine's own links, opened when the start-up joins them.
pub const Host = struct {
    star: ?star_mod.Star = null,
};

/// One in-place all-gather part: every rank's `bytes` at `buf + rank * bytes`, gathered into `buf` rank after rank (the plan of a
/// gather: the transport's own plans carry the same shape).
pub const InPlace = struct { buf: u64, bytes: usize };

/// TF_DS_TWO_HOP=1 (model.zig sets it, every rank alike; off by default): four nodes cabled as a ring, where node g
/// reaches only its column partner g ^ 2 and its row partner g ^ 1 (no cross links between the two pairs). A world-4
/// exchange then goes between direct partners only (ring2d.zig's rings; this engine's all-gathers are the RDMA rings' and
/// the host collectives', never NCCL's, so only the rings read this switch).
pub var two_hop: bool = false;

pub const Comm = struct {
    rank: u32,
    world: u32,
    host: Host = .{},

    pub fn init(rank: u32, world: u32) Comm {
        return .{ .rank = rank, .world = world };
    }

    pub fn deinit(c: *Comm) void {
        c.* = undefined;
    }

    /// `bytes` at device address `send` from every rank, gathered into `recv` rank after rank, over the links (the stream
    /// drained, the device bytes read, every rank's gathered by rank 0, written into `recv`). Synchronous, for a caller that
    /// synchronizes at once, as all of them do.
    pub fn gatherHost(c: Comm, d: *const cuda.Driver, send: u64, recv: u64, bytes: usize, stream: cuda.Stream) !void {
        const star = c.host.star orelse return error.NoHostLinks;
        return gatherVia(star, DriverMem{ .d = d, .stream = stream }, send, recv, bytes);
    }

    /// `count` bytes at `buf` made, byte by byte, the largest of every rank's, in place, over the links (the stream drained
    /// first): the graph captures' votes.
    pub fn maxHost(c: Comm, d: *const cuda.Driver, buf: u64, count: usize, stream: cuda.Stream) !void {
        const star = c.host.star orelse return error.NoHostLinks;
        return maxVia(star, DriverMem{ .d = d, .stream = stream }, buf, count);
    }
};

/// The device memory a host collective reads and writes: the driver, after the stream is drained.
const DriverMem = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,

    fn settle(m: DriverMem) !void {
        try m.stream.synchronize();
    }

    fn read(m: DriverMem, dst: []u8, src: u64) !void {
        try m.d.check(m.d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "cuMemcpyDtoH");
    }

    fn write(m: DriverMem, dst: u64, src: []const u8) !void {
        try m.d.copyHtoD(@src(), dst, src.ptr, src.len);
    }
};

/// `bytes` at device address `send` from every rank, gathered over `star` into `recv` rank after rank. `mem` is the device
/// memory (settle, read, write): the driver in the engine, plain memory in a test.
fn gatherVia(star: star_mod.Star, mem: anytype, send: u64, recv: u64, bytes: usize) !void {
    if (bytes > star_mod.max_part) return error.BadPart;
    var mine: [star_mod.max_part]u8 = undefined;
    var table: [4 * star_mod.max_part]u8 = undefined;
    const all = table[0 .. star.world * bytes];
    try mem.settle();
    try mem.read(mine[0..bytes], send);
    try star.allGather(mine[0..bytes], all);
    try mem.write(recv, all);
}

/// `count` bytes at device address `buf` made, byte by byte, the largest of every rank's, in place.
fn maxVia(star: star_mod.Star, mem: anytype, buf: u64, count: usize) !void {
    if (count > star_mod.max_part) return error.BadPart;
    var bytes: [star_mod.max_part]u8 = undefined;
    try mem.settle();
    try mem.read(bytes[0..count], buf);
    try star.maxBytes(bytes[0..count]);
    try mem.write(buf, bytes[0..count]);
}

// -- the host collectives ------------------------------------------------------------------------------------------------

/// Plain memory standing for device memory in the tests of gatherVia and maxVia: an address is a pointer.
const PlainMem = struct {
    fn settle(_: PlainMem) !void {}
    fn read(_: PlainMem, dst: []u8, src: u64) !void {
        @memcpy(dst, @as([*]const u8, @ptrFromInt(src))[0..dst.len]);
    }
    fn write(_: PlainMem, dst: u64, src: []const u8) !void {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..src.len], src);
    }
};

test "gatherVia: every rank's bytes land in recv rank after rank on every rank" {
    try star_mod.runRanks(struct {
        fn f(s: star_mod.Star, g: u32) !void {
            var send: [6]u8 = .{ @intCast(g), 1, 2, 3, 4, @intCast(g + 100) };
            var recv: [24]u8 = @splat(0xee);
            try gatherVia(s, PlainMem{}, @intFromPtr(&send), @intFromPtr(&recv), send.len);
            for (0..4) |r| {
                try std.testing.expectEqual(@as(u8, @intCast(r)), recv[r * 6]);
                try std.testing.expectEqual(@as(u8, @intCast(r + 100)), recv[r * 6 + 5]);
            }
        }
    }.f);
}

test "maxVia: the vote of a failed capture reaches every rank, and all-clear stays clear" {
    try star_mod.runRanks(struct {
        fn f(s: star_mod.Star, g: u32) !void {
            // the capture's flag byte: 1 is a failure; rank 2 alone failed
            var flag = [_]u8{if (g == 2) 1 else 0};
            try maxVia(s, PlainMem{}, @intFromPtr(&flag), 1);
            try std.testing.expectEqual(@as(u8, 1), flag[0]);
            var clear = [_]u8{ 0, 0 };
            try maxVia(s, PlainMem{}, @intFromPtr(&clear), 2);
            try std.testing.expectEqualSlices(u8, &.{ 0, 0 }, &clear);
        }
    }.f);
}

test "a host collective on a rank without links fails by name: nothing else could carry it" {
    const c = Comm.init(1, 4);
    const stream: cuda.Stream = .{ .d = undefined, .handle = null };
    const d: *const cuda.Driver = undefined;
    try std.testing.expectError(error.NoHostLinks, c.gatherHost(d, 0x1000, 0x2000, 1, stream));
    try std.testing.expectError(error.NoHostLinks, c.maxHost(d, 0x3000, 1, stream));
}

test "a host word reaches a collective in the collective's stream order: no synchronous copy feeds one, and the copy names its stream" {
    // a synchronous copy from pageable memory can return before its DMA lands, on the legacy stream, which the family's
    // non-blocking streams do not wait for: the collective after it could send a rank's word before it is there
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var dir = std.Io.Dir.cwd().openDir(io, "zig/src/families/dsv41", .{ .iterate = true }) catch |e| {
        std.debug.print("zig/src/families/dsv41: {s} (the tests run from the tree's root)\n", .{@errorName(e)});
        return error.TestUnexpectedResult;
    };
    defer dir.close(io);
    const collectives = [_][]const u8{ "gatherHost(", "maxHost(" };
    var it = dir.iterate();
    var fed: usize = 0;
    var bad = false;
    while (try it.next(io)) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.name, ".zig") or std.mem.endsWith(u8, e.name, "_test.zig")) continue;
        const text = try dir.readFileAlloc(io, e.name, gpa, .limited(1 << 24));
        defer gpa.free(text);
        var lines: std.ArrayList([]const u8) = .empty;
        defer lines.deinit(gpa);
        var li = std.mem.splitScalar(u8, text, '\n');
        while (li.next()) |l| try lines.append(gpa, l);
        var in_test = false;
        for (lines.items, 0..) |l, i| {
            if (std.mem.startsWith(u8, l, "test \"")) in_test = true;
            if (in_test) {
                if (std.mem.eql(u8, l, "}")) in_test = false;
                continue;
            }
            const sync = std.mem.indexOf(u8, l, ".upload(") != null and std.mem.indexOf(u8, l, "fn upload(") == null;
            const async_at = std.mem.indexOf(u8, l, ".uploadAsync(");
            if (!sync and async_at == null) continue;
            // the next three lines' collective, and the stream it runs on (its last argument)
            for (lines.items[i + 1 .. @min(i + 4, lines.items.len)]) |n| {
                const c = for (collectives) |name| {
                    if (std.mem.indexOf(u8, n, name)) |at| break at;
                } else continue;
                if (sync) {
                    std.debug.print("{s}:{d}: a synchronous copy feeds a collective: {s}\n", .{ e.name, i + 1, std.mem.trim(u8, l, " ") });
                    bad = true;
                } else {
                    const close = std.mem.lastIndexOf(u8, n, ");") orelse continue;
                    const comma = std.mem.lastIndexOfScalar(u8, n[c..close], ',') orelse continue;
                    const stream = std.mem.trim(u8, n[c + comma + 1 .. close], " ");
                    const want = try std.fmt.allocPrint(gpa, ", {s}.handle);", .{stream});
                    defer gpa.free(want);
                    if (std.mem.indexOf(u8, l, want) == null) {
                        std.debug.print("{s}:{d}: the copy is not on the collective's stream {s}: {s}\n", .{ e.name, i + 1, stream, std.mem.trim(u8, l, " ") });
                        bad = true;
                    }
                    fed += 1;
                }
                break;
            }
        }
    }
    try std.testing.expect(!bad);
    // the start's agreements that upload a word and gather it: Two's idx-shard and topology words, ranksThatCannot's flag and
    // the rings' queue-pair infos: none may go unseen
    try std.testing.expect(fed >= 4);
}
