//! The ranks' host collectives: the start-up agreements, the queue-pair tables, the graph captures' votes and the SSD tier's
//! digests, over the engine's own links (star.zig, a star from rank 0). Every exchange between the Sparks while serving is
//! Rocinante's (rocinante.zig) or the decode rings'; there is no other transport.
const std = @import("std");
const cuda = @import("cuda");
const Link = @import("link.zig").Link;

/// Where the ranks' host collectives go: `star` is the engine's own links, opened when the start-up joins them.
/// One in-place all-gather part: every rank's `bytes` at `buf + rank * bytes`, gathered into `buf` rank after rank (the plan of a
/// gather: the transport's own plans carry the same shape).
pub const InPlace = struct { buf: u64, bytes: usize };

/// TF_DS_TWO_HOP=1 (model.zig sets it, every rank alike; off by default): four nodes cabled as a ring, where node g
/// reaches only its column partner g ^ 2 and its row partner g ^ 1 (no cross links between the two pairs). A world-4
/// exchange then goes between direct partners only (ring2d.zig's rings; this engine's all-gathers are the RDMA rings' and
pub var two_hop: bool = false;

pub const Comm = struct {
    rank: u32,
    world: u32,

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
        return gatherVia(star, DriverMem{ .d = d, .stream = stream }, send, recv, bytes);
    }

    /// `count` bytes at `buf` made, byte by byte, the largest of every rank's, in place, over the links (the stream drained
    /// first): the graph captures' votes.
    pub fn maxHost(c: Comm, d: *const cuda.Driver, buf: u64, count: usize, stream: cuda.Stream) !void {
        return maxVia(star, DriverMem{ .d = d, .stream = stream }, buf, count);
    }

    /// `count` elements of `dtype` to rank `peer` and `rcount` from it, in one NCCL group (point to point).
    pub fn exchange(c: Comm, send: u64, count: usize, recv: u64, rcount: usize, dtype: nccl.DataType, peer: u32, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        const s = c.lib.api.ncclSend(send, count, dtype, @intCast(peer), c.comm, stream.handle);
        const r = c.lib.api.ncclRecv(recv, rcount, dtype, @intCast(peer), c.comm, stream.handle);
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(s, "ncclSend");
        try c.lib.check(r, "ncclRecv");
    }
};

test "two hops on a ring of four: every rank ends with every slice in rank order, partners only" {
    // four ranks' recv buffers, slot q holding rank q's slice once it arrived (0: not yet)
    var have: [4][4]u8 = @splat(@splat(0));
    for (0..4) |g| have[g][g] = @intCast(g + 1);
    for (0..2) |step| {
        var next = have;
        for (0..4) |g| {
            const h = hops(@intCast(g))[step];
            try std.testing.expect(h.peer == g ^ 1 or h.peer == g ^ 2); // a direct partner, never g ^ 3
            const back = hops(h.peer)[step];
            try std.testing.expectEqual(@as(u32, @intCast(g)), back.peer);
            try std.testing.expectEqual(h.n, back.n);
            // what g posts to receive is, in order, what its peer posts to send, and it already holds them
            for (0..h.n) |i| {
                try std.testing.expectEqual(h.recv[i], back.send[i]);
                try std.testing.expect(have[h.peer][back.send[i]] != 0);
                next[g][h.recv[i]] = have[h.peer][back.send[i]];
// -- the host collectives ------------------------------------------------------------------------------------------------
test "a host collective on a rank without links fails by name: nothing else could carry it" {
    const c = Comm.init(1, 4);
    try std.testing.expectError(error.NoHostLinks, c.gatherHost(d, 0x1000, 0x2000, 1, stream));
    try std.testing.expectError(error.NoHostLinks, c.maxHost(d, 0x3000, 1, stream));
            }
        }
    }
    for (0..4) |g| for (0..4) |q| try std.testing.expectEqual(@as(u8, @intCast(q + 1)), have[g][q]);
}
