//! The 2D split's decode-size exchanges over RDMA rings (DESIGN-zig-tp4.md section 4; the Python lane's Comm2D with
//! TF_DS_2D_INTO and TF_DS_2D_GROUPS): quarters over the four nodes' ring, cat_rank over the column pair's (node g and
//! g ^ 2, the same TP2 rank of each pair), the pair gather over the row pair's (g and g ^ 1, TP2 inside a pair), each
//! part written straight into its place in the caller's layout (no [4, ...] buffer, no copies after it). The rings
//! open on the fabric's verbs devices; their queue-pair infos go round the four nodes over the host links. Parts
//! travel as fp32 words (16-bit parts reinterpreted), so a part's row must be a whole number of 16 bytes. A part
//! larger than the rings take returns error.TooLarge, and the caller ends the exchange by name (prompt2d.zig).
const std = @import("std");
const cuda = @import("cuda");
const rdma = @import("rdma.zig");
const Comm = @import("comm.zig").Comm;

pub const Rings = struct {
    g: u32,
    r: u32,
    p: u32,
    all4: *rdma.Ring,
    col: *rdma.Ring,
    pair: ?*rdma.Ring, // null: none opened (TF_DS_DRAFT_4WAY: the drafter, its only user, gathers four ways)

    /// The three rings of node `g` (of 4; without the row pair's when `pair_ring` is false, every node alike): created,
    /// their infos exchanged over `comm` (world 4, rank i serving node ord[i]: TF_DS_2D_ORDER), connected and
    /// started on every node before any returns. On a ring without a switch each ring routes its peers over the devices
    /// on their cables, and the four nodes' ring relays a node's part to the node across through the two between
    /// (rdma.zig).
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, g: u32, ord: [4]u32, s: rdma.Settings, comm: *const Comm, stream: cuda.Stream, pair_ring: bool) !Rings {
        if (g >= 4 or comm.world != 4 or comm.rank >= 4) return error.NotFourNodes;
        var seen: [4]bool = @splat(false);
        for (ord) |x| {
            if (x >= 4 or seen[x]) return error.BadOrder;
            seen[x] = true;
        }
        if (ord[comm.rank] != g) return error.NotFourNodes;
        const r = g % 2;
        const p = g / 2;
        var rank_of: [4]u32 = undefined; // node -> rank
        for (ord, 0..) |x, i| rank_of[x] = @intCast(i);
        // TF_DS_RDMA_CPUS=a,b,c: the three proxies' CPUs (the four nodes' ring's, the column pair's, the row pair's)
        var cpus: [3]?u16 = .{ null, null, null };
        if (std.c.getenv("TF_DS_RDMA_CPUS")) |v| {
            var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
            for (&cpus) |*c| c.* = if (it.next()) |t| std.fmt.parseInt(u16, std.mem.trim(u8, t, " "), 10) catch null else null;
        }
        var s4 = s;
        s4.cpu = cpus[0];
        var sc = s;
        sc.cpu = cpus[1];
        var sr = s;
        sr.cpu = cpus[2];
        const all4 = try rdma.Ring.create(gpa, d, k, devices, g, 4, s4);
        errdefer all4.destroy(gpa);
        const col = try rdma.Ring.create(gpa, d, k, devices, p, 2, sc); // members (r, 0), (r, 1): ring rank = pair
        errdefer col.destroy(gpa);
        const pair: ?*rdma.Ring = if (pair_ring) try rdma.Ring.create(gpa, d, k, devices, r, 2, sr) else null; // members (0, p), (1, p): ring rank = TP2 rank
        errdefer if (pair) |x| x.destroy(gpa);
        // this node's infos: the four-node ring's toward each node, the column pair's and the row pair's toward each member
        const per = @sizeOf(rdma.Info);
        var mine: [8]rdma.Info = undefined;
        for (0..4) |q| mine[q] = all4.info(@intCast(q));
        for (0..2) |q| mine[4 + q] = col.info(@intCast(q));
        for (0..2) |q| mine[6 + q] = if (pair) |x| x.info(@intCast(q)) else std.mem.zeroes(rdma.Info);
        var dev = try cuda.DeviceBuffer.alloc(d, 5 * 8 * per);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 8 * per, 8 * per, .u8, stream);
        try comm.gatherHost(d, dev.ptr, dev.ptr + 8 * per, 8 * per, stream);
        try stream.synchronize();
        var all: [4][8]rdma.Info = undefined; // by rank
        try dev.download(8 * per, std.mem.sliceAsBytes(&all));
        var remote4: [4]rdma.Info = undefined;
        for (0..4) |q| remote4[q] = all[rank_of[q]][g];
        try all4.connect(&remote4);
        var remote_col: [2]rdma.Info = undefined;
        for (0..2) |pp| remote_col[pp] = all[rank_of[r + 2 * pp]][4 + p];
        try col.connect(&remote_col);
        if (pair) |x| {
            var remote_pair: [2]rdma.Info = undefined;
            for (0..2) |rr| remote_pair[rr] = all[rank_of[2 * p + rr]][6 + r];
            try x.connect(&remote_pair);
        }
        // every queue pair ready on every node before any ring sends: one more all-gather as the barrier
        try comm.gatherHost(d, dev.ptr, dev.ptr + 8 * per, 1, stream);
        try stream.synchronize();
        try all4.start();
        try col.start();
        if (pair) |x| try x.start();
        const rt = all4.routes();
        if (pair) |x| {
            std.log.info("2D RDMA rings on node {d}: lanes to nodes 0-3 {d},{d},{d},{d} (0: relayed), {d} relay(s); column {d} lane(s), row {d}", .{ g, rt.lanes[0], rt.lanes[1], rt.lanes[2], rt.lanes[3], rt.relays, col.routes().lanes[1 - p], x.routes().lanes[1 - r] });
        } else std.log.info("2D RDMA rings on node {d}: lanes to nodes 0-3 {d},{d},{d},{d} (0: relayed), {d} relay(s); column {d} lane(s), no row pair ring", .{ g, rt.lanes[0], rt.lanes[1], rt.lanes[2], rt.lanes[3], rt.relays, col.routes().lanes[1 - p] });
        return .{ .g = g, .r = r, .p = p, .all4 = all4, .col = col, .pair = pair };
    }

    pub fn close(rs: *Rings, gpa: std.mem.Allocator) void {
        for ([_]?*rdma.Ring{ rs.pair, rs.col, rs.all4 }) |y| if (y) |x| {
            x.stop();
            x.destroy(gpa);
        };
        rs.* = undefined;
    }

    fn float4s(bytes: usize) !u32 {
        if (bytes % 16 != 0) return error.NotFloat4;
        return @intCast(bytes / 16);
    }

    /// quarters (Comm2D.quarters): `src` [rows, w[p]] (this node's columns of its TP2 rank's part) -> `dst`
    /// [2, rows, w0 + w1]: each TP2 rank's whole part, pair 0's columns then pair 1's.
    pub fn quarters(rs: *const Rings, stream: cuda.Stream, src: u64, dst: u64, rows: usize, w: [2]usize, esize: usize) !void {
        const dsts = try quartersDsts(dst, rows, 0, w, esize);
        try rs.all4.gatherInto(stream, src, @intCast(rows), try float4s(w[rs.p] * esize), &dsts);
    }

    /// quarters of rows r0 .. r0 + rows - 1 of an exchange of n_total rows: `src` this node's part of those rows only
    /// ([rows, w[p]]), `dst` the whole [2, n_total, w0 + w1], where one quarters call of every row lands them.
    pub fn quartersRows(rs: *const Rings, stream: cuda.Stream, src: u64, dst: u64, n_total: usize, r0: usize, rows: usize, w: [2]usize, esize: usize) !void {
        const dsts = try quartersDsts(dst, n_total, r0, w, esize);
        try rs.all4.gatherInto(stream, src, @intCast(rows), try float4s(w[rs.p] * esize), &dsts);
    }

    /// One part of `rows` rows of 16 bytes (four fp32) from every node into `dst` [4, rows, 16 bytes], the parts in
    /// `slot_of[node]` order (the communicator's rank order, which Markov's pick reads): the four nodes' ring, no copy after
    /// it.
    pub fn nodesGather(rs: *const Rings, stream: cuda.Stream, src: u64, dst: u64, rows: usize, slot_of: [4]u32) !void {
        const dsts = nodesDsts(dst, rows, slot_of);
        try rs.all4.gatherInto(stream, src, @intCast(rows), 1, &dsts);
    }

    /// The bytes a part of one exchange takes at most (each ring's slot).
    pub fn slotBytes(rs: *const Rings) usize {
        return rs.all4.s.max_bytes;
    }

    /// cat_rank (Comm2D.cat_rank): `src` [rows, w[p]] -> `dst` [rows, w0 + w1], pair 0's part then pair 1's, from the
    /// column partner.
    pub fn catRank(rs: *const Rings, stream: cuda.Stream, src: u64, dst: u64, rows: usize, w: [2]usize, esize: usize) !void {
        const row = (w[0] + w[1]) * esize;
        var dsts: [2]rdma.Ring.Dst = undefined;
        for (0..2) |pp| dsts[pp] = .{ .ptr = dst + pp * w[0] * esize, .row4 = try float4s(w[pp] * esize), .stride4 = try float4s(row) };
        try rs.col.gatherInto(stream, src, @intCast(rows), try float4s(w[rs.p] * esize), &dsts);
    }

    /// The pair gather (Comm2D.gather: TP2 inside a pair): `src` [rows, width] -> `dst` [2, rows, width] in TP2 rank order;
    /// error.NoPairRing when the row pair's ring was not opened.
    pub fn pairGather(rs: *const Rings, stream: cuda.Stream, src: u64, dst: u64, rows: usize, width: usize, esize: usize) !void {
        const pair = rs.pair orelse return error.NoPairRing;
        var dsts: [2]rdma.Ring.Dst = undefined;
        for (0..2) |rr| dsts[rr] = .{ .ptr = dst + rr * rows * width * esize, .row4 = try float4s(width * esize), .stride4 = try float4s(width * esize) };
        try pair.gatherInto(stream, src, @intCast(rows), try float4s(width * esize), &dsts);
    }
};

/// Where each node's part of quarters lands: node q (TP2 rank q % 2, pair q / 2) at its rank's [n_total, w0 + w1]
/// plane, row r0, its pair's columns.
pub fn quartersDsts(dst: u64, n_total: usize, r0: usize, w: [2]usize, esize: usize) ![4]rdma.Ring.Dst {
    const row = (w[0] + w[1]) * esize;
    var dsts: [4]rdma.Ring.Dst = undefined;
    for (0..4) |q| dsts[q] = .{ .ptr = dst + (q % 2) * n_total * row + r0 * row + (q / 2) * w[0] * esize, .row4 = try Rings.float4s(w[q / 2] * esize), .stride4 = try Rings.float4s(row) };
    return dsts;
}

/// Where each node's part of nodesGather lands: node q's `rows` rows of one float4 at slot `slot_of[q]` of `dst`.
pub fn nodesDsts(dst: u64, rows: usize, slot_of: [4]u32) [4]rdma.Ring.Dst {
    var dsts: [4]rdma.Ring.Dst = undefined;
    for (0..4) |q| dsts[q] = .{ .ptr = dst + @as(u64, slot_of[q]) * rows * 16, .row4 = 1, .stride4 = 1 };
    return dsts;
}

/// The rows of one exchange whose widest part (`row_bytes` a row) fits a slot of `slot` bytes; a row wider than a slot:
/// error.TooLarge.
pub fn rowsPerSlot(row_bytes: usize, slot: usize) !usize {
    if (row_bytes == 0 or row_bytes > slot) return error.TooLarge;
    return slot / row_bytes;
}

/// Where node q's word j of part `kind` lands, for the checks: (TP2 rank, pair) of the node owning it.
pub fn owner(q: u32) [2]u32 {
    return .{ q % 2, q / 2 };
}

test "a 2D node's ring ranks and members" {
    // node 3 = TP2 rank 1 of pair 1: the column pair is nodes 1 and 3 (ring rank 1), the row pair nodes 2 and 3 (rank 1)
    try std.testing.expectEqual([2]u32{ 1, 1 }, owner(3));
    try std.testing.expectEqual([2]u32{ 0, 1 }, owner(2));
    try std.testing.expectError(error.NotFloat4, Rings.float4s(10));
    try std.testing.expectEqual(@as(u32, 640), try Rings.float4s(2560 * 4));
}

test "quartersRows lands rows [r0, r0 + n) where one quarters call lands them" {
    // a host model of gatherInto's destinations: float4 j of row i of node q's part lands at ptr + i * stride4 + j
    const Model = struct {
        fn at(d: rdma.Ring.Dst, i: usize, j: usize) u64 {
            return d.ptr + i * d.stride4 * 16 + j * 16;
        }
    };
    const dst: u64 = 1 << 32;
    // the head's quarters (both pair widths) and the partials, fp32
    for ([_][2]usize{ .{ 32384, 32256 }, .{ 2560, 2560 } }) |w| for ([_]usize{ 35, 48 }) |n_total| {
        const whole = try quartersDsts(dst, n_total, 0, w, 4);
        for ([_][2]usize{ .{ 0, 34 }, .{ 34, n_total - 34 } }) |blk| {
            const part = try quartersDsts(dst, n_total, blk[0], w, 4);
            for (0..4) |q| {
                try std.testing.expectEqual(whole[q].row4, part[q].row4);
                try std.testing.expectEqual(whole[q].stride4, part[q].stride4);
                for ([_]usize{ 0, blk[1] - 1 }) |i| for ([_]usize{ 0, whole[q].row4 - 1 }) |j| {
                    try std.testing.expectEqual(Model.at(whole[q], blk[0] + i, j), Model.at(part[q], i, j));
                };
            }
        }
    };
    // a rank's plane holds both pairs' columns: rank 1, pair 1 starts its row past pair 0's columns of rank 1's plane
    const d3 = (try quartersDsts(dst, 48, 34, .{ 32384, 32256 }, 4))[3];
    try std.testing.expectEqual(dst + 48 * (32384 + 32256) * 4 + 34 * (32384 + 32256) * 4 + 32384 * 4, d3.ptr);
}

test "a decode gather's rows per slot" {
    try std.testing.expectEqual(@as(usize, 34), try rowsPerSlot(129536, 4456448)); // the head's pair-0 row
    try std.testing.expectEqual(@as(usize, 435), try rowsPerSlot(10240, 4456448)); // a partial's row: 80 rows in one
    try std.testing.expectError(error.TooLarge, rowsPerSlot(4456449, 4456448));
    try std.testing.expectError(error.TooLarge, rowsPerSlot(0, 4456448));
}

test "without the row pair's ring the pair gather says so" {
    // (TF_DS_DRAFT_4WAY opens none on any node: its only user, the TP2 drafter, does not run)
    var rs: Rings = undefined;
    rs.pair = null;
    try std.testing.expectError(error.NoPairRing, rs.pairGather(undefined, 0, 0, 5, 5120, 4));
}

test "quarters is quartersRows of every row" {
    const src = @embedFile("ring2d.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn quarters(").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "const dsts = try quartersDsts(dst, rows, 0, w, esize);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "try rs.all4.gatherInto(stream, src, @intCast(rows), try float4s(w[rs.p] * esize), &dsts);"));
    const rr = std.mem.indexOf(u8, src, "    pub fn quartersRows(").?;
    const rbody = src[rr .. rr + std.mem.indexOf(u8, src[rr..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, rbody, "const dsts = try quartersDsts(dst, n_total, r0, w, esize);"));
}

test "nodesDsts: a node's rows land at its communicator rank's slot, contiguous, whatever the ring order" {
    const dst: u64 = 1 << 32;
    // the identity order, then one where nodes 1 and 2 swap (TF_DS_2D_ORDER 0,2,1,3)
    for ([_][4]u32{ .{ 0, 1, 2, 3 }, .{ 0, 2, 1, 3 }, .{ 3, 2, 1, 0 } }) |order| {
        const d = nodesDsts(dst, 5, order);
        var seen: [4]bool = @splat(false);
        for (0..4) |q| {
            try std.testing.expectEqual(dst + order[q] * 5 * 16, d[q].ptr);
            try std.testing.expectEqual(@as(u32, 1), d[q].row4);
            try std.testing.expectEqual(@as(u32, 1), d[q].stride4);
            seen[order[q]] = true;
        }
        for (seen) |x| try std.testing.expect(x);
    }
    // control: the order matters, the slots of two different orders differ
    try std.testing.expect(nodesDsts(dst, 5, .{ 0, 1, 2, 3 })[1].ptr != nodesDsts(dst, 5, .{ 0, 2, 1, 3 })[1].ptr);
}
