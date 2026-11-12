//! tf-dsv41-rdma-bench: the RDMA ring (rdma.zig) across real nodes, one process a rank. Rank 0 listens on
//! `--port` for the other ranks' queue-pair infos and sends every rank the whole table; once all are connected, each
//! rank runs back-to-back gathers at each row count (50 a CUDA graph, replayed, as the Python ring's --gather-bench),
//! the plain gather and the strided gather_into, and checks every float it received. One JSON line a rank and row count.
//!   tf-dsv41-rdma-bench RANK WORLD MASTER_IP FATBIN DEVICES [--port P] [--rows 1,6,16,48] [--width 2560] [--reps 500]
//!                       [--pdl 0|1] [--gid N]
//! --wc: the rings' send and landing data slots from cudaHostAlloc(Mapped | WriteCombined) (TF_DS_RING_WC's memory);
//! --inplace: the timed gathers take the ring's fixed send slot as their source and wait for its previous send before
//! each publish (TF_DS_RING_INPLACE's path), the row set copied there once (the bytes are the same every gather).
//! --roci word|atomic: TF_DS_RING_WORD's arrival signal (each lane's data a plain RDMA WRITE, then its 64-bit
//! generation word as an inline 8-byte WRITE or an ATOMIC fetch-and-add on the same queue pair; the collecting kernel
//! waits on the words in-kernel), against the baseline path when unset: the same bytes, timed here.
//! --mixed K (needs --inplace 1 --wc 1): after the timed loops of the first row count, K exchanges back to back with no host wait between them,
//! from three streams in one fixed order (in-place user 0, in-place user 1, a staged send; events keep the order the same on every rank), each
//! exchange's payload stamped with its index and sender, every received part checked against the stamp it must carry. It is the replay the
//! shortcuts on the in-place publish need: a send slot overwritten while the card still reads it, or a landing slot reused too early, shows as
//! a wrong stamp. One JSON line: rank, K, exchanges checked, bad.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const ring = dsv41.rdma;
const posix = std.posix;

const usage = "usage: tf-dsv41-rdma-bench RANK WORLD MASTER_IP FATBIN DEVICES [--port P] [--rows 1,6,16,48] [--width 2560] [--reps 500] [--pdl 0|1] [--gid N] [--collect rank,peers] [--rounds N] [--stream GBPS] [--wc 0|1] [--inplace 0|1] [--roci word|atomic]\n";

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 6) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const world = try std.fmt.parseInt(u32, args[2], 10);
    const master = try ip4(args[3]);
    var port: u16 = 29_700;
    var rows_text: []const u8 = "1,6,16,48";
    var width: u32 = 2560;
    var reps: u32 = 500;
    var pdl = true;
    var gid: ?u8 = null;
    var wc = false;
    var inplace = false;
    var mixed: u32 = 0;
    var roci: ring.Roci = .off;
    var i: usize = 6;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--port")) port = try std.fmt.parseInt(u16, v, 10) else if (std.mem.eql(u8, k, "--rows")) rows_text = v else if (std.mem.eql(u8, k, "--width")) width = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--reps")) reps = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--pdl")) pdl = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--gid")) gid = try std.fmt.parseInt(u8, v, 10) else if (std.mem.eql(u8, k, "--collect")) collects_text = v else if (std.mem.eql(u8, k, "--rounds")) rounds = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--stream")) stream_gbps = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, k, "--wc")) wc = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--inplace")) inplace = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--mixed")) mixed = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--roci")) {
            if (std.mem.eql(u8, v, "word")) roci = .word else if (std.mem.eql(u8, v, "atomic")) roci = .atomic else {
                std.debug.print("{s}", .{usage});
                return 2;
            }
        } else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    if (width % 4 != 0) return error.WidthNotFloat4;
    var rows: std.ArrayList(u32) = .empty;
    var it = std.mem.splitScalar(u8, rows_text, ',');
    while (it.next()) |t| try rows.append(a, try std.fmt.parseInt(u32, t, 10));
    var devices: std.ArrayList([]const u8) = .empty;
    var dit = std.mem.splitScalar(u8, args[5], ',');
    while (dit.next()) |t| try devices.append(a, t);
    if (inplace and !wc) {
        std.debug.print("--inplace needs --wc 1 (its send slot is the write-combined one)\n", .{});
        return 2;
    }
    if (mixed > 0 and !inplace) {
        std.debug.print("--mixed needs --inplace 1 --wc 1 (two in-place users and a staged send on one ring)\n", .{});
        return 2;
    }
    const per_graph = 50;
    const loops = @max(1, reps / per_graph);
    var most: u32 = 1;
    for (rows.items) |n| most = @max(most, n);

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const image = try std.Io.Dir.cwd().readFileAlloc(io, args[4], a, .limited(1 << 26));
    var module = try cuda.Module.load(&driver, image);
    defer module.unload();
    const r = try ring.Ring.create(init.gpa, &driver, try ring.Kernels.load(module), devices.items, rank, world, .{ .max_bytes = std.mem.alignForward(usize, @as(usize, most) * width * 4, 64), .pdl = pdl, .gid_index = gid });
    if (inplace and !kernels.hasInplace()) {
        std.debug.print("--inplace: the fatbin has no dsv41_rdma_publish / dsv41_rdma_await_send\n", .{});
        return 2;
    }
    if (roci.on() and !kernels.hasRoci()) {
        std.debug.print("--roci: the fatbin has no dsv41_rdma_collect_roci / _into_roci / _peers_roci\n", .{});
        return 2;
    }
    const r = try ring.Ring.create(init.gpa, &driver, kernels, devices.items, rank, world, .{ .max_bytes = std.mem.alignForward(usize, @as(usize, most) * width * 4, 64), .pdl = pdl, .gid_index = gid, .wc = wc, .inplace = inplace, .roci = roci, .roci_deadline_ns = 120 * std.time.ns_per_s });
    defer r.destroy(init.gpa);

    // every rank's info(p) for every p, gathered by rank 0 and handed back whole; then a barrier once all are connected
    const row_bytes = @sizeOf(ring.Info) * world;
    const mine = try a.alloc(u8, row_bytes);
    for (0..world) |p| @memcpy(mine[p * @sizeOf(ring.Info) ..][0..@sizeOf(ring.Info)], std.mem.asBytes(&r.info(@intCast(p))));
    const table = try a.alloc(u8, row_bytes * world);
    try exchange(io, rank, world, master, port, mine, table);
    var remote: [ring.max_ranks]ring.Info = undefined;
    for (0..world) |q| remote[q] = std.mem.bytesToValue(ring.Info, table[q * row_bytes + rank * @sizeOf(ring.Info) ..][0..@sizeOf(ring.Info)]);
    try r.connect(remote[0..world]);
    var one = [_]u8{1};
    const all = try a.alloc(u8, world);
    try exchange(io, rank, world, master, port, &one, all);
    try r.start();

    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    for (rows.items) |n_rows| {
        const n: usize = @as(usize, n_rows) * width; // floats a rank
        const host = try a.alloc(f32, n * world);
        for (0..n) |j| host[j] = value(rank, j);
        var send = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(host[0..n]));
        defer send.free();
        var recv = try cuda.DeviceBuffer.alloc(&driver, n * world * 4);
        defer recv.free();
        var into = try cuda.DeviceBuffer.alloc(&driver, n * world * 4); // [rows][world x width]: rank q at columns q x width
        defer into.free();
        const n4: u32 = @intCast(n / 4);
        // --inplace: the producer writes this stream's fixed send slot; the bench copies the row set there once (the
        // loops send the same bytes), and each gather waits for the previous send as a producer would
        const send_src: u64 = if (inplace) blk: {
            const slot = (try r.sendSlotDev(stream, n * 4)) orelse return error.BadInplace;
            try driver.check(driver.api.cuMemcpyDtoDAsync_v2(slot, send.ptr, n * 4, stream.handle), "cuMemcpyDtoDAsync");
            try stream.synchronize();
            break :blk slot;
        } else send.ptr;
        var dsts: [ring.max_ranks]ring.Ring.Dst = undefined;
        for (0..world) |q| dsts[q] = .{ .ptr = into.ptr + @as(u64, q) * width * 4, .row4 = width / 4, .stride4 = world * width / 4 };
        for (0..20) |_| try r.gather(stream, send.ptr, n4, recv.ptr);
        try stream.synchronize();
        // 50 gathers captured in one graph and replayed, as the Python ring's --gather-bench times them
        try cuda.graph.beginCapture(stream, .thread_local);
        for (0..per_graph) |_| try r.gather(stream, send.ptr, n4, recv.ptr);
        var g1 = try cuda.graph.endCapture(stream);
        defer g1.deinit();
        var e1 = try g1.instantiate();
        defer e1.deinit();
        try e1.launchOn(stream);
        try stream.synchronize();
        var t0 = std.Io.Timestamp.now(io, .awake);
        for (0..loops) |_| try e1.launchOn(stream);
        try stream.synchronize();
        const us_gather = micros(t0, io) / @as(f64, @floatFromInt(loops * per_graph));
        try recv.download(0, std.mem.sliceAsBytes(host));
        var bad: usize = 0;
        for (0..world) |q| for (0..n) |j| {
            if (host[q * n + j] != value(@intCast(q), j)) bad += 1;
            for (0..20) |_| {
                if (inplace) _ = try r.prepareSend(stream);
                try r.gather(stream, send_src, n4, recv.ptr);
            }
            try stream.synchronize();
            // 50 gathers captured in one graph and replayed, as the Python ring's --gather-bench times them
            try cuda.graph.beginCapture(stream, .thread_local);
            for (0..per_graph) |_| {
                if (inplace) _ = try r.prepareSend(stream);
                try r.gather(stream, send_src, n4, recv.ptr);
            }
            var g1 = try cuda.graph.endCapture(stream);
            defer g1.deinit();
            var e1 = try g1.instantiate();
            defer e1.deinit();
            for (0..20) |_| {
                if (inplace) _ = try r.prepareSend(stream);
                try r.gatherInto(stream, send_src, n_rows, width / 4, dsts[0..world]);
            }
            try stream.synchronize();
            try cuda.graph.beginCapture(stream, .thread_local);
            for (0..per_graph) |_| {
                if (inplace) _ = try r.prepareSend(stream);
                try r.gatherInto(stream, send_src, n_rows, width / 4, dsts[0..world]);
            }
            var g2 = try cuda.graph.endCapture(stream);
            defer g2.deinit();
            var e2 = try g2.instantiate();
            defer e2.deinit();
            try out.interface.print("{{\"rank\": {d}, \"world\": {d}, \"rows\": {d}, \"bytes\": {d}, \"gathers\": {d}, \"pdl\": {d}, \"us_gather\": {d:.2}, \"us_into\": {d:.2}, \"bad\": {d}, \"bad_into\": {d}, \"collect\": \"{s}\", \"wc\": {d}, \"inplace\": {d}, \"roci\": \"{s}\", \"round\": {d}, \"stream_gbps\": {d:.0}, \"stream_ran_gbps\": [{d:.1}, {d:.1}], \"stream_outlasted\": [{}, {}], \"failure\": \"{s}\"}}\n", .{ rank, world, n_rows, n * 4, loops * per_graph, @intFromBool(pdl), us_gather, us_into, bad, bad_into, if (peers) "peers" else "rank", @intFromBool(wc), @intFromBool(inplace), if (roci.on()) roci.signalName() else "", round, stream_gbps, ran_gather.gbps, ran_into.gbps, ran_gather.outlasted, ran_into.outlasted, r.failure() orelse "" });
            try out.interface.flush();
        };
        if (mixed > 0 and n_rows == rows.items[0]) {
            const bad = try mixedReplay(init.gpa, &driver, r, stream, rank, world, n, mixed);
            try out.interface.print("{{\"rank\": {d}, \"mixed\": {d}, \"exchanges_checked\": {d}, \"bad\": {d}}}\n", .{ rank, mixed, mixed * world, bad });
            try out.interface.flush();
            if (bad != 0) return 1;
        }
    }
    return 0;
}

fn stamp(q: u32, k: u32, j: usize) f32 {
    return @floatFromInt((k + 1) * 100_000 + q * 1_000 + @as(u32, @intCast(j % 1_000)));
}

/// `count` exchanges of `n` floats back to back from three streams in one fixed order (k % 3: in-place user 0 on `stream_a`, in-place user 1 on
/// a second stream, a staged send on a third), ordered by events so every rank launches the same sequence; no host wait until all are queued.
/// Every payload carries its exchange index and sender; the received parts are checked against what each sender stamped. Returns the parts wrong.
fn mixedReplay(gpa: std.mem.Allocator, driver: *const cuda.Driver, r: *ring.Ring, stream_a: cuda.Stream, rank: u32, world: u32, n: usize, count: u32) !usize {
    var sb = try cuda.Stream.init(driver, true);
    defer sb.deinit();
    var sc = try cuda.Stream.init(driver, true);
    defer sc.deinit();
    const streams = [3]cuda.Stream{ stream_a, sb, sc };
    const n4: u32 = @intCast(n / 4);
    const srcs = try gpa.alloc(cuda.DeviceBuffer, count);
    defer gpa.free(srcs);
    const recvs = try gpa.alloc(cuda.DeviceBuffer, count);
    defer gpa.free(recvs);
    const evs = try gpa.alloc(cuda.Event, count);
    defer gpa.free(evs);
    const host = try gpa.alloc(f32, n * world);
    defer gpa.free(host);
    var made: u32 = 0;
    defer for (0..made) |i| {
        srcs[i].free();
        recvs[i].free();
        evs[i].deinit();
    };
    while (made < count) : (made += 1) {
        for (0..n) |j| host[j] = stamp(rank, made, j);
        srcs[made] = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(host[0..n]));
        recvs[made] = try cuda.DeviceBuffer.alloc(driver, n * world * 4);
        try driver.check(driver.api.cuMemsetD8_v2(recvs[made].ptr, 0xff, n * world * 4), "cuMemsetD8");
        evs[made] = try cuda.Event.init(driver, false);
    }
    // the two in-place users take their slots first, in the same order on every rank
    const slot_a = (try r.sendSlotDev(streams[0], n * 4)) orelse return error.BadInplace;
    const slot_b = (try r.sendSlotDev(streams[1], n * 4)) orelse return error.BadInplace;
    for (0..count) |k| {
        const which = k % 3;
        const s = streams[which];
        if (k > 0) try s.wait(evs[k - 1]);
        if (which == 2) {
            try r.gather(s, srcs[k].ptr, n4, recvs[k].ptr);
        } else {
            const slot = if (which == 0) slot_a else slot_b;
            _ = try r.prepareSend(s);
            try driver.check(driver.api.cuMemcpyDtoDAsync_v2(slot, srcs[k].ptr, n * 4, s.handle), "cuMemcpyDtoDAsync");
            try r.gather(s, slot, n4, recvs[k].ptr);
        }
        try evs[k].record(s);
    }
    for (streams) |s| try s.synchronize();
    var bad: usize = 0;
    for (0..count) |k| {
        try recvs[k].download(0, std.mem.sliceAsBytes(host));
        for (0..world) |q| for (0..n) |j| {
            if (host[q * n + j] != stamp(@intCast(q), @intCast(k), j)) {
                bad += 1;
                break;
            }
        };
    }
    return bad;
}

/// The float rank `q` sends at index `j`: exact in f32 (below 2^24).
fn value(q: u32, j: usize) f32 {
    return @floatFromInt(@as(u32, q) * 1_000_000 + @as(u32, @intCast(j % 1_000_000)));
}

fn micros(t0: std.Io.Timestamp, io: std.Io) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e3;
}

fn ip4(text: []const u8) !u32 {
    var out: u32 = 0;
    var parts: u32 = 0;
    var it = std.mem.splitScalar(u8, text, '.');
    while (it.next()) |t| : (parts += 1) out = out << 8 | try std.fmt.parseInt(u8, t, 10);
    if (parts != 4) return error.BadAddress;
    return out;
}

fn writeAll(fd: posix.socket_t, bytes: []const u8) !void {
    var sent: usize = 0;
    while (sent < bytes.len) {
        const rc = posix.system.write(fd, bytes[sent..].ptr, bytes.len - sent);
        switch (posix.errno(rc)) {
            .SUCCESS => sent += @intCast(rc),
            .INTR => {},
            else => return error.WriteFailed,
        }
    }
}

fn readAll(fd: posix.socket_t, buf: []u8) !void {
    var got: usize = 0;
    while (got < buf.len) {
        const rc = posix.system.read(fd, buf[got..].ptr, buf.len - got);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.PeerClosed;
                got += @intCast(rc);
            },
            .INTR => {},
            else => return error.ReadFailed,
        }
    }
}

fn socketTcp() !posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    return @intCast(rc);
}

/// Rank 0 collects every rank's `mine` (all the same length) into `table` in rank order and sends it to every rank; the
/// others send theirs and read the table back. Ranks other than 0 retry their connect for a minute.
fn exchange(io: std.Io, rank: u32, world: u32, master: u32, port: u16, mine: []const u8, table: []u8) !void {
    var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, if (rank == 0) 0 else master) };
    if (rank == 0) {
        const fd = try socketTcp();
        defer _ = posix.system.close(fd);
        const yes: c_int = 1;
        try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&yes));
        if (posix.errno(posix.system.bind(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(fd, 16)) != .SUCCESS) return error.ListenFailed;
        @memcpy(table[0..mine.len], mine);
        var conns: [ring.max_ranks]posix.socket_t = undefined;
        for (1..world) |k| {
            const rc = posix.system.accept(fd, null, null);
            if (posix.errno(rc) != .SUCCESS) return error.AcceptFailed;
            conns[k] = @intCast(rc);
            var who: [4]u8 = undefined;
            try readAll(conns[k], &who);
            const q = std.mem.readInt(u32, &who, .little);
            if (q == 0 or q >= world) return error.BadRank;
            try readAll(conns[k], table[q * mine.len ..][0..mine.len]);
        }
        for (1..world) |k| {
            try writeAll(conns[k], table);
            _ = posix.system.close(conns[k]);
        }
        return;
    }
    var fd: posix.socket_t = undefined;
    var tries: u32 = 0;
    while (true) : (tries += 1) {
        const s = try socketTcp();
        if (posix.errno(posix.system.connect(s, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) {
            fd = s;
            break;
        }
        _ = posix.system.close(s);
        if (tries > 600) return error.ConnectFailed;
        std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
    }
    defer _ = posix.system.close(fd);
    var who: [4]u8 = undefined;
    std.mem.writeInt(u32, &who, rank, .little);
    try writeAll(fd, &who);
    try writeAll(fd, mine);
    try readAll(fd, table);
}

test "an IPv4 address in dotted form" {
    try std.testing.expectEqual(@as(u32, 0xc0000201), try ip4("192.0.2.1"));
    try std.testing.expectError(error.BadAddress, ip4("10.200.10"));
}
