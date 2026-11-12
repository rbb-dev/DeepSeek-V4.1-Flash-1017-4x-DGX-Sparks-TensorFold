//!   - TF_DS_RING_WORD: the generation-word collecting kernel (dsv41_rdma_collect_into_roci) on a receive ring's
//!     slots against the baseline dsv41_rdma_collect_into on the same slots, a host thread writing the lanes' words after
//!     the launch: the word signal (the sequence inline) and the atomic signal (one fetch-and-add a lane) give the baseline
//!     bytes (also out of write-combined landing slots); one landing byte changed differs; words written 20 ms late
//!     still collect under the deadline; a lane's
//!     word that never arrives ends in the failure record within the deadline (the ring's abort word set, nothing
//!     copied); the abort word alone ends a wait with the record's abort reason, far under the deadline.
//!   - TF_DS_ROCI's two-block readers (_hc_mix_pf_2b, _hc_post_2b, _engram_gate_2b): the blocks as the transport lands them against
//!     widenPairs + the baseline readers (the fp32 planes), byte for byte, at 1 to 1,024 rows with NaN, +-0, subnormals and infinities
//!     in the sums; a sum changed must change each output.
//!   - TF_DS_ROCI's whole-line GEMM stores (the _st kernels): a dense FP8 GEMM into write-combined host memory against the same into
//!     device memory, byte for byte, with the time of both; a changed activation code must change the bytes.
//!   - TF_DS_ROCI_COLLECT's copy (Ops.collect2): two blocks from device memory into write-combined host memory, the bytes equal, the time
//!     of the launch printed; a changed source must differ.
//!     written to the other three regions, copied out to their rows, against the in-place gather's layout, with the region in cudaMalloc, in
//!   - TF_DS_ROCI_STAGGER's two lanes (roci_pair): two passes' exchanges in flight at once on two streams a node, each lane's bytes in its own
//!     lane; a late peer on one lane holds that lane's wait and not the other's streams; the cache gate (pass B's read behind pass A's event is
//!     A's write, without the event it is not).
//!   - TF_DS_ROCI_STAGGER_LANES=3 (roci_triple): three passes' exchanges in flight at once on three lanes, each lane's bytes in its own lane; the cache
//!     gate chain (pass C's read behind B's event behind A's event is A's write, with B's wait on A's left out it is not).
//!   - The decoder half on owned rows (roci_dec_rows, roci_dec_mixes, roci_dec_kset): at every decoder window of d = 1 to 128 rows, each
//!     launch class layers 20-39 make at a node's slab of the plan (slab_plan: q(d) = 16 or 32 rows a rank) against the window's one
//!     launch, the baseline decoder half's: act_quant, wq_a / wkv / wq_b, the q RMSNorm, the router's logits and routing, hcPre; the posted
//!     mixes and posts from the transport's two blocks at the plan's stride against _hc_mix_pf / _hc_post on the quarters' planes; a
//!     tap's mean; the experts' gate / up from the slabs' codes against gateUp's own quantization. Every byte equal; a slab one row short
//!     and blocks read one row short of their stride must differ. And the decoder half's start check on the kit (every layer's launches at
//!     every slab of a window, the two-block readers in the extra set), with a control the check must refuse.
            r.report(try std.fmt.bufPrint(&line, "pf_split {s} control: one code of row 700 changed ({s})", .{ sh.name, config_names[@backingInt(cfg_on) - 2] }), differs(want, got), false);
        for (names, want, got) |nm, w, g| r.report(try std.fmt.bufPrint(&line, "idx_proj_owned q={d} (pf_split {s}): {s} at every owned block against the whole chunk's", .{ q, mx.PfSplit.name(@backingInt(sw)).?, nm }), differs(w, g), true);
    /// The generation-word copy-out (dsv41_rdma_collect_into_roci) of both ranks' slots into `dst`, the signal the
    /// waiting kernel sees and the failure record it writes, through the same launch the rings use.
    fn collectRoci(g: *RingRig, recv_dev: u64, dst: u64, n4: usize, gen: u64, lanes: u64, fault: u64, deadline_ns: u64, sig: dsv41.rdma.Roci) !void {
        var dd: dsv41.rdma.Dsts = .{};
        for (0..world) |p| {
            dd.ptr[p] = dst + p * n4 * 16;
            dd.row4[p] = @intCast(n4);
            dd.stride4[p] = @intCast(n4);
        }
        var copy4: u64 = 0;
        for (0..world) |p| if (p != rank) {
            copy4 += n4;
        };
        const ring: dsv41.rdma.CollectRing = .{ .recv = recv_dev, .flags = g.metaDev(0), .seq = g.seq.ptr, .abort = g.metaDev(abort_w), .rank = rank, .world = world, .gen = gen, .lanes = lanes, .fault = fault, .deadline_ns = deadline_ns, .roci = sig };
        try dsv41.rdma.launchCollect(g.k, .{ .max_bytes = max_bytes, .slots = slots, .roci = sig }, g.s, dd, 1, ring, copy4);
    }

// -- TF_DS_RING_WORD: the generation word wakes the collecting kernel ----------------------------------------------

/// A host writer, standing in for the sending NIC: it stores (or fetch-adds) one lane's 64-bit generation word after
/// the collecting kernel is queued, so the kernel's wait is what the test exercises. A separate thread, because the
/// kernel only returns once the word is in.
const WordWriter = struct {
    fn store(io: std.Io, p: *u64, delay_ms: i64, value: u64) void {
        std.Io.sleep(io, .fromMilliseconds(delay_ms), .awake) catch {};
        @atomicStore(u64, p, value, .release);
    }
    fn add(io: std.Io, p: *u64, delay_ms: i64) void {
        std.Io.sleep(io, .fromMilliseconds(delay_ms), .awake) catch {};
        _ = @atomicRmw(u64, p, .Add, 1, .release);
    }
    fn abort(io: std.Io, p: *u64, delay_ms: i64) void {
        std.Io.sleep(io, .fromMilliseconds(delay_ms), .awake) catch {};
        @atomicStore(u64, p, 1, .release);
    }
};

/// TF_DS_RING_WORD's collecting kernel against the baseline on the same slots: the word signal and the atomic signal
/// (the host thread writes the words after the launch), a changed landing byte, a late word, a lane that never arrives
/// (the failure record, the poison and the untouched destination, within the deadline) and the abort word ending a wait
/// early. The bytes are the baseline collect kernel's from the same receive ring.
fn ringWord(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var g = try RingRig.init(r.d, r.s, r.io orelse return error.NoIo);
    defer g.deinit();
    r.note("ring_word: the rings' fatbin has the generation-word kernels (dsv41_rdma_collect_roci, _into_roci, _peers_roci)", g.k.hasRoci());
    if (!g.k.hasRoci()) {
        for (0..7) |_| r.note("ring_word: no generation-word kernel to run", false);
        return;
    }
    var prng = std.Random.DefaultPrng.init(0x0c1dec);
    const rng = prng.random();
    const n4: usize = 192;
    const nbytes = n4 * 16;
    const peer = try r.a.alloc(u8, nbytes);
    defer r.a.free(peer);
    for (peer) |*b| b.* = rng.int(u8);
    const slot = RingRig.q % RingRig.slots;
    const peer_off = @as(usize, @intCast(slot)) * RingRig.world * RingRig.max_bytes + RingRig.max_bytes;

    // the generation words (ordinary pinned memory: the CPU and the peers' NICs read-modify-write them), the lanes
    // toward each peer, and the first expired wait's record
    var gen = try dsv41.rdma.Pinned.alloc(r.d, &g.hr, dsv41.rdma.rociWords() * 8);
    defer gen.free(&g.hr);
    var lanes_buf = try cuda.DeviceBuffer.alloc(r.d, 2 * 4);
    defer lanes_buf.free();
    const lane_counts = [2]u32{ 0, 2 };
    var fault = try cuda.DeviceBuffer.alloc(r.d, 64);
    defer fault.free();
    try fault.fill8(0, null);
    const words: [*]u64 = @ptrCast(@alignCast(gen.mem.ptr));
    const lane_at = 1 * dsv41.rdma.max_devices; // the sender is peer 1; rank 0 reads its words
    var dst_today = try cuda.DeviceBuffer.alloc(r.d, 2 * nbytes);
    defer dst_today.free();
    var dst = try cuda.DeviceBuffer.alloc(r.d, 2 * nbytes);
    defer dst.free();

    // the baseline collecting kernel: the peer's slot in, its flag at q, into the reference bytes
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    g.words()[@as(usize, @intCast(slot)) * RingRig.world + 1] = RingRig.q;
    try poison(dst_today.ptr, 2 * nbytes, r.d, r.s);
    try g.collect(g.recv.dev, dst_today.ptr, n4);
    try r.s.synchronize();
    const want = try get(dst_today.ptr, 2 * nbytes, r.d);
    defer free(want);

    const deadline_ns: u64 = 500 * std.time.ns_per_ms;

    // the word signal: the same slot, the host thread writing both lanes' words after the launch
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .word);
    var w0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 2, RingRig.q });
    var w1 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at + 1], 2, RingRig.q });
    w0.join();
    w1.join();
    try r.s.synchronize();
    const got = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(got);

    // the same signal out of write-combined landing slots (TF_DS_RING_WC's memory), against the baseline collect there
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv_wc.mem[peer_off..][0..nbytes], peer);
    g.words()[@as(usize, @intCast(slot)) * RingRig.world + 1] = RingRig.q;
    try poison(dst_today.ptr, 2 * nbytes, r.d, r.s);
    try g.collect(g.recv_wc.dev, dst_today.ptr, n4);
    try r.s.synchronize();
    const want_wc = try get(dst_today.ptr, 2 * nbytes, r.d);
    defer free(want_wc);
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv_wc.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv_wc.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .word);
    var wc0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 2, RingRig.q });
    var wc1 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at + 1], 2, RingRig.q });
    wc0.join();
    wc1.join();
    try r.s.synchronize();
    const got_wc = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(got_wc);

    // the atomic signal: the host fetch-adds one per lane; the word the kernel waits on counts to the sequence
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .atomic);
    var a0 = try std.Thread.spawn(.{}, WordWriter.add, .{ g.io, &words[lane_at], 2 });
    var a1 = try std.Thread.spawn(.{}, WordWriter.add, .{ g.io, &words[lane_at + 1], 2 });
    a0.join();
    a1.join();
    try r.s.synchronize();
    const atomic_got = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(atomic_got);

    // the control: one landing byte changed differs
    var peer2 = try r.a.dupe(u8, peer);
    defer r.a.free(peer2);
    peer2[0] ^= 0xff;
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer2);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .word);
    var c0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 2, RingRig.q });
    var c1 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at + 1], 2, RingRig.q });
    c0.join();
    c1.join();
    try r.s.synchronize();
    const ctl = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(ctl);
    r.report("ring_word: one landing byte changed differs the collected bytes", differs(want, ctl), false);

    // a late word (20 ms) still collects under a 500 ms deadline: the wait is bounded, not a fixed sleep
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, deadline_ns, .word);
    var l0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 20, RingRig.q });
    var l1 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at + 1], 20, RingRig.q });
    l0.join();
    l1.join();
    try r.s.synchronize();
    const late = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(late);

    // one lane's word never arrives: the deadline ends the wait with the record, the ring poisoned, nothing copied
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try fault.fill8(0, null);
    const short_deadline: u64 = 50 * std.time.ns_per_ms;
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, short_deadline, .word);
    var n0 = try std.Thread.spawn(.{}, WordWriter.store, .{ g.io, &words[lane_at], 2, RingRig.q });
    n0.join();
    try r.s.synchronize();
    var rec: [8]u64 = undefined;
    const dead = try get(dst.ptr, 2 * nbytes, r.d);
    defer free(dead);
    var untouched: usize = 0;
    for (dead) |b| untouched += @intFromBool(b != 0xa5);
    const record_ok = rec[0] == 0x524f_4349_0000_0001 and rec[1] == RingRig.q and (rec[3] >> 32) == 1 and
        (rec[3] & 0xffff_ffff) == 1 and rec[4] == 0 and rec[7] == 2 and g.words()[RingRig.abort_w] == 1;
    r.note("ring_word: a lane's word that never arrives ends in the failure record (magic, sequence, peer 1 lane 1, the word seen 0, the deadline) with the ring's abort word set and the destination untouched", record_ok and untouched == 0);
    r.note("ring_word: the expired wait ended within its deadline and well before the replay would give up", rec[5] >= 45 * std.time.ns_per_ms and rec[5] < 5 * std.time.ns_per_s);

    // the abort word alone ends a wait early (why 1), with a deadline far away
    @memset(gen.mem, 0);
    g.reset();
    try g.setSeq(RingRig.q);
    @memcpy(g.recv.mem[peer_off..][0..nbytes], peer);
    try poison(dst.ptr, 2 * nbytes, r.d, r.s);
    try fault.fill8(0, null);
    try g.collectRoci(g.recv.dev, dst.ptr, n4, gen.dev, lanes_buf.ptr, fault.ptr, std.time.ns_per_s, .word);
    var ab = try std.Thread.spawn(.{}, WordWriter.abort, .{ g.io, &g.words()[RingRig.abort_w], 2 });
    ab.join();
    try r.s.synchronize();
    r.note("ring_word: the abort word ends a waiting kernel well before its 1 s deadline (the record's why is the abort)", rec[0] == 0x524f_4349_0000_0001 and rec[7] == 1 and rec[5] < 900 * std.time.ns_per_ms);
}

// -- TF_DS_ROCI's two-block readers ----------------------------------------------------------------------------------

/// bf16 [2, rows, w] sums with the special values the identity argument names scattered in (quiet NaN, +-0, subnormals,
/// +-inf, the largest finite).
fn rociBlocks(rows: usize, w: usize, rng: std.Random, a: std.mem.Allocator) ![]u16 {
    const v = try a.alloc(u16, 2 * rows * w);
    for (v) |*x| x.* = bf16Rne(rng.floatNorm(f32) * 2.0);
    const special = [_]u16{ 0x7fc0, 0x0000, 0x8000, 0x7f80, 0xff80, 0x7f7f, 0x0001, 0x8001, 0xff7f };
    for (0..@min(v.len / 4, 3000)) |_| v[rng.uintLessThan(usize, v.len)] = special[rng.uintLessThan(usize, special.len)];
    return v;
}

const RociIn = struct { blk: cuda.DeviceBuffer, planes: cuda.DeviceBuffer, rows: usize };

/// The baseline input of the readers from the blocks: widenPairs into plane 0 of the tight fp32 buffer, plane 1 -0.0.
fn rociPlanes(r: *Run, blk: cuda.DeviceBuffer, rows: usize, w: usize) !cuda.DeviceBuffer {
    const planes = try cuda.DeviceBuffer.alloc(r.d, 2 * rows * 2 * w * 4);
    try r.ops.?.widenPairs(r.s, blk.ptr, blk.ptr + rows * w * 2, planes.ptr, rows, w);
    try r.d.check(r.d.api.cuMemsetD32Async(planes.ptr + rows * 2 * w * 4, dsv41.prompt2d.plane1_word, rows * 2 * w, r.s.handle), "cuMemsetD32Async");
    return planes;
}

fn rociReaders(r: *Run, rng: std.Random) !void {
    const D: usize = 5120;
    const W: usize = D / 2;
    const EW: usize = 12800;
    var line: [256]u8 = undefined;
    if (r.t.extra == null or r.ops == null) return r.note("roci_readers: the kit's extra set (aot-extra/) loads and the ops are available", false);
    r.note("roci_readers: the kit's extra set (aot-extra/) loads", true);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    const hc_blocks = dsv41.tri_basic.hc_blocks;
    for ([_]usize{ 1, 5, 13, 16, 100, 512, 1024 }) |rows| {
        const sums = try rociBlocks(rows, W, rng, ha);
        var blk = try upload(r.d, std.mem.sliceAsBytes(sums));
        defer blk.free();
        var planes = try rociPlanes(r, blk, rows, W);
        defer planes.free();
        var h = try upload(r.d, try bf16s(rows * 4 * D, rng, 1.0, ha));
        defer h.free();
        var post = try upload(r.d, try f32s(rows * 4, rng, ha));
        defer post.free();
        var comb = try upload(r.d, try f32s(rows * 16, rng, ha));
        defer comb.free();
        var fnw = try upload(r.d, try f32s(24 * 4 * D, rng, ha));
        defer fnw.free();
        var outs: [2]cuda.DeviceBuffer = undefined;
        var parts: [2]cuda.DeviceBuffer = undefined;
        for (&outs, &parts, 0..) |*o, *pt, i| {
            o.* = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            pt.* = try cuda.DeviceBuffer.alloc(r.d, rows * hc_blocks * 32 * 4);
            try poison(o.ptr, rows * 4 * D * 2, r.d, r.s);
            try poison(pt.ptr, rows * hc_blocks * 32 * 4, r.d, r.s);
            const posted: dsv41.tri_hc.Posted = if (i == 0) .{ .gathered = planes.ptr, .world = 2, .h_out = o.ptr } else .{ .gathered = blk.ptr, .world = 2, .h_out = o.ptr, .blk_rs = rows * W };
            _ = try dsv41.tri_hc.hcMixPf(r.t, h.ptr, fnw.ptr, pt.ptr, posted, post.ptr, comb.ptr, rows, D);
        }
        try r.s.synchronize();
        const xo = [2][]u8{ try get(outs[0].ptr, rows * 4 * D * 2, r.d), try get(outs[1].ptr, rows * 4 * D * 2, r.d) };
        const pp = [2][]u8{ try get(parts[0].ptr, rows * hc_blocks * 32 * 4, r.d), try get(parts[1].ptr, rows * hc_blocks * 32 * 4, r.d) };
        r.report(try std.fmt.bufPrint(&line, "roci_readers {d} rows: _hc_mix_pf_2b's streams and partial dots against widenPairs + _hc_mix_pf (NaN, +-0, subnormals, +-inf in the sums)", .{rows}), differs(xo[0], xo[1]) + differs(pp[0], pp[1]), true);
        for (&outs) |*o| o.free();
        for (&parts) |*pt| pt.free();
        // hc_post in place on a copy of the streams
        var hp: [2]cuda.DeviceBuffer = undefined;
        for (&hp, 0..) |*b, i| {
            b.* = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(b.ptr, h.ptr, rows * 4 * D * 2, r.s.handle), "cuMemcpyDtoDAsync");
            if (i == 0) try dsv41.tri_basic.hcPost(r.t, planes.ptr, b.ptr, post.ptr, comb.ptr, b.ptr, 2, rows, D) else try dsv41.tri_basic.hcPost2b(r.t, blk.ptr, rows * W, b.ptr, post.ptr, comb.ptr, b.ptr, rows, D);
        }
        try r.s.synchronize();
        const ph = [2][]u8{ try get(hp[0].ptr, rows * 4 * D * 2, r.d), try get(hp[1].ptr, rows * 4 * D * 2, r.d) };
        r.report(try std.fmt.bufPrint(&line, "roci_readers {d} rows: _hc_post_2b (the tile at column 2,560 straddles the blocks) against widenPairs + _hc_post", .{rows}), differs(ph[0], ph[1]), true);
        for (&hp) |*b| b.free();
        // the Engram gate: the key and value row [2 blocks of 12,800] as one row of 25,600 against the blocks as they lie
        const eb = try rociBlocks(rows, EW, rng, ha);
        var ekb = try upload(r.d, std.mem.sliceAsBytes(eb));
        defer ekb.free();
        var kv = try cuda.DeviceBuffer.alloc(r.d, rows * 2 * EW * 2);
        defer kv.free();
        for (0..2) |b| try r.ops.?.copyRows(r.s, ekb.ptr + b * rows * EW * 2, EW * 2, kv.ptr + b * EW * 2, 2 * EW * 2, EW * 2, rows);
        var qk = try upload(r.d, try f32s(4 * D, rng, ha));
        defer qk.free();
        var eo: [2]cuda.DeviceBuffer = undefined;
        for (&eo, 0..) |*o, i| {
            o.* = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            try poison(o.ptr, rows * 4 * D * 2, r.d, r.s);
            if (i == 0) try dsv41.tri_basic.engramGate(r.t, h.ptr, kv.ptr, qk.ptr, o.ptr, 1e-20, rows, D) else try dsv41.tri_basic.engramGate2b(r.t, h.ptr, ekb.ptr, rows * EW, qk.ptr, o.ptr, 1e-20, rows, D);
        }
        try r.s.synchronize();
        const eg = [2][]u8{ try get(eo[0].ptr, rows * 4 * D * 2, r.d), try get(eo[1].ptr, rows * 4 * D * 2, r.d) };
        r.report(try std.fmt.bufPrint(&line, "roci_readers {d} rows: _engram_gate_2b against the key and value rows assembled + _engram_gate", .{rows}), differs(eg[0], eg[1]), true);
        for (&eo) |*o| o.free();
        // controls, once: one sum changed in the last row's last column must change each reader's output
        if (rows == 512) {
            const bad = try ha.dupe(u16, sums);
            bad[2 * rows * W - 1] = if (bad[2 * rows * W - 1] == 0x4100) 0x4200 else 0x4100;
            var blk2 = try upload(r.d, std.mem.sliceAsBytes(bad));
            defer blk2.free();
            var c1 = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            defer c1.free();
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(c1.ptr, h.ptr, rows * 4 * D * 2, r.s.handle), "cuMemcpyDtoDAsync");
            try dsv41.tri_basic.hcPost2b(r.t, blk2.ptr, rows * W, c1.ptr, post.ptr, comb.ptr, c1.ptr, rows, D);
            try r.s.synchronize();
            r.report("roci_readers control: _hc_post_2b with one sum changed (the last row's last column, block 1)", differs(ph[0], try get(c1.ptr, rows * 4 * D * 2, r.d)), false);
            const bad_e = try ha.dupe(u16, eb);
            bad_e[2 * rows * EW - 1] = if (bad_e[2 * rows * EW - 1] == 0x4100) 0x4200 else 0x4100;
            var ekb2 = try upload(r.d, std.mem.sliceAsBytes(bad_e));
            defer ekb2.free();
            var c2 = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            defer c2.free();
            try poison(c2.ptr, rows * 4 * D * 2, r.d, r.s);
            try dsv41.tri_basic.engramGate2b(r.t, h.ptr, ekb2.ptr, rows * EW, qk.ptr, c2.ptr, 1e-20, rows, D);
            try r.s.synchronize();
            r.report("roci_readers control: _engram_gate_2b with the value's last column changed", differs(eg[0], try get(c2.ptr, rows * 4 * D * 2, r.d)), false);
            var c3 = try cuda.DeviceBuffer.alloc(r.d, rows * 4 * D * 2);
            defer c3.free();
            var c3p = try cuda.DeviceBuffer.alloc(r.d, rows * hc_blocks * 32 * 4);
            defer c3p.free();
            try poison(c3.ptr, rows * 4 * D * 2, r.d, r.s);
            try poison(c3p.ptr, rows * hc_blocks * 32 * 4, r.d, r.s);
            _ = try dsv41.tri_hc.hcMixPf(r.t, h.ptr, fnw.ptr, c3p.ptr, .{ .gathered = blk2.ptr, .world = 2, .h_out = c3.ptr, .blk_rs = rows * W }, post.ptr, comb.ptr, rows, D);
            try r.s.synchronize();
            r.report("roci_readers control: _hc_mix_pf_2b with one sum changed", differs(xo[0], try get(c3.ptr, rows * 4 * D * 2, r.d)), false);
        }
    }
}

// -- TF_DS_ROCI's whole-line GEMM stores -------------------------------------------------------------------------------

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A dense FP8 GEMM into y in write-combined host memory (the transport's arena), which takes the _st kernels, against the
/// same GEMM into device memory, byte for byte: wo_b's shape and an Engram-width one, the fold / split A / split C7 tiles and
/// the exact block-scaled kernel (its key proven), at 64 to 4,096 rows. The time of each, in ms a launch, is printed beside.
fn rociStage(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x57a9e);
    const rng = prng.random();
    const d = r.d;
    var line: [320]u8 = undefined;
    const shapes = [_]Shape{ .{ .name = "wo_b", .k = 4096, .n = 2560 }, .{ .name = "engram", .k = 5120, .n = 12800 } };
    const rows_list = [_]usize{ 64, 129, 1000, 4096 };
    const max_m: usize = 4096;
    var arena_mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&arena_mem, max_m * 12800 * 4, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(arena_mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem, 0), "cuMemHostGetDevicePointer");
    try r.k.useStaged(wc, wc + max_m * 12800 * 4);
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
        r.k.sf_nproven = 0;
    }
    var last_control: []u8 = &.{};
    for (shapes) |sh| {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
        const x = try upload(d, try bf16s(max_m * sh.k, rng, 0.5, ha));
        try keep.append(ha, x);
        const q = try cuda.DeviceBuffer.alloc(d, max_m * sh.k);
        try keep.append(ha, q);
        const sc = try cuda.DeviceBuffer.alloc(d, max_m * (sh.k / 32) * 4);
        try keep.append(ha, sc);
        try mx.quant(r.k, r.s, x.ptr, .bf16, sh.k, max_m, sh.k, q.ptr, sh.k, sc.ptr, sh.k / 32);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, max_m), 1) * 4);
        try keep.append(ha, z);
        const y = try cuda.DeviceBuffer.alloc(d, max_m * sh.n * 4);
        try keep.append(ha, y);
        const names = [_][]const u8{ "fold / tile", "split A", "split C7", "block-scaled exact" };
        for (names, 0..) |nm, v| {
            var diff: usize = 0;
            var t_dev: u64 = 0;
            var t_arena: u64 = 0;
            for (rows_list) |m| {
                const run = struct {
                    fn go(rr: *Run, vv: usize, ll: weights.Linear, qq: u64, ss: u64, zz: u64, mm: usize, yy: u64) ![]u8 {
                        return switch (vv) {
                            0 => gemmOnce(rr, .off, false, ll, qq, ss, zz, 0, mm, yy, .f32, null),
                            1 => gemmOnce(rr, .a, false, ll, qq, ss, zz, 0, mm, yy, .f32, null),
                            2 => gemmOnce(rr, .c7, false, ll, qq, ss, zz, 0, mm, yy, .f32, null),
                            else => gemmExact(rr, true, ll, qq, ss, zz, mm, yy, .f32),
                        };
                    }
                };
                if (v == 3) {
                    if (r.k.sfk == null) continue;
                    try mx.sfProve(r.k, l.s);
                }
                r.k.wc = .{ 0, 0 };
                const a = try run.go(r, v, l, q.ptr, sc.ptr, z.ptr, m, y.ptr);
                defer free(a);
                r.k.wc = .{ wc, wc + max_m * 12800 * 4 };
                const b = try run.go(r, v, l, q.ptr, sc.ptr, z.ptr, m, wc);
                defer free(b);
                diff += differs(a, b);
                if (v == 0 and m == 4096 and sh.n == 2560) last_control = try std.heap.page_allocator.dupe(u8, a);
                if (m == 4096) {
                    for ([_]u1{ 0, 1 }) |on| {
                        r.k.wc = if (on == 1) .{ wc, wc + max_m * 12800 * 4 } else .{ 0, 0 };
                        const yy = if (on == 1) wc else y.ptr;
                        const t0 = nowNs();
                        for (0..10) |_| free(try run.go(r, v, l, q.ptr, sc.ptr, z.ptr, m, yy));
                        if (on == 1) t_arena = nowNs() - t0 else t_dev = nowNs() - t0;
                    }
                }
            }
            if (v == 3 and r.k.sfk == null) {
                r.note(try std.fmt.bufPrint(&line, "roci_stage {s} ({d} x {d}) {s}: the block-scaled kernels (native_sf.cu) are not loaded on this GPU", .{ sh.name, sh.k, sh.n, nm }), false);
                continue;
            }
            std.debug.print("roci_stage {s} {s} at 4,096 rows (ms a launch incl. the readback): device y {d:.2}, arena y {d:.2}\n", .{ sh.name, nm, @as(f64, @floatFromInt(t_dev)) / 1e7, @as(f64, @floatFromInt(t_arena)) / 1e7 });
            r.report(try std.fmt.bufPrint(&line, "roci_stage {s} ({d} x {d}), {s}: y in write-combined host memory against y in device memory at 64, 129, 1,000 and 4,096 rows", .{ sh.name, sh.k, sh.n, nm }), diff, true);
        }
    }
    // control: one activation code changed in row 700 changes the arena GEMM's bytes (the comparison is live)
    r.k.wc = .{ wc, wc + max_m * 12800 * 4 };
    defer if (last_control.len != 0) free(last_control);
    {
        var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer sarena.deinit();
        const ha = sarena.allocator();
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        const sh = shapes[0];
        const l = try fp8Linear(d, sh.n, sh.k, .fast, rng, ha, &keep);
        const qb = try ha.alloc(u8, 1024 * sh.k);
        @memset(qb, 0x38);
        const sb = try ha.alloc(f32, 1024 * (sh.k / 32));
        @memset(sb, 1.0);
        const q = try upload(d, qb);
        try keep.append(ha, q);
        const sc = try upload(d, std.mem.sliceAsBytes(sb));
        try keep.append(ha, sc);
        const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, 1024), 1) * 4);
        try keep.append(ha, z);
        const a = try gemmOnce(r, .off, false, l, q.ptr, sc.ptr, z.ptr, 0, 1000, wc, .f32, null);
        defer free(a);
        const b = try gemmOnce(r, .off, false, l, q.ptr, sc.ptr, z.ptr, 0, 1000, wc, .f32, null);
        defer free(b);
        r.report("roci_stage control: one activation code of row 700 changed, the arena GEMM's bytes", differs(a, b), false);
    }
}

// -- TF_DS_ROCI_COLLECT's copy ---------------------------------------------------------------------------------------

/// Ops.collect2: two blocks from device memory into write-combined host memory (the transport's arena) in one launch, at the
/// owner sums' part 1 sizes (wo_b's and MoE's psum blocks 10,485,760 bytes, Engram's 52,428,800): the bytes equal, and the
/// launch's time printed (ms, 20 launches). A source changed after the copy must differ from the copy.
fn rociCollect(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const d = r.d;
    var prng = std.Random.DefaultPrng.init(0xc011ec7);
    const rng = prng.random();
    var line: [256]u8 = undefined;
    const ops = r.ops orelse return r.note("roci_collect: the ops are available", false);
    const big: usize = 52_428_800;
    var mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&mem, 3 * big, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, mem, 0), "cuMemHostGetDevicePointer");
    var src = try cuda.DeviceBuffer.alloc(d, 2 * big);
    defer src.free();
    const host = try std.heap.page_allocator.alloc(u8, 2 * big);
    defer std.heap.page_allocator.free(host);
    rng.bytes(host);
    var kept: []u8 = &.{};
    defer if (kept.len != 0) free(kept);
    for ([_]usize{ 10_485_760, big }) |bytes| {
        // the blocks sit at different offsets of the source and of the arena, as the plan's rows do
        try poison(wc, 3 * big, d, r.s);
        try ops.collect2(r.s, src.ptr, wc + big, src.ptr + big, wc, bytes);
        try r.s.synchronize();
        const a = try get(wc + big, bytes, d);
        defer free(a);
        const b = try get(wc, bytes, d);
        defer free(b);
        r.report(try std.fmt.bufPrint(&line, "roci_collect {d} bytes a block: both blocks in the arena equal their sources", .{bytes}), differs(a, host[0..bytes]) + differs(b, host[big..][0..bytes]), true);
        const t0 = nowNs();
        for (0..20) |_| try ops.collect2(r.s, src.ptr, wc + big, src.ptr + big, wc, bytes);
        try r.s.synchronize();
        std.debug.print("roci_collect {d} bytes a block, two blocks: {d:.3} ms a launch\n", .{ bytes, @as(f64, @floatFromInt(nowNs() - t0)) / 20e6 });
        if (bytes == big) kept = try std.heap.page_allocator.dupe(u8, a);
    }
    // control: the first block's source changed after the copy no longer equals the copy
    host[7] ^= 0x55;
    r.report("roci_collect control: the source changed after the copy", differs(kept, host[0..big]), false);
}

    // a node of the second pair, its ranks in an order that is not the identity
/// The row counts a T-ANY class runs at and how its lines name them.
const AnyPlan = struct {
    /// the chunks' rows (a decoder window's for the decoder half)
    ns: []const usize,
    /// the most of them: the rows the buffers hold
    max: usize,
    /// how the lines name the row counts, and the chunks
    text: []const u8,
    unit: []const u8,
};
const any_plan: AnyPlan = .{ .ns = &any_ns, .max = any_max, .text = "n = 13, 63, 64, 65, 500, 1,087, 1,956, 3,000", .unit = "chunks" };

/// The decoder half's windows: every row count a chunk's decoder half runs at (1 row to the window's 128, prompt.decoderRows), whose slabs
/// (q(d) = 16 or 32 rows) hold every row count from 1 to 32.
const dec_max: usize = 128;
const dec_ds = blk: {
    var ds: [dec_max]usize = undefined;
    for (&ds, 1..) |*x, i| x.* = i;
    break :blk ds;
};
const dec_plan: AnyPlan = .{ .ns = &dec_ds, .max = dec_max, .text = "every decoder window d = 1 to 128 (slabs of q(d) = 16 or 32 rows)", .unit = "windows" };

    /// The row counts it runs at.
    plan: *const AnyPlan = &any_plan,
/// The control's short slab of a chunk of `n` rows (replay_runner.shortSlab).
const anyShort = dsv41.replay_runner.shortSlab;

comptime {
    // the outputs are poisoned with the byte the null control reads as "not written"
    if (dsv41.replay_runner.poison_byte != 0xa5) @compileError("poison() and the runner's poison byte differ");
}

/// What a class's run over its chunks found: the bytes that differ (a chunk whose null control failed counts every byte, uncompared),
/// the chunks whose control differed, the chunks whose null control held, and the first that did not (its rows, and whether the one
/// launch or the slabs left their output unwritten).
const AnyRes = struct { bad: usize = 0, controls_ok: usize = 0, null_ok: usize = 0, first_null: ?struct { n: usize, slabs: bool } = null };

/// Whole launch of n rows against the four ranks' slabs (slab_plan.slabs), the rows of the output equal; before the comparison the null
/// control, each run's output written (replay_runner.wrotePoisoned: at least half its bytes no longer the poison, not all zero); and the
/// control, one row of the last slab that has more than one row left out (anyShort), which must differ. A failing step names itself: the
/// class, the chunk row count, the slab and the step.
fn anyClassRun(r: *Run, c: AnyClass) !AnyRes {
    var res: AnyRes = .{};
    const max = c.plan.max;
    for (c.plan.ns) |n| {
        poison(c.out, max * c.row, r.d, r.s) catch |e| return anyFail(c, n, null, "poison", e);
        poison(c.out, max * c.row, r.d, r.s) catch |e| return anyFail(c, n, null, "poison", e);
        const ref_wrote = dsv41.replay_runner.wrotePoisoned(ref);
        if (ref_wrote and dsv41.replay_runner.wrotePoisoned(got)) {
            res.null_ok += 1;
            res.bad += differs(ref, got);
        } else {
            res.bad += bytes;
            if (res.first_null == null) res.first_null = .{ .n = n, .slabs = ref_wrote };
        }
        poison(c.out, max * c.row, r.d, r.s) catch |e| return anyFail(c, n, null, "poison", e);
        const short = anyShort(n);
            if (m > 0) c.run(c.ctx, sb.start, m) catch |e| return anyFail(c, n, .{ k, sb.start, m }, "the control's slab launch", e);
        res.controls_ok += @intFromBool(differs(ref, cut) > 0);
    return res;
    r.report(try std.fmt.bufPrint(&line, "{s} {s}: the node's slab launches against the chunk's one launch at {s}", .{ c.section, c.name, c.plan.text }), res.bad, true);
    if (res.first_null) |f| {
        r.note(try std.fmt.bufPrint(&line, "{s} {s} null control: the one launch and the slabs wrote their outputs in {d} of {d} {s}; first at {d} rows: {s} did not", .{ c.section, c.name, res.null_ok, c.plan.ns.len, c.plan.unit, f.n, if (f.slabs) "the slabs' launches" else "the chunk's one launch" }), false);
    } else r.note(try std.fmt.bufPrint(&line, "{s} {s} null control: the one launch and the slabs wrote their outputs, in all {d} {s}", .{ c.section, c.name, c.plan.ns.len, c.plan.unit }), res.null_ok == c.plan.ns.len);
    r.report(try std.fmt.bufPrint(&line, "{s} {s} control: the last slab one row short, in all {d} {s}", .{ c.section, c.name, c.plan.ns.len, c.plan.unit }), if (res.controls_ok == c.plan.ns.len) 1 else 0, false);
    return rociRowsAt(r, rng_in, "roci_any_rows", &any_plan);
}

/// The decoder half on owned rows (layers 20-39 at a window of d decoder rows, the gate of moving its gathers onto the transport): the same
/// launch classes on each node's slab of every window d = 1 to 128 against the window's one launch, the baseline decoder half's.
fn rociDecRows(r: *Run, rng_in: std.Random) !void {
    return rociRowsAt(r, rng_in, "roci_dec_rows", &dec_plan);
}

fn rociRowsAt(r: *Run, rng_in: std.Random, section: []const u8, plan: *const AnyPlan) !void {
    return rociAnyRowsBody(r, rng_in, &stage, section, plan) catch |e| {
        std.debug.print("{s}: FAILED in setup step \"{s}\": {s}\n", .{ section, stage, @errorName(e) });
fn rociAnyRowsBody(r: *Run, rng_in: std.Random, stage: *[]const u8, section: []const u8, plan: *const AnyPlan) !void {
    stage.* = "the activation of the plan's most rows and its act_quant";
    const M = plan.max;
    try anyReport(r, .{ .name = "act_quant", .ctx = &qc, .run = Quant.run, .out = out_big.ptr, .row = D, .section = section, .plan = plan });
    try anyReport(r, .{ .name = "wq_a GEMM", .ctx = &ga, .run = Gemm.run, .out = out_big.ptr, .row = 1280 * 2, .section = section, .plan = plan });
    try anyReport(r, .{ .name = "wkv GEMM", .ctx = &gk, .run = Gemm.run, .out = out_big.ptr, .row = 512 * 2, .section = section, .plan = plan });
    try anyReport(r, .{ .name = "wq_b GEMM", .ctx = &gb, .run = Gemm.run, .out = out_big.ptr, .row = 8192 * 2, .section = section, .plan = plan });
    try anyReport(r, .{ .name = "q RMSNorm", .ctx = &nc, .run = Norm.run, .out = out_big.ptr, .row = 1280 * 2, .section = section, .plan = plan });
    try anyReport(r, .{ .name = "router logits", .ctx = &lc, .run = Logits.run, .out = out_big.ptr, .row = 384 * 4, .section = section, .plan = plan });
    try anyReport(r, .{ .name = "ffnRoute", .ctx = &rc, .run = Route.run, .out = out_big.ptr, .row = 7 * 4, .section = section, .plan = plan });
        var nl: [128]u8 = undefined;
        r.note(try std.fmt.bufPrint(&nl, "{s} hcPre: the kit's extra set loads", .{section}), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcPre null control: the kit's extra set loads", .{section}), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcPre control: the kit's extra set loads", .{section}), false);
    try anyReport(r, .{ .name = "hcPre (x)", .ctx = &hcc, .run = Hc.run, .out = out_big.ptr, .row = D * 2, .section = section, .plan = plan });
// -- The decoder half on owned rows: the launch classes it adds to the encoder half's ---------------------------------------------

/// The hidden width, a pair's columns, a row of the streams.
const DM_D: usize = 5120;
const DM_W: usize = DM_D / 2;
const DM_RH: usize = 4 * DM_D * 2;

/// fp32 values, normal * scale.
fn f32sAt(n: usize, rng: std.Random, scale: f32, a: std.mem.Allocator) ![]u8 {
    const v = try a.alloc(f32, n);
    for (v) |*x| x.* = rng.floatNorm(f32) * scale;
    return std.mem.sliceAsBytes(v);
}

/// The device bytes of `parts` (address, bytes) read back one after another (freed with free).
fn getParts(d: *const cuda.Driver, parts: []const [2]u64) ![]u8 {
    var n: usize = 0;
    for (parts) |pt| n += @intCast(pt[1]);
    const x = try std.heap.page_allocator.alloc(u8, n);
    errdefer std.heap.page_allocator.free(x);
    var at: usize = 0;
    for (parts) |pt| {
        const len: usize = @intCast(pt[1]);
        at += len;
    }
    return x;
}

/// One bf16 value at `addr` in device memory changed (an exponent bit flipped, `saved` its old bits) or put back: the null control's change
/// of an input value. Synchronous: the runner has synchronized the stream before it.
fn flipBf16(r: *Run, addr: u64, on: bool, saved: *u16) !void {
    if (on) {
        const v: u16 = saved.* ^ 0x4000;
}

/// The sums a mix posts for a window of d rows in the two forms its two paths read: the baseline fp32 planes [2, d, D] over the window (plane 0
/// the bf16 sums widened, plane 1 -0.0: the readers' input from the quarters' gather) and, for each slab of the plan, the two bf16 blocks
/// [2, q, D / 2] the transport lands for the slab's owner (block 0 pair 0's columns, block 1 pair 1's; the slab's rows first, the rest
/// poisoned), q = slab_plan.q(d).
const DecSums = struct {
    r: *Run,
    sums: u64, // bf16 [2, dec_max, D / 2]: block b's row i
    planes: u64, // fp32 [2, dec_max, D]
    blk: [4]u64, // bf16 [2, 32, D / 2] a slab

    fn prep(c: *DecSums, d: usize) !void {
        const ops = c.r.ops.?;
        try ops.widenPairs(c.r.s, c.sums, c.sums + dec_max * DM_W * 2, c.planes, d, DM_W);
        try c.r.d.check(c.r.d.api.cuMemsetD32Async(c.planes + d * DM_D * 4, dsv41.prompt2d.plane1_word, d * DM_D, c.r.s.handle), "cuMemsetD32Async");
        const q = dsv41.slab_plan.q(d);
        for (dsv41.slab_plan.slabs(d), 0..) |sb, k| {
            try poison(c.blk[k], 2 * 32 * DM_W * 2, c.r.d, c.r.s);
            for (0..2) |b| try ops.copyRows(c.r.s, c.sums + b * dec_max * DM_W * 2 + sb.start * DM_W * 2, DM_W * 2, c.blk[k] + b * q * DM_W * 2, DM_W * 2, DM_W * 2, sb.n);
        }
    }
};

/// A mix with the previous sublayer's sums posted (attnMixes' and ffnMixes' hcPrePf): the baseline _hc_mix_pf on the planes and _hc_finish over the
/// window's d rows; the owned path's _hc_mix_pf_2b on the slab's blocks at the plan's stride q (`short` rows less: the stride control) and
/// _hc_finish at the slab's rows. post and comb are read by the mix and written by the finish: restored before each run.
const DecMix = struct {
    r: *Run,
    sums: *DecSums,
    h: u64,
    fnw: u64,
    scale: u64,
    base: u64,
    pre_in: u64,
    nw: u64,
    post_in: u64,
    comb_in: u64,
    h_out: u64,
    x: u64,
    pre_o: u64,
    post: u64,
    comb: u64,
    part: u64,
    short: usize = 0,
    saved: u16 = 0,

    pub const strided = true;
    const part_row = dsv41.tri_basic.hc_blocks * 32 * 4;

    /// The window's last row's first sum (block 0), the source prep builds the planes and the blocks from.
    pub fn perturb(c: *DecMix, d: usize, on: bool) !void {
        try flipBf16(c.r, c.sums.sums + (d - 1) * DM_W * 2, on, &c.saved);
    }

    pub fn prep(c: *DecMix, d: usize) !void {
        try c.sums.prep(d);
    }
    pub fn reset(c: *DecMix) !void {
        const r = c.r;
        try poison(c.h_out, dec_max * DM_RH, r.d, r.s);
        try poison(c.x, dec_max * DM_D * 2, r.d, r.s);
        try poison(c.pre_o, dec_max * 16, r.d, r.s);
        try poison(c.part, dec_max * part_row, r.d, r.s);
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(c.post, c.post_in, dec_max * 16, r.s.handle), "cuMemcpyDtoDAsync");
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(c.comb, c.comb_in, dec_max * 64, r.s.handle), "cuMemcpyDtoDAsync");
    }
    pub fn whole(c: *DecMix, d: usize) !void {
        _ = try dsv41.tri_hc.hcPrePf(c.r.t, c.h, c.fnw, c.scale, c.base, c.pre_in, c.nw, 1e-20, 1e-6, 20, c.x, c.pre_o, c.post, c.comb, c.part, .{ .gathered = c.sums.planes, .world = 2, .h_out = c.h_out }, d, DM_D);
    }
    pub fn slab(c: *DecMix, d: usize, k: usize, s: usize, m: usize) !void {
        const q = dsv41.slab_plan.q(d);
        _ = try dsv41.tri_hc.hcPrePf(c.r.t, c.h + s * DM_RH, c.fnw, c.scale, c.base, c.pre_in + s * 16, c.nw, 1e-20, 1e-6, 20, c.x + s * DM_D * 2, c.pre_o + s * 16, c.post + s * 16, c.comb + s * 64, c.part + s * part_row, .{ .gathered = c.sums.blk[k], .world = 2, .h_out = c.h_out + s * DM_RH, .blk_rs = (q - c.short) * DM_W }, m, DM_D);
    }
    pub fn after(c: *DecMix, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn read(c: *DecMix, d: usize) ![]u8 {
        return getParts(c.r.d, &.{ .{ c.h_out, d * DM_RH }, .{ c.x, d * DM_D * 2 }, .{ c.pre_o, d * 16 }, .{ c.post, d * 16 }, .{ c.comb, d * 64 } });
    }
};

/// A pending sum posted without a mix (a DSpark tap's and the head's hc_post), in place on the streams: the baseline _hc_post on the planes over
/// the window's d rows; the owned path's _hc_post_2b on the slab's blocks at the plan's stride q (`short` rows less: the stride control).
const DecPost = struct {
    r: *Run,
    sums: *DecSums,
    h_in: u64,
    h: u64,
    post: u64,
    comb: u64,
    short: usize = 0,
    saved: u16 = 0,

    pub const strided = true;

    /// The window's last row's first sum (block 0), the source prep builds the planes and the blocks from.
    pub fn perturb(c: *DecPost, d: usize, on: bool) !void {
        try flipBf16(c.r, c.sums.sums + (d - 1) * DM_W * 2, on, &c.saved);
    }
    pub fn prep(c: *DecPost, d: usize) !void {
        try c.sums.prep(d);
    }
    pub fn reset(c: *DecPost) !void {
        try c.r.d.check(c.r.d.api.cuMemcpyDtoDAsync_v2(c.h, c.h_in, dec_max * DM_RH, c.r.s.handle), "cuMemcpyDtoDAsync");
    }
    pub fn whole(c: *DecPost, d: usize) !void {
        try dsv41.tri_basic.hcPost(c.r.t, c.sums.planes, c.h, c.post, c.comb, c.h, 2, d, DM_D);
    }
    pub fn slab(c: *DecPost, d: usize, k: usize, s: usize, m: usize) !void {
        const q = dsv41.slab_plan.q(d);
        try dsv41.tri_basic.hcPost2b(c.r.t, c.sums.blk[k], (q - c.short) * DM_W, c.h + s * DM_RH, c.post + s * 16, c.comb + s * 64, c.h + s * DM_RH, m, DM_D);
    }
    pub fn after(c: *DecPost, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn read(c: *DecPost, d: usize) ![]u8 {
        return get(c.h, d * DM_RH, c.r.d);
    }
};

/// A DSpark tap's mean of the streams (hcMean4) into the tap's rows: the window's d rows in one launch, or each slab's.
const DecMean = struct {
    r: *Run,
    ex: *const dsv41.exact.Exact,
    h: u64,
    out: u64,
    saved: u16 = 0,

    pub const strided = false;

    /// The window's last row's first stream value.
    pub fn perturb(c: *DecMean, d: usize, on: bool) !void {
        try flipBf16(c.r, c.h + (d - 1) * DM_RH, on, &c.saved);
    }

    pub fn prep(c: *DecMean, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn reset(c: *DecMean) !void {
        try poison(c.out, dec_max * DM_D * 2, c.r.d, c.r.s);
    }
    pub fn whole(c: *DecMean, d: usize) !void {
        try c.ex.hcMean4(c.r.s, c.h, c.out, DM_D, d, DM_D);
    }
    pub fn slab(c: *DecMean, d: usize, k: usize, s: usize, m: usize) !void {
        _ = d;
        _ = k;
        try c.ex.hcMean4(c.r.s, c.h + s * DM_RH, c.out + s * DM_D * 2, DM_D, m, DM_D);
    }
    pub fn after(c: *DecMean, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn read(c: *DecMean, d: usize) ![]u8 {
        return get(c.out, d * DM_D * 2, c.r.d);
    }
};

/// The experts' gate / up at a window of d rows: the baseline mx.gateUp quantizing the d rows itself; the owned path's act_quant of each slab's rows
/// into the scratch (ffnRows) and mx.gateUpCodes on the codes the slabs left.
const DecExperts = struct {
    r: *Run,
    ex: weights.Experts,
    sc: mx.ExpertScratch,
    x: u64,
    pick: u64,
    wts: u64,
    hb: u64,
    saved: u16 = 0,

    pub const strided = false;

    /// The window's last row's first activation value (its codes and its block's scale change).
    pub fn perturb(c: *DecExperts, d: usize, on: bool) !void {
        try flipBf16(c.r, c.x + (d - 1) * X_D * 2, on, &c.saved);
    }

    pub fn prep(c: *DecExperts, d: usize) !void {
        _ = c;
        _ = d;
    }
    pub fn reset(c: *DecExperts) !void {
        try poison(c.sc.q, dec_max * X_D, c.r.d, c.r.s);
        try poison(c.sc.s, dec_max * (X_D / 32) * 4, c.r.d, c.r.s);
        try poison(c.hb, dec_max * X_SL * X_W * 2, c.r.d, c.r.s);
    }
    pub fn whole(c: *DecExperts, d: usize) !void {
        try mx.gateUp(c.r.k, c.r.s, c.r.d, c.ex, c.sc, c.x, X_D, c.pick, c.wts, d, 7.0, c.hb, X_MAIN, false);
    }
    pub fn slab(c: *DecExperts, d: usize, k: usize, s: usize, m: usize) !void {
        _ = d;
        _ = k;
        try mx.quant(c.r.k, c.r.s, c.x + s * X_D * 2, .bf16, X_D, m, X_D, c.sc.q + s * X_D, X_D, c.sc.s + s * (X_D / 32) * 4, X_D / 32);
    }
    pub fn after(c: *DecExperts, d: usize) !void {
        try mx.gateUpCodes(c.r.k, c.r.s, c.r.d, c.ex, c.sc, c.pick, c.wts, d, 7.0, c.hb, X_MAIN);
    }
    pub fn read(c: *DecExperts, d: usize) ![]u8 {
        return get(c.hb, d * X_SL * X_W * 2, c.r.d);
    }
};

// Every decoder class's interface (replay_runner.zig), public: the runner calls each method on every class (a private method is invisible
// to @hasDecl, and a dispatch by it skips the method silently).
comptime {
    for ([_]type{ DecMix, DecPost, DecMean, DecExperts }) |T| for ([_][]const u8{ "prep", "reset", "whole", "slab", "after", "read", "perturb", "strided" }) |m| {
        if (!@hasDecl(T, m)) @compileError("the decoder class " ++ @typeName(T) ++ " has no public " ++ m);
    };
}

/// A decoder class at every window of dec_ds through the runner (replay_runner.runClass: the null control before the comparison, the
/// controls); a failing step names the section, the class, the window, the slab and the step.
fn decClassRun(r: *Run, section: []const u8, name: []const u8, c: anytype) !dsv41.replay_runner.Result {
    var fail: dsv41.replay_runner.Failure = .{};
    return dsv41.replay_runner.runClass(r.s, c, &dec_ds, &fail) catch |e| {
        if (fail.slab) |sl| {
            std.debug.print("{s} {s}: FAILED at window d = {d}, slab {d} (start {d}, {d} rows), {s}: {s}\n", .{ section, name, fail.d, sl[0], sl[1], sl[2], fail.step, @errorName(e) });
        } else std.debug.print("{s} {s}: FAILED at window d = {d}, {s}: {s}\n", .{ section, name, fail.d, fail.step, @errorName(e) });
        return e;
    };
}

/// A decoder class's lines: the comparison, the null control (named failure: which run, at which window), the control and, strided, the
/// stride control.
fn decReport(r: *Run, section: []const u8, what: []const u8, res: dsv41.replay_runner.Result, stride: bool) !void {
    var line: [512]u8 = undefined;
    const name = what[0..std.mem.indexOfScalar(u8, what, ':').?];
    r.report(try std.fmt.bufPrint(&line, "{s} {s}, at {s}", .{ section, what, dec_plan.text }), res.bad, true);
    if (res.first_null) |f| {
    r.report(try std.fmt.bufPrint(&line, "{s} {s} control: the last slab one row short, in all {d} windows", .{ section, name, dec_ds.len }), if (res.cut_ok == dec_ds.len) 1 else 0, false);
    if (stride) r.report(try std.fmt.bufPrint(&line, "{s} {s} stride control: the slabs' blocks read q - 1 rows apart, in all {d} windows", .{ section, name, dec_ds.len }), if (res.stride_ok == dec_ds.len) 1 else 0, false);
}

/// The decoder half on owned rows, the launch classes the encoder half's sections do not cover (roci_dec_rows runs those at the windows):
/// the posted mixes and the posts from the transport's two blocks at a window's plan (stride q(d), the last slab short) against the baseline
/// readers of the quarters' planes over the window's rows, a DSpark tap's mean, and the experts' gate / up from the slabs' codes, at every
/// window d = 1 to 128; each with its control.
fn rociDecMixes(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const section = "roci_dec_mixes";
    var prng = std.Random.DefaultPrng.init(0xdec0);
    const rng = prng.random();
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const M = dec_max;
    const h = try upload(d, try bf16s(M * 4 * DM_D, rng, 1.0, ha));
    try keep.append(ha, h);
    const mix_what = "hcPrePf: _hc_mix_pf_2b + _hc_finish on each slab's two blocks at the plan's stride q against _hc_mix_pf + _hc_finish on the planes over the window's d rows (the streams, x, pre, post, comb)";
    const post_what = "hcPost: _hc_post_2b in place on each slab's rows from its two blocks against _hc_post on the planes over the window's d rows";
    if (r.t.extra == null or r.ops == null) {
        for ([_][]const u8{ "hcPrePf", "hcPost" }) |nm| for ([_][]const u8{ "", " null control", " control", " stride control" }) |sfx| {
            var nl: [160]u8 = undefined;
            r.note(try std.fmt.bufPrint(&nl, "{s} {s}{s}: the kit's extra set (aot-extra/) loads and the ops are available", .{ section, nm, sfx }), false);
        };
    } else {
        var sums: DecSums = .{ .r = r, .sums = 0, .planes = 0, .blk = undefined };
        const sb = try upload(d, try bf16s(2 * M * DM_W, rng, 1.0, ha));
        try keep.append(ha, sb);
        sums.sums = sb.ptr;
        const pl = try cuda.DeviceBuffer.alloc(d, 2 * M * DM_D * 4);
        try keep.append(ha, pl);
        sums.planes = pl.ptr;
        for (&sums.blk) |*b| {
            const bb = try cuda.DeviceBuffer.alloc(d, 2 * 32 * DM_W * 2);
            try keep.append(ha, bb);
            b.* = bb.ptr;
        }
        // the mixes' weights small enough that the Sinkhorn finish stays finite
        const fnw = try upload(d, try f32sAt(24 * 4 * DM_D, rng, 0.01, ha));
        const scale = try upload(d, try f32sAt(3, rng, 1.0, ha));
        const base = try upload(d, try f32sAt(24, rng, 1.0, ha));
        const pre_in = try upload(d, try f32sAt(M * 4, rng, 0.5, ha));
        const nw = try upload(d, try bf16s(DM_D, rng, 1.0, ha));
        const post_in = try upload(d, try f32sAt(M * 4, rng, 0.5, ha));
        const comb_in = try upload(d, try f32sAt(M * 16, rng, 0.5, ha));
        for ([_]cuda.DeviceBuffer{ fnw, scale, base, pre_in, nw, post_in, comb_in }) |b| try keep.append(ha, b);
        var outs: [6]u64 = undefined;
        for (&outs, [_]usize{ M * DM_RH, M * DM_D * 2, M * 16, M * 16, M * 64, M * DecMix.part_row }) |*o, n| {
            const b = try cuda.DeviceBuffer.alloc(d, n);
            try keep.append(ha, b);
            o.* = b.ptr;
        }
        var mix: DecMix = .{ .r = r, .sums = &sums, .h = h.ptr, .fnw = fnw.ptr, .scale = scale.ptr, .base = base.ptr, .pre_in = pre_in.ptr, .nw = nw.ptr, .post_in = post_in.ptr, .comb_in = comb_in.ptr, .h_out = outs[0], .x = outs[1], .pre_o = outs[2], .post = outs[3], .comb = outs[4], .part = outs[5] };
        try decReport(r, section, mix_what, try decClassRun(r, section, "hcPrePf", &mix), true);
        const hw = try cuda.DeviceBuffer.alloc(d, M * DM_RH);
        try keep.append(ha, hw);
        var post: DecPost = .{ .r = r, .sums = &sums, .h_in = h.ptr, .h = hw.ptr, .post = post_in.ptr, .comb = comb_in.ptr };
        try decReport(r, section, post_what, try decClassRun(r, section, "hcPost", &post), true);
    }
    // a tap's mean
    const mean_what = "hcMean4 (a DSpark tap's mean): each slab's rows against the window's one launch";
    if (dsv41.exact.Exact.load(d, cuda.kernels.dsv41_torch)) |ex_loaded| {
        var ex = ex_loaded;
        defer ex.unload();
        const out = try cuda.DeviceBuffer.alloc(d, M * DM_D * 2);
        try keep.append(ha, out);
        var mean: DecMean = .{ .r = r, .ex = &ex, .h = h.ptr, .out = out.ptr };
        try decReport(r, section, mean_what, try decClassRun(r, section, "hcMean4", &mean), false);
    } else |e| {
        var nl: [160]u8 = undefined;
        r.note(try std.fmt.bufPrint(&nl, "{s} hcMean4: the torch kernels load ({s})", .{ section, @errorName(e) }), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcMean4 null control: the torch kernels load", .{section}), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcMean4 control: the torch kernels load", .{section}), false);
    }
    // the experts' gate / up
    const ex = try xExperts(d, rng, ha, &keep);
    const sc = try xScratch(d, M, true, ha, &keep);
    const x = try upload(d, try bf16s(M * X_D, rng, 0.5, ha));
    const pw = try xPicks(M, rng, ha);
    const pb = try upload(d, std.mem.sliceAsBytes(pw.pick));
    const wb = try upload(d, std.mem.sliceAsBytes(pw.wts));
    const hb = try cuda.DeviceBuffer.alloc(d, M * X_SL * X_W * 2);
    for ([_]cuda.DeviceBuffer{ x, pb, wb, hb }) |b| try keep.append(ha, b);
    var gu: DecExperts = .{ .r = r, .ex = ex, .sc = sc, .x = x.ptr, .pick = pb.ptr, .wts = wb.ptr, .hb = hb.ptr };
    try decReport(r, section, "gate / up: act_quant at each slab's rows and mx.gateUpCodes on the slabs' codes against mx.gateUp's own quantization of the window's d rows", try decClassRun(r, section, "gate / up", &gu), false);
}

/// The kit's base and extra sets, looked up as a launch finds its variant (Set.lookup: find's variant, without its log); `hide` a name the
/// extra set is taken not to hold (the control).
const KitFinds = struct {
    set: *const cuda.aot.Set,
    extra: ?*const cuda.aot.Set,
    hide: ?[]const u8 = null,

    pub fn has(f: KitFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        return f.set.lookup(name, args, consts) != null;
    }
    pub fn hasExtra(f: KitFinds, name: []const u8, args: []const cuda.aot.Arg, consts: []const cuda.aot.Const) bool {
        if (f.hide) |x| if (std.mem.eql(u8, x, name)) return false;
        const set = f.extra orelse return false;
        return set.lookup(name, args, consts) != null;
    }
};

/// The decoder half's start check on this kit (any_check.decoderOff: every launch of layers 0-39 at every slab of a decoder window of 1 to 128
/// rows, the two-block readers in the extra set), TF_DS_ROCI_ANY's (layers 0-19, chunks of 1 to 4,096 rows) with the readers looked up as well,
/// and the control: the extra set taken to lack _hc_post_2b, which the decoder half's check must name.
fn rociDecKset(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const section = "roci_dec_kset";
    var line: [512]u8 = undefined;
    const gpa = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var why: dsv41.config.Why = .{};
    const cfg = try dsv41.config.parse(a, dsv41.config.test_config, &why);
    var lays: [40]weights.Layer = undefined;
    dsv41.any_check.layers(&lays, cfg);
    const finds: KitFinds = .{ .set = r.t.set.?, .extra = r.t.extra };
    const dec = try dsv41.any_check.decoderOff(gpa, a, finds, cfg, &lays);
    r.note(try std.fmt.bufPrint(&line, "{s}: the decoder half's start check on this kit (every launch of layers 0-39 at every slab of a decoder window of 1 to 128 rows, the two-block readers in the extra set): {s}", .{ section, dec orelse "every variant found" }), dec == null);
    const enc = try dsv41.any_check.anyOwnedOff(gpa, a, finds, cfg, &lays, cfg.layers / 2, true, 4096);
    r.note(try std.fmt.bufPrint(&line, "{s}: TF_DS_ROCI_ANY's start check on this kit (layers 0-19, chunks of 1 to 4,096 rows) with the two-block readers looked up: {s}", .{ section, enc orelse "every variant found" }), enc == null);
    const ctl = try dsv41.any_check.decoderOff(gpa, a, KitFinds{ .set = r.t.set.?, .extra = r.t.extra, .hide = "_hc_post_2b" }, cfg, &lays);
    r.note(try std.fmt.bufPrint(&line, "{s} control: the extra set taken to lack _hc_post_2b, the decoder half's check names it", .{section}), ctl != null and std.mem.indexOf(u8, ctl.?, "_hc_post_2b") != null);
}

/// three that landed out to their rows: every array of every node must hold the bytes the in-place gather leaves, the groups around
                r.report(try std.fmt.bufPrint(&line, "roci_boundary {s}: four nodes' cache arrays after the staged gather against the in-place gather's layout (rows around the chunk untouched)", .{nm}), bad, true);
// -- TF_DS_ROCI_STAGGER ----------------------------------------------------------------------------------------------

const PairNode = struct {
    arena: []align(64) u8,
    arena_dev: u64,
    ctl: *dsv41.rocinante.Control,
    ctl_dev: u64,
    t: *dsv41.rocinante.Transport = undefined,
    peer: *PairNode = undefined,
    me: u8 = 0,
    hold: std.atomic.Value(bool) = .init(false),
};

fn pairHostAlloc(d: *const cuda.Driver, len: usize, flags: c_uint) !struct { p: [*]align(64) u8, dev: u64 } {
    var p: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&p, len, flags), "cuMemHostAlloc");
    var dev: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, p, 0), "cuMemHostGetDevicePointer");
    return .{ .p = @ptrCast(@alignCast(p.?)), .dev = dev };
}

fn pairNap(ns: u64) void {
    const ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}

fn pairPost(ctx: *anyopaque, rail: u8, peer: u8, class: dsv41.rocinante.Class, chain: *const dsv41.rocinante.Chain, word: u64) dsv41.rocinante.PostError!void {
    const me: *PairNode = @ptrCast(@alignCast(ctx));
    _ = peer;
    while (me.hold.load(.acquire)) pairNap(100_000);
    for (chain.slice()) |p| @memcpy(me.peer.arena[p.dst..][0..p.len], me.arena[p.src..][0..p.len]);
    @atomicStore(u64, &me.peer.ctl.recv[@backingInt(class)][me.me][rail].v, word, .release);
}
fn pairFail(_: *anyopaque) void {}
fn pairReap(_: *anyopaque, _: u8) dsv41.rocinante.PostError!void {}

const PairLane = struct {
    nodes: [2]PairNode = undefined,

    fn open(l: *PairLane, d: *const cuda.Driver) !void {
        for (&l.nodes, 0..) |*n, i| {
            const a = try pairHostAlloc(d, 64 << 20, 0x01 | 0x02 | 0x04);
            const c = try pairHostAlloc(d, @sizeOf(dsv41.rocinante.Control), 0x01 | 0x02);
            n.* = .{ .arena = a.p[0 .. 64 << 20], .arena_dev = a.dev, .ctl = @ptrCast(c.p), .ctl_dev = c.dev, .me = @intCast(i) };
            n.ctl.* = .{};
        }
        for (&l.nodes, 0..) |*n, i| {
            n.peer = &l.nodes[1 - i];
            n.t = try dsv41.rocinante.Transport.init(std.heap.page_allocator, @intCast(i), dsv41.rocinante.identity, n.ctl, n.ctl_dev, .{ .ctx = n, .post_fn = pairPost, .fail_fn = pairFail, .reap_fn = pairReap }, .{ .nap = false });
            try n.t.start(null);
        }
    }

    fn close(l: *PairLane, d: *const cuda.Driver) void {
        for (&l.nodes) |*n| n.t.deinit();
        for (&l.nodes) |*n| {
            _ = d.api.cuMemFreeHost(n.arena.ptr);
            _ = d.api.cuMemFreeHost(@ptrCast(n.ctl));
        }
    }
};

fn pairWord(lane: u32, node: u32, block: u32, gen: u32) u32 {
    return (lane << 30) | (node << 28) | (block << 20) | gen;
}

fn pairWrite(d: *const cuda.Driver, nd: *PairNode, s: cuda.Stream, lane: u32, gen: u32) !void {
    const l = dsv41.rocinante.layout();
    const blk: usize = 1024 * 2560 * 4;
    for (0..4) |b| try d.check(d.api.cuMemsetD32Async(nd.arena_dev + l.at(.psum) + b * blk, pairWord(lane, nd.me, @intCast(b), gen), blk / 4, s.handle), "memset");
}

/// Words of node `nd`'s stage regions that differ from the peer's lane `from` words.
fn pairStageBad(nd: *PairNode, from: u32, gen: u32) usize {
    const l = dsv41.rocinante.layout();
    const blk: usize = 1024 * 2560 * 4;
    const peer: u32 = 1 - nd.me;
    const w = [2]u32{ pairWord(from, peer, nd.me, gen), pairWord(from, peer, nd.me ^ 2, gen) };
    var bad: usize = 0;
    for (0..2) |k| {
        const got = std.mem.bytesAsSlice(u32, nd.arena[l.at(.stage) + k * blk ..][0..blk]);
        for (got) |x| bad += @intFromBool(x != w[k]);
    }
    return bad;
}

fn pairStreamDone(d: *const cuda.Driver, s: cuda.Stream, ms: u64) bool {
    var waited: u64 = 0;
    while (waited < ms) : (waited += 1) {
        if (d.api.cuStreamQuery(s.handle) == 0) return true;
        pairNap(1_000_000);
    }
    return false;
}

/// Two passes of a pair on one rank's two lanes (each lane its own arena, control block, transport; two nodes a lane, one stream a node): both
/// lanes' exchanges in flight at once land every byte in their own lane; a lane-1 peer that is late holds lane 1's wait while lane 0's
/// streams finish; lane 1 miswired into lane 0's arena must differ; pass B's cache read behind the event sees A's write and, without the event,
/// does not.
fn rociPair(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const d = r.d;
    var lanes: [2]PairLane = undefined;
    try lanes[0].open(d);
    try lanes[1].open(d);
    defer for (&lanes) |*l| l.close(d);
    var st: [2][2]cuda.Stream = undefined;
    for (&st) |*row| {
        for (row) |*x| x.* = try cuda.Stream.init(d, true);
    }
    defer for (&st) |*row| {
        for (row) |*x| x.deinit();
    };
    var bad: usize = 0;
    for (1..5) |gen| {
        for (0..2) |lane| {
            for (0..2) |n| {
                try pairWrite(d, &lanes[lane].nodes[n], st[lane][n], @intCast(lane), @intCast(gen));
                try lanes[lane].nodes[n].t.send(st[lane][n], .own1, 0);
            }
        }
        for (0..2) |lane| {
            for (0..2) |n| try lanes[lane].nodes[n].t.wait(st[lane][n], .own1);
        }
        for (0..2) |lane| {
            for (0..2) |n| try st[lane][n].synchronize();
        }
        for (0..2) |lane| {
            for (0..2) |n| bad += pairStageBad(&lanes[lane].nodes[n], @intCast(lane), @intCast(gen));
        }
    }
    r.report("roci_pair: two lanes in flight at once, four generations: every word of each lane's stage regions is its own lane's", bad, true);
    // lane 1's node 1 is late; lane 0's streams finish meanwhile and lane 1's node 0 still waits
    lanes[1].nodes[1].hold.store(true, .release);
    for (0..2) |n| {
        try pairWrite(d, &lanes[1].nodes[n], st[1][n], 1, 9);
        try lanes[1].nodes[n].t.send(st[1][n], .own1, 0);
    }
    for (0..2) |n| {
        try pairWrite(d, &lanes[0].nodes[n], st[0][n], 0, 9);
        try lanes[0].nodes[n].t.send(st[0][n], .own1, 0);
        try lanes[0].nodes[n].t.wait(st[0][n], .own1);
    }
    try lanes[1].nodes[0].t.wait(st[1][0], .own1);
    const live = pairStreamDone(d, st[0][0], 3000) and pairStreamDone(d, st[0][1], 3000);
    const pending = !pairStreamDone(d, st[1][0], 150);
    r.note("roci_pair: lane 0's streams finish while lane 1's wait for its late peer is pending (a pending wait on one stream holds the other)", live and pending);
    r.report("roci_pair control: the wait for lane 1's words not yet released, lane 1's stage holds lane 1's bytes", pairStageBad(&lanes[1].nodes[0], 1, 9), false);
    lanes[1].nodes[1].hold.store(false, .release);
    try lanes[1].nodes[1].t.wait(st[1][1], .own1);
    try st[1][0].synchronize();
    try st[1][1].synchronize();
    r.report("roci_pair: after the late peer posts, lane 1's stage holds its bytes", pairStageBad(&lanes[1].nodes[0], 1, 9) + pairStageBad(&lanes[0].nodes[0], 0, 9), true);
    // the cache gate
    const flag = try pairHostAlloc(d, 4096, 0x01 | 0x02);
    defer _ = d.api.cuMemFreeHost(flag.p);
    const flag_p: *u64 = @ptrCast(@alignCast(flag.p));
    var cache = try cuda.DeviceBuffer.alloc(d, 4096);
    defer cache.free();
    var out = try cuda.DeviceBuffer.alloc(d, 8192);
    defer out.free();
    var ev = try cuda.Event.init(d, false);
    defer ev.deinit();
    var diff: [2]usize = .{ 0, 0 };
    for ([_]bool{ true, false }, 0..) |gated, gi| {
        flag_p.* = 0;
        const base: u64 = if (gated) 0 else 4096;
        const val: u32 = if (gated) 0x22222222 else 0x33333333;
        try st[0][0].waitValues64(&.{flag.dev}, &.{1});
        try d.check(d.api.cuMemsetD32Async(cache.ptr, val, 1024, st[0][0].handle), "memset");
        try ev.record(st[0][0]);
        if (gated) try st[1][0].wait(ev);
        try d.check(d.api.cuMemcpyDtoDAsync_v2(out.ptr + base, cache.ptr, 4096, st[1][0].handle), "copy");
        pairNap(100_000_000);
        @atomicStore(u64, flag_p, 1, .release);
        try st[0][0].synchronize();
        try st[1][0].synchronize();
        var got: [1024]u32 = undefined;
        for (got) |x| diff[gi] += @intFromBool(x != val);
    }
    r.report("roci_pair: pass B's read of a layer's cache behind pass A's event is A's write", diff[0], true);
    r.report("roci_pair control: the same read without the event", diff[1], false);
}

/// TF_DS_ROCI_STAGGER_LANES=3: three passes on one rank's three lanes (each its own arena, control block, transport; two nodes a lane, one stream a
/// node): all three lanes' exchanges in flight at once land every byte in their own lane; and the cache gate chains, pass C's read of a layer's
/// cache behind B's event behind A's event is A's write, and with B's wait left out is not.
fn rociTriple(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    const d = r.d;
    var lanes: [3]PairLane = undefined;
    for (&lanes) |*l| try l.open(d);
    defer for (&lanes) |*l| l.close(d);
    var st: [3][2]cuda.Stream = undefined;
    for (&st) |*row| {
        for (row) |*x| x.* = try cuda.Stream.init(d, true);
    }
    defer for (&st) |*row| {
        for (row) |*x| x.deinit();
    };
    var bad: usize = 0;
    for (1..5) |gen| {
        for (0..3) |lane| {
            for (0..2) |n| {
                try pairWrite(d, &lanes[lane].nodes[n], st[lane][n], @intCast(lane), @intCast(gen));
                try lanes[lane].nodes[n].t.send(st[lane][n], .own1, 0);
            }
        }
        for (0..3) |lane| {
            for (0..2) |n| try lanes[lane].nodes[n].t.wait(st[lane][n], .own1);
        }
        for (0..3) |lane| {
            for (0..2) |n| try st[lane][n].synchronize();
        }
        for (0..3) |lane| {
            for (0..2) |n| bad += pairStageBad(&lanes[lane].nodes[n], @intCast(lane), @intCast(gen));
        }
    }
    r.report("roci_triple: three lanes in flight at once, four generations: every word of each lane's stage regions is its own lane's", bad, true);
    // the cache gate chain A -> B -> C
    const flag = try pairHostAlloc(d, 4096, 0x01 | 0x02);
    defer _ = d.api.cuMemFreeHost(flag.p);
    const flag_p: *u64 = @ptrCast(@alignCast(flag.p));
    var cache = try cuda.DeviceBuffer.alloc(d, 4096);
    defer cache.free();
    var mid = try cuda.DeviceBuffer.alloc(d, 8192);
    defer mid.free();
    var out = try cuda.DeviceBuffer.alloc(d, 8192);
    defer out.free();
    var ev_a = try cuda.Event.init(d, false);
    defer ev_a.deinit();
    var ev_b = try cuda.Event.init(d, false);
    defer ev_b.deinit();
    var diff: [2]usize = .{ 0, 0 };
    for ([_]bool{ true, false }, 0..) |gated, gi| {
        flag_p.* = 0;
        const base: u64 = if (gated) 0 else 4096;
        const val: u32 = if (gated) 0x22222222 else 0x33333333;
        // stale contents everywhere, so a read that runs early sees a value that is not `val`
        try d.check(d.api.cuMemsetD32Async(cache.ptr, 0x11111111, 1024, st[0][0].handle), "memset");
        try d.check(d.api.cuMemsetD32Async(mid.ptr + base, 0x11111111, 1024, st[0][0].handle), "memset");
        try st[0][0].synchronize();
        try st[0][0].waitValues64(&.{flag.dev}, &.{1});
        try d.check(d.api.cuMemsetD32Async(cache.ptr, val, 1024, st[0][0].handle), "memset");
        try ev_a.record(st[0][0]);
        if (gated) try st[1][0].wait(ev_a);
        try d.check(d.api.cuMemcpyDtoDAsync_v2(mid.ptr + base, cache.ptr, 4096, st[1][0].handle), "copy");
        try ev_b.record(st[1][0]);
        try st[2][0].wait(ev_b); // C always waits B: the chain breaks only where B skips A
        try d.check(d.api.cuMemcpyDtoDAsync_v2(out.ptr + base, mid.ptr + base, 4096, st[2][0].handle), "copy");
        pairNap(100_000_000);
        @atomicStore(u64, flag_p, 1, .release);
        try st[0][0].synchronize();
        try st[1][0].synchronize();
        try st[2][0].synchronize();
        var got: [1024]u32 = undefined;
        for (got) |x| diff[gi] += @intFromBool(x != val);
    }
    r.report("roci_triple: pass C's read of a layer's cache behind B's event behind A's event is A's write", diff[0], true);
    r.report("roci_triple control: the same chain with B's wait on A's event left out", diff[1], false);
}

    .{ "ring_word", ringWord, 9 },
    .{ "roci_readers", rociReaders, 1 + 7 * 3 + 3 },
    .{ "roci_stage", rociStage, 2 * 4 + 1 },
    .{ "roci_collect", rociCollect, 3 },
    .{ "roci_any_rows", rociAnyRows, 3 * 8 },
    .{ "roci_any_index", rociAnyIndex, 6 },
    .{ "roci_any_engram", rociAnyEngram, 5 },
    .{ "roci_dec_rows", rociDecRows, 3 * 8 },
    .{ "roci_dec_mixes", rociDecMixes, 4 + 4 + 3 + 3 },
    .{ "roci_dec_kset", rociDecKset, 3 },
    .{ "roci_pair", rociPair, 6 },
    .{ "roci_triple", rociTriple, 3 },