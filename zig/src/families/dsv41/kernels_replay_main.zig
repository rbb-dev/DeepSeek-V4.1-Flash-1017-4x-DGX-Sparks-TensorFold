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
//!   - the cost of splitting wo_b's GEMM and the MoE combine into four row quarters (TF_DS_ROCI sends two of four): kernel-only times, device y
//!     and arena y, 4,096 rows as one launch and as four of 1,024.
//!   - TF_DS_ROCI_COMBINE_FIRST's routed sum (the quarters an owner sum's first part sends, a host step, the quarters the node keeps)
//!     against one tf_dsn_combine launch of 4,096 rows, into device memory and into write-combined host memory, byte for byte; the
//!     host step sees the sent quarters complete and the kept ones unwritten; a kept quarter left out must differ.
//!   - the kernel classes the owner path moves onto the arena (roci_wc_classes: act_quant in planar and in records, hcPrePf, ffnRoute,
//!     rmsnorm, pf_gu and down from codes, the block-scaled GEMM from codes, sparseAttn from cidx), kernel only at the 4,096-row shapes,
//!     into or out of cudaMalloc, a second cudaMalloc set (the null control, within 2 %) and write-combined host memory: ms a launch and a pass.
//!   - TF_DS_ROCI_GATHERS' boundary staging (roci_boundary): four nodes' cache arrays staged into the arena's boundary region, the slabs
//!     written to the other three regions, copied out to their rows, against the in-place gather's layout, with the region in cudaMalloc, in
//!     write-combined host memory and in locked registered pages; a node that stages nothing must leave its rows missing; the ms of the
//!     16 copies.
//!   - TF_DS_ROCI_CODES' two exchanges on two column partners (roci_codes): wo_b's part quantized into the arena's send region, written to
//!     the partner's land region, read by the four copies that assemble wo_b's input; the experts' pack quantized, written, read by the
//!     assembly into down's input; the regions in cudaMalloc, write-combined host memory and locked registered pages: every consumer byte
//!     equal to the run in device memory; a pack that never lands must differ; the ms of the quantizers, the copies and the assembly.
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
//!   - TF_DS_QGATE_CANARY 2 / 3 (the body canary, mx.qgateBody) on layer 0's MoE partials' shapes (1, 3, 17, 48 and
//!     2,048 rows of 2,560 and 5,120 fp32 columns; every group at its own power-of-two magnitude, zero groups,
//!     negative zeros, a value far above its group): mode 2 in place against act_quant's quant-dequant of the same
//!     values (mx.fake, bf16, widened) byte-equal, and the values it moved; mode 3 against each value's bits plus one
//!     (zeros, infinities, NaN and the largest finite values kept) byte-equal.
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

// -- the cost of splitting wo_b and combine into row quarters (TF_DS_ROCI sends only two of four row blocks) ----------

/// Milliseconds a launch of `f` takes, kernel only: `reps` launches between two timing events, no readback, the mean.
fn kernelMs(r: *Run, comptime Ctx: type, ctx: Ctx, comptime f: fn (Ctx) anyerror!void, reps: usize) !f32 {
    var t0 = try cuda.Event.init(r.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(r.d, true);
    defer t1.deinit();
    for (0..3) |_| try f(ctx); // warm
    try t0.record(r.s);
    for (0..reps) |_| try f(ctx);
    try t1.record(r.s);
    try t1.synchronize();
    return (try cuda.Event.elapsedMs(t0, t1)) / @as(f32, @floatFromInt(reps));
}

/// wo_b's block-scaled exact GEMM (K 4096, N 2560) and the MoE combine (7 slots, 2,560 columns), each as one launch of 4,096 rows
/// and as four of 1,024 (the sent quarters are not adjacent), into device memory and into the arena's write-combined host memory:
/// kernel time only (events around the launches, no readback). The combine indexes rows relative to its arguments, so a row start
/// is a pointer offset.
fn rociSplitCost(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x59117);
    const rng = prng.random();
    const d = r.d;
    var line: [640]u8 = undefined;
    const K: usize = 4096;
    const N: usize = 2560;
    const M: usize = 4096;
    var arena_mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&arena_mem, M * N * 4, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(arena_mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem, 0), "cuMemHostGetDevicePointer");
    try r.k.useStaged(wc, wc + M * N * 4);
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    const x = try upload(d, try bf16s(M * K, rng, 0.5, ha));
    try keep.append(ha, x);
    const q = try cuda.DeviceBuffer.alloc(d, M * K);
    try keep.append(ha, q);
    const sc = try cuda.DeviceBuffer.alloc(d, M * (K / 32) * 4);
    try keep.append(ha, sc);
    try mx.quant(r.k, r.s, x.ptr, .bf16, K, M, K, q.ptr, K, sc.ptr, K / 32);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, M), 1) * 4);
    try keep.append(ha, z);
    const y = try cuda.DeviceBuffer.alloc(d, M * N * 4);
    try keep.append(ha, y);
    if (r.k.sfk == null) return r.note("roci_split_cost: the block-scaled kernels (native_sf.cu) are loaded on this GPU", false);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    const G = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        sc: u64,
        z: u64,
        y: u64,
        parts: usize,
        fn go(g: @This()) anyerror!void {
            const m = M / g.parts;
            for (0..g.parts) |i| try mx.gemm(g.r.k, g.r.s, g.l, g.q + i * m * K, K, g.sc + i * m * (K / 32) * 4, K / 32, m, g.y + i * m * N * 4, N, .f32, false, g.z);
        }
    };
    var g_ms: [2][2]f32 = undefined; // [y: device, arena][parts: 1, 4]
    for ([_]u64{ y.ptr, wc }, 0..) |yy, yi| {
        r.k.wc = if (yi == 1) .{ wc, wc + M * N * 4 } else .{ 0, 0 };
        for ([_]usize{ 1, 4 }, 0..) |parts, pi| g_ms[yi][pi] = try kernelMs(r, G, .{ .r = r, .l = l, .q = q.ptr, .sc = sc.ptr, .z = z.ptr, .y = yy, .parts = parts }, G.go, 20);
    }
    const txt = try std.fmt.bufPrint(&line, "roci_split_cost wo_b (K {d}, N {d}) block-scaled exact, kernel only, ms for 4,096 rows: device y {d:.3} as 1 launch, {d:.3} as 4 x 1,024; arena y (whole-line stores) {d:.3} as 1, {d:.3} as 4 x 1,024", .{ K, N, g_ms[0][0], g_ms[0][1], g_ms[1][0], g_ms[1][1] });
    r.note(txt, true);
    std.debug.print("{s}\n", .{txt});
    // combine: 4,096 rows x 7 slots, expert ids distinct in a row
    const S: usize = 7;
    const E: usize = 384;
    const picks = try ha.alloc(i32, M * S);
    for (0..M) |row| {
        var seen: [S]i32 = undefined;
        var k: usize = 0;
        while (k < S) {
            const e: i32 = @intCast(rng.uintLessThan(usize, E));
            if (std.mem.indexOfScalar(i32, seen[0..k], e) != null) continue;
            seen[k] = e;
            picks[row * S + k] = e;
            k += 1;
        }
    }
    const pk = try upload(d, std.mem.sliceAsBytes(picks));
    try keep.append(ha, pk);
    const outs = try cuda.DeviceBuffer.alloc(d, M * S * N * 4);
    try keep.append(ha, outs);
    try d.check(d.api.cuMemsetD32Async(outs.ptr, 0x3f800000, M * S * N, r.s.handle), "cuMemsetD32Async");
    const C = struct {
        r: *Run,
        out: u64,
        pick: u64,
        y: u64,
        parts: usize,
        fn go(c: @This()) anyerror!void {
            const m = M / c.parts;
            for (0..c.parts) |i| {
                var ca: cuda.Args = .{};
                ca.add(c.out + i * m * S * N * 4);
                ca.add(@as(i64, @intCast(N)));
                ca.add(c.pick + i * m * S * 4);
                ca.add(@as(c_int, @intCast(m)));
                ca.add(@as(c_int, @intCast(S)));
                ca.add(@as(c_int, @intCast(E)));
                ca.add(@as(c_int, @intCast(N)));
                ca.add(c.y + i * m * N * 4);
                ca.add(@as(i64, @intCast(N)));
                ca.add(@as(c_int, 0));
                try cuda.launch.launch(c.r.k.combine, .{ .grid = .{ .x = @intCast((m * N + 255) / 256) }, .block = .{ .x = 256 } }, c.r.s, &ca);
            }
        }
    };
    var c_ms: [2][2]f32 = undefined;
    for ([_]u64{ y.ptr, wc }, 0..) |yy, yi| for ([_]usize{ 1, 4 }, 0..) |parts, pi| {
        c_ms[yi][pi] = try kernelMs(r, C, .{ .r = r, .out = outs.ptr, .pick = pk.ptr, .y = yy, .parts = parts }, C.go, 20);
    };
    const txt2 = try std.fmt.bufPrint(&line, "roci_split_cost combine (7 slots, N {d}), kernel only, ms for 4,096 rows: device y {d:.3} as 1 launch, {d:.3} as 4 x 1,024; arena y {d:.3} as 1, {d:.3} as 4 x 1,024", .{ N, c_ms[0][0], c_ms[0][1], c_ms[1][0], c_ms[1][1] });
    r.note(txt2, true);
    std.debug.print("{s}\n", .{txt2});
}

// -- TF_DS_ROCI_COMBINE_FIRST ------------------------------------------------------------------------------------------

/// What the host step between the combine's two stages sees (mx.CombineFirst.between): the stream drained, the first stage's quarters
/// equal to the reference's and the last stage's quarters not written yet.
const CombineStep = struct {
    r: *Run,
    y: u64,
    ref: []const u8,
    cf: mx.CombineFirst,
    row_bytes: usize,
    called: usize = 0,
    first_wrong: usize = 0,
    last_written: usize = 0,

    fn between(p: *anyopaque) anyerror!void {
        const c: *CombineStep = @ptrCast(@alignCast(p));
        c.called += 1;
        try c.r.s.synchronize();
        const blk = c.cf.rows * c.row_bytes;
        for (c.cf.first) |row0| {
            const got = try get(c.y + row0 * c.row_bytes, blk, c.r.d);
            defer free(got);
            c.first_wrong += differs(got, c.ref[row0 * c.row_bytes ..][0..blk]);
        }
        for (c.cf.last) |row0| {
            const got = try get(c.y + row0 * c.row_bytes, blk, c.r.d);
            defer free(got);
            for (got) |b| c.last_written += @intFromBool(b != 0xa5);
        }
    }
};

/// TF_DS_ROCI_COMBINE_FIRST's routed sum (mx.combineFirst: the quarters an owner sum's first part sends, a host step, the quarters the
/// node keeps) against tf_dsn_combine over all 4,096 rows in one launch (7 slots, 2,560 columns, random entries' outputs, picks
/// of distinct expert ids), the rows into device memory and into the arena's write-combined host memory: every output byte equal.
/// The host step sees the sent quarters complete and the kept ones unwritten. Controls: a kept quarter left out must differ.
fn rociCombineFirst(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xc0b1);
    const rng = prng.random();
    const d = r.d;
    var line: [320]u8 = undefined;
    const S: usize = 7;
    const E: usize = 384;
    const N: usize = 2560;
    const M: usize = 4096;
    const row_bytes = N * 4;
    var arena_mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&arena_mem, M * row_bytes, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(arena_mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem, 0), "cuMemHostGetDevicePointer");
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const picks = try ha.alloc(i32, M * S);
    for (0..M) |row| {
        var seen: [S]i32 = undefined;
        var k: usize = 0;
        while (k < S) {
            const e: i32 = @intCast(rng.uintLessThan(usize, E));
            if (std.mem.indexOfScalar(i32, seen[0..k], e) != null) continue;
            seen[k] = e;
            picks[row * S + k] = e;
            k += 1;
        }
    }
    const pk = try upload(d, std.mem.sliceAsBytes(picks));
    try keep.append(ha, pk);
    const outs = try upload(d, try f32s(M * S * N, rng, ha));
    try keep.append(ha, outs);
    const ydev = try cuda.DeviceBuffer.alloc(d, M * row_bytes);
    try keep.append(ha, ydev);
    var sc: mx.ExpertScratch = undefined;
    sc.slots = S;
    sc.out = outs.ptr;
    // the reference: one launch over every row
    try mx.combine(r.k, r.s, sc, pk.ptr, M, E, N, ydev.ptr, N, false);
    try r.s.synchronize();
    const ref = try get(ydev.ptr, M * row_bytes, d);
    defer free(ref);
    // a node of the second pair, its ranks in an order that is not the identity
    const pl = dsv41.prompt2d.ownerPlan(1, .{ 2, 0, 3, 1 }, M / 4);
    var step: CombineStep = .{ .r = r, .y = 0, .ref = ref, .cf = undefined, .row_bytes = row_bytes };
    const cf: mx.CombineFirst = .{ .rows = M / 4, .first = pl.send1, .last = pl.recv1, .ctx = &step, .between = CombineStep.between };
    step.cf = cf;
    const names = [_][]const u8{ "device memory", "the arena's write-combined host memory" };
    for ([_]u64{ ydev.ptr, wc }, 0..) |y, yi| {
        step.y = y;
        step.called = 0;
        step.first_wrong = 0;
        step.last_written = 0;
        try poison(y, M * row_bytes, d, r.s);
        try mx.combineFirst(r.k, r.s, sc, pk.ptr, M, E, N, y, N, false, &cf);
        try r.s.synchronize();
        const got = try get(y, M * row_bytes, d);
        defer free(got);
        r.report(try std.fmt.bufPrint(&line, "roci_combine_first y in {s}: the quarters sent first, a host step, the quarters kept, against one launch of 4,096 rows", .{names[yi]}), differs(got, ref), true);
        r.note(try std.fmt.bufPrint(&line, "roci_combine_first y in {s}: the host step ran once and saw the sent quarters equal to the reference ({d} bytes differ) and the kept ones unwritten ({d} bytes written)", .{ names[yi], step.first_wrong, step.last_written }), step.called == 1 and step.first_wrong == 0 and step.last_written == 0);
    }
    // control: the second kept quarter never launched leaves its rows unwritten, so the bytes differ from the reference
    try poison(wc, M * row_bytes, d, r.s);
    for (cf.first) |row0| try mx.combineRows(r.k, r.s, sc, pk.ptr, row0, cf.rows, E, N, wc, N, false);
    try mx.combineRows(r.k, r.s, sc, pk.ptr, cf.last[0], cf.rows, E, N, wc, N, false);
    try r.s.synchronize();
    const miss = try get(wc, M * row_bytes, d);
    defer free(miss);
    r.report("roci_combine_first control: one kept quarter left out, the arena's bytes", differs(miss, ref), false);
}

// -- the kernel classes on the arena's host memory, in every page type a NIC can register -----------------------------------------

/// Where a measured buffer lives: cudaMalloc, or host memory of one page type, all mapped into the GPU's address space:
/// `arena` cuMemHostAlloc portable|mapped|write-combined (the arena in the baseline, (b)); `cached` the same without write-combined (e);
/// `mmapreg` anonymous pages (2 MiB aligned, huge pages when the kernel gives them), locked, cuMemHostRegister'd mapped|portable
/// (the decode rings' type, (f)); `mmaphuge` the same on MAP_HUGETLB 2 MiB pages only (f2: refused when none are reserved); `vmm` cuMemCreate on the host (a NUMA node, else the host), cuMemMap'd and made accessible to the
/// CPU and the device (g).
const Where = enum { device, arena, cached, mmapreg, mmaphuge, vmm };

/// The driver's virtual memory management entry points, which the runtime's driver table does not hold.
const Vmm = struct {
    lib: std.DynLib,
    create: *const fn (*u64, usize, *const VmmProp, u64) callconv(.c) c_int,
    reserve: *const fn (*u64, usize, usize, u64, u64) callconv(.c) c_int,
    map: *const fn (u64, usize, usize, u64, u64) callconv(.c) c_int,
    set_access: *const fn (u64, usize, [*]const VmmAccess, usize) callconv(.c) c_int,
    gran: *const fn (*usize, *const VmmProp, c_int) callconv(.c) c_int,
    unmap: *const fn (u64, usize) callconv(.c) c_int,
    addr_free: *const fn (u64, usize) callconv(.c) c_int,
    release: *const fn (u64) callconv(.c) c_int,
    dev_attr: *const fn (*c_int, c_int, c_int) callconv(.c) c_int,
    err_name: *const fn (c_int, *[*:0]const u8) callconv(.c) c_int,

    fn open() !Vmm {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.NoLibcuda;
        errdefer lib.close();
        var v: Vmm = undefined;
        v.lib = lib;
        inline for (.{ .{ "create", "cuMemCreate" }, .{ "reserve", "cuMemAddressReserve" }, .{ "map", "cuMemMap" }, .{ "set_access", "cuMemSetAccess" }, .{ "gran", "cuMemGetAllocationGranularity" }, .{ "unmap", "cuMemUnmap" }, .{ "addr_free", "cuMemAddressFree" }, .{ "release", "cuMemRelease" }, .{ "dev_attr", "cuDeviceGetAttribute" }, .{ "err_name", "cuGetErrorName" } }) |p| {
            @field(v, p[0]) = v.lib.lookup(@TypeOf(@field(v, p[0])), p[1]) orelse return error.NoVmmSymbol;
        }
        return v;
    }

    fn name(v: *const Vmm, res: c_int) []const u8 {
        var s: [*:0]const u8 = undefined;
        if (v.err_name(res, &s) != 0) return "?";
        return std.mem.span(s);
    }
};

/// CUmemAllocationProp and CUmemAccessDesc (cuda.h): 32 and 12 bytes.
const VmmProp = extern struct { kind: c_int, handle_types: c_int, loc_type: c_int, loc_id: c_int, win32: ?*anyopaque = null, compression: u8 = 0, rdma: u8 = 0, usage: u16 = 0, reserved: [4]u8 = @splat(0) };
const VmmAccess = extern struct { loc_type: c_int, loc_id: c_int, flags: c_int };

extern "c" fn mlock(addr: *const anyopaque, len: usize) c_int;

/// A block of one page type: the GPU's address of it, the CPU's (what ibv_reg_mr would be given), and what frees it.
const WcBlock = struct { kind: Where, dev: u64, host: ?*anyopaque, len: usize, map: []align(std.heap.page_size_min) u8 = &.{}, vmm_handle: u64 = 0 };

const huge: usize = 2 << 20;

const WcPool = struct {
    d: *const cuda.Driver,
    blocks: std.ArrayList(WcBlock) = .empty,
    vmm: ?Vmm = null,
    /// What the last cuMemCreate attempts said (for the page-type line).
    vmm_note: [320]u8 = undefined,
    vmm_note_len: usize = 0,
    /// How the mmap type got its pages: "2 MiB huge pages (MAP_HUGETLB)", "4 KiB pages with MADV_HUGEPAGE", ...
    mmap_how: []const u8 = "",

    fn noteVmm(p: *WcPool, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(p.vmm_note[p.vmm_note_len..], fmt, args) catch return;
        p.vmm_note_len += s.len;
    }

    fn hostAlloc(p: *WcPool, kind: Where, bytes: usize, flags: c_uint) !WcBlock {
        var mem: ?*anyopaque = null;
        try p.d.check(p.d.api.cuMemHostAlloc(&mem, bytes, flags), "cuMemHostAlloc");
        errdefer _ = p.d.api.cuMemFreeHost(mem);
        var dev: u64 = 0;
        try p.d.check(p.d.api.cuMemHostGetDevicePointer_v2(&dev, mem, 0), "cuMemHostGetDevicePointer");
        return .{ .kind = kind, .dev = dev, .host = mem, .len = bytes };
    }

    fn mmapReg(p: *WcPool, bytes: usize, huge_only: bool) !WcBlock {
        const len = std.mem.alignForward(usize, bytes, huge);
        const prot: std.posix.PROT = .{ .READ = true, .WRITE = true };
        var map: []align(std.heap.page_size_min) u8 = undefined;
        var at: usize = 0;
        if (std.posix.mmap(null, len, prot, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .POPULATE = true, .HUGETLB = true }, -1, 0)) |m| {
            map = m;
            p.mmap_how = "2 MiB huge pages (MAP_HUGETLB)";
        } else |err| {
            if (huge_only) {
                p.mmap_how = @errorName(err);
                return error.HugePagesRefused;
            }
            map = try std.posix.mmap(null, len + huge, prot, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .POPULATE = true }, -1, 0);
            at = std.mem.alignForward(usize, @intFromPtr(map.ptr), huge) - @intFromPtr(map.ptr);
            std.posix.madvise(@alignCast(map.ptr + at), len, std.posix.MADV.HUGEPAGE) catch {};
            p.mmap_how = "4 KiB pages, 2 MiB aligned, MADV_HUGEPAGE (MAP_HUGETLB refused)";
        }
        errdefer std.posix.munmap(map);
        const base = map.ptr + at;
        @memset(base[0..len], 0);
        _ = mlock(base, len);
        var hr = try dsv41.rdma.HostReg.open();
        defer hr.lib.close();
        try p.d.check(hr.register(base, len, 0x01 | 0x02), "cuMemHostRegister");
        var dev: u64 = 0;
        try p.d.check(p.d.api.cuMemHostGetDevicePointer_v2(&dev, base, 0), "cuMemHostGetDevicePointer");
        return .{ .kind = if (huge_only) .mmaphuge else .mmapreg, .dev = dev, .host = base, .len = len, .map = map };
    }

    fn vmmAlloc(p: *WcPool, bytes: usize) !WcBlock {
        if (p.vmm == null) p.vmm = try Vmm.open();
        const v = &p.vmm.?;
        var numa: c_int = 0;
        _ = v.dev_attr(&numa, 134, 0); // CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
        var supported: c_int = 0;
        _ = v.dev_attr(&supported, 141, 0); // ..._HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
        p.vmm_note_len = 0;
        p.noteVmm("device attribute HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED {d}, host NUMA id {d}", .{ supported, numa });
        // the NUMA location first (the docs' EGM recipe), then the plain host location
        const tries = [_]struct { loc: c_int, id: c_int, label: []const u8 }{ .{ .loc = 3, .id = if (numa < 0) 0 else numa, .label = "HOST_NUMA" }, .{ .loc = 2, .id = 0, .label = "HOST" } };
        for (tries) |t| {
            var prop: VmmProp = .{ .kind = 1, .handle_types = 0, .loc_type = t.loc, .loc_id = t.id };
            var g: usize = 0;
            var res = v.gran(&g, &prop, 0);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemGetAllocationGranularity {s}", .{ t.label, v.name(res) });
                continue;
            }
            const len = std.mem.alignForward(usize, bytes, @max(g, 1));
            var h: u64 = 0;
            res = v.create(&h, len, &prop, 0);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemCreate {s}", .{ t.label, v.name(res) });
                continue;
            }
            var va: u64 = 0;
            res = v.reserve(&va, len, 0, 0, 0);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemAddressReserve {s}", .{ t.label, v.name(res) });
                _ = v.release(h);
                continue;
            }
            res = v.map(va, len, 0, h, 0);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemMap {s}", .{ t.label, v.name(res) });
                _ = v.addr_free(va, len);
                _ = v.release(h);
                continue;
            }
            const acc = [2]VmmAccess{ .{ .loc_type = t.loc, .loc_id = t.id, .flags = 3 }, .{ .loc_type = 1, .loc_id = 0, .flags = 3 } };
            res = v.set_access(va, len, &acc, 2);
            if (res != 0) {
                p.noteVmm("; {s}: cuMemSetAccess (CPU + device read/write) {s}", .{ t.label, v.name(res) });
                _ = v.unmap(va, len);
                _ = v.addr_free(va, len);
                _ = v.release(h);
                continue;
            }
            p.noteVmm("; created with the {s} location (granularity {d} B)", .{ t.label, g });
            return .{ .kind = .vmm, .dev = va, .host = @ptrFromInt(va), .len = len, .vmm_handle = h };
        }
        return error.VmmRefused;
    }

    fn alloc(p: *WcPool, kind: Where, bytes: usize) !WcBlock {
        const b: WcBlock = switch (kind) {
            .arena => try p.hostAlloc(.arena, bytes, 0x01 | 0x02 | 0x04),
            .cached => try p.hostAlloc(.cached, bytes, 0x01 | 0x02),
            .mmapreg => try p.mmapReg(bytes, false),
            .mmaphuge => try p.mmapReg(bytes, true),
            .vmm => try p.vmmAlloc(bytes),
            .device => unreachable,
        };
        try p.blocks.append(std.heap.page_allocator, b);
        return b;
    }

    fn release(p: *WcPool, b: WcBlock) void {
        switch (b.kind) {
            .arena, .cached => _ = p.d.api.cuMemFreeHost(b.host),
            .mmapreg, .mmaphuge => {
                if (dsv41.rdma.HostReg.open()) |hr_| {
                    var hr = hr_;
                    _ = hr.unregister(b.host);
                    hr.lib.close();
                } else |_| {}
                std.posix.munmap(b.map);
            },
            .vmm => {
                const v = &p.vmm.?;
                _ = v.unmap(b.dev, b.len);
                _ = v.addr_free(b.dev, b.len);
                _ = v.release(b.vmm_handle);
            },
            .device => {},
        }
    }

    /// Frees every block still held.
    fn freeAll(p: *WcPool) void {
        for (p.blocks.items) |b| p.release(b);
        p.blocks.clearRetainingCapacity();
    }

    fn deinit(p: *WcPool) void {
        p.freeAll();
        p.blocks.deinit(std.heap.page_allocator);
        if (p.vmm) |*v| v.lib.close();
    }
};

/// A variant of a class: where its buffers live and what the line calls it.
const WcVar = struct { w: Where, label: []const u8 };

/// The class measurements' shared state: the arena pool, the device buffers kept, the allocator for host-side inputs, and the
/// variants to run (cudaMalloc, the null control, then every page type this machine allocates).
const WcEnv = struct {
    r: *Run,
    pool: *WcPool,
    keep: *std.ArrayList(cuda.DeviceBuffer),
    ha: std.mem.Allocator,
    vars: [7]WcVar,
    nvars: usize,

    fn list(e: *const WcEnv) []const WcVar {
        return e.vars[0..e.nvars];
    }

    fn buf(e: WcEnv, w: Where, bytes: usize) !u64 {
        if (w != .device) return (try e.pool.alloc(w, bytes)).dev;
        const b = try cuda.DeviceBuffer.alloc(e.r.d, bytes);
        try e.keep.append(e.ha, b);
        return b.ptr;
    }

    fn up(e: WcEnv, bytes: []const u8) !u64 {
        const b = try upload(e.r.d, bytes);
        try e.keep.append(e.ha, b);
        return b.ptr;
    }
};

const wc_reps: usize = 20;
/// The launches of a kernel class in a 4,096-row pass: one a layer of the model's 40, the ceiling (the owner path carries
/// layers 0-19, so about half of it); the hc mixes are two a layer.
const wc_layers: usize = 40;
/// Under this many ms a launch a kernel is launch-bound: its launches are repeated until a measurement lasts about 20 ms, and a
/// null control still outside 2 % is reported as too short to time rather than as wrong.
const wc_short_ms: f32 = 0.05;
/// A null control outside 2 % of a long kernel but within this many % is NOISY (printed, not failed); beyond it the line fails.
const wc_noisy_pct: f32 = 5.0;

/// A line of text built in pieces.
const WcText = struct {
    buf: [2048]u8 = undefined,
    n: usize = 0,

    fn add(t: *WcText, comptime fmt: []const u8, args: anytype) !void {
        t.n += (try std.fmt.bufPrint(t.buf[t.n..], fmt, args)).len;
    }
};

/// The variants' times of a class.
const WcRes = struct {
    labels: [8][]const u8 = undefined,
    ms: [8]f32 = undefined,
    n: usize = 0,

    fn add(s: *WcRes, label: []const u8, ms: f32) void {
        s.labels[s.n] = label;
        s.ms[s.n] = ms;
        s.n += 1;
    }
};

/// ms a launch: the best of three measurements of `reps` launches (kernelMs), `reps` raised for a launch-bound kernel so that a
/// measurement lasts about 20 ms.
fn wcTime(r: *Run, comptime Ctx: type, ctx: Ctx, comptime f: fn (Ctx) anyerror!void) !f32 {
    var reps: usize = wc_reps;
    var best = try kernelMs(r, Ctx, ctx, f, reps);
    if (best < wc_short_ms * 2) {
        reps = @min(4000, @max(reps, @as(usize, @intFromFloat(20.0 / @max(best, 0.001)))));
        best = try kernelMs(r, Ctx, ctx, f, reps);
    }
    for (0..2) |_| best = @min(best, try kernelMs(r, Ctx, ctx, f, reps));
    return best;
}

/// One line for a class: ms a launch of each variant (variant 0 is cudaMalloc, variant 1 a second cudaMalloc set: the null
/// control, which must read within 2 % of variant 0; the others are the page types), each page type over variant 0, and ms a
fn wcLine(r: *Run, name: []const u8, res: *const WcRes, launches: usize) !void {
    const ms = res.ms[0..res.n];
    const labels = res.labels[0..res.n];
    var t: WcText = .{};
    try t.add("roci_wc_classes {s}: ms a launch", .{name});
    for (labels, ms) |l, v| try t.add(", {s} {d:.4}", .{ l, v });
    const null_pct = (ms[1] / ms[0] - 1.0) * 100.0;
    if (short) {
        try t.add("; null control {d:.1} % on a kernel under {d:.0} us: TOO SHORT TO TIME, the percentages below are launch noise", .{ null_pct, wc_short_ms * 1000 });
    } else if (noisy) {
    } else try t.add("; null control {d:.1} %", .{null_pct});
    for (labels[2..], ms[2..]) |l, v| try t.add("; {s} over cudaMalloc {d:.1} %", .{ l, (v / ms[0] - 1.0) * 100.0 });
    try t.add("; ms a 4,096-row pass at {d} launches:", .{launches});
    for (labels, ms) |l, v| try t.add(" {s} {d:.2},", .{ l, v * @as(f32, @floatFromInt(launches)) });
}

/// mx.quant of bf16 rows into each variant planar, and (the arena type of the baseline) in records: a row's `k` codes then its scales as
/// floats in one record of `rec` bytes (the scales at +k, lds = rec / 4 floats).
fn wcActQuant(r: *Run, env: WcEnv, name: []const u8, rng: std.Random, rows: usize, k: usize, rec: usize) !void {
    const x = try env.up(try bf16s(rows * k, rng, 1.0, env.ha));
    const C = struct {
        r: *Run,
        x: u64,
        rows: usize,
        k: usize,
        q: u64,
        ldq: usize,
        s: u64,
        lds: usize,
        fn go(c: @This()) anyerror!void {
            try mx.quant(c.r.k, c.r.s, c.x, .bf16, c.k, c.rows, c.k, c.q, c.ldq, c.s, c.lds);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{ .r = r, .x = x, .rows = rows, .k = k, .q = try env.buf(vr.w, rows * k), .ldq = k, .s = try env.buf(vr.w, rows * (k / 32) * 4), .lds = k / 32 };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    const base = try env.buf(.arena, rows * rec);
    const c: C = .{ .r = r, .x = x, .rows = rows, .k = k, .q = base, .ldq = rec, .s = base + k, .lds = rec / 4 };
    res.add("(c) WC records", try wcTime(r, C, c, C.go));
    try wcLine(r, name, &res, wc_layers);
}

fn wcActQuantWob(r: *Run, env: WcEnv, rng: std.Random) !void {
    // wo_b's input a node sends: 4,096 rows of 2,048 codes (+ 64 scales: a 2,304-byte record)
    try wcActQuant(r, env, "act_quant of wo_b's input (4,096 rows x 2,048)", rng, 4096, 2048, 2304);
}

fn wcActQuantPack(r: *Run, env: WcEnv, rng: std.Random) !void {
    // the experts' pack: 28,672 rows (4,096 x 7 slots) of 640 codes (+ 20 scales: a 720-byte record)
    try wcActQuant(r, env, "act_quant of the experts' pack (28,672 rows x 640)", rng, 4096 * 7, 640, 720);
}

/// _hc_mix_pf_2b then _hc_finish (tri_hc.hcPrePf, the owner path's attention-side mix): h_out (the streams) and the normalised
/// rows `x` written into each variant.
fn wcHcPre(r: *Run, env: WcEnv, rng: std.Random) !void {
    if (r.t.extra == null) return r.note("roci_wc_classes hcPrePf: the kit's extra set (aot-extra/) loads", false);
    const D: usize = 5120;
    const W: usize = D / 2;
    const rows: usize = 4096;
    const hc_blocks = dsv41.tri_basic.hc_blocks;
    const blk = try env.up(std.mem.sliceAsBytes(try rociBlocks(rows, W, rng, env.ha)));
    const h = try env.up(try bf16s(rows * 4 * D, rng, 1.0, env.ha));
    const fnw = try env.up(try f32s(24 * 4 * D, rng, env.ha));
    const scale = try env.up(try f32s(3, rng, env.ha));
    const base = try env.up(try f32s(24, rng, env.ha));
    const pre_in = try env.up(try f32s(rows * 4, rng, env.ha));
    const nw = try env.up(try bf16s(D, rng, 1.0, env.ha));
    const C = struct {
        r: *Run,
        h: u64,
        blk: u64,
        fnw: u64,
        scale: u64,
        base: u64,
        pre_in: u64,
        nw: u64,
        x: u64,
        h_out: u64,
        pre_out: u64,
        post: u64,
        comb: u64,
        part: u64,
        fn go(c: @This()) anyerror!void {
            _ = try dsv41.tri_hc.hcPrePf(c.r.t, c.h, c.fnw, c.scale, c.base, c.pre_in, c.nw, 1e-20, 1e-6, 20, c.x, c.pre_out, c.post, c.comb, c.part, .{ .gathered = c.blk, .world = 2, .h_out = c.h_out, .blk_rs = rows * W }, rows, D);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{
            .r = r,
            .h = h,
            .blk = blk,
            .fnw = fnw,
            .scale = scale,
            .base = base,
            .pre_in = pre_in,
            .nw = nw,
            .x = try env.buf(vr.w, rows * D * 2),
            .h_out = try env.buf(vr.w, rows * 4 * D * 2),
            .pre_out = try env.buf(.device, rows * 4 * 4),
            .post = try env.buf(.device, rows * 4 * 4),
            .comb = try env.buf(.device, rows * 16 * 4),
            .part = try env.buf(.device, rows * hc_blocks * 32 * 4),
        };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    try wcLine(r, "hcPrePf into x and the streams (4,096 rows x 4 x 5,120)", &res, 2 * wc_layers);
}

/// tri_norm.route (the router's top-6 and the shared expert, one warp a row): pick and wts into each variant.
fn wcRoute(r: *Run, env: WcEnv, rng: std.Random) !void {
    const rows: usize = 4096;
    const E: usize = 384;
    const logits = try env.up(try f32s(rows * E, rng, env.ha));
    const bias = try env.up(try f32s(E, rng, env.ha));
    const C = struct {
        r: *Run,
        logits: u64,
        bias: u64,
        pick: u64,
        wts: u64,
        fn go(c: @This()) anyerror!void {
            try dsv41.tri_norm.route(c.r.t, c.logits, 0, c.bias, 6, 1.5, E, c.pick, c.wts, rows, E, 7);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{ .r = r, .logits = logits, .bias = bias, .pick = try env.buf(vr.w, rows * 7 * 4), .wts = try env.buf(vr.w, rows * 7 * 4) };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    try wcLine(r, "ffnRoute into pick and wts (4,096 rows)", &res, wc_layers);
}

/// tri_basic.rmsnorm of the q_a projection (1,280 columns) into qr, into each variant.
fn wcRmsnorm(r: *Run, env: WcEnv, rng: std.Random) !void {
    const rows: usize = 4096;
    const Dq: usize = 1280;
    const x = try env.up(try bf16s(rows * Dq, rng, 1.0, env.ha));
    const w_ = try env.up(try bf16s(Dq, rng, 1.0, env.ha));
    const C = struct {
        r: *Run,
        x: u64,
        w: u64,
        out: u64,
        fn go(c: @This()) anyerror!void {
            try dsv41.tri_basic.rmsnorm(c.r.t, c.x, Dq, c.w, c.out, Dq, 1e-20, rows, Dq);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{ .r = r, .x = x, .w = w_, .out = try env.buf(vr.w, rows * Dq * 2) };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    try wcLine(r, "rmsnorm into qr (4,096 rows x 1,280)", &res, wc_layers);
}

/// The experts' consumers of codes: gate / up from the activation codes q / s (the loader reads them by erow) and down from the
/// intermediate's codes hq / hs, the codes in each variant (planar). The codes are made once by the quantising calls and the timed
/// calls only read them (mx.gateUpCodes, mx.downCodes; down's closing tf_dsn_combine reads and writes cudaMalloc in every variant).
fn wcExperts(r: *Run, env: WcEnv, rng: std.Random, comptime down: bool) !void {
    const R: usize = 4096;
    const P = R * X_SL;
    const ex = try xExperts(r.d, rng, env.ha, env.keep);
    const x = try env.up(try bf16s(R * X_D, rng, 0.5, env.ha));
    const hdn = try env.up(try bf16s(P * X_I, rng, 0.5, env.ha));
    const pw = try xPicks(R, rng, env.ha);
    const pb = try env.up(std.mem.sliceAsBytes(pw.pick));
    const wb = try env.up(std.mem.sliceAsBytes(pw.wts));
    const hb = try env.buf(.device, P * X_W * 2);
    const ob = try env.buf(.device, R * X_N * 4);
    const base = try xScratch(r.d, R, true, env.ha, env.keep);
    const C = struct {
        r: *Run,
        ex: weights.Experts,
        sc: mx.ExpertScratch,
        pk: u64,
        wt: u64,
        hb: u64,
        ob: u64,
        fn go(c: @This()) anyerror!void {
            if (down) try mx.downCodes(c.r.k, c.r.s, c.ex, c.sc, c.pk, R, c.ob, X_N, false) else try mx.gateUpCodes(c.r.k, c.r.s, c.r.d, c.ex, c.sc, c.pk, c.wt, R, 7.0, c.hb, X_MAIN);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        var sc = base;
        if (down) {
            sc.hq = try env.buf(vr.w, P * X_I);
            sc.hs = try env.buf(vr.w, P * (X_I / 32) * 4);
        } else {
            sc.q = try env.buf(vr.w, R * X_D);
            sc.s = try env.buf(vr.w, R * (X_D / 32) * 4);
        }
        // the codes: the quantising call writes them where this variant keeps them
        if (down) try mx.down(r.k, r.s, ex, sc, hdn, X_I, pb, R, ob, X_N, false, false) else try mx.gateUp(r.k, r.s, r.d, ex, sc, x, X_D, pb, wb, R, 7.0, hb, X_MAIN, false);
        try r.s.synchronize();
        const c: C = .{ .r = r, .ex = ex, .sc = sc, .pk = pb, .wt = wb, .hb = hb, .ob = ob };
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    if (down) {
        try wcLine(r, "sf_dn / pf_dn from the intermediate's codes (4,096 rows x 7 slots)", &res, wc_layers);
    } else try wcLine(r, "pf_gu from the activation codes by erow (4,096 rows)", &res, wc_layers);
}

fn wcPfGu(r: *Run, env: WcEnv, rng: std.Random) !void {
    return wcExperts(r, env, rng, false);
}

fn wcPfDn(r: *Run, env: WcEnv, rng: std.Random) !void {
    return wcExperts(r, env, rng, true);
}

/// The block-scaled exact GEMM (sf_split128 family, wo_b's shape: K 2,048, N 2,560, 4,096 rows) reading its codes and scales from
/// each variant planar, and from the arena type of the baseline in 2,304-byte records.
fn wcGemmCodes(r: *Run, env: WcEnv, rng: std.Random) !void {
    const K: usize = 2048;
    const N: usize = 2560;
    const M: usize = 4096;
    if (r.k.sfk == null) return r.note("roci_wc_classes sf GEMM: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    const l = try fp8Linear(r.d, N, K, .fast, rng, env.ha, env.keep);
    const x = try env.up(try bf16s(M * K, rng, 0.5, env.ha));
    const z = try env.buf(.device, @max(mx.zFloats(l, M), 1) * 4);
    const y = try env.buf(.device, M * N * 4);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    const C = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        ldq: usize,
        s: u64,
        lds: usize,
        z: u64,
        y: u64,
        fn go(c: @This()) anyerror!void {
            try mx.gemm(c.r.k, c.r.s, c.l, c.q, c.ldq, c.s, c.lds, M, c.y, N, .f32, false, c.z);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const c: C = .{ .r = r, .l = l, .q = try env.buf(vr.w, M * K), .ldq = K, .s = try env.buf(vr.w, M * (K / 32) * 4), .lds = K / 32, .z = z, .y = y };
        // the codes where this variant keeps them (the quantising call writes the layout it reads)
        try mx.quant(r.k, r.s, x, .bf16, K, M, K, c.q, c.ldq, c.s, c.lds);
        try r.s.synchronize();
        res.add(vr.label, try wcTime(r, C, c, C.go));
    }
    const b = try env.buf(.arena, M * 2304);
    const c: C = .{ .r = r, .l = l, .q = b, .ldq = 2304, .s = b + K, .lds = 2304 / 4, .z = z, .y = y };
    try mx.quant(r.k, r.s, x, .bf16, K, M, K, c.q, c.ldq, c.s, c.lds);
    try r.s.synchronize();
    res.add("(c) WC records", try wcTime(r, C, c, C.go));
    try wcLine(r, "sf_split128 / wo_b GEMM from the codes (4,096 rows, K 2,048, N 2,560)", &res, wc_layers);
}

/// tri_attn.sparseAttn (the base set's _sparse_attn_part: 4,096 rows, 16 heads, 512 picks over 262,144 compressed rows, one split)
/// with the picks' index rows (cidx) in each variant.
fn wcSparseAttn(r: *Run, env: WcEnv, rng: std.Random) !void {
    const H: usize = 16;
    const HD: usize = 512;
    const WIN: usize = 128;
    const end: usize = 1 << 20;
    const ncomp: usize = end / 4;
    const R: usize = 4096;
    const n_idx: usize = 512;
    const start = end - R;
    if (tri_attn.splits(R, false) > 1) return r.note("roci_wc_classes sparseAttn: 4,096 rows take one split", false);
    const codes = try env.ha.alloc(u8, ncomp * HD / 2);
    rng.bytes(codes);
    const scales = try env.ha.alloc(u8, ncomp * HD / 16);
    for (scales) |*s| s.* = 0x18 + rng.uintLessThan(u8, 0x3f - 0x18);
    const cb = try env.up(codes);
    const sb = try env.up(scales);
    const q = try env.up(try bf16s(R * H * HD, rng, 0.5, env.ha));
    const wsrc = try env.up(try bf16s((R + WIN) * HD, rng, 0.5, env.ha));
    const idx = try env.ha.alloc(i64, R * n_idx);
    for (0..R) |i| for (0..n_idx) |j| {
        idx[i * n_idx + j] = if ((j + i) % 29 == 7) -1 else @intCast(rng.uintLessThan(usize, (start + i + 1) / 4));
    };
    const pos = try env.ha.alloc(i64, R);
    for (pos, 0..) |*p, i| p.* = @intCast(start + i);
    const pb = try env.up(std.mem.sliceAsBytes(pos));
    var wlo = [1]i64{@intCast(start - WIN)};
    const wb = try env.up(std.mem.sliceAsBytes(&wlo));
    const sink = try env.up(try f32s(H, rng, env.ha));
    const out = try env.buf(.device, R * H * HD * 2);
    const C = struct {
        r: *Run,
        a: tri_attn.Attn,
        fn go(c: @This()) anyerror!void {
            try tri_attn.sparseAttn(c.r.t, c.a);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const ib = try env.buf(vr.w, R * n_idx * 8);
        const a: tri_attn.Attn = .{ .q = q, .out = out, .rows = R, .h = H, .hd = HD, .sink = sink, .wsrc = wsrc, .wsrc_rows = R + WIN, .wlo = wb, .ring = false, .comp = .{ .fp4 = .{ .codes = cb, .scales = sb } }, .idx = ib, .n_idx = n_idx, .pos = pb, .scale = dsv41.prompt.scale(HD), .window = WIN, .parts = .{ .pm = 0, .pl = 0, .po = 0 } };
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .a = a }, C.go));
    }
    try wcLine(r, "sparseAttn reading cidx (4,096 rows, 512 picks)", &res, wc_layers);
}

/// Points the staged (whole-line) GEMM kernels at a variant's y: the arena types take the _st kernels (the engine's choice for a y in the
/// arena), cudaMalloc takes the baseline.
fn wcStaged(r: *Run, w: Where, y: u64, bytes: usize) !void {
    if (w == .device) r.k.wc = .{ 0, 0 } else try r.k.useStaged(y, y + bytes);
}

/// wo_b's block-scaled exact GEMM (sf_split128_exact; _st into the arena), K 4,096, N 2,560, 4,096 rows, its fp32 y (psum) in each variant.
fn wcWobPsum(r: *Run, env: WcEnv, rng: std.Random) !void {
    const K: usize = 4096;
    const N: usize = 2560;
    const M: usize = 4096;
    if (r.k.sfk == null) return r.note("roci_wc_classes wo_b into psum: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    const l = try fp8Linear(r.d, N, K, .fast, rng, env.ha, env.keep);
    const x = try env.up(try bf16s(M * K, rng, 0.5, env.ha));
    const q = try env.buf(.device, M * K);
    const s = try env.buf(.device, M * (K / 32) * 4);
    const z = try env.buf(.device, @max(mx.zFloats(l, M), 1) * 4);
    try mx.quant(r.k, r.s, x, .bf16, K, M, K, q, K, s, K / 32);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer {
        r.k.sf = .off;
        r.k.wc = .{ 0, 0 };
    }
    const C = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        z: u64,
        y: u64,
        fn go(c: @This()) anyerror!void {
            try mx.gemm(c.r.k, c.r.s, c.l, c.q, K, c.s, K / 32, M, c.y, N, .f32, false, c.z);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const y = try env.buf(vr.w, M * N * 4);
        try wcStaged(r, vr.w, y, M * N * 4);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .l = l, .q = q, .s = s, .z = z, .y = y }, C.go));
    }
    try wcLine(r, "wo_b sf_split128_exact(_st) into psum (4,096 rows, K 4,096, N 2,560)", &res, wc_layers);
}

/// tf_dsn_combine (7 slots, 2,560 columns, 4,096 rows) into pm, in each variant; the entries' outputs are cudaMalloc.
fn wcCombinePm(r: *Run, env: WcEnv, rng: std.Random) !void {
    const M: usize = 4096;
    const S: usize = 7;
    const E: usize = 384;
    const N: usize = 2560;
    const picks = try env.ha.alloc(i32, M * S);
    for (0..M) |row| {
        var seen: [S]i32 = undefined;
        var n: usize = 0;
        while (n < S) {
            const e: i32 = @intCast(rng.uintLessThan(usize, E));
            if (std.mem.indexOfScalar(i32, seen[0..n], e) != null) continue;
            seen[n] = e;
            picks[row * S + n] = e;
            n += 1;
        }
    }
    const pk = try env.up(std.mem.sliceAsBytes(picks));
    const outs = try env.buf(.device, M * S * N * 4);
    try r.d.check(r.d.api.cuMemsetD32Async(outs, 0x3f800000, M * S * N, r.s.handle), "cuMemsetD32Async");
    var sc: mx.ExpertScratch = undefined;
    sc.slots = S;
    sc.out = outs;
    const C = struct {
        r: *Run,
        sc: mx.ExpertScratch,
        pk: u64,
        y: u64,
        fn go(c: @This()) anyerror!void {
            try mx.combine(c.r.k, c.r.s, c.sc, c.pk, M, E, N, c.y, N, false);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const y = try env.buf(vr.w, M * N * 4);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .sc = sc, .pk = pk, .y = y }, C.go));
    }
    try wcLine(r, "combine (7 slots) into pm (4,096 rows x 2,560)", &res, wc_layers);
}

/// Engram's projection (a dense FP8 GEMM, K 5,120, N 12,800, 4,096 rows; the fold / tile configuration, _st into the arena) into ek.
fn wcPfGemmEk(r: *Run, env: WcEnv, rng: std.Random) !void {
    const K: usize = 5120;
    const N: usize = 12800;
    const M: usize = 4096;
    const l = try fp8Linear(r.d, N, K, .fast, rng, env.ha, env.keep);
    const x = try env.up(try bf16s(M * K, rng, 0.5, env.ha));
    const q = try env.buf(.device, M * K);
    const s = try env.buf(.device, M * (K / 32) * 4);
    const z = try env.buf(.device, @max(mx.zFloats(l, M), 1) * 4);
    try mx.quant(r.k, r.s, x, .bf16, K, M, K, q, K, s, K / 32);
    r.k.sf = .off;
    defer r.k.wc = .{ 0, 0 };
    const C = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        z: u64,
        y: u64,
        fn go(c: @This()) anyerror!void {
            try mx.gemm(c.r.k, c.r.s, c.l, c.q, K, c.s, K / 32, M, c.y, N, .f32, false, c.z);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const y = try env.buf(vr.w, M * N * 4);
        try wcStaged(r, vr.w, y, M * N * 4);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .l = l, .q = q, .s = s, .z = z, .y = y }, C.go));
    }
    try wcLine(r, "pf_gemm(_st) into ek (4,096 rows, K 5,120, N 12,800)", &res, 2);
}

/// add2_bf16 of an owner sum's part 2 (one quarter, 1,024 x 2,560): this node's fp32 partial plus the landed stage block, rounded to
/// bf16 into the block, all three in each variant.
fn wcAdd2(r: *Run, env: WcEnv, rng: std.Random) !void {
    const ops = r.ops orelse return r.note("roci_wc_classes add2: the ops are available", false);
    const n: usize = 1024 * 2560;
    const a_host = try f32s(n, rng, env.ha);
    const b_host = try f32s(n, rng, env.ha);
    const C = struct {
        ops: *const dsv41.ops.Ops,
        s: cuda.Stream,
        a: u64,
        b: u64,
        out: u64,
        fn go(c: @This()) anyerror!void {
            try c.ops.add2Bf16(c.s, c.a, c.b, c.out, n);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const a = try env.buf(vr.w, n * 4);
        const b = try env.buf(vr.w, n * 4);
        const out = try env.buf(vr.w, n * 2);
        res.add(vr.label, try wcTime(r, C, .{ .ops = ops, .s = r.s, .a = a, .b = b, .out = out }, C.go));
    }
    try wcLine(r, "add2_bf16 reading the partial and the landed stage block, into the sum block (1,024 x 2,560)", &res, 83);
}

/// _hc_mix_pf_2b (tri_hc.hcMixPf with the blocks as they land) reading the two bf16 sum blocks from each variant: 4,096 rows x 5,120.
fn wcHcMix2b(r: *Run, env: WcEnv, rng: std.Random) !void {
    if (r.t.extra == null) return r.note("roci_wc_classes _hc_mix_pf_2b: the kit's extra set (aot-extra/) loads", false);
    const D: usize = 5120;
    const W: usize = D / 2;
    const rows: usize = 4096;
    const hc_blocks = dsv41.tri_basic.hc_blocks;
    const blk_host = try bf16s(2 * rows * W, rng, 1.0, env.ha);
    const h = try env.up(try bf16s(rows * 4 * D, rng, 1.0, env.ha));
    const fnw = try env.up(try f32s(24 * 4 * D, rng, env.ha));
    const post = try env.buf(.device, rows * 4 * 4);
    const comb = try env.buf(.device, rows * 16 * 4);
    const part = try env.buf(.device, rows * hc_blocks * 32 * 4);
    const h_out = try env.buf(.device, rows * 4 * D * 2);
    const C = struct {
        r: *Run,
        h: u64,
        fnw: u64,
        blk: u64,
        part: u64,
        h_out: u64,
        post: u64,
        comb: u64,
        fn go(c: @This()) anyerror!void {
            _ = try dsv41.tri_hc.hcMixPf(c.r.t, c.h, c.fnw, c.part, .{ .gathered = c.blk, .world = 2, .h_out = c.h_out, .blk_rs = rows * W }, c.post, c.comb, rows, D);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const blk = try env.buf(vr.w, 2 * rows * W * 2);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .h = h, .fnw = fnw, .blk = blk, .part = part, .h_out = h_out, .post = post, .comb = comb }, C.go));
    }
    try wcLine(r, "_hc_mix_pf_2b reading the landed sum blocks (4,096 rows x 5,120)", &res, 2 * wc_layers);
}

/// _engram_gate_2b reading Engram's two bf16 sum blocks (12,800 columns each) from each variant: 4,096 rows.
fn wcEngramGate2b(r: *Run, env: WcEnv, rng: std.Random) !void {
    if (r.t.extra == null) return r.note("roci_wc_classes _engram_gate_2b: the kit's extra set (aot-extra/) loads", false);
    const D: usize = 5120;
    const EW: usize = 12800;
    const rows: usize = 4096;
    const blk_host = try bf16s(2 * rows * EW, rng, 1.0, env.ha);
    const h = try env.up(try bf16s(rows * 4 * D, rng, 1.0, env.ha));
    const qk = try env.up(try f32s(4 * D, rng, env.ha));
    const out = try env.buf(.device, rows * 4 * D * 2);
    const C = struct {
        r: *Run,
        h: u64,
        blk: u64,
        qk: u64,
        out: u64,
        fn go(c: @This()) anyerror!void {
            try dsv41.tri_basic.engramGate2b(c.r.t, c.h, c.blk, rows * EW, c.qk, c.out, 1e-20, rows, D);
        }
    };
    var res: WcRes = .{};
    for (env.list()) |vr| {
        const blk = try env.buf(vr.w, 2 * rows * EW * 2);
        res.add(vr.label, try wcTime(r, C, .{ .r = r, .h = h, .blk = blk, .qk = qk, .out = out }, C.go));
    }
    try wcLine(r, "_engram_gate_2b reading the landed sum blocks (4,096 rows, 12,800 columns)", &res, 2);
}

const wc_labels = [_][]const u8{ "(a) cudaMalloc", "null (a) again", "(b) WC hostalloc (the arena)", "(e) cached hostalloc", "(f) mmap+cuMemHostRegister", "(f2) mmap MAP_HUGETLB", "(g) cuMemCreate host" };

/// Each page type this machine allocates (a 64 MiB block): whether it allocates, how, and whether ibv_reg_mr accepts it. With no verbs
/// device to ask (this workstation) the registration answer is the construction's: (b) (e) cudaHostAlloc and (f) mlock'd anonymous
/// pages are ordinary pinned host memory, which the driver registers for DMA; (g) is a CUDA-managed mapping, and CUDA's documentation
/// (Virtual Memory Management, Extended GPU Memory) says nothing about registering its CPU mapping with a NIC.
fn wcPageTypes(r: *Run, pool: *WcPool, avail: *[7]bool) !void {
    var t: WcText = .{};
    try t.add("roci_wc_classes page types (64 MiB blocks)", .{});
    const dev_env = std.c.getenv("TF_DS_RDMA_DEVICES");
    const first: []const u8 = if (dev_env) |v| std.mem.sliceTo(std.mem.span(v), ',') else "mlx5_0";
    var probe: ?dsv41.rocinante_verbs.RegProbe = dsv41.rocinante_verbs.RegProbe.open(first) catch null;
    defer if (probe) |*p| p.close();
    if (probe == null) try t.add("; ibv_reg_mr not probed (no verbs device {s} to open here)", .{first}) else try t.add("; ibv_reg_mr probed on {s}", .{first});
    const kinds = [_]Where{ .arena, .cached, .mmapreg, .mmaphuge, .vmm };
    const names = [_][]const u8{ "(b) WC hostalloc", "(e) cached hostalloc", "(f) mmap+register", "(f2) mmap MAP_HUGETLB", "(g) cuMemCreate host" };
    const construction = [_][]const u8{ "registerable by construction (pinned host memory)", "registerable by construction (pinned host memory)", "registerable by construction (locked anonymous pages, the decode rings' registration)", "registerable by construction (locked huge pages)", "registration of its CPU mapping is not documented by CUDA" };
    for (kinds, names, construction, 0..) |k, nm, why, i| {
        const b = pool.alloc(k, 64 << 20) catch |err| {
            try t.add("; {s}: NOT allocated ({s}{s}{s})", .{ nm, @errorName(err), if (k == .vmm or k == .mmaphuge) ": " else "", if (k == .vmm) pool.vmm_note[0..pool.vmm_note_len] else if (k == .mmaphuge) pool.mmap_how else "" });
            continue;
        };
        avail[i + 2] = true;
        try t.add("; {s}: allocated", .{nm});
        if (k == .mmapreg or k == .mmaphuge) try t.add(" with {s}", .{pool.mmap_how});
        if (k == .vmm) try t.add(" ({s})", .{pool.vmm_note[0..pool.vmm_note_len]});
        if (probe) |*p| {
            if (p.refused(b.host.?, b.len)) |errno| {
                try t.add(", ibv_reg_mr REFUSED (errno {d})", .{errno});
            } else try t.add(", ibv_reg_mr ACCEPTED", .{});
        } else try t.add(", {s}", .{why});
    }
    pool.freeAll();
    r.note(t.buf[0..t.n], avail[2]);
}

/// U8, the gate before the steps that move more producers and consumers to the arena: each kernel class the owner path moves
/// (act_quant in planar and in records, hcPrePf, ffnRoute, rmsnorm, pf_gu and down from the codes, the block-scaled GEMM from the
/// codes, sparseAttn from cidx) at the real shapes of a 4,096-row pass, kernel only (events, best of three runs of 20 launches or
/// more, no readback), into or out of cudaMalloc, a second cudaMalloc set (the null control, within 2 %) and every page type of
/// host memory the machine allocates. This changes no engine path. A class that cannot run says why on its own line.
fn rociWcClasses(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x0c1a55e5);
    const rng = prng.random();
    var pool: WcPool = .{ .d = r.d };
    defer pool.deinit();
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var avail: [7]bool = .{ true, true, false, false, false, false, false };
    try wcPageTypes(r, &pool, &avail);
    var env: WcEnv = .{ .r = r, .pool = &pool, .keep = &keep, .ha = ha, .vars = undefined, .nvars = 0 };
    const wheres = [_]Where{ .device, .device, .arena, .cached, .mmapreg, .mmaphuge, .vmm };
    for (wheres, wc_labels, avail) |w, l, ok| if (ok) {
        env.vars[env.nvars] = .{ .w = w, .label = l };
        env.nvars += 1;
    };
    const classes = [_]struct { name: []const u8, f: *const fn (*Run, WcEnv, std.Random) anyerror!void }{
        .{ .name = "act_quant of wo_b's input", .f = wcActQuantWob },
        .{ .name = "act_quant of the experts' pack", .f = wcActQuantPack },
        .{ .name = "hcPrePf", .f = wcHcPre },
        .{ .name = "ffnRoute", .f = wcRoute },
        .{ .name = "rmsnorm", .f = wcRmsnorm },
        .{ .name = "pf_gu", .f = wcPfGu },
        .{ .name = "pf_dn", .f = wcPfDn },
        .{ .name = "sf GEMM", .f = wcGemmCodes },
        .{ .name = "sparseAttn", .f = wcSparseAttn },
        .{ .name = "wo_b into psum", .f = wcWobPsum },
        .{ .name = "combine into pm", .f = wcCombinePm },
        .{ .name = "pf_gemm into ek", .f = wcPfGemmEk },
        .{ .name = "add2", .f = wcAdd2 },
        .{ .name = "_hc_mix_pf_2b", .f = wcHcMix2b },
        .{ .name = "_engram_gate_2b", .f = wcEngramGate2b },
    };
    for (classes) |cl| {
        const before = r.lines;
        cl.f(r, env, rng) catch |err| {
            var line: [256]u8 = undefined;
            if (r.lines == before) r.note(try std.fmt.bufPrint(&line, "roci_wc_classes {s}: could not run ({s})", .{ cl.name, @errorName(err) }), false);
        };
        // each class's buffers are freed before the next (the device buffers kept and the host blocks)
        for (keep.items) |*b| b.free();
        keep.clearRetainingCapacity();
        pool.freeAll();
    }
    r.note("roci_wc_classes act_quant (d), the record writer through shared memory: no such kernel exists in the engine tree, so it is not measured", true);
}

// -- the cost of a local-first tile order (a consumer starts on the rows this node holds, then waits for the others') --------------

/// `rows` rows' picks as a skewed router makes them: the routed experts in a random fixed order, the one of rank i drawn with
/// weight 1 / (i + 1)^0.8, six distinct a row, the shared expert last.
fn xPicksSkew(rows: usize, rng: std.Random, ha: std.mem.Allocator) !struct { pick: []i32, wts: []f32 } {
    const n = X_E - 1;
    const order = try ha.alloc(i32, n);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    rng.shuffle(i32, order);
    const cdf = try ha.alloc(f64, n);
    var total: f64 = 0;
    for (cdf, 0..) |*c, i| {
        total += std.math.pow(f64, @as(f64, @floatFromInt(i + 1)), -0.8);
        c.* = total;
    }
    const pick = try ha.alloc(i32, rows * X_SL);
    const wts = try ha.alloc(f32, rows * X_SL);
    for (0..rows) |row| {
        var used: [6]i32 = undefined;
        for (0..6) |sl| {
            var e: i32 = undefined;
            while (true) {
                const u = rng.float(f64) * total;
                var i: usize = 0;
                while (i + 1 < n and cdf[i] < u) i += 1;
                e = order[i];
                if (std.mem.indexOfScalar(i32, used[0..sl], e) == null) break;
            }
            used[sl] = e;
            pick[row * X_SL + sl] = e;
            wts[row * X_SL + sl] = rng.float(f32) * 0.5;
        }
        pick[row * X_SL + 6] = X_E - 1;
        wts[row * X_SL + 6] = 1;
    }
    return .{ .pick = pick, .wts = wts };
}

/// One kernel launch ready to repeat: the function, its arguments and grid (256 threads a CTA).
const Lau = struct {
    f: cuda.Function,
    args: cuda.Args,
    gx: usize,
    gy: usize,
    smem: u32,
};

/// Launches `ls` in order on the run's stream; with kernelMs the milliseconds of the whole sequence.
const LauRun = struct {
    r: *Run,
    ls: []const Lau,

    fn go(c: @This()) anyerror!void {
        for (c.ls) |l| {
            var a = l.args;
            try cuda.launch.launch(l.f, .{ .grid = .{ .x = @intCast(l.gx), .y = @intCast(l.gy), .z = 1 }, .block = .{ .x = 256 }, .shared = l.smem }, c.r.s, &a);
        }
    }
};

fn lauMs(r: *Run, ls: []const Lau) !f32 {
    return kernelBest(r, LauRun, .{ .r = r, .ls = ls }, LauRun.go);
}

/// The ints of a device array, read back (memory from `a`).
fn getInts(d: *const cuda.Driver, ptr: u64, n: usize, a: std.mem.Allocator) ![]i32 {
    const v = try a.alloc(i32, n);
    return v;
}

/// The tile list a layout launches: local tiles then rest tiles in one device array, and their two counts in another.
const LfDevice = struct {
    off: u64,
    ent: u64,
    erow: u64,
    tiles: u64,
    counts: u64,
    nlocal: usize,
    nrest: usize,

    fn make(d: *const cuda.Driver, l: dsv41.local_first.Layout, ha: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer)) !LfDevice {
        const both = try ha.alloc([2]i32, l.local.len + l.rest.len);
        @memcpy(both[0..l.local.len], l.local);
        @memcpy(both[l.local.len..], l.rest);
        const cnt = [2]i32{ @intCast(l.local.len), @intCast(l.rest.len) };
        const o = try upload(d, std.mem.sliceAsBytes(l.off));
        const en = try upload(d, std.mem.sliceAsBytes(l.ent));
        const er = try upload(d, std.mem.sliceAsBytes(l.erow));
        const tl = try upload(d, std.mem.sliceAsBytes(both));
        const ct = try upload(d, std.mem.sliceAsBytes(&cnt));
        for ([_]cuda.DeviceBuffer{ o, en, er, tl, ct }) |b| try keep.append(ha, b);
        return .{ .off = o.ptr, .ent = en.ptr, .erow = er.ptr, .tiles = tl.ptr, .counts = ct.ptr, .nlocal = l.local.len, .nrest = l.rest.len };
    }
};

/// The grouped experts' kernel of one phase over one set of operands: gate / up (`down` false) or down, on `h` / `out`.
const LfKernel = struct {
    r: *Run,
    down: bool,
    gu: mx.GuArgs,
    dn: mx.DnArgs,
    f: cuda.Function,
    gx: usize,
    smem: u32,

    /// A launch of `tiles` (device list) counted by *ntiles over grouping arrays (off, ent, erow), `gy` tile rows of CTAs, into `dst`
    /// (the pack h or the entries' outputs).
    fn lau(k: LfKernel, off: u64, ent: u64, erow: u64, tiles: u64, ntiles: u64, gy: usize, dst: u64) Lau {
        var args: cuda.Args = .{};
        if (k.down) {
            var a = k.dn;
            a.off = off;
            a.ent = ent;
            a.out = dst;
            args.add(a);
        } else {
            var a = k.gu;
            a.off = off;
            a.ent = ent;
            a.erow = erow;
            a.h = dst;
            args.add(a);
        }
        args.add(tiles);
        args.add(ntiles);
        return .{ .f = k.f, .args = args, .gx = k.gx, .gy = gy, .smem = k.smem };
    }
};

fn pct(x: f32, base: f32) f32 {
    return (x / base - 1.0) * 100.0;
}

/// What the local-first order costs the grouped expert kernels at a 4,096-row pass's real shapes, kernel only (events, 20 launches
/// averaged after 3): gate / up (tf_dsn_sf_gu_exact) and down (tf_dsn_sf_dn_exact) in one launch of the baseline tile list (a) against
/// the same rows laid out so a tile's rows come from one part of the chunk (dsv41.local_first), the node's own quarter's tiles
/// launched first and the rest in a second launch, with the rest cut into the three other quarters' tiles (four segments) and
/// kept whole (two): tile counts, ms of each launch, the pair against (a), the share of the work in the first launch. Rows 0 ..
/// 1,023 are the node's own. Routing: the replay's uniform draw of six experts a row, and a skewed draw (a stand-in for a router's
/// hot experts, not a capture). Then the attention front's dense GEMM (wq_a) on the gathered rows as one 4,096-row launch against
/// 1,024 then 3,072. Outputs byte for byte, each with a control.
fn rociLocalFirst(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x10ca1f);
    const rng = prng.random();
    const d = r.d;
    var line: [1024]u8 = undefined;
    const sk = r.k.sfk orelse return r.note("roci_local_first: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const R: usize = 4096;
    const P = R * X_SL;
    const quarter = R / 4;
    const ex = try xExperts(d, rng, ha, &keep);
    const nx = ex.native.?;
    const x = try upload(d, try bf16s(R * X_D, rng, 0.5, ha));
    const hdn = try upload(d, try bf16s(P * X_I, rng, 0.5, ha));
    const h_a = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
    const h_n = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
    const h_b = try cuda.DeviceBuffer.alloc(d, P * X_W * 2);
    const o_a = try cuda.DeviceBuffer.alloc(d, P * X_N * 4);
    const o_n = try cuda.DeviceBuffer.alloc(d, P * X_N * 4);
    const o_b = try cuda.DeviceBuffer.alloc(d, P * X_N * 4);
    for ([_]cuda.DeviceBuffer{ x, hdn, h_a, h_n, h_b, o_a, o_n, o_b }) |b| try keep.append(ha, b);
    const names = [_][]const u8{ "uniform routing", "skewed routing" };
    for (names, 0..) |rname, ri| {
        var picks: []i32 = undefined;
        var weights_: []f32 = undefined;
        if (ri == 0) {
            const p = try xPicks(R, rng, ha);
            picks = p.pick;
            weights_ = p.wts;
        } else {
            const p = try xPicksSkew(R, rng, ha);
            picks = p.pick;
            weights_ = p.wts;
        }
        const pb = try upload(d, std.mem.sliceAsBytes(picks));
        const wb = try upload(d, std.mem.sliceAsBytes(weights_));
        try keep.append(ha, pb);
        try keep.append(ha, wb);
        const sc = try xScratch(d, R, true, ha, &keep);
        // the codes, the grouping and the baseline tile list, once: what gate / up and down read
        try mx.quant(r.k, r.s, x.ptr, .bf16, X_D, R, X_D, sc.q, X_D, sc.s, X_D / 32);
        try mx.grouping(r.k, r.s, d, X_E, sc, pb.ptr, R);
        try mx.tileList(r.k, r.s, sc);
        try mx.quant(r.k, r.s, hdn.ptr, .bf16, X_I, P, X_I, sc.hq, X_I, sc.hs, X_I / 32);
        try r.s.synchronize();
        const off = try getInts(d, sc.off, X_E + 1, ha);
        const ent = try getInts(d, sc.ent, P, ha);
        const erow = try getInts(d, sc.erow, P, ha);
        const nt_a: usize = @intCast((try getInts(d, sc.ntiles, 1, ha))[0]);
        const want = try dsv41.local_first.todays(ha, off, X_E);
        const lay2 = try dsv41.local_first.segmented(ha, off, ent, erow, X_E, 2, quarter);
        const lay4 = try dsv41.local_first.segmented(ha, off, ent, erow, X_E, 4, quarter);
        const g2 = try LfDevice.make(d, lay2, ha, &keep);
        const g4 = try LfDevice.make(d, lay4, ha, &keep);
        const holds = want.len == nt_a and dsv41.local_first.holds(lay2, off, ent, erow, X_E, 2, quarter) and dsv41.local_first.holds(lay4, off, ent, erow, X_E, 4, quarter);
        const grid_a: usize = P / mx.pf_bm + X_E; // the baseline grid: the most tiles a grouping can make
        for ([_]bool{ false, true }) |down| {
            const kname: []const u8 = if (down) "sf_dn_exact" else "sf_gu_exact";
            const k: LfKernel = .{
                .r = r,
                .down = down,
                .gu = .{ .q = sc.q, .s = sc.s, .ldq = X_D, .lds = X_D / 32, .ent = sc.ent, .erow = sc.erow, .off = sc.off, .act = sc.act, .nact = sc.nact, .w1 = nx.w1, .w3 = nx.w3, .s1 = nx.s1, .s3 = nx.s3, .ldw = X_D / 2, .ldws = X_D / 32, .fmt = 1, .D = X_D, .shared = X_E - 1, .sldw = X_D, .sldws = X_D / 32, .owner = nx.owner, .n_tiles = X_TILES, .rest_tiles = X_REST, .wts = wb.ptr, .limit = 7.0, .h = h_a.ptr, .ldh = X_MAIN, .P = @intCast(P) },
                .dn = .{ .q = sc.hq, .s = sc.hs, .ldq = X_I, .lds = X_I / 32, .ent = sc.ent, .off = sc.off, .act = sc.act, .nact = sc.nact, .w2 = nx.w2, .s2 = nx.s2, .ldw = X_I / 2, .ldws = X_I / 32, .fmt = 1, .shared = X_E - 1, .sldw = X_I, .sldws = X_I / 32, .I = X_I, .N = X_N, .round_bf16 = 0, .out = o_a.ptr, .ldo = X_N },
                .f = if (down) sk.dn[1] else sk.gu[1],
                .gx = if (down) X_N / 128 else X_TILES * 32 / 128,
                .smem = if (down) mx.pf_dn_r128_smem else mx.pf_gu_r128_smem,
            };
            const dst_a: u64 = if (down) o_a.ptr else h_a.ptr;
            const dst_n: u64 = if (down) o_n.ptr else h_n.ptr;
            const dst_b: u64 = if (down) o_b.ptr else h_b.ptr;
            const bytes: usize = if (down) P * X_N * 4 else P * X_W * 2;
            const la = [1]Lau{k.lau(sc.off, sc.ent, sc.erow, sc.tiles, sc.ntiles, grid_a, dst_a)};
            const ln = [1]Lau{k.lau(sc.off, sc.ent, sc.erow, sc.tiles, sc.ntiles, grid_a, dst_n)};
            const lx = [1]Lau{k.lau(sc.off, sc.ent, sc.erow, sc.tiles, sc.ntiles, nt_a, dst_a)};
            const gs = [2]LfDevice{ g2, g4 };
            var pair: [2][2]Lau = undefined;
            for (gs, 0..) |g, gi| {
                pair[gi][0] = k.lau(g.off, g.ent, g.erow, g.tiles, g.counts, g.nlocal, dst_b);
                pair[gi][1] = k.lau(g.off, g.ent, g.erow, g.tiles + g.nlocal * 8, g.counts + 4, g.nrest, dst_b);
            }
            const ms_a = try lauMs(r, &la);
            const ms_n = try lauMs(r, &ln);
            const ms_x = try lauMs(r, &lx);
            var ms_l: [2]f32 = undefined;
            var ms_r: [2]f32 = undefined;
            var ms_p: [2]f32 = undefined;
            for (0..2) |gi| {
                ms_l[gi] = try lauMs(r, pair[gi][0..1]);
                ms_r[gi] = try lauMs(r, pair[gi][1..2]);
                ms_p[gi] = try lauMs(r, &pair[gi]);
            }
            var t: WcText = .{};
            try t.add("roci_local_first {s} {s} (4,096 rows x 7 slots, kernel only): tiles (a) {d}; two parts local {d} + rest {d} = {d}; four quarters local {d} + rest {d} = {d}; ", .{ rname, kname, nt_a, g2.nlocal, g2.nrest, g2.nlocal + g2.nrest, g4.nlocal, g4.nrest, g4.nlocal + g4.nrest });
            try t.add("ms: (a) {d:.3} (null control, (a) again, {d:.1} %: {s}; grid of the exact tile count {d:.1} %); ", .{ ms_a, pct(ms_n, ms_a), resolved(pct(ms_n, ms_a)), pct(ms_x, ms_a) });
            for (0..2) |gi| try t.add("{s} local {d:.3} + rest {d:.3}, back to back {d:.3} = {d:.1} % over (a), the local launch is {d:.0} % of the tiles and {d:.0} % of the ms{s} ", .{ if (gi == 0) "two parts" else "four quarters", ms_l[gi], ms_r[gi], ms_p[gi], pct(ms_p[gi], ms_a), 100.0 * @as(f32, @floatFromInt(gs[gi].nlocal)) / @as(f32, @floatFromInt(gs[gi].nlocal + gs[gi].nrest)), 100.0 * ms_l[gi] / (ms_l[gi] + ms_r[gi]), if (gi == 0) ";" else "." });
            try t.add("The research's kill line (more than 1.5 % slower): two parts {s}, four quarters {s}.", .{ if (pct(ms_p[0], ms_a) > 1.5) "crossed" else "not crossed", if (pct(ms_p[1], ms_a) > 1.5) "crossed" else "not crossed" });
            r.note(t.buf[0..t.n], @abs(pct(ms_n, ms_a)) <= null_bar and holds);
            // the bytes: (a), then each layout's two launches, every output the same
            try poison(dst_a, bytes, d, r.s);
            try LauRun.go(.{ .r = r, .ls = &la });
            try r.s.synchronize();
            const ref = try get(dst_a, bytes, d);
            defer free(ref);
            var bad: [2]usize = undefined;
            for (0..2) |gi| {
                try poison(dst_b, bytes, d, r.s);
                try LauRun.go(.{ .r = r, .ls = &pair[gi] });
                try r.s.synchronize();
                const got = try get(dst_b, bytes, d);
                defer free(got);
                bad[gi] = differs(ref, got);
            }
            // control: the last tile of the rest launch never runs
            const cut = [2]Lau{ pair[1][0], blk: {
                var c = pair[1][1];
                c.gy -= 1;
                break :blk c;
            } };
            try poison(dst_b, bytes, d, r.s);
            try LauRun.go(.{ .r = r, .ls = &cut });
            try r.s.synchronize();
            const got = try get(dst_b, bytes, d);
            defer free(got);
            r.report(try std.fmt.bufPrint(&line, "roci_local_first {s} {s} control: the four-quarter layout's last rest tile not launched", .{ rname, kname }), differs(ref, got), false);
        }
    }
    // the attention front's dense GEMM (wq_a: K 5120, N 1280) on the gathered rows
    const K: usize = 5120;
    const N: usize = 1280;
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    const q = try cuda.DeviceBuffer.alloc(d, R * K);
    const qs = try cuda.DeviceBuffer.alloc(d, R * (K / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, R), 1) * 4);
    const y_a = try cuda.DeviceBuffer.alloc(d, R * N * 4);
    const y_n = try cuda.DeviceBuffer.alloc(d, R * N * 4);
    const y_b = try cuda.DeviceBuffer.alloc(d, R * N * 4);
    for ([_]cuda.DeviceBuffer{ q, qs, z, y_a, y_n, y_b }) |b| try keep.append(ha, b);
    try mx.quant(r.k, r.s, x.ptr, .bf16, X_D, R, K, q.ptr, K, qs.ptr, K / 32);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    const G = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        z: u64,
        y: u64,
        /// 0: one launch of all rows; 1: rows 0 .. 1,023 then 1,024 ..; 2: rows 0 .. 1,023 then 1,025 ..
        mode: u8,

        fn rows(g: @This(), r0: usize, m: usize) anyerror!void {
            try mx.gemm(g.r.k, g.r.s, g.l, g.q + r0 * K, K, g.s + r0 * (K / 32) * 4, K / 32, m, g.y + r0 * N * 4, N, .f32, false, g.z);
        }

        fn go(g: @This()) anyerror!void {
            switch (g.mode) {
                0 => try g.rows(0, R),
                1 => {
                    try g.rows(0, quarter);
                    try g.rows(quarter, R - quarter);
                },
                else => {
                    try g.rows(0, quarter);
                    try g.rows(quarter + 1, R - quarter - 1);
                },
            }
        }
    };
    const one: G = .{ .r = r, .l = l, .q = q.ptr, .s = qs.ptr, .z = z.ptr, .y = y_a.ptr, .mode = 0 };
    var two = one;
    two.y = y_b.ptr;
    two.mode = 1;
    var again = one;
    again.y = y_n.ptr;
    const ms1 = try kernelBest(r, G, one, G.go);
    const msn = try kernelBest(r, G, again, G.go);
    const ms2 = try kernelBest(r, G, two, G.go);
    const ybytes = R * N * 4;
    try poison(y_a.ptr, ybytes, d, r.s);
    try one.go();
    try r.s.synchronize();
    const ref = try get(y_a.ptr, ybytes, d);
    defer free(ref);
    try poison(y_b.ptr, ybytes, d, r.s);
    try two.go();
    try r.s.synchronize();
    const got = try get(y_b.ptr, ybytes, d);
    defer free(got);
    var cut = two;
    cut.mode = 2;
    try poison(y_b.ptr, ybytes, d, r.s);
    try cut.go();
    try r.s.synchronize();
    const got_cut = try get(y_b.ptr, ybytes, d);
    defer free(got_cut);
    const kinds = [3]?mx.SfGemm{ mx.sfRoute(r.k, l, q.ptr, K, qs.ptr, K / 32, y_a.ptr, N, false, R), mx.sfRoute(r.k, l, q.ptr, K, qs.ptr, K / 32, y_a.ptr, N, false, quarter), mx.sfRoute(r.k, l, q.ptr, K, qs.ptr, K / 32, y_a.ptr, N, false, R - quarter) };
    var t: WcText = .{};
    try t.add("roci_local_first wq_a (K {d}, N {d}) block-scaled exact on the gathered rows, kernel only, ms: one launch of 4,096 rows {d:.3} ({s}), null control {d:.1} % ({s}); 1,024 rows then 3,072 rows {d:.3} = {d:.1} % over one launch ({s} then {s}). The research's kill line (more than 1.5 % slower): {s}", .{
        if (pct(ms2, ms1) > 1.5) "crossed" else "not crossed",
    });
    r.report("roci_local_first wq_a: 1,024 rows then 3,072 rows against one launch of 4,096", differs(ref, got), true);
    r.report("roci_local_first wq_a control: row 1,024 not computed", differs(ref, got_cut), false);
}

// -- wq_b after the qr gather: the node's rows first -------------------------------------------------------------------------------

/// The cost of running wq_b (K 1,280, N 8,192, bf16 out) over the rows a node holds before the other nodes' qr rows have landed: the
/// owned quarter's launch first (a node of the second row block: rows 1,024 .. 2,047), then the rows before it, then the rows after it
/// (the three launches the gather's wait would sit before), against one launch of 4,096 rows, kernel only, and the bytes of the output
/// equal. The research's kill line: more than 1.5 % slower than one launch. A row left out must differ.
fn rociLocalFirstWqb(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x7b0b);
    const rng = prng.random();
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const R: usize = 4096;
    const quarter = R / 4;
    const K: usize = 1280;
    const N: usize = 8192;
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    const x = try upload(d, try bf16s(R * K, rng, 0.5, ha));
    const q = try cuda.DeviceBuffer.alloc(d, R * K);
    const qs = try cuda.DeviceBuffer.alloc(d, R * (K / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, R), 1) * 4);
    const y_a = try cuda.DeviceBuffer.alloc(d, R * N * 2);
    const y_n = try cuda.DeviceBuffer.alloc(d, R * N * 2);
    const y_b = try cuda.DeviceBuffer.alloc(d, R * N * 2);
    for ([_]cuda.DeviceBuffer{ x, q, qs, z, y_a, y_n, y_b }) |b| try keep.append(ha, b);
    try mx.quant(r.k, r.s, x.ptr, .bf16, K, R, K, q.ptr, K, qs.ptr, K / 32);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    const G = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        z: u64,
        y: u64,
        /// 0: one launch; 1: the owned quarter, the rows before it, the rows after it; 2: as 1 with the row after the owned quarter left out
        mode: u8,

        fn rows(g: @This(), r0: usize, m: usize) anyerror!void {
            if (m == 0) return;
            try mx.gemm(g.r.k, g.r.s, g.l, g.q + r0 * K, K, g.s + r0 * (K / 32) * 4, K / 32, m, g.y + r0 * N * 2, N, .bf16, false, g.z);
        }

        fn go(g: @This()) anyerror!void {
            switch (g.mode) {
                0 => try g.rows(0, R),
                1 => {
                    try g.rows(quarter, quarter);
                    try g.rows(0, quarter);
                    try g.rows(2 * quarter, 2 * quarter);
                },
                else => {
                    try g.rows(quarter, quarter);
                    try g.rows(0, quarter);
                    try g.rows(2 * quarter + 1, 2 * quarter - 1);
                },
            }
        }
    };
    const one: G = .{ .r = r, .l = l, .q = q.ptr, .s = qs.ptr, .z = z.ptr, .y = y_a.ptr, .mode = 0 };
    var three = one;
    three.y = y_b.ptr;
    three.mode = 1;
    var again = one;
    again.y = y_n.ptr;
    const ms1 = try kernelBest(r, G, one, G.go);
    const msn = try kernelBest(r, G, again, G.go);
    const ms3 = try kernelBest(r, G, three, G.go);
    const owned_only = try kernelBest(r, G, .{ .r = r, .l = l, .q = q.ptr, .s = qs.ptr, .z = z.ptr, .y = y_n.ptr, .mode = 3 }, struct {
        fn go(g: G) anyerror!void {
            try g.rows(quarter, quarter);
        }
    }.go);
    const ybytes = R * N * 2;
    try poison(y_a.ptr, ybytes, d, r.s);
    try one.go();
    try r.s.synchronize();
    const ref = try get(y_a.ptr, ybytes, d);
    defer free(ref);
    try poison(y_b.ptr, ybytes, d, r.s);
    try three.go();
    try r.s.synchronize();
    const got = try get(y_b.ptr, ybytes, d);
    defer free(got);
    var cut = three;
    cut.mode = 2;
    try poison(y_b.ptr, ybytes, d, r.s);
    try cut.go();
    try r.s.synchronize();
    const got_cut = try get(y_b.ptr, ybytes, d);
    defer free(got_cut);
    var t: WcText = .{};
    try t.add("roci_local_first wq_b (K {d}, N {d}) block-scaled exact on the gathered qr rows, kernel only, ms: one launch of 4,096 rows {d:.3} (null control {d:.1} % {s}); the owned quarter alone {d:.3}, then the rows before and after it {d:.3} = {d:.1} % over one launch. Kill line (more than 1.5 % slower): {s}", .{
        K, N, ms1, pct(msn, ms1), resolved(pct(msn, ms1)), owned_only, ms3, pct(ms3, ms1), if (pct(ms3, ms1) > 1.5) "crossed" else "not crossed",
    });
    r.report("roci_local_first wq_b: the owned quarter, the rows before it, the rows after it, against one launch of 4,096", differs(ref, got), true);
    r.report("roci_local_first wq_b control: the row after the owned quarter not computed", differs(ref, got_cut), false);
}

// -- wo_b on the two pairs' parts where they lie ---------------------------------------------------------------------------------

/// wo_b's block-scaled exact GEMM (K 4,096, N 2,560, 4,096 rows, y in the arena's write-combined memory) reading its activation rows from the
/// two pairs' codes parts where they lie (mx.gemmTwo: K blocks 0 .. 63 from the node's part, 64 .. 127 from the partner's) against the baseline path:
/// the four copies that assemble the input (prompt2d.catCodesCopies) and the GEMM on it. Every output byte equal; one code byte of the
/// partner's part changed must differ. The ms are the GEMM on the parts, and the copies plus the GEMM on the assembled input.
fn rociTwoBase(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x2ba5e);
    const rng = prng.random();
    const d = r.d;
    var line: [640]u8 = undefined;
    const n: usize = 4096;
    const uw: usize = 2048;
    const K: usize = 2 * uw;
    const N: usize = 2560;
    const ops = r.ops orelse return r.note("roci_two_base: the ops are available", false);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const ybytes = n * N * 4;
    var arena_mem: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&arena_mem, 2 * ybytes, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    defer _ = d.api.cuMemFreeHost(arena_mem);
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem, 0), "cuMemHostGetDevicePointer");
    try r.k.useStaged(wc, wc + 2 * ybytes);
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    const part_bytes = dsv41.prompt2d.catCodesBytes(n, uw);
    var parts: [2]u64 = undefined;
    for (0..2) |i| {
        const x = try upload(d, try bf16s(n * uw, rng, 0.5, ha));
        try keep.append(ha, x);
        const pb = try cuda.DeviceBuffer.alloc(d, part_bytes);
        try keep.append(ha, pb);
        parts[i] = pb.ptr;
        try mx.quant(r.k, r.s, x.ptr, .bf16, uw, n, uw, pb.ptr, uw, pb.ptr + n * uw, uw / 32);
    }
    const q = try cuda.DeviceBuffer.alloc(d, n * K);
    const qs = try cuda.DeviceBuffer.alloc(d, n * (K / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, n), 1) * 4);
    for ([_]cuda.DeviceBuffer{ q, qs, z }) |b| try keep.append(ha, b);
    try mx.sfProve(r.k, l.s);
    const y_ref = wc;
    const y_two = wc + ybytes;
    // node 1 (pair 0): its part is the first, the partner's the second
    const pl = dsv41.prompt2d.catCodesPlan(1, .{ 0, 1, 2, 3 }, n, .{ uw, uw }, 0, parts[0], parts[1], q.ptr, qs.ptr);
    if (!mx.twoTakes(r.k, l, parts[0], uw, parts[0] + n * uw, uw / 32, y_two, N, n)) return r.note("roci_two_base: the kernels take the two-base GEMM (exact, split128, y in the arena)", false);
    const C = struct {
        r: *Run,
        l: weights.Linear,
        pl: dsv41.prompt2d.CatCodes,
        p: [2]u64,
        z: u64,
        y: u64,
        two: bool,
        fn go(c: @This()) anyerror!void {
            if (c.two) {
                try mx.gemmTwo(c.r.k, c.r.s, c.l, c.p[0], uw, c.p[0] + n * uw, uw / 32, .{ .q1 = c.p[1], .s1 = c.p[1] + n * uw, .kb_split = uw / 32 }, n, c.y, N, c.z);
            } else {
                try dsv41.prompt2d.catCodesCopies(c.r.ops.?, c.r.s, c.pl);
                try mx.gemm(c.r.k, c.r.s, c.l, c.pl.q, c.pl.k, c.pl.s, c.pl.k / 32, n, c.y, N, .f32, false, c.z);
            }
        }
    };
    _ = ops;
    const today: C = .{ .r = r, .l = l, .pl = pl, .p = parts, .z = z.ptr, .y = y_ref, .two = false };
    var two = today;
    two.y = y_two;
    two.two = true;
    try poison(y_ref, ybytes, d, r.s);
    try today.go();
    try r.s.synchronize();
    try poison(y_two, ybytes, d, r.s);
    try two.go();
    try r.s.synchronize();
    const ref = try get(y_ref, ybytes, d);
    defer free(ref);
    const got = try get(y_two, ybytes, d);
    defer free(got);
    const ms_today = try kernelBest(r, C, today, C.go);
    const ms_two = try kernelBest(r, C, two, C.go);
    // the control: one code byte of the partner's part changed
    var one: [1]u8 = undefined;
    const flipped = [1]u8{one[0] ^ 0x08};
    try poison(y_two, ybytes, d, r.s);
    try two.go();
    try r.s.synchronize();
    const cut = try get(y_two, ybytes, d);
    defer free(cut);
    var t: WcText = .{};
    r.note(t.buf[0..t.n], true);
    r.report(try std.fmt.bufPrint(&line, "roci_two_base wo_b: the GEMM on the node's part and the partner's part against the copies and the GEMM on the assembled input", .{}), differs(ref, got), true);
    r.report("roci_two_base wo_b control: one code byte of the partner's part changed", differs(ref, cut), false);
}

// -- wo_b on the two pairs' parts: where each base lives -------------------------------------------------------------------------

/// roci_two_base in the engine's own memory: the node's part and the landed part, and wo_b's y, in the same block kind (the engine's arena is one
/// kind: write-combined host memory, locked anonymous 4 KiB pages registered with the driver as rdma.Pinned makes them, or those with
/// MADV_HUGEPAGE before first touch), against the replay's earlier setup (the parts in device memory, y in write-combined host memory).
/// For each kind: the exact whole-line kernel on the two bases (mx.gemmTwo) and the path it replaced, four copies into device memory and the plain
/// GEMM on them, kernel only, ms for 4,096 rows; the outputs byte for byte against the device-parts reference.
fn rociTwoBasePages(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x2ba5f);
    const rng = prng.random();
    const d = r.d;
    var line: [640]u8 = undefined;
    const n: usize = 4096;
    const uw: usize = 2048;
    const K: usize = 2 * uw;
    const N: usize = 2560;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const ybytes = n * N * 4;
    const part_bytes = dsv41.prompt2d.catCodesBytes(n, uw);
    const l = try fp8Linear(d, N, K, .fast, rng, ha, &keep);
    var src: [2]cuda.DeviceBuffer = undefined;
    for (0..2) |i| {
        const x = try upload(d, try bf16s(n * uw, rng, 0.5, ha));
        try keep.append(ha, x);
        src[i] = try cuda.DeviceBuffer.alloc(d, part_bytes);
        try keep.append(ha, src[i]);
        try mx.quant(r.k, r.s, x.ptr, .bf16, uw, n, uw, src[i].ptr, uw, src[i].ptr + n * uw, uw / 32);
    }
    const q = try cuda.DeviceBuffer.alloc(d, n * K);
    const qs = try cuda.DeviceBuffer.alloc(d, n * (K / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, n), 1) * 4);
    for ([_]cuda.DeviceBuffer{ q, qs, z }) |b| try keep.append(ha, b);
    try r.s.synchronize();
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    var hr = try dsv41.rdma.HostReg.open();
    defer hr.lib.close();
    var pool: WcPool = .{ .d = d };
    defer pool.deinit();
    const names = [_][]const u8{ "parts in device memory, y in write-combined host memory (the earlier replay)", "all in write-combined host memory", "all in locked 4 KiB pages (rdma.Pinned: no madvise)", "all in locked pages with MADV_HUGEPAGE before first touch" };
    var ref: ?[]u8 = null;
    defer if (ref) |x| free(x);
    var ms_all: [4][2]f32 = .{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } };
    var control_diff: usize = 0;
    for (0..4) |v| {
        // the three blocks of this kind: the two parts and y
        var p: [2]u64 = undefined;
        var y: u64 = undefined;
        var pinned: [3]?dsv41.rdma.Pinned = .{ null, null, null };
        defer for (&pinned) |*x| if (x.*) |*pp| pp.free(&hr);
        const sizes = [3]usize{ part_bytes, part_bytes, 2 * ybytes };
        var addr: [3]u64 = undefined;
        for (0..3) |bi| {
            addr[bi] = switch (v) {
                0 => if (bi < 2) src[bi].ptr else (try pool.alloc(.arena, sizes[bi])).dev,
                1 => (try pool.alloc(.arena, sizes[bi])).dev,
                2 => blk: {
                    pinned[bi] = try dsv41.rdma.Pinned.alloc(d, &hr, sizes[bi]);
                    break :blk pinned[bi].?.dev;
                },
                else => (try pool.alloc(.mmapreg, sizes[bi])).dev,
            };
        }
        p = .{ addr[0], addr[1] };
        y = addr[2];
        try r.s.synchronize();
        try r.k.useStaged(y, y + 2 * ybytes);
        const pl = dsv41.prompt2d.catCodesPlan(1, .{ 0, 1, 2, 3 }, n, .{ uw, uw }, 0, p[0], p[1], q.ptr, qs.ptr);
        if (!mx.twoTakes(r.k, l, p[0], uw, p[0] + n * uw, uw / 32, y + ybytes, N, n)) return r.note("roci_two_base_pages: the kernels take the two-base GEMM", false);
        const C = struct {
            r: *Run,
            l: weights.Linear,
            pl: dsv41.prompt2d.CatCodes,
            p: [2]u64,
            z: u64,
            y: u64,
            two: bool,
            fn go(c: @This()) anyerror!void {
                if (c.two) {
                    try mx.gemmTwo(c.r.k, c.r.s, c.l, c.p[0], uw, c.p[0] + n * uw, uw / 32, .{ .q1 = c.p[1], .s1 = c.p[1] + n * uw, .kb_split = uw / 32 }, n, c.y, N, c.z);
                } else {
                    try dsv41.prompt2d.catCodesCopies(c.r.ops.?, c.r.s, c.pl);
                    try mx.gemm(c.r.k, c.r.s, c.l, c.pl.q, c.pl.k, c.pl.s, c.pl.k / 32, n, c.y, N, .f32, false, c.z);
                }
            }
        };
        const today: C = .{ .r = r, .l = l, .pl = pl, .p = p, .z = z.ptr, .y = y, .two = false };
        var two = today;
        two.y = y + ybytes;
        two.two = true;
        try poison(y, 2 * ybytes, d, r.s);
        try today.go();
        try two.go();
        try r.s.synchronize();
        const a = try get(y, ybytes, d);
        const b = try get(y + ybytes, ybytes, d);
        defer free(b);
        if (ref == null) ref = a else free(a);
        ms_all[v][0] = try kernelBest(r, C, two, C.go);
        ms_all[v][1] = try kernelBest(r, C, today, C.go);
        r.note(try std.fmt.bufPrint(&line, "roci_two_base_pages {s}: wo_b on the two bases {d:.3} ms, the four copies and the plain GEMM {d:.3} ms ({d:.1} % on the two bases)", .{ names[v], ms_all[v][0], ms_all[v][1], pct(ms_all[v][0], ms_all[v][1]) }), true);
        r.report(try std.fmt.bufPrint(&line, "roci_two_base_pages {s}: the two-base output against the device-parts reference", .{names[v]}), differs(ref.?, b), true);
        if (v == 3) {
            var one: [1]u8 = undefined;
            const flipped = [1]u8{one[0] ^ 0x08};
            try poison(y + ybytes, ybytes, d, r.s);
            try two.go();
            try r.s.synchronize();
            const cut = try get(y + ybytes, ybytes, d);
            defer free(cut);
            control_diff = differs(ref.?, cut);
        }
    }
    r.report("roci_two_base_pages control: one code byte of the second part changed, in the last kind's block", control_diff, false);
}

// -- T-ANY: the owned path's launch classes on a node's slab against the whole chunk's launch ----------------------------------------

/// The chunk row counts T-ANY runs: below a block, around the 64-row variants, a tail of the 4,096 pass, and the sizes the lead measures.
const any_ns = [_]usize{ 13, 63, 64, 65, 500, 1087, 1956, 3000 };
const any_max: usize = 3000;

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

/// One launch class of the owned path over rows [start, start + m): `out` is its output buffer (every row-indexed pointer offset by the
/// start, as the engine launches a slab), `row` the output bytes a row.
const AnyClass = struct {
    name: []const u8,
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, start: usize, m: usize) anyerror!void,
    out: u64,
    row: usize,
    /// The section's name, the prefix of its lines and of a failing step's message.
    section: []const u8 = "roci_any_rows",
    /// The row counts it runs at.
    plan: *const AnyPlan = &any_plan,
};

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
        const bytes = n * c.row;
        poison(c.out, max * c.row, r.d, r.s) catch |e| return anyFail(c, n, null, "poison", e);
        c.run(c.ctx, 0, n) catch |e| return anyFail(c, n, null, "the one launch of the chunk's rows", e);
        r.s.synchronize() catch |e| return anyFail(c, n, null, "synchronize after the one launch", e);
        const ref = get(c.out, bytes, r.d) catch |e| return anyFail(c, n, null, "read back the one launch", e);
        defer free(ref);
        poison(c.out, max * c.row, r.d, r.s) catch |e| return anyFail(c, n, null, "poison", e);
        const sl = dsv41.slab_plan.slabs(n);
        for (sl, 0..) |sb, k| if (sb.n > 0) c.run(c.ctx, sb.start, sb.n) catch |e| return anyFail(c, n, .{ k, sb.start, sb.n }, "the slab's launch", e);
        r.s.synchronize() catch |e| return anyFail(c, n, null, "synchronize after the slabs", e);
        const got = get(c.out, bytes, r.d) catch |e| return anyFail(c, n, null, "read back the slabs", e);
        defer free(got);
        const ref_wrote = dsv41.replay_runner.wrotePoisoned(ref);
        if (ref_wrote and dsv41.replay_runner.wrotePoisoned(got)) {
            res.null_ok += 1;
            res.bad += differs(ref, got);
        } else {
            res.bad += bytes;
            if (res.first_null == null) res.first_null = .{ .n = n, .slabs = ref_wrote };
        }
        // the control: the last slab with more than one row, one row short
        poison(c.out, max * c.row, r.d, r.s) catch |e| return anyFail(c, n, null, "poison", e);
        const short = anyShort(n);
        for (sl, 0..) |sb, k| if (sb.n > 0) {
            const m = if (short != null and k == short.?) sb.n - 1 else sb.n;
            if (m > 0) c.run(c.ctx, sb.start, m) catch |e| return anyFail(c, n, .{ k, sb.start, m }, "the control's slab launch", e);
        };
        r.s.synchronize() catch |e| return anyFail(c, n, null, "synchronize after the control", e);
        const cut = get(c.out, bytes, r.d) catch |e| return anyFail(c, n, null, "read back the control", e);
        defer free(cut);
        res.controls_ok += @intFromBool(differs(ref, cut) > 0);
    }
    return res;
}

fn anyFail(c: AnyClass, n: usize, slab: ?[3]usize, step: []const u8, e: anyerror) anyerror {
    if (slab) |s| {
        std.debug.print("{s} {s}: FAILED at chunk n = {d}, slab {d} (start {d}, {d} rows), {s}: {s}\n", .{ c.section, c.name, n, s[0], s[1], s[2], step, @errorName(e) });
    } else std.debug.print("{s} {s}: FAILED at chunk n = {d}, {s}: {s}\n", .{ c.section, c.name, n, step, @errorName(e) });
    return e;
}

fn anyReport(r: *Run, c: AnyClass) !void {
    var line: [320]u8 = undefined;
    const res = try anyClassRun(r, c);
    r.report(try std.fmt.bufPrint(&line, "{s} {s}: the node's slab launches against the chunk's one launch at {s}", .{ c.section, c.name, c.plan.text }), res.bad, true);
    if (res.first_null) |f| {
        r.note(try std.fmt.bufPrint(&line, "{s} {s} null control: the one launch and the slabs wrote their outputs in {d} of {d} {s}; first at {d} rows: {s} did not", .{ c.section, c.name, res.null_ok, c.plan.ns.len, c.plan.unit, f.n, if (f.slabs) "the slabs' launches" else "the chunk's one launch" }), false);
    } else r.note(try std.fmt.bufPrint(&line, "{s} {s} null control: the one launch and the slabs wrote their outputs, in all {d} {s}", .{ c.section, c.name, c.plan.ns.len, c.plan.unit }), res.null_ok == c.plan.ns.len);
    r.report(try std.fmt.bufPrint(&line, "{s} {s} control: the last slab one row short, in all {d} {s}", .{ c.section, c.name, c.plan.ns.len, c.plan.unit }), if (res.controls_ok == c.plan.ns.len) 1 else 0, false);
}

/// TF_DS_ROCI_ANY's gate (T-ANY): each launch class the owned path makes at a slab's rows (act_quant, the dense FP8 GEMMs wq_a / wkv / wq_b on the
/// block-scaled exact kernels, the q RMSNorm, the router's logits and routing, the hyper-connection mix) at n_k rows of the slab plan against the one
/// launch of the chunk's n rows, at chunk row counts 13 to 3,000: every row's bytes equal. No engine path runs; this is the kernels' row-count
/// independence the ownership rule rests on.
fn rociAnyRows(r: *Run, rng_in: std.Random) !void {
    return rociRowsAt(r, rng_in, "roci_any_rows", &any_plan);
}

/// The decoder half on owned rows (layers 20-39 at a window of d decoder rows, the gate of moving its gathers onto the transport): the same
/// launch classes on each node's slab of every window d = 1 to 128 against the window's one launch, the baseline decoder half's.
fn rociDecRows(r: *Run, rng_in: std.Random) !void {
    return rociRowsAt(r, rng_in, "roci_dec_rows", &dec_plan);
}

fn rociRowsAt(r: *Run, rng_in: std.Random, section: []const u8, plan: *const AnyPlan) !void {
    var stage: []const u8 = "start";
    return rociAnyRowsBody(r, rng_in, &stage, section, plan) catch |e| {
        std.debug.print("{s}: FAILED in setup step \"{s}\": {s}\n", .{ section, stage, @errorName(e) });
        return e;
    };
}

fn rociAnyRowsBody(r: *Run, rng_in: std.Random, stage: *[]const u8, section: []const u8, plan: *const AnyPlan) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xa17);
    const rng = prng.random();
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    stage.* = "the activation of the plan's most rows and its act_quant";
    const M = plan.max;
    const D: usize = 5120;
    const x = try upload(d, try bf16s(M * D, rng, 0.5, ha));
    try keep.append(ha, x);
    const q5 = try cuda.DeviceBuffer.alloc(d, M * D);
    const s5 = try cuda.DeviceBuffer.alloc(d, M * (D / 32) * 4);
    try keep.append(ha, q5);
    try keep.append(ha, s5);
    try mx.quant(r.k, r.s, x.ptr, .bf16, D, M, D, q5.ptr, D, s5.ptr, D / 32);
    stage.* = "the q_a activation and its act_quant";
    const qa = try upload(d, try bf16s(M * 1280, rng, 1.0, ha));
    try keep.append(ha, qa);
    const q1 = try cuda.DeviceBuffer.alloc(d, M * 1280);
    const s1 = try cuda.DeviceBuffer.alloc(d, M * (1280 / 32) * 4);
    try keep.append(ha, q1);
    try keep.append(ha, s1);
    try mx.quant(r.k, r.s, qa.ptr, .bf16, 1280, M, 1280, q1.ptr, 1280, s1.ptr, 1280 / 32);
    stage.* = "the FP8 linears wq_a, wkv, wq_b";
    const wa = try fp8Linear(d, 1280, 5120, .fast, rng, ha, &keep);
    const wk = try fp8Linear(d, 512, 5120, .fast, rng, ha, &keep);
    const wb = try fp8Linear(d, 8192, 1280, .fast, rng, ha, &keep);
    stage.* = "the scale guards of the three linears (mx.sfProve)";
    try mx.sfProve(r.k, wa.s);
    try mx.sfProve(r.k, wk.s);
    try mx.sfProve(r.k, wb.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    stage.* = "the output, split-partial and router buffers";
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(wb, M), mx.zFloats(wa, M), mx.zFloats(wk, M), 1) * 4);
    try keep.append(ha, z);
    const out_big = try cuda.DeviceBuffer.alloc(d, M * 8192 * 2);
    try keep.append(ha, out_big);
    const nw = try upload(d, try bf16s(1280, rng, 1.0, ha));
    try keep.append(ha, nw);
    const gate_w = try upload(d, try bf16s(384 * D, rng, 0.5, ha));
    const gate_b = try upload(d, try f32s(384, rng, ha));
    try keep.append(ha, gate_w);
    try keep.append(ha, gate_b);
    const logits_in = try upload(d, try f32s(M * 384, rng, ha));
    try keep.append(ha, logits_in);

    stage.* = "the launch classes";
    const Quant = struct {
        r: *Run,
        x: u64,
        q: u64,
        s: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try mx.quant(c.r.k, c.r.s, c.x + start * D * 2, .bf16, D, m, D, c.q + start * D, D, c.s + start * (D / 32) * 4, D / 32);
        }
    };
    var qc: Quant = .{ .r = r, .x = x.ptr, .q = out_big.ptr, .s = out_big.ptr + M * D };
    // (quant's two outputs are checked through one buffer: codes then scales; the scales are compared by the GEMMs' use of them as well)
    try anyReport(r, .{ .name = "act_quant", .ctx = &qc, .run = Quant.run, .out = out_big.ptr, .row = D, .section = section, .plan = plan });

    const Gemm = struct {
        r: *Run,
        l: weights.Linear,
        q: u64,
        s: u64,
        k: usize,
        z: u64,
        y: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try mx.gemm(c.r.k, c.r.s, c.l, c.q + start * c.k, c.k, c.s + start * (c.k / 32) * 4, c.k / 32, m, c.y + start * c.l.n * 2, c.l.n, .bf16, false, c.z);
        }
    };
    var ga: Gemm = .{ .r = r, .l = wa, .q = q5.ptr, .s = s5.ptr, .k = 5120, .z = z.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "wq_a GEMM", .ctx = &ga, .run = Gemm.run, .out = out_big.ptr, .row = 1280 * 2, .section = section, .plan = plan });
    var gk: Gemm = .{ .r = r, .l = wk, .q = q5.ptr, .s = s5.ptr, .k = 5120, .z = z.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "wkv GEMM", .ctx = &gk, .run = Gemm.run, .out = out_big.ptr, .row = 512 * 2, .section = section, .plan = plan });
    var gb: Gemm = .{ .r = r, .l = wb, .q = q1.ptr, .s = s1.ptr, .k = 1280, .z = z.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "wq_b GEMM", .ctx = &gb, .run = Gemm.run, .out = out_big.ptr, .row = 8192 * 2, .section = section, .plan = plan });

    const Norm = struct {
        r: *Run,
        x: u64,
        w: u64,
        y: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try dsv41.tri_basic.rmsnorm(c.r.t, c.x + start * 1280 * 2, 1280, c.w, c.y + start * 1280 * 2, 1280, 1e-20, m, 1280);
        }
    };
    var nc: Norm = .{ .r = r, .x = qa.ptr, .w = nw.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "q RMSNorm", .ctx = &nc, .run = Norm.run, .out = out_big.ptr, .row = 1280 * 2, .section = section, .plan = plan });

    const Logits = struct {
        r: *Run,
        x: u64,
        w: u64,
        y: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try mx.bf16Gemm(c.r.k, c.r.s, c.x + start * D * 2, D, c.w, D, m, 384, D, c.y + start * 384 * 4, 384, .f32, 1, false, 1, 0, 0, 0, null);
        }
    };
    var lc: Logits = .{ .r = r, .x = x.ptr, .w = gate_w.ptr, .y = out_big.ptr };
    try anyReport(r, .{ .name = "router logits", .ctx = &lc, .run = Logits.run, .out = out_big.ptr, .row = 384 * 4, .section = section, .plan = plan });

    const Route = struct {
        r: *Run,
        logits: u64,
        bias: u64,
        pick: u64,
        wts: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try dsv41.tri_norm.route(c.r.t, c.logits + start * 384 * 4, 0, c.bias, 6, 1.5, 384, c.pick + start * 7 * 4, c.wts + start * 7 * 4, m, 384, 7);
        }
    };
    var rc: Route = .{ .r = r, .logits = logits_in.ptr, .bias = gate_b.ptr, .pick = out_big.ptr, .wts = out_big.ptr + M * 7 * 4 };
    // picks and weights are compared through the pick rows; the weights lie in the buffer after them
    try anyReport(r, .{ .name = "ffnRoute", .ctx = &rc, .run = Route.run, .out = out_big.ptr, .row = 7 * 4, .section = section, .plan = plan });

    if (r.t.extra == null) {
        var nl: [128]u8 = undefined;
        r.note(try std.fmt.bufPrint(&nl, "{s} hcPre: the kit's extra set loads", .{section}), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcPre null control: the kit's extra set loads", .{section}), false);
        r.note(try std.fmt.bufPrint(&nl, "{s} hcPre control: the kit's extra set loads", .{section}), false);
        return;
    }
    stage.* = "the hyper-connection mix buffers";
    const hc_blocks = dsv41.tri_basic.hc_blocks;
    const h = try upload(d, try bf16s(M * 4 * D, rng, 1.0, ha));
    const fnw = try upload(d, try f32s(24 * 4 * D, rng, ha));
    const scale = try upload(d, try f32s(3, rng, ha));
    const base = try upload(d, try f32s(24, rng, ha));
    const pre_in = try upload(d, try f32s(M * 4, rng, ha));
    const nw5 = try upload(d, try bf16s(D, rng, 1.0, ha));
    const part = try cuda.DeviceBuffer.alloc(d, M * hc_blocks * 32 * 4);
    const post = try cuda.DeviceBuffer.alloc(d, M * 4 * 4);
    const comb = try cuda.DeviceBuffer.alloc(d, M * 16 * 4);
    const pre_o = try cuda.DeviceBuffer.alloc(d, M * 4 * 4);
    for ([_]cuda.DeviceBuffer{ h, fnw, scale, base, pre_in, nw5, part, post, comb, pre_o }) |b| try keep.append(ha, b);
    const Hc = struct {
        r: *Run,
        h: u64,
        fnw: u64,
        scale: u64,
        base: u64,
        pre_in: u64,
        nw: u64,
        x: u64,
        pre_o: u64,
        post: u64,
        comb: u64,
        part: u64,
        fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try dsv41.tri_basic.hcPre(c.r.t, c.h + start * 4 * D * 2, c.fnw, c.scale, c.base, c.pre_in + start * 16, c.nw, 1e-20, 1e-6, 20, c.x + start * D * 2, c.pre_o + start * 16, c.post + start * 16, c.comb + start * 64, c.part + start * hc_blocks * 32 * 4, m, D);
        }
    };
    var hcc: Hc = .{ .r = r, .h = h.ptr, .fnw = fnw.ptr, .scale = scale.ptr, .base = base.ptr, .pre_in = pre_in.ptr, .nw = nw5.ptr, .x = out_big.ptr, .pre_o = pre_o.ptr, .post = post.ptr, .comb = comb.ptr, .part = part.ptr };
    try anyReport(r, .{ .name = "hcPre (x)", .ctx = &hcc, .run = Hc.run, .out = out_big.ptr, .row = D * 2, .section = section, .plan = plan });
}

// -- T-ANY, the rest of the owned path's launch classes: the grouped experts, the index queries, the Engram gate ---------------------------

/// The grouped experts at a chunk of any row count: gate / up from the activation codes the nodes' slabs quantized into the scratch
/// (mx.quant of each slab's rows at the slab's row offsets, as ffnRows does; the codes of the whole chunk gathered), mx.gateUpCodes at the
/// chunk's n rows, against mx.gateUp's own quantization of the n rows: every byte of the pack equal, at chunk row counts 13 to 3,000.
/// The control leaves the last row of the last slab with more than one row unquantized: the pack must differ.
fn rociAnyExperts(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xa1e);
    const rng = prng.random();
    const d = r.d;
    var line: [256]u8 = undefined;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const M = any_max;
    const ex = try xExperts(d, rng, ha, &keep);
    const sc = try xScratch(d, M, true, ha, &keep);
    const hb_bytes = M * X_SL * X_W * 2;
    const hb_ref = try cuda.DeviceBuffer.alloc(d, hb_bytes);
    const hb_got = try cuda.DeviceBuffer.alloc(d, hb_bytes);
    try keep.append(ha, hb_ref);
    try keep.append(ha, hb_got);
    var bad: usize = 0;
    var controls_ok: usize = 0;
    for (any_ns) |n| {
        const x = try upload(d, try bf16s(n * X_D, rng, 0.5, ha));
        const pw = try xPicks(n, rng, ha);
        const pb = try upload(d, std.mem.sliceAsBytes(pw.pick));
        const wb = try upload(d, std.mem.sliceAsBytes(pw.wts));
        for ([_]cuda.DeviceBuffer{ x, pb, wb }) |b| try keep.append(ha, b);
        const bytes = n * X_SL * X_W * 2;
        try poison(hb_ref.ptr, hb_bytes, d, r.s);
        try mx.gateUp(r.k, r.s, d, ex, sc, x.ptr, X_D, pb.ptr, wb.ptr, n, 7.0, hb_ref.ptr, X_MAIN, false);
        try r.s.synchronize();
        const ref = try get(hb_ref.ptr, bytes, d);
        defer free(ref);
        const sl = dsv41.slab_plan.slabs(n);
        var short: ?usize = null;
        for (sl, 0..) |sb, k| if (sb.n > 1) {
            short = k;
        };
        for ([_]bool{ false, true }) |control| {
            try poison(sc.q, M * X_D, d, r.s);
            try poison(sc.s, M * (X_D / 32) * 4, d, r.s);
            try poison(hb_got.ptr, hb_bytes, d, r.s);
            for (sl, 0..) |sb, k| if (sb.n > 0) {
                const m = if (control and short != null and k == short.?) sb.n - 1 else sb.n;
                try mx.quant(r.k, r.s, x.ptr + sb.start * X_D * 2, .bf16, X_D, m, X_D, sc.q + sb.start * X_D, X_D, sc.s + sb.start * (X_D / 32) * 4, X_D / 32);
            };
            try mx.gateUpCodes(r.k, r.s, d, ex, sc, pb.ptr, wb.ptr, n, 7.0, hb_got.ptr, X_MAIN);
            try r.s.synchronize();
            const got = try get(hb_got.ptr, bytes, d);
            defer free(got);
            if (control) controls_ok += @intFromBool(differs(ref, got) > 0) else bad += differs(ref, got);
        }
    }
    r.report(try std.fmt.bufPrint(&line, "roci_any_experts: gate / up from the slabs' codes at the chunk's n rows against mx.gateUp's own quantization at n = 13, 63, 64, 65, 500, 1,087, 1,956, 3,000", .{}), bad, true);
    r.report(try std.fmt.bufPrint(&line, "roci_any_experts control: the last slab one row unquantized, in all {d} chunks", .{any_ns.len}), if (controls_ok == any_ns.len) 1 else 0, false);
}

const AI_QW: usize = 1280;
const AI_D: usize = 5120;
const AI_IH: usize = 32;
const AI_ID: usize = 128;
const AI_RD: usize = 64;
const AI_START: usize = 70000;

/// The indexer's per-row launches on a slab (prompt.indexer with TF_DS_IDX_SHARD): wq_b through mx.linear into iq, idxQueries (the RoPE of iq's
/// rows and fp4_qd's bytes into iq4), weights_proj into iw, each at the slab's row offsets, against the one launch of the chunk's n rows.
const IdxCtx = struct {
    r: *Run,
    sc: mx.Scratch,
    wqb: weights.Linear,
    qr: u64,
    x: u64,
    proj: u64,
    cos: u64,
    sin: u64,
    pos: u64,
    iq: u64,
    iq4: u64,
    iw: u64,
    fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
        const c: *@This() = @ptrCast(@alignCast(p));
        try idxProjectRows(c.r, c.sc, c.wqb, c.qr, c.x, c.proj, c.cos, c.sin, c.pos, .{ .start = start, .n = m }, c.iq, c.iq4, c.iw);
    }
};

fn idxProjectRows(r: *Run, sc: mx.Scratch, wqb: weights.Linear, qr: u64, x: u64, proj: u64, cos: u64, sin: u64, pos: u64, rows: dsv41.prompt.QRows, iq: u64, iq4: u64, iw: u64) !void {
    try mx.linear(r.k, r.s, sc, wqb, qr + rows.start * AI_QW * 2, .bf16, AI_QW, rows.n, iq + rows.start * AI_IH * AI_ID * 2, AI_IH * AI_ID, .bf16, false, null);
    try dsv41.prompt.idxQueries(r.t, iq, iq4, cos, sin, pos, rows, AI_RD, AI_IH, AI_ID);
    try mx.bf16Gemm(r.k, r.s, x + rows.start * AI_D * 2, AI_D, proj, AI_D, rows.n, AI_IH, AI_D, iw + rows.start * AI_IH * 2, AI_IH, .bf16, dsv41.exact.indexScale(AI_ID, AI_IH), true, 1, 0, 0, 0, null);
}

fn rociAnyIndex(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xa1d);
    const rng = prng.random();
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const M = any_max;
    const wqb = try fp8Linear(d, AI_IH * AI_ID, AI_QW, .fast, rng, ha, &keep);
    const qr = try upload(d, try bf16s(M * AI_QW, rng, 0.5, ha));
    const x = try upload(d, try bf16s(M * AI_D, rng, 0.5, ha));
    const proj = try upload(d, try bf16s(AI_IH * AI_D, rng, 0.05, ha));
    const tab = try ha.alloc(f32, (AI_START + M) * (AI_RD / 2));
    for (tab) |*v| v.* = rng.float(f32) * 2 - 1;
    const cos = try upload(d, std.mem.sliceAsBytes(tab));
    for (tab) |*v| v.* = rng.float(f32) * 2 - 1;
    const sin = try upload(d, std.mem.sliceAsBytes(tab));
    const ph = try ha.alloc(i64, M);
    for (ph, 0..) |*pv, i| pv.* = @intCast(AI_START + i);
    const pos = try upload(d, std.mem.sliceAsBytes(ph));
    for ([_]cuda.DeviceBuffer{ qr, x, proj, cos, sin, pos }) |b| try keep.append(ha, b);
    var sb: [4]cuda.DeviceBuffer = undefined;
    for (&sb, mx.Scratch.sizes(M, AI_QW, @max(mx.zFloats(wqb, M), 1))) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n);
        try keep.append(ha, b.*);
    }
    try mx.sfProve(r.k, wqb.s);
    r.k.sf = .exact;
    defer r.k.sf = .off;
    const iq = try cuda.DeviceBuffer.alloc(d, M * AI_IH * AI_ID * 2);
    const iq4 = try cuda.DeviceBuffer.alloc(d, M * AI_IH * AI_ID * 2);
    const iw = try cuda.DeviceBuffer.alloc(d, M * AI_IH * 2);
    for ([_]cuda.DeviceBuffer{ iq, iq4, iw }) |b| try keep.append(ha, b);
    var ctx: IdxCtx = .{ .r = r, .sc = .{ .q = sb[0].ptr, .s = sb[1].ptr, .z = sb[2].ptr, .fake = sb[3].ptr, .rows = M, .max_k = AI_QW }, .wqb = wqb, .qr = qr.ptr, .x = x.ptr, .proj = proj.ptr, .cos = cos.ptr, .sin = sin.ptr, .pos = pos.ptr, .iq = iq.ptr, .iq4 = iq4.ptr, .iw = iw.ptr };
    try anyReport(r, .{ .name = "index queries (fp4_qd bytes)", .ctx = &ctx, .run = IdxCtx.run, .out = iq4.ptr, .row = AI_IH * AI_ID * 2, .section = "roci_any_index" });
    try anyReport(r, .{ .name = "index weights (weights_proj)", .ctx = &ctx, .run = IdxCtx.run, .out = iw.ptr, .row = AI_IH * 2, .section = "roci_any_index" });
}

/// The Engram gate on a slab: _engram_gate on h and the assembled key and value rows at the slab's row offsets against the chunk's one launch; and
/// _engram_gate_2b reading the transport's two blocks with a block stride of q rows (the slab plan's stride, the last slab shorter than q) against
/// the assembled rows and _engram_gate.
const EgCtx = struct {
    r: *Run,
    h: u64,
    kv: u64,
    qk: u64,
    out: u64,
    fn run(p: *anyopaque, start: usize, m: usize) anyerror!void {
        const c: *@This() = @ptrCast(@alignCast(p));
        try dsv41.tri_basic.engramGate(c.r.t, c.h + start * 4 * 5120 * 2, c.kv + start * 5 * 5120 * 2, c.qk, c.out + start * 4 * 5120 * 2, 1e-20, m, 5120);
    }
};

fn rociAnyEngram(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xa1c);
    const rng = prng.random();
    const d = r.d;
    const D: usize = 5120;
    const EW: usize = 12800;
    var line: [256]u8 = undefined;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    const M = any_max;
    const h = try upload(d, try bf16s(M * 4 * D, rng, 1.0, ha));
    const kv = try upload(d, try bf16s(M * 5 * D, rng, 1.0, ha));
    const qk = try upload(d, try f32s(4 * D, rng, ha));
    const out = try cuda.DeviceBuffer.alloc(d, M * 4 * D * 2);
    for ([_]cuda.DeviceBuffer{ h, kv, qk, out }) |b| try keep.append(ha, b);
    var ctx: EgCtx = .{ .r = r, .h = h.ptr, .kv = kv.ptr, .qk = qk.ptr, .out = out.ptr };
    try anyReport(r, .{ .name = "_engram_gate", .ctx = &ctx, .run = EgCtx.run, .out = out.ptr, .row = 4 * D * 2, .section = "roci_any_engram" });
    if (r.t.extra == null) return r.note("roci_any_engram _engram_gate_2b: the kit's extra set loads", false);
    // the two blocks at the slab plan's stride: the slab's rows only are read, q rows apart
    var bad: usize = 0;
    var controls_ok: usize = 0;
    var tested: usize = 0;
    for ([_]usize{ 13, 63, 65, 500, 1087 }) |n| {
        const q = dsv41.slab_plan.q(n);
        const m = dsv41.slab_plan.slab(n, 3).n; // the last slab: m <= q rows, the stride q
        if (m == 0 or m == q) continue;
        tested += 1;
        const eb = try rociBlocks(q, EW, rng, ha);
        var ekb = try upload(d, std.mem.sliceAsBytes(eb));
        defer ekb.free();
        var kva = try cuda.DeviceBuffer.alloc(d, m * 2 * EW * 2);
        defer kva.free();
        for (0..2) |b| try r.ops.?.copyRows(r.s, ekb.ptr + b * q * EW * 2, EW * 2, kva.ptr + b * EW * 2, 2 * EW * 2, EW * 2, m);
        var o: [2]cuda.DeviceBuffer = undefined;
        for (&o, 0..) |*ob, i| {
            ob.* = try cuda.DeviceBuffer.alloc(d, m * 4 * D * 2);
            try poison(ob.ptr, m * 4 * D * 2, d, r.s);
            if (i == 0) try dsv41.tri_basic.engramGate(r.t, h.ptr, kva.ptr, qk.ptr, ob.ptr, 1e-20, m, D) else try dsv41.tri_basic.engramGate2b(r.t, h.ptr, ekb.ptr, q * EW, qk.ptr, ob.ptr, 1e-20, m, D);
        }
        try r.s.synchronize();
        const g0 = try get(o[0].ptr, m * 4 * D * 2, d);
        const g1 = try get(o[1].ptr, m * 4 * D * 2, d);
        bad += differs(g0, g1);
        free(g1);
        for (&o) |*ob| ob.free();
        // the control: the stride one row short reads the second block from the wrong place
        var oc = try cuda.DeviceBuffer.alloc(d, m * 4 * D * 2);
        defer oc.free();
        try poison(oc.ptr, m * 4 * D * 2, d, r.s);
        try dsv41.tri_basic.engramGate2b(r.t, h.ptr, ekb.ptr, (q - 1) * EW, qk.ptr, oc.ptr, 1e-20, m, D);
        try r.s.synchronize();
        const gc = try get(oc.ptr, m * 4 * D * 2, d);
        controls_ok += @intFromBool(differs(g0, gc) > 0);
        free(gc);
        free(g0);
    }
    r.report(try std.fmt.bufPrint(&line, "roci_any_engram: _engram_gate_2b on the last slab at the plan's block stride q against the assembled rows and _engram_gate ({d} chunk row counts)", .{tested}), bad, true);
    r.report("roci_any_engram control: the block stride one row short in every chunk", if (controls_ok == tested and tested != 0) 1 else 0, false);
}

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

// -- wo_b and the MoE's routed sum with the sent quarters first, in one launch ------------------------------------------------

/// The noise bar of a null control: the rig's measured noise floor. A time line is "as it must" (its instrument holds) while the null
/// control, the same launch again, reads within it; a verdict (a kill line crossed, a difference outside the null) is in the text.
const null_bar: f32 = 5.0;

/// ms a launch: the best of three measurements of 20 launches.
fn kernelBest(r: *Run, comptime Ctx: type, ctx: Ctx, comptime f: fn (Ctx) anyerror!void) !f32 {
    var best = try kernelMs(r, Ctx, ctx, f, 20);
    for (0..2) |_| best = @min(best, try kernelMs(r, Ctx, ctx, f, 20));
    return best;
}

/// "within" / "outside" the null control's reading, for a text.
fn resolved(null_pct: f32) []const u8 {
    return if (@abs(null_pct) <= 2.0) "resolved" else if (@abs(null_pct) <= null_bar) "noisy: differences under the null control's reading are not resolved" else "NOT resolved";
}

const sf_m: usize = 4096;
const sf_k: usize = 4096;
const sf_n: usize = 2560;
const sf_rt: u32 = sf_m / 128; // row tiles
const sf_ct: u32 = sf_n / 128; // column tiles

/// A launch of wo_b's block-scaled exact GEMM (K 4,096, N 2,560, 4,096 rows, y in the arena) in `ro`'s order: `f` the kernel that
/// takes (MxArgs, RowOrder), plus the Door when `door` and the stamps when `ts`.
const SentRun = struct {
    r: *Run,
    f: cuda.Function,
    a: mx.MxArgs,
    ro: mx.RowOrder,
    door: ?mx.Door = null,
    /// The timed twins' stamps [CTA in launch order][begin, end] (u64 ns) and, past them, the ring's; 0: untimed.
    ts: u64 = 0,

    fn go(c: @This()) anyerror!void {
        var args: cuda.Args = .{};
        args.add(c.a);
        args.add(c.ro);
        if (c.door) |dr| args.add(dr);
        if (c.ts != 0) args.add(c.ts);
        try cuda.launch.launch(c.f, .{ .grid = .{ .x = sf_rt, .y = sf_ct, .z = 1 }, .block = .{ .x = 256 }, .shared = mx.sf_split128_smem }, c.r.s, &args);
    }
};

/// The baseline launch: mx.gemm with y in the arena (the whole-line kernel).
const TodayRun = struct {
    r: *Run,
    l: weights.Linear,
    q: u64,
    s: u64,
    z: u64,
    y: u64,

    fn go(c: @This()) anyerror!void {
        try mx.gemm(c.r.k, c.r.s, c.l, c.q, sf_k, c.s, sf_k / 32, sf_m, c.y, sf_n, .f32, false, c.z);
    }
};

/// The span of a timed launch (stamps [begin, end] per CTA in launch order, and past them the ring's) in %: the CTA `n` of them
/// that ended last gives 100; `upto` is the time to place (a CTA set's last end, or the ring's stamp). Null when a CTA did not stamp.
fn stampSpan(ts: []const u64, n: usize, upto: u64) ?f64 {
    var t0: u64 = std.math.maxInt(u64);
    var t1: u64 = 0;
    for (0..n) |i| {
        const b = ts[2 * i];
        const e = ts[2 * i + 1];
        if (b == 0 or e < b) return null;
        t0 = @min(t0, b);
        t1 = @max(t1, e);
    }
    if (upto < t0 or t1 <= t0) return null;
    return 100.0 * @as(f64, @floatFromInt(upto - t0)) / @as(f64, @floatFromInt(t1 - t0));
}

/// The last end among the CTAs `sent[i]` names.
fn lastEnd(ts: []const u64, sent: []const bool) u64 {
    var e: u64 = 0;
    for (sent, 0..) |s, i| {
        if (s) e = @max(e, ts[2 * i + 1]);
    }
    return e;
}

fn u64s(raw: []u8) []const u64 {
    return @alignCast(std.mem.bytesAsSlice(u64, raw));
}

/// The wo_b launch's operands: the arena y blocks and the quantized operand, built once for the two sections.
const SfRig = struct {
    l: weights.Linear,
    q: u64,
    s: u64,
    z: u64,
    margs: mx.MxArgs,
    wc: u64,
    y_ref: u64,
    y_new: u64,
    y_null: u64,
};

fn sfRig(r: *Run, rng: std.Random, ha: std.mem.Allocator, keep: *std.ArrayList(cuda.DeviceBuffer), arena_mem: *?*anyopaque) !?SfRig {
    const d = r.d;
    const ybytes = sf_m * sf_n * 4;
    try d.check(d.api.cuMemHostAlloc(arena_mem, 3 * ybytes, 0x01 | 0x02 | 0x04), "cuMemHostAlloc");
    var wc: u64 = 0;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&wc, arena_mem.*, 0), "cuMemHostGetDevicePointer");
    try r.k.useStaged(wc, wc + 3 * ybytes);
    const l = try fp8Linear(d, sf_n, sf_k, .fast, rng, ha, keep);
    const x = try upload(d, try bf16s(sf_m * sf_k, rng, 0.5, ha));
    const q = try cuda.DeviceBuffer.alloc(d, sf_m * sf_k);
    const qs = try cuda.DeviceBuffer.alloc(d, sf_m * (sf_k / 32) * 4);
    const z = try cuda.DeviceBuffer.alloc(d, @max(mx.zFloats(l, sf_m), 1) * 4);
    for ([_]cuda.DeviceBuffer{ x, q, qs, z }) |b| try keep.append(ha, b);
    try mx.quant(r.k, r.s, x.ptr, .bf16, sf_k, sf_m, sf_k, q.ptr, sf_k, qs.ptr, sf_k / 32);
    try mx.sfProve(r.k, l.s);
    r.k.sf = .exact;
    const kind = mx.sfRoute(r.k, l, q.ptr, sf_k, qs.ptr, sf_k / 32, wc, sf_n, false, sf_m);
    if (kind == null or kind.? != .split128) return null;
    const sp = mx.splits(sf_k, sf_n);
    const margs: mx.MxArgs = .{ .q = q.ptr, .s = qs.ptr, .ldq = sf_k, .lds = sf_k / 32, .w = l.w, .ws = l.s, .ldw = sf_k, .ldws = sf_k / 32, .fmt = 0, .M = sf_m, .N = sf_n, .K = sf_k, .splits = @intCast(sp), .kblocks = @intCast((sf_k / 32 + sp - 1) / sp), .z = z.ptr, .y = wc + ybytes, .ldy = sf_n, .y_dtype = @backingInt(mx.DType.f32), .accumulate = 0 };
    return .{ .l = l, .q = q.ptr, .s = qs.ptr, .z = z.ptr, .margs = margs, .wc = wc, .y_ref = wc, .y_new = wc + ybytes, .y_null = wc + 2 * ybytes };
}

/// wo_b's block-scaled exact GEMM (K 4,096, N 2,560, 4,096 rows, y in the arena's write-combined host memory: the whole-line
/// kernel) in one launch whose CTAs take the row tiles of the two quarters a node's first-part send takes before the other two's
/// (tf_dsn_sf_split128_exact_st_first: a row order, every column of the sent quarters first), against the baseline launch: ms, bytes,
/// and when the sent quarters are done, from each CTA's %globaltimer at its begin and end (the timed twin).
fn rociSentFirst(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0x5e17f1);
    const rng = prng.random();
    const d = r.d;
    const sk = r.k.sfk orelse return r.note("roci_sent_first: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    const f_first = try sk.module.function("tf_dsn_sf_split128_exact_st_first_kernel");
    try f_first.allowDynamicShared(mx.sf_split128_smem);
    const f_time = try sk.module.function("tf_dsn_sf_split128_exact_st_first_t_kernel");
    try f_time.allowDynamicShared(mx.sf_split128_smem);
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var arena_mem: ?*anyopaque = null;
    defer if (arena_mem != null) {
        _ = d.api.cuMemFreeHost(arena_mem);
    };
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    const ybytes = sf_m * sf_n * 4;
    const rig = (try sfRig(r, rng, ha, &keep, &arena_mem)) orelse return r.note("roci_sent_first: wo_b's 4,096 rows take tf_dsn_sf_split128 here", false);
    const stamps = try cuda.DeviceBuffer.alloc(d, (sf_rt * sf_ct * 2 + 1) * 8);
    try keep.append(ha, stamps);
    const q = sf_m / 4;
    const ident = mx.identityRowOrder(sf_m, sf_ct).?;
    var orders: [4]mx.RowOrder = undefined;
    for (0..4) |g| orders[g] = mx.rowOrder(.{ (g ^ 1) * q, (g ^ 3) * q }, .{ g * q, (g ^ 2) * q }, q, sf_ct).?;
    // a permutation that is not one: quarter g twice, quarter g ^ 2 never
    var dup = orders[0];
    dup.quarters = (dup.quarters & 0x00ffffff) | (@as(u32, 0) << 24);
    const today: TodayRun = .{ .r = r, .l = rig.l, .q = rig.q, .s = rig.s, .z = rig.z, .y = rig.y_ref };
    var today_n = today;
    today_n.y = rig.y_null;
    const sent_run: SentRun = .{ .r = r, .f = f_first, .a = rig.margs, .ro = orders[0] };
    var id_run = sent_run;
    id_run.ro = ident;
    var t_sent = sent_run;
    t_sent.f = f_time;
    t_sent.ts = stamps.ptr;
    var t_id = id_run;
    t_id.f = f_time;
    t_id.ts = stamps.ptr;
    const ms_t = try kernelBest(r, TodayRun, today, TodayRun.go);
    const ms_n = try kernelBest(r, TodayRun, today_n, TodayRun.go);
    const ms_s = try kernelBest(r, SentRun, sent_run, SentRun.go);
    const ms_i = try kernelBest(r, SentRun, id_run, SentRun.go);
    const ms_ts = try kernelBest(r, SentRun, t_sent, SentRun.go);
    const ms_ti = try kernelBest(r, SentRun, t_id, SentRun.go);
    const null_pct = pct(ms_n, ms_t);
    const bar = @max(1.0, @abs(null_pct));
    var t: WcText = .{};
    r.note(t.buf[0..t.n], @abs(null_pct) <= null_bar);
    // when the sent quarters are done: 5 launches of each order, the stamps read back
    var sent_first_l: [sf_rt * sf_ct]bool = undefined;
    var sent_id_l: [sf_rt * sf_ct]bool = undefined;
    for (0..sf_rt * sf_ct) |i| {
        sent_first_l[i] = i < ident.span / 2;
        const quarter = (i % sf_rt) / 8;
        sent_id_l[i] = quarter == 1 or quarter == 3;
    }
    var span: [2][5]f64 = undefined;
    var sane = true;
    var res_ns: u64 = std.math.maxInt(u64);
    for ([_]SentRun{ t_sent, t_id }, 0..) |run, ki| for (0..5) |rep| {
        try d.check(d.api.cuMemsetD8Async(stamps.ptr, 0, sf_rt * sf_ct * 16, r.s.handle), "cuMemsetD8Async");
        try run.go();
        try r.s.synchronize();
        const raw = try get(stamps.ptr, sf_rt * sf_ct * 16, d);
        defer free(raw);
        const ts = u64s(raw);
        const sent = if (ki == 0) &sent_first_l else &sent_id_l;
        const sc_v = stampSpan(ts, sf_rt * sf_ct, lastEnd(ts, sent));
        sane = sane and sc_v != null;
        span[ki][rep] = sc_v orelse 0;
        // the stamps' resolution: the smallest difference between two different stamps
        var sorted: [sf_rt * sf_ct * 2]u64 = undefined;
        @memcpy(&sorted, ts[0 .. sf_rt * sf_ct * 2]);
        std.mem.sort(u64, &sorted, {}, std.sort.asc(u64));
        for (sorted[1..], sorted[0 .. sorted.len - 1]) |hi, lo| if (hi > lo) {
            res_ns = @min(res_ns, hi - lo);
        };
    };
    for (&span) |*s_| std.mem.sort(f64, s_, {}, std.sort.asc(f64));
    var t2: WcText = .{};
    try t2.add("the stamps: {d} CTAs, smallest step between two stamps {d} ns. The send could start there, less the stores' completion and the fence", .{ sf_rt * sf_ct, res_ns });
    r.note(t2.buf[0..t2.n], sane and span[1][2] >= 90.0);
    // the bytes
    try poison(rig.y_ref, ybytes, d, r.s);
    try today.go();
    try r.s.synchronize();
    const ref = try get(rig.y_ref, ybytes, d);
    defer free(ref);
    var bad: usize = 0;
    for (0..4) |g| {
        var run = sent_run;
        run.ro = orders[g];
        try poison(rig.y_new, ybytes, d, r.s);
        try run.go();
        try r.s.synchronize();
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(ref, got);
    }
    bad = 0;
    for ([_]SentRun{ id_run, t_sent, t_id }) |run| {
        try poison(rig.y_new, ybytes, d, r.s);
        try run.go();
        try r.s.synchronize();
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(ref, got);
    }
    var cut = sent_run;
    cut.ro = dup;
    try poison(rig.y_new, ybytes, d, r.s);
    try cut.go();
    try r.s.synchronize();
    const got = try get(rig.y_new, ybytes, d);
    defer free(got);
    r.report("roci_sent_first wo_b control: a row order that runs one quarter twice and another never", differs(ref, got), false);
}

/// The doorbell line a test kernel rings (host-mapped memory, not write-combined, as the transport's control block), the
/// arrival count (a device word, zero between launches) and the observer's two result words.
const DoorRig = struct {
    d: *const cuda.Driver,
    mem: ?*anyopaque = null,
    dev: u64 = 0,
    cnt: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    ticket: u64 = 0,

    fn init(d: *const cuda.Driver) !DoorRig {
        var g: DoorRig = .{ .d = d, .cnt = try cuda.DeviceBuffer.alloc(d, 256), .out = try cuda.DeviceBuffer.alloc(d, 256) };
        errdefer g.cnt.free();
        errdefer g.out.free();
        try d.check(d.api.cuMemHostAlloc(&g.mem, 4096, 0x01 | 0x02), "cuMemHostAlloc");
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&g.dev, g.mem, 0), "cuMemHostGetDevicePointer");
        @memset(@as([*]u8, @ptrCast(g.mem.?))[0..4096], 0);
        return g;
    }

    fn deinit(g: *DoorRig) void {
        _ = g.d.api.cuMemFreeHost(g.mem);
        g.cnt.free();
        g.out.free();
    }

    fn line(g: *const DoorRig) *volatile u64 {
        return @ptrCast(@alignCast(g.mem.?));
    }

    /// The next ticket, the line cleared; the Door a producer rings.
    fn arm(g: *DoorRig) mx.Door {
        g.ticket += 1;
        g.line().* = 0;
        return .{ .cnt = g.cnt.ptr, .door = g.dev, .ticket = g.ticket };
    }

    /// Whether the line shows the ticket and the arrival count is zero again: the producer rang once and left the count clean.
    fn rang(g: *const DoorRig) !bool {
        var c: u32 = 1;
        return g.line().* == g.ticket and c == 0;
    }
};

/// One producer launch, observed: the observer kernel (a CTA on `sw`) is started and given a moment to be waiting; then the
/// producer `P.go(p)` runs on the run's stream (it must ring `rig.ticket`); once both have ended the observer's count of words that
/// differed from their references at the moment the doorbell showed, and whether it showed.
fn observed(r: *Run, rig: *DoorRig, wf: cuda.Function, sw: cuda.Stream, ref: [2]u64, got: [2]u64, words: u64, comptime P: type, p: P) !struct { wrong: u32, seen: bool } {
    try r.d.check(r.d.api.cuMemsetD8Async(rig.out.ptr, 0, 256, sw.handle), "cuMemsetD8Async");
    var a: cuda.Args = .{};
    a.add(rig.dev);
    a.add(rig.ticket);
    a.add(ref[0]);
    a.add(got[0]);
    a.add(ref[1]);
    a.add(got[1]);
    a.add(words);
    a.add(rig.out.ptr);
    a.add(@as(u64, 3_000_000_000));
    try cuda.launch.launch(wf, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024 } }, sw, &a);
    std.Io.sleep(r.io.?, .fromMilliseconds(3), .awake) catch {};
    try P.go(p);
    try r.s.synchronize();
    try sw.synchronize();
    var o: [2]u32 = undefined;
    return .{ .wrong = o[0], .seen = o[1] != 0 };
}

/// wo_b as two launches of 2,048 rows (mx.gemm at row offsets).
const HalfRun = struct {
    r: *Run,
    l: weights.Linear,
    q: u64,
    s: u64,
    z: u64,
    y: u64,

    fn go(c: @This()) anyerror!void {
        for ([_]usize{ 0, sf_m / 2 }) |r0| try mx.gemm(c.r.k, c.r.s, c.l, c.q + r0 * sf_k, sf_k, c.s + r0 * (sf_k / 32) * 4, sf_k / 32, sf_m / 2, c.y + r0 * sf_n * 4, sf_n, .f32, false, c.z);
    }
};

/// tf_dsn_combine over all rows: one launch of 4,096 rows, and four of 1,024 (mx.combineRows) in the quarter order of a node.
const CombRun = struct {
    r: *Run,
    sc: mx.ExpertScratch,
    pick: u64,
    y: u64,
    /// 0: one launch; 1: four launches in the order `first`, `last`
    mode: u8,
    first: [2]usize = .{ 0, 0 },
    last: [2]usize = .{ 0, 0 },

    fn go(c: @This()) anyerror!void {
        if (c.mode == 0) return mx.combine(c.r.k, c.r.s, c.sc, c.pick, sf_m, comb_e, sf_n, c.y, sf_n, false);
        for ([_]usize{ c.first[0], c.first[1], c.last[0], c.last[1] }) |row0| try mx.combineRows(c.r.k, c.r.s, c.sc, c.pick, row0, sf_m / 4, comb_e, sf_n, c.y, sf_n, false);
    }
};
const comb_e: usize = 384;
const comb_s: usize = 7;

/// tf_dsn_combine_door over all rows in `qo`'s order (variants: the plain kernel, the early-ring control, the timed twin).
const CombDoor = struct {
    r: *Run,
    f: cuda.Function,
    sc: mx.ExpertScratch,
    pick: u64,
    y: u64,
    qo: mx.QuarterOrder,
    door: mx.Door,
    ts: u64 = 0,

    fn go(c: @This()) anyerror!void {
        var ca: cuda.Args = .{};
        ca.add(c.sc.out);
        ca.add(@as(i64, @intCast(sf_n)));
        ca.add(c.pick);
        ca.add(@as(c_int, @intCast(sf_m)));
        ca.add(@as(c_int, @intCast(comb_s)));
        ca.add(@as(c_int, @intCast(comb_e)));
        ca.add(@as(c_int, @intCast(sf_n)));
        ca.add(c.y);
        ca.add(@as(i64, @intCast(sf_n)));
        ca.add(@as(c_int, 0));
        ca.add(c.qo);
        ca.add(c.door);
        if (c.ts != 0) ca.add(c.ts);
        try cuda.launch.launch(c.f, .{ .grid = .{ .x = @intCast(sf_m * sf_n / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, c.r.s, &ca);
    }
};

/// The doorbell rung from inside the producer (TF_DS_ROCI_SENT_FIRST): wo_b's launch with the sent quarters first
/// (tf_dsn_sf_split128_exact_st_door: the last CTA of the sent quarters rings) and the MoE's routed sum in one launch in the same
/// order (tf_dsn_combine_door), each against the baseline launch (and the combine against the four launches of the quarter order):
/// ms, the ring's place in the launch, bytes for the four nodes, and an observer kernel on a second stream that waits for the
/// doorbell and then reads the sent quarters against the reference: zero words wrong at the ring. Controls: the early-ring kernels
/// (the arrival counted before the CTA's stores) must show words wrong at the ring.
fn rociDoor(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xd007b311);
    const rng = prng.random();
    const d = r.d;
    const sk = r.k.sfk orelse return r.note("roci_door: the block-scaled kernels (native_sf.cu) load on this GPU", false);
    var fns: [6]cuda.Function = undefined;
    for (&fns, [_][:0]const u8{ "tf_dsn_sf_split128_exact_st_door_kernel", "tf_dsn_sf_split128_exact_st_door_early_kernel", "tf_dsn_sf_split128_exact_st_door_t_kernel", "tf_dsn_sf_split128_exact_st_door_b_kernel", "tf_dsn_sf_split128_exact_st_door_b_t_kernel", "tf_dsn_sf_split128_exact_st_door_g_kernel" }) |*f, name| {
        f.* = try sk.module.function(name);
        try f.allowDynamicShared(mx.sf_split128_smem);
    }
    const wf = try r.k.module.function("tf_dsn_door_watch_kernel");
    const cf = try r.k.module.function("tf_dsn_combine_door_kernel");
    const cf_early = try r.k.module.function("tf_dsn_combine_door_early_kernel");
    const cf_time = try r.k.module.function("tf_dsn_combine_door_t_kernel");
    const cf_pt = try r.k.module.function("tf_dsn_combine_door_pt_kernel");
    const cf_g = try r.k.module.function("tf_dsn_combine_door_g_kernel");
    const cf_g_early = try r.k.module.function("tf_dsn_combine_door_g_early_kernel");
    var sw = try cuda.Stream.init(d, true);
    defer sw.deinit();
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var arena_mem: ?*anyopaque = null;
    defer if (arena_mem != null) {
        _ = d.api.cuMemFreeHost(arena_mem);
    };
    defer {
        r.k.wc = .{ 0, 0 };
        r.k.sf = .off;
    }
    var rigd = try DoorRig.init(d);
    defer rigd.deinit();
    const ybytes = sf_m * sf_n * 4;
    const rig = (try sfRig(r, rng, ha, &keep, &arena_mem)) orelse return r.note("roci_door: wo_b's 4,096 rows take tf_dsn_sf_split128 here", false);
    const q = sf_m / 4;
    const qwords: u64 = q * sf_n; // a quarter's 32-bit words
    const ref_dev = try cuda.DeviceBuffer.alloc(d, ybytes);
    const stamps = try cuda.DeviceBuffer.alloc(d, (sf_rt * sf_ct * 2 + 1) * 8);
    try keep.append(ha, ref_dev);
    try keep.append(ha, stamps);
    // ---- wo_b
    const today: TodayRun = .{ .r = r, .l = rig.l, .q = rig.q, .s = rig.s, .z = rig.z, .y = rig.y_ref };
    var today_n = today;
    today_n.y = rig.y_null;
    try poison(rig.y_ref, ybytes, d, r.s);
    try today.go();
    try r.s.synchronize();
    const ref = try get(ref_dev.ptr, ybytes, d);
    defer free(ref);
    var orders: [4]mx.RowOrder = undefined;
    for (0..4) |g| orders[g] = mx.rowOrder(.{ (g ^ 1) * q, (g ^ 3) * q }, .{ g * q, (g ^ 2) * q }, q, sf_ct).?;
    const mk = struct {
        fn run(rr: *Run, f: cuda.Function, a: mx.MxArgs, ro: mx.RowOrder, door: mx.Door, ts: u64) SentRun {
            return .{ .r = rr, .f = f, .a = a, .ro = ro, .door = door, .ts = ts };
        }
    }.run;
    // ms: the baseline, the null, the plain door kernel (a ring each launch: the tickets only need to be distinct per launch for the
    // timing, so one armed Door serves the 60 launches), the timed twin
    const door0 = rigd.arm();
    const ms_t = try kernelBest(r, TodayRun, today, TodayRun.go);
    const ms_n = try kernelBest(r, TodayRun, today_n, TodayRun.go);
    const ms_d = try kernelBest(r, SentRun, mk(r, fns[0], rig.margs, orders[0], door0, 0), SentRun.go);
    const ms_dt = try kernelBest(r, SentRun, mk(r, fns[2], rig.margs, orders[0], door0, stamps.ptr), SentRun.go);
    const ms_db = try kernelBest(r, SentRun, mk(r, fns[3], rig.margs, orders[0], door0, 0), SentRun.go);
    const ms_dg = try kernelBest(r, SentRun, mk(r, fns[5], rig.margs, orders[0], door0, 0), SentRun.go);
    var halves: HalfRun = .{ .r = r, .l = rig.l, .q = rig.q, .s = rig.s, .z = rig.z, .y = rig.y_new };
    const ms_h = try kernelBest(r, HalfRun, halves, HalfRun.go);
    halves.y = rig.y_null;
    const ms_hn = try kernelBest(r, HalfRun, halves, HalfRun.go);
    {
        var th: WcText = .{};
        try th.add("roci_door wo_b as two launches of 2,048 rows (rows 0 .. 2,047 then 2,048 .. 4,095: the sent half then the kept half in time, whatever rows they are), kernel only, best of three, ms: one launch of 4,096 rows {d:.3}; two of 2,048 rows {d:.3} ({d:.1} % over one launch; null control, the pair again, {d:.1} %: {s}); for comparison the sent-first order in one launch with the door {d:.3} ({d:.1} %), with the count released at GPU scope {d:.3} ({d:.1} %)", .{ ms_t, ms_h, pct(ms_h, ms_t), pct(ms_hn, ms_h), resolved(pct(ms_hn, ms_h)), ms_d, pct(ms_d, ms_t), ms_dg, pct(ms_dg, ms_t) });
        r.note(th.buf[0..th.n], @abs(pct(ms_hn, ms_h)) <= null_bar);
    }
    // where the ring falls: 5 timed launches, the ring's stamp past the CTAs'
    var ring_pct: [5]f64 = undefined;
    var ring_b: [5]f64 = undefined;
    var sane = true;
    for (0..5) |rep| {
        try d.check(d.api.cuMemsetD8Async(stamps.ptr, 0, (sf_rt * sf_ct * 2 + 1) * 8, r.s.handle), "cuMemsetD8Async");
        const dr = rigd.arm();
        try mk(r, fns[4], rig.margs, orders[0], dr, stamps.ptr).go();
        try r.s.synchronize();
        {
            const raw = try get(stamps.ptr, (sf_rt * sf_ct * 2 + 1) * 8, d);
            defer free(raw);
            const ts = u64s(raw);
            ring_b[rep] = stampSpan(ts, sf_rt * sf_ct, ts[sf_rt * sf_ct * 2]) orelse 0;
            sane = sane and ring_b[rep] != 0 and try rigd.rang();
        }
        try d.check(d.api.cuMemsetD8Async(stamps.ptr, 0, (sf_rt * sf_ct * 2 + 1) * 8, r.s.handle), "cuMemsetD8Async");
        const dr2 = rigd.arm();
        try mk(r, fns[2], rig.margs, orders[0], dr2, stamps.ptr).go();
        try r.s.synchronize();
        const raw = try get(stamps.ptr, (sf_rt * sf_ct * 2 + 1) * 8, d);
        defer free(raw);
        const ts = u64s(raw);
        const v = stampSpan(ts, sf_rt * sf_ct, ts[sf_rt * sf_ct * 2]);
        sane = sane and v != null and try rigd.rang();
        ring_pct[rep] = v orelse 0;
    }
    std.mem.sort(f64, &ring_pct, {}, std.sort.asc(f64));
    std.mem.sort(f64, &ring_b, {}, std.sort.asc(f64));
    const null_pct = pct(ms_n, ms_t);
    var t: WcText = .{};
    try t.add("the ring falls at {d:.1} % of the launch (per-thread release, median of 5, {d:.1} to {d:.1}), {d:.1} % with the block-level release ({d:.1} to {d:.1}); the doorbell showed the ticket and the count was zero after each", .{ ring_pct[2], ring_pct[0], ring_pct[4], ring_b[2], ring_b[0], ring_b[4] });
    r.note(t.buf[0..t.n], @abs(null_pct) <= null_bar and sane);
    // bytes and the observer, the four nodes
    var bad: usize = 0;
    var wrong: u32 = 0;
    var unseen: usize = 0;
    var unclean: usize = 0;
    for (0..4) |g| {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const ro = mk(r, fns[0], rig.margs, orders[g], dr, 0);
        const rows: [2]usize = .{ (g ^ 1) * q, (g ^ 3) * q };
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + rows[0] * sf_n * 4, ref_dev.ptr + rows[1] * sf_n * 4 }, .{ rig.y_new + rows[0] * sf_n * 4, rig.y_new + rows[1] * sf_n * 4 }, qwords, SentRun, ro);
        wrong += o.wrong;
        unseen += @intFromBool(!o.seen);
        unclean += @intFromBool(!(try rigd.rang()));
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(ref, got);
    }
    var line: [512]u8 = undefined;
    r.report("roci_door wo_b: words of the two sent quarters that differed from the reference when the observer saw the doorbell, four nodes", wrong, true);
    // the block-level release and the GPU-scope release, the same four nodes
    for ([_]usize{ 3, 5 }, [_][]const u8{ "the block-level release (thread 0 fences once per CTA)", "the count released at GPU scope and the system fence by the last arrival alone" }) |fi, what| {
        var badb: usize = 0;
        var wrongb: u32 = 0;
        var offb: usize = 0;
        for (0..4) |g| {
            const dr = rigd.arm();
            try poison(rig.y_new, ybytes, d, r.s);
            const rows: [2]usize = .{ (g ^ 1) * q, (g ^ 3) * q };
            const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + rows[0] * sf_n * 4, ref_dev.ptr + rows[1] * sf_n * 4 }, .{ rig.y_new + rows[0] * sf_n * 4, rig.y_new + rows[1] * sf_n * 4 }, qwords, SentRun, mk(r, fns[fi], rig.margs, orders[g], dr, 0));
            wrongb += o.wrong;
            offb += @intFromBool(!o.seen) + @intFromBool(!(try rigd.rang()));
            const got = try get(rig.y_new, ybytes, d);
            defer free(got);
            badb += differs(ref, got);
        }
    }
    // control: the arrival counted before the CTA's stores
    var early_wrong: u32 = 0;
    {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + q * sf_n * 4, ref_dev.ptr + 3 * q * sf_n * 4 }, .{ rig.y_new + q * sf_n * 4, rig.y_new + 3 * q * sf_n * 4 }, qwords, SentRun, mk(r, fns[1], rig.margs, orders[0], dr, 0));
        early_wrong = o.wrong;
        if (!o.seen) early_wrong = 0;
    }
    r.report("roci_door wo_b control: the early-ring kernel (the arrival counted before the CTA's stores), words of the sent quarters that differed when the observer saw the doorbell", early_wrong, false);
    // ---- the routed sum
    const pk = try ha.alloc(i32, sf_m * comb_s);
    for (0..sf_m) |row| {
        var seen: [comb_s]i32 = undefined;
        var n: usize = 0;
        while (n < comb_s) {
            const e: i32 = @intCast(rng.uintLessThan(usize, comb_e));
            if (std.mem.indexOfScalar(i32, seen[0..n], e) != null) continue;
            seen[n] = e;
            pk[row * comb_s + n] = e;
            n += 1;
        }
    }
    // the entries' outputs: a value that differs by entry and column, so a row put in the wrong place shows
    const ev = try ha.alloc(f32, sf_m * comb_s * sf_n);
    for (ev, 0..) |*x, i| x.* = @as(f32, @floatFromInt((i *% 2654435761) >> 12 & 0xffff)) / 256.0;
    const pick = try upload(d, std.mem.sliceAsBytes(pk));
    const outs = try upload(d, std.mem.sliceAsBytes(ev));
    const cstamps = try cuda.DeviceBuffer.alloc(d, (sf_m * sf_n / 256 * 2 + 1) * 8);
    try keep.append(ha, pick);
    try keep.append(ha, outs);
    try keep.append(ha, cstamps);
    var csc = std.mem.zeroes(mx.ExpertScratch);
    csc.out = outs.ptr;
    csc.slots = comb_s;
    const one: CombRun = .{ .r = r, .sc = csc, .pick = pick.ptr, .y = rig.y_ref, .mode = 0 };
    var one_n = one;
    one_n.y = rig.y_null;
    try poison(rig.y_ref, ybytes, d, r.s);
    try one.go();
    try r.s.synchronize();
    const cref = try get(ref_dev.ptr, ybytes, d);
    defer free(cref);
    var qos: [4]mx.QuarterOrder = undefined;
    for (0..4) |g| qos[g] = mx.quarterOrder(.{ (g ^ 1) * q, (g ^ 3) * q }, .{ g * q, (g ^ 2) * q }, q, sf_n).?;
    const four: CombRun = .{ .r = r, .sc = csc, .pick = pick.ptr, .y = rig.y_new, .mode = 1, .first = .{ q, 3 * q }, .last = .{ 0, 2 * q } };
    const cdoor0 = rigd.arm();
    const cd = struct {
        fn make(rr: *Run, f: cuda.Function, sc_: mx.ExpertScratch, pk_: u64, y: u64, qo: mx.QuarterOrder, door: mx.Door, ts: u64) CombDoor {
            return .{ .r = rr, .f = f, .sc = sc_, .pick = pk_, .y = y, .qo = qo, .door = door, .ts = ts };
        }
    }.make;
    const cms_1 = try kernelBest(r, CombRun, one, CombRun.go);
    const cms_n = try kernelBest(r, CombRun, one_n, CombRun.go);
    const cms_4 = try kernelBest(r, CombRun, four, CombRun.go);
    const cms_d = try kernelBest(r, CombDoor, cd(r, cf, csc, pick.ptr, rig.y_new, qos[0], cdoor0, 0), CombDoor.go);
    const cms_pt = try kernelBest(r, CombDoor, cd(r, cf_pt, csc, pick.ptr, rig.y_new, qos[0], cdoor0, 0), CombDoor.go);
    const cms_g = try kernelBest(r, CombDoor, cd(r, cf_g, csc, pick.ptr, rig.y_new, qos[0], cdoor0, 0), CombDoor.go);
    const cms_t = try kernelBest(r, CombDoor, cd(r, cf_time, csc, pick.ptr, rig.y_new, qos[0], cdoor0, cstamps.ptr), CombDoor.go);
    const ncta = sf_m * sf_n / 256;
    var cring: [5]f64 = undefined;
    var csane = true;
    for (0..5) |rep| {
        try d.check(d.api.cuMemsetD8Async(cstamps.ptr, 0, (ncta * 2 + 1) * 8, r.s.handle), "cuMemsetD8Async");
        const dr = rigd.arm();
        try cd(r, cf_time, csc, pick.ptr, rig.y_new, qos[0], dr, cstamps.ptr).go();
        try r.s.synchronize();
        const raw = try get(cstamps.ptr, (ncta * 2 + 1) * 8, d);
        defer free(raw);
        const ts = u64s(raw);
        const v = stampSpan(ts, ncta, ts[ncta * 2]);
        csane = csane and v != null and try rigd.rang();
        cring[rep] = v orelse 0;
    }
    std.mem.sort(f64, &cring, {}, std.sort.asc(f64));
    const cnull = pct(cms_n, cms_1);
    var t3: WcText = .{};
    try t3.add("the door kernel (one launch, sent quarters first, the ring from inside, the block-level release) {d:.3} ({d:.1} % over one launch, {d:.1} % over the four); with every thread fencing its own stores {d:.3} ({d:.1} % over one launch); the GPU-scope release with the last arrival's system fence {d:.3} ({d:.1} % over one launch); the block-level one's timed twin {d:.3}; ", .{ cms_d, pct(cms_d, cms_1), pct(cms_d, cms_4), cms_pt, pct(cms_pt, cms_1), cms_g, pct(cms_g, cms_1), cms_t });
    try t3.add("the ring falls at {d:.1} % of the launch (median of 5, {d:.1} to {d:.1}); the doorbell showed the ticket and the count was zero after each of the 5", .{ cring[2], cring[0], cring[4] });
    r.note(t3.buf[0..t3.n], @abs(cnull) <= null_bar and csane);
    bad = 0;
    wrong = 0;
    unseen = 0;
    unclean = 0;
    for (0..4) |g| {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const rows: [2]usize = .{ (g ^ 1) * q, (g ^ 3) * q };
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + rows[0] * sf_n * 4, ref_dev.ptr + rows[1] * sf_n * 4 }, .{ rig.y_new + rows[0] * sf_n * 4, rig.y_new + rows[1] * sf_n * 4 }, qwords, CombDoor, cd(r, cf, csc, pick.ptr, rig.y_new, qos[g], dr, 0));
        wrong += o.wrong;
        unseen += @intFromBool(!o.seen);
        unclean += @intFromBool(!(try rigd.rang()));
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(cref, got);
    }
    r.report(try std.fmt.bufPrint(&line, "roci_door combine: the door kernel of each of the four nodes against the one launch (the doorbell showed {d} of 4 times, the count clean {d} of 4)", .{ 4 - unseen, 4 - unclean }), bad + unseen + unclean, true);
    r.report("roci_door combine: words of the two sent quarters that differed from the reference when the observer saw the doorbell, four nodes", wrong, true);
    early_wrong = 0;
    {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + q * sf_n * 4, ref_dev.ptr + 3 * q * sf_n * 4 }, .{ rig.y_new + q * sf_n * 4, rig.y_new + 3 * q * sf_n * 4 }, qwords, CombDoor, cd(r, cf_early, csc, pick.ptr, rig.y_new, qos[0], dr, 0));
        early_wrong = if (o.seen) o.wrong else 0;
    }
    r.report("roci_door combine control: the early-ring kernel (the arrival counted before the CTA's stores), words of the sent quarters that differed when the observer saw the doorbell", early_wrong, false);
    // the engine's kernel: the count released at GPU scope, the system fence by the last arrival alone
    bad = 0;
    wrong = 0;
    unseen = 0;
    unclean = 0;
    for (0..4) |g| {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const rows: [2]usize = .{ (g ^ 1) * q, (g ^ 3) * q };
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + rows[0] * sf_n * 4, ref_dev.ptr + rows[1] * sf_n * 4 }, .{ rig.y_new + rows[0] * sf_n * 4, rig.y_new + rows[1] * sf_n * 4 }, qwords, CombDoor, cd(r, cf_g, csc, pick.ptr, rig.y_new, qos[g], dr, 0));
        wrong += o.wrong;
        unseen += @intFromBool(!o.seen);
        unclean += @intFromBool(!(try rigd.rang()));
        const got = try get(rig.y_new, ybytes, d);
        defer free(got);
        bad += differs(cref, got);
    }
    r.report("roci_door combine with the count released at GPU scope and the system fence by the last arrival alone (the engine's kernel): output bytes against the one launch, words wrong at the ring, and doorbells not seen or counts not clean, summed over the four nodes", bad + wrong + unseen + unclean, true);
    {
        const dr = rigd.arm();
        try poison(rig.y_new, ybytes, d, r.s);
        const o = try observed(r, &rigd, wf, sw, .{ ref_dev.ptr + q * sf_n * 4, ref_dev.ptr + 3 * q * sf_n * 4 }, .{ rig.y_new + q * sf_n * 4, rig.y_new + 3 * q * sf_n * 4 }, qwords, CombDoor, cd(r, cf_g_early, csc, pick.ptr, rig.y_new, qos[0], dr, 0));
        early_wrong = if (o.seen) o.wrong else 0;
    }
    r.report("roci_door combine control for the GPU-scope release: its early-ring kernel (the arrival counted before the CTA's stores, delayed stores), words of the sent quarters that differed when the observer saw the doorbell", early_wrong, false);
}

// -- TF_DS_ROCI_GATHERS' boundary staging -----------------------------------------------------------------------------------------

/// TF_DS_ROCI_GATHERS' gather of the boundary layer's cache rows on four nodes in one process: each node's rows of the four cache
/// arrays (device memory) are staged by prompt.boundaryOut into its slab of the arena's boundary region (cudaMalloc, the write-combined
/// host memory of the arena in the baseline, or the locked anonymous pages registered with the driver: TF_DS_ROCI_PAGES=mmap), each slab is
/// written to the same place in the other three regions (the NIC's WRITEs, as device-to-device copies), and prompt.boundaryIn copies the
/// three that landed out to their rows: every array of every node must hold the bytes the in-place gather leaves, the groups around
/// the chunk's untouched. The ms are the 16 copies of one gather (4 out and 12 in) at the real sizes (1,024 rows an owner). A node that
/// stages nothing must leave its rows missing.
fn rociBoundary(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xb0da);
    const rng = prng.random();
    const d = r.d;
    var line: [400]u8 = undefined;
    const world = 4;
    const q: usize = 1024;
    const start: usize = 8192; // the chunk's first group
    const hd: usize = 512;
    const id: usize = 128;
    const rows = [4]usize{ hd / 2, hd / 16, id / 2, id / 32 };
    const total_rows = start + world * q + 16;
    const slab = q * dsv41.rocinante.gb_row;
    var pool: WcPool = .{ .d = d };
    defer pool.deinit();
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    // the whole update's bytes of each array: the rows of the chunk
    var whole: [4][]u8 = undefined;
    for (&whole, rows) |*w, row| {
        w.* = try ha.alloc(u8, total_rows * row);
        rng.bytes(w.*);
    }
    const kinds = [_]Where{ .device, .arena, .mmapreg };
    const names = [_][]const u8{ "cudaMalloc", "write-combined host memory", "locked anonymous pages (cuMemHostRegister)" };
    var ms_out: [3]f64 = .{ 0, 0, 0 };
    for (kinds, names, 0..) |kind, nm, ki| {
        var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
        defer for (keep.items) |*b| b.free();
        var arrays: [world][4]u64 = undefined;
        var parts: [world][4]dsv41.comm.InPlace = undefined;
        var regions: [world]u64 = undefined;
        var ok = true;
        for (0..world) |k| {
            for (0..4) |a| {
                const b = try cuda.DeviceBuffer.alloc(d, total_rows * rows[a]);
                try keep.append(ha, b);
                try d.check(d.api.cuMemsetD8Async(b.ptr, 0xee, b.len, r.s.handle), "cuMemsetD8Async");
                // this node's own groups hold the update's bytes
                const lo = (start + k * q) * rows[a];
                arrays[k][a] = b.ptr;
            }
            var cs: dsv41.prompt.Caches = undefined;
            cs.comp_codes[20] = arrays[k][0];
            cs.comp_scales[20] = arrays[k][1];
            cs.idx_codes[20] = arrays[k][2];
            cs.idx_scales[20] = arrays[k][3];
            parts[k] = dsv41.prompt.boundaryGathers(&cs, 20, start, .{ .start = k * q, .n = q }, hd, id);
            if (kind == .device) {
                const b = try cuda.DeviceBuffer.alloc(d, world * slab);
                try keep.append(ha, b);
                regions[k] = b.ptr;
            } else regions[k] = (pool.alloc(kind, world * slab) catch |err| {
                r.note(try std.fmt.bufPrint(&line, "roci_boundary {s}: could not allocate the region ({s})", .{ nm, @errorName(err) }), false);
                ok = false;
                break;
            }).dev;
        }
        if (!ok) continue;
        try r.s.synchronize();
        // (the staging out, the wire, the staging in) as many times as the timing wants; the first run is the checked one
        var t_total: u64 = 0;
        for (0..11) |rep| {
            const t0 = nowNs();
            for (0..world) |k| for (dsv41.prompt.boundaryOut(&parts[k], regions[k], slab, k)) |cp| try d.check(d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, r.s.handle), "cuMemcpyDtoDAsync");
            for (0..world) |k| for (0..world) |dd| {
                if (dd == k) continue;
                const at = dsv41.prompt.boundarySlab(0, slab, k);
                try d.check(d.api.cuMemcpyDtoDAsync_v2(regions[dd] + at, regions[k] + at, slab, r.s.handle), "cuMemcpyDtoDAsync");
            };
            for (0..world) |dd| for (0..world) |k| {
                if (dd == k) continue;
                for (dsv41.prompt.boundaryIn(&parts[dd], regions[dd], slab, k)) |cp| try d.check(d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, r.s.handle), "cuMemcpyDtoDAsync");
            };
            try r.s.synchronize();
            if (rep > 0) t_total += nowNs() - t0;
            if (rep == 0) {
                var bad: usize = 0;
                for (0..world) |k| for (0..4) |a| {
                    const got = try get(arrays[k][a], total_rows * rows[a], d);
                    defer free(got);
                    const lo = start * rows[a];
                    const hi = (start + world * q) * rows[a];
                    bad += differs(got[lo..hi], whole[a][lo..hi]);
                    for (got[0..lo]) |b| bad += @intFromBool(b != 0xee);
                    for (got[hi..]) |b| bad += @intFromBool(b != 0xee);
                };
                r.report(try std.fmt.bufPrint(&line, "roci_boundary {s}: four nodes' cache arrays after the staged gather against the in-place gather's layout (rows around the chunk untouched)", .{nm}), bad, true);
            }
        }
        ms_out[ki] = @as(f64, @floatFromInt(t_total)) / 10.0 / 1e6;
        if (kind == .mmapreg) {
            // the control: node 2 stages nothing, so its rows never reach the other three (the regions hold the checked runs' bytes: poisoned first)
            for (regions) |rg| try poison(rg, world * slab, d, r.s);
            for (0..world) |k| for (0..4) |a| {
                const lo = start * rows[a];
                try d.check(d.api.cuMemsetD8Async(arrays[k][a] + lo, 0xee, world * q * rows[a], r.s.handle), "cuMemsetD8Async");
                const own = (start + k * q) * rows[a];
            };
            try r.s.synchronize();
            for (0..world) |k| {
                if (k == 2) continue;
                for (dsv41.prompt.boundaryOut(&parts[k], regions[k], slab, k)) |cp| try d.check(d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, r.s.handle), "cuMemcpyDtoDAsync");
            }
            for (0..world) |k| for (0..world) |dd| {
                if (dd == k) continue;
                const at = dsv41.prompt.boundarySlab(0, slab, k);
                try d.check(d.api.cuMemcpyDtoDAsync_v2(regions[dd] + at, regions[k] + at, slab, r.s.handle), "cuMemcpyDtoDAsync");
            };
            for (0..world) |dd| for (0..world) |k| {
                if (dd == k) continue;
                for (dsv41.prompt.boundaryIn(&parts[dd], regions[dd], slab, k)) |cp| try d.check(d.api.cuMemcpyDtoDAsync_v2(cp.dst, cp.src, cp.bytes, r.s.handle), "cuMemcpyDtoDAsync");
            };
            try r.s.synchronize();
            var bad: usize = 0;
            for (0..world) |k| for (0..4) |a| {
                const got = try get(arrays[k][a], total_rows * rows[a], d);
                defer free(got);
                const lo = start * rows[a];
                const hi = (start + world * q) * rows[a];
                bad += differs(got[lo..hi], whole[a][lo..hi]);
            };
            r.report("roci_boundary control: node 2 stages nothing, the arrays of the four nodes", bad, false);
        }
    }
    r.note(try std.fmt.bufPrint(&line, "roci_boundary: ms for one gather's copies (4 out, 12 in) and its 12 slab writes at 1,024 rows an owner: cudaMalloc {d:.3}, write-combined {d:.3}, locked pages {d:.3}; a pass makes one", .{ ms_out[0], ms_out[1], ms_out[2] }), true);
}

// -- TF_DS_ROCI_CODES -------------------------------------------------------------------------------------------------------------

/// The inputs of TF_DS_ROCI_CODES' two exchanges for two column partners (nodes 1 and 3: pairs 0 and 1): wo_b's part of each (u, bf16
/// [4,096, 2,048]), the experts' pack of each (xd, bf16 [28,672, 640]), the picks and the owner of each expert's rest.
const CodesIn = struct { u: [2]u64, xd: [2]u64, pick: u64, owner: u64 };

const codes_n: usize = 4096;
const codes_uw: usize = 2048;
const codes_slots: usize = 7;
const codes_experts: usize = 384;

/// What the two nodes' consumers hold after the exchange: wo_b's input (codes [4,096, 4,096], scales [4,096, 128] fp32) and down's
/// (hq [28,672, 1,152], hs [28,672, 36] fp32), each node's, read back.
const CodesOut = struct { wq: [2][]u8, ws: [2][]u8, hq: [2][]u8, hs: [2][]u8 };

/// Both exchanges as the engine runs them with TF_DS_ROCI_CODES, the wire (the NIC's WRITE of a send region into the partner's land
/// region) a device-to-device copy, the send and land regions in `kind` memory (device memory: the reference). `skip_pack_wire`: the
/// pack's land regions never written (the control). ms: the producer's launches, the wo_b copies and the assembly, each alone.
fn codesRun(r: *Run, pool: *WcPool, kind: Where, in: CodesIn, skip_pack_wire: bool, ms: ?*[3]f32) !CodesOut {
    const d = r.d;
    const k = r.k;
    const ops = r.ops.?;
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (keep.items) |*b| b.free();
        keep.deinit(std.heap.page_allocator);
    }
    const wob_bytes = dsv41.prompt2d.catCodesBytes(codes_n, codes_uw);
    const pc = try dsv41.prompt2d.packCodesPlan(codes_n * codes_slots, 512, 128);
    var out: CodesOut = undefined;
    var wob_send: [2]u64 = undefined;
    var wob_land: [2]u64 = undefined;
    var pk_send: [2]u64 = undefined;
    var pk_land: [2]u64 = undefined;
    var wq: [2]u64 = undefined;
    var ws: [2]u64 = undefined;
    var hq: [2]u64 = undefined;
    var hs: [2]u64 = undefined;
    const region = struct {
        fn at(rr: *Run, pl: *WcPool, kp: *std.ArrayList(cuda.DeviceBuffer), kd: Where, bytes: usize) !u64 {
            if (kd == .device) {
                const b = try cuda.DeviceBuffer.alloc(rr.d, bytes);
                try kp.append(std.heap.page_allocator, b);
                return b.ptr;
            }
            return (try pl.alloc(kd, bytes)).dev;
        }
        fn dev(rr: *Run, kp: *std.ArrayList(cuda.DeviceBuffer), bytes: usize) !u64 {
            const b = try cuda.DeviceBuffer.alloc(rr.d, bytes);
            try kp.append(std.heap.page_allocator, b);
            return b.ptr;
        }
    };
    for (0..2) |i| {
        wob_send[i] = try region.at(r, pool, &keep, kind, wob_bytes);
        wob_land[i] = try region.at(r, pool, &keep, kind, wob_bytes);
        pk_send[i] = try region.at(r, pool, &keep, kind, pc.bytes);
        pk_land[i] = try region.at(r, pool, &keep, kind, pc.bytes);
        wq[i] = try region.dev(r, &keep, codes_n * 2 * codes_uw);
        ws[i] = try region.dev(r, &keep, codes_n * (2 * codes_uw / 32) * 4);
        hq[i] = try region.dev(r, &keep, codes_n * codes_slots * (2 * 512 + 128));
        hs[i] = try region.dev(r, &keep, codes_n * codes_slots * ((2 * 512 + 128) / 32) * 4);
        try poison(wob_land[i], wob_bytes, d, r.s);
        try poison(pk_land[i], pc.bytes, d, r.s);
        try poison(wq[i], codes_n * 2 * codes_uw, d, r.s);
        try poison(hq[i], codes_n * codes_slots * (2 * 512 + 128), d, r.s);
    }
    const plans = [2]dsv41.prompt2d.CatCodes{
        dsv41.prompt2d.catCodesPlan(1, .{ 0, 1, 2, 3 }, codes_n, .{ codes_uw, codes_uw }, in.u[0], wob_send[0], wob_land[0], wq[0], ws[0]),
        dsv41.prompt2d.catCodesPlan(3, .{ 0, 1, 2, 3 }, codes_n, .{ codes_uw, codes_uw }, in.u[1], wob_send[1], wob_land[1], wq[1], ws[1]),
    };
    for (plans) |pl| try dsv41.prompt2d.catCodesQuant(k, r.s, pl);
    for (0..2) |i| try dsv41.prompt2d.packQuant(k, r.s, in.xd[i], pk_send[i], pc);
    // the wire
    for (0..2) |i| {
        try d.check(d.api.cuMemcpyDtoDAsync_v2(wob_land[1 - i], wob_send[i], wob_bytes, r.s.handle), "cuMemcpyDtoDAsync");
        if (!skip_pack_wire) try d.check(d.api.cuMemcpyDtoDAsync_v2(pk_land[1 - i], pk_send[i], pc.bytes, r.s.handle), "cuMemcpyDtoDAsync");
    }
    // the consumers
    for (plans) |pl| for (pl.copies) |cp| try ops.copyRows(r.s, cp.src, cp.src_stride, cp.dst, cp.dst_stride, cp.width, cp.rows);
    for (0..2) |i| {
        var sc: mx.ExpertScratch = undefined;
        sc.hq = hq[i];
        sc.hs = hs[i];
        const part0 = if (i == 0) pk_send[0] else pk_land[1];
        const part1 = if (i == 1) pk_send[1] else pk_land[0];
        try dsv41.prompt2d.packAssemble(ops, r.s, part0, part1, in.pick, in.owner, sc, pc);
    }
    try r.s.synchronize();
    if (ms) |m| {
        // each stage alone, kernel only (the data stays as the checked run left it)
        const Q = struct {
            r: *Run,
            pl: dsv41.prompt2d.CatCodes,
            pc: dsv41.prompt2d.PackCodes,
            xd: u64,
            send: u64,
            fn quant(c: @This()) anyerror!void {
                try dsv41.prompt2d.catCodesQuant(c.r.k, c.r.s, c.pl);
                try dsv41.prompt2d.packQuant(c.r.k, c.r.s, c.xd, c.send, c.pc);
            }
            fn copies(c: @This()) anyerror!void {
                for (c.pl.copies) |cp| try c.r.ops.?.copyRows(c.r.s, cp.src, cp.src_stride, cp.dst, cp.dst_stride, cp.width, cp.rows);
            }
        };
        const q: Q = .{ .r = r, .pl = plans[0], .pc = pc, .xd = in.xd[0], .send = pk_send[0] };
        m[0] = try wcTime(r, Q, q, Q.quant);
        m[1] = try wcTime(r, Q, q, Q.copies);
        const A = struct {
            r: *Run,
            p0: u64,
            p1: u64,
            pick: u64,
            owner: u64,
            sc: mx.ExpertScratch,
            pc: dsv41.prompt2d.PackCodes,
            fn go(c: @This()) anyerror!void {
                try dsv41.prompt2d.packAssemble(c.r.ops.?, c.r.s, c.p0, c.p1, c.pick, c.owner, c.sc, c.pc);
            }
        };
        var sc: mx.ExpertScratch = undefined;
        sc.hq = hq[0];
        sc.hs = hs[0];
        m[2] = try wcTime(r, A, .{ .r = r, .p0 = pk_send[0], .p1 = pk_land[0], .pick = in.pick, .owner = in.owner, .sc = sc, .pc = pc }, A.go);
    }
    for (0..2) |i| {
        out.wq[i] = try get(wq[i], codes_n * 2 * codes_uw, d);
        out.ws[i] = try get(ws[i], codes_n * (2 * codes_uw / 32) * 4, d);
        out.hq[i] = try get(hq[i], codes_n * codes_slots * (2 * 512 + 128), d);
        out.hs[i] = try get(hs[i], codes_n * codes_slots * ((2 * 512 + 128) / 32) * 4, d);
    }
    return out;
}

fn codesFree(o: CodesOut) void {
    for (0..2) |i| {
        free(o.wq[i]);
        free(o.ws[i]);
        free(o.hq[i]);
        free(o.hs[i]);
    }
}

fn codesDiffer(a: CodesOut, b: CodesOut) [2]usize {
    var wob: usize = 0;
    var pack: usize = 0;
    for (0..2) |i| {
        wob += differs(a.wq[i], b.wq[i]) + differs(a.ws[i], b.ws[i]);
        pack += differs(a.hq[i], b.hq[i]) + differs(a.hs[i], b.hs[i]);
    }
    return .{ wob, pack };
}

/// TF_DS_ROCI_CODES' two exchanges on two column partners in one process: wo_b's part quantized into the send region of each page type
/// (device memory, write-combined host memory, locked registered pages), written to the partner's land region, read by the four copies
/// that assemble wo_b's input; the experts' pack quantized into its send region, written to the land region, read by the assembly into
/// down's input. Every consumer byte must equal the run in device memory throughout. A pack that never reaches the land region must
/// leave the assembly's bytes different. ms a launch of the quantizers, the four copies and the assembly in each page type.
fn rociCodes(r: *Run, rng_in: std.Random) !void {
    _ = rng_in;
    var prng = std.Random.DefaultPrng.init(0xc0de5);
    const rng = prng.random();
    const d = r.d;
    var line: [512]u8 = undefined;
    if (r.ops == null) {
        r.note("roci_codes: the ops are available", false);
        return;
    }
    var pool: WcPool = .{ .d = d };
    defer pool.deinit();
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (keep.items) |*b| b.free();
    var in: CodesIn = undefined;
    for (0..2) |i| {
        const ub = try upload(d, try bf16s(codes_n * codes_uw, rng, 1.0, ha));
        try keep.append(ha, ub);
        in.u[i] = ub.ptr;
        const xb = try upload(d, try bf16s(codes_n * codes_slots * 640, rng, 1.0, ha));
        try keep.append(ha, xb);
        in.xd[i] = xb.ptr;
    }
    const picks = try ha.alloc(i32, codes_n * codes_slots);
    for (picks) |*p| p.* = @intCast(rng.uintLessThan(usize, codes_experts));
    const owner = try ha.alloc(i32, codes_experts + 1);
    for (owner) |*o| o.* = @intFromBool(rng.boolean());
    owner[codes_experts] = 0;
    const pb = try upload(d, std.mem.sliceAsBytes(picks));
    try keep.append(ha, pb);
    const ob = try upload(d, std.mem.sliceAsBytes(owner));
    try keep.append(ha, ob);
    in.pick = pb.ptr;
    in.owner = ob.ptr;
    var ms: [3][3]f32 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
    const ref = try codesRun(r, &pool, .device, in, false, &ms[0]);
    defer codesFree(ref);
    pool.freeAll();
    const kinds = [_]Where{ .arena, .mmapreg };
    const names = [_][]const u8{ "write-combined host memory", "locked anonymous pages (cuMemHostRegister)" };
    for (kinds, names, 1..) |kind, nm, ki| {
        const got = codesRun(r, &pool, kind, in, false, &ms[ki]) catch |err| {
            r.note(try std.fmt.bufPrint(&line, "roci_codes {s}: could not run ({s})", .{ nm, @errorName(err) }), false);
            r.note(try std.fmt.bufPrint(&line, "roci_codes {s}: could not run ({s})", .{ nm, @errorName(err) }), false);
            pool.freeAll();
            continue;
        };
        defer codesFree(got);
        const df = codesDiffer(ref, got);
        r.report(try std.fmt.bufPrint(&line, "roci_codes {s}: wo_b's input (codes and scales) on both partners, the send and land regions in this memory, against device memory", .{nm}), df[0], true);
        r.report(try std.fmt.bufPrint(&line, "roci_codes {s}: down's input (hq and hs) on both partners, the pack's regions in this memory, against device memory", .{nm}), df[1], true);
        pool.freeAll();
    }
    // the control: the pack never written into the land regions
    const ctl = try codesRun(r, &pool, .mmapreg, in, true, null);
    defer codesFree(ctl);
    pool.freeAll();
    r.report("roci_codes control: the pack's land regions never written, down's input on both partners", codesDiffer(ref, ctl)[1], false);
    r.note(try std.fmt.bufPrint(&line, "roci_codes: ms a pass-launch (one of the 20 layers' each) cudaMalloc / write-combined / locked pages: the quantizers of wo_b's part and of the pack {d:.3} / {d:.3} / {d:.3}, wo_b's four copies from the land region {d:.3} / {d:.3} / {d:.3}, the pack's assembly {d:.3} / {d:.3} / {d:.3}", .{ ms[0][0], ms[1][0], ms[2][0], ms[0][1], ms[1][1], ms[2][1], ms[0][2], ms[1][2], ms[2][2] }), true);
}

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

// -- TF_DS_QGATE_CANARY 2 / 3 ---------------------------------------------------------------------------------------

/// The body canary's values: every 32-value group normal values at its own power of two (2^-30 .. 2^30), every 97th
/// group zeros, a negative zero in every 89th, a value 5,000 times its group's magnitude in every 83rd.
fn canaryValues(n: usize, rng: std.Random, a: std.mem.Allocator) ![]f32 {
    const v = try a.alloc(f32, n);
    for (0..n / 32) |g| {
        const sc = std.math.ldexp(@as(f32, 1.0), rng.intRangeAtMost(i32, -30, 30));
        const grp = v[g * 32 ..][0..32];
        for (grp) |*x| x.* = rng.floatNorm(f32) * sc;
        if (g % 97 == 5) @memset(grp, 0);
        if (g % 89 == 7) grp[3] = -0.0;
        if (g % 83 == 11) grp[9] = 5000 * sc;
    }
    return v;
}

fn qgateBody(r: *Run, rng: std.Random) !void {
    const d = r.d;
    var sarena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer sarena.deinit();
    const ha = sarena.allocator();
    const shapes = [_][2]usize{ .{ 1, 2560 }, .{ 3, 5120 }, .{ 17, 2560 }, .{ 48, 2560 }, .{ 2048, 5120 } };
    var diff2: usize = 0;
    var moved: usize = 0;
    var diff3: usize = 0;
    for (shapes) |sh| {
        const rows = sh[0];
        const w = sh[1];
        const n = rows * w;
        const x = try canaryValues(n, rng, ha);
        // (the copies through replay_io: in the stream's order)
        var xb = try cuda.DeviceBuffer.alloc(d, n * 4);
        defer xb.free();
        try rio.write(xb, 0, std.mem.sliceAsBytes(x));
        var fb = try cuda.DeviceBuffer.alloc(d, n * 2);
        defer fb.free();
        // mode 2: act_quant's own quant-dequant of the same rows (bf16), then the canary in place
        try poison(fb.ptr, fb.len, d, r.s);
        try mx.fake(r.k, r.s, xb.ptr, .f32, w, rows, w, fb.ptr, w);
        try mx.qgateBody(r.k, r.s, xb.ptr, n, 2);
        try r.s.synchronize();
        const fake = try get(fb.ptr, n * 2, d);
        defer free(fake);
        const got2 = try get(xb.ptr, n * 4, d);
        defer free(got2);
        const want2 = try ha.alloc(u32, n);
        for (want2, 0..) |*o, i| o.* = @as(u32, std.mem.readInt(u16, fake[i * 2 ..][0..2], .little)) << 16;
        diff2 += differs(std.mem.sliceAsBytes(want2), got2);
        moved += differs(std.mem.sliceAsBytes(x), got2);
        // mode 3: the specials first, then each value's bits plus one
        const special = [_]u32{ 0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x7f7fffff, 0xff7fffff, 0x7f7ffffe, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x3f800000, 0xbf800000 };
        const bits = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(x));
        @memcpy(bits[0..special.len], &special);
        try rio.write(xb, 0, std.mem.sliceAsBytes(x));
        try mx.qgateBody(r.k, r.s, xb.ptr, n, 3);
        try r.s.synchronize();
        const got3 = try get(xb.ptr, n * 4, d);
        defer free(got3);
        const want3 = try ha.alloc(u32, n);
        for (want3, bits) |*o, u| {
            const mag = u & 0x7fffffff;
            o.* = if (mag != 0 and mag < 0x7f7fffff) u + 1 else u;
        }
        diff3 += differs(std.mem.sliceAsBytes(want3), got3);
    }
    r.report("qgate_body 2: layer 0's partials (1 .. 2,048 rows of 2,560 and 5,120 columns) one FP8 step in place against act_quant's quant-dequant of the same values (mx.fake, widened)", diff2, true);
    r.report("qgate_body 2 control: the partials against their FP8 step", moved, false);
    r.report("qgate_body 3: each nonzero finite value's bits plus one, zeros, infinities, NaN and the largest finite values kept", diff3, true);
}

    .{ "qgate_body", qgateBody, 3 },
    .{ "ring_word", ringWord, 9 },
    .{ "roci_readers", rociReaders, 1 + 7 * 3 + 3 },
    .{ "roci_stage", rociStage, 2 * 4 + 1 },
    .{ "roci_collect", rociCollect, 3 },
    .{ "roci_split_cost", rociSplitCost, 2 },
    .{ "roci_combine_first", rociCombineFirst, 5 },
    .{ "roci_local_first", rociLocalFirst, 15 },
    .{ "roci_sent_first", rociSentFirst, 5 },
    .{ "roci_door", rociDoor, 13 },
    .{ "roci_boundary", rociBoundary, 3 + 1 + 1 },
    .{ "roci_codes", rociCodes, 2 * 2 + 1 + 1 },
    .{ "roci_local_first_wqb", rociLocalFirstWqb, 3 },
    .{ "roci_two_base", rociTwoBase, 3 },
    .{ "roci_any_rows", rociAnyRows, 3 * 8 },
    .{ "roci_two_base_pages", rociTwoBasePages, 4 * 2 + 1 },
    .{ "roci_any_experts", rociAnyExperts, 2 },
    .{ "roci_any_index", rociAnyIndex, 6 },
    .{ "roci_any_engram", rociAnyEngram, 5 },
    .{ "roci_dec_rows", rociDecRows, 3 * 8 },
    .{ "roci_dec_mixes", rociDecMixes, 4 + 4 + 3 + 3 },
    .{ "roci_dec_kset", rociDecKset, 3 },
    .{ "roci_pair", rociPair, 6 },
    .{ "roci_triple", rociTriple, 3 },
    try k.useQgateBody();