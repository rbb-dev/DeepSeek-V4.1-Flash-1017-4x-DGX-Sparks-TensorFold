//! The torch operations of the served forward that only move or widen bytes, on TensorFold's own torch-op kernels
//! (zig/kernels/cuda/torch_ops: pointwise.cu, movement.cu; ours in dsv41_ops.cu) with their development launchers'
//! geometry: `.float()` of bf16 and fp16 (exact), `.to(bf16)` of fp32 (round to nearest even), `.contiguous()` and row
//! copies, row gathers and scatters by index (`t[idx]`, `t[idx] = rows`), and the integer selections of the indexer
//! (torch's top-k of int64 keys, kernels.topk_indices). Their bytes are the same as torch's by construction; the layer
//! gate checks them.
const std = @import("std");
const cuda = @import("cuda");

/// tf_ds_assemble_rest_kernel's 16-byte units of a main row of `ig` bytes and a rest row of `ir` (assembleRest's ig / 8
/// and ir / 8 of fp16 values).
pub fn restUnits(ig: usize, ir: usize) ![2]usize {
    if (ig % 16 != 0 or ir % 16 != 0) return error.UnsupportedShape;
    return .{ ig / 16, ir / 16 };
}

/// launch_shape.h's tf_launch_blocks: a block a `threads` elements, at most 65535.
fn blocks(count: usize, threads: usize) u32 {
    return @intCast(@min((count + threads - 1) / threads, 65535));
}

/// The paced prefetch's settings (TF_EXL3_L2_DELAY_NS, TF_EXL3_L2_RATE_GBPS at Model.open; 0 GB/s: every chunk at once).
pub var l2_delay_ns: c_longlong = 3000;
pub var l2_rate_gbps: f64 = 230;
/// TF_DS_L2_CHUNK_KIB in effect (Model.open): the bulk prefetch's chunk in bytes, 8 KiB served. A wave is 128 chunks, so the DRAM queue a wave puts ahead of a NIC
/// read or a staging copy is 128 x chunk; the wave period scales with it (waveNs) and the rate and the bytes touched are the same: it changes no value.
pub var l2_chunk_bytes: u64 = 8192;
/// TF_DS_L2_CHUNK_KIB as asked (KiB): unset or empty is the served 8; 1, 2, 4, 8 or 16; null for any other text (Model.open refuses the start with it).
pub fn l2ChunkWanted(v: ?[]const u8) ?u64 {
    const t = v orelse return 8192;
    if (t.len == 0) return 8192;
    for ([_]u64{ 1, 2, 4, 8, 16 }) |kib| {
        var b: [4]u8 = undefined;
        if (std.mem.eql(u8, t, std.fmt.bufPrint(&b, "{d}", .{kib}) catch unreachable)) return kib * 1024;
    }
    return null;
}

/// tf_ds_l2_prefetch_kernel's argument: up to 16 device ranges (their first byte and length).
pub const Prefetch = extern struct { ptr: [16]u64 = @splat(0), bytes: [16]u64 = @splat(0), n: c_int = 0 };

    return waveNs(gbps, 8192);
}

/// A wave's period at `gbps` for chunks of `chunk` bytes: 128 chunks a wave (0 gbps: no pacing).
pub fn waveNs(gbps: f64, chunk: u64) c_longlong {
    return if (gbps > 0) @intFromFloat(128.0 * @as(f64, @floatFromInt(chunk)) / gbps) else 0;
pub const Ops = struct {
    pointwise: cuda.Module,
    movement: cuda.Module,
    family: cuda.Module,
    to_f32: cuda.Function,
    f16_to_f32: cuda.Function,
    add2_bf16: cuda.Function,
    widen_pairs: cuda.Function,
    collect2_fn: cuda.Function,
    ywiden: cuda.Function,
    topk_i64: cuda.Function,
    topk_indices: cuda.Function,
    scatter: cuda.Function,
    cand_fast: cuda.Function,
    iota_vis: cuda.Function,
    l2_prefetch: cuda.Function,
    block_max: cuda.Function,
    pool_pick: cuda.Function,
    apply_pool: cuda.Function,
    to_bf16: cuda.Function,
    strided: cuda.Function,
    gather: cuda.Function,
    copy16: cuda.Function,
    argmax_rows: cuda.Function,
    top_cand_rows: cuda.Function,
    gpu_clock: cuda.Function,
    rest_group: cuda.Function,
    remap_picks: cuda.Function,
    assemble_rest: cuda.Function,
    /// tf_ds_add2_bf16_v4_kernel (TF_DS_ADD2_VEC): add2Bf16 four elements a thread; null when the image lacks it.
    add2_v4: ?cuda.Function = null,
    keys_rebase: cuda.Function,

    /// Whether add2Bf16 takes the four-wide kernel where it can (TF_DS_ADD2_VEC, on unless 0, read at Model.open; the bytes are the scalar kernel's).
    pub var add2_vec_on: bool = true;

    /// Whether a call of add2Bf16 can take the four-wide kernel: a count of whole fours, 16-byte aligned operands, an 8-byte aligned output.
    pub fn add2VecFits(g0: u64, g1: u64, out: u64, count: usize) bool {
        return count % 4 == 0 and g0 % 16 == 0 and g1 % 16 == 0 and out % 8 == 0;
    }

    /// The three images (cuda.kernels.torch_pointwise, torch_movement and dsv41_ops, or the fatbins' bytes).
    pub fn load(d: *const cuda.Driver, pointwise: []const u8, movement: []const u8, family: []const u8) !Ops {
        var pw = try cuda.Module.load(d, pointwise);
        errdefer pw.unload();
        var mv = try cuda.Module.load(d, movement);
        errdefer mv.unload();
        var fm = try cuda.Module.load(d, family);
        errdefer fm.unload();
        return .{
            .pointwise = pw,
            .movement = mv,
            .family = fm,
            .to_f32 = try pw.function("tf_bf16_to_f32_kernel"),
            .f16_to_f32 = try fm.function("tf_ds_f16_to_f32_kernel"),
            .add2_bf16 = try fm.function("tf_ds_add2_bf16_kernel"),
            .widen_pairs = try fm.function("tf_ds_widen_pairs_kernel"),
            .collect2_fn = try fm.function("tf_ds_collect2_kernel"),
            .ywiden = try fm.function("tf_ds_ywiden_kernel"),
            .topk_i64 = try fm.function("tf_ds_topk_i64_kernel"),
            .topk_indices = try fm.function("tf_ds_topk_indices_kernel"),
            .scatter = try fm.function("tf_ds_scatter_rows_kernel"),
            .cand_fast = try fm.function("tf_ds_cand_fast_kernel"),
            .iota_vis = try fm.function("tf_ds_iota_vis_kernel"),
            .l2_prefetch = try fm.function("tf_ds_l2_prefetch_kernel"),
            .block_max = try fm.function("tf_ds_block_max_kernel"),
            .pool_pick = try fm.function("tf_ds_pool_pick_kernel"),
            .apply_pool = try fm.function("tf_ds_apply_pool_kernel"),
            .to_bf16 = try pw.function("tf_f32_to_bf16_kernel"),
            .strided = try mv.function("tf_strided_copy_kernel"),
            .gather = try mv.function("tf_gather_rows_kernel"),
            .copy16 = try fm.function("tf_ds_copy_rows16_kernel"),
            .argmax_rows = try fm.function("tf_ds_argmax_rows_kernel"),
            .top_cand_rows = try fm.function("tf_ds_top_cand_rows_kernel"),
            .gpu_clock = try fm.function("tf_ds_clock_kernel"),
            .rest_group = try fm.function("tf_ds_rest_group_kernel"),
            .remap_picks = try fm.function("tf_ds_remap_picks_kernel"),
            .assemble_rest = try fm.function("tf_ds_assemble_rest_kernel"),
            .add2_v4 = fm.function("tf_ds_add2_bf16_v4_kernel") catch null,
            .keys_rebase = try fm.function("tf_ds_keys_rebase_kernel"),
        };
    }

    pub fn unload(o: *Ops) void {
        o.pointwise.unload();
        o.movement.unload();
        o.family.unload();
        o.* = undefined;
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: usize, block: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = block } }, s, args);
    }

    /// `.float()` of `count` bf16 values (exact).
    pub fn toF32(o: *const Ops, s: cuda.Stream, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.to_f32, s, blocks(count, 256), 256, &a);
    }

    /// `.float()` of `count` fp16 values (exact).
    pub fn f16ToF32(o: *const Ops, s: cuda.Stream, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.f16_to_f32, s, blocks(count, 256), 256, &a);
    }

    /// Comm.sum of two ranks' fp32 partials then `.to(bf16)`: out = bf16(g0 + g1), one fp32 add in rank order.
    pub fn add2Bf16(o: *const Ops, s: cuda.Stream, g0: u64, g1: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(g0);
        a.add(g1);
        a.add(out);
        if (add2_vec_on and add2VecFits(g0, g1, out, count)) {
            if (o.add2_v4) |f| {
                a.add(@as(u64, count / 4));
                return go(f, s, blocks(count / 4, 256), 256, &a);
            }
        }
        a.add(@as(u64, count));
        try go(o.add2_bf16, s, blocks(count, 256), 256, &a);
    }

    /// TF_DS_ROCI_COLLECT: two parts of `bytes` each (a multiple of 16, 16-byte aligned) copied from s0, s1 to d0, d1 in one launch,
    /// 16 bytes a thread along each part: the owner sums' send blocks from device memory into the transport's arena as whole lines.
    pub fn collect2(o: *const Ops, s: cuda.Stream, s0: u64, d0: u64, s1: u64, d1: u64, bytes: usize) !void {
        if (bytes == 0) return;
        if (bytes % 16 != 0 or (s0 | d0 | s1 | d1) % 16 != 0) return error.Unaligned;
        const n16: u64 = bytes / 16;
        var a: cuda.Args = .{};
        a.add(s0);
        a.add(d0);
        a.add(s1);
        a.add(d1);
        a.add(n16);
        const bx: u32 = @intCast(@min((n16 + 255) / 256, 384));
        try cuda.launch.launch(o.collect2_fn, .{ .grid = .{ .x = bx, .y = 2 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// dst fp32 [q, 2 w] = (float(p0) | float(p1)) row by row, p0 and p1 bf16 [q, w] (exact): the owner sums' two
    /// pairs' bf16 sums as one plane of the readers' gathered partials.
    pub fn widenPairs(o: *const Ops, s: cuda.Stream, p0: u64, p1: u64, dst: u64, q: usize, w: usize) !void {
        if (q == 0 or w == 0) return;
        if (q > 65535 or (w + 255) / 256 > 65535) return error.TooLarge;
        var a: cuda.Args = .{};
        a.add(p0);
        a.add(p1);
        a.add(dst);
        a.add(@as(u64, w));
        try cuda.launch.launch(o.widen_pairs, .{ .grid = .{ .x = blocks(w, 256), .y = @intCast(q) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// prompt2d ySums: pieces bf16 [4, half, w] -> dst fp32 [n, 2 w], widened exactly (tf_ds_ywiden_kernel).
    pub fn yWiden(o: *const Ops, s: cuda.Stream, pieces: u64, dst: u64, n: usize, half: usize, w: usize) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(pieces);
        a.add(dst);
        a.add(@as(u64, n));
        a.add(@as(u64, half));
        a.add(@as(u64, w));
        try go(o.ywiden, s, blocks(n * 2 * w, 256), 256, &a);
    }

    /// keys.topk(k, sorted=False).values of unique int64 keys [rows, n] (row stride ks) into top [rows, k], the set in no
    /// particular order (_topk_finish sorts the indices).
    pub fn topkI64(o: *const Ops, s: cuda.Stream, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) !void {
        if (rows == 0) return;
        if (k > n) return error.BadTopK;
        var a: cuda.Args = .{};
        a.add(keys);
        a.add(@as(c_longlong, @intCast(ks)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        a.add(top);
        try cuda.launch.launch(o.topk_i64, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// kernels.topk_indices(score, k) with torch.where(top < vis, top, -1): each row's k highest fp32 scores' columns
    /// ascending (ties to the lower column), a column at or past vis[r] as -1. score [rows, n] (row stride ss), vis
    /// [rows] int64, out [rows, k] int64.
    pub fn topkIndices(o: *const Ops, s: cuda.Stream, score: u64, ss: usize, rows: usize, n: usize, k: usize, vis: u64, out: u64) !void {
        if (rows == 0) return;
        if (k > n or k > 2048) return error.BadTopK;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ss)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        a.add(vis);
        a.add(out);
        try cuda.launch.launch(o.topk_indices, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// dst row idx[i] = src row i (`t[idx] = rows`, distinct int64 indices): `rows` rows of `bytes` bytes, src rows
    /// `src_ld` bytes apart, dst rows `dst_ld`.
    pub fn scatterRows(o: *const Ops, s: cuda.Stream, src: u64, src_ld: usize, idx: u64, dst: u64, dst_ld: usize, bytes: usize, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(c_longlong, @intCast(src_ld)));
        a.add(idx);
        a.add(dst);
        a.add(@as(c_longlong, @intCast(dst_ld)));
        a.add(@as(c_longlong, @intCast(bytes)));
        a.add(@as(c_int, @intCast(rows)));
        try go(o.scatter, s, rows, 256, &a);
    }

    /// rounds.py _candidates_fast's mask when the pool takes every block: out [rows, nb] u8 (row stride os) = the block
    /// of bsize scores (row stride ss) has a maximum above -inf, or is the row's newest, (vis - 1) // bsize.
    pub fn candFast(o: *const Ops, s: cuda.Stream, score: u64, ss: usize, nb: usize, bsize: usize, vis: u64, out: u64, os: usize, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ss)));
        a.add(@as(c_int, @intCast(nb)));
        a.add(@as(c_int, @intCast(bsize)));
        a.add(vis);
        a.add(out);
        a.add(@as(c_longlong, @intCast(os)));
        try cuda.launch.launch(o.cand_fast, .{ .grid = .{ .x = blocks(nb, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Every 128-byte line of `pf`'s ranges touched into L2 (one block of 128 threads); writes nothing.
    pub fn l2Prefetch(o: *const Ops, s: cuda.Stream, pf: Prefetch) !void {
        if (pf.n == 0) return;
        // the served decode settings (exl3/prefetch.py): 3 us before the first wave, waves of 128 chunks of 8 KiB
        // at 230 GB/s (TF_EXL3_L2_DELAY_NS, TF_EXL3_L2_RATE_GBPS), one block
        const delay_ns: c_longlong = l2_delay_ns;
        const wave_ns: c_longlong = waveNs(l2_rate_gbps, l2_chunk_bytes);
        var a: cuda.Args = .{};
        a.add(pf);
        a.add(delay_ns);
        a.add(wave_ns);
        a.add(@as(c_longlong, @intCast(l2_chunk_bytes)));
        try cuda.launch.launch(o.l2_prefetch, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, s, &a);
    }

    /// The indexer's selection of every scanned key (its top-k takes all nb): out [rows, nb] int64 = j where j <
    /// vis[r], else -1 (vis [rows] int64), on the device.
    pub fn iotaVis(o: *const Ops, s: cuda.Stream, vis: u64, out: u64, nb: usize, rows: usize) !void {
        if (rows == 0 or nb == 0) return;
        var a: cuda.Args = .{};
        a.add(vis);
        a.add(out);
        a.add(@as(c_int, @intCast(nb)));
        try cuda.launch.launch(o.iota_vis, .{ .grid = .{ .x = blocks(nb, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The candidate pool's block maxima (model.py _candidates, rounds.py _candidate_blocks): out [rows, nb] fp32 (row
    /// stride os, nb = ceil(width / bsize)) = each block's maximum of the scores [rows, width] (row stride ss; the last
    /// block padded with -inf, a NaN wins), the row's newest block (vis - 1) // bsize pinned to +inf.
    pub fn blockMax(o: *const Ops, s: cuda.Stream, score: u64, ss: usize, width: usize, bsize: usize, vis: u64, out: u64, os: usize, rows: usize) !void {
        if (rows == 0) return;
        const nb = (width + bsize - 1) / bsize;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ss)));
        a.add(@as(c_int, @intCast(width)));
        a.add(@as(c_int, @intCast(bsize)));
        a.add(vis);
        a.add(out);
        a.add(@as(c_longlong, @intCast(os)));
        a.add(@as(c_int, @intCast(nb)));
        try cuda.launch.launch(o.block_max, .{ .grid = .{ .x = blocks(nb, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The pool from each row's top blocks idx [rows, k] int64 of the block maxima bmax [rows, nb] (row stride bs):
    /// _candidates' mask u8 [rows, nb] (row stride ms; zeros but the picked blocks above -inf) and/or
    /// _candidate_blocks' list int32 [rows, k] (the block, or -1 when its maximum is not above -inf); 0 leaves one out.
    pub fn poolPick(o: *const Ops, s: cuda.Stream, bmax: u64, bs: usize, nb: usize, idx: u64, k: usize, mask: u64, ms: usize, cblk: u64, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(bmax);
        a.add(@as(c_longlong, @intCast(bs)));
        a.add(@as(c_int, @intCast(nb)));
        a.add(idx);
        a.add(@as(c_int, @intCast(k)));
        a.add(mask);
        a.add(@as(c_longlong, @intCast(ms)));
        a.add(cblk);
        try cuda.launch.launch(o.pool_pick, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// model.py apply_candidates, in place: -inf at every column of score [rows, width] (row stride ss) whose block of
    /// bsize the row's pool mask (u8, row stride ms) leaves out.
    pub fn applyPool(o: *const Ops, s: cuda.Stream, score: u64, ss: usize, width: usize, bsize: usize, mask: u64, ms: usize, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(score);
        a.add(@as(c_longlong, @intCast(ss)));
        a.add(@as(c_int, @intCast(width)));
        a.add(@as(c_int, @intCast(bsize)));
        a.add(mask);
        a.add(@as(c_longlong, @intCast(ms)));
        try cuda.launch.launch(o.apply_pool, .{ .grid = .{ .x = blocks(width, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// `.to(torch.bfloat16)` of `count` fp32 values (round to nearest even).
    pub fn toBf16(o: *const Ops, s: cuda.Stream, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try go(o.to_bf16, s, blocks(count, 256), 256, &a);
    }

    /// `rows` rows of `bytes` bytes from src (rows `src_ld` bytes apart) to dst (rows `dst_ld` bytes apart).
    pub fn copyRows(o: *const Ops, s: cuda.Stream, src: u64, src_ld: usize, dst: u64, dst_ld: usize, bytes: usize, rows: usize) !void {
        if (rows == 0) return;
        if (src_ld < bytes or dst_ld < bytes) return error.Overlap;
        if ((src | dst | src_ld | dst_ld | bytes) % 16 == 0 and rows <= 65535) {
            // 16-byte words, a word a thread (the same bytes)
            var w: cuda.Args = .{};
            w.add(src);
            w.add(@as(u64, src_ld / 16));
            w.add(dst);
            w.add(@as(u64, dst_ld / 16));
            w.add(@as(u64, bytes / 16));
            return cuda.launch.launch(o.copy16, .{ .grid = .{ .x = blocks(bytes / 16, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &w);
        }
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        // tf_strided_copy (outer, middle, inner, source_outer, source_middle, destination_outer, destination_middle):
        // a row a block
        for ([_]u64{ rows, 1, bytes, src_ld, bytes, dst_ld, bytes }) |v| a.add(v);
        try go(o.strided, s, rows, 256, &a);
    }

    /// Each row's greedy token (sampling.argmax's: the first largest, a NaN never larger, 0 when the first value is
    /// NaN): out [rows] u32 of logits [rows, n] fp32 (row stride ld floats).
    pub fn argmaxRows(o: *const Ops, s: cuda.Stream, logits: u64, ld: usize, n: usize, rows: usize, out: u64) !void {
        if (rows == 0 or n == 0) return;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, ld));
        a.add(@as(c_int, @intCast(n)));
        a.add(out);
        try go(o.argmax_rows, s, rows, 1024, &a);
    }

    /// Track B: each row's first `count` (value, id) in sampling.topCandidates' order into vals [rows, count] fp32 and ids
    /// [rows, count] i64; flags [rows] i32 (0, or a row the host must read whole: 1 a NaN, 2 too many ties at the cut).
    pub fn topCandRows(o: *const Ops, s: cuda.Stream, logits: u64, ld: usize, n: usize, rows: usize, count: usize, vals: u64, ids: u64, flags: u64) !void {
        if (rows == 0 or n == 0) return;
        if (count == 0 or count > 64) return error.TooManyCandidates;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, ld));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(count)));
        a.add(vals);
        a.add(ids);
        a.add(flags);
        try go(o.top_cand_rows, s, rows, 1024, &a);
    }

    /// A parity split's rest grouping (experts2d _rest_group_kernel): the places of (uids, count, members [maxu, maxm])
    /// whose expert's rest this pair computes, in order, into (ruids, rcount, rmembers).
    pub fn restGroup(o: *const Ops, s: cuda.Stream, uids: u64, count: u64, members: u64, owner: u64, mine: u32, ruids: u64, rcount: u64, rmembers: u64, maxu: usize, maxm: usize) !void {
        if (maxu == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ uids, count, members, owner }) |v| a.add(v);
        a.add(@as(c_int, @intCast(mine)));
        for ([_]u64{ ruids, rcount, rmembers }) |v| a.add(v);
        a.add(@as(c_int, @intCast(maxu)));
        a.add(@as(c_int, @intCast(maxm)));
        try go(o.rest_group, s, @intCast(maxu), 32, &a);
    }

    /// rest_picks: out int32 [n] = remap[pick[i]].
    pub fn remapPicks(o: *const Ops, s: cuda.Stream, pick: u64, remap: u64, out: u64, n: usize) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ pick, remap, out }) |v| a.add(v);
        a.add(@as(c_int, @intCast(n)));
        try go(o.remap_picks, s, blocks(n, 256), 256, &a);
    }

    /// assemble_rest: out fp16 [P, 2 ig + ir] from the pairs' packs (main rows [P, ig] then rest rows [P, ir] each),
    /// the rest of a slot from owner[pick].
    pub fn assembleRest(o: *const Ops, s: cuda.Stream, part0: u64, part1: u64, pick: u64, owner: u64, out: u64, p: usize, ig: usize, ir: usize) !void {
        if (p == 0) return;
        if (ig % 8 != 0 or ir % 8 != 0) return error.UnsupportedShape;
        var a: cuda.Args = .{};
        for ([_]u64{ part0, part1, pick, owner, out }) |v| a.add(v);
        for ([_]usize{ p, ig / 8, ir / 8 }) |v| a.add(@as(c_int, @intCast(v)));
        try go(o.assemble_rest, s, @intCast(p), 64, &a);
    }

    /// assemble_rest in bytes: out [P, 2 ig + ir] bytes from the pairs' parts (main rows [P, ig] then rest rows [P, ir]
    /// bytes each), the rest of a slot from owner[pick]; assembleRest's launch at its 16-byte units (restUnits). The
    /// pack's codes (a byte a value) or its scales (four bytes a 32-value group).
    pub fn assembleRestBytes(o: *const Ops, s: cuda.Stream, part0: u64, part1: u64, pick: u64, owner: u64, out: u64, p: usize, ig: usize, ir: usize) !void {
        if (p == 0) return;
        const u = try restUnits(ig, ir);
        var a: cuda.Args = .{};
        for ([_]u64{ part0, part1, pick, owner, out }) |v| a.add(v);
        for ([_]usize{ p, u[0], u[1] }) |v| a.add(@as(c_int, @intCast(v)));
        try go(o.assemble_rest, s, @intCast(p), 64, &a);
    }

    /// TF_DS_IDX_SPLIT (idx_split.zig), in place: a node's top keys [rows, k] int64 (row stride ld elements) of its
    /// columns [c0, c0 + n) as keys of the whole row: each key of a column of the node's less c0 (its inverted column
    /// the row's, its score word kept), a pad (a low word of no such column) as it was.
    pub fn keysRebase(o: *const Ops, s: cuda.Stream, keys: u64, ld: usize, k: usize, n: usize, c0: usize, rows: usize) !void {
        if (rows == 0 or k == 0) return;
        var a: cuda.Args = .{};
        a.add(keys);
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(@as(c_int, @intCast(k)));
        a.add(@as(c_longlong, @intCast(n)));
        a.add(@as(c_longlong, @intCast(c0)));
        try cuda.launch.launch(o.keys_rebase, .{ .grid = .{ .x = blocks(k, 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The GPU clock (ns) into buf[i] when the stream reaches it (a profile).
    pub fn gpuClock(o: *const Ops, s: cuda.Stream, buf: u64, tag: u32, reset: bool, cap: u32) !void {
        var a: cuda.Args = .{};
        a.add(buf);
        a.add(tag);
        a.add(@as(u32, @intFromBool(reset)));
        a.add(cap);
        try go(o.gpu_clock, s, 1, 32, &a);
    }

    /// dst row i = src row idx[i] (int64 indices, rows of `bytes` bytes); an index outside [0, src_rows) sets the
    /// caller's zeroed uint32 flag `invalid` and leaves that row as it was.
    pub fn gatherRows(o: *const Ops, s: cuda.Stream, src: u64, src_rows: usize, idx: u64, dst: u64, bytes: usize, rows: usize, invalid: u64) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(idx);
        a.add(@as(u64, rows));
        a.add(@as(u64, src_rows));
        a.add(@as(u64, bytes));
        a.add(invalid);
        try go(o.gather, s, rows, 256, &a);
    }
};

test "launch geometry of the development launchers" {
    try std.testing.expectEqual(@as(u32, 1), blocks(1, 256));
    try std.testing.expectEqual(@as(u32, 2), blocks(257, 256));
    try std.testing.expectEqual(@as(u32, 65535), blocks(1 << 40, 256));
}
    // TF_DS_L2_CHUNK_KIB: the wave's period scales with the chunk so the rate (and every byte touched) stays: 128 x chunk / period = 230 GB/s within the rounding
    try std.testing.expectEqual(@as(c_longlong, 4559), waveNs(230, 8192));
    try std.testing.expectEqual(@as(c_longlong, 2279), waveNs(230, 4096));
    try std.testing.expectEqual(@as(c_longlong, 1139), waveNs(230, 2048));
    try std.testing.expectEqual(@as(c_longlong, 0), waveNs(0, 2048));
    for ([_]u64{ 1024, 2048, 4096, 8192, 16384 }) |c| {
        const rate = 128.0 * @as(f64, @floatFromInt(c)) / @as(f64, @floatFromInt(waveNs(230, c)));
        try std.testing.expect(rate > 229.0 and rate < 231.0);
    }
}

test "TF_DS_L2_CHUNK_KIB: unset or empty is 8 KiB, 1 / 2 / 4 / 8 / 16 are KiB, any other text is null (the start is refused at its call site)" {
    try std.testing.expectEqual(@as(?u64, 8192), l2ChunkWanted(null));
    try std.testing.expectEqual(@as(?u64, 8192), l2ChunkWanted(""));
    try std.testing.expectEqual(@as(?u64, 1024), l2ChunkWanted("1"));
    try std.testing.expectEqual(@as(?u64, 2048), l2ChunkWanted("2"));
    try std.testing.expectEqual(@as(?u64, 4096), l2ChunkWanted("4"));
    try std.testing.expectEqual(@as(?u64, 8192), l2ChunkWanted("8"));
    try std.testing.expectEqual(@as(?u64, 16384), l2ChunkWanted("16"));
    for ([_][]const u8{ "0", "3", "32", "2048", "-2", " 2", "02", "two", "2k" }) |bad| try std.testing.expectEqual(@as(?u64, null), l2ChunkWanted(bad));

test "TF_DS_ADD2_VEC: the four-wide add takes whole fours at 16-byte operands and an 8-byte output, anything else the scalar kernel; off unless asked" {
    const O = Ops;
    try std.testing.expect(O.add2_vec_on); // on unless TF_DS_ADD2_VEC=0 (a bit-identical speed change)
    try std.testing.expect(O.add2VecFits(0x1000, 0x2000, 0x3008, 2560 * 1024));
    try std.testing.expect(!O.add2VecFits(0x1000, 0x2000, 0x3000, 7)); // not whole fours
    try std.testing.expect(!O.add2VecFits(0x1004, 0x2000, 0x3000, 8)); // a partial not 16-byte aligned
    try std.testing.expect(!O.add2VecFits(0x1000, 0x2008, 0x3000, 8));
    try std.testing.expect(!O.add2VecFits(0x1000, 0x2000, 0x3004, 8)); // output not 8-byte aligned
    // the owner sums' real operands: a quarter (q rows x w columns) at the plan's offsets of 4-float multiples
    for ([_]usize{ 1024, 512, 16 }) |q| try std.testing.expect(O.add2VecFits(0x40000 + q * 2560 * 4, 0x80000, 0xc0000 + q * 2560 * 2, q * 2560));
    // the call: the vector launch unless switched off and only when the kernel is loaded, the scalar launch after it
    const src = @embedFile("ops.zig");
    const f = src[std.mem.indexOf(u8, src, "pub fn add2Bf16(").?..];
    const v = std.mem.indexOf(u8, f, "if (add2_vec_on and add2VecFits(g0, g1, out, count))").?;
    const sc = std.mem.indexOf(u8, f, "try go(o.add2_bf16, s, blocks(count, 256), 256, &a);").?;
    try std.testing.expect(v < sc);
}
