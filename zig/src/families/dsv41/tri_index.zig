//! kernels.py's indexer launches: the compressed keys' scores (a row a program in decode windows, row tiles in prompt
//! chunks), their top-k keys and tile maxima, the pool-only keys and the selection, each as its Python wrapper does.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const aot = cuda.aot;
const Tri = tri.Tri;
const u = tri.u;
const p = aot.ptr;

/// kernels.DECODE_ROWS (TF_DS_DECODE_ROWS unset): calls of at most this many rows take the row-invariant kernels.
pub const decode_rows = 16;
/// The score kernels' key tile: index_score's bn and TOPK_PRUNE_BN (the tile maxima's).
pub const tile = 64;
/// kernels.TOPK_FUSED_MAX (TF_DS_TOPK_FUSED_MAX unset): keys a row the fused selection takes at most.
pub const topk_fused_max = 4096;
/// kernels.IDX_RB with IDX_TREE on (TF_DS_IDX_TREE / TF_DS_IDX_RB unset): prompt chunks' scores one row a program.
/// WP6 (docs/prefill-plan.md): the kit's own `_index_score_rows` is already captured at RB=2 (dequantizing a
/// 64-key tile once for 2 rows instead of 1; TF_DS_IDX_RB=2 set once at startup, model.zig), but not at RB=1's
/// "TREE" constexpr -- the matching capture simply has no TREE entry, so the call below only asks for it at RB=1.
pub var idx_rb: usize = 1;
/// TF_DS_IDX_U in effect (Model.open): a prompt chunk's index score asks for the kit's twin _index_score_rows_u (each
/// packed key code byte's two FP4 values unpacked in hardware: the same scores and keys); a variant without its twin
/// keeps the baseline kernel (Tri.runTwin).
pub var idx_u: bool = false;
/// TF_DS_IDX_V in effect (Model.open): a prompt chunk's index score asks for the kit's twin _index_score_rows_v (the
/// key tile dequantized once and each of a program's rows dotted in turn, its head sum written out, at four warps: the
/// same scores and keys); it replaces _index_score_rows_u when both are on, and a variant without its twin takes the
/// _u twin if TF_DS_IDX_U is on, else the baseline kernel.
pub var idx_v: bool = false;
/// TF_DS_IDX_TPP in effect (Model.open): key tiles a program of a prompt chunk's index score. 0 is the baseline kernel (the base set's, or
/// _index_score_rows_u with TF_DS_IDX_U); 4 asks for the kit's twin _index_score_rows_t (a program takes 4 key tiles in turn: its
/// queries, head weights and visible lengths read once, the next tile's rows prefetched into L1; grid y = cdiv(nt, 4), TPP in the
/// variant's consts; the same scores, keys and tile maxima). A variant without its twin keeps the next choice (u twin or the baseline),
/// counted in tri.twin_misses.
pub var idx_tpp: usize = 0;
/// What topk_select_pruned's `every` holds in each row: a visible count past every tile position.
pub const every_vis: i64 = 1 << 40;

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

/// kernels._pow2.
fn pow2(n: usize) bool {
    return n > 0 and n & (n - 1) == 0;
}

/// index_score's k: the indexer's keys, a packed FP4 pair (codes u8 [N, ID/2], E8M0 scales u8 [N, ID/32]) or bf16 [N, ID].
pub const IndexK = union(enum) {
    fp4: struct { codes: u64, scales: u64 },
    bf16: u64,

    fn ty(k: IndexK) []const u8 {
        return if (k == .fp4) "*u8" else "*bf16";
    }

    /// KS's type: the E8M0 scale tensor, u8 in every captured variant. A bf16-key launch (PACKED 0) does not read it, but the kit
    /// captured those variants with KS as *u8, and a variant is found by its arguments' types (the replay's idx_deq probe asked
    /// for *bf16 and found none).
    fn scalesTy(_: IndexK) []const u8 {
        return "*u8";
    }

    fn codes(k: IndexK) u64 {
        return switch (k) {
            .fp4 => |f| f.codes,
            .bf16 => |a| a,
        };
    }

    fn scales(k: IndexK) u64 {
        return switch (k) {
            .fp4 => |f| f.scales,
            .bf16 => |a| a,
        };
    }
};

/// index_keys' cand: the candidate pool's block mask (bool [R, >= cdiv(n, cand_block)], read as u8) and its row stride.
pub const Cand = struct { mask: u64, stride: usize };

/// _idx_keys_deq (the kit's extra set): the indexer's `n` packed keys dequantized once to bf16 [n, id] in `out`, the keys
/// _index_score_rows_u builds in every program. A TIMING PROBE (replay section idx_deq): scores read from them are not the packed
/// kernel's bytes. Error.MissingTritonVariant when the set has no kernel.
pub fn idxKeysDeq(t: Tri, codes: u64, scales: u64, out: u64, n: usize, id: usize) !void {
    try t.runExtra("_idx_keys_deq", .{ tri.cdiv(n, tile), 1, 1 }, &.{ p("K", "*u8", codes), p("KS", "*u8", scales), p("OUT", "*bf16", out), int("n", n) }, &.{ ci("ID", id), ci("BN", tile) });
}

/// index_score: q [R, IH, ID], k, w [R, IH], vis [R] -> out [R, n] fp32 scores; `keys`: int64 keys; `tmax`: tile maxima.
pub fn indexScore(t: Tri, q: u64, k: IndexK, w: u64, vis: u64, n: usize, out: u64, base: ?u64, keys: bool, tmax: ?u64, rowwise: bool, rows: usize, ih: usize, id: usize) !void {
    const nt = tri.cdiv(n, tile);
    const ot: []const u8 = if (keys) "*i64" else "*fp32";
    // tmax [R, cdiv(n, 64)] int64 (its maxima only with keys); without it the kernel gets out there
    const tm = p("TMAX", if (tmax != null) "*i64" else ot, tmax orelse out);
    if (rows <= tri.decode_rows or rowwise) {
        // base [R]: each row's first pool row (concurrent rounds); vis stands in for it and for the candidate mask
        // HAS_CAND, CB and DEAD_PAD at the kernel's defaults (index_score passes none of them)
        // SKIP_DEAD: switch "tile_skip" (served on), a tile wholly at or past vis skips its dots
        return t.run("_index_score", .{ u(rows), nt, 1 }, &.{
            p("Q", "*bf16", q),     p("K", k.ty(), k.codes()), p("KS", k.scalesTy(), k.scales()), p("Wt", "*bf16", w),
            p("VIS", "*i64", vis),  p("OUT", ot, out),         int("n", n),                       p("KB", "*i64", base orelse vis),
            p("CAND", "*i64", vis), int("cs", 0),              tm,                                int("nt", nt),
        }, &.{
            ci("IH", ih),          ci("ID", id), ci("BN", tile),   cb("PACKED", k == .fp4),               cb("HAS_BASE", base != null), t.pdlConst(),
            cb("HAS_CAND", false), ci("CB", 8),  cb("KEYS", keys), cb("HAS_TMAX", tmax != null and keys), cb("SKIP_DEAD", true),        cb("DEAD_PAD", false),
        });
    }
    std.debug.assert(base == null);
    const rb_args = [_]aot.Arg{
        p("Q", "*bf16", q),    p("K", k.ty(), k.codes()), p("KS", k.scalesTy(), k.scales()), p("Wt", "*bf16", w),
        p("VIS", "*i64", vis), p("OUT", ot, out),         int("n", n),                       int("rows", rows),
        tm,                    int("nt", nt),
    };
    const rb_base_consts = [_]aot.Const{
        ci("IH", ih),            ci("ID", id),     ci("BN", tile),                        ci("RB", idx_rb),
        cb("PACKED", k == .fp4), cb("KEYS", keys), cb("HAS_TMAX", tmax != null and keys),
    };
    // RB=1 is the only row count the kit captured with a "TREE" constexpr at all; _index_score_rows_v is captured at
    // both with it (its head sum is written out), so every RB asks its variant for TREE.
    const with_tree = rb_base_consts ++ [_]aot.Const{cb("TREE", true)};
    const base_consts: []const aot.Const = if (idx_rb == 1) &with_tree else &rb_base_consts;
    try scoreRows(t, .{ tri.cdiv(rows, idx_rb), nt, 1 }, &rb_args, base_consts, &with_tree);
}

/// _index_score_rows' launch, from the extra set's twin _index_score_rows_v with TF_DS_IDX_V (idx_v; its consts carry
/// TREE), else from its twin _index_score_rows_u with TF_DS_IDX_U (idx_u), else the baseline kernel. A missing V twin falls
/// back to the u twin and then to the baseline kernel, each with the consts its variant was captured with.
fn scoreRows(t: Tri, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const, v_consts: []const aot.Const) !void {
    if (idx_v) {
        if (t.log != null) return t.runTwin("_index_score_rows_v", "_index_score_rows", grid, args, v_consts);
        if (t.extra) |set| if (set.lookup("_index_score_rows_v", args, v_consts)) |v| return aot.Set.launchVariant(v, t.stream, grid, args);
        _ = tri.twin_misses.fetchAdd(1, .monotonic);
    }
    if (idx_tpp != 0 and consts.len < 15) {
        var g = grid;
        g[1] = tri.cdiv(grid[1], idx_tpp);
        var cs: [16]aot.Const = undefined;
        @memcpy(cs[0..consts.len], consts);
        cs[consts.len] = ci("TPP", idx_tpp);
        const tc = cs[0 .. consts.len + 1];
        if (t.log != null) return t.runTwin("_index_score_rows_t", "_index_score_rows", g, args, tc);
        if (t.extra) |set| if (set.lookup("_index_score_rows_t", args, tc)) |v| return aot.Set.launchVariant(v, t.stream, g, args);
        _ = tri.twin_misses.fetchAdd(1, .monotonic);
    }
    if (idx_u) return t.runTwin("_index_score_rows_u", "_index_score_rows", grid, args, consts);
    try t.run("_index_score_rows", grid, args, consts);
}

/// index_keys: index_score's int64 keys (decode windows), apply_candidates' mask folded in; pruned_k: topk_select_pruned's k.
pub fn indexKeys(t: Tri, q: u64, k: IndexK, w: u64, vis: u64, n: usize, base: ?u64, cand: ?Cand, cand_block: usize, tmax: ?u64, pruned_k: usize, out: u64, rows: usize, ih: usize, id: usize) !void {
    std.debug.assert(rows <= tri.decode_rows);
    const nt = tri.cdiv(n, tile);
    // switch "tile_skip" (served on): dead tiles skip their dots, and write no keys when only the pruned selection reads
    const skip = true;
    // the mask as u8 (cand.view(torch.uint8)), its row stride; without one vis stands in, stride 0
    const cv = if (cand) |m| p("CAND", "*u8", m.mask) else p("CAND", "*i64", vis);
    const cs: usize = if (cand) |m| m.stride else 0;
    try t.run("_index_score", .{ u(rows), nt, 1 }, &.{
        p("Q", "*bf16", q),    p("K", k.ty(), k.codes()), p("KS", k.scalesTy(), k.scales()),  p("Wt", "*bf16", w),
        p("VIS", "*i64", vis), p("OUT", "*i64", out),     int("n", n),                        p("KB", "*i64", base orelse vis),
        cv,                    int("cs", cs),             p("TMAX", "*i64", tmax orelse out), int("nt", nt),
    }, &.{
        ci("IH", ih),                 ci("ID", id),         ci("BN", tile),   cb("PACKED", k == .fp4),      cb("HAS_BASE", base != null), t.pdlConst(),
        cb("HAS_CAND", cand != null), ci("CB", cand_block), cb("KEYS", true), cb("HAS_TMAX", tmax != null), cb("SKIP_DEAD", skip),        cb("DEAD_PAD", skip and tmax != null and prunes(n, pruned_k)),
    });
}

/// index_keys_cand: the keys of the pool's blocks only, cblk [R, nblk] i32 (-1: none; row stride cbs) -> out [R, nblk * CB].
pub fn indexKeysCand(t: Tri, q: u64, k: IndexK, w: u64, vis: u64, n: usize, cblk: u64, cbs: usize, nblk: usize, cand_block: usize, base: ?u64, out: u64, rows: usize, ih: usize, id: usize) !void {
    std.debug.assert(rows <= tri.decode_rows);
    const nout = nblk * cand_block;
    try t.run("_index_score_cand", .{ u(rows), tri.cdiv(nout, tile), 1 }, &.{
        p("Q", "*bf16", q),               p("K", k.ty(), k.codes()), p("KS", k.scalesTy(), k.scales()), p("Wt", "*bf16", w),
        p("VIS", "*i64", vis),            p("OUT", "*i64", out),     int("n", n),                       int("nout", nout),
        p("KB", "*i64", base orelse vis), p("CBLK", "*i32", cblk),   int("cbs", cbs),
    }, &.{ ci("IH", ih), ci("ID", id), ci("BN", tile), cb("PACKED", k == .fp4), cb("HAS_BASE", base != null), ci("CB", cand_block), t.pdlConst() });
}

/// torch's top-k, topk_select's one step that is no Triton launch: keys.topk(k, sorted=False).values into top [R, k] int64.
pub const TopK = struct {
    top: u64,
    ctx: ?*anyopaque = null,
    run: *const fn (ctx: ?*anyopaque, t: Tri, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) anyerror!void,
};

/// topk_select: keys int64 [R, n] (row stride ks) -> out [R, k] int64, the k largest keys' indices sorted, -1 past vis[r].
pub fn topkSelect(t: Tri, keys: u64, ks: usize, k: usize, vis: u64, out: u64, top: TopK, rows: usize, n: usize) !void {
    // switch "topk_fused" (served on): one launch a row for power-of-two k <= n <= TOPK_FUSED_MAX, contiguous int64 keys
    if (pow2(n) and pow2(k) and k <= n and n <= topk_fused_max and (ks == n or rows <= 1)) {
        return t.run("_topk_sel", .{ u(rows), 1, 1 }, &.{ p("KEYS", "*i64", keys), p("VIS", "*i64", vis), p("OUT", "*i64", out) }, &.{ ci("N", n), ci("KK", k), t.pdlConst() });
    }
    try topkSelectTorch(t, keys, ks, k, vis, out, top, rows, n);
}

/// topk_select's steps past the fused selection, whatever n: torch's top-k into top.top, then _topk_finish.
pub fn topkSelectTorch(t: Tri, keys: u64, ks: usize, k: usize, vis: u64, out: u64, top: TopK, rows: usize, n: usize) !void {
    try top.run(top.ctx, t, keys, ks, rows, n, k, top.top);
    try t.run("_topk_finish", .{ u(rows), 1, 1 }, &.{ p("TOP", "*i64", top.top), p("VIS", "*i64", vis), p("OUT", "*i64", out) }, &.{ ci("KK", k), t.pdlConst() });
}

/// prunes: topk_select_pruned searches tiles for rows of n keys (more than TOPK_FUSED_MAX keys and more than k tiles).
pub fn prunes(n: usize, k: usize) bool {
    // switch "topk_prune" (served on)
    return k > 0 and n > topk_fused_max and tri.cdiv(n, tile) > k;
}

/// topk_select_pruned's buffers: tpos [R, k], cand [R, k * 64], every [>= R] all every_vis (constant: filled once).
pub const PrunedBufs = struct { tpos: u64, cand: u64, every: u64 };

/// topk_select_pruned: topk_select(keys, k, vis) from the k 64-key tiles whose maxima tmax [R, cdiv(n, 64)] are largest.
pub fn topkSelectPruned(t: Tri, keys: u64, ks: usize, tmax: u64, k: usize, vis: u64, out: u64, bufs: PrunedBufs, top: TopK, rows: usize, n: usize) !void {
    // kernels.py also falls back to topk_select when keys.stride(1) != 1: every caller's keys have unit column stride
    if (!prunes(n, k)) return topkSelect(t, keys, ks, k, vis, out, top, rows, n);
    try leadTiles(t, keys, ks, tmax, k, bufs, top, rows, n);
    try topkSelect(t, bufs.cand, k * tile, k, vis, out, top, rows, k * tile);
}

/// topk_select_pruned's tile step: the k tiles whose maxima tmax are largest (tpos), their keys into cand [R, k * 64].
fn leadTiles(t: Tri, keys: u64, ks: usize, tmax: u64, k: usize, bufs: PrunedBufs, top: TopK, rows: usize, n: usize) !void {
    const nt = tri.cdiv(n, tile);
    try topkSelect(t, tmax, nt, k, bufs.every, bufs.tpos, top, rows, nt);
    try t.run("_gather_tiles", .{ u(rows), u(k), 1 }, &.{
        p("KEYS", "*i64", keys), p("TPOS", "*i64", bufs.tpos), p("OUT", "*i64", bufs.cand), int("n", n), int("ks", ks),
    }, &.{ ci("BN", tile), ci("KK", k), t.pdlConst() });
}

/// TF_DS_IDX_SPLIT (idx_split.zig): topk_select_pruned's k largest keys themselves, not their columns: the tile step,
/// then torch's top-k of the k tiles' keys into out [R, k] (rows k apart, the set in no particular order); torch's
/// top-k of every key when n does not prune. The same k keys topk_select_pruned takes its columns from.
pub fn topKeysPruned(t: Tri, keys: u64, ks: usize, tmax: u64, k: usize, out: u64, bufs: PrunedBufs, top: TopK, rows: usize, n: usize) !void {
    if (!prunes(n, k)) return top.run(top.ctx, t, keys, ks, rows, n, k, out);
    try leadTiles(t, keys, ks, tmax, k, bufs, top, rows, n);
    try top.run(top.ctx, t, bufs.cand, k * tile, rows, k * tile, k, out);
}

/// score_keys: fp32 scores [R, n] (contiguous) -> out [R, n] int64 keys; tmax: each 64-key tile's largest [R, cdiv(n, 64)].
pub fn scoreKeys(t: Tri, score: u64, tmax: ?u64, out: u64, rows: usize, n: usize) !void {
    try t.run("_score_keys", .{ u(rows), tri.cdiv(n, 1024), 1 }, &.{
        p("S", "*fp32", score), p("OUT", "*i64", out), int("n", n), p("TMAX", "*i64", tmax orelse out), int("nt", tri.cdiv(n, tile)),
    }, &.{ ci("BN", 1024), cb("HAS_TMAX", tmax != null), ci("TB", tile), t.pdlConst() });
}

// -- conformance with the recorded launches ----------------------------------------------------------------------

const Case = tri.Case;

/// config.json's index_topk: rounds.py's k (and index_keys' pruned_k) is min(index_topk, n).
const index_topk = 512;

/// The recorded key cache: FP4 codes [N, ID/2] with their scales, or bf16 rows [N, ID].
fn caseK(c: Case) IndexK {
    if (2 * c.dim("K", 1) == c.dim("Q", 2)) return .{ .fp4 = .{ .codes = c.ptr("K"), .scales = c.ptr("KS") } };
    return .{ .bf16 = c.ptr("K") };
}

/// A launch of a decode round (rounds.py passes the pool's per-row bases), not of the prompt forward (attention_k: none).
fn inRound(c: Case) bool {
    return std.mem.startsWith(u8, c.v.get("phase").?.string, "round:");
}

/// A recorded int argument, passed or folded by Triton (the value 1) into a constexpr.
fn num(c: Case, name: []const u8) usize {
    return @intCast(if (c.hasConst(name)) c.constInt(name) else c.int(name));
}

/// The tests' torch top-k: nothing (no Triton launch to check).
fn noTopK(_: ?*anyopaque, _: Tri, _: u64, _: usize, _: usize, _: usize, _: usize, _: u64) anyerror!void {}

test "index_score and index_keys launch _index_score as recorded" {
    try tri.conform("_index_score", struct {
        fn call(t: Tri, c: Case) !void {
            const rows = c.dim("Q", 0);
            const n = c.dim("OUT", 1);
            const keys = c.constInt("KEYS") != 0;
            const tmax: ?u64 = if (c.constInt("HAS_TMAX") != 0) c.ptr("TMAX") else null;
            const base: ?u64 = if (inRound(c)) c.ptr("KB") else null;
            if (keys and inRound(c)) {
                // rounds.py's index_keys: a pool mask is [R, blocks] (vis [R] stands in without one)
                const cand: ?Cand = if (c.rank("CAND") == 2) .{ .mask = c.ptr("CAND"), .stride = c.stride("CAND", 0) } else null;
                return indexKeys(t, c.ptr("Q"), caseK(c), c.ptr("Wt"), c.ptr("VIS"), n, base, cand, @intCast(c.constInt("CB")), tmax, @min(index_topk, n), c.ptr("OUT"), rows, c.dim("Q", 1), c.dim("Q", 2));
            }
            try indexScore(t, c.ptr("Q"), caseK(c), c.ptr("Wt"), c.ptr("VIS"), n, c.ptr("OUT"), base, keys, tmax, false, rows, c.dim("Q", 1), c.dim("Q", 2));
        }
    }.call);
}

test "index_score (prompt chunks) launches _index_score_rows as recorded" {
    try tri.conform("_index_score_rows", struct {
        fn call(t: Tri, c: Case) !void {
            const tmax: ?u64 = if (c.constInt("HAS_TMAX") != 0) c.ptr("TMAX") else null;
            try indexScore(t, c.ptr("Q"), caseK(c), c.ptr("Wt"), c.ptr("VIS"), c.dim("OUT", 1), c.ptr("OUT"), null, c.constInt("KEYS") != 0, tmax, false, c.dim("Q", 0), c.dim("Q", 1), c.dim("Q", 2));
        }
    }.call);
}

test "bf16 keys: K is *bf16 and KS the scale tensor's own type *u8 in every launch (the kit's PACKED 0 variants were captured so; a variant is found by its arguments' types)" {
    const keep_rb = idx_rb;
    defer idx_rb = keep_rb;
    for ([_]usize{ 1, 2 }) |rb| for ([_]usize{ 1, 512 }) |rows| for ([_]bool{ true, false }) |bf16| {
        idx_rb = rb;
        var log: tri.Log = .{ .gpa = std.testing.allocator };
        defer log.deinit();
        const k: IndexK = if (bf16) .{ .bf16 = 0x200 } else .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } };
        try indexScore(.{ .log = &log }, 0x100, k, 0x400, 0x500, 40000, 0x600, null, false, null, false, rows, 32, 128);
        try std.testing.expectEqual(@as(usize, 1), log.items.items.len);
        var seen: usize = 0;
        for (log.items.items[0].args) |a| {
            if (std.mem.eql(u8, a.name, "K")) {
                try std.testing.expectEqualStrings(if (bf16) "*bf16" else "*u8", a.value.ptr.ty);
                seen += 1;
            }
            if (std.mem.eql(u8, a.name, "KS")) {
                try std.testing.expectEqualStrings("*u8", a.value.ptr.ty);
                seen += 1;
            }
        }
        try std.testing.expectEqual(@as(usize, 2), seen);
    };
}

test "TF_DS_IDX_U: a prompt chunk's index score launches _index_score_rows' twin with its arguments; decode windows, rowwise and unset as the baseline" {
    const keep_u = idx_u;
    defer idx_u = keep_u;
    const keep_rb = idx_rb;
    defer idx_rb = keep_rb;
    for ([_]usize{ 1, 2 }) |rb| for ([_]bool{ true, false }) |keyed| {
        idx_rb = rb;
        var today: tri.Log = .{ .gpa = std.testing.allocator };
        defer today.deinit();
        idx_u = false;
        try indexScore(.{ .log = &today }, 0x100, .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } }, 0x400, 0x500, 40000, 0x600, null, keyed, if (keyed) 0x700 else null, false, 512, 32, 128);
        var twin: tri.Log = .{ .gpa = std.testing.allocator };
        defer twin.deinit();
        idx_u = true;
        try indexScore(.{ .log = &twin }, 0x100, .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } }, 0x400, 0x500, 40000, 0x600, null, keyed, if (keyed) 0x700 else null, false, 512, 32, 128);
        try std.testing.expectEqual(@as(usize, 1), today.items.items.len);
        try std.testing.expectEqual(@as(usize, 1), twin.items.items.len);
        const p0 = today.items.items[0];
        const p1 = twin.items.items[0];
        try std.testing.expectEqualStrings("_index_score_rows", p0.name);
        try std.testing.expect(!p0.extra);
        try std.testing.expectEqualStrings("_index_score_rows_u", p1.name);
        try std.testing.expect(p1.extra);
        try std.testing.expectEqual(p0.grid, p1.grid);
        try std.testing.expectEqual(p0.args.len, p1.args.len);
        try std.testing.expectEqual(p0.consts.len, p1.consts.len);
        for (p0.args, p1.args) |q0, q1| try std.testing.expect(std.mem.eql(u8, q0.name, q1.name) and std.meta.eql(q0.value, q1.value));
        for (p0.consts, p1.consts) |q0, q1| try std.testing.expect(std.mem.eql(u8, q0.name, q1.name) and std.meta.eql(q0.int, q1.int));
        // (RB 1 asks for the TREE variants)
        var tree = false;
        for (p1.consts) |q| tree = tree or (std.mem.eql(u8, q.name, "TREE") and q.int == 1);
        try std.testing.expectEqual(rb == 1, tree);
    };
    // decode windows and rowwise calls: _index_score, the baseline, with or without the switch
    idx_u = true;
    for ([_]struct { rows: usize, rowwise: bool }{ .{ .rows = 16, .rowwise = false }, .{ .rows = 512, .rowwise = true } }) |c| {
        var log: tri.Log = .{ .gpa = std.testing.allocator };
        defer log.deinit();
        try indexScore(.{ .log = &log }, 0x100, .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } }, 0x400, 0x500, 40000, 0x600, null, true, 0x700, c.rowwise, c.rows, 32, 128);
        try std.testing.expectEqualStrings("_index_score", log.items.items[0].name);
        try std.testing.expect(!log.items.items[0].extra);
    }
}

test "TF_DS_IDX_TPP: a prompt chunk's index score launches _index_score_rows_t with TPP 4 in its consts and grid y = cdiv(nt, 4), its arguments unchanged; 0 and decode windows as the baseline" {
    const keep_t = idx_tpp;
    defer idx_tpp = keep_t;
    const keep_u = idx_u;
    defer idx_u = keep_u;
    const keep_v = idx_v;
    defer idx_v = keep_v;
    const keep_rb = idx_rb;
    defer idx_rb = keep_rb;
    idx_v = false;
    for ([_]usize{ 1, 2 }) |rb| for ([_]bool{ true, false }) |keyed| {
        idx_rb = rb;
        idx_tpp = 0;
        var today: tri.Log = .{ .gpa = std.testing.allocator };
        defer today.deinit();
        try indexScore(.{ .log = &today }, 0x100, .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } }, 0x400, 0x500, 40000, 0x600, null, keyed, if (keyed) 0x700 else null, false, 512, 32, 128);
        var twin: tri.Log = .{ .gpa = std.testing.allocator };
        defer twin.deinit();
        idx_tpp = 4;
        try indexScore(.{ .log = &twin }, 0x100, .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } }, 0x400, 0x500, 40000, 0x600, null, keyed, if (keyed) 0x700 else null, false, 512, 32, 128);
        try std.testing.expectEqual(@as(usize, 1), today.items.items.len);
        try std.testing.expectEqual(@as(usize, 1), twin.items.items.len);
        const p0 = today.items.items[0];
        const p1 = twin.items.items[0];
        try std.testing.expectEqualStrings("_index_score_rows", p0.name);
        try std.testing.expectEqualStrings("_index_score_rows_t", p1.name);
        try std.testing.expect(p1.extra);
        try std.testing.expectEqual(p0.grid[0], p1.grid[0]);
        try std.testing.expectEqual(p0.grid[1] / 4 + @intFromBool(p0.grid[1] % 4 != 0), p1.grid[1]);
        try std.testing.expectEqual(p0.grid[2], p1.grid[2]);
        for (p0.args, p1.args) |q0, q1| try std.testing.expect(std.mem.eql(u8, q0.name, q1.name) and std.meta.eql(q0.value, q1.value));
        try std.testing.expectEqual(p0.consts.len + 1, p1.consts.len);
        for (p0.consts, p1.consts[0..p0.consts.len]) |q0, q1| try std.testing.expect(std.mem.eql(u8, q0.name, q1.name) and std.meta.eql(q0.int, q1.int));
        try std.testing.expectEqualStrings("TPP", p1.consts[p0.consts.len].name);
        try std.testing.expectEqual(@as(?i64, 4), p1.consts[p0.consts.len].int);
    };
}

test "TF_DS_IDX_V: a prompt chunk's index score launches _index_score_rows_v with TREE, its arguments and its grid; it replaces the _u twin; decode windows and rowwise as the baseline" {
    const keep_v = idx_v;
    defer idx_v = keep_v;
    const keep_u = idx_u;
    defer idx_u = keep_u;
    const keep_rb = idx_rb;
    defer idx_rb = keep_rb;
    for ([_]usize{ 1, 2 }) |rb| for ([_]bool{ true, false }) |keyed| {
        idx_rb = rb;
        idx_v = false;
        idx_u = false;
        var today: tri.Log = .{ .gpa = std.testing.allocator };
        defer today.deinit();
        try indexScore(.{ .log = &today }, 0x100, .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } }, 0x400, 0x500, 40000, 0x600, null, keyed, if (keyed) 0x700 else null, false, 512, 32, 128);
        var twin: tri.Log = .{ .gpa = std.testing.allocator };
        defer twin.deinit();
        idx_v = true;
        try indexScore(.{ .log = &twin }, 0x100, .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } }, 0x400, 0x500, 40000, 0x600, null, keyed, if (keyed) 0x700 else null, false, 512, 32, 128);
        var both: tri.Log = .{ .gpa = std.testing.allocator };
        defer both.deinit();
        idx_u = true;
        try indexScore(.{ .log = &both }, 0x100, .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } }, 0x400, 0x500, 40000, 0x600, null, keyed, if (keyed) 0x700 else null, false, 512, 32, 128);
        try std.testing.expectEqual(@as(usize, 1), today.items.items.len);
        try std.testing.expectEqual(@as(usize, 1), twin.items.items.len);
        try std.testing.expectEqual(@as(usize, 1), both.items.items.len);
        const p0 = today.items.items[0];
        const p1 = twin.items.items[0];
        const p2 = both.items.items[0];
        try std.testing.expectEqualStrings("_index_score_rows", p0.name);
        try std.testing.expectEqualStrings("_index_score_rows_v", p1.name);
        try std.testing.expect(p1.extra);
        try std.testing.expectEqualStrings("_index_score_rows_v", p2.name);
        try std.testing.expect(p2.extra);
        try std.testing.expectEqual(p0.grid, p1.grid);
        try std.testing.expectEqual(p0.args.len, p1.args.len);
        for (p0.args, p1.args) |q0, q1| try std.testing.expect(std.mem.eql(u8, q0.name, q1.name) and std.meta.eql(q0.value, q1.value));
        for (p1.consts, p2.consts) |q1, q2| try std.testing.expect(std.mem.eql(u8, q1.name, q2.name) and std.meta.eql(q1.int, q2.int));
        // its consts are the baseline, with TREE on at every RB (RB 2's capture has no TREE entry and the twin adds it)
        try std.testing.expectEqual(p0.consts.len + (if (rb == 2) @as(usize, 1) else 0), p1.consts.len);
        for (p0.consts) |q0| {
            var found = false;
            for (p1.consts) |q1| {
                if (std.mem.eql(u8, q0.name, q1.name)) {
                    found = true;
                    try std.testing.expectEqual(q0.int, q1.int);
                }
            }
            try std.testing.expect(found);
        }
        var tree = false;
        for (p1.consts) |q| tree = tree or (std.mem.eql(u8, q.name, "TREE") and q.int == 1);
        try std.testing.expect(tree);
    };
    // decode windows and rowwise calls: _index_score, the baseline, with the switch on
    idx_v = true;
    idx_u = false;
    for ([_]struct { rows: usize, rowwise: bool }{ .{ .rows = 16, .rowwise = false }, .{ .rows = 512, .rowwise = true } }) |c| {
        var log: tri.Log = .{ .gpa = std.testing.allocator };
        defer log.deinit();
        try indexScore(.{ .log = &log }, 0x100, .{ .fp4 = .{ .codes = 0x200, .scales = 0x300 } }, 0x400, 0x500, 40000, 0x600, null, true, 0x700, c.rowwise, c.rows, 32, 128);
        try std.testing.expectEqualStrings("_index_score", log.items.items[0].name);
        try std.testing.expect(!log.items.items[0].extra);
    }
}

test "every _index_score_rows variant of the kit has its twin _index_score_rows_u, compiled the same way" {
    // fixtures/kit-drafter*.json (tools/dsv41-zig/kit/kit_fixture.py): the base set's variants and the extra set's
    var base = try cuda.aot.Manifest.parse(std.testing.allocator, @embedFile("fixtures/kit-drafter.json"));
    defer base.deinit();
    var extra = try cuda.aot.Manifest.parse(std.testing.allocator, @embedFile("fixtures/kit-drafter-extra.json"));
    defer extra.deinit();
    try std.testing.expectEqual([2]usize{ 102, 102 }, base.twinCoverage(&extra, "_index_score_rows", "_index_score_rows_u"));
}

test "index_keys_cand launches _index_score_cand as recorded" {
    try tri.conform("_index_score_cand", struct {
        fn call(t: Tri, c: Case) !void {
            const base: ?u64 = if (inRound(c)) c.ptr("KB") else null;
            try indexKeysCand(t, c.ptr("Q"), caseK(c), c.ptr("Wt"), c.ptr("VIS"), num(c, "n"), c.ptr("CBLK"), c.stride("CBLK", 0), c.dim("CBLK", 1), @intCast(c.constInt("CB")), base, c.ptr("OUT"), c.dim("Q", 0), c.dim("Q", 1), c.dim("Q", 2));
        }
    }.call);
}

test "topk_select (fused) launches _topk_sel as recorded" {
    try tri.conform("_topk_sel", struct {
        fn call(t: Tri, c: Case) !void {
            const top: TopK = .{ .top = 0x7e00_0000_0000, .run = noTopK };
            try topkSelect(t, c.ptr("KEYS"), c.stride("KEYS", 0), c.dim("OUT", 1), c.ptr("VIS"), c.ptr("OUT"), top, c.dim("KEYS", 0), c.dim("KEYS", 1));
        }
    }.call);
}

test "topk_select (torch's top-k) launches _topk_finish as recorded" {
    try tri.conform("_topk_finish", struct {
        fn call(t: Tri, c: Case) !void {
            // the keys are not in the launch (torch's top-k read them): a width past the fused selection's stands in
            const n = 2 * topk_fused_max;
            const top: TopK = .{ .top = c.ptr("TOP"), .run = noTopK };
            try topkSelect(t, 0x7e00_0000_0000, n, c.dim("TOP", 1), c.ptr("VIS"), c.ptr("OUT"), top, c.dim("TOP", 0), n);
        }
    }.call);
}

test "topk_select_pruned launches _gather_tiles as recorded" {
    try tri.conform("_gather_tiles", struct {
        fn call(t: Tri, c: Case) !void {
            const bufs: PrunedBufs = .{ .tpos = c.ptr("TPOS"), .cand = c.ptr("OUT"), .every = 0x7e00_0000_0000 };
            const top: TopK = .{ .top = 0x7e00_0000_0100, .run = noTopK };
            try topkSelectPruned(t, c.ptr("KEYS"), c.stride("KEYS", 0), 0x7e00_0000_0200, c.dim("TPOS", 1), 0x7e00_0000_0300, 0x7e00_0000_0400, bufs, top, c.dim("KEYS", 0), c.dim("KEYS", 1));
        }
    }.call);
}

test "score_keys launches _score_keys as recorded" {
    try tri.conform("_score_keys", struct {
        fn call(t: Tri, c: Case) !void {
            const tmax: ?u64 = if (c.constInt("HAS_TMAX") != 0) c.ptr("TMAX") else null;
            try scoreKeys(t, c.ptr("S"), tmax, c.ptr("OUT"), c.dim("S", 0), c.dim("S", 1));
        }
    }.call);
}
