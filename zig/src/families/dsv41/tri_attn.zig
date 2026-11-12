//! kernels.py's sparse attention and packed FP4 rows: sparse_attn (the split keys and their merge, the merge with the
//! inverse RoPE and wo_a's input rotation, or one pass for prompt chunks), fp4_store and fp4_qd_p2, as Python launches them.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const aot = cuda.aot;
const Tri = tri.Tri;
const u = tri.u;
const p = aot.ptr;

/// kernels.DECODE_ROWS (TF_DS_DECODE_ROWS unset in the served build): calls of up to this many rows split the keys.
pub const decode_rows = 16;
/// kernels.ATTN_SPLITS: the key splits of a decode / verify window's rows.
pub const attn_splits = 8;

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

/// kernels._pow2: n is a power of two.
fn isPow2(n: usize) bool {
    return n > 0 and n & (n - 1) == 0;
}

/// sparse_attn's compressed keys: none, bf16 rows [N, HD], or a packed FP4 pair (codes u8 [N, HD/2], scales u8 [N, HD/16]).
pub const Comp = union(enum) { none, bf16: u64, fp4: struct { codes: u64, scales: u64 } };

/// sparse_attn's rot = (cos, sin, rope dim, wo_a's suh concatenated, its rotated rows' buffer, heads a slice).
pub const Rot = struct { cos: u64, sin: u64, rd: usize, suh: u64, xh: u64, gh: usize };

/// A split call's partials (rows <= decode_rows; Python allocates them): pm, pl fp32 [R * H * 8], po fp32 [R * H * 8 * HD].
pub const Parts = struct { pm: u64, pl: u64, po: u64 };

/// sparse_attn's arguments, Python's keyword defaults as field defaults.
pub const Attn = struct {
    /// q [rows, h, hd] bf16; out the same shape (Python: empty_like(q) when not given)
    q: u64,
    out: u64,
    rows: usize,
    h: usize,
    hd: usize,
    /// sink fp32 [h]
    sink: u64,
    /// the window keys bf16 [wsrc_rows, hd]: a ring (slot = position % ring size) or rows from position wlo[0] (int64 [1])
    wsrc: u64,
    wsrc_rows: usize,
    wlo: u64,
    ring: bool,
    comp: Comp = .none,
    /// int64 [rows, n_idx] (-1 = none; n_idx 0: no idx); prompt chunks read rows of n_idx rounded up to 16, -1 past n_idx
    idx: u64 = 0,
    n_idx: usize = 0,
    /// pos int64 [rows]
    pos: u64,
    /// Python's float as Triton passes it: rounded to fp32 (hd ** -0.5: 0x3d3504f3)
    scale: f32,
    window: usize,
    /// per-row ring / compressed row bases, int64 [rows] (concurrent rounds and batched drafts: rings only)
    wbase: ?u64 = null,
    cbase: ?u64 = null,
    /// a pool's ring size (0: none, wsrc's rows)
    ring_rows: usize = 0,
    rot: ?Rot = null,
    /// unused by prompt chunks (rows > decode_rows): one split writes out directly
    parts: Parts = .{ .pm = 0, .pl = 0, .po = 0 },
    /// a small prompt chunk run as a large one (one split, out directly: the prompt chunks' variant)
    one_split: bool = false,
    /// TF_DS_ATTN_U (a prompt chunk's call, `attn_u`) or TF_DS_ATTN_U_DECODE (a decode round's, `attn_u_decode`): with
    /// packed latents the part from its twin _sparse_attn_part_u (Tri.runTwin: the hardware unpack, the same keys), else
    /// from _sparse_attn_part
    u: bool = false,
};

/// TF_DS_ATTN_U in effect (Model.open): a prompt chunk's sparse attention asks for _sparse_attn_part_u.
pub var attn_u: bool = false;
/// TF_DS_ATTN_U_DECODE in effect (Model.open): a decode round's sparse attention asks for _sparse_attn_part_u (the
/// drafter's passes read bf16 compressed keys: nothing to unpack).
pub var attn_u_decode: bool = false;

/// sparse_attn's key splits for a call of `rows` rows: a decode window's (decode_rows rows or fewer) split the keys and
/// merge them; a prompt chunk, and a small chunk run as one (`one_split`), take one split that writes out directly.
pub fn splits(rows: usize, one_split: bool) usize {
    return if (rows <= tri.decode_rows and !one_split) attn_splits else 1;
}

/// sparse_attn: q -> out (o [R, H, HD]); window keys from wsrc, compressed keys comp[idx[r, j]].
pub fn sparseAttn(t: Tri, a: Attn) !void {
    const hb = 16;
    const bn = 32;
    const has = a.comp != .none and a.n_idx > 0;
    const pack = has and a.comp == .fp4;
    const codes: u64 = if (!has) a.wsrc else switch (a.comp) {
        .none => unreachable,
        .bf16 => |k| k,
        .fp4 => |k| k.codes,
    };
    const csc: u64 = if (pack) a.comp.fp4.scales else a.wsrc;
    const ct: []const u8 = if (pack) "*u8" else "*bf16";
    var n_idx: usize = if (has) a.n_idx else 0;
    const sp: usize = splits(a.rows, a.one_split);
    const groups = a.h / hb;
    const final = sp == 1;
    var picks: usize = if (has) tri.cdiv(n_idx, bn) else 0;
    if (final and picks > 0) {
        // prompt chunks: the pick list padded with -1 to a multiple of 16 entries, the pick blocks to a power of two
        n_idx = (n_idx + 15) / 16 * 16;
        picks = tri.pow2(picks);
    }
    const nblk = a.window / bn + picks;
    const ring_size: usize = if (a.ring) (if (a.ring_rows != 0) a.ring_rows else a.wsrc_rows) else 16;
    const pp: Parts = if (final) .{ .pm = a.out, .pl = a.out, .po = a.out } else a.parts;
    const pt: []const u8 = if (final) "*bf16" else "*fp32";
    const based = a.wbase != null;
    if (based and !a.ring) return error.BasesNeedRing;
    const rbase = a.wbase orelse a.pos;
    const cbase = if (based) a.cbase orelse a.wbase.? else a.pos;
    // switch "attn_pf" (on in the served build): a split's window keys into L2 before its PDL wait
    const wpf = a.ring and !final;
    const part_args = [_]aot.Arg{
        p("Q", "*bf16", a.q),              p("WSRC", "*bf16", a.wsrc),                  p("WLO", "*i64", a.wlo),    p("COMP", ct, codes),
        p("CSC", ct, csc),                 p("IDX", "*i64", if (has) a.idx else a.pos), p("POS", "*i64", a.pos),    p("PM", pt, pp.pm),
        p("PL", pt, pp.pl),                p("PO", pt, pp.po),                          p("SINK", "*fp32", a.sink), p("OUT", "*bf16", a.out),
        aot.float("scale", a.scale),       p("RBASE", "*i64", rbase),                   p("CBASE", "*i64", cbase),  aot.int("ring_size", @intCast(ring_size)),
        aot.int("n_idx", @intCast(n_idx)),
    };
    const part_consts = [_]aot.Const{
        ci("H", a.h),       ci("HD", a.hd),        ci("HB", hb),       ci("WIN", a.window), ci("BN", bn),
        cb("RING", a.ring), cb("HAS_COMP", has),   cb("PACKED", pack), ci("SPLITS", sp),    ci("NBLK", nblk),
        cb("FINAL", final), cb("HAS_BASE", based), t.pdlConst(),       cb("WPF", wpf),
    };
    const grid: [3]u32 = .{ u(a.rows), u(groups), u(sp) };
    if (a.u and pack and !a.extra) {
    } else try (if (a.extra) &Tri.runExtra else &Tri.run)(t, "_sparse_attn_part", grid, &part_args, &part_consts);
    if (final) return;
    const merged = [_]aot.Arg{ p("PM", "*fp32", pp.pm), p("PL", "*fp32", pp.pl), p("PO", "*fp32", pp.po), p("SINK", "*fp32", a.sink), p("OUT", "*bf16", a.out) };
    if (a.rot) |r| {
        // switch "merge_reg" (on in the served build): the head's row kept in registers when HD is a power of two
        const merge: []const u8 = if (isPow2(a.hd)) "_sparse_attn_merge_rot2" else "_sparse_attn_merge_rot";
        // TF_EXL3_L2_DISCARD unset in the served build (on): the merge drops the split outputs it read from L2
        const discard = a.hd % 32 == 0 and isPow2(sp * (a.hd / 32));
        const args = merged ++ [_]aot.Arg{ p("COS", "*fp32", r.cos), p("SIN", "*fp32", r.sin), p("POS", "*i64", a.pos), p("SUH", "*fp16", r.suh), p("XH", "*fp16", r.xh), aot.int("rows", @intCast(a.rows)) };
        try t.run(merge, .{ u(a.rows), u(a.h), 1 }, &args, &.{
            ci("H", a.h), ci("HD", a.hd), ci("HB", hb), ci("SPLITS", sp), ci("RD", r.rd), ci("GH", r.gh), cb("DISCARD", discard), t.pdlConst(),
        });
    } else {
        try t.run("_sparse_attn_merge", .{ u(a.rows), u(groups), 1 }, &merged, &.{ ci("H", a.h), ci("HD", a.hd), ci("HB", hb), ci("SPLITS", sp), t.pdlConst() });
    }
}

/// fp4_store: model.store_rows(cache, rows, x, block, e4m3) of a packed cache (codes, scales); x [n, d] bf16, row stride xs.
pub fn fp4Store(t: Tri, x: u64, xs: usize, codes: u64, scales: u64, rows: u64, n: usize, d: usize, block: usize, e4m3: bool) !void {
    try t.run("_fp4_store", .{ u(n), 1, 1 }, &.{ p("X", "*bf16", x), aot.int("xs", @intCast(xs)), p("CODES", "*u8", codes), p("SCALES", "*u8", scales), p("ROWS", "*i64", rows) }, &.{ ci("D", d), ci("BLOCK", block), cb("E4M3", e4m3), t.pdlConst() });
}

/// fp4_qd_p2: ops.fp4_qd(x, 32, e4m3_scale=False) of bf16 x (numel elements, a multiple of 32) into out, x's shape.
pub fn fp4QdP2(t: Tri, x: u64, out: u64, numel: usize) !void {
    const nblk = numel / 32;
    const blks = 32;
    try t.run("_fp4qd_p2", .{ tri.cdiv(nblk, blks), 1, 1 }, &.{ p("X", "*bf16", x), p("OUT", "*bf16", out), aot.int("nblk", @intCast(nblk)) }, &.{ ci("BLKS", blks), t.pdlConst() });
}

// -- conformance with the recorded launches ----------------------------------------------------------------------

const Case = tri.Case;

fn dtype(c: Case, tensor: []const u8) []const u8 {
    return c.v.get("tensors").?.object.get(tensor).?.array.items[0].string;
}

test "sparse_attn's split keys and prompt chunks launch as recorded" {
    try tri.conform("_sparse_attn_part", struct {
        fn call(t: Tri, c: Case) !void {
            // a pick list with keys was given when IDX is [R, n]; else the kernel reads pos there
            const has = c.rank("IDX") == 2;
            const fp4 = std.mem.eql(u8, dtype(c, "COMP"), "uint8");
            const comp: Comp = if (!has) .none else if (fp4) .{ .fp4 = .{ .codes = c.ptr("COMP"), .scales = c.ptr("CSC") } } else .{ .bf16 = c.ptr("COMP") };
            const ring = c.constInt("RING") != 0;
            // wbase given or not: no tensor tells (RBASE and CBASE are pos otherwise), so HAS_BASE from the case
            const based = c.constInt("HAS_BASE") != 0;
            // the caller's ring_rows (a pool's ring size, not wsrc's rows): the recorded ring_size where it differs from them
            const rs: usize = @intCast(c.int("ring_size"));
            try sparseAttn(t, .{
                .q = c.ptr("Q"),
                .out = c.ptr("OUT"),
                .rows = c.dim("Q", 0),
                .h = c.dim("Q", 1),
                .hd = c.dim("Q", 2),
                .sink = c.ptr("SINK"),
                .wsrc = c.ptr("WSRC"),
                .wsrc_rows = c.dim("WSRC", 0),
                .wlo = c.ptr("WLO"),
                .ring = ring,
                .comp = comp,
                .idx = if (has) c.ptr("IDX") else 0,
                .n_idx = if (has) c.dim("IDX", 1) else 0,
                .pos = c.ptr("POS"),
                .scale = c.float("scale"),
                .window = @intCast(c.constInt("WIN")),
                .wbase = if (based) c.ptr("RBASE") else null,
                .cbase = if (based) c.ptr("CBASE") else null,
                .ring_rows = if (ring and rs != c.dim("WSRC", 0)) rs else 0,
                .parts = .{ .pm = c.ptr("PM"), .pl = c.ptr("PL"), .po = c.ptr("PO") },
            });
        }
    }.call);
}

/// sparse_attn's arguments for a recorded merge: the merge's from the case, the split pass's own (not recorded) fixed.
fn mergeArgs(c: Case, rot: ?Rot) Attn {
    return .{
        .q = 0x7e00_0000_0000,
        .out = c.ptr("OUT"),
        .rows = c.dim("OUT", 0),
        .h = c.dim("OUT", 1),
        .hd = c.dim("OUT", 2),
        .sink = c.ptr("SINK"),
        .wsrc = 0x7e00_0000_0100,
        .wsrc_rows = 576,
        .wlo = 0x7e00_0000_0200,
        .ring = true,
        .pos = if (c.has("POS")) c.ptr("POS") else 0x7e00_0000_0300,
        .scale = 0.044194173,
        .window = 128,
        .ring_rows = 144,
        .rot = rot,
        .parts = .{ .pm = c.ptr("PM"), .pl = c.ptr("PL"), .po = c.ptr("PO") },
    };
}

test "sparse_attn's merges launch as recorded" {
    try tri.conform("_sparse_attn_merge", struct {
        fn call(t: Tri, c: Case) !void {
            try sparseAttn(t, mergeArgs(c, null));
        }
    }.call);
    try tri.conform("_sparse_attn_merge_rot2", struct {
        fn call(t: Tri, c: Case) !void {
            const rot: Rot = .{ .cos = c.ptr("COS"), .sin = c.ptr("SIN"), .rd = @intCast(c.constInt("RD")), .suh = c.ptr("SUH"), .xh = c.ptr("XH"), .gh = @intCast(c.constInt("GH")) };
            try sparseAttn(t, mergeArgs(c, rot));
        }
    }.call);
}

test "a prompt chunk's pick list is padded to 16 entries and its pick blocks to a power of two" {
    var log: tri.Log = .{ .gpa = std.testing.allocator };
    defer log.deinit();
    const comp: Comp = .{ .fp4 = .{ .codes = 0x600, .scales = 0x700 } };
    try sparseAttn(.{ .log = &log }, .{ .q = 0x100, .out = 0x200, .rows = 17, .h = 32, .hd = 512, .sink = 0x300, .wsrc = 0x400, .wsrc_rows = 17, .wlo = 0x500, .ring = false, .comp = comp, .idx = 0x800, .n_idx = 70, .pos = 0x900, .scale = 0.044194173, .window = 128 });
    try std.testing.expectEqual(@as(usize, 1), log.items.items.len);
    const k = log.items.items[0];
    const n_idx = for (k.args) |x| {
        if (std.mem.eql(u8, x.name, "n_idx")) break x.value.i32;
    } else return error.TestUnexpectedResult;
    const nblk = for (k.consts) |x| {
        if (std.mem.eql(u8, x.name, "NBLK")) break x.int;
    } else return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(i32, 80), n_idx);
    try std.testing.expectEqual(@as(?i64, 128 / 32 + 4), nblk);
}

test "a decode window's rows split the keys; a prompt chunk, and a small chunk run as one, take one split" {
    try std.testing.expectEqual(@as(usize, attn_splits), splits(16, false));
    try std.testing.expectEqual(@as(usize, attn_splits), splits(1, false));
    try std.testing.expectEqual(@as(usize, 1), splits(16, true));
    try std.testing.expectEqual(@as(usize, 1), splits(1, true));
    try std.testing.expectEqual(@as(usize, 1), splits(17, false));
    try std.testing.expectEqual(@as(usize, 1), splits(17, true));
}

test "a small chunk run as one split launches the prompt chunks' variant: one pass, its pick list padded, no merge" {
    const comp: Comp = .{ .fp4 = .{ .codes = 0x600, .scales = 0x700 } };
    const a: Attn = .{ .q = 0x100, .out = 0x200, .rows = 16, .h = 32, .hd = 512, .sink = 0x300, .wsrc = 0x400, .wsrc_rows = 16, .wlo = 0x500, .ring = false, .comp = comp, .idx = 0x800, .n_idx = 70, .pos = 0x900, .scale = 0.044194173, .window = 128, .one_split = true };
    for ([_]bool{ true, false }) |one| {
        var log: tri.Log = .{ .gpa = std.testing.allocator };
        defer log.deinit();
        var b = a;
        b.one_split = one;
        try sparseAttn(.{ .log = &log }, b);
        // one split: the part kernel alone, FINAL; split keys: the part kernel, then the merge
        try std.testing.expectEqual(@as(usize, if (one) 1 else 2), log.items.items.len);
        const k = log.items.items[0];
        try std.testing.expectEqualStrings("_sparse_attn_part", k.name);
        const n_idx = for (k.args) |x| {
            if (std.mem.eql(u8, x.name, "n_idx")) break x.value.i32;
        } else return error.TestUnexpectedResult;
        var got: [2]?i64 = .{ null, null };
        for (k.consts) |x| {
            if (std.mem.eql(u8, x.name, "SPLITS")) got[0] = x.int;
            if (std.mem.eql(u8, x.name, "FINAL")) got[1] = x.int;
        }
        try std.testing.expectEqual(@as(?i64, if (one) 1 else attn_splits), got[0]);
        try std.testing.expectEqual(@as(?i64, @intFromBool(one)), got[1]);
        try std.testing.expectEqual(@as(i32, if (one) 80 else 70), n_idx);
        try std.testing.expectEqual(@as(u32, 16), k.grid[0]);
    }
}

test "TF_DS_ATTN_U: a call with packed latents launches the part's twin with its arguments; else the part, as the baseline" {
    const fp4: Comp = .{ .fp4 = .{ .codes = 0x600, .scales = 0x700 } };
    const a: Attn = .{ .q = 0x100, .out = 0x200, .rows = 2048, .h = 16, .hd = 512, .sink = 0x300, .wsrc = 0x400, .wsrc_rows = 2048, .wlo = 0x500, .ring = false, .comp = fp4, .idx = 0x800, .n_idx = 512, .pos = 0x900, .scale = 0.044194173, .window = 128 };
    // a prompt chunk (one split, FINAL) and a small chunk on the ring (8 splits, then the merge)
    for ([_]Attn{ a, blk: {
        var b = a;
        b.rows = 6;
        b.ring = true;
        b.parts = .{ .pm = 0xa00, .pl = 0xb00, .po = 0xc00 };
        break :blk b;
    } }) |x| {
        var today: tri.Log = .{ .gpa = std.testing.allocator };
        defer today.deinit();
        try sparseAttn(.{ .log = &today }, x);
        var twin: tri.Log = .{ .gpa = std.testing.allocator };
        defer twin.deinit();
        var xu = x;
        xu.u = true;
        try sparseAttn(.{ .log = &twin }, xu);
        try std.testing.expectEqual(today.items.items.len, twin.items.items.len);
        for (today.items.items, twin.items.items, 0..) |p0, p1, i| {
            // the part: its twin from the extra set, the same arguments and constexprs; the merge the baseline
            try std.testing.expectEqualStrings(if (i == 0) "_sparse_attn_part_u" else p0.name, p1.name);
            try std.testing.expectEqual(i == 0, p1.extra);
            try std.testing.expectEqual(p0.grid, p1.grid);
            try std.testing.expect(std.meta.eql(p0.args.len, p1.args.len) and std.meta.eql(p0.consts.len, p1.consts.len));
            for (p0.args, p1.args) |q0, q1| try std.testing.expect(std.mem.eql(u8, q0.name, q1.name) and std.meta.eql(q0.value, q1.value));
            for (p0.consts, p1.consts) |q0, q1| try std.testing.expect(std.mem.eql(u8, q0.name, q1.name) and std.meta.eql(q0.int, q1.int));
        }
    }
    // keys without packed latents (bf16, or none): the part itself
    for ([_]Comp{ .{ .bf16 = 0x600 }, .none }) |cmp| {
        var log: tri.Log = .{ .gpa = std.testing.allocator };
        defer log.deinit();
        var b = a;
        b.comp = cmp;
        b.u = true;
        try sparseAttn(.{ .log = &log }, b);
        try std.testing.expectEqualStrings("_sparse_attn_part", log.items.items[0].name);
        try std.testing.expect(!log.items.items[0].extra);
    }
}

test "every _sparse_attn_part variant of the kit has its twin _sparse_attn_part_u, compiled the same way" {
    // fixtures/kit-drafter*.json (tools/dsv41-zig/kit/kit_fixture.py): the base set's variants and the extra set's
    var base = try cuda.aot.Manifest.parse(std.testing.allocator, @embedFile("fixtures/kit-drafter.json"));
    defer base.deinit();
    var extra = try cuda.aot.Manifest.parse(std.testing.allocator, @embedFile("fixtures/kit-drafter-extra.json"));
    defer extra.deinit();
    try std.testing.expectEqual([2]usize{ 36, 36 }, base.twinCoverage(&extra, "_sparse_attn_part", "_sparse_attn_part_u"));
    // (and the four-way drafter's two, in the extra set itself)
    try std.testing.expectEqual([2]usize{ 2, 2 }, extra.twinCoverage(&extra, "_sparse_attn_part", "_sparse_attn_part_u"));
}

test "fp4_store launches as recorded" {
    try tri.conform("_fp4_store", struct {
        fn call(t: Tri, c: Case) !void {
            try fp4Store(t, c.ptr("X"), c.stride("X", 0), c.ptr("CODES"), c.ptr("SCALES"), c.ptr("ROWS"), c.dim("X", 0), c.dim("X", 1), @intCast(c.constInt("BLOCK")), c.constInt("E4M3") != 0);
        }
    }.call);
}

test "fp4_qd_p2 launches as recorded" {
    try tri.conform("_fp4qd_p2", struct {
        fn call(t: Tri, c: Case) !void {
            var numel: usize = 1;
            for (0..c.rank("X")) |i| numel *= c.dim("X", i);
            try fp4QdP2(t, c.ptr("X"), c.ptr("OUT"), numel);
        }
    }.call);
}
