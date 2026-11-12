//! A test driver's logits-log (tf-dsv41-lanes --logits-log): one record per scored row of a stream the driver forced
//! onto reference tokens (the lane core's Stream.force): the reference ("label") token's logit and rank, the row's
//! logsumexp and its top-K ids and logits, little-endian after a header. xp/qgate.py reads two logs (a base and an arm)
//! and joins them on (request, position). A row predicts the token at its position (a round's row: the window's
//! position; the prompt's last row: the prompt's length); its label is the stream's reference token there.
const std = @import("std");

pub const magic = "DSQG".*;
/// 4: version 1's 64-byte header and records, then (after the header) HeaderTail: kern2, the speed word and the settings hash
/// (3 had the first two; qgate.py's version 2 is a different, unwritten layout: a stream table and replay fields in the records).
pub const version: u32 = 4;
pub const no_label: u32 = 0xFFFF_FFFF;

/// A record's flags: its row's sha256 follows its pairs; its whole row (vocab fp32) follows that.
pub const flag_sha: u16 = 1;
pub const flag_row: u16 = 2;

/// The log's header: what a reader checks before it joins two logs.
pub const Header = extern struct {
    magic: [4]u8 = magic,
    version: u32 = version,
    vocab: u32,
    topk: u32,
    full_every: u32, // every Nth record carries its whole row (0: none)
    flags: u32, // bit 0: every record carries its row's sha256
    kern: u32, // the start-up settings' kernel switches (prompt2d.kernAsked)
    exch: u32, // and exchange switches (prompt2d.exchAsked)
    requests_sha: [32]u8, // sha256 of the requests file
};

comptime {
    if (@sizeOf(Header) != 64) @compileError("the logits-log header is 64 bytes");
}

/// What follows the 64 bytes in version 3: the second kernel word and the decode defaults' state.
pub const HeaderTail = extern struct {
    kern2: u32, // the kernel switches beyond kern's 32 bits (prompt2d.kern2Asked): HC_2S, DRAFT_VOCAB
    speed: u32, // speedWord: the state IN EFFECT (not asked) of the decode defaults that carry no kernel-word bit
    settings_hash: u64, // settingsHash of the settings text the run wrote beside the log (<log>.settings): the engine's own start report
};

comptime {
    if (@sizeOf(HeaderTail) != 16) @compileError("the logits-log header tail is 16 bytes");
}

/// 64 bits of the settings text (the first 8 bytes of its SHA-256, little-endian): a log without its sidecar still shows
/// whether two runs' settings differed.
pub fn settingsHash(text: []const u8) u64 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
    return std.mem.readInt(u64, d[0..8], .little);
}

/// The decode defaults that carry no kernel-word bit, as the run has them in effect (a switch the kit or a rank could not
/// run is off here, whatever was asked).
pub const Speed = struct {
    twin: u2, // TF_DS_ATTN_TWIN: 0 u, 1 w, 2 x, 3 z (tri_attn.Twin's order)
    rope_w: bool, // TF_DS_ROPE_W
    merge_cols: usize, // TF_DS_MERGE_D in effect: the columns a program (128, 64, 16) or 0 (off)
    group2: bool, // TF_DS_DEC_GROUP2
    expert_exit: bool, // TF_DS_EXPERT_EXIT
    pack_epilogue: bool, // TF_DS_PACK_EPILOGUE
    hc_2s: bool, // TF_DS_HC_2S
    qkv_plain: bool, // TF_DS_QKV_PLAIN: a decode window's q norm and window KV in one launch (round.qkv_plain)
};

/// The speed word's bits: 0-1 the twin, 2 ROPE_W, 3-4 MERGE_D (0 off, 1 = 128 columns, 2 = 64, 3 = 16), 5 DEC_GROUP2,
/// 6 EXPERT_EXIT, 7 PACK_EPILOGUE, 8 HC_2S, 9 QKV_PLAIN. (xp/qgate.py speed_text reads the same.)
/// The speed word keeps the decode defaults that were first; every other switch, value and future one is in the settings
/// sidecar (the engine's own start report) whose hash the header carries. model.zig's test pins that the report names every
/// default-on row.
pub fn speedWord(s: Speed) u32 {
    var x: u32 = s.twin;
    if (s.rope_w) x |= 1 << 2;
    x |= @as(u32, switch (s.merge_cols) {
        128 => 1,
        64 => 2,
        16 => 3,
        else => 0,
    }) << 3;
    if (s.group2) x |= 1 << 5;
    if (s.expert_exit) x |= 1 << 6;
    if (s.pack_epilogue) x |= 1 << 7;
    if (s.hc_2s) x |= 1 << 8;
    if (s.qkv_plain) x |= 1 << 9;
    return x;
}

/// The switches the speed word carries, in bit order of their first bit.
pub const speed_names = [_][]const u8{ "TF_DS_ATTN_TWIN", "TF_DS_ROPE_W", "TF_DS_MERGE_D", "TF_DS_DEC_GROUP2", "TF_DS_EXPERT_EXIT", "TF_DS_PACK_EPILOGUE", "TF_DS_HC_2S", "TF_DS_QKV_PLAIN" };

/// Bytes before the first record.
pub const header_bytes = @sizeOf(Header) + @sizeOf(HeaderTail);

/// The fixed part of a record before its pairs.
pub const prefix_bytes = 36;

/// A record's fixed fields.
pub const Record = struct {
    request: u32,
    position: u32,
    input: u32,
    label: u32 = no_label,
    label_logit: f32 = std.math.nan(f32),
    label_rank: u32 = no_label,
    lse: f64,
    k: u16,
    flags: u16 = 0,
};

/// Ordering key: NaN as -inf (never above a number).
fn key(x: f32) f32 {
    return if (std.math.isNan(x)) -std.math.inf(f32) else x;
}

/// The min(ids.len, row.len) largest values of `row` into ids and vals, in descending order, equal values by ascending
/// id (the engine's greedy pick, the first largest, is ids[0]); returns how many.
pub fn topK(row: []const f32, ids: []u32, vals: []f32) usize {
    const n = @min(ids.len, row.len);
    if (n == 0) return 0;
    for (0..n) |i| {
        ids[i] = @intCast(i);
        vals[i] = row[i];
    }
    // the kept value that would go first: the least, the last id among equal ones
    const worst = struct {
        fn of(v: []const f32, d: []const u32) usize {
            var w: usize = 0;
            for (v, d, 0..) |x, id, i| {
                if (key(x) < key(v[w]) or (key(x) == key(v[w]) and id > d[w])) w = i;
            }
            return w;
        }
    }.of;
    var w = worst(vals[0..n], ids[0..n]);
    // a later id ranks below an equal kept one, so only a greater value replaces
    for (row[n..], n..) |x, i| if (key(x) > key(vals[w])) {
        ids[w] = @intCast(i);
        vals[w] = x;
        w = worst(vals[0..n], ids[0..n]);
    };
    const Pair = struct { id: u32, v: f32 };
    var pairs: [1024]Pair = undefined;
    if (n <= pairs.len) {
        for (pairs[0..n], ids[0..n], vals[0..n]) |*p, id, v| p.* = .{ .id = id, .v = v };
        std.sort.pdq(Pair, pairs[0..n], {}, struct {
            fn less(_: void, x: Pair, y: Pair) bool {
                return key(x.v) > key(y.v) or (key(x.v) == key(y.v) and x.id < y.id);
            }
        }.less);
        for (pairs[0..n], ids[0..n], vals[0..n]) |p, *id, *v| {
            id.* = p.id;
            v.* = p.v;
        }
        return n;
    }
    // past 1,024 kept values: an insertion sort in place (slow, the tests' and a full row's case alone)
    var i: usize = 1;
    while (i < n) : (i += 1) {
        var j = i;
        while (j > 0 and (key(vals[j]) > key(vals[j - 1]) or (key(vals[j]) == key(vals[j - 1]) and ids[j] < ids[j - 1]))) : (j -= 1) {
            std.mem.swap(f32, &vals[j], &vals[j - 1]);
            std.mem.swap(u32, &ids[j], &ids[j - 1]);
        }
    }
    return n;
}

/// log(sum(exp(row))) in f64, the row's largest value subtracted first; -inf for a row of -inf alone.
pub fn logsumexp(row: []const f32) f64 {
    var m: f64 = -std.math.inf(f64);
    for (row) |x| m = @max(m, @as(f64, key(x)));
    if (m == -std.math.inf(f64)) return m;
    var sum: f64 = 0;
    for (row) |x| sum += @exp(@as(f64, key(x)) - m);
    return m + @log(sum);
}

/// The label's rank in its row: the values above it, and the equal ones before it (rank 0: the row's first largest).
pub fn rank(row: []const f32, label: u32) u32 {
    const v = key(row[label]);
    var n: u32 = 0;
    for (row, 0..) |x, i| n += @intFromBool(key(x) > v or (key(x) == v and i < label));
    return n;
}

/// A record's fixed fields and its pairs into `buf` (prefix_bytes + 8 k bytes); returns the bytes written.
pub fn encode(buf: []u8, r: Record, ids: []const u32, vals: []const f32) usize {
    std.mem.writeInt(u32, buf[0..4], r.request, .little);
    std.mem.writeInt(u32, buf[4..8], r.position, .little);
    std.mem.writeInt(u32, buf[8..12], r.input, .little);
    std.mem.writeInt(u32, buf[12..16], r.label, .little);
    std.mem.writeInt(u32, buf[16..20], @bitCast(r.label_logit), .little);
    std.mem.writeInt(u32, buf[20..24], r.label_rank, .little);
    std.mem.writeInt(u64, buf[24..32], @bitCast(r.lse), .little);
    std.mem.writeInt(u16, buf[32..34], r.k, .little);
    std.mem.writeInt(u16, buf[34..36], r.flags, .little);
    var off: usize = prefix_bytes;
    for (ids[0..r.k], vals[0..r.k]) |id, v| {
        std.mem.writeInt(u32, buf[off..][0..4], id, .little);
        std.mem.writeInt(u32, buf[off + 4 ..][0..4], @bitCast(v), .little);
        off += 8;
    }
    return off;
}

/// What the driver registered for a stream: its request's index, its prompt's length and its reference tokens.
pub const Reg = struct { request: u32, prompt_len: u32, ref: []const u32 };

/// The label of a row predicting the token at `position` of a registered stream: its reference token there, or none
/// (a row past the reference).
pub fn labelOf(reg: Reg, position: u64) ?u32 {
    if (position < reg.prompt_len) return null;
    const i = position - reg.prompt_len;
    return if (i < reg.ref.len) reg.ref[@intCast(i)] else null;
}

/// The records of the registered streams' rows, written to `w`.
pub const Sink = struct {
    comptime {
        // the full rows go out as the host's bytes: little-endian hosts only (aarch64, x86-64)
        if (@import("builtin").cpu.arch.endian() != .little) @compileError("the logits-log writes little-endian rows");
    }
    gpa: std.mem.Allocator,
    w: *std.Io.Writer,
    vocab: u32,
    topk: u32,
    full_every: u32,
    row_sha: bool,
    regs: std.AutoHashMapUnmanaged(*const anyopaque, Reg) = .empty,
    ids: []u32,
    vals: []f32,
    buf: []u8,
    records: u64 = 0, // records written
    unlabeled: u64 = 0, // rows of registered streams past their reference
    bytes: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, w: *std.Io.Writer, vocab: u32, topk: u32, full_every: u32, row_sha: bool) !Sink {
        const k: usize = @min(topk, vocab);
        const ids = try gpa.alloc(u32, k);
        errdefer gpa.free(ids);
        const vals = try gpa.alloc(f32, k);
        errdefer gpa.free(vals);
        const buf = try gpa.alloc(u8, prefix_bytes + 8 * k);
        return .{ .gpa = gpa, .w = w, .vocab = vocab, .topk = @intCast(k), .full_every = full_every, .row_sha = row_sha, .ids = ids, .vals = vals, .buf = buf };
    }

    pub fn deinit(s: *Sink) void {
        s.regs.deinit(s.gpa);
        s.gpa.free(s.ids);
        s.gpa.free(s.vals);
        s.gpa.free(s.buf);
    }

    /// The header, first: the requests file's sha256 and the switch words the run's start-up settings carry.
    pub fn header(s: *Sink, requests_sha: [32]u8, kern: u32, exch: u32, kern2: u32, speed: u32, settings_hash: u64) !void {
        const h: Header = .{ .vocab = s.vocab, .topk = s.topk, .full_every = s.full_every, .flags = @intFromBool(s.row_sha), .kern = kern, .exch = exch, .requests_sha = requests_sha };
        const t: HeaderTail = .{ .kern2 = kern2, .speed = speed, .settings_hash = settings_hash };
        try s.w.writeAll(std.mem.asBytes(&h));
        try s.w.writeAll(std.mem.asBytes(&t));
        s.bytes += header_bytes;
    }

    pub fn register(s: *Sink, stream: *const anyopaque, reg: Reg) !void {
        try s.regs.put(s.gpa, stream, reg);
    }

    /// A row of logits (vocab fp32) of a stream, predicting the token at `position` after the row's own token `input`:
    /// its record when the stream is registered and its reference has a token there.
    pub fn row(s: *Sink, stream: *const anyopaque, position: u64, input: u32, logits: []const f32) !void {
        const reg = s.regs.get(stream) orelse return;
        const label = labelOf(reg, position) orelse {
            s.unlabeled += 1;
            return;
        };
        if (logits.len != s.vocab or label >= s.vocab) return error.BadLogitsRow;
        const k = topK(logits, s.ids, s.vals);
        const full = s.full_every > 0 and s.records % s.full_every == s.full_every - 1;
        const flags: u16 = (if (s.row_sha) flag_sha else 0) | (if (full) flag_row else 0);
        const r: Record = .{ .request = reg.request, .position = @intCast(position), .input = input, .label = label, .label_logit = logits[label], .label_rank = rank(logits, label), .lse = logsumexp(logits), .k = @intCast(k), .flags = flags };
        const n = encode(s.buf, r, s.ids[0..k], s.vals[0..k]);
        try s.w.writeAll(s.buf[0..n]);
        s.bytes += n;
        if (s.row_sha) {
            var dg: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(logits), &dg, .{});
            try s.w.writeAll(&dg);
            s.bytes += 32;
        }
        if (full) {
            try s.w.writeAll(std.mem.sliceAsBytes(logits));
            s.bytes += logits.len * 4;
        }
        s.records += 1;
    }
};

// -- tests ----------------------------------------------------------------------------------------------------------

const testing = std.testing;
const inf = std.math.inf(f32);

/// The test's naive top-k: every index sorted by (value descending, id ascending).
fn naiveTop(a: std.mem.Allocator, row: []const f32) ![]u32 {
    const all = try a.alloc(u32, row.len);
    for (all, 0..) |*x, i| x.* = @intCast(i);
    const Ctx = struct {
        r: []const f32,
        fn less(c: @This(), x: u32, y: u32) bool {
            const kx = key(c.r[x]);
            const ky = key(c.r[y]);
            return kx > ky or (kx == ky and x < y);
        }
    };
    std.sort.pdq(u32, all, Ctx{ .r = row }, Ctx.less);
    return all;
}

test "topK: the largest values in descending order, equal ones by ascending id, the whole row when K is past it" {
    const a = testing.allocator;
    const row = [_]f32{ 1.5, -inf, 3.0, 3.0, 0.25, -2.0, 3.0, 1.5, -inf, 0.0 };
    const want = try naiveTop(a, &row);
    defer a.free(want);
    for ([_]usize{ 1, 2, 3, 4, 7, 10, 16 }) |k| {
        var ids: [16]u32 = undefined;
        var vals: [16]f32 = undefined;
        const n = topK(&row, ids[0..k], vals[0..k]);
        try testing.expectEqual(@min(k, row.len), n);
        try testing.expectEqualSlices(u32, want[0..n], ids[0..n]);
        for (ids[0..n], vals[0..n]) |id, v| try testing.expectEqual(row[id], v);
    }
    // the ties: ids 2, 3 and 6 hold 3.0, so the first three are 2, 3, 6; a NaN never outranks a number
    var ids: [3]u32 = undefined;
    var vals: [3]f32 = undefined;
    _ = topK(&row, &ids, &vals);
    try testing.expectEqualSlices(u32, &.{ 2, 3, 6 }, &ids);
    const nanrow = [_]f32{ std.math.nan(f32), -1.0, 2.0 };
    _ = topK(&nanrow, ids[0..2], vals[0..2]);
    try testing.expectEqualSlices(u32, &.{ 2, 1 }, ids[0..2]);
    // a long random row against the naive sort
    var prng = std.Random.DefaultPrng.init(7);
    const big = try a.alloc(f32, 5000);
    defer a.free(big);
    for (big) |*x| x.* = @floor(prng.random().floatNorm(f32) * 8) / 4; // many equal values
    const wb = try naiveTop(a, big);
    defer a.free(wb);
    var bi: [128]u32 = undefined;
    var bv: [128]f32 = undefined;
    _ = topK(big, &bi, &bv);
    try testing.expectEqualSlices(u32, wb[0..128], &bi);
}

test "logsumexp: in f64, the largest value subtracted first; a row of -inf alone gives -inf" {
    const row = [_]f32{ 1.0, 2.0, -inf, 0.5 };
    const want = @log(@exp(@as(f64, 1.0)) + @exp(@as(f64, 2.0)) + @exp(@as(f64, 0.5)));
    try testing.expectApproxEqAbs(want, logsumexp(&row), 1e-12);
    // large values do not overflow: 1000 twice is 1000 + ln 2
    try testing.expectApproxEqAbs(1000.0 + @log(@as(f64, 2.0)), logsumexp(&.{ 1000.0, 1000.0 }), 1e-9);
    try testing.expectEqual(-std.math.inf(f64), logsumexp(&.{ -inf, -inf }));
}

test "rank: the values above the label, then the equal ones before it; rank 0 is the first largest" {
    const row = [_]f32{ 1.5, 3.0, 0.0, 3.0, 2.0 };
    try testing.expectEqual(@as(u32, 0), rank(&row, 1)); // the first 3.0
    try testing.expectEqual(@as(u32, 1), rank(&row, 3)); // the second 3.0
    try testing.expectEqual(@as(u32, 2), rank(&row, 4));
    try testing.expectEqual(@as(u32, 3), rank(&row, 0));
    try testing.expectEqual(@as(u32, 4), rank(&row, 2));
}

/// A record of K 2 as qgate.py's selftest decodes it (xp/qgate.py RECORD_HEX).
const record_hex = "0300000041000000070000000900000000002040010000000000000000800440020001000900000000004040070000000000c03f";

test "a record's bytes: the fixed fields little-endian, then its pairs (qgate.py's RECORD_HEX)" {
    var buf: [prefix_bytes + 16]u8 = undefined;
    const r: Record = .{ .request = 3, .position = 65, .input = 7, .label = 9, .label_logit = 2.5, .label_rank = 1, .lse = 2.5625, .k = 2, .flags = flag_sha };
    const n = encode(&buf, r, &.{ 9, 7 }, &.{ 3.0, 1.5 });
    try testing.expectEqual(@as(usize, prefix_bytes + 16), n);
    var want: [prefix_bytes + 16]u8 = undefined;
    _ = try std.fmt.hexToBytes(&want, record_hex);
    try testing.expectEqualSlices(u8, &want, buf[0..n]);
}

test "the sink labels each row with the reference token its position predicts and skips rows past the reference" {
    const a = testing.allocator;
    var out: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&out);
    var s = try Sink.init(a, &w, 8, 3, 0, false);
    defer s.deinit();
    try s.header(@splat(0xab), 0x10, 0x20, 0, 0, 0);
    var st: u8 = 0; // a stream's address
    var other: u8 = 0;
    try s.register(&st, .{ .request = 5, .prompt_len = 10, .ref = &.{ 4, 6, 1 } });
    try testing.expectEqual(@as(?u32, 4), labelOf(s.regs.get(&st).?, 10)); // the prompt's last row predicts ref[0]
    try testing.expectEqual(@as(?u32, 1), labelOf(s.regs.get(&st).?, 12));
    try testing.expectEqual(@as(?u32, null), labelOf(s.regs.get(&st).?, 13)); // past the reference
    try testing.expectEqual(@as(?u32, null), labelOf(s.regs.get(&st).?, 9)); // inside the prompt
    const row = [_]f32{ 0.5, 2.0, -1.0, 3.0, 0.0, 1.0, 1.5, -2.0 };
    for ([_]u64{ 10, 11, 12, 13 }, [_]u32{ 99, 4, 6, 1 }) |pos, input| try s.row(&st, pos, input, &row);
    try s.row(&other, 11, 4, &row); // an unregistered stream
    try testing.expectEqual(@as(u64, 3), s.records);
    try testing.expectEqual(@as(u64, 1), s.unlabeled);
    // the header, then three records of K 3: request 5, positions 10..12, labels 4, 6, 1, ranks in the row
    const bytes = w.buffered();
    try testing.expectEqual(@as(usize, header_bytes + 3 * (prefix_bytes + 24)), bytes.len);
    try testing.expectEqualSlices(u8, "DSQG", bytes[0..4]);
    var off: usize = header_bytes;
    for ([_]u32{ 10, 11, 12 }, [_]u32{ 99, 4, 6 }, [_]u32{ 4, 6, 1 }, [_]u32{ 5, 2, 1 }) |pos, input, label, rk| {
        const f = bytes[off..];
        try testing.expectEqual(@as(u32, 5), std.mem.readInt(u32, f[0..4], .little));
        try testing.expectEqual(pos, std.mem.readInt(u32, f[4..8], .little));
        try testing.expectEqual(input, std.mem.readInt(u32, f[8..12], .little));
        try testing.expectEqual(label, std.mem.readInt(u32, f[12..16], .little));
        try testing.expectEqual(@as(u32, @bitCast(row[label])), std.mem.readInt(u32, f[16..20], .little));
        try testing.expectEqual(rk, std.mem.readInt(u32, f[20..24], .little));
        try testing.expectEqual(@as(u64, @bitCast(logsumexp(&row))), std.mem.readInt(u64, f[24..32], .little));
        try testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, f[32..34], .little));
        // the pairs: ids 3, 1, 6 (3.0, 2.0, 1.5)
        for ([_]u32{ 3, 1, 6 }, 0..) |id, j| try testing.expectEqual(id, std.mem.readInt(u32, f[prefix_bytes + 8 * j ..][0..4], .little));
        off += prefix_bytes + 24;
    }
}

test "every Nth record carries its whole row, and with row_sha each record its row's sha256" {
    const a = testing.allocator;
    var out: [16384]u8 = undefined;
    var w: std.Io.Writer = .fixed(&out);
    var s = try Sink.init(a, &w, 4, 2, 2, true);
    defer s.deinit();
    var st: u8 = 0;
    try s.register(&st, .{ .request = 0, .prompt_len = 1, .ref = &.{ 0, 1, 2 } });
    const row = [_]f32{ 0.5, 2.0, -1.0, 3.0 };
    for ([_]u64{ 1, 2, 3 }) |pos| try s.row(&st, pos, 0, &row);
    const rec = prefix_bytes + 16 + 32;
    const bytes = w.buffered();
    try testing.expectEqual(@as(usize, 3 * rec + 4 * 4), bytes.len); // the second record (index 1) carries its row
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(&row), &sha, .{});
    try testing.expectEqualSlices(u8, &sha, bytes[prefix_bytes + 16 ..][0..32]);
    try testing.expectEqual(flag_sha, std.mem.readInt(u16, bytes[34..36], .little));
    try testing.expectEqual(flag_sha | flag_row, std.mem.readInt(u16, bytes[rec + 34 ..][0..2], .little));
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&row), bytes[2 * rec ..][0..16]);
}

test "a forced stream's rounds: every row reads the token before its position, and its record is labeled with the reference token committed there" {
    const lanes = @import("lanes");
    const be = lanes.backend;
    const fake = lanes.fake;
    const a = testing.allocator;
    // the lanes driver's facts (lanes.zig Lanes.facts): six-row windows, the head's drafts, no pipelined one-token rounds
    var costs: [6]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, wd| c.* = .{ .width = @intCast(wd), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(wd)) };
    var cfg = try lanes.Config.init(a, .{ .exact_width = 6, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .streams_exact = true, .hidden_rows = true, .batch_rows = 32, .max_streams = 4, .draft_streams = true }, 32, 31);
    defer cfg.deinit(a);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    var sink = try Sink.init(a, &out.writer, fake.vocab, 4, 0, false);
    defer sink.deinit();
    try sink.header(@splat(0), 0, 0, 0, 0, 0);
    // the fake target's verify, then each window's rows into the sink as lanes.zig's verifyFn gives them: the pending
    // token, then the window's drafts (the verified rows after it), each at the window's position for that row
    const Spy = struct {
        var inner: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined;
        var to: *Sink = undefined;
        var widest: usize = 0; // the widest window a registered stream had
        fn verify(ptr: *anyopaque, windows: []const be.Window, res: []be.Verified) anyerror!void {
            try inner(ptr, windows, res);
            for (windows, res) |w, v| {
                if (to.regs.contains(w.stream)) widest = @max(widest, w.rows());
                var row: [fake.vocab]f32 = undefined;
                for (0..w.rows()) |r| {
                    for (&row, 0..) |*x, t| x.* = @floatFromInt((t * 31 + w.positions[r] * 7) % fake.vocab);
                    try to.row(w.stream, w.positions[r], if (r == 0) w.pending else v.drafts[r - 1], &row);
                }
            }
        }
    };
    var vt = target.backend().vtable.*;
    Spy.inner = vt.verify;
    Spy.to = &sink;
    Spy.widest = 0;
    vt.verify = Spy.verify;
    var clock: fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, .{ .ptr = &target, .vtable = &vt }, clock.clock());
    defer engine.deinit();
    // two forced streams and one free one, in shared rounds
    const prompts = [_][]const u32{ &.{ 3, 1, 4, 1, 5 }, &.{ 2, 7, 1, 8, 2, 8, 1, 8, 2 }, &.{ 9, 9, 8 } };
    const refs = [_][]const u32{ &.{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 11, 21, 31, 41, 51, 61, 71, 81 }, &.{ 5, 6, 7, 8 }, &.{} };
    const names = [_][]const u8{ "f0", "f1", "free" };
    var streams: [3]lanes.Stream = undefined;
    var props: [3]lanes.SuffixLookup = undefined;
    for (&streams, &props, prompts, refs, names, 0..) |*s, *p, pr, rf, nm, i| {
        p.* = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
        s.* = try lanes.Stream.init(a, .{ .id = nm, .prompt = pr, .max_new = if (rf.len > 0) @intCast(rf.len) else 12, .eos = &.{96}, .sampling = null, .drafts = true, .proposer = p.proposer() });
        try s.force.appendSlice(a, rf);
        if (rf.len > 0) try sink.register(s, .{ .request = @intCast(i), .prompt_len = @intCast(pr.len), .ref = rf });
    }
    defer for (&streams, &props) |*s, *p| {
        s.deinit(a);
        p.deinit();
    };
    for (&streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    // the replies are the references, and forced windows ran (rows past one)
    for (streams[0..2], refs[0..2]) |*s, rf| try testing.expectEqualSlices(u32, rf, s.emitted());
    try testing.expectEqual(@as(usize, 6), Spy.widest);
    // each record: a row after the prompt's own (lanes.zig's prompt end scores that one), its input the token before its
    // position, its label the reference token committed at its position, every position once
    var seen: [2][32]bool = @splat(@splat(false));
    const bytes = out.written();
    var off: usize = header_bytes;
    var n: usize = 0;
    while (off < bytes.len) : (n += 1) {
        const f = bytes[off..];
        const request = std.mem.readInt(u32, f[0..4], .little);
        const position = std.mem.readInt(u32, f[4..8], .little);
        try testing.expect(request < 2);
        const pl = prompts[request].len;
        const rf = refs[request];
        try testing.expect(position > pl and position < pl + rf.len);
        const before = position - 1;
        try testing.expectEqual(if (before < pl) prompts[request][before] else rf[before - pl], std.mem.readInt(u32, f[8..12], .little));
        try testing.expectEqual(rf[position - pl], std.mem.readInt(u32, f[12..16], .little));
        try testing.expect(!seen[request][position - pl]);
        seen[request][position - pl] = true;
        off += prefix_bytes + 8 * @as(usize, std.mem.readInt(u16, f[32..34], .little));
    }
    try testing.expectEqual(bytes.len, off);
    try testing.expectEqual(refs[0].len - 1 + refs[1].len - 1, n);
    try testing.expectEqual(@as(u64, n), sink.records);
}

test "lanes.zig scores every round's rows where the sink reads them: no pipelined one-token rounds, each row at its own position" {
    const src = @embedFile("lanes.zig");
    // a pipelined one-token round (Backend.queue) would bypass verifyFn and the logits-log with it
    const facts = src[std.mem.indexOf(u8, src, "pub fn facts(self: *const Lanes) lanes.Model {").?..];
    const body = facts[0..std.mem.indexOf(u8, facts, "\n    }\n").?];
    try testing.expect(std.mem.indexOf(u8, body, ".gpu_tokens = false,") != null);
    // a window's row r: the pending token, then the held drafts, then the host's tokens; its record at positions[r]
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "ids[k][0] = @intCast(w.pending);"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "try lg.row(w.stream, w.positions[r], @intCast(ids[k][r]), logits[(row0 + r) * m.vocab ..][0..m.vocab]);"));
    // the prompt's last row predicts the token at the prompt's length, after the prompt's last token
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "try lg.row(s, len, @intCast(seq.items[len - 1]), logits);"));
}

test "version 4's header tail holds kern2, the speed word and the settings hash (8 bytes); version 1's 64 bytes come first" {
    const a = testing.allocator;
    var out: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&out);
    var s = try Sink.init(a, &w, 8, 3, 0, false);
    defer s.deinit();
    const sp = speedWord(.{ .twin = 3, .rope_w = true, .merge_cols = 64, .group2 = true, .expert_exit = false, .pack_epilogue = true, .hc_2s = true, .qkv_plain = false });
    const h = settingsHash("INFO speed features: 3 on, 0 off\n");
    try s.header(@splat(0xab), 0x10, 0x20, 0x1, sp, h);
    const bytes = w.buffered();
    try testing.expectEqual(@as(usize, 80), bytes.len);
    try testing.expectEqual(@as(usize, 80), header_bytes);
    try testing.expectEqual(@as(u32, 4), std.mem.readInt(u32, bytes[4..8], .little));
    try testing.expectEqual(h, std.mem.readInt(u64, bytes[72..80], .little));
    try testing.expectEqual(@as(u32, 0x10), std.mem.readInt(u32, bytes[24..28], .little)); // v1's kern and exch where they were
    try testing.expectEqual(@as(u32, 0x20), std.mem.readInt(u32, bytes[28..32], .little));
    try testing.expectEqual(@as(u32, 0x1), std.mem.readInt(u32, bytes[64..68], .little));
    try testing.expectEqual(sp, std.mem.readInt(u32, bytes[68..72], .little));
}

test "the speed word: the effective state of the decode defaults that carry no kernel-word bit" {
    const all_off: Speed = .{ .twin = 0, .rope_w = false, .merge_cols = 0, .group2 = false, .expert_exit = false, .pack_epilogue = false, .hc_2s = false, .qkv_plain = false };
    try testing.expectEqual(@as(u32, 0), speedWord(all_off));
    // every field moves its own bits, and only those (twin bits 0-1, rope_w 2, merge 3-4, group2 5, expert_exit 6, pack_epilogue 7, hc_2s 8)
    var one = all_off;
    one.twin = 3;
    try testing.expectEqual(@as(u32, 0b11), speedWord(one));
    one = all_off;
    one.rope_w = true;
    try testing.expectEqual(@as(u32, 1 << 2), speedWord(one));
    one = all_off;
    one.merge_cols = 128;
    try testing.expectEqual(@as(u32, 1 << 3), speedWord(one));
    one.merge_cols = 64;
    try testing.expectEqual(@as(u32, 2 << 3), speedWord(one));
    one.merge_cols = 16;
    try testing.expectEqual(@as(u32, 3 << 3), speedWord(one));
    one.merge_cols = 32; // a width the word has no code for reads as off
    try testing.expectEqual(@as(u32, 0), speedWord(one));
    one = all_off;
    one.group2 = true;
    try testing.expectEqual(@as(u32, 1 << 5), speedWord(one));
    one = all_off;
    one.expert_exit = true;
    try testing.expectEqual(@as(u32, 1 << 6), speedWord(one));
    one = all_off;
    one.pack_epilogue = true;
    try testing.expectEqual(@as(u32, 1 << 7), speedWord(one));
    one = all_off;
    one.hc_2s = true;
    try testing.expectEqual(@as(u32, 1 << 8), speedWord(one));
    // T3's defaults (z twin, RoPE-W, merge 128, group2, expert exit, pack epilogue, 2S) against the pre-11-09 set (u twin, the rest off)
    try testing.expectEqual(@as(u32, 0x1EF), speedWord(.{ .twin = 3, .rope_w = true, .merge_cols = 128, .group2 = true, .expert_exit = true, .pack_epilogue = true, .hc_2s = true, .qkv_plain = false }));
}

test "the speed word's bit 9 is TF_DS_QKV_PLAIN in effect, and nothing else moves with it" {
    var s: Speed = .{ .twin = 0, .rope_w = false, .merge_cols = 0, .group2 = false, .expert_exit = false, .pack_epilogue = false, .hc_2s = false, .qkv_plain = true };
    try testing.expectEqual(@as(u32, 1 << 9), speedWord(s));
    s.qkv_plain = false;
    try testing.expectEqual(@as(u32, 0), speedWord(s));
    // T3's defaults with QKV_PLAIN on (T4's): 0x1EF plus bit 9
    try testing.expectEqual(@as(u32, 0x3EF), speedWord(.{ .twin = 3, .rope_w = true, .merge_cols = 128, .group2 = true, .expert_exit = true, .pack_epilogue = true, .hc_2s = true, .qkv_plain = true }));
    try testing.expectEqual(@as(usize, 8), speed_names.len);
}

test "the settings hash: the same text the same 64 bits, any changed line another" {
    const a = "INFO   ON  prompt passes: TF_DS_PASS_ROWS=4096, on by default\nINFO   ON  the drafter: TF_DS_DRAFT_HEAD=fp4, on by default\n";
    const b = "INFO   ON  prompt passes: TF_DS_PASS_ROWS=2048, on by default\nINFO   ON  the drafter: TF_DS_DRAFT_HEAD=fp4, on by default\n";
    try testing.expectEqual(settingsHash(a), settingsHash(a));
    try testing.expect(settingsHash(a) != settingsHash(b));
    try testing.expect(settingsHash("") != settingsHash(a));
}
