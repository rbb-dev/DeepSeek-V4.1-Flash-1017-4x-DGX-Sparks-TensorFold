//! Engram's rows from packed shards (TF_DS_ENGRAM_PACKED, engram_pack.zig) instead of the tables' separate weight and
//! scale rows: each slot's record staged on the host (a cache hit copied, a miss read in one go, a row another node
//! owns zeroed), uploaded as it is, combined over the nodes when each owns a range of the rows (TF_DS_ENGRAM_OWNED: a
//! byte max, one node holding a slot's bytes and the rest zeros) and decoded to bf16 on the GPU (engram_gpu.zig).
//! A layer's rows keep the host path's bits; only where they come from changes.
const std = @import("std");
const cuda = @import("cuda");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const engram_pack = @import("engram_pack.zig");
const engram_cache = @import("engram_cache.zig");
pub const anonHugeKiB = engram_cache.anonHugeKiB;
pub const Fault = engram_cache.Fault;
const engram_gpu = @import("engram_gpu.zig");
const engram_partner = @import("engram_partner.zig");
const comm = @import("comm.zig");

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Whether environment switch `name` is set to anything but 0.
pub fn envOn(name: [:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// How many different ids `ids` holds (a miss list: a row asked by several slots of a step is read once per slot in the baseline).
pub fn uniqueIds(gpa: std.mem.Allocator, ids: []const i64) !usize {
    if (ids.len == 0) return 0;
    var cap: usize = 16;
    while (cap < 2 * ids.len) cap *= 2;
    const set = try gpa.alloc(u64, cap);
    defer gpa.free(set);
    @memset(set, 0);
    var uniq: usize = 0;
    for (ids) |x| {
        const key: u64 = @as(u64, @bitCast(x)) +% 1; // (0 is the empty mark)
        var at: usize = @intCast((key *% 0x9e3779b97f4a7c15) >> @intCast(64 - @ctz(cap)));
        while (set[at] != 0 and set[at] != key) at = (at + 1) & (cap - 1);
        if (set[at] == 0) {
            set[at] = key;
            uniq += 1;
        }
    }
    return uniq;
}

test "uniqueIds counts the different ids of a miss list: repeats once, a list without repeats whole, the empty list none" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 0), try uniqueIds(a, &.{}));
    try std.testing.expectEqual(@as(usize, 5), try uniqueIds(a, &.{ 7, 9, 11, 13, 1_000_000_007 })); // the control: no repeats
    try std.testing.expectEqual(@as(usize, 3), try uniqueIds(a, &.{ 7, 9, 7, 7, 11, 9 }));
    try std.testing.expectEqual(@as(usize, 1), try uniqueIds(a, &.{ 0, 0, 0, 0 })); // (id 0 is an id, not the empty mark)
    // a long list with a known number of different ids, as a step's 24,576 slots
    var prng = std.Random.DefaultPrng.init(3);
    const big = try a.alloc(i64, 24576);
    defer a.free(big);
    for (big) |*x| x.* = prng.random().intRangeLessThan(i64, 0, 9000);
    var seen = std.AutoHashMap(i64, void).init(a);
    defer seen.deinit();
    for (big) |x| try seen.put(x, {});
    try std.testing.expectEqual(@as(usize, seen.count()), try uniqueIds(a, big));
}

/// A staging buffer: rows x slots records, the misses' row ids and where each lands, their AIO batch.
pub const step_ring = 64;
pub const StepRec = struct { tag: u64 = 0, count: u32 = 0, ord: u32 = 0, set: u32 = 0 };

/// Hashes of a miss list: one that depends on the order of the ids, one that does not.
pub fn missHashes(ids: []const i64) struct { ord: u32, set: u32 } {
    var o: u32 = 0x811c9dc5;
    var s: u32 = 0;
    for (ids) |x| {
        const v: u64 = @bitCast(x);
        o = (o ^ @as(u32, @truncate(v))) *% 16777619;
        o = (o ^ @as(u32, @truncate(v >> 32))) *% 16777619;
        var z = v +% 0x9e3779b97f4a7c15;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        s +%= @as(u32, @truncate(z ^ (z >> 31)));
    }
    return .{ .ord = o, .set = s };
}

test "missHashes: the order hash sees a swap, the set hash does not, a changed id changes both" {
    const a = missHashes(&.{ 5, 9, 12, 7 });
    const b = missHashes(&.{ 9, 5, 12, 7 });
    const c = missHashes(&.{ 5, 9, 12, 8 });
    try std.testing.expect(a.ord != b.ord);
    try std.testing.expectEqual(a.set, b.set);
    try std.testing.expect(a.ord != c.ord and a.set != c.set);
    try std.testing.expectEqual(missHashes(&.{ 5, 9, 12, 7 }).ord, a.ord);
}

pub const Stage = struct {
    /// A stage that holds nothing: the slots a Rows does not make.
    pub const empty: Stage = .{ .j = 0, .bytes = &.{}, .ids = &.{}, .dsts = &.{} };

    j: usize, // the Engram layer slot it serves (its cache)
    bytes: []u8,
    ids: []i64,
    dsts: [][*]u8,
    misses: usize = 0,
    batch: u64 = 0, // the misses' AIO batch (0: none)
    host: ?cuda.HostBuffer = null, // pinned bytes (a round's), else the allocator's
    rows: ?*Rows = null, // what a stream's host step reads
    failed: std.atomic.Value(bool) = .init(false),

    fn init(gpa: std.mem.Allocator, d: ?*const cuda.Driver, j: usize, records: usize, rec: usize) !Stage {
        var st: Stage = .{ .j = j, .bytes = &.{}, .ids = &.{}, .dsts = &.{} };
        errdefer st.deinit(gpa);
        if (d) |dr| {
            st.host = try cuda.HostBuffer.alloc(dr, records * rec);
            st.bytes = st.host.?.bytes[0 .. records * rec];
        } else st.bytes = try gpa.alloc(u8, records * rec);
        st.ids = try gpa.alloc(i64, records);
        st.dsts = try gpa.alloc([*]u8, records);
        return st;
    }

    fn deinit(st: *Stage, gpa: std.mem.Allocator) void {
        if (st.host) |*h| h.free() else if (st.bytes.len > 0) gpa.free(st.bytes);
        if (st.ids.len > 0) gpa.free(st.ids);
        if (st.dsts.len > 0) gpa.free(st.dsts);
        st.* = undefined;
    }
};

/// The resume units (2,048 rows) in the largest pass (TF_DS_PASS_ROWS=16384).
pub const max_units = 8;

/// The most chunk slots the worker holds: two passes of the largest pass's units.
pub const max_wk = 2 * max_units;

/// The chunk slots the worker's ring holds for passes of `units` resume units: a pass reads all its units before the
/// layers give the first back, so the ring needs `units` at the least (fewer hangs the worker, silently); two passes in
/// flight (`pair`) need a whole pass more for the worker to read ahead into the next, so twice the units. Never fewer
/// than two, the ring before passes of more than one unit existed.
pub fn workerSlots(units: usize, pair: bool) usize {
    return workerSlotsN(units, if (pair) 2 else 1);
}

/// `workerSlots` for `lanes` passes in flight (TF_DS_ROCI_STAGGER_LANES: 3 takes three times the units, up to max_wk).
pub fn workerSlotsN(units: usize, lanes: usize) usize {
    return @max(2, if (lanes >= 2) lanes * units else units);
}

/// Why a ring of `slots` cannot serve passes of `units` resume units (null: it can): the name of the setting and the
/// slots it needs.
pub fn slotsRefusal(buf: []u8, slots: usize, units: usize) ?[]const u8 {
    if (slots >= units and slots <= max_wk) return null;
    return std.fmt.bufPrint(buf, "the Engram worker's ring has {d} chunk slots and a pass of {d} resume units ({d} rows, TF_DS_PASS_ROWS) needs {d} to {d}: a ring with fewer slots than a pass's units never lets the worker finish a pass", .{ slots, units, units * 2048, units, max_wk }) catch "the Engram worker's ring does not hold a pass's units";
}

pub const Options = struct {
    dir: []const u8, // the packed shards
    owned: bool, // node `node` of `nodes` reads its own range of each table's rows (else the whole tables, node 0 of 1)
    node: u32,
    nodes: u32,
    lo: usize, // this node's hash columns [lo, lo + k) of a row's `cols`
    k: usize,
    cols: usize,
    dim: usize, // a row's values (engram_head_dim)
    layer_ids: []const u16, // the Engram layers in order
    round_rows: usize, // a round's rows at most
    chunk_rows: usize, // a prompt chunk's
    cache_bytes: usize, // the row cache over the layers (0: none)
    prefault: bool = false, // its records faulted in at the open (TF_DS_ENGRAM_PREFAULT: engram_cache.Cache.initFaulted)
    instream: bool, // a round's landing as a host step on the stream (needs the round's AIO ring)
    pinned: bool = false, // the fill worker's stages in pinned host memory (TF_DS_ENGRAM_PINNED)
    round_dev: u64, // device records of a round (round_rows * slots * record bytes)
    chunk_dev: u64, // of a chunk (chunk_rows * slots * record bytes)
    chunk_dev_b: u64 = 0, // the second lane's pass: its own (0: one pass at a time)
    chunk_dev_c: u64 = 0, // the third lane's pass (TF_DS_ROCI_STAGGER_LANES=3; 0: two at most)
    wk_slots: usize = 2, // the worker's chunk slots (workerSlots): 2 to max_wk
    unit_rows: usize = 0, // the rows one worker slot holds: a resume unit (0: chunk_rows)
};

pub const Rows = struct {
    gpa: std.mem.Allocator,
    owned: bool,
    node: u32,
    nodes: u32,
    lo: usize,
    k: usize,
    cols: usize,
    dim: usize,
    rec: usize,
    layers: usize,
    shards: engram_pack.Shards,
    pk: [2]engram_io.Packed,
    caches: [2]?engram_cache.Cache,
    kernels: engram_gpu.Kernels,
    round: [2]Stage, // a round's records by Engram layer slot (pinned)
    chunk: Stage, // a prompt chunk's, read on the pass's thread
    ahead: [2]Stage, // a prompt chunk's read ahead, by Engram layer slot
    wk: [max_wk][2]Stage, // a fill's worker's records, by chunk slot and Engram layer slot (prompt.engramWorker); the first wk_n slots exist
    wk_n: usize,
    round_dev: u64,
    chunk_dev: u64,
    chunk_dev_b: u64,
    chunk_dev_c: u64,
    pool: *engram_io.Pool,
    aio: ?*engram_aio.Aio, // a round's reads (TF_DS_ENGRAM_AIO), else the pool
    paio: ?*engram_aio.Aio, // a chunk's
    instream: bool,
    fault: [2]engram_cache.Fault = .{ .{}, .{} }, // how each layer's cache was faulted in (Options.prefault)
    fault_ns: u64 = 0, // and the time it took
    // TF_DS_ROCI_PAIR_LOG's count of repeated ids: the misses of every staged step and how many different ids they name
    dup_on: bool = false,
    // TF_DS_ENGRAM_PARTNER: the fill worker's steps read half of their misses and take the other half from the column partner
    partner: ?*engram_partner.Link = null,
    // the worker's steps' miss lists, for comparing two nodes that need the same rows (TF_DS_ROCI_PAIR_LOG): per step the tag the worker
    // set (absolute chunk index << 8 | layer slot), the miss count, a hash of the ids in list order and a hash of the set (order-free)
    steps: [step_ring]StepRec = @splat(.{}),
    steps_n: std.atomic.Value(u64) = .init(0),
    step_tag: std.atomic.Value(u64) = .init(0),
    other_plans: std.atomic.Value(u64) = .init(0), // plans by anything but the fill worker (rounds, read-ahead, inline reads): they fill the cache too
    dup_misses: std.atomic.Value(u64) = .init(0),
    dup_unique: std.atomic.Value(u64) = .init(0),

    /// The shards of node o.node of o.nodes (owned) or the whole ones, checked against `tables`; the kernels; the
    /// staging buffers and the caches. Needs the CUDA context current.
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, tables: *const engram_io.Tables, pool: *engram_io.Pool, aio: ?*engram_aio.Aio, paio: ?*engram_aio.Aio, o: Options) !*Rows {
        if (o.layer_ids.len == 0 or o.layer_ids.len > 2 or o.dim == 0 or o.dim % 32 != 0 or o.lo + o.k > o.cols) return error.EngramShape;
        if (o.instream and aio == null) return error.EngramInStreamNeedsAio;
        if (!cuda.kernels.available) return error.NoKernelImages;
        const rec = o.dim + o.dim / 32;
        const nodes: u32 = if (o.owned) o.nodes else 1;
        const node: u32 = if (o.owned) o.node else 0;
        if (node >= nodes) return error.EngramShape;
        const sl = if (o.owned) o.cols else o.k;
        var shards = try engram_pack.Shards.open(gpa, o.dir, tables, node, nodes);
        errdefer shards.close();
        var pks: [2]engram_io.Packed = undefined;
        for (o.layer_ids, 0..) |li, j| {
            pks[j] = shards.layers.get(li) orelse return error.MissingEngramTable;
            if (pks[j].rec != rec) return error.EngramShape;
        }
        var kernels = try engram_gpu.Kernels.load(d, cuda.kernels.dsv41_engram);
        errdefer kernels.unload();
        var round0 = try Stage.init(gpa, d, 0, o.round_rows * sl, rec);
        errdefer round0.deinit(gpa);
        var round1 = try Stage.init(gpa, d, 1, o.round_rows * sl, rec);
        errdefer round1.deinit(gpa);
        var chunk = try Stage.init(gpa, null, 0, o.chunk_rows * sl, rec);
        errdefer chunk.deinit(gpa);
        var ahead0 = try Stage.init(gpa, null, 0, o.chunk_rows * sl, rec);
        errdefer ahead0.deinit(gpa);
        var ahead1 = try Stage.init(gpa, null, 1, o.chunk_rows * sl, rec);
        errdefer ahead1.deinit(gpa);
        if (o.wk_slots < 2 or o.wk_slots > max_wk) return error.EngramShape;
        const unit_rows = if (o.unit_rows == 0) o.chunk_rows else o.unit_rows;
        var wk: [max_wk][2]Stage = @splat(@splat(Stage.empty));
        var made: usize = 0;
        errdefer for (0..made) |i| wk[i / 2][i % 2].deinit(gpa);
        for (0..2 * o.wk_slots) |i| {
            wk[i / 2][i % 2] = try Stage.init(gpa, if (o.pinned) d else null, i % 2, unit_rows * sl, rec);
            made += 1;
        }
        var caches: [2]?engram_cache.Cache = .{ null, null };
        errdefer for (&caches) |*c| {
            if (c.*) |*x| x.deinit(gpa);
        };
        var fault: [2]engram_cache.Fault = .{ .{}, .{} };
        const t0 = nowNs();
        if (o.cache_bytes > 0) {
            for (0..o.layer_ids.len) |j| caches[j] = if (o.prefault) try engram_cache.Cache.initFaulted(gpa, o.cache_bytes / o.layer_ids.len, rec, 16, &fault[j]) else try engram_cache.Cache.init(gpa, o.cache_bytes / o.layer_ids.len, rec, 16);
        }
        const fault_ns = nowNs() - t0;
        const r = try gpa.create(Rows);
        r.* = .{
            .gpa = gpa,
            .owned = o.owned,
            .node = node,
            .nodes = nodes,
            .lo = o.lo,
            .k = o.k,
            .cols = o.cols,
            .dim = o.dim,
            .rec = rec,
            .layers = o.layer_ids.len,
            .shards = shards,
            .pk = pks,
            .caches = caches,
            .kernels = kernels,
            .round = .{ round0, round1 },
            .chunk = chunk,
            .ahead = .{ ahead0, ahead1 },
            .wk = wk,
            .wk_n = o.wk_slots,
            .round_dev = o.round_dev,
            .chunk_dev = o.chunk_dev,
            .chunk_dev_b = o.chunk_dev_b,
            .chunk_dev_c = o.chunk_dev_c,
            .pool = pool,
            .aio = aio,
            .paio = paio,
            .instream = o.instream,
            .fault = fault,
            .fault_ns = fault_ns,
        };
        for (&r.round) |*st| st.rows = r;
        return r;
    }

    pub fn close(r: *Rows) void {
        const gpa = r.gpa;
        for (&r.round) |*st| st.deinit(gpa);
        r.chunk.deinit(gpa);
        for (&r.ahead) |*st| st.deinit(gpa);
        for (r.wk[0..r.wk_n]) |*row| {
            for (row) |*st| st.deinit(gpa);
        }
        for (&r.caches) |*c| {
            if (c.*) |*x| x.deinit(gpa);
        }
        r.kernels.unload();
        r.shards.close();
        gpa.destroy(r);
    }

    /// A row's slots this node stages: every hash column (owned), else its own k.
    pub fn slots(r: *const Rows) usize {
        return if (r.owned) r.cols else r.k;
    }

    /// `st` for `n` rows of Engram layer slot `j` from `hashes` ([n][layers][cols] row ids): each slot a cache hit's
    /// record, zeros (a row another node owns), or a miss listed with where it lands.
    fn plan(r: *Rows, st: *Stage, j: usize, hashes: []const i64, n: usize, layers: usize) !void {
        const sl = r.slots();
        if (n * sl * r.rec > st.bytes.len or hashes.len < n * layers * r.cols or j >= r.layers) return error.EngramShape;
        const first = if (r.owned) 0 else r.lo;
        const pk = r.pk[j];
        const cache: ?*engram_cache.Cache = if (r.caches[j]) |*c| c else null;
        var m: usize = 0;
        for (0..n) |row| {
            for (0..sl) |q| {
                const x = hashes[(row * layers + j) * r.cols + first + q];
                if (x < 0) return error.BadEngramRow;
                const id: u64 = @intCast(x);
                const dst = st.bytes[(row * sl + q) * r.rec ..][0..r.rec];
                if (r.owned and (id < pk.lo or id >= pk.hi)) {
                    @memset(dst, 0);
                    continue;
                }
                if (cache) |c| {
                    if (c.get(id, dst)) continue;
                }
                st.ids[m] = x;
                st.dsts[m] = dst.ptr;
                m += 1;
            }
        }
        st.j = j;
        st.misses = m;
        st.batch = 0;
        if (r.dup_on) {
            const w0 = @intFromPtr(&r.wk[0][0]);
            const w1 = @intFromPtr(&r.wk[max_wk - 1][1]) + @sizeOf(Stage);
            if (@intFromPtr(st) >= w0 and @intFromPtr(st) < w1) {
                const h = missHashes(st.ids[0..m]);
                const at = r.steps_n.load(.monotonic);
                r.steps[at % step_ring] = .{ .tag = r.step_tag.load(.monotonic), .count = @intCast(m), .ord = h.ord, .set = h.set };
                r.steps_n.store(at + 1, .release);
            } else _ = r.other_plans.fetchAdd(1, .monotonic);
        }
        if (r.dup_on and m > 0) {
            const u = uniqueIds(r.gpa, st.ids[0..m]) catch m;
            _ = r.dup_misses.fetchAdd(m, .monotonic);
            _ = r.dup_unique.fetchAdd(u, .monotonic);
        }
    }

    /// The staged misses' records kept in their layer's cache.
    fn keep(r: *Rows, st: *const Stage) void {
        const c = if (r.caches[st.j]) |*x| x else return;
        for (st.ids[0..st.misses], st.dsts[0..st.misses]) |x, dst| c.put(@intCast(x), dst[0..r.rec]);
    }

    /// The staged misses read now (an AIO ring when given, else the reader pool) and kept.
    fn readMisses(r: *Rows, st: *Stage, aio: ?*engram_aio.Aio) !void {
        const m = st.misses;
        if (m == 0) return;
        const pk = r.pk[st.j];
        if (aio) |x| try aioRecords(x, pk, st.ids[0..m], st.dsts[0..m]) else try r.pool.gatherRecords(pk, st.ids[0..m], st.dsts[0..m]);
        r.keep(st);
    }

    /// A round's rows of Engram layer slot `j` staged from `hashes` ([rows][layers][cols]), the misses submitted to the
    /// round's AIO ring when there is one (else they are read when the layer lands).
    pub fn roundAhead(r: *Rows, j: usize, hashes: []const i64, rows: usize, layers: usize) !void {
        const st = &r.round[j];
        st.failed.store(false, .monotonic);
        try r.plan(st, j, hashes, rows, layers);
        if (st.misses > 0) {
            if (r.aio) |x| {
                if (st.misses <= x.slots) st.batch = try x.startRecords(r.pk[j], st.ids[0..st.misses], st.dsts[0..st.misses]);
            }
        }
    }

    /// Engram layer slot `j`'s rows of the round into `out` (bf16 [rows][k][dim]) in `s`'s order: the misses landed
    /// (read now when none were submitted) and kept, the records uploaded, combined over the nodes (owned) and
    /// decoded. In stream mode the landing is a host step on `s`, so the host goes on: roundCheck after the round.
    pub fn roundLand(r: *Rows, j: usize, rows: usize, s: cuda.Stream, out: u64) !void {
        const st = &r.round[j];
        const sl = r.slots();
        const bytes = rows * sl * r.rec;
        if (st.batch != 0 and r.instream) {
            try s.hostFunc(stepLand, st);
        } else if (st.batch != 0) {
            const id = st.batch;
            st.batch = 0;
            try r.aio.?.wait(id);
            r.keep(st);
        } else try r.readMisses(st, r.aio);
        try upload(s, r.round_dev, st.bytes[0..bytes]);
        try r.kernels.decodeRows(s, r.round_dev, out, rows, sl, if (r.owned) r.lo else 0, r.k, r.dim);
    }

    /// A round's layer slot `j` that will not land (its round failed, G0.3): its submitted batch waited for and dropped
    /// (nothing kept), so the slot's staging and batch id are free for the next round. Errors are ignored.
    pub fn roundDrain(r: *Rows, j: usize) void {
        const st = &r.round[j];
        if (st.batch == 0) return;
        const id = st.batch;
        st.batch = 0;
        if (r.aio) |x| x.wait(id) catch {};
    }

    /// After a round's synchronize: error.ReadFailed when a stream host step's read failed.
    pub fn roundCheck(r: *Rows) !void {
        var bad = false;
        for (&r.round) |*st| bad = st.failed.swap(false, .acq_rel) or bad;
        if (bad) return error.ReadFailed;
    }

    /// The pinned bytes of the fill worker's stages (Options.pinned; 0 without).
    pub fn pinnedBytes(r: *const Rows) usize {
        var n: usize = 0;
        for (r.wk[0..r.wk_n]) |*row| for (row) |*st| {
            if (st.host) |h| n += h.bytes.len;
        };
        return n;
    }

    /// A prompt chunk's rows of Engram layer slot `j` staged in `st` (the chunk's own, or a read-ahead's) and read.
    pub fn chunkRead(r: *Rows, st: *Stage, j: usize, hashes: []const i64, n: usize, layers: usize) !void {
        try r.plan(st, j, hashes, n, layers);
        try r.readMisses(st, r.paio);
    }

    /// chunkRead for the fill worker with the partner split on (else chunkRead): the step's misses are planned as ever, then each of the two
    /// column partners reads half and they swap (engram_partner.splitRead), and the cache keeps the whole list in the list's order. `tag`: the
    /// step's place in the fill, the same on both partners.
    pub fn chunkReadSplit(r: *Rows, st: *Stage, j: usize, hashes: []const i64, n: usize, layers: usize, tag: u64) !void {
        const link = r.partner orelse return r.chunkRead(st, j, hashes, n, layers);
        try r.plan(st, j, hashes, n, layers);
        var ctx: ReadCtx = .{ .r = r, .j = j };
        _ = try engram_partner.splitRead(link, st.ids[0..st.misses], st.dsts[0..st.misses], tag, .{ .ctx = &ctx, .read = readIds }, r.rec);
        r.keep(st);
    }

    /// A staged chunk's records uploaded, combined over the nodes (owned) and decoded into `out` (bf16 [n][k][dim]).
    pub fn chunkApply(r: *Rows, st: *const Stage, n: usize, s: cuda.Stream, out: u64) !void {
        return r.chunkApplyOn(st, n, s, out, 0);
    }

    /// chunkApply on a lane: the second and third lanes' passes decode through their own device records, so passes in flight never share them.
    pub fn chunkApplyOn(r: *Rows, st: *const Stage, n: usize, s: cuda.Stream, out: u64, lane: u2) !void {
        const sl = r.slots();
        const bytes = n * sl * r.rec;
        const dev = switch (lane) {
            0 => r.chunk_dev,
            1 => r.chunk_dev_b,
            else => r.chunk_dev_c,
        };
        try upload(s, dev, st.bytes[0..bytes]);
        try r.kernels.decodeRows(s, dev, out, n, sl, if (r.owned) r.lo else 0, r.k, r.dim);
    }
};

const ReadCtx = struct { r: *Rows, j: usize };

/// The partner split's reader: records of row ids of layer slot `j` through the prompt ring (else the reader pool).
fn readIds(ctx: *anyopaque, ids: []const i64, dsts: []const [*]u8) anyerror!void {
    const c: *ReadCtx = @ptrCast(@alignCast(ctx));
    const pk = c.r.pk[c.j];
    if (c.r.paio) |x| try aioRecords(x, pk, ids, dsts) else try c.r.pool.gatherRecords(pk, ids, dsts);
}

/// A stream's host step for a staged round layer: its AIO batch waited for and its misses kept (no CUDA call).
fn stepLand(data: ?*anyopaque) callconv(.c) void {
    const st: *Stage = @ptrCast(@alignCast(data.?));
    const r = st.rows.?;
    const id = st.batch;
    st.batch = 0;
    r.aio.?.wait(id) catch {
        st.failed.store(true, .release);
        return;
    };
    r.keep(st);
}

/// Bytes to the device in `s`'s order (pinned bytes copy when the stream gets there; pageable ones are staged now).
fn upload(s: cuda.Stream, dst: u64, src: []const u8) !void {
    if (src.len == 0) return;
    try s.d.copyHtoDAsync(@src(), dst, src.ptr, src.len, s.handle);
}

/// Records of `ids` from `pk` into `dsts` through `aio`: batches of half its ring, all submitted, waited for in order.
fn aioRecords(aio: *engram_aio.Aio, pk: engram_io.Packed, ids: []const i64, dsts: []const [*]u8) !void {
    const per = @max(aio.slots / 2, 1);
    var batch: [64]u64 = undefined;
    var nb: usize = 0;
    var at: usize = 0;
    errdefer for (batch[0..nb]) |id| aio.wait(id) catch {};
    while (at < ids.len) : (at += per) {
        const n = @min(per, ids.len - at);
        if (nb == batch.len) return error.TooManyRows;
        batch[nb] = try aio.startRecords(pk, ids[at..][0..n], dsts[at..][0..n]);
        nb += 1;
    }
    for (batch[0..nb], 0..) |id, i| {
        aio.wait(id) catch |err| {
            for (batch[i + 1 .. nb]) |rest| aio.wait(rest) catch {};
            nb = 0;
            return err;
        };
    }
    nb = 0;
}

test "an owned plan stages hits, zeros another node's rows and lists the misses" {
    const a = std.testing.allocator;
    var cache = try engram_cache.Cache.init(a, 64 * 4, 4, 4);
    defer cache.deinit(a);
    cache.put(12, &.{ 1, 2, 3, 4 });
    var st = try Stage.init(a, null, 0, 2 * 3, 4);
    defer st.deinit(a);
    var r: Rows = undefined;
    r.owned = true;
    r.lo = 1;
    r.k = 2;
    r.cols = 3;
    r.rec = 4;
    r.layers = 1;
    r.pk = .{ .{ .fd = -1, .base = 0, .rec = 4, .rows = 40, .lo = 10, .hi = 20 }, undefined };
    r.caches = .{ cache, null };
    // two rows, one layer, three columns: 12 cached, 5 and 25 another node's, 11 / 13 / 19 misses
    const hashes = [_]i64{ 12, 5, 11, 13, 25, 19 };
    try r.plan(&st, 0, &hashes, 2, 1);
    try std.testing.expectEqual(@as(usize, 3), st.misses);
    try std.testing.expectEqualSlices(i64, &.{ 11, 13, 19 }, st.ids[0..3]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, st.bytes[0..4]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, st.bytes[4..8]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, st.bytes[16..20]);
    try std.testing.expectEqual(@intFromPtr(st.bytes[8..].ptr), @intFromPtr(st.dsts[0]));
    try std.testing.expectEqual(@intFromPtr(st.bytes[20..].ptr), @intFromPtr(st.dsts[2]));
    // the misses kept: the next plan of the same rows reads nothing
    @memset(st.bytes[8..12], 7);
    @memset(st.bytes[12..16], 8);
    @memset(st.bytes[20..24], 9);
    r.keep(&st);
    try r.plan(&st, 0, &hashes, 2, 1);
    try std.testing.expectEqual(@as(usize, 0), st.misses);
    try std.testing.expectEqualSlices(u8, &.{ 9, 9, 9, 9 }, st.bytes[20..24]);
    // the whole tables (not owned): this node's columns only, every row a miss or a hit
    r.owned = false;
    r.caches = .{ null, null };
    r.pk[0].lo = 0;
    r.pk[0].hi = 40;
    try r.plan(&st, 0, &hashes, 2, 1);
    try std.testing.expectEqualSlices(i64, &.{ 5, 11, 25, 19 }, st.ids[0..st.misses]);
}

test "TF_DS_ENGRAM_PREFAULT: the row caches are faulted in at the open, each layer's, and the open keeps how and how long" {
    const src = @embedFile("engram_rows.zig");
    const at = std.mem.indexOf(u8, src, "    pub fn open(").?;
    const body = src[at .. at + std.mem.indexOf(u8, src[at..], "\n    }\n").?];
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "caches[j] = if (o.prefault) try engram_cache.Cache.initFaulted(gpa, o.cache_bytes / o.layer_ids.len, rec, 16, &fault[j]) else try engram_cache.Cache.init(gpa, o.cache_bytes / o.layer_ids.len, rec, 16);"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, ".fault = fault,"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, ".fault_ns = fault_ns,"));
    const model = @embedFile("model.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, model, "const prefault = engram_rows.envOn(\"TF_DS_ENGRAM_PREFAULT\");"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, model, ".prefault = prefault,"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, model, "if (prefault) logPrefault(m.erows.?, mib, huge0, hugeKiB(io));"));
}

test "the worker's ring: a pass's resume units, twice them with two passes in flight, never fewer than two; a ring under a pass's units is refused naming the setting and the slots" {
    try std.testing.expectEqual(@as(usize, 8), max_units); // 16,384 rows
    try std.testing.expectEqual(@as(usize, 2), workerSlots(1, false)); // 2,048-row passes
    try std.testing.expectEqual(@as(usize, 2), workerSlots(2, false)); // 4,096
    try std.testing.expectEqual(@as(usize, 4), workerSlots(4, false)); // 8,192
    try std.testing.expectEqual(@as(usize, 8), workerSlots(8, false)); // 16,384
    try std.testing.expectEqual(@as(usize, 2), workerSlots(1, true));
    try std.testing.expectEqual(@as(usize, 4), workerSlots(2, true));
    try std.testing.expectEqual(@as(usize, 8), workerSlots(4, true));
    try std.testing.expectEqual(@as(usize, 16), workerSlots(8, true));
    try std.testing.expectEqual(max_wk, workerSlots(max_units, true));
    var buf: [320]u8 = undefined;
    for (1..max_units + 1) |units| {
        try std.testing.expectEqual(@as(?[]const u8, null), slotsRefusal(&buf, workerSlots(units, false), units));
        try std.testing.expectEqual(@as(?[]const u8, null), slotsRefusal(&buf, workerSlots(units, true), units));
        // one slot short of a pass's units hangs the worker: refused, whatever the size
        if (units > 2) try std.testing.expect(slotsRefusal(&buf, units - 1, units) != null);
    }
    const w = slotsRefusal(&buf, 2, 4).?;
    try std.testing.expect(std.mem.indexOf(u8, w, "2 chunk slots") != null);
    try std.testing.expect(std.mem.indexOf(u8, w, "4 resume units (8192 rows, TF_DS_PASS_ROWS)") != null);
    const w8 = slotsRefusal(&buf, 4, 8).?;
    try std.testing.expect(std.mem.indexOf(u8, w8, "8 resume units (16384 rows, TF_DS_PASS_ROWS) needs 8 to 16") != null);
    try std.testing.expect(slotsRefusal(&buf, 1, 2) != null);
    try std.testing.expect(slotsRefusal(&buf, max_wk + 1, 1) != null);
}
