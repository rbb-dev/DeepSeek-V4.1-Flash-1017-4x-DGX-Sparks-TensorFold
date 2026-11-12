//! TF_DS_ENGRAM_PARTNER: the Engram records of a prompt step, read half by each of two column partners.
//!
//! Nodes g and g ^ 2 (the same rank in the two pairs of the 2D split) need the same hash columns' rows, so on every step they hold the same
//! list of cache misses and read every record twice. Here each reads the records at its own positions of that list (the pair's index picks
//! even or odd positions), packs them, sends them to the partner on the host lane (a header first, then the records), and scatters both
//! halves into the stage where a whole local read would have put them. The cache is filled by the caller from the stage in the list's
//! order, as before, so the two partners' caches stay alike.
//!
//! Correctness never depends on the lists being equal: the header carries the step, the position tag, the count, an order hash and whether
//! the node splits this step; when the partner's differs, or arrives late, or its records are late, this node reads the missing records
//! itself. A step always sends one header and one records event, so the partners' word counters stay equal whatever each decided.
//!
//! What a late partner costs is bounded by what reading its half here would cost: after its own half is read and sent, a node waits for
//! the partner at most as long as that read took (and a little for the crossing), then reads the rest. The worst step is so 1.5 times a
//! whole local read. A node that was let down backs off: it reads whole lists for the next steps (2, 4, 8, then 16 at most) and says so in
//! its headers, so a partner that is still splitting reads the rest at once instead of waiting for records that were not meant for it.

const std = @import("std");
const roci = @import("rocinante.zig");
const rows = @import("engram_rows.zig");

pub const Outcome = enum {
    split, // both halves exchanged
    local_mismatch, // the partner's header named another list: everything read here
    partner_declined, // the partner read its whole list this step (backing off): the rest read here
    skipped, // this node backed off: everything read here, nothing waited for
    local_header_late, // no header in time: everything read here
    partner_half_late, // the partner's records were late: its half read here
};

/// Reads the records of `ids` (row ids of the step's layer) into `dsts`, one record each.
pub const Reader = struct {
    ctx: *anyopaque,
    read: *const fn (ctx: *anyopaque, ids: []const i64, dsts: []const [*]u8) anyerror!void,
};

pub const Counters = struct {
    steps: u64 = 0,
    split: u64 = 0,
    mismatch: u64 = 0,
    header_late: u64 = 0,
    half_late: u64 = 0,
    skipped: u64 = 0, // steps this node read whole lists because it was backing off
    declined: u64 = 0, // steps the partner said it was reading its whole list
    records_read: u64 = 0, // records read from this node's drive
    records_got: u64 = 0, // records that came from the partner
    records_sent: u64 = 0, // records this node's half sent to the partner
    hdr_wait_ns: u64 = 0, // waiting for the partner's header, after this node's half was read
    rec_wait_ns: u64 = 0, // waiting for the partner's records, after this node's half was sent
    wait_max_ns: u64 = 0, // the longest single wait of either kind (a maximum: `since` keeps the later one)
    waits: u64 = 0, // waits made
    wait_looks: u64 = 0, // looks at the partner's words in them
    wait_naps: u64 = 0, // naps in them
    read_ns: u64 = 0, // reading this node's half of the steps that split
    full_ns: u64 = 0, // what reading the whole lists of those steps would have taken at the same rate (an estimate: read_ns scaled by the list)
    redo_ns: u64 = 0, // reading the rest of a list after a mismatch, a late header or late records
    copy_ns: u64 = 0, // moving the halves between the exchange buffers and the stage

    /// What happened since `base` (a copy taken earlier); the longest wait stays as it is.
    pub fn since(c: Counters, base: Counters) Counters {
        var d: Counters = c;
        d.steps = c.steps -% base.steps;
        d.split = c.split -% base.split;
        d.mismatch = c.mismatch -% base.mismatch;
        d.header_late = c.header_late -% base.header_late;
        d.half_late = c.half_late -% base.half_late;
        d.skipped = c.skipped -% base.skipped;
        d.declined = c.declined -% base.declined;
        d.records_read = c.records_read -% base.records_read;
        d.records_got = c.records_got -% base.records_got;
        d.records_sent = c.records_sent -% base.records_sent;
        d.hdr_wait_ns = c.hdr_wait_ns -% base.hdr_wait_ns;
        d.rec_wait_ns = c.rec_wait_ns -% base.rec_wait_ns;
        d.waits = c.waits -% base.waits;
        d.wait_looks = c.wait_looks -% base.wait_looks;
        d.wait_naps = c.wait_naps -% base.wait_naps;
        d.read_ns = c.read_ns -% base.read_ns;
        d.full_ns = c.full_ns -% base.full_ns;
        d.redo_ns = c.redo_ns -% base.redo_ns;
        d.copy_ns = c.copy_ns -% base.copy_ns;
        return d;
    }
};

/// The step's side of the exchange. One per node, used by the fill worker's thread only.
pub const Link = struct {
    t: *roci.Transport,
    arena: []u8, // the host lane's arena
    role: u1, // which positions of a list are this node's: its pair's index
    step: u64 = 0, // steps run: the next step's index (the same on both partners)
    slack_ns: u64 = 2 * std.time.ns_per_ms, // what the partner is given beyond the time this node's own half took (the crossing)
    late_streak: u5 = 0, // consecutive steps the partner was late
    skip: u8 = 0, // steps still to read whole (backing off)
    c: Counters = .{},
    scratch_ids: std.ArrayList(i64) = .empty,
    scratch_dsts: std.ArrayList([*]u8) = .empty,
    gpa: std.mem.Allocator,

    pub fn deinit(l: *Link) void {
        l.scratch_ids.deinit(l.gpa);
        l.scratch_dsts.deinit(l.gpa);
    }

    /// One wait on the partner's words, its length added to `into`, its looks and naps and the longest wait to the counters.
    fn wait(l: *Link, class: roci.Class, want: u64, deadline_ns: u64, into: *u64) !bool {
        var st: roci.Transport.HostWait = .{};
        defer {
            into.* += st.ns;
            l.c.waits += 1;
            l.c.wait_looks += st.looks;
            l.c.wait_naps += st.naps;
            l.c.wait_max_ns = @max(l.c.wait_max_ns, st.ns);
        }
        return l.t.waitHostStats(class, want, deadline_ns, &st);
    }

    /// Reads the `count` positions of the list that belong to the role `prole` into the stage.
    fn readRest(l: *Link, ids: []const i64, dsts: []const [*]u8, prole: u1, count: usize, reader: Reader) !void {
        if (count == 0) return;
        try l.scratch_ids.resize(l.gpa, count);
        try l.scratch_dsts.resize(l.gpa, count);
        for (0..count) |j| {
            l.scratch_ids.items[j] = ids[position(j, prole)];
            l.scratch_dsts.items[j] = dsts[position(j, prole)];
        }
        const t0 = nowNs();
        try reader.read(reader.ctx, l.scratch_ids.items, l.scratch_dsts.items);
        l.c.redo_ns += nowNs() - t0;
        l.c.records_read += count;
    }

    fn sendBuf(l: *Link, slot: usize) []u8 {
        const L = roci.layoutOf(.host);
        return l.arena[L.at(.rec_send) + slot * roci.host_slot_bytes ..][0..roci.host_slot_bytes];
    }
    fn landBuf(l: *Link, slot: usize) []u8 {
        const L = roci.layoutOf(.host);
        return l.arena[L.at(.rec_land) + slot * roci.host_slot_bytes ..][0..roci.host_slot_bytes];
    }
    fn hdrOut(l: *Link, slot: usize) []u8 {
        const L = roci.layoutOf(.host);
        return l.arena[L.at(.rec_hs) + slot * roci.host_hdr_bytes ..][0..roci.host_hdr_bytes];
    }
    fn hdrIn(l: *Link, slot: usize) []u8 {
        const L = roci.layoutOf(.host);
        return l.arena[L.at(.rec_hl) + slot * roci.host_hdr_bytes ..][0..roci.host_hdr_bytes];
    }
};

/// A step's header: what the partner must see to take part in the split.
const Header = struct { step: u64, tag: u64, count: u64, ord: u64, splits: bool };

fn putHeader(buf: []u8, h: Header) void {
    std.mem.writeInt(u64, buf[0..8], h.step, .little);
    std.mem.writeInt(u64, buf[8..16], h.tag, .little);
    std.mem.writeInt(u64, buf[16..24], h.count, .little);
    std.mem.writeInt(u64, buf[24..32], h.ord, .little);
    std.mem.writeInt(u64, buf[32..40], @intFromBool(h.splits), .little);
}

fn getHeader(buf: []const u8) Header {
    return .{
        .step = std.mem.readInt(u64, buf[0..8], .little),
        .tag = std.mem.readInt(u64, buf[8..16], .little),
        .count = std.mem.readInt(u64, buf[16..24], .little),
        .ord = std.mem.readInt(u64, buf[24..32], .little),
        .splits = std.mem.readInt(u64, buf[32..40], .little) != 0,
    };
}

/// Positions of a list of `m` that belong to `role` (even for 0, odd for 1): how many, and the position of the j-th.
pub fn mineCount(m: usize, role: u1) usize {
    return if (role == 0) (m + 1) / 2 else m / 2;
}

pub fn position(j: usize, role: u1) usize {
    return 2 * j + role;
}

/// One step on a node. `ids`/`dsts`: the step's misses, in the cache's order, `dsts[i]` where record i belongs in the stage.
/// `tag`: where the step is in the fill (the same on both partners). `rec`: the record size in bytes.
///
/// The header goes out and this node's half is read at once, without waiting for the partner's header: the header crosses while the
/// drive works, and a partner that has the same list (nearly always) is found out afterwards at no cost to the read. The records go
/// out as soon as they are read, whatever the header says (a partner that does not use them ignores them). When the headers differ
/// the other half is read here, so a mismatch costs a whole local read, as before.
pub fn splitRead(l: *Link, ids: []const i64, dsts: []const [*]u8, tag: u64, reader: Reader, rec: usize) !Outcome {
    std.debug.assert(rec == roci.host_rec);
    std.debug.assert(ids.len == dsts.len);
    const m = ids.len;
    if (m > 2 * roci.host_max_recs) return error.StepTooLong;
    const step = l.step;
    l.step += 1;
    l.c.steps += 1;
    const slot = step % roci.host_slots;
    const h = rows.missHashes(ids);
    const splits = l.skip == 0;
    putHeader(l.hdrOut(slot), .{ .step = step, .tag = tag, .count = m, .ord = h.ord, .splits = splits });
    try l.t.sendRec(.rech, @intCast(slot), 1);
    if (!splits) {
        // backing off: the whole list here, nothing waited for; the records event of zero keeps the words equal
        l.skip -= 1;
        l.c.skipped += 1;
        try l.t.sendRec(.rec, @intCast(slot), 0);
        const t0 = nowNs();
        if (m > 0) try reader.read(reader.ctx, ids, dsts);
        l.c.redo_ns += nowNs() - t0;
        l.c.records_read += m;
        return .skipped;
    }
    // my positions, read into the contiguous send buffer
    const mine = mineCount(m, l.role);
    const send = l.sendBuf(slot);
    try l.scratch_ids.resize(l.gpa, mine);
    try l.scratch_dsts.resize(l.gpa, mine);
    for (0..mine) |j| {
        l.scratch_ids.items[j] = ids[position(j, l.role)];
        l.scratch_dsts.items[j] = send[j * rec ..].ptr;
    }
    const t0 = nowNs();
    if (mine > 0) try reader.read(reader.ctx, l.scratch_ids.items, l.scratch_dsts.items);
    const took = nowNs() - t0;
    l.c.records_read += mine;
    l.c.read_ns += took;
    if (mine > 0) l.c.full_ns += @intCast(@as(u128, took) * m / mine);
    // the records out, then the node's half into the stage
    try l.t.sendRec(.rec, @intCast(slot), mine);
    l.c.records_sent += mine;
    const c0 = nowNs();
    for (0..mine) |j| @memcpy(dsts[position(j, l.role)][0..rec], send[j * rec ..][0..rec]);
    l.c.copy_ns += nowNs() - c0;
    // the partner, for as long as reading its half here would take
    const budget = took + l.slack_ns;
    const w0 = nowNs();
    var outcome: Outcome = .split;
    if (!try l.wait(.rech, step + 1, budget, &l.c.hdr_wait_ns)) {
        outcome = .local_header_late;
        l.c.header_late += 1;
    } else {
        const ph = getHeader(l.hdrIn(slot));
        if (ph.step != step or ph.tag != tag or ph.count != m or ph.ord != h.ord) {
            outcome = .local_mismatch;
            l.c.mismatch += 1;
        } else if (!ph.splits) {
            outcome = .partner_declined;
            l.c.declined += 1;
        }
    }
    const prole: u1 = l.role ^ 1;
    const theirs = m - mine;
    if (outcome == .split and m > 0) {
        const left = budget -| (nowNs() - w0);
        if (try l.wait(.rec, step + 1, @max(left, 1), &l.c.rec_wait_ns)) {
            const land = l.landBuf(slot);
            const c1 = nowNs();
            for (0..theirs) |j| @memcpy(dsts[position(j, prole)][0..rec], land[j * rec ..][0..rec]);
            l.c.copy_ns += nowNs() - c1;
            l.c.records_got += theirs;
            l.c.split += 1;
            l.late_streak = 0;
            return .split;
        }
        outcome = .partner_half_late;
        l.c.half_late += 1;
    }
    // everything else of the list is read here (a late partner's records may still land in the slot; nothing else uses it until the ring
    // comes round), and a partner that let this node down makes it back off
    if (outcome == .local_header_late or outcome == .partner_half_late) {
        l.late_streak +|= 1;
        l.skip = @as(u8, 1) << @min(l.late_streak, 4);
    }
    try l.readRest(ids, dsts, prole, theirs, reader);
    return outcome;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "positions: the two roles take the even and the odd positions of a list, together every one once" {
    for ([_]usize{ 0, 1, 2, 3, 10, 11, 12287, 24576 }) |m| {
        var seen = try std.testing.allocator.alloc(u8, m);
        defer std.testing.allocator.free(seen);
        @memset(seen, 0);
        for ([_]u1{ 0, 1 }) |role| {
            const n = mineCount(m, role);
            for (0..n) |j| {
                const p = position(j, role);
                try std.testing.expect(p < m);
                seen[p] += 1;
            }
        }
        for (seen) |c| try std.testing.expectEqual(@as(u8, 1), c);
        try std.testing.expect(mineCount(m, 0) + mineCount(m, 1) == m);
        try std.testing.expect(mineCount(m, 0) <= roci.host_max_recs and mineCount(m, 1) <= roci.host_max_recs);
    }
}
