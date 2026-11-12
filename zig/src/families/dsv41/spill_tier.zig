//! The DeepSeek-V4.1 spill tier on rank 0: core/prompt_cache's `Tier` over core/spill's store and spill_state's
//! packing. A restore read here reaches the other ranks through the landing (land.zig); the engine thread only
//! enqueues copies and polls their events, and never waits on the disk.
const std = @import("std");
const core = @import("core");
const engine_api = @import("engine_api");
pub const prompt_cache = engine_api.prompt_cache;
const state = @import("spill_state.zig");
const sync = @import("spill_sync.zig");
const wire = @import("spill_wire.zig");
const Star = @import("star.zig").Star;

const spill = core.spill;
const format = spill.format;
const pc = prompt_cache;
const Allocator = std.mem.Allocator;

pub const Digest = format.Digest;

/// Between two settles of operations some rank still has pending.
pub const settle_gap_ns = std.time.ns_per_ms;

/// Why a restore failed or was dropped, for the counters and the log.
pub const Failure = enum(u8) {
    none = 0,
    presence = 1, // the store lacks the point
    reserve = 2, // the destination could not be reserved
    read = 3, // a disk read failed
    crc = 4, // a block's CRC or digest did not match
    truncated = 5, // a file was shorter than its manifest
    missing = 6, // a file was gone
    cancelled = 7, // the request went away
    timeout = 8,
    unpack = 9, // the small state or a block did not install
    superseded = 10,
    shutdown = 11,
};

/// An Rx with no failing block.
pub const no_block: u32 = 0xffff_ffff;

/// Restores in flight at most (TF_DS_SPILL_RESTORES is checked against it).
pub const max_restores = 16;

/// The counters' copy's age at which a counts call refreshes it (a service and a flush refresh it anyway).
pub const counts_max_age_ns = std.time.ns_per_s;

/// A landing's step (land.zig): every rank holds the restored state, still landing, skipped by every rank, or failed.
pub const LandStep = @import("land.zig").Step;
/// The landing's own counts (land.zig's Lander), which the tier reports as /metrics' landings.
pub const LandCounts = @import("land.zig").Lander.Counts;

/// A probe's verdict on one kept state (probe.zig over every rank's digest): the ranks hold the same bytes, the stream
/// was busy (ask again at the next pump), the first rank that differs, or the probe could not run.
pub const ProbeVerdict = union(enum) { same, busy, differs: u16, failed: anyerror };

/// The small state's one section in a point (snapCopy's bytes, the layout's small planes in order).
pub const section_name = "dsv41.small";
/// Blocks a pump stages at most (GLM 0110's PUMP_BLOCKS).
pub const pump_blocks = 4;

/// The smallest staging cap a rank takes (TF_DS_SPILL_STAGING_MIB / --spill-staging-mib), on every rank alike.
pub const min_staging_mib: u32 = 16;

/// The tier's switches on rank 0, from engine_api.Spill (the one parse of TF_DS_SPILL_* and TF_SPILL_*: fromSpill).
/// Off unless `dir` and `gib > 0`.
pub const Settings = struct {
    dir: []const u8,
    gib: f64,
    min_free_gib: f64 = 50,
    quota: f64 = 0,
    min_tokens: u32 = 8192,
    turns: u32 = 4,
    flush_ms: u32 = 60_000,
    /// Pinned host staging for every write, read and small state together (the plan's 128 MiB a rank).
    staging_mib: u32 = 128,
    /// Exact staging bytes instead of staging_mib (0: staging_mib's; tests and measurements).
    staging_bytes: u64 = 0,
    /// Small-state slots for queued writes, and restores at once (each holds a small-state slot).
    smalls: u32 = 4,
    restores: u32 = 2,
    readers: u32 = 4,
    direct: bool = true,
    /// Pictures' points kept and found (TF_DS_IMAGE_CACHE, on unless "0"). Off: a prompt with pictures keeps
    /// nothing on disk and finds nothing there.
    image_cache: bool = true,
    /// Unpublished write bytes before a keep is refused (TF_DS_SPILL_BACKLOG_GIB).
    backlog_gib: f64 = 16,
    /// Earlier builds' stores kept beside this one's (TF_DS_SPILL_KEEP_BUILDS).
    keep_builds: u32 = 1,
    /// A sampled re-check of the ranks' bytes every N keeps (TF_DS_SPILL_PROBE_EVERY; 0: the start probe only).
    probe_every: u32 = 0,
    /// The store's writer and readers' I/O priority as an ioprio value (class << 13 | level; TF_DS_SPILL_IOPRIO; null:
    /// the process's).
    io_priority: ?u16 = null,

    /// The settings engine_api.Spill.fromEnv read: the one parse, range-checked there.
    pub fn fromSpill(sp: engine_api.Spill) Settings {
        return .{
            .dir = sp.dir orelse "",
            .gib = sp.gib,
            .min_free_gib = sp.min_free_gib,
            .quota = sp.quota,
            .min_tokens = sp.min_tokens,
            .turns = sp.turns,
            .flush_ms = sp.flush_ms,
            .staging_mib = sp.staging_mib,
            .smalls = sp.smalls,
            .restores = sp.restores,
            .readers = sp.readers,
            .image_cache = sp.image_cache,
            .backlog_gib = sp.backlog_gib,
            .keep_builds = sp.keep_builds,
            .probe_every = sp.probe_every,
            .io_priority = ioPriority(sp.ioprio),
        };
    }

    /// Linux's ioprio value of TF_DS_SPILL_IOPRIO: best effort (class 2) at its level, idle (class 3), none (null).
    pub fn ioPriority(p: engine_api.Spill.IoPrio) ?u16 {
        return switch (p.class) {
            .be => (2 << 13) | @as(u16, p.level),
            .idle => 3 << 13,
            .none => null,
        };
    }

    pub fn enabled(s: Settings) bool {
        return s.dir.len > 0 and s.gib > 0;
    }

    pub fn capBytes(s: Settings) u64 {
        return @intFromFloat(s.gib * (1 << 30));
    }

    pub fn floorBytes(s: Settings) u64 {
        return @intFromFloat(@max(s.min_free_gib, 0) * (1 << 30));
    }

    pub fn stagingBytes(s: Settings) u64 {
        if (s.staging_bytes > 0) return s.staging_bytes;
        return @as(u64, s.staging_mib) << 20;
    }

    pub fn backlogBytes(s: Settings) u64 {
        return @intFromFloat(@max(s.backlog_gib, 0) * (1 << 30));
    }
};

/// The staging cap split for a layout (slots.plan: small slots first, the rest half writes, half reads).
pub fn stagingPlan(s: Settings, l: *const state.StateLayout) !spill.slots.Plan {
    return spill.slots.plan(s.stagingBytes(), l.block_bytes, l.small_bytes, s.smalls, s.restores);
}

/// Events a rank needs (one a staging slot, stable for the store's life, and one for clears).
pub fn eventCount(p: spill.slots.Plan) usize {
    return p.write + p.read + p.small + p.restore + 1;
}

/// The store's owner layout and the wire's for a state layout.
pub fn storeLayout(l: *const state.StateLayout) format.OwnerLayout {
    return if (l.owner_layout == .owner_sharded) .owner_sharded else .replicated;
}

/// The store's namespace: its name (spill_identity.name: every rank's weights and the common inputs) and the format's
/// fields, the owner layout among them.
pub fn storeCompat(name: Digest, l: *const state.StateLayout) Digest {
    return format.compatDigest(name, storeLayout(l), @intCast(l.geo.block_tokens), l.block_bytes);
}

/// The id the prompt cache knows a point by: its key's first 8 bytes (the key is a SHA-256).
pub fn pointId(key: Digest) u64 {
    return std.mem.readInt(u64, key[0..8], .little);
}

pub fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A failed restore's reason as an error the prompt cache logs.
fn failureError(why: Failure) anyerror {
    return switch (why) {
        .presence => error.SpillNotOnEveryRank,
        .reserve => error.SpillReserve,
        .read => error.SpillRead,
        .crc => error.SpillDamaged,
        .truncated => error.SpillTruncated,
        .missing => error.SpillMissing,
        .cancelled => error.Cancelled,
        .timeout => error.SpillTimeout,
        .unpack => error.SpillUnpack,
        .shutdown => error.SpillShutdown,
        else => error.SpillAborted,
    };
}

/// Whether a failure is damage to the stored point (every rank then forgets it).
fn damage(why: Failure) bool {
    return switch (why) {
        .crc, .truncated, .missing, .unpack => true,
        else => false,
    };
}

fn readWhy(e: ?spill.store.ReadError) Failure {
    return switch (e orelse return .read) {
        .missing => .missing,
        .damaged, .small => .crc,
        .io => .read,
        .cancelled => .cancelled,
    };
}

// -- pictures (9014's content keys, recorded with their full digests) -----------------------------------------

comptime {
    // the prompt cache's media are the store's, byte for byte (Want.media and Kept.media go to the store as they are)
    std.debug.assert(@sizeOf(pc.Media) == @sizeOf(format.Media) and @alignOf(pc.Media) == @alignOf(format.Media));
    std.debug.assert(@offsetOf(pc.Media, "start") == @offsetOf(format.Media, "start"));
    std.debug.assert(@offsetOf(pc.Media, "len") == @offsetOf(format.Media, "len"));
    std.debug.assert(@offsetOf(pc.Media, "digest") == @offsetOf(format.Media, "digest"));
    std.debug.assert(pc.picture_ids == format.picture_ids);
}

/// The store's view of the prompt cache's media (the same bytes: checked above).
pub fn asFormat(m: []const pc.Media) []const format.Media {
    return @ptrCast(m);
}

/// Pictures a point records at most; a point with more is not written.
pub const max_note_media = 256;

/// A point's pictures as the store records them: the prompt's media that start before the cut (a prefix of them),
/// and the model id their raw positions hold (the placeholder, the same for every picture).
pub const Pictures = struct { media: []const format.Media = &.{}, placeholder: u32 = 0 };

/// The pictures of the point `keyed[0..at]` of a prompt whose pictures are `media` (all of them, in order) and whose
/// model ids are `raw`: null when the ids do not agree with the media (a picture `media` does not name, ids not made
/// from its digest, a raw id that is not the placeholder: the point is then not written), or past max_note_media.
/// A point no picture reaches is a text point (`media` empty).
pub fn pointPictures(keyed: []const u32, raw: []const u32, media: []const pc.Media, at: u32) ?Pictures {
    if (keyed.len < at or !pc.mediaKeys(keyed, at, media)) return null;
    var n: usize = 0;
    while (n < media.len and media[n].start < at) n += 1;
    if (n == 0) return .{};
    if (n > max_note_media or raw.len < at) return null;
    const placeholder = raw[media[0].start];
    if (placeholder >= pc.picture_ids) return null;
    var i: usize = 0;
    for (media[0..n]) |m| {
        if (!std.mem.eql(u32, keyed[i..m.start], raw[i..m.start])) return null;
        const stop = @min(@as(usize, m.start) + m.len, at);
        for (raw[m.start..stop]) |x| if (x != placeholder) return null;
        i = stop;
    }
    if (!std.mem.eql(u32, keyed[i..at], raw[i..at])) return null;
    return .{ .media = asFormat(media[0..n]), .placeholder = placeholder };
}

/// The store's lock times as the prompt cache's Timing: its buckets of up to 2^i us hold what Timing's of up to
/// 2^(10 + i) ns hold (1,024 ns >= 1 us), and what the store counted past its last bucket goes to the overflow.
pub fn asTiming(t: spill.store.LockTimes) pc.Timing {
    var out: pc.Timing = .{ .count = t.count, .sum_ns = t.sum_ns, .max_ns = t.max_ns };
    var in: u64 = 0;
    for (t.le_us, 0..) |n, i| {
        out.buckets[@min(i, pc.Timing.edges)] += n;
        in += n;
    }
    out.buckets[pc.Timing.edges] += t.count -| in;
    return out;
}

// -- one rank's side: its store, packing, staging events, writes and restores ----------------------------------------

/// What a keep stores on a rank.
pub const KeepArgs = struct {
    keyed: []const u32, // [0, at)
    key: Digest, // the caller's format.pointKeyChained, hashed once
    /// keyed[0..at]'s id chain (format.chain at the block tokens), from which `key` came: the store takes it instead of
    /// a second pass over the ids. Null: the store hashes them itself.
    chain: ?[]const Digest = null,
    at: u32,
    base: usize, // the kept state's extent
    snap: u64, // its Snap.id
    buf: u64, // its device buffer (the small state)
    lineage: format.Lineage,
    origin: ?format.Origin = null,
    shared: bool = false,
    group: u64 = 0,
    /// With pictures before `at`: the prompt's model ids [0, at) and the pictures (a point's media, in order).
    raw: ?[]const u32 = null,
    media: []const format.Media = &.{},
};

pub const Kept = enum { queued, stored, refused };

/// A restore's destination and progress on one rank.
pub const Rx = struct {
    id: u64,
    key: Digest,
    at: u32,
    blocks: u32,
    rid: ?spill.ReadId = null,
    dst: ?state.ExtentRef = null,
    snap: state.SnapRef = .{ .id = 0, .buf = 0 },
    copied: u32 = 0, // blocks whose copy into the destination is enqueued (in order)
    inflight: u32 = 0, // of those, copies whose event has not completed
    small: enum { none, pending, done } = .none,
    why: Failure = .none,
    bad_block: u32 = no_block,

    fn fail(rx: *Rx, why: Failure) void {
        if (rx.why == .none) {
            rx.why = why;
            rx.bad_block = rx.copied;
        }
    }
};

pub fn Side(comptime Dev: type) type {
    return struct {
        const Self = @This();
        pub const Packer = state.Packer(Dev);

        const Write = struct { job: spill.JobId, key: Digest, pin: state.ExtentRef, capturing: bool = true };
        const Copy = struct {
            ev: u32,
            what: union(enum) {
                stage: struct { t: spill.Ticket, pin: state.ExtentRef, pt: state.Ticket },
                capture: struct { job: spill.JobId, snap: u64 },
                read: struct { rx: u64, slot: spill.Slot },
                small: struct { rx: u64 },
            },
        };
        gpa: Allocator,
        io: std.Io,
        store: *spill.Store,
        layout: state.StateLayout,
        gens: state.Generations,
        pk: Packer,
        dev: Dev,
        plan: spill.slots.Plan,
        regions: [4][]align(spill.slots.alignment) u8,
        lists: [4][1]spill.slots.Buffer,
        events: []Dev.Event,
        bt: u32,
        copies: std.ArrayList(Copy) = .empty,
        writes: std.AutoHashMapUnmanaged(u64, Write) = .empty, // by Snap.id
        rxs: std.AutoHashMapUnmanaged(u64, *Rx) = .empty,
        reqs: []spill.StageReq,
        cached: pc.TierCounts = .{},
        /// When `cached` was last brought up to date (monotonic ns).
        refreshed_ns: u64 = 0,
        cancelled_shutdown: u64 = 0,
        /// The stop's bound for the store's close (Settings.flush_ms, SPILL_FLUSH_MS): the flush and the close share
        /// it, so the tier's whole stop stays within SPILL_FLUSH_MS.
        flush_ns: u64,
        /// The stop's flush deadline (monotonic), once a flush ran: the close gets what is left of it.
        stop_deadline: ?u64 = null,
        /// Writes waiting for the engine (store.pendingStage) as of the last service, plus keeps queued since: what a
        /// round's busy check reads instead of locking the store.
        stage_pending: usize = 0,
        // the store lock's wait and hold times as the prompt cache's Timing (refresh)
        lock_wait: pc.Timing = .{},
        lock_hold: pc.Timing = .{},
        // the store's reads as the prompt cache's Timing (refresh): each block's stages, each read's wall time
        block_open: pc.Timing = .{},
        block_read: pc.Timing = .{},
        block_crc: pc.Timing = .{},
        block_close: pc.Timing = .{},
        read_wall: pc.Timing = .{},

        /// Opens this rank's store under `s.dir` (core/spill names the namespace from `compat`) over `staging`
        /// (pinned, at least stagingPlan's bytes, 4 KiB aligned) and `events` (eventCount of them); both stay the
        /// caller's. `gens` is moved in. `max_tokens`: the longest point (the window).
        pub fn create(gpa: Allocator, io: std.Io, dev: Dev, layout: state.StateLayout, gens: state.Generations, s: Settings, rank: u32, compat: Digest, max_tokens: u32, staging: []align(spill.slots.alignment) u8, events: []Dev.Event, hooks: spill.store.Hooks) !*Self {
            const plan = try stagingPlan(s, &layout);
            if (staging.len < plan.bytes()) return error.StagingTooSmall;
            if (events.len < eventCount(plan)) return error.TooFewEvents;
            // one store serves every rank: the pool must be replicated
            if (layout.owner_layout != .replicated) return error.NotReplicated;
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .io = io, .store = undefined, .layout = layout, .gens = gens, .pk = undefined, .dev = dev, .plan = plan, .regions = undefined, .lists = undefined, .events = events, .bt = @intCast(layout.geo.block_tokens), .reqs = &.{}, .flush_ns = @as(u64, s.flush_ms) * std.time.ns_per_ms };
            self.pk = .{ .dev = dev, .layout = &self.layout, .gens = &self.gens, .rank = rank };
            var off: usize = 0;
            for ([_]usize{ plan.write * plan.block_slot, plan.read * plan.block_slot, plan.small * plan.small_slot, plan.restore * plan.small_slot }, 0..) |n, i| {
                self.regions[i] = @alignCast(staging[off..][0..n]);
                self.lists[i] = .{self.regions[i]};
                off += n;
            }
            self.reqs = try gpa.alloc(spill.StageReq, @max(plan.write, 1));
            errdefer gpa.free(self.reqs);
            const dir = try gpa.dupeSentinel(u8, s.dir, 0);
            defer gpa.free(dir);
            self.store = try spill.Store.open(gpa, io, .{
                .root = dir,
                .rank = rank,
                .compat = compat,
                .owner_layout = storeLayout(&layout),
                .limits = .{ .max_tokens = max_tokens, .block_tokens = self.bt, .block_bytes = layout.block_bytes, .own_part = rank, .max_small_bytes = std.mem.alignForward(u64, layout.small_bytes, 256) },
                .cap_bytes = s.capBytes(),
                .floor_bytes = s.floorBytes(),
                .quota = s.quota,
                .turns = s.turns,
                .min_tokens = s.min_tokens,
                .staging = .{ .write = &self.lists[0], .read = &self.lists[1], .small = &self.lists[2], .restore = &self.lists[3] },
                .readers = s.readers,
                .max_reads = s.restores,
                .direct = s.direct,
                .backlog_bytes = s.backlogBytes(),
                .keep_builds = s.keep_builds,
                .io_priority = s.io_priority,
                .hooks = hooks,
            });
            self.refresh();
            return self;
        }

        /// Waits every copy on the device, closes the store (unpublished points stay unpublished) and frees. The close
        /// is bounded by the stop's flush deadline when a flush ran, else by SPILL_FLUSH_MS. False: a store thread was
        /// in a disk call at the bound; the store is left to its threads, its staging kept until the process exits.
        pub fn close(self: *Self) bool {
            for (self.copies.items) |cp| self.dev.wait(self.events[cp.ev]);
            _ = self.retire();
            var it = self.rxs.valueIterator();
            while (it.next()) |rx| self.gpa.destroy(rx.*);
            self.rxs.deinit(self.gpa);
            const bound = if (self.stop_deadline) |d| d -| monotonicNs() else self.flush_ns;
            const r = self.store.closeWithin(bound);
            if (!r.clean and !@import("builtin").is_test) std.log.warn("dsv41 spill: the store did not close within {d} ms (a disk call still running): {d} queued metadata write(s) dropped; the store is left to its threads until the process exits", .{ bound / std.time.ns_per_ms, r.dropped });
            self.gens.deinit(self.gpa);
            self.copies.deinit(self.gpa);
            self.writes.deinit(self.gpa);
            self.gpa.free(self.reqs);
            self.gpa.destroy(self);
            return r.clean;
        }

        pub fn destroy(self: *Self) void {
            _ = self.close();
        }

        fn slotEvent(self: *const Self, slot: spill.Slot) u32 {
            const p = self.plan;
            const base: usize = switch (slot.pool) {
                .write => 0,
                .read => p.write,
                .small => p.write + p.read,
                .restore => p.write + p.read + p.small,
            };
            return @intCast(base + slot.index);
        }

        /// The event of the slot a store-given slice starts in (small and restore slots: Queued.small, SmallView).
        fn addrEvent(self: *const Self, pool: spill.slots.Pool, bytes: []const u8) u32 {
            const region = self.regions[@backingInt(pool)];
            const per = if (pool == .write or pool == .read) self.plan.block_slot else self.plan.small_slot;
            const i = (@intFromPtr(bytes.ptr) - @intFromPtr(region.ptr)) / per;
            return self.slotEvent(.{ .pool = pool, .index = @intCast(i) });
        }

        fn clearEvent(self: *const Self) u32 {
            return @intCast(eventCount(self.plan) - 1);
        }

        // -- generations ------------------------------------------------------------------------------------------

        /// Pool positions [lo, hi) are about to change (a prompt chunk, a round's rows, a copy's destination).
        pub fn bump(self: *Self, lo: usize, hi: usize) void {
            self.gens.bump(lo, @min(hi, self.gens.cap));
        }

        pub fn bumpSlot(self: *Self, slot: usize) void {
            self.gens.bumpSlot(slot);
        }

        // -- writes -----------------------------------------------------------------------------------------------

        /// The write of a kept state: its rows pinned as they stand, its small state copied from its Snap buffer now
        /// (packSnap, rule a: the Snap is live and owns `buf` at this call), its blocks staged later by `service`.
        pub fn keepPoint(self: *Self, k: KeepArgs) Kept {
            const at = k.at;
            if (at == 0 or at % self.bt != 0 or k.keyed.len < at) return .refused;
            if (self.writes.contains(k.snap)) return .stored;
            const pin = self.gens.pin(k.base, k.base + at) catch return .refused;
            self.writes.ensureUnusedCapacity(self.gpa, 1) catch return .refused;
            self.copies.ensureUnusedCapacity(self.gpa, 1) catch return .refused;
            const sections = [_]spill.store.SectionSpec{.{ .name = section_name, .len = self.layout.small_bytes }};
            const r = self.store.keep(.{ .keyed = k.keyed[0..at], .raw = if (k.raw) |x| x[0..at] else null, .media = k.media, .lineage = k.lineage, .origin = k.origin, .shared = k.shared, .group = k.group, .sections = &sections, .source = .{ .id = k.snap, .generation = pin.stamp }, .chain = k.chain });
            switch (r) {
                .stored => return .stored,
                .queued => |q| {
                    const ev = self.addrEvent(.small, q.small);
                    if (q.small.len < self.layout.small_bytes or q.sections.len != 1 or q.sections[0].offset != 0) {
                        self.store.captured(q.job, false);
                        return .refused;
                    }
                    _ = self.pk.packSnap(.{ .id = k.snap, .buf = k.buf }, q.small[0..self.layout.small_bytes], self.events[ev]) catch {
                        self.dev.wait(self.events[ev]); // (a copy that failed partway still recorded its event)
                        self.store.captured(q.job, false);
                        return .refused;
                    };
                    self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .capture = .{ .job = q.job, .snap = k.snap } } });
                    self.writes.putAssumeCapacity(k.snap, .{ .job = q.job, .key = k.key, .pin = pin });
                    self.stage_pending += 1; // (busy until a service has staged it)
                    return .queued;
                },
                else => return .refused,
            }
        }

        /// One staged block's pack copy (its event the slot's), or the store told it could not be made.
        fn packReq(self: *Self, req: spill.StageReq) void {
            const w = self.writes.get(req.source.id) orelse return self.store.staged(req.ticket, false);
            if (w.pin.stamp != req.source.generation or req.first % self.bt != 0) return self.store.staged(req.ticket, false);
            const ev = self.slotEvent(req.ticket.slot);
            self.copies.ensureUnusedCapacity(self.gpa, 1) catch return self.store.staged(req.ticket, false);
            const pt = self.pk.packBlocks(w.pin, req.first / self.bt, 1, req.dst, self.events[ev]) catch {
                self.dev.wait(self.events[ev]);
                return self.store.staged(req.ticket, false); // StaleExtent: the rows changed under the pin
            };
            self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .stage = .{ .t = req.ticket, .pin = w.pin, .pt = pt } } });
        }

        /// Kept state `snap`'s rows from token `from` on are about to change: its unpublished write stages what it
        /// still reads there now (the copies queued before the change, in stream order) or is cancelled.
        pub fn guard(self: *Self, snap: u64, from: u32) void {
            const w = self.writes.get(snap) orelse return;
            const r = self.store.guard(.{ .id = snap, .generation = w.pin.stamp }, from, self.reqs);
            for (self.reqs[0..r.staged]) |req| self.packReq(req);
        }

        /// Kept state `snap` is about to be freed: its write guarded from 0, its small-state capture waited for
        /// (rule b: the buffer's address cannot come back under a copy still reading it).
        pub fn dropSnap(self: *Self, snap: u64) void {
            self.guard(snap, 0);
            var i: usize = 0;
            while (i < self.copies.items.len) {
                const cp = self.copies.items[i];
                if (cp.what == .capture and cp.what.capture.snap == snap) {
                    self.dev.wait(self.events[cp.ev]);
                    _ = self.copies.swapRemove(i);
                    self.done(cp);
                } else i += 1;
            }
        }

        /// Whether kept state `snap`'s unpublished write still reads its rows (blocks not yet staged): rank 0's
        /// planner.setPending (a stale, move or carry of it is then preceded by a stage op).
        pub fn writePending(self: *Self, snap: u64) bool {
            const w = self.writes.get(snap) orelse return false;
            return self.store.pins(.{ .id = snap, .generation = w.pin.stamp }, 0);
        }

        /// writePending for many kept states at once (one store lock: Store.pinsAny), `out[i]` for `snaps[i]`.
        pub fn writesPending(self: *Self, snaps: []const u64, out: []bool) void {
            self.store.pinsAny(snaps, out);
            for (snaps, out) |id, *o| o.* = o.* and self.writes.contains(id);
        }

        /// Whether a write of point `key` is still queued here (not yet published, cancelled or failed).
        pub fn writing(self: *const Self, key: Digest) bool {
            var it = self.writes.valueIterator();
            while (it.next()) |w| if (std.mem.eql(u8, &w.key, &key)) return true;
            return false;
        }

        /// Cancel kept state `snap`'s unpublished write (rank 0 dropped its own).
        pub fn cancelWrite(self: *Self, snap: u64) void {
            const w = self.writes.get(snap) orelse return;
            _ = self.store.cancel(w.job);
        }

        /// Cancel the unpublished write of point `key`, whichever kept state wrote it (the four stores' drop).
        pub fn cancelKey(self: *Self, key: Digest) void {
            var it = self.writes.valueIterator();
            while (it.next()) |w| {
                if (std.mem.eql(u8, &w.key, &key)) _ = self.store.cancel(w.job);
            }
        }

        /// A copy's event completed: hand its result on.
        fn done(self: *Self, cp: Copy) void {
            switch (cp.what) {
                .stage => |x| self.store.staged(x.t, x.pt.belongsTo(x.pin)),
                .capture => |x| {
                    self.store.captured(x.job, true);
                    if (self.writes.getPtr(x.snap)) |w| w.capturing = false;
                },
                .read => |x| {
                    self.store.readRelease(x.slot);
                    if (self.rxs.get(x.rx)) |rx| rx.inflight -= 1;
                },
                .small => |x| if (self.rxs.get(x.rx)) |rx| {
                    rx.small = .done;
                },
            }
        }

        /// Every copy whose event completed, handed on (never waits); how many.
        pub fn retire(self: *Self) usize {
            var n: usize = 0;
            var i: usize = 0;
            while (i < self.copies.items.len) {
                const cp = self.copies.items[i];
                if (self.dev.done(self.events[cp.ev])) {
                    _ = self.copies.swapRemove(i);
                    self.done(cp);
                    n += 1;
                } else i += 1;
            }
            return n;
        }

        /// Between rounds and prompt chunks, and at idle: completed copies handed on, at most `max_blocks` more
        /// blocks staged within `budget_ns`, finished writes collected. Bounded; never waits. The copies retired and
        /// blocks staged (0: nothing moved).
        pub fn service(self: *Self, budget_ns: u64, max_blocks: usize) usize {
            const deadline = monotonicNs() +| budget_ns;
            var moved = self.retire();
            var left = max_blocks;
            while (left > 0 and monotonicNs() < deadline) {
                const n = self.store.stage(self.reqs[0..@min(self.reqs.len, left, 4)]);
                if (n == 0) break;
                for (self.reqs[0..n]) |req| self.packReq(req);
                left -|= n;
                moved += n;
            }
            self.collect();
            self.refresh();
            self.stage_pending = self.store.pendingStage();
            return moved;
        }

        /// The store's completions: finished writes leave `writes`.
        fn collect(self: *Self) void {
            var out: [16]spill.Completion = undefined;
            while (true) {
                const n = self.store.poll(&out);
                for (out[0..n]) |c| {
                    const job: spill.JobId = switch (c) {
                        inline else => |x| x.job,
                    };
                    var it = self.writes.iterator();
                    while (it.next()) |e| if (e.value_ptr.job == job) {
                        _ = self.writes.remove(e.key_ptr.*);
                        break;
                    };
                }
                if (n < out.len) break;
            }
        }

        /// Copies on the device, or writes waiting for the engine as of the last service plus any queued since. Read
        /// every round, so it takes no store lock.
        pub fn busy(self: *Self) bool {
            return self.copies.items.len > 0 or self.stage_pending > 0;
        }

        /// A clean stop: every queued write staged and drained by `deadline_ns` (monotonic); the points left
        /// unpublished (never published to make the count). The store's close gets what is left of the deadline.
        pub fn flush(self: *Self, deadline_ns: u64) u64 {
            self.stop_deadline = deadline_ns;
            while (monotonicNs() < deadline_ns and (self.store.pendingStage() > 0 or self.copies.items.len > 0)) {
                // nothing retired or staged (the device busy, or every write slot still held by a writer): sleep, never spin
                if (self.service(deadline_ns -| monotonicNs(), std.math.maxInt(usize)) == 0) std.Io.sleep(self.io, .fromMilliseconds(1), .awake) catch {};
            }
            for (self.copies.items) |cp| self.dev.wait(self.events[cp.ev]);
            _ = self.retire();
            self.collect();
            const d = self.store.drain(deadline_ns -| monotonicNs());
            self.cancelled_shutdown += d.unpublished;
            self.refresh();
            return d.unpublished;
        }

        // -- restores ---------------------------------------------------------------------------------------------

        pub fn newRx(self: *Self, id: u64, key: Digest, at: u32, blocks: u32) !*Rx {
            try self.rxs.ensureUnusedCapacity(self.gpa, 1);
            const rx = try self.gpa.create(Rx);
            rx.* = .{ .id = id, .key = key, .at = at, .blocks = blocks };
            self.rxs.putAssumeCapacity(id, rx);
            return rx;
        }

        /// Ready blocks before `upto` (null: all) copied into the destination, within `budget_ns`; never waits.
        pub fn rxStep(self: *Self, rx: *Rx, upto: ?u32, budget_ns: u64) void {
            const rid = rx.rid orelse return;
            const dst = rx.dst orelse return;
            const stop = @min(upto orelse rx.blocks, rx.blocks);
            const deadline = monotonicNs() +| budget_ns;
            while (rx.why == .none and rx.copied < stop) {
                var one: [1]spill.ReadyBlock = undefined;
                const got = self.store.readReady(rid, stop, &one);
                if (got.n == 0) {
                    if (got.status == .failed) rx.fail(readWhy(self.store.readFailure(rid).err));
                    return;
                }
                const b = one[0];
                const ev = self.slotEvent(b.slot);
                self.copies.ensureUnusedCapacity(self.gpa, 1) catch {
                    self.store.readRelease(b.slot);
                    return rx.fail(.unpack);
                };
                _ = self.pk.unpackBlocks(dst, b.index, 1, b.data, self.events[ev]) catch |e| {
                    self.dev.wait(self.events[ev]);
                    self.store.readRelease(b.slot);
                    return rx.fail(if (e == error.StaleExtent) .reserve else .unpack);
                };
                self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .read = .{ .rx = rx.id, .slot = b.slot } } });
                rx.inflight += 1;
                rx.copied = b.index + 1;
                if (monotonicNs() >= deadline) return;
            }
        }

        /// With every block copied: the small state, once read and checked, into the destination's Snap buffer.
        pub fn rxSmall(self: *Self, rx: *Rx) void {
            if (rx.why != .none or rx.small != .none or rx.copied < rx.blocks) return;
            const rid = rx.rid orelse return;
            const view = self.store.readSmallView(rid) orelse {
                const f = self.store.readFailure(rid);
                if (f.err != null) rx.fail(readWhy(f.err));
                return;
            };
            const l = self.layout.small_bytes;
            if (view.sections.len != 1 or !std.mem.eql(u8, view.sections[0].name.slice(), section_name) or view.sections[0].offset != 0 or view.sections[0].len != l or view.bytes.len < l)
                return rx.fail(.unpack);
            self.copies.ensureUnusedCapacity(self.gpa, 1) catch return rx.fail(.unpack);
            const ev = self.addrEvent(.restore, view.bytes);
            _ = self.pk.unpackSnap(view.bytes[0..l], rx.snap, self.events[ev]) catch {
                self.dev.wait(self.events[ev]);
                return rx.fail(.unpack);
            };
            self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .small = .{ .rx = rx.id } } });
            rx.small = .pending;
        }

        /// Every block and the small state installed (their events completed).
        pub fn rxReady(rx: *const Rx) bool {
            return rx.why == .none and rx.copied == rx.blocks and rx.inflight == 0 and rx.small == .done;
        }

        /// Nothing moved the destination since it was reserved (Generations.valid of every unpack).
        pub fn rxValid(self: *const Self, rx: *const Rx) bool {
            const dst = rx.dst orelse return false;
            return self.gens.current(dst);
        }

        /// Every copy of `rx` on the device waited for and handed on.
        pub fn rxQuiesce(self: *Self, rx: *const Rx) void {
            var i: usize = 0;
            while (i < self.copies.items.len) {
                const cp = self.copies.items[i];
                const mine = switch (cp.what) {
                    .read => |x| x.rx == rx.id,
                    .small => |x| x.rx == rx.id,
                    else => false,
                };
                if (mine) {
                    self.dev.wait(self.events[cp.ev]);
                    _ = self.copies.swapRemove(i);
                    self.done(cp);
                } else i += 1;
            }
        }

        /// Every copy of `rx` on the device completed and handed on by `deadline_ns` (monotonic), polling its events
        /// (never an unbounded device wait); false when one is still running at the deadline.
        pub fn rxSettle(self: *Self, rx: *const Rx, deadline_ns: u64) bool {
            while (true) {
                _ = self.retire();
                const left = for (self.copies.items) |cp| {
                    const mine = switch (cp.what) {
                        .read => |x| x.rx == rx.id,
                        .small => |x| x.rx == rx.id,
                        else => false,
                    };
                    if (mine) break true;
                } else false;
                if (!left) return true;
                if (monotonicNs() >= deadline_ns) return false;
                std.Io.sleep(self.io, .fromMicroseconds(200), .awake) catch {};
            }
        }

        /// A restore ends without publishing: its copies waited for, the point forgotten on this rank when
        /// `forget` (with this rank's own damaged/missing block lists), its reads ended, its destination cleared to
        /// fresh-pool bytes (scale sentinel rows) unless the rows already went to someone else.
        pub fn rxEnd(self: *Self, rx: *Rx, outcome: spill.Store.Outcome, forget: bool) void {
            self.rxQuiesce(rx);
            if (rx.rid) |rid| {
                if (forget) {
                    const f = self.store.readFailure(rid);
                    self.store.forget(rx.key, f.bad, f.lost);
                }
                _ = self.store.endRead(rid, outcome) catch {};
            } else if (forget) self.store.forget(rx.key, &.{}, &.{});
            if (rx.dst) |dst| {
                const ev = self.events[self.clearEvent()];
                if (self.pk.clearExtent(dst, ev)) |_| self.dev.wait(ev) else |_| {}
            }
            self.rxDrop(rx);
        }

        /// A committed restore's lineage (gpa-owned): its reads ended as committed. Its destination stays.
        pub fn rxCommit(self: *Self, rx: *Rx) !spill.ReadDone {
            const rid = rx.rid orelse return error.Incomplete;
            const done_ = try self.store.endRead(rid, .commit) orelse return error.Incomplete;
            return done_;
        }

        pub fn rxDrop(self: *Self, rx: *Rx) void {
            _ = self.rxs.remove(rx.id);
            self.gpa.destroy(rx);
        }

        // -- counters ---------------------------------------------------------------------------------------------

        /// The cached counters brought up to date (from service and flush: counts() never locks the store).
        pub fn refresh(self: *Self) void {
            self.refreshed_ns = monotonicNs();
            const c = self.store.counts();
            self.cached = .{
                .points = c.points,
                .blocks = c.blocks,
                .stored_bytes = c.stored_bytes,
                .cap_bytes = self.store.cfg.cap_bytes,
                .disk_free_bytes = c.disk_free_bytes,
                .staging_bytes = c.staging_bytes,
                .staging_peak = c.staging_peak_bytes,
                .writes_pending = c.pending_jobs,
                .writes_queued = c.queued,
                .writes_committed = c.committed,
                .writes_cancelled = c.cancelled,
                .writes_cancelled_guard = c.cancelled_guard + c.cancelled_source,
                .writes_cancelled_staging = c.no_staging + c.backlog_refused,
                .writes_cancelled_shutdown = self.cancelled_shutdown,
                .writes_failed = c.write_failed,
                .forced_staging = c.guard_staged,
                .no_space = c.no_space,
                .dedup_blocks = c.dedup_blocks,
                .dedup_bytes = c.dedup_bytes,
                .written_bytes = c.written_bytes + c.written_small_bytes,
                .crc_errors = c.crc_errors,
                .size_errors = c.size_errors,
                .missing_errors = c.missing_errors,
                .reads = c.loaded,
                .read_bytes = c.loaded_bytes,
                .read_failed = c.load_failed,
                .evicted_turns = c.pruned_turns + c.trimmed_turns,
                .evicted_forks = c.pruned_forks,
                .evicted_conversations = c.pruned_conversations,
                .evicted_shared = c.pruned_shared,
                .quota_trims = c.quota_trims,
                .recovered_points = c.recovered_points,
                .dropped_temporaries = c.temps_removed + c.recovered_dropped,
                .direct_refused = c.direct_refused,
                .recover_deferred = c.recover_deferred,
                .orphans_removed = c.orphans_removed,
                .builds_pruned = c.builds_pruned,
                .quarantined = c.quarantined,
                .marker_failed = c.marker_failed,
                .ioprio_refused = c.ioprio_refused,
            };
            self.lock_wait = asTiming(c.lock_wait);
            self.lock_hold = asTiming(c.lock_hold);
            self.block_open = asTiming(c.block_open);
            self.block_read = asTiming(c.block_read);
            self.block_crc = asTiming(c.block_crc);
            self.block_close = asTiming(c.block_close);
            self.read_wall = asTiming(c.read_wall);
        }
    };
}

// -- rank 0 ---------------------------------------------------------------------------------------------------------

/// A restore's destination as the family made it: a fresh kept state (lanes.zig: a Snap with its extent reserved, in
/// `snaps` but not usable until installed) and its device buffer.
pub const Dest = struct { saved: *anyopaque, id: u64, base: usize, buf: u64 };
/// A kept state as the tier needs it.
pub const SnapInfo = struct { id: u64, base: usize, at: usize, buf: u64, slot: u32 };

/// Rank 0's family side (lanes.zig's Lanes).
pub const Family = struct {
    ptr: *anyopaque,
    /// A destination for a restore of `at` positions (placement, its buffer); an error: no room now. `room`: the
    /// positions the request waiting on it needs from the same base (its prompt; >= at), a hint: the family holds
    /// that much free past the rows while room allows, so the resume runs in place instead of copying the extent.
    reserve: *const fn (ptr: *anyopaque, at: usize, room: usize) anyerror!Dest,
    /// The restored state is installed on every rank: usable from now (Lanes.usable).
    install: *const fn (ptr: *anyopaque, saved: *anyopaque) void,
    /// A restore that did not commit: its state removed and freed.
    discard: *const fn (ptr: *anyopaque, saved: *anyopaque) void,
    info: *const fn (ptr: *anyopaque, saved: *anyopaque) SnapInfo,
    /// The restored state (rank 0 holds it: read, committed, not yet installed) given to every other rank, a step at
    /// a time within `budget_ns` (frame kind snap_land; lanes.zig over land.zig); discard aborts a landing in flight.
    /// Null: one rank, nothing to land (installed at once).
    land: ?*const fn (ptr: *anyopaque, saved: *anyopaque, budget_ns: u64) LandStep = null,
    /// Every rank's bytes of kept state `saved` compared (frame kind snap_land's probe; lanes.zig over probe.zig), only
    /// at an idle stream (busy otherwise). Null: one rank, nothing to compare.
    probe: ?*const fn (ptr: *anyopaque, saved: *anyopaque) ProbeVerdict = null,
    /// The family's landing steps that did work, timed on rank 0's engine thread (lanes.zig: an open with its ready
    /// vote, one piece's broadcast, an end); the tier reports them as /metrics' land_piece. Null: none (one rank).
    land_time: ?*const pc.Timing = null,
    /// The family's landing counts (lanes.zig: its Lander's): landed, skipped, aborted, failed, pieces, bytes, read live
    /// by the tier's counts. Null: none (one rank).
    land_counts: ?*const LandCounts = null,
    /// Every rank keeps its own store (spill_sync): a snap_sync frame's bytes (a sub-op byte, then its fields) sent to every
    /// other rank in the ordered stream. With it the family has no `land`: a restore is read by every rank from its own disk.
    sync_send: ?*const fn (ptr: *anyopaque, bytes: []const u8) anyerror!void = null,
    /// Rank 0's stream is at a point where a settle (a gather every rank enters) may run: idle, or a prompt pass's boundary.
    sync_idle: ?*const fn (ptr: *anyopaque) bool = null,
};

pub fn Leader(comptime Dev: type) type {
    return struct {
        const Self = @This();
        pub const S = Side(Dev);

        /// A restore: reading (`rx`), then, read and committed, landing on the other ranks (`done` holds its lineage).
        const Job = struct {
            rx: ?*Rx,
            point: pc.Point,
            saved: *anyopaque,
            key: Digest,
            blocks: u32 = 0,
            done: ?spill.ReadDone = null,
            t_begin: u64 = 0, // monotonic ns: its begin, and when its blocks and small state were on the device
            t_copied: u64 = 0,
            vote: enum { none, adopt, drop } = .none, // the four ranks' decision on a restore (spill_sync)
        };

        /// The four stores' state (TF_DS_SSD_EVERY_RANK): the protocol's peer and leader, the decisions a call made and the
        /// body of a keep being built.
        const Sync = struct {
            peer: sync.Peer,
            lead: sync.Leader,
            out: std.ArrayList(sync.Decided) = .empty,
            body: std.ArrayList(u8) = .empty,
            /// When the last settle ran (monotonic ns): operations still pending on some rank are asked again after a millisecond, not at every poll.
            last_ns: u64 = 0,
        };
        const PointRef = struct { key: Digest, at: u32, room: u32 = 0 }; // room: the last lookup's prompt length

        gpa: Allocator,
        side: *S,
        fam: Family,
        min_tokens: u32,
        image_cache: bool, // Settings.image_cache: pictures' points kept and found
        pictures_kept: u64 = 0, // points written with pictures before their cut
        pictures_refused: u64 = 0, // keeps and lookups with pictures turned away (their ids and media disagree, or image cache off)
        max_restores: u32, // restores in flight at most (Settings.restores, the store's reads; <= max_restores)
        flush_ms: u32,
        jobs: std.AutoHashMapUnmanaged(u64, *Job) = .empty,
        next_job: u64 = 1,
        points: std.AutoHashMapUnmanaged(u64, PointRef) = .empty,
        fwd: std.AutoHashMapUnmanaged(u64, [16]u8) = .empty, // lineage names: the cache's u64s to the store's
        rev: std.AutoHashMapUnmanaged([16]u8, u64) = .empty,
        next_name: u64 = 0xC000_0000_0000_0000,
        boot: [16]u8,
        lin: std.ArrayList(pc.Fork) = .empty, // a ready restore's lineage (valid during its poll)
        presence_refused: u64 = 0,
        misses: u64 = 0,
        /// Why the tier turned off at run time (the first reason kept; null: on). Off stops keeps, lookups and
        /// restores; guards, bumps, pins, staging and drops' capture waits go on.
        off: ?[]const u8 = null,
        off_buf: [64]u8 = undefined,
        /// The sampled re-check: every probe_every-th keep is due, probed at the next pump on an idle stream.
        probe_every: u32 = 0,
        keeps_since: u32 = 0,
        due: ?*anyopaque = null,
        probe_runs: u64 = 0,
        probe_failed: u64 = 0,
        /// Every probe that ran (the start probe through noteProbe, sampled re-checks at pumps), for /metrics.
        probe_time: pc.Timing = .{},
        // each restore: its blocks and small state on the device, its landing on the other ranks, the whole
        restore_copy: pc.Timing = .{},
        restore_land: pc.Timing = .{},
        restore_ready: pc.Timing = .{},
        /// Restored points whose state is not used yet, by point id: the store keeps a quarantine marker for each (a
        /// crash in its first pass forgets the point at the next open) until restored(key), called once for each
        /// committed restore at the state's first resume (touch), its release, a flush or the tier's end.
        unconfirmed: std.AutoHashMapUnmanaged(u64, Unconfirmed) = .empty,
        /// Set when every rank keeps its own store (enableSync); null: one store on rank 0.
        sy: ?*Sync = null,

        /// The family states restored (by Snap id) and their points: a state the prompt cache drops unadopted (a
        /// usable twin won) was never used, so dropSnap confirms its point.
        installed: std.AutoHashMapUnmanaged(u64, u64) = .empty,

        const Unconfirmed = struct { key: Digest, holds: u32 };

        const vt: pc.Tier.VTable = .{
            .lookup = lookupFn,
            .begin = beginFn,
            .full = fullFn,
            .poll = pollFn,
            .cancel = cancelFn,
            .keep = keepFn,
            .release = releaseFn,
            .share = shareFn,
            .touch = touchFn,
            .forget = forgetFn,
            .pump = pumpFn,
            .busy = busyFn,
            .flush = flushFn,
            .counts = countsFn,
            .off = offFn,
            .times = timesFn,
        };

        /// `side` is moved in (destroy closes it). `boot` salts this process's lineage names. TF_DS_SPILL_RESTORES
        /// past max_restores is refused here, not clamped.
        pub fn create(gpa: Allocator, side: *S, fam: Family, s: Settings, boot: [16]u8) !*Self {
            if (s.restores == 0 or s.restores > max_restores) return error.BadSpillSetting;
            const self = try gpa.create(Self);
            self.* = .{ .gpa = gpa, .side = side, .fam = fam, .min_tokens = s.min_tokens, .image_cache = s.image_cache, .max_restores = s.restores, .flush_ms = s.flush_ms, .boot = boot, .probe_every = s.probe_every };
            return self;
        }

        /// Cancels what is still restoring, closes the store (Side.close, bounded by the stop's flush deadline) and
        /// frees. False: the store's threads were still in a disk call, the store is left to them, and the caller keeps
        /// the staging and the events until the process exits.
        pub fn close(self: *Self) bool {
            var it = self.jobs.valueIterator();
            while (it.next()) |j| self.endJob(j.*, .cancelled, false);
            self.jobs.deinit(self.gpa);
            self.confirmAll(); // a clean end: nothing crashed in a restored state's first use
            self.unconfirmed.deinit(self.gpa);
            self.installed.deinit(self.gpa);
            if (self.sy) |sy| {
                sy.lead.deinit();
                sy.peer.deinit();
                sy.out.deinit(self.gpa);
                sy.body.deinit(self.gpa);
                self.gpa.destroy(sy);
            }
            const clean = self.side.close();
            self.points.deinit(self.gpa);
            self.fwd.deinit(self.gpa);
            self.rev.deinit(self.gpa);
            self.lin.deinit(self.gpa);
            self.gpa.destroy(self);
            return clean;
        }

        pub fn destroy(self: *Self) void {
            _ = self.close();
        }

        /// Every rank keeps its own store from now: this Leader decides, the other ranks carry the decisions out (spill_sync). The
        /// ranks' start-up reconcile (`reconcile`) has run, and left `peer.visible` the entries all four hold. `fam.sync_send`
        /// must be set.
        pub fn enableSync(self: *Self, star: Star, cfg: sync.Digest, visible_from: *sync.Peer) !void {
            const sy = try self.gpa.create(Sync);
            sy.* = .{ .peer = sync.Peer.init(self.gpa, star, 0, self.syncDisk()), .lead = undefined };
            sy.peer.cfg = cfg;
            sy.peer.visible = visible_from.visible;
            visible_from.visible = .empty;
            sy.lead = .{ .peer = &sy.peer, .tx = .{ .ptr = self, .send = syncTx } };
            self.sy = sy;
        }

        /// The Disk of this rank's own store for the protocol's Peer on rank 0 (the reconcile before the Leader exists uses it too).
        pub fn syncDisk(self: *Self) sync.Disk {
            return .{ .ptr = self, .keep = dkKeep, .keepReport = dkKeepReport, .read = dkRead, .readReport = dkReadReport, .finishRead = dkFinish, .remove = dkRemove, .list = dkList, .damage = dkDamage };
        }

        fn syncTx(ptr: *anyopaque, frame: []const u8) anyerror!void {
            const self = of(ptr);
            const send = self.fam.sync_send orelse return error.NoFollowers;
            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(self.gpa);
            try buf.append(self.gpa, @backingInt(wire.SubOp.proto));
            try buf.appendSlice(self.gpa, frame);
            try send(self.fam.ptr, buf.items);
        }

        // the Disk: rank 0's keep and restore are begun by keepFn and beginFn as always; these report on them
        fn dkKeep(p: *anyopaque, key: sync.Key, _: []const u8) bool {
            const self = of(p);
            return self.side.writing(key) or self.side.store.presence(key);
        }

        fn dkKeepReport(p: *anyopaque, key: sync.Key) sync.Report {
            const self = of(p);
            if (self.side.writing(key)) return .{ .status = .pending };
            const d = self.side.store.pointDigest(key) orelse return .{ .status = .fail };
            return .{ .status = .ok, .digest = d };
        }

        fn dkRead(p: *anyopaque, _: sync.Key, id: u64, _: []const u8) bool {
            return of(p).jobs.contains(id);
        }

        fn dkReadReport(p: *anyopaque, id: u64) sync.Report {
            const self = of(p);
            const job = self.jobs.get(id) orelse return .{ .status = .fail };
            const rx = job.rx orelse return .{ .status = .fail };
            _ = self.side.retire();
            self.side.rxStep(rx, null, 0);
            self.side.rxSmall(rx);
            if (rx.why != .none) return .{ .status = .fail };
            if (!S.rxReady(rx)) return .{ .status = .pending };
            if (!self.side.rxValid(rx)) return .{ .status = .fail };
            const d = self.side.store.pointDigest(rx.key) orelse return .{ .status = .fail };
            return .{ .status = .ok, .digest = d };
        }

        fn dkFinish(p: *anyopaque, id: u64, adopt: bool) void {
            const self = of(p);
            if (self.jobs.get(id)) |job| job.vote = if (adopt) .adopt else .drop;
        }

        fn dkRemove(p: *anyopaque, key: sync.Key) void {
            const self = of(p);
            self.side.cancelKey(key);
            self.side.store.forget(key, &.{}, &.{});
        }

        fn dkDamage(p: *anyopaque, key: sync.Key) void {
            _ = of(p).side.store.damage(key);
        }

        /// The drill's faults rank 0 carries out (TF_DS_SSD_FAULT); after `enableSync`.
        pub fn armFault(self: *Self, f: sync.Armed) void {
            if (self.sy) |sy| sy.peer.fault = f;
        }

        fn dkList(p: *anyopaque, g: Allocator, out: *std.ArrayList(sync.Listed)) anyerror!void {
            const self = of(p);
            var pts: std.ArrayList(spill.store.PointEntry) = .empty;
            defer pts.deinit(g);
            try self.side.store.pointList(g, &pts);
            for (pts.items) |e| try out.append(g, .{ .key = e.key, .digest = e.digest });
        }

        /// What a settle or a send decided, applied to the tier's own tables: an entry dropped on all four is not a point here.
        fn syncApply(self: *Self) void {
            const sy = self.sy orelse return;
            for (sy.out.items) |d| switch (d.kind) {
                .keep_dropped, .restore_dropped => _ = self.points.remove(pointId(d.key)),
                .kept, .adopt => {},
            };
            sy.out.clearRetainingCapacity();
            if (sy.peer.down) self.setOff("a rank's link ended: the four stores can no longer be kept in step");
        }

        /// The operations in flight settled when the stream is at a point every rank can enter the gather.
        fn syncSettle(self: *Self) void {
            const sy = self.sy orelse return;
            if (sy.peer.pending() == 0) return;
            const idle = self.fam.sync_idle orelse return;
            if (!idle(self.fam.ptr)) return;
            const now = monotonicNs();
            if (now -% sy.last_ns < settle_gap_ns) return;
            sy.last_ns = now;
            sy.lead.settle(&sy.out) catch {};
            self.syncApply();
        }

        pub fn tier(self: *Self) pc.Tier {
            return .{ .ptr = self, .vtable = &vt };
        }

        fn of(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        /// Turn the tier off for the rest of the process (the probe's verdict), naming why: logged once.
        pub fn setOff(self: *Self, why: []const u8) void {
            if (self.off != null) return;
            self.off = why;
            if (!@import("builtin").is_test) std.log.err("dsv41 spill: the SSD tier is off for this process: {s}", .{why});
        }

        fn offFn(ptr: *anyopaque) ?[]const u8 {
            return of(ptr).off;
        }

        /// The tier's own times for /metrics: the store lock's wait and hold (as of the last service).
        fn timesFn(ptr: *anyopaque) pc.TierTimes {
            const self = of(ptr);
            const sd = self.side;
            return .{ .lock_wait = sd.lock_wait, .lock_hold = sd.lock_hold, .probe = self.probe_time, .land_piece = if (self.fam.land_time) |t| t.* else .{}, .block_open = sd.block_open, .block_read = sd.block_read, .block_crc = sd.block_crc, .block_close = sd.block_close, .read_wall = sd.read_wall, .restore_copy = self.restore_copy, .restore_land = self.restore_land, .restore_ready = self.restore_ready };
        }

        /// The start probe's time and run (native.zig runs it before the Leader exists).
        pub fn noteProbe(self: *Self, ns: u64) void {
            self.probe_time.observe(ns);
            self.probe_runs += 1; // (it passed: a start probe that differs leaves the tier off, no Leader made)
        }

        /// A restored point's state was used (or is gone, or the tier ends cleanly): its quarantine marker goes.
        fn confirm(self: *Self, point: u64) void {
            const kv = self.unconfirmed.fetchRemove(point) orelse return;
            for (0..kv.value.holds) |_| self.side.store.restored(kv.value.key);
        }

        fn confirmAll(self: *Self) void {
            var it = self.unconfirmed.valueIterator();
            while (it.next()) |u| for (0..u.holds) |_| self.side.store.restored(u.key);
            self.unconfirmed.clearRetainingCapacity();
        }

        /// A keep counted for the sampled re-check: every probe_every-th becomes due.
        fn noteKeep(self: *Self, saved: *anyopaque) void {
            if (self.probe_every == 0) return;
            self.keeps_since += 1;
            if (self.keeps_since < self.probe_every) return;
            self.keeps_since = 0;
            self.due = saved;
        }

        /// The due probe, run at a pump (an idle stream only: busy keeps it due). A rank whose bytes differ turns the
        /// tier off, named; nothing fails.
        fn probeDue(self: *Self) void {
            if (self.off != null) return;
            const saved = self.due orelse return;
            const f = self.fam.probe orelse {
                self.due = null;
                return;
            };
            const t0 = monotonicNs();
            const verdict = f(self.fam.ptr, saved);
            if (verdict != .busy) self.probe_time.observe(monotonicNs() -| t0);
            switch (verdict) {
                .busy => return,
                .same => self.probe_runs += 1,
                .differs => |rank| {
                    self.probe_runs += 1;
                    self.probe_failed += 1;
                    self.setOff(std.fmt.bufPrint(&self.off_buf, "probe: rank {d} differs", .{rank}) catch "probe: a rank differs");
                },
                .failed => |err| if (!@import("builtin").is_test) std.log.warn("dsv41 spill: a probe could not run ({s}); the next one tries again", .{@errorName(err)}),
            }
            self.due = null;
        }

        // -- lineage names ----------------------------------------------------------------------------------------

        pub fn name16(self: *const Self, n: u64) [16]u8 {
            if (self.fwd.get(n)) |x| return x;
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            h.update("tf-dsv41-lineage");
            h.update(&self.boot);
            h.update(std.mem.asBytes(&n));
            return h.finalResult()[0..16].*;
        }

        fn nameOf(self: *Self, n: [16]u8) !u64 {
            if (self.rev.get(n)) |x| return x;
            try self.fwd.ensureUnusedCapacity(self.gpa, 1);
            try self.rev.ensureUnusedCapacity(self.gpa, 1);
            const id = self.next_name;
            self.next_name += 1;
            self.fwd.putAssumeCapacity(id, n);
            self.rev.putAssumeCapacity(n, id);
            return id;
        }

        /// The store's lineage of a cache lineage (forks in `out`).
        fn toLineage(self: *Self, forks: []const pc.Fork, out: *std.ArrayList(format.Fork)) !format.Lineage {
            if (forks.len == 0) {
                self.next_name += 1;
                return .{ .root = self.name16(self.next_name) };
            }
            for (forks[1..]) |f| try out.append(self.gpa, .{ .name = self.name16(f.name), .at = f.at });
            return .{ .root = self.name16(forks[0].name), .part = null, .forks = out.items };
        }

        fn fromLineage(self: *Self, l: format.Lineage) ![]const pc.Fork {
            self.lin.clearRetainingCapacity();
            try self.lin.append(self.gpa, .{ .name = try self.nameOf(l.root), .at = 0 });
            for (l.forks) |f| try self.lin.append(self.gpa, .{ .name = try self.nameOf(f.name), .at = f.at });
            return self.lin.items;
        }

        // -- lanes.zig's guards -----------------------------------------------------------------------------------

        /// Kept state `snap`'s rows from token `from` on are about to change (placeKept victim, staleOver): stage or
        /// cancel its write before the change.
        pub fn guardSnap(self: *Self, snap: u64, from: u32) void {
            self.side.guard(snap, from);
            // the other ranks' write of it stages what it still reads before the rows change (the frame precedes the change)
            if (self.sy) |sy| if (!sy.peer.down) if (self.fam.sync_send) |send| {
                var b: [1 + wire.Guard.len]u8 = undefined;
                b[0] = @backingInt(wire.SubOp.guard);
                (wire.Guard{ .snap = snap, .from = from }).encode(b[1..][0..wire.Guard.len]);
                send(self.fam.ptr, &b) catch {
                    sy.peer.down = true;
                };
            };
        }

        /// Kept state `snap` is about to be freed (snapDropFn, the store's release): guarded from 0, its small-state
        /// capture waited for.
        pub fn dropSnap(self: *Self, snap: u64) void {
            self.guardSnap(snap, 0);
            self.side.dropSnap(snap);
            if (self.installed.fetchRemove(snap)) |kv| self.confirm(kv.value); // (dropped unused: nothing crashed in it)
        }

        pub fn bump(self: *Self, lo: usize, hi: usize) void {
            self.side.bump(lo, hi);
        }

        pub fn bumpSlot(self: *Self, slot: usize) void {
            self.side.bumpSlot(slot);
        }

        /// Rank 0's own write of kept state `snap` still reads its rows (the planner's pending flag).
        pub fn writePending(self: *Self, snap: u64) bool {
            return self.side.writePending(snap);
        }

        pub fn writesPending(self: *Self, snaps: []const u64, out: []bool) void {
            self.side.writesPending(snaps, out);
        }

        // -- the Tier ---------------------------------------------------------------------------------------------

        fn lookupFn(ptr: *anyopaque, want: *const pc.Want) ?pc.Point {
            const self = of(ptr);
            if (self.off != null) return null;
            // pictures: every one accounted for by its full identity (the store then matches spans and digests before
            // each point's cut), or no point at all; a missing or mismatched identity is a miss
            if (want.images or want.media.len > 0) if (!self.image_cache or !pc.mediaKeys(want.key, want.key.len, want.media)) {
                self.pictures_refused += 1;
                self.misses += 1;
                return null;
            };
            const hit = self.side.store.lookup(.{ .keyed = want.key, .media = asFormat(want.media), .longer_than = want.beyond, .max_at = want.max_at, .grid = want.grid }) orelse {
                self.misses += 1;
                return null;
            };
            // with a store on every rank only an entry all four hold is a hit
            if (self.sy) |sy| if (!sy.peer.isVisible(hit.key)) {
                self.misses += 1;
                return null;
            };
            const id = pointId(hit.key);
            self.points.put(self.gpa, id, .{ .key = hit.key, .at = hit.at, .room = @intCast(@min(want.key.len, std.math.maxInt(u32))) }) catch return null;
            return .{ .id = id, .at = hit.at, .tokens = hit.at + want.lookahead, .bytes = 0, .group = hit.group, .shared = hit.shared };
        }

        fn keepFn(ptr: *anyopaque, kept: *const pc.Kept) ?u64 {
            const self = of(ptr);
            if (self.off != null) return null;
            const bt = self.side.bt;
            const at = kept.at;
            if (at == 0 or at % bt != 0 or at < self.min_tokens or kept.key.len < at) return null;
            // pictures: recorded with their spans and full digests (those before the cut), the keyed ids checked to
            // derive from them and the prompt's model ids kept beside them; any disagreement keeps nothing on disk
            var pics: Pictures = .{};
            if (kept.images or kept.media.len > 0) {
                if (!self.image_cache) {
                    self.pictures_refused += 1;
                    return null;
                }
                pics = pointPictures(kept.key, kept.raw, kept.media, at) orelse {
                    self.pictures_refused += 1;
                    return null;
                };
            }
            const info = self.fam.info(self.fam.ptr, kept.saved);
            if (info.at != at) return null;
            var forks: std.ArrayList(format.Fork) = .empty;
            defer forks.deinit(self.gpa);
            const lin = self.toLineage(kept.lineage, &forks) catch return null;
            const origin: ?format.Origin = if (kept.parent) |p| (if (self.points.get(p)) |r| .{ .key = r.key, .at = kept.fork } else null) else null;
            const keyed = kept.key[0..at];
            // one pass over the ids: the chain gives the key here and the block names in the store
            var chain: std.ArrayList(Digest) = .empty;
            defer chain.deinit(self.gpa);
            format.chain(self.gpa, keyed, bt, &chain) catch return null;
            const key = format.pointKeyChained(chain.items, keyed, bt, pics.media);
            const raw: ?[]const u32 = if (pics.media.len > 0) kept.raw[0..at] else null;
            const side = self.side;
            const r = side.keepPoint(.{ .keyed = keyed, .key = key, .chain = chain.items, .at = at, .base = info.base, .snap = info.id, .buf = info.buf, .lineage = lin, .origin = origin, .shared = kept.shared, .group = kept.group, .raw = raw, .media = pics.media });
            if (r == .refused) return null;
            if (self.sy) |sy| {
                const body: wire.KeepBody = .{ .snap = info.id, .base = info.base, .at = at, .shared = kept.shared, .group = kept.group, .origin = origin, .lineage_root = lin.root, .lineage_part = lin.part, .forks = forks.items, .keyed = keyed, .raw = raw, .media = pics.media };
                sy.body.clearRetainingCapacity();
                body.encode(self.gpa, &sy.body) catch {
                    side.cancelWrite(info.id);
                    return null;
                };
                const sent = sy.lead.keep(key, sy.body.items, &sy.out);
                self.syncApply();
                if (sent == .refused) {
                    side.cancelWrite(info.id);
                    return null;
                }
            }
            if (pics.media.len > 0) self.pictures_kept += 1;
            const id = pointId(key);
            const gop = self.points.getOrPut(self.gpa, id) catch return null;
            if (!gop.found_existing) gop.value_ptr.room = 0;
            gop.value_ptr.key = key;
            gop.value_ptr.at = at;
            self.noteKeep(kept.saved);
            return id;
        }

        fn releaseFn(ptr: *anyopaque, point: u64, saved: pc.Saved) void {
            const self = of(ptr);
            if (self.due == saved) self.due = null; // (a state about to be freed is never probed)
            self.confirm(point);
            self.dropSnap(self.fam.info(self.fam.ptr, saved).id);
        }

        fn shareFn(ptr: *anyopaque, point: u64) void {
            const self = of(ptr);
            const r = self.points.get(point) orelse return;
            self.side.store.markShared(r.key);
        }

        fn touchFn(ptr: *anyopaque, point: u64) void {
            const self = of(ptr);
            self.confirm(point); // its first pass ran: the prompt cache resumed it
            const r = self.points.get(point) orelse return;
            self.side.store.touch(r.key);
        }

        fn forgetFn(ptr: *anyopaque, point: u64) void {
            const self = of(ptr);
            const r = self.points.fetchRemove(point) orelse return;
            self.side.store.forget(r.value.key, &.{}, &.{});
            if (self.sy) |sy| {
                sy.lead.evict(r.value.key, &sy.out);
                self.syncApply();
            }
        }

        fn pumpFn(ptr: *anyopaque, budget_ns: u64) void {
            const self = of(ptr);
            _ = self.side.service(budget_ns, pump_blocks);
            self.syncSettle();
            self.probeDue();
        }

        fn busyFn(ptr: *anyopaque) bool {
            return of(ptr).side.busy();
        }

        fn flushFn(ptr: *anyopaque, budget_ns: u64) u64 {
            const self = of(ptr);
            var it = self.jobs.valueIterator();
            while (it.next()) |j| self.endJob(j.*, .shutdown, false);
            self.jobs.clearRetainingCapacity();
            self.confirmAll();
            return self.side.flush(monotonicNs() +| budget_ns);
        }

        fn countsFn(ptr: *anyopaque, out: []pc.TierCounts) usize {
            const self = of(ptr);
            if (out.len == 0) return 0;
            // no service runs while the engine has no tier work, and the store's own threads publish, evict and read:
            // the copy is refreshed here once it is counts_max_age_ns old (the store lock at most that often)
            if (monotonicNs() -| self.side.refreshed_ns >= counts_max_age_ns) self.side.refresh();
            var c = self.side.cached;
            c.presence_refused = self.presence_refused;
            c.compatible_misses = self.misses;
            c.pictures_kept = self.pictures_kept;
            c.pictures_refused = self.pictures_refused;
            c.probe_runs = self.probe_runs;
            c.probe_failed = self.probe_failed;
            if (self.fam.land_counts) |lc| {
                c.landings = lc.landings;
                c.landings_skipped = lc.skipped;
                c.landings_failed = lc.failed + lc.aborted; // every landing begun that did not land
                c.landing_pieces = lc.pieces;
                c.landing_bytes = lc.bytes;
            }
            out[0] = c; // the one store's, rank 0's
            return 1;
        }

        // -- restores ---------------------------------------------------------------------------------------------

        /// beginFn's capacity refusals asked before the prompt cache makes room (robustness): the restores in flight at
        /// max_restores, or the store out of reads or restore slots. An off tier is not full: begin names it.
        fn fullFn(ptr: *anyopaque) bool {
            const self = of(ptr);
            if (self.off != null) return false;
            return self.jobs.count() >= self.max_restores or self.side.store.readsFull();
        }

        fn beginFn(ptr: *anyopaque, point: *const pc.Point) anyerror!u64 {
            const self = of(ptr);
            if (self.off != null) return error.SpillOff;
            const side = self.side;
            // refused before anything changes: the store's reads (Settings.restores) bound how many run at once, and a
            // placement must not stale kept states for a restore that cannot start
            if (self.jobs.count() >= self.max_restores) return error.TooManyRestores;
            const ref = self.points.get(point.id) orelse return error.UnknownPoint;
            if (point.at != ref.at or point.at % side.bt != 0) return error.BadPoint;
            const blocks: u32 = point.at / side.bt;
            if (!side.store.presence(ref.key)) {
                self.presence_refused += 1;
                return error.SpillNotOnEveryRank;
            }
            try self.jobs.ensureUnusedCapacity(self.gpa, 1);
            const id = self.next_job;
            const rx = try side.newRx(id, ref.key, point.at, blocks);
            // the reads first (TooMany / NoSlot / Missing: nothing placed); they only need staging slots, the destination
            // is first written by rxStep
            rx.rid = side.store.beginRead(ref.key) catch |e| {
                side.rxDrop(rx);
                return e;
            };
            const room = @max(@as(usize, point.at), ref.room);
            const dest = self.fam.reserve(self.fam.ptr, point.at, room) catch |e| {
                side.rxEnd(rx, .cancel, false);
                return e;
            };
            rx.snap = .{ .id = dest.id, .buf = dest.buf };
            rx.dst = side.gens.reserve(dest.base, dest.base + point.at) catch |e| {
                side.rxEnd(rx, .cancel, false);
                self.fam.discard(self.fam.ptr, dest.saved);
                return e;
            };
            const job = self.gpa.create(Job) catch |e| {
                side.rxEnd(rx, .cancel, false);
                self.fam.discard(self.fam.ptr, dest.saved);
                return e;
            };
            job.* = .{ .rx = rx, .point = point.*, .saved = dest.saved, .key = ref.key, .t_begin = monotonicNs() };
            self.next_job += 1;
            self.jobs.putAssumeCapacity(id, job);
            if (self.sy) |sy| {
                // every other rank reads its own copy into the same destination, now, beside this rank's read
                var rb: [wire.RestoreBody.len]u8 = undefined;
                (wire.RestoreBody{ .snap = dest.id, .base = dest.base, .at = point.at }).encode(&rb);
                const began = sy.lead.restore(ref.key, id, &rb, &sy.out);
                self.syncApply();
                if (!began) {
                    _ = self.jobs.remove(id);
                    self.endJob(job, .cancelled, false);
                    return error.SpillNotOnEveryRank;
                }
            }
            return id;
        }

        /// A job ends without publishing: its copies waited, its destination cleared and discarded, the point
        /// forgotten when `forget` (damage).
        fn endJob(self: *Self, job: *Job, why: Failure, forget: bool) void {
            if (job.rx) |rx| self.side.rxEnd(rx, if (why == .cancelled or why == .shutdown) .cancel else .abort, forget);
            if (job.done) |*d| { // read and committed, never installed (its landing refused, failed or cut): never used,
                // nothing crashed in it, so its quarantine marker goes now (else the next open forgets a good point)
                d.lineage.deinit(self.gpa);
                self.side.store.restored(job.key);
            }
            self.fam.discard(self.fam.ptr, job.saved); // (a landing in flight: the family aborts it)
            if (forget) _ = self.points.remove(job.point.id);
            self.gpa.destroy(job);
        }

        fn failJob(self: *Self, id: u64, job: *Job, why: Failure) pc.Poll {
            _ = self.jobs.remove(id);
            const forget = damage(why);
            self.side.rxEnd(job.rx.?, .abort, forget);
            self.fam.discard(self.fam.ptr, job.saved);
            if (forget) _ = self.points.remove(job.point.id);
            self.gpa.destroy(job);
            return .{ .failed = failureError(why) };
        }

        fn pollFn(ptr: *anyopaque, id: u64, budget_ns: u64) pc.Poll {
            const self = of(ptr);
            const side = self.side;
            const job = self.jobs.get(id) orelse return .{ .failed = error.UnknownJob };
            if (self.off != null) {
                _ = self.jobs.remove(id);
                self.endJob(job, .cancelled, false);
                return .{ .failed = error.SpillOff };
            }
            if (job.rx) |rx| {
                _ = side.retire();
                side.rxStep(rx, null, budget_ns);
                side.rxSmall(rx);
                if (self.sy != null) {
                    // this rank's result in (read whole, or failed): the four ranks vote on it before anything is adopted
                    if (job.vote == .none) {
                        if (rx.why == .none and !S.rxReady(rx)) return .{ .pending = .{ .done = rx.copied, .total = rx.blocks } };
                        self.syncSettle();
                        if (job.vote == .none) return .{ .pending = .{ .done = rx.copied, .total = rx.blocks } };
                    }
                    if (job.vote == .drop) return self.failJob(id, job, if (rx.why != .none) rx.why else .crc);
                }
                if (rx.why != .none) return self.failJob(id, job, rx.why);
                if (!S.rxReady(rx)) return .{ .pending = .{ .done = rx.copied, .total = rx.blocks } };
                if (!side.rxValid(rx)) return self.failJob(id, job, .reserve);
                // rank 0's own end of the read: a failure here (out of memory for the lineage, an incomplete or failed
                // read the checks above missed) is not damage to the stored point, so nobody forgets it
                job.done = side.rxCommit(rx) catch return self.failJob(id, job, .read);
                job.blocks = rx.blocks;
                side.rxDrop(rx);
                job.rx = null;
                job.t_copied = monotonicNs();
                self.restore_copy.observe(job.t_copied -| job.t_begin);
            }
            // rank 0 holds the state (its Snap still loading: never a victim, never usable); every other rank next
            if (self.fam.land) |land| switch (land(self.fam.ptr, job.saved, budget_ns)) {
                .pending => return .{ .pending = .{ .done = job.blocks, .total = job.blocks } },
                .landed => {},
                // the point stays on disk (not damage): the request runs from memory or from its start
                .refused => return self.endLanding(id, job, error.SpillLandRefused),
                .failed => return self.endLanding(id, job, error.SpillLandFailed),
            };
            const now = monotonicNs();
            if (self.fam.land != null) self.restore_land.observe(now -| job.t_copied);
            self.restore_ready.observe(now -| job.t_begin);
            if (!@import("builtin").is_test) std.log.info("dsv41 spill: restored {d} tokens ({d} blocks) in {d} ms: on rank 0's device after {d} ms, landed on every rank {d} ms later", .{ job.point.at, job.blocks, (now -| job.t_begin) / std.time.ns_per_ms, (job.t_copied -| job.t_begin) / std.time.ns_per_ms, (now -| job.t_copied) / std.time.ns_per_ms });
            var done_ = job.done.?;
            defer done_.lineage.deinit(self.gpa);
            const lin = self.fromLineage(done_.lineage) catch &.{};
            _ = self.jobs.remove(id);
            // its marker stays until the state is used; without room to track it, it goes now (never left behind)
            if (self.unconfirmed.getOrPut(self.gpa, job.point.id)) |gop| {
                if (gop.found_existing) gop.value_ptr.holds += 1 else gop.value_ptr.* = .{ .key = job.key, .holds = 1 };
            } else |_| self.side.store.restored(job.key);
            self.installed.put(self.gpa, self.fam.info(self.fam.ptr, job.saved).id, job.point.id) catch {};
            self.fam.install(self.fam.ptr, job.saved);
            const saved = job.saved;
            self.gpa.destroy(job);
            return .{ .ready = .{ .saved = saved, .bytes = 0, .lineage = lin } };
        }

        /// A landing every rank skipped or that failed: the destination discarded (the family aborts what it opened),
        /// the point kept on disk.
        fn endLanding(self: *Self, id: u64, job: *Job, err: anyerror) pc.Poll {
            _ = self.jobs.remove(id);
            self.endJob(job, .cancelled, false);
            return .{ .failed = err };
        }

        fn cancelFn(ptr: *anyopaque, id: u64) void {
            const self = of(ptr);
            const kv = self.jobs.fetchRemove(id) orelse return;
            self.endJob(kv.value, .cancelled, false);
        }
    };
}
