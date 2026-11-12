//! Port of wojo's GLM 0088 (PR #78) and 0110 (PR #124; BadAd84's block layout, lineage, write-at-keep staging ring,
//! release and fsync order; eviction by conversation for issue #122, reported by xiepengqi) spill tier: the native,
//! family-neutral, GPU-free spill v2 store of one rank.
//!
//! A kept prompt state is its rows, in blocks of `limits.block_tokens` tokens of the family's packed per-rank planes,
//! plus named small-state sections. The family packs and unpacks bytes on its CUDA stream; this store never touches
//! the device, never allocates staging and issues no CUDA call. It hands the engine thread aligned host slots out of
//! the caller's staging buffers, takes them back when the engine says the device copy into or out of them is done,
//! and does the disk work on its own threads:
//!
//! - writes (`keep` at an eligible keep, `stage` between rounds, `staged`/`captured` once the engine's events
//!   complete): each missing block under a unique temporary name, its CRC-32 and SHA-256 computed, fsynced, renamed
//!   (or hard-linked to a block with equal bytes), then the point's ids and small state, the directories fsynced, and
//!   the manifest last. `guard` stages or cancels unpublished points before their source rows change.
//! - reads (`lookup`, `beginRead`, `readReady`/`readRelease` between rounds, `readAwait` for a follower's bounded
//!   wait that recycles slots while it waits, `endRead`): reader threads read and check blocks into read slots ahead
//!   of the engine's copies; damage to a block invalidates every name linked to its inode, a missing name only itself.
//! - room: a cap, a free-disk floor and an optional per-conversation quota, made in 0110's order: old turns past
//!   retention on a conversation's active path, abandoned forks, whole conversations by last use, shared prefixes
//!   last. A conversation is the branch below the deepest shared point a state extends, merged with any states that
//!   carry the same 0111 `group`.
//! - recovery at open: temporaries deleted, manifests parsed with fixed bounds, points with any missing or wrongly
//!   sized file dropped, references rebuilt, unreferenced blocks collected. An exclusive lock keeps a second process
//!   from scanning or deleting a live store.
//!
//! Every public function runs on the engine thread unless it says otherwise. Engine-thread calls take the store's
//! lock only briefly and do no file I/O (deletions, touches and metadata rewrites are queued to the writer).
const std = @import("std");
const builtin = @import("builtin");
const c = std.c;
const format = @import("format.zig");
const fsio = @import("fsio.zig");
const slots_mod = @import("slots.zig");
const direct_io = @import("../direct_io.zig");
const mutants = @import("mutants.zig");
const mut = mutants.on;
const Allocator = std.mem.Allocator;

pub const Digest = format.Digest;
pub const Lineage = format.Lineage;
pub const Media = format.Media;
pub const Slot = slots_mod.Slot;
pub const Staging = slots_mod.Staging;
pub const Hooks = fsio.Hooks;

/// The logical rows a write reads from: the family's extent and its generation. A later occupant of the same base
/// has another generation, so a delayed staging can never attach to it.
pub const Source = struct {
    id: u64,
    generation: u64,

    pub fn eql(a: Source, b: Source) bool {
        return a.id == b.id and a.generation == b.generation;
    }
};

pub const Config = struct {
    /// The spill directory (a local NVMe per rank; never the shared model/Engram mount). Must exist.
    root: [:0]const u8,
    rank: u32,
    /// The agreed compatibility digest (`format.compatDigest` over the common digest every rank exchanged).
    /// rank_wire's startup agreement exchanges the common digest and the ordered per-rank digests and refuses
    /// differing configurations before any spill collective (fingerprint.Agreement.storeCommon); this store only
    /// trusts what it is given.
    compat: Digest,
    /// Written into every manifest; a point of another layout is foreign here (owner_sharded with TF_DS_KV_SHARD).
    owner_layout: format.OwnerLayout = .replicated,
    limits: format.Limits,
    /// Bytes this rank's store may hold.
    cap_bytes: u64,
    /// No write leaves less than this free on the store's file system.
    floor_bytes: u64 = 50 << 30,
    /// A conversation holding more than this fraction of the cap trims its own old turns and forks first (0: off).
    quota: f64 = 0,
    /// Newest states kept on a conversation's active path (0: every turn until room is needed).
    turns: u32 = 4,
    /// Shorter states are not written.
    min_tokens: u32 = 8192,
    /// Bytes of unpublished writes (rows still in device memory, queued staging) past which a keep is refused.
    backlog_bytes: u64 = 16 << 30,
    /// The caller's staging buffers (block slots of `limits.block_bytes`, small slots of `limits.max_small_bytes`).
    staging: Staging,
    readers: u32 = 4,
    /// Restores at once (each holds a restore slot for its small state).
    max_reads: u32 = 4,
    direct: bool = true,
    /// Other compat namespaces kept at open, newest first (a live one, still locked, is never touched).
    keep_builds: u32 = 1,
    /// The I/O priority of the store's writer and reader threads, an ioprio value (class << 13 | level: be/7 is
    /// (2 << 13) | 7), so their fsyncs and restore reads yield to the engine's own disk reads (Engram) under an I/O
    /// scheduler that honours priorities (mq-deadline, bfq; not `none`). null: the process's priority. Linux only; a
    /// value the kernel refuses is counted (ioprio_refused) and the thread keeps the process's.
    io_priority: ?u16 = null,
    hooks: Hooks = .{},
};

pub const SectionSpec = struct { name: []const u8, len: u64 };

/// What `keep` stores.
pub const Keep = struct {
    /// Canonical key ids of [0, at) (the prompt cache's keyed ids, picture keys included).
    keyed: []const u32,
    /// The raw ids when they differ from `keyed` (pictures), else null.
    raw: ?[]const u32 = null,
    media: []const Media = &.{},
    lineage: Lineage,
    origin: ?format.Origin = null,
    shared: bool = false,
    group: u64 = 0,
    sections: []const SectionSpec,
    /// Bytes of the last block when `keyed.len` is not a whole number of blocks (captured now: a mutable tail).
    tail_bytes: u64 = 0,
    source: Source,
    /// `keyed`'s id chain (`format.chain(keyed, limits.block_tokens)`) when the caller hashed it already, taken in
    /// place of a second pass over the ids. Its length is checked (else `.invalid`), its digests are trusted: a wrong
    /// chain gives a key no lookup finds and a point the next open drops (recovery hashes the ids again).
    chain: ?[]const Digest = null,
};

/// A whole point and its digest (Store.pointList).
pub const PointEntry = struct { key: Digest, digest: Digest };

pub const JobId = u64;
pub const ReadId = u64;

pub const Queued = struct {
    job: JobId,
    /// The small state's bytes: the caller fills each section at its offset (asynchronously), then `captured`.
    small: []u8,
    sections: []const format.Section,
    /// The tail block's bytes when the point has a tail (filled with the small state, before `captured`).
    tail: ?[]u8 = null,
};

pub const KeepResult = union(enum) {
    queued: Queued,
    /// Stored or on its way already (a shared request made it shared).
    stored,
    short,
    no_space,
    no_staging,
    backlog,
    invalid,
};

/// One block to pack into `dst`: the source's tokens [first, first + tokens), `bytes` long.
pub const StageReq = struct {
    ticket: Ticket,
    source: Source,
    first: u32,
    tokens: u32,
    dst: []u8,
};

pub const Ticket = struct { job: JobId, index: u32, slot: Slot };

pub const WriteFailure = enum { io, no_space, capture, stage, missing_block, crashed, invalid };
pub const CancelReason = enum { guard, source_gone, explicit, shutdown };

pub const Completion = union(enum) {
    published: struct { job: JobId, key: Digest, at: u32 },
    failed: struct { job: JobId, key: Digest, why: WriteFailure },
    cancelled: struct { job: JobId, key: Digest, why: CancelReason },
};

pub const GuardResult = struct { staged: usize = 0, cancelled: usize = 0 };

pub const Query = struct {
    keyed: []const u32,
    media: []const Media = &.{},
    /// Only points longer than this.
    longer_than: u32 = 0,
    /// The longest usable point (`Lanes.usable`: at < prompt.len, at <= prompt.len - window, ...).
    max_at: u32,
    /// Points on this grid only (0: any).
    grid: u32 = 0,
};

/// A lookup's point: `part` is its lineage's part salt (owner-sharded stores: the part its rows were written from,
/// where a restore must put them back; null in a replicated store).
pub const Hit = struct { key: Digest, at: u32, shared: bool, group: u64, part: ?u32 = null };

pub const ReadyBlock = struct { index: u32, first: u32, tokens: u32, data: []const u8, slot: Slot };

pub const ReadStatus = enum { more, waiting, done, failed };

pub const ReadError = enum { missing, damaged, io, small, cancelled };

pub const ReadDone = struct {
    key: Digest,
    at: u32,
    shared: bool,
    group: u64,
    has_raw: bool,
    lineage: Lineage, // gpa-owned
    origin: ?format.Origin,
};

pub const SmallView = struct { bytes: []const u8, sections: []const format.Section };

/// The store lock's wait or hold times: count, sum and max in ns, and log2 buckets: `le_us[i]` counts the times of at
/// most 2^i microseconds (i = 0 .. 23: 1 us .. about 8.4 s; a longer one is in count, sum and max only). The buckets
/// are not cumulative. A thread waiting on one of the store's conditions does not hold the lock.
pub const LockTimes = struct {
    pub const buckets = 24;
    count: u64 = 0,
    sum_ns: u64 = 0,
    max_ns: u64 = 0,
    le_us: [buckets]u64 = @splat(0),

    pub fn note(t: *LockTimes, ns: u64) void {
        t.count += 1;
        t.sum_ns +|= ns;
        t.max_ns = @max(t.max_ns, ns);
        const us = std.math.divCeil(u64, ns, std.time.ns_per_us) catch unreachable;
        const i: usize = if (us <= 1) 0 else std.math.log2_int_ceil(u64, us);
        if (i < buckets) t.le_us[i] += 1;
    }
};

pub const Counters = struct {
    keeps: u64 = 0,
    queued: u64 = 0,
    stored_skips: u64 = 0,
    short: u64 = 0,
    no_space: u64 = 0,
    no_staging: u64 = 0,
    backlog_refused: u64 = 0,
    invalid: u64 = 0,
    committed: u64 = 0,
    write_failed: u64 = 0,
    cancelled: u64 = 0,
    cancelled_guard: u64 = 0,
    cancelled_source: u64 = 0,
    staged_blocks: u64 = 0,
    guard_staged: u64 = 0,
    written_blocks: u64 = 0,
    written_bytes: u64 = 0,
    written_small_bytes: u64 = 0,
    dedup_blocks: u64 = 0,
    dedup_bytes: u64 = 0,
    shared_marked: u64 = 0,
    shared_rewrites: u64 = 0,
    pruned_turns: u64 = 0,
    pruned_forks: u64 = 0,
    pruned_conversations: u64 = 0, // whole conversations evicted at the cap or the floor (once a conversation)
    pruned_conversation_points: u64 = 0, // their points
    pruned_shared: u64 = 0,
    trimmed_turns: u64 = 0,
    quota_trims: u64 = 0,
    dropped: u64 = 0,
    disk_lookups: u64 = 0,
    disk_hits: u64 = 0,
    read_deferred: u64 = 0,
    loaded: u64 = 0,
    loaded_tokens: u64 = 0,
    loaded_bytes: u64 = 0,
    load_failed: u64 = 0,
    crc_errors: u64 = 0,
    size_errors: u64 = 0,
    missing_errors: u64 = 0,
    recovered_points: u64 = 0,
    recovered_dropped: u64 = 0,
    /// Points recovery could not read for a reason other than damage (an I/O error, EMFILE, a busy file): left on
    /// disk untouched, with every unreferenced block, until the next open.
    recover_deferred: u64 = 0,
    temps_removed: u64 = 0,
    orphans_removed: u64 = 0,
    builds_pruned: u64 = 0,
    /// Opens with O_DIRECT the file system refused, reads and writes (O_DIRECT is then off for this store: it reads
    /// and writes buffered and drops each file's pages after use).
    direct_refused: u64 = 0,
    /// Chowns of a folder or file the store made to the root's owner that failed (the file is used as it is).
    owner_failed: u64 = 0,
    /// Points an open forgot because a restore marker named them (a restore was in flight, or its state had not
    /// served a round, when the engine stopped).
    quarantined: u64 = 0,
    /// Restore markers that could not be made (the restore failed as an I/O error; the point stays).
    marker_failed: u64 = 0,
    /// Store threads whose I/O priority (Config.io_priority) the kernel refused (they run at the process's).
    ioprio_refused: u64 = 0,
    /// How long callers waited for the store lock, and how long they held it.
    lock_wait: LockTimes = .{},
    lock_hold: LockTimes = .{},
    // each block a restore reads: its open (and size check), its read, its CRC and its close; and each read's wall
    // time, from its begin to its last block read
    block_open: LockTimes = .{},
    block_read: LockTimes = .{},
    block_crc: LockTimes = .{},
    block_close: LockTimes = .{},
    read_wall: LockTimes = .{},
    /// Writer tasks dropped because closeWithin's deadline passed (metadata that did not run, unpublished writes).
    close_dropped: u64 = 0,
    // gauges
    points: u64 = 0,
    blocks: u64 = 0,
    stored_bytes: u64 = 0,
    queued_bytes: u64 = 0,
    pending_jobs: u64 = 0,
    reads: u64 = 0,
    staging_bytes: u64 = 0,
    staging_peak_bytes: u64 = 0,
    disk_free_bytes: u64 = 0,
};

const BlockInfo = struct { crc: u32, sha: Digest, ino: u64, bytes: u64 };
const Inode = struct { bytes: u64, names: std.ArrayList(Digest) = .empty };

const Point = struct {
    key: Digest,
    at: u32,
    shared: bool,
    group: u64,
    used: i96,
    created_ns: i64,
    has_raw: bool,
    mutable_tail: bool,
    lineage: Lineage,
    origin: ?format.Origin,
    chain: []Digest,
    tail: []u32, // ids after the last whole block
    media: []Media,
    blocks: []format.BlockRef,
    sections: []format.Section,
    ids_bytes: u64,
    ids_crc: u32,
    small_bytes: u64,
    small_crc: u32,
    files: u64 = 0, // bytes of its .ids, .st and .man
    pins: u32 = 0, // restores holding it (freed only once none does)
    removed: bool = false,

    fn deinit(p: *Point, gpa: Allocator) void {
        p.lineage.deinit(gpa);
        gpa.free(p.chain);
        gpa.free(p.tail);
        gpa.free(p.media);
        gpa.free(p.blocks);
        gpa.free(p.sections);
        gpa.destroy(p);
    }

    fn whole(p: *const Point) usize {
        return p.chain.len;
    }
};

const WriteJob = struct {
    id: JobId,
    point: *Point,
    source: Source,
    new: std.ArrayList(u32) = .empty, // block indices this job stages (no tail)
    next: usize = 0,
    tail: ?u32 = null,
    tail_slot: ?Slot = null,
    outstanding: u32 = 0, // staged blocks not yet written (or given back)
    engine: u32 = 0, // of those, slots the engine holds (its copy not yet reported by `staged`)
    small_slot: Slot,
    capture: enum { pending, ok, failed } = .pending,
    state: enum { open, publishing, failed, cancelled } = .open,
    fail: WriteFailure = .io,
    cancel: CancelReason = .explicit,
    size: u64,
    ids: []u8,
    listed: bool = true, // in the staging FIFO
    parked: bool = false,
    shared_at_write: bool = false,
};

const ReadJob = struct {
    id: ReadId,
    point: *Point,
    total: u32,
    state: []BState,
    slot: []Slot,
    next_assign: u32 = 0,
    next_hand: u32 = 0,
    small_slot: Slot,
    small: enum { queued, ready, failed } = .queued,
    err: ?ReadError = null,
    bad: std.ArrayList(Digest) = .empty,
    lost: std.ArrayList(Digest) = .empty,
    disk: u32 = 0, // tasks queued or running on readers
    held: u32 = 0, // read slots it holds (reading, read, or handed and not released)
    read: u32 = 0, // blocks read and checked (its wall time is noted when the last one is)
    ended: bool = false,
    t0: i96,

    const BState = enum(u8) { idle, reading, ready, handed, failed };
};

const Task = union(enum) {
    block: struct { job: *WriteJob, index: u32, slot: Slot },
    point: *WriteJob,
    shared: Digest,
    touch: struct { key: Digest, ns: i96 },
    unlink_block: struct { name: Digest, bytes: u64 },
    unlink_point: struct { key: Digest, bytes: u64 },
    sync: u64,
    unmark: Digest, // a restore marker no restore holds any more
};

/// One key's restore quarantine marker, in the store's books while anything about it is live.
const Mark = struct {
    restores: u32 = 0, // restores holding it: beginRead until endRead's abort or cancel, or restored(key) after a commit
    writing: u32 = 0, // readers writing it now
    on_disk: bool = false, // a marker file may be there (a write started; cleared once the writer unlinked it)
    queued: bool = false, // an unmark task is queued
    unlinking: bool = false, // the writer unlinks it now (a reader about to write a new one waits)
};

const RTask = struct { job: *ReadJob, index: ?u32, slot: Slot };

/// The longest a drain or a close sleeps between two looks at its deadline (and at `crashed`): a constant, not a
/// setting. The writer and the readers wake them sooner (`done` after every task and at their exit), so it bounds
/// only how late a passed deadline is seen.
const wait_slice_ns: i96 = 50 * std.time.ns_per_ms;

pub const Store = struct {
    gpa: Allocator,
    io: std.Io,
    cfg: Config,
    hooks: Hooks,
    root_fd: c.fd_t = -1,
    top_fd: c.fd_t = -1,
    rank_fd: c.fd_t = -1,
    blocks_fd: c.fd_t = -1,
    points_fd: c.fd_t = -1,
    lock: fsio.Lock = .{ .fd = -1 },
    writer: fsio.Writer = undefined,
    block_slot: usize,
    small_slot: usize,

    mutex: std.Io.Mutex = .init,
    hold_t0: i96 = 0, // when the lock's holder took it (only the holder reads or writes it)
    work: std.Io.Condition = .init,
    rwork: std.Io.Condition = .init,
    done: std.Io.Condition = .init,

    points: std.AutoHashMapUnmanaged(Digest, *Point) = .empty,
    have: std.AutoHashMapUnmanaged(Digest, BlockInfo) = .empty,
    content: std.AutoHashMapUnmanaged(Digest, Digest) = .empty,
    inodes: std.AutoHashMapUnmanaged(u64, Inode) = .empty,
    refs: std.AutoHashMapUnmanaged(Digest, u32) = .empty,
    inflight: std.AutoHashMapUnmanaged(Digest, JobId) = .empty,
    parent: std.AutoHashMapUnmanaged(Digest, ?Digest) = .empty,
    reading: std.AutoHashMapUnmanaged(Digest, u32) = .empty,
    marks: std.AutoHashMapUnmanaged(Digest, Mark) = .empty,
    waiting: std.AutoHashMapUnmanaged(Digest, *WriteJob) = .empty,
    jobs_by_id: std.AutoHashMapUnmanaged(JobId, *WriteJob) = .empty,
    fifo: std.ArrayList(*WriteJob) = .empty, // jobs with blocks left to stage, oldest first
    parked: std.ArrayList(*WriteJob) = .empty,
    tasks: std.ArrayList(Task) = .empty,
    task_head: usize = 0,
    rtasks: std.ArrayList(RTask) = .empty,
    rtask_head: usize = 0,
    reads: std.ArrayList(*ReadJob) = .empty,
    read_turn: usize = 0,
    completions: std.ArrayList(Completion) = .empty,
    slots: slots_mod.Slots = .{},
    counters: Counters = .{},

    next_id: u64 = 1,
    clock: i96 = 0,
    stored_bytes: u64 = 0,
    queued_bytes: u64 = 0,
    disk_free: u64 = 0,
    pending_free: u64 = 0, // bytes removed from the index whose files the writer has not unlinked yet
    sync_sent: u64 = 0,
    sync_done: u64 = 0,
    busy: u32 = 0, // writer tasks running
    stopping: bool = false,
    crashed: bool = false,
    /// closeWithin's deadline passed: the writer drops every task left (counted in close_dropped) and nothing waits
    /// for the threads; they finish on their own while the process exits.
    abandon: bool = false,
    writer_exited: bool = false,
    readers_exited: usize = 0,
    writer_thread: ?std.Thread = null,
    reader_threads: std.ArrayList(std.Thread) = .empty,
    /// The kernel ids of the writer and the readers, each recorded once its loop started (tests: their priority).
    thread_ids: std.ArrayList(std.Thread.Id) = .empty,

    // -- open / close --------------------------------------------------------------------------------------------
    /// Opens (or makes) this rank's store under `cfg.root` and recovers it; fails with StoreBusy, before scanning or
    /// deleting anything, while another process owns it.
    pub fn open(gpa: Allocator, io: std.Io, cfg: Config) !*Store {
        const l = cfg.limits;
        if (l.block_tokens == 0 or l.block_bytes == 0 or l.max_tokens == 0 or cfg.cap_bytes == 0) return error.BadConfig;
        if (l.foreign_block_bytes > l.block_bytes or (l.foreign_block_bytes != 0 and cfg.owner_layout != .owner_sharded)) return error.BadConfig;
        if (!std.math.isFinite(cfg.quota) or cfg.quota < 0 or cfg.quota > 1) return error.BadConfig;
        const s = try gpa.create(Store);
        s.* = .{
            .gpa = gpa,
            .io = io,
            .cfg = cfg,
            .hooks = cfg.hooks,
            .block_slot = std.mem.alignForward(usize, @intCast(l.block_bytes), fsio.alignment),
            .small_slot = std.mem.alignForward(usize, @intCast(@max(l.max_small_bytes, 1)), fsio.alignment),
        };
        s.writer = fsio.Writer.init(&s.hooks, cfg.direct);
        errdefer s.destroy();
        s.slots = try slots_mod.Slots.init(gpa, cfg.staging, s.block_slot, s.small_slot);
        s.root_fd = try fsio.openDir(cfg.root);
        // everything the store makes is the root's owner's, set before each rename
        const root_st = try fsio.statFd(s.root_fd);
        s.writer.owner = .{ .uid = root_st.uid, .gid = root_st.gid };
        var nb: [64]u8 = undefined;
        const top = try namespaceName(&nb, cfg.compat);
        try fsio.ensureDirAt(s.root_fd, top, &s.writer);
        s.top_fd = try fsio.openDirAt(s.root_fd, top);
        var rb: [32]u8 = undefined;
        const rank = std.fmt.bufPrintSentinel(&rb, "rank{d}", .{cfg.rank}, 0) catch unreachable;
        try fsio.ensureDirAt(s.top_fd, rank, &s.writer);
        s.rank_fd = try fsio.openDirAt(s.top_fd, rank);
        s.lock = fsio.Lock.take(s.rank_fd, &s.writer) catch |e| if (mut(.owner_lock_ignored) and e == error.Busy) fsio.Lock{ .fd = -1 } else return if (e == error.Busy) error.StoreBusy else e;
        // marks this build's namespace used: pruneBuilds orders by its directory's mtime (nothing else changes it, the
        // writes go two levels down), so an older build opened after a newer one keeps its store
        if (!mut(.prune_by_creation)) fsio.touchAt(s.root_fd, top, std.Io.Clock.real.now(io).toNanoseconds());
        try fsio.ensureDirAt(s.rank_fd, "blocks", &s.writer);
        try fsio.ensureDirAt(s.rank_fd, "points", &s.writer);
        s.blocks_fd = try fsio.openDirAt(s.rank_fd, "blocks");
        s.points_fd = try fsio.openDirAt(s.rank_fd, "points");
        s.pruneBuilds(top, rank);
        try s.scan();
        s.disk_free = fsio.diskFree(&s.hooks, s.blocks_fd);
        try s.reader_threads.ensureTotalCapacity(gpa, @max(cfg.readers, 1));
        try s.thread_ids.ensureTotalCapacity(gpa, @max(cfg.readers, 1) + 1);
        s.writer_thread = try std.Thread.spawn(.{}, writerLoop, .{s});
        for (0..@max(cfg.readers, 1)) |_| {
            const t = std.Thread.spawn(.{}, readerLoop, .{s}) catch |e| {
                s.stopThreads();
                return e;
            };
            s.reader_threads.appendAssumeCapacity(t);
        }
        return s;
    }

    fn stopThreads(s: *Store) void {
        s.lockS();
        s.stopping = true;
        s.work.broadcast(s.io);
        s.rwork.broadcast(s.io);
        s.unlockS();
        if (s.writer_thread) |t| t.join();
        s.writer_thread = null;
        for (s.reader_threads.items) |t| t.join();
        s.reader_threads.clearRetainingCapacity();
    }

    /// Stops the workers (deletions, touches and metadata rewrites already queued still run; unpublished points
    /// stay unpublished and are reported), frees memory and releases the owner lock. Never deletes a published point.
    /// Waits as long as that takes (closeWithin(maxInt)); a stop bounded in time calls closeWithin.
    pub fn close(s: *Store) void {
        _ = s.closeWithin(std.math.maxInt(u64));
    }

    pub const CloseResult = struct {
        /// The writer and every reader exited by the deadline: joined, the memory freed, the owner lock released.
        clean: bool,
        /// Not clean: the writer's tasks still queued at the deadline. They do not run (the writer drops them once
        /// its disk call returns, counted in close_dropped). 0 when clean.
        dropped: usize,
    };

    /// Stops the workers and waits at most `timeout_ns` for the queued metadata to run and the threads to exit. Past
    /// the deadline the tasks left are dropped (counted), the threads detached and the store left allocated with its
    /// lock held: only a process about to exit calls this, and it calls nothing on the store afterwards.
    pub fn closeWithin(s: *Store, timeout_ns: u64) CloseResult {
        const deadline = std.Io.Clock.awake.now(s.io).toNanoseconds() + @as(i96, timeout_ns);
        s.lockS();
        s.stopping = true;
        s.work.broadcast(s.io);
        s.rwork.broadcast(s.io);
        while (!(s.writer_exited and s.readers_exited == s.reader_threads.items.len)) {
            const left = deadline - std.Io.Clock.awake.now(s.io).toNanoseconds();
            if (left <= 0 and !mut(.close_waits)) {
                s.abandon = true;
                const dropped = s.tasks.items.len - s.task_head;
                s.work.broadcast(s.io);
                s.unlockS();
                if (s.writer_thread) |t| t.detach();
                s.writer_thread = null;
                for (s.reader_threads.items) |t| t.detach(); // the list stays: readers_exited counts against it
                return .{ .clean = false, .dropped = dropped };
            }
            const slice: i96 = if (left > 0) @min(left, wait_slice_ns) else wait_slice_ns; // else: the mutant waits on
            s.condWaitFor(&s.done, slice);
        }
        s.unlockS();
        if (s.writer_thread) |t| t.join();
        s.writer_thread = null;
        for (s.reader_threads.items) |t| t.join();
        s.reader_threads.clearRetainingCapacity();
        s.destroy();
        return .{ .clean = true, .dropped = 0 };
    }

    fn destroy(s: *Store) void {
        const gpa = s.gpa;
        var jit = s.jobs_by_id.valueIterator();
        while (jit.next()) |j| freeJob(gpa, j.*, true);
        s.jobs_by_id.deinit(gpa);
        for (s.reads.items) |r| {
            r.point.pins -= 1;
            if (r.point.pins == 0 and r.point.removed) r.point.deinit(gpa);
            freeRead(gpa, r);
        }
        s.reads.deinit(gpa);
        var pit = s.points.valueIterator();
        while (pit.next()) |p| p.*.deinit(gpa);
        s.points.deinit(gpa);
        s.have.deinit(gpa);
        s.content.deinit(gpa);
        var iit = s.inodes.valueIterator();
        while (iit.next()) |x| x.names.deinit(gpa);
        s.inodes.deinit(gpa);
        s.refs.deinit(gpa);
        s.inflight.deinit(gpa);
        s.parent.deinit(gpa);
        s.reading.deinit(gpa);
        s.marks.deinit(gpa);
        s.waiting.deinit(gpa);
        s.fifo.deinit(gpa);
        s.parked.deinit(gpa);
        s.tasks.deinit(gpa);
        s.rtasks.deinit(gpa);
        s.completions.deinit(gpa);
        s.reader_threads.deinit(gpa);
        s.thread_ids.deinit(gpa);
        s.slots.deinit(gpa);
        if (s.lock.fd >= 0) s.lock.release();
        for ([_]c.fd_t{ s.points_fd, s.blocks_fd, s.rank_fd, s.top_fd, s.root_fd }) |fd| if (fd >= 0) {
            _ = c.close(fd);
        };
        gpa.destroy(s);
    }

    fn freeJob(gpa: Allocator, j: *WriteJob, with_point: bool) void {
        j.new.deinit(gpa);
        gpa.free(j.ids);
        if (with_point) j.point.deinit(gpa);
        gpa.destroy(j);
    }

    fn freeRead(gpa: Allocator, r: *ReadJob) void {
        gpa.free(r.state);
        gpa.free(r.slot);
        r.bad.deinit(gpa);
        r.lost.deinit(gpa);
        gpa.destroy(r);
    }

    fn namespaceName(buf: []u8, compat: Digest) ![:0]const u8 {
        const h = format.hex(&compat);
        return std.fmt.bufPrintSentinel(buf, "native-spill-{s}", .{h[0..16]}, 0) catch unreachable;
    }

    fn lockS(s: *Store) void {
        const t0 = monoNow(s.io);
        s.mutex.lockUncancelable(s.io);
        s.hold_t0 = monoNow(s.io);
        s.counters.lock_wait.note(@intCast(@max(0, s.hold_t0 - t0)));
    }

    fn unlockS(s: *Store) void {
        s.endHold();
        s.mutex.unlock(s.io);
    }

    /// The lock's holder: its hold ends now (counted), before it unlocks or waits on a condition.
    fn endHold(s: *Store) void {
        s.counters.lock_hold.note(@intCast(@max(0, monoNow(s.io) - s.hold_t0)));
    }

    fn monoNow(io: std.Io) i96 {
        return std.Io.Clock.awake.now(io).toNanoseconds();
    }

    /// A wait on one of the store's conditions, lock held: the hold ends before it (the mutex is released inside)
    /// and starts again once it returns with the mutex: a thread waiting for work never counts as holding the lock.
    fn condWait(s: *Store, cond: *std.Io.Condition) void {
        s.endHold();
        cond.waitUncancelable(s.io, &s.mutex);
        s.hold_t0 = monoNow(s.io);
    }

    /// `condWait` for at most `ns` (it returns with the mutex either way).
    fn condWaitFor(s: *Store, cond: *std.Io.Condition, ns: i96) void {
        s.endHold();
        cond.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(ns), .clock = .awake } }) catch {};
        s.hold_t0 = monoNow(s.io);
    }

    /// Tests: holds the store lock `ns` on the calling thread (`held` set once it has it).
    pub fn testHoldLock(s: *Store, ns: u64, held: *std.atomic.Value(bool)) void {
        comptime std.debug.assert(builtin.is_test);
        s.lockS();
        defer s.unlockS();
        held.store(true, .release);
        std.Io.sleep(s.io, .fromNanoseconds(@intCast(ns)), .awake) catch {};
    }

    fn now(s: *Store) i96 {
        const t = std.Io.Clock.real.now(s.io).toNanoseconds();
        s.clock = @max(t, s.clock + 1); // strictly increasing: last use orders every point
        return s.clock;
    }

    /// Other builds' namespaces: the `keep_builds` used last stay (a namespace directory's mtime is its build's last
    /// open); of the rest, this rank's directory goes unless a live process holds its lock. The lock is taken (not
    /// probed) and held until the directory is gone, so a process
    /// of that build starting meanwhile fails with StoreBusy instead of losing its store underneath it; one that holds
    /// it already is skipped. Symlinks are unlinked, never followed.
    fn pruneBuilds(s: *Store, mine: []const u8, rank: [:0]const u8) void {
        const gpa = s.gpa;
        var names = fsio.list(gpa, s.root_fd) catch return;
        defer fsio.freeList(gpa, &names);
        const Other = struct { name: [:0]const u8, mtime: i96 };
        var others: std.ArrayList(Other) = .empty;
        defer others.deinit(gpa);
        for (names.items) |n| {
            if (!std.mem.startsWith(u8, n, "native-spill-") or std.mem.eql(u8, n, mine)) continue;
            if (fsio.isSymlinkAt(s.root_fd, n)) { // never ours: the link goes, its target is left alone
                fsio.unlinkAt(s.root_fd, n);
                continue;
            }
            const st = fsio.statAt(s.root_fd, n) catch continue;
            others.append(gpa, .{ .name = n, .mtime = st.mtime_ns }) catch return;
        }
        std.mem.sort(Other, others.items, {}, struct {
            fn newer(_: void, a: Other, b: Other) bool {
                return a.mtime > b.mtime;
            }
        }.newer);
        if (others.items.len <= s.cfg.keep_builds) return;
        for (others.items[s.cfg.keep_builds..]) |o| {
            const dir = fsio.openDirAt(s.root_fd, o.name) catch continue;
            defer _ = c.close(dir);
            if (fsio.isSymlinkAt(dir, rank)) fsio.unlinkAt(dir, rank);
            const sub = fsio.openDirAt(dir, rank) catch {
                _ = c.unlinkat(s.root_fd, o.name, c.AT.REMOVEDIR); // empty once every rank pruned its own
                continue;
            };
            defer _ = c.close(sub);
            var lock = fsio.Lock.take(sub, &s.writer) catch continue; // live (or unlockable): left alone
            defer lock.release();
            fsio.removeTree(gpa, dir, rank); // (its owner.lock goes too, while held)
            s.counters.builds_pruned += 1;
            _ = c.unlinkat(s.root_fd, o.name, c.AT.REMOVEDIR);
        }
    }

    // -- recovery ------------------------------------------------------------------------------------------------
    fn pointName(buf: []u8, key: *const Digest, comptime suffix: []const u8) [:0]const u8 {
        const h = format.hex(key);
        return std.fmt.bufPrintSentinel(buf, "{s}" ++ suffix, .{h}, 0) catch unreachable;
    }

    fn blockName(buf: []u8, name: *const Digest) [:0]const u8 {
        const h = format.hex(name);
        return std.fmt.bufPrintSentinel(buf, "{s}.blk", .{h}, 0) catch unreachable;
    }

    /// Points whose manifest, ids, small state and every block are on disk at the sizes the manifest says; the rest
    /// (a write a crash cut short, a damaged or foreign point) go, and so do blocks no point names. Only damage
    /// deletes: a point that could not be read for another reason (I/O error, EMFILE, a busy file) keeps its files,
    /// and while any did, so does every unreferenced block (it may be that point's); the next open tries again. Out of
    /// memory fails the open with the store untouched.
    fn scan(s: *Store) !void {
        const gpa = s.gpa;
        for ([_]c.fd_t{ s.points_fd, s.blocks_fd }) |dir| {
            var names = try fsio.list(gpa, dir);
            defer fsio.freeList(gpa, &names);
            for (names.items) |n| if (n[0] == '.' and !mut(.scan_keeps_temps)) {
                fsio.unlinkAt(dir, n);
                s.counters.temps_removed += 1;
            };
        }
        var marked: std.AutoHashMapUnmanaged(Digest, void) = .empty;
        defer marked.deinit(gpa);
        try s.quarantine(&marked);
        var present: std.AutoHashMapUnmanaged(Digest, fsio.Stat) = .empty;
        defer present.deinit(gpa);
        var unsure: std.AutoHashMapUnmanaged(Digest, void) = .empty; // blocks whose stat failed, not as missing
        defer unsure.deinit(gpa);
        var kept: std.AutoHashMapUnmanaged(Digest, void) = .empty; // points left on disk unread
        defer kept.deinit(gpa);
        {
            var names = try fsio.list(gpa, s.blocks_fd);
            defer fsio.freeList(gpa, &names);
            for (names.items) |n| {
                s.scanned(n);
                const d = if (std.mem.endsWith(u8, n, ".blk")) format.parseHex(n[0 .. n.len - 4]) else null;
                if (d == null) {
                    fsio.unlinkAt(s.blocks_fd, n);
                    s.counters.orphans_removed += 1;
                    continue;
                }
                const st = fsio.statAt(s.blocks_fd, n) catch |e| {
                    if (e != error.Missing) try unsure.put(gpa, d.?, {});
                    continue;
                };
                try present.put(gpa, d.?, st);
            }
        }
        var names = try fsio.list(gpa, s.points_fd);
        defer fsio.freeList(gpa, &names);
        for (names.items) |n| {
            s.scanned(n);
            if (!std.mem.endsWith(u8, n, ".man")) continue;
            const key = format.parseHex(n[0 .. n.len - 4]) orelse continue;
            if (marked.contains(key)) continue; // its manifest could not be unlinked: still never recovered
            const p = s.recover(key, &present, &unsure) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Corrupt, error.Foreign, error.Missing, error.TooLarge, error.Short => null,
                else => if (mut(.recover_drops_on_io)) null else {
                    try kept.put(gpa, key, {});
                    s.counters.recover_deferred += 1;
                    continue;
                },
            };
            if (p) |pt| {
                try s.points.put(gpa, key, pt);
                s.counters.recovered_points += 1;
            } else s.counters.recovered_dropped += 1;
        }
        // files of no point, then blocks no point names
        for (names.items) |n| {
            const dot = std.mem.indexOfScalar(u8, n, '.') orelse n.len;
            const key = format.parseHex(n[0..dot]);
            if (key == null or !(s.points.contains(key.?) or kept.contains(key.?))) {
                fsio.unlinkAt(s.points_fd, n);
                s.counters.orphans_removed += 1;
            }
        }
        var pit = s.points.valueIterator();
        while (pit.next()) |pp| {
            const p = pp.*;
            for (p.blocks) |b| {
                if (!s.have.contains(b.name)) try s.addBlock(b.name, b.crc, b.sha, present.get(b.name).?.ino, b.bytes);
            }
            s.stored_bytes += p.files;
            s.ref(p.blocks, 1);
        }
        // blocks no point names (kept while a point was left unread: they may be its)
        if (kept.count() == 0 and unsure.count() == 0) {
            var bit = present.iterator();
            while (bit.next()) |e| {
                if (s.have.contains(e.key_ptr.*)) continue;
                var bb: [80]u8 = undefined;
                fsio.unlinkAt(s.blocks_fd, blockName(&bb, e.key_ptr));
                s.counters.orphans_removed += 1;
            }
        }
        // parents, shortest first
        var order: std.ArrayList(*Point) = .empty;
        defer order.deinit(gpa);
        pit = s.points.valueIterator();
        while (pit.next()) |p| try order.append(gpa, p.*);
        std.mem.sort(*Point, order.items, {}, struct {
            fn shorter(_: void, a: *Point, b: *Point) bool {
                return a.at < b.at;
            }
        }.shorter);
        for (order.items) |p| try s.parent.put(gpa, p.key, s.parentOf(p));
        s.clock = 0;
        for (order.items) |p| s.clock = @max(s.clock, p.used);
    }

    /// At open: every point a restore marker names is forgotten before anything reads it (a restore was in flight, or
    /// its state had not served a round, when the engine stopped: it may be what stopped it). Its manifest, small
    /// state and ids go, the points folder is fsynced, then the marker; its blocks go with the scan's unreferenced
    /// ones. A stale marker (its point gone already) is removed too. `out` gets every marked key; recovery skips them.
    fn quarantine(s: *Store, out: *std.AutoHashMapUnmanaged(Digest, void)) !void {
        const gpa = s.gpa;
        var names = try fsio.list(gpa, s.points_fd);
        defer fsio.freeList(gpa, &names);
        var nb: [80]u8 = undefined;
        for (names.items) |n| {
            if (!std.mem.endsWith(u8, n, ".rst")) continue;
            const key = format.parseHex(n[0 .. n.len - 4]) orelse continue; // no key: an orphan, collected below
            try out.put(gpa, key, {});
            const had = c.unlinkat(s.points_fd, pointName(&nb, &key, ".man"), 0) == 0;
            fsio.unlinkAt(s.points_fd, pointName(&nb, &key, ".st"));
            fsio.unlinkAt(s.points_fd, pointName(&nb, &key, ".ids"));
            if (had) {
                s.counters.quarantined += 1;
                const x = format.hex(&key);
                std.log.warn("spill: rank {d}: point {s} was being restored when the engine stopped: forgotten (quarantine)", .{ s.cfg.rank, x[0..16] });
            } else s.counters.orphans_removed += 1;
        }
        if (out.count() == 0) return;
        fsio.syncDir(&s.hooks, s.points_fd, "points") catch {}; // the point files gone durably before their markers
        var it = out.keyIterator();
        while (it.next()) |k| fsio.unlinkAt(s.points_fd, pointName(&nb, k, ".rst"));
    }

    /// One name the scan read: counted for a watcher on another thread (Hooks.progress), and the test's hook.
    fn scanned(s: *Store, name: []const u8) void {
        if (s.hooks.progress) |p| _ = p.fetchAdd(1, .monotonic);
        if (s.hooks.scanned) |f| f(s.hooks.ctx, name);
    }

    fn recover(s: *Store, key: Digest, present: *const std.AutoHashMapUnmanaged(Digest, fsio.Stat), unsure: *const std.AutoHashMapUnmanaged(Digest, void)) !*Point {
        const gpa = s.gpa;
        const l = s.cfg.limits;
        var nb: [80]u8 = undefined;
        var man = try fsio.readSmall(&s.writer, gpa, s.points_fd, pointName(&nb, &key, ".man"), l.maxManifest());
        defer man.deinit(gpa);
        const raw = man.bytes();
        var m = try format.Manifest.parse(gpa, raw, s.cfg.compat, l);
        defer {
            gpa.free(m.media);
            gpa.free(m.blocks);
            gpa.free(m.sections);
        }
        var lineage_owned = true;
        defer if (lineage_owned) gpa.free(m.lineage.forks);
        if (!std.mem.eql(u8, &m.key, &key) or m.owner_layout != s.cfg.owner_layout) return error.Corrupt;
        const man_st = try fsio.statAt(s.points_fd, pointName(&nb, &key, ".man"));
        var ids_file = try fsio.readSmall(&s.writer, gpa, s.points_fd, pointName(&nb, &key, ".ids"), m.ids_bytes);
        defer ids_file.deinit(gpa);
        // its length and CRC, the key its ids' chain gives, and with pictures its media (format.checkIds)
        const ids = try format.checkIds(gpa, ids_file.bytes(), &m);
        const keyed = ids.keyed;
        defer gpa.free(keyed);
        const ch = ids.chain;
        var ch_owned = true;
        defer if (ch_owned) gpa.free(ch);
        const small_st = try fsio.statAt(s.points_fd, pointName(&nb, &key, ".st"));
        if (small_st.size != m.small_bytes) return error.Corrupt;
        const spans = try gpa.alloc(format.Span, m.blocks.len);
        defer gpa.free(spans);
        format.blockNames(m.lineage, ch, keyed, l.block_tokens, spans);
        for (m.blocks, spans) |b, sp| {
            if (!std.mem.eql(u8, &b.name, &sp.name)) return error.Corrupt;
            const st = present.get(b.name) orelse return if (unsure.contains(b.name)) error.Io else error.Missing;
            if (st.size != b.bytes and !mut(.recover_no_size_check)) return error.Corrupt;
        }
        const p = try s.makePoint(keyed, ch, m.media, m.lineage, .{
            .key = key,
            .at = m.at,
            .shared = m.flags.shared,
            .group = m.group,
            .created_ns = m.created_ns,
            .has_raw = m.flags.has_raw,
            .mutable_tail = m.flags.mutable_tail,
            .origin = m.origin,
        });
        lineage_owned = false; // the point owns them now
        ch_owned = false;
        errdefer p.deinit(gpa);
        @memcpy(p.blocks, m.blocks);
        gpa.free(p.sections);
        p.sections = try gpa.dupe(format.Section, m.sections);
        p.ids_bytes = m.ids_bytes;
        p.ids_crc = m.ids_crc;
        p.small_bytes = m.small_bytes;
        p.small_crc = m.small_crc;
        p.used = man_st.mtime_ns;
        p.files = man_st.size + m.ids_bytes + m.small_bytes;
        return p;
    }

    const PointInit = struct {
        key: Digest,
        at: u32,
        shared: bool,
        group: u64,
        created_ns: i64,
        has_raw: bool,
        mutable_tail: bool,
        origin: ?format.Origin,
    };

    /// `keyed`'s id chain (gpa-owned): the one pass over a state's ids its key and block names derive from.
    fn chainOf(gpa: Allocator, keyed: []const u32, block: u32) ![]Digest {
        var ch: std.ArrayList(Digest) = .empty;
        errdefer ch.deinit(gpa);
        try format.chain(gpa, keyed, block, &ch);
        return ch.toOwnedSlice(gpa);
    }

    /// A point for `keyed` (tail, media, block names) with its chain `ch`; on success it owns `ch` and `lineage`'s
    /// forks (on an error the caller still owns both).
    fn makePoint(s: *Store, keyed: []const u32, ch: []Digest, media: []const Media, lineage: Lineage, i: PointInit) !*Point {
        const gpa = s.gpa;
        const block = s.cfg.limits.block_tokens;
        const p = try gpa.create(Point);
        errdefer gpa.destroy(p);
        const whole = ch.len * block;
        const tail = try gpa.dupe(u32, keyed[whole..]);
        errdefer gpa.free(tail);
        const md = try gpa.dupe(Media, media);
        errdefer gpa.free(md);
        const nblocks = std.math.divCeil(usize, keyed.len, block) catch unreachable;
        const spans = try gpa.alloc(format.Span, nblocks);
        defer gpa.free(spans);
        format.blockNames(lineage, ch, keyed, block, spans);
        const blocks = try gpa.alloc(format.BlockRef, nblocks);
        for (blocks, spans) |*b, sp| b.* = .{ .name = sp.name, .tokens = sp.end - sp.start, .bytes = 0, .crc = 0, .sha = format.zero_digest };
        p.* = .{
            .key = i.key,
            .at = i.at,
            .shared = i.shared,
            .group = i.group,
            .used = 0,
            .created_ns = i.created_ns,
            .has_raw = i.has_raw,
            .mutable_tail = i.mutable_tail,
            .lineage = lineage,
            .origin = i.origin,
            .chain = ch,
            .tail = tail,
            .media = md,
            .blocks = blocks,
            .sections = &.{},
            .ids_bytes = 0,
            .ids_crc = 0,
            .small_bytes = 0,
            .small_crc = 0,
        };
        return p;
    }

    // -- block and reference bookkeeping (lock held) -------------------------------------------------------------
    fn addBlock(s: *Store, name: Digest, crc: u32, sha: Digest, ino: u64, bytes: u64) !void {
        const gpa = s.gpa;
        try s.have.put(gpa, name, .{ .crc = crc, .sha = sha, .ino = ino, .bytes = bytes });
        const gop = try s.inodes.getOrPut(gpa, ino);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{ .bytes = bytes };
            s.stored_bytes += bytes;
        }
        try gop.value_ptr.names.append(gpa, name);
        const cg = try s.content.getOrPut(gpa, sha);
        if (!cg.found_existing) cg.value_ptr.* = name;
    }

    /// Block `name` leaves the index; its bytes once no name holds its inode. Its file is unlinked by the writer.
    fn dropBlock(s: *Store, name: Digest) void {
        const info = (s.have.fetchRemove(name) orelse return).value;
        var freed: u64 = 0;
        if (s.inodes.getPtr(info.ino)) |ino| {
            for (ino.names.items, 0..) |n, i| if (std.mem.eql(u8, &n, &name)) {
                _ = ino.names.swapRemove(i);
                break;
            };
            if (s.content.get(info.sha)) |cur| if (std.mem.eql(u8, &cur, &name)) {
                if (ino.names.items.len > 0) s.content.putAssumeCapacity(info.sha, ino.names.items[0]) else _ = s.content.remove(info.sha);
            };
            if (ino.names.items.len == 0) {
                s.stored_bytes -= ino.bytes;
                freed = ino.bytes;
                ino.names.deinit(s.gpa);
                _ = s.inodes.remove(info.ino);
            }
        }
        s.pending_free += freed;
        s.queue(.{ .unlink_block = .{ .name = name, .bytes = freed } });
    }

    fn ref(s: *Store, blocks: []const format.BlockRef, k: i32) void {
        for (blocks) |b| {
            const gop = s.refs.getOrPut(s.gpa, b.name) catch continue;
            const cur: i64 = if (gop.found_existing) gop.value_ptr.* else 0;
            const v = cur + k;
            if (v > 0) gop.value_ptr.* = @intCast(v) else _ = s.refs.remove(b.name);
        }
    }

    /// Of `blocks`, those nothing names any more (no point, waiting write or write in flight) go.
    fn gc(s: *Store, blocks: []const format.BlockRef) void {
        for (blocks) |b| {
            if (s.have.contains(b.name) and !s.refs.contains(b.name) and !s.inflight.contains(b.name)) s.dropBlock(b.name);
        }
    }

    fn queue(s: *Store, t: Task) void {
        s.tasks.append(s.gpa, t) catch {
            std.log.err("spill: rank {d}: writer queue allocation failed", .{s.cfg.rank});
            return;
        };
        s.work.signal(s.io);
    }

    fn complete(s: *Store, x: Completion) void {
        s.completions.append(s.gpa, x) catch {};
    }

    /// A point goes from the index (its files queued for the writer), then the blocks only it named.
    fn remove(s: *Store, key: Digest, comptime why: []const u8) void {
        const kv = s.points.fetchRemove(key) orelse return;
        const p = kv.value;
        s.ref(p.blocks, -1);
        const up = if (s.parent.fetchRemove(key)) |e| e.value else null;
        var it = s.parent.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.*) |v| if (std.mem.eql(u8, &v, &key)) {
                e.value_ptr.* = up;
            };
        }
        s.stored_bytes -= p.files;
        s.pending_free += p.files;
        s.queue(.{ .unlink_point = .{ .key = key, .bytes = p.files } });
        @field(s.counters, why) += 1;
        s.gc(p.blocks);
        if (p.pins > 0) p.removed = true else p.deinit(s.gpa);
    }

    /// Whether `p`'s ids strictly extend `q`'s (block chains, then q's tail). A tail of q inside one of p's whole
    /// blocks cannot be compared from memory (only the last block's ids are kept): not counted as an extension.
    fn extends(s: *Store, p: *const Point, q: *const Point) bool {
        _ = s;
        if (q.at >= p.at) return false;
        const k = q.chain.len;
        if (k > p.chain.len) return false;
        for (q.chain, p.chain[0..k]) |a, b| if (!std.mem.eql(u8, &a, &b)) return false;
        if (q.tail.len == 0) return true;
        if (k != p.chain.len) return false;
        return std.mem.eql(u32, p.tail[0..q.tail.len], q.tail);
    }

    fn parentOf(s: *Store, p: *const Point) ?Digest {
        var best: ?*Point = null;
        var it = s.points.valueIterator();
        while (it.next()) |qq| {
            const q = qq.*;
            if (q == p) continue;
            if ((best == null or q.at > best.?.at) and s.extends(p, q)) best = q;
        }
        return if (best) |b| b.key else null;
    }

    /// `p` just entered the index: its parent, and the points it now stands between.
    fn adopt(s: *Store, p: *Point) void {
        s.parent.put(s.gpa, p.key, s.parentOf(p)) catch {};
        var it = s.points.valueIterator();
        while (it.next()) |qq| {
            const q = qq.*;
            if (q == p or !s.extends(q, p)) continue;
            const cur = s.parent.get(q.key) orelse null;
            if (cur == null or s.points.get(cur.?).?.at < p.at) s.parent.put(s.gpa, q.key, p.key) catch {};
        }
    }

    // -- conversations and room (lock held) ----------------------------------------------------------------------
    const Tree = struct {
        keys: []Digest,
        conv: []?usize, // a conversation id per key (null: a shared point)
        index: std.AutoHashMapUnmanaged(Digest, usize),

        fn deinit(t: *Tree, gpa: Allocator) void {
            gpa.free(t.keys);
            gpa.free(t.conv);
            t.index.deinit(gpa);
        }
    };

    fn find(uf: []usize, x: usize) usize {
        var r = x;
        while (uf[r] != r) r = uf[r];
        var y = x;
        while (uf[y] != r) {
            const nx = uf[y];
            uf[y] = r;
            y = nx;
        }
        return r;
    }

    /// Each point's conversation: the topmost non-shared point on its path below a shared one, merged with every
    /// point carrying the same nonzero 0111 group (a rewound or restored conversation counts once).
    fn tree(s: *Store) !Tree {
        const gpa = s.gpa;
        const n = s.points.count();
        const keys = try gpa.alloc(Digest, n);
        errdefer gpa.free(keys);
        var index: std.AutoHashMapUnmanaged(Digest, usize) = .empty;
        errdefer index.deinit(gpa);
        var it = s.points.keyIterator();
        var i: usize = 0;
        while (it.next()) |k| : (i += 1) {
            keys[i] = k.*;
            try index.put(gpa, k.*, i);
        }
        std.mem.sort(Digest, keys, {}, struct {
            fn lt(_: void, a: Digest, b: Digest) bool {
                return std.mem.order(u8, &a, &b) == .lt;
            }
        }.lt);
        for (keys, 0..) |k, j| index.putAssumeCapacity(k, j);
        const uf = try gpa.alloc(usize, n);
        defer gpa.free(uf);
        for (uf, 0..) |*u, j| u.* = j;
        const conv = try gpa.alloc(?usize, n);
        errdefer gpa.free(conv);
        var groups: std.AutoHashMapUnmanaged(u64, usize) = .empty;
        defer groups.deinit(gpa);
        for (keys, 0..) |k, j| {
            const p = s.points.get(k).?;
            if (p.shared) {
                conv[j] = null;
                continue;
            }
            var top = j;
            var cur = s.parent.get(k) orelse null;
            while (cur) |ck| {
                const q = s.points.get(ck) orelse break;
                if (q.shared) break;
                top = index.get(ck).?;
                cur = s.parent.get(ck) orelse null;
            }
            conv[j] = top;
        }
        for (keys, 0..) |k, j| {
            const top = conv[j] orelse continue;
            const p = s.points.get(k).?;
            if (p.group == 0) continue;
            const gop = try groups.getOrPut(gpa, p.group);
            if (!gop.found_existing) gop.value_ptr.* = top else {
                const a = find(uf, top);
                const b = find(uf, gop.value_ptr.*);
                if (a != b) uf[@max(a, b)] = @min(a, b);
            }
        }
        for (conv) |*x| if (x.*) |t| {
            x.* = find(uf, t);
        };
        return .{ .keys = keys, .conv = conv, .index = index };
    }

    pub const Why = enum { pruned_turns, pruned_forks, pruned_conversations, pruned_shared };
    /// `group`: a whole conversation's points share one (1, 2, ... in eviction order; 0 otherwise): makeRoom takes a
    /// conversation it begins whole.
    pub const Victim = struct { key: Digest, why: Why, group: u32 = 0 };

    /// The order room would be made in now (gpa-owned): what /health or a test inspects, nothing removed.
    pub fn evictionOrder(s: *Store) ![]Victim {
        s.lockS();
        defer s.unlockS();
        var v = try s.victims(null);
        return v.toOwnedSlice(s.gpa);
    }

    /// Tests: member visits and path steps victims() made, counted in test builds only.
    pub var test_victim_steps: std.atomic.Value(u64) = .init(0);

    fn countSteps(k: usize) void {
        if (builtin.is_test) _ = test_victim_steps.fetchAdd(k, .monotonic);
    }

    /// Points in eviction order: old turns, abandoned forks, whole conversations by last use, shared states last.
    /// `only`: a conversation's own old turns and forks (its quota). The points are grouped by conversation once (a
    /// counting sort over the conversation ids, which are point indices) and each conversation walks only its own
    /// members, so a keep at the cap costs O(n log n). The order is `victimsScan`'s exactly (members in ascending
    /// index, conversations in their first member's order, the same strict picks and stable sorts).
    fn victims(s: *Store, only: ?usize) !std.ArrayList(Victim) {
        const gpa = s.gpa;
        var t = try s.tree();
        defer t.deinit(gpa);
        const n = t.keys.len;
        var out: std.ArrayList(Victim) = .empty;
        errdefer out.deinit(gpa);
        const used = try gpa.alloc(i96, n);
        defer gpa.free(used);
        const at = try gpa.alloc(u32, n);
        defer gpa.free(at);
        const par = try gpa.alloc(?usize, n);
        defer gpa.free(par);
        const has_kid = try gpa.alloc(bool, n);
        defer gpa.free(has_kid);
        @memset(has_kid, false);
        for (t.keys, 0..) |k, j| {
            const p = s.points.get(k).?;
            used[j] = p.used;
            at[j] = p.at;
            par[j] = if (s.parent.get(k) orelse null) |pk| t.index.get(pk) else null;
        }
        for (0..n) |j| if (par[j]) |pj| {
            if (t.conv[j] != null and t.conv[pj] == t.conv[j]) has_kid[pj] = true;
        };
        const kind = try gpa.alloc(u8, n); // 0 keep, 1 turn, 2 fork
        defer gpa.free(kind);
        @memset(kind, 0);
        // each conversation's members, once: `first[cv] ..first[cv + 1]` in `members`, ascending point index
        const first = try gpa.alloc(usize, n + 1);
        defer gpa.free(first);
        @memset(first, 0);
        for (t.conv) |x| if (x) |cv| {
            first[cv + 1] += 1;
        };
        for (0..n) |i| first[i + 1] += first[i];
        const members = try gpa.alloc(usize, first[n]);
        defer gpa.free(members);
        const fill = try gpa.dupe(usize, first[0..n]);
        defer gpa.free(fill);
        for (t.conv, 0..) |x, j| if (x) |cv| {
            members[fill[cv]] = j;
            fill[cv] += 1;
        };
        const Whole = struct { conv: usize, used: i96 };
        var wholes: std.ArrayList(Whole) = .empty;
        defer wholes.deinit(gpa);
        const path_of = try gpa.alloc(usize, n); // the conversation (+1) whose active path a point is on (0: none)
        defer gpa.free(path_of);
        @memset(path_of, 0);
        var steps: usize = 0;
        for (0..n) |j0| {
            const cv = t.conv[j0] orelse continue;
            const mine = members[first[cv]..first[cv + 1]];
            if (mine[0] != j0) continue; // each conversation once, at its first member (victimsScan's order)
            if (only != null and only.? != cv) continue;
            var newest: ?usize = null;
            var newest_any: ?usize = null;
            var last: i96 = std.math.minInt(i96);
            for (0..mine.len) |m| {
                const j = mine[if (mut(.victims_tie_order)) mine.len - 1 - m else m];
                last = @max(last, used[j]);
                if (newest_any == null or used[j] > used[newest_any.?]) newest_any = j;
                if (!has_kid[j] and (newest == null or used[j] > used[newest.?])) newest = j;
            }
            var path_len: usize = 0;
            var cur: ?usize = newest orelse newest_any;
            const keep_n = @max(1, s.cfg.turns);
            while (cur) |j| {
                if (t.conv[j] != cv) break;
                path_of[j] = cv + 1;
                if (path_len >= keep_n) kind[j] = 1;
                path_len += 1;
                cur = par[j];
            }
            for (mine) |j| {
                if (path_of[j] == cv + 1) continue;
                kind[j] = if (has_kid[j]) 1 else 2;
            }
            steps += 2 * mine.len + path_len;
            try wholes.append(gpa, .{ .conv = cv, .used = last });
        }
        const Ent = struct { j: usize, used: i96, at: u32 };
        var turns: std.ArrayList(Ent) = .empty;
        defer turns.deinit(gpa);
        var forks: std.ArrayList(Ent) = .empty;
        defer forks.deinit(gpa);
        for (0..n) |j| switch (kind[j]) {
            1 => try turns.append(gpa, .{ .j = j, .used = used[j], .at = at[j] }),
            2 => try forks.append(gpa, .{ .j = j, .used = used[j], .at = at[j] }),
            else => {},
        };
        const byUsed = struct {
            fn lt(_: void, a: Ent, b: Ent) bool {
                return a.used < b.used or (a.used == b.used and a.at > b.at);
            }
        }.lt;
        std.mem.sort(Ent, turns.items, {}, byUsed);
        std.mem.sort(Ent, forks.items, {}, byUsed);
        for (turns.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_turns });
        for (forks.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_forks });
        if (only != null) {
            countSteps(steps);
            return out;
        }
        if (mut(.evict_shared_first)) {
            var sh: std.ArrayList(Ent) = .empty;
            defer sh.deinit(gpa);
            for (0..n) |j| if (t.conv[j] == null) try sh.append(gpa, .{ .j = j, .used = used[j], .at = at[j] });
            std.mem.sort(Ent, sh.items, {}, byUsed);
            for (sh.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_shared });
        }
        std.mem.sort(Whole, wholes.items, {}, struct {
            fn lt(_: void, a: Whole, b: Whole) bool {
                return a.used < b.used;
            }
        }.lt);
        var rest: std.ArrayList(Ent) = .empty;
        defer rest.deinit(gpa);
        for (wholes.items, 1..) |w, g| {
            rest.clearRetainingCapacity();
            const mine = members[first[w.conv]..first[w.conv + 1]];
            for (mine) |j| if (kind[j] == 0) try rest.append(gpa, .{ .j = j, .used = used[j], .at = at[j] });
            steps += mine.len;
            std.mem.sort(Ent, rest.items, {}, struct {
                fn longer(_: void, a: Ent, b: Ent) bool {
                    return a.at > b.at;
                }
            }.longer);
            for (rest.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_conversations, .group = @intCast(g) });
        }
        rest.clearRetainingCapacity();
        if (!mut(.evict_shared_first)) for (0..n) |j| if (t.conv[j] == null) try rest.append(gpa, .{ .j = j, .used = used[j], .at = at[j] });
        std.mem.sort(Ent, rest.items, {}, byUsed);
        for (rest.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_shared });
        countSteps(steps);
        return out;
    }

    /// Tests: the ungrouped victims(), the differential test's reference (`testVictims`): for every conversation two
    /// passes over every point, a memset and one more pass per whole conversation.
    fn victimsScan(s: *Store, only: ?usize) !std.ArrayList(Victim) {
        const gpa = s.gpa;
        var t = try s.tree();
        defer t.deinit(gpa);
        const n = t.keys.len;
        var out: std.ArrayList(Victim) = .empty;
        errdefer out.deinit(gpa);
        const used = try gpa.alloc(i96, n);
        defer gpa.free(used);
        const at = try gpa.alloc(u32, n);
        defer gpa.free(at);
        const par = try gpa.alloc(?usize, n);
        defer gpa.free(par);
        const has_kid = try gpa.alloc(bool, n);
        defer gpa.free(has_kid);
        @memset(has_kid, false);
        for (t.keys, 0..) |k, j| {
            const p = s.points.get(k).?;
            used[j] = p.used;
            at[j] = p.at;
            par[j] = if (s.parent.get(k) orelse null) |pk| t.index.get(pk) else null;
        }
        for (0..n) |j| if (par[j]) |pj| {
            if (t.conv[j] != null and t.conv[pj] == t.conv[j]) has_kid[pj] = true;
        };
        const kind = try gpa.alloc(u8, n); // 0 keep, 1 turn, 2 fork
        defer gpa.free(kind);
        @memset(kind, 0);
        const Whole = struct { conv: usize, used: i96 };
        var wholes: std.ArrayList(Whole) = .empty;
        defer wholes.deinit(gpa);
        var seen: std.AutoHashMapUnmanaged(usize, void) = .empty;
        defer seen.deinit(gpa);
        const on_path = try gpa.alloc(bool, n);
        defer gpa.free(on_path);
        for (0..n) |j0| {
            const cv = t.conv[j0] orelse continue;
            if ((try seen.getOrPut(gpa, cv)).found_existing) continue;
            if (only != null and only.? != cv) continue;
            var newest: ?usize = null;
            var newest_any: ?usize = null;
            var last: i96 = std.math.minInt(i96);
            for (0..n) |j| {
                if (t.conv[j] != cv) continue;
                last = @max(last, used[j]);
                if (newest_any == null or used[j] > used[newest_any.?]) newest_any = j;
                if (!has_kid[j] and (newest == null or used[j] > used[newest.?])) newest = j;
            }
            @memset(on_path, false);
            var path_len: usize = 0;
            var cur: ?usize = newest orelse newest_any;
            const keep_n = @max(1, s.cfg.turns);
            while (cur) |j| {
                if (t.conv[j] != cv) break;
                on_path[j] = true;
                if (path_len >= keep_n) kind[j] = 1;
                path_len += 1;
                cur = par[j];
            }
            for (0..n) |j| {
                if (t.conv[j] != cv or on_path[j]) continue;
                kind[j] = if (has_kid[j]) 1 else 2;
            }
            try wholes.append(gpa, .{ .conv = cv, .used = last });
        }
        const Ent = struct { j: usize, used: i96, at: u32 };
        var turns: std.ArrayList(Ent) = .empty;
        defer turns.deinit(gpa);
        var forks: std.ArrayList(Ent) = .empty;
        defer forks.deinit(gpa);
        for (0..n) |j| switch (kind[j]) {
            1 => try turns.append(gpa, .{ .j = j, .used = used[j], .at = at[j] }),
            2 => try forks.append(gpa, .{ .j = j, .used = used[j], .at = at[j] }),
            else => {},
        };
        const byUsed = struct {
            fn lt(_: void, a: Ent, b: Ent) bool {
                return a.used < b.used or (a.used == b.used and a.at > b.at);
            }
        }.lt;
        std.mem.sort(Ent, turns.items, {}, byUsed);
        std.mem.sort(Ent, forks.items, {}, byUsed);
        for (turns.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_turns });
        for (forks.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_forks });
        if (only != null) return out;
        if (mut(.evict_shared_first)) {
            var sh: std.ArrayList(Ent) = .empty;
            defer sh.deinit(gpa);
            for (0..n) |j| if (t.conv[j] == null) try sh.append(gpa, .{ .j = j, .used = used[j], .at = at[j] });
            std.mem.sort(Ent, sh.items, {}, byUsed);
            for (sh.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_shared });
        }
        std.mem.sort(Whole, wholes.items, {}, struct {
            fn lt(_: void, a: Whole, b: Whole) bool {
                return a.used < b.used;
            }
        }.lt);
        var rest: std.ArrayList(Ent) = .empty;
        defer rest.deinit(gpa);
        for (wholes.items) |w| {
            rest.clearRetainingCapacity();
            for (0..n) |j| if (t.conv[j] == w.conv and kind[j] == 0) try rest.append(gpa, .{ .j = j, .used = used[j], .at = at[j] });
            std.mem.sort(Ent, rest.items, {}, struct {
                fn longer(_: void, a: Ent, b: Ent) bool {
                    return a.at > b.at;
                }
            }.longer);
            for (rest.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_conversations });
        }
        rest.clearRetainingCapacity();
        if (!mut(.evict_shared_first)) for (0..n) |j| if (t.conv[j] == null) try rest.append(gpa, .{ .j = j, .used = used[j], .at = at[j] });
        std.mem.sort(Ent, rest.items, {}, byUsed);
        for (rest.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_shared });
        return out;
    }

    // -- test seams (never referenced outside test builds) -------------------------------------------------------
    /// Tests: a point in the index only (no files, no blocks), with `used` and `group` as given, linked to its parent
    /// and its children as a publish links it (`adopt`). Its key (an existing key is left as it is).
    pub fn testInsert(s: *Store, keyed: []const u32, lineage: Lineage, shared: bool, group: u64, used: i96) !Digest {
        comptime std.debug.assert(builtin.is_test);
        const gpa = s.gpa;
        const block = s.cfg.limits.block_tokens;
        const ch = try chainOf(gpa, keyed, block);
        const key = format.pointKeyChained(ch, keyed, block, &.{});
        s.lockS();
        defer s.unlockS();
        if (s.points.contains(key)) {
            gpa.free(ch);
            return key;
        }
        var lineage_copy = lineage.clone(gpa) catch |e| {
            gpa.free(ch);
            return e;
        };
        const p = s.makePoint(keyed, ch, &.{}, lineage_copy, .{
            .key = key,
            .at = @intCast(keyed.len),
            .shared = shared,
            .group = group,
            .created_ns = 0,
            .has_raw = false,
            .mutable_tail = keyed.len % block != 0,
            .origin = null,
        }) catch |e| {
            lineage_copy.deinit(gpa);
            gpa.free(ch);
            return e;
        };
        p.used = used;
        s.points.put(gpa, key, p) catch |e| {
            p.deinit(gpa);
            return e;
        };
        s.adopt(p);
        return key;
    }

    /// Tests: every in-memory point gone (`testInsert`'s, or a store's without files).
    pub fn testClear(s: *Store) void {
        comptime std.debug.assert(builtin.is_test);
        s.lockS();
        defer s.unlockS();
        var it = s.points.valueIterator();
        while (it.next()) |p| p.*.deinit(s.gpa);
        s.points.clearRetainingCapacity();
        s.parent.clearRetainingCapacity();
    }

    /// Tests: victims(only), or the reference victimsScan(only) when the flag is set, gpa-owned.
    pub fn testVictims(s: *Store, only: ?usize, by_scan: bool) ![]Victim {
        comptime std.debug.assert(builtin.is_test);
        s.lockS();
        defer s.unlockS();
        var v = if (by_scan) try s.victimsScan(only) else try s.victims(only);
        return v.toOwnedSlice(s.gpa);
    }

    /// Tests: each conversation tree() gives now, once (gpa-owned).
    pub fn testConversations(s: *Store) ![]usize {
        comptime std.debug.assert(builtin.is_test);
        s.lockS();
        defer s.unlockS();
        var t = try s.tree();
        defer t.deinit(s.gpa);
        var out: std.ArrayList(usize) = .empty;
        errdefer out.deinit(s.gpa);
        for (t.conv) |cv| if (cv) |x| if (std.mem.indexOfScalar(usize, out.items, x) == null) try out.append(s.gpa, x);
        return out.toOwnedSlice(s.gpa);
    }

    fn removeWhy(s: *Store, key: Digest, why: Why) void {
        switch (why) {
            inline else => |w| s.remove(key, @tagName(w)),
        }
    }

    /// Frees at least `nbytes` in eviction order, never `protect` or a point being read. A conversation it begins to
    /// evict goes whole, past `nbytes` if need be, so it never loses only its newest resume points; it is counted once
    /// in pruned_conversations, its points in pruned_conversation_points.
    fn makeRoom(s: *Store, nbytes: u64, protect: ?Digest) void {
        var v = s.victims(null) catch return;
        defer v.deinit(s.gpa);
        const target = s.stored_bytes -| nbytes;
        var whole: u32 = 0; // the conversation being evicted (its group), 0: none
        for (v.items) |x| {
            const finishing = whole != 0 and x.why == .pruned_conversations and x.group == whole and !mut(.conversation_pieces);
            if (s.stored_bytes <= target and !finishing) return;
            if (protect != null and std.mem.eql(u8, &protect.?, &x.key)) continue;
            if (!s.points.contains(x.key) or s.reading.contains(x.key)) continue;
            if (x.why == .pruned_conversations) {
                if (x.group != whole) {
                    whole = x.group;
                    s.counters.pruned_conversations += 1;
                }
                s.remove(x.key, "pruned_conversation_points");
            } else s.removeWhy(x.key, x.why);
        }
    }

    /// After `p` is published: on its conversation's path (its ancestors below a shared point) the newest `turns`
    /// stay, older ones go now (they free their small files only: `p` names every block they named).
    fn trimPath(s: *Store, p: *Point) void {
        if (s.cfg.turns == 0 or p.shared) return;
        var path: std.ArrayList(Digest) = .empty;
        defer path.deinit(s.gpa);
        var cur: ?Digest = p.key;
        while (cur) |k| {
            const q = s.points.get(k) orelse break;
            if (q.shared) break;
            path.append(s.gpa, k) catch return;
            cur = s.parent.get(k) orelse null;
        }
        if (path.items.len <= s.cfg.turns) return;
        for (path.items[s.cfg.turns..]) |k| {
            if (!s.reading.contains(k)) s.remove(k, "trimmed_turns");
        }
    }

    /// Bytes only `keys`' points hold: their files and blocks no other point names (each inode once).
    fn convBytes(s: *Store, keys: []const Digest) u64 {
        const gpa = s.gpa;
        var mine: std.AutoHashMapUnmanaged(Digest, void) = .empty;
        defer mine.deinit(gpa);
        for (keys) |k| mine.put(gpa, k, {}) catch return 0;
        var others: std.AutoHashMapUnmanaged(Digest, void) = .empty;
        defer others.deinit(gpa);
        var it = s.points.iterator();
        while (it.next()) |e| {
            if (mine.contains(e.key_ptr.*)) continue;
            for (e.value_ptr.*.blocks) |b| others.put(gpa, b.name, {}) catch return 0;
        }
        var inos: std.AutoHashMapUnmanaged(u64, void) = .empty;
        defer inos.deinit(gpa);
        var n: u64 = 0;
        for (keys) |k| {
            const p = s.points.get(k).?;
            n += p.files;
            for (p.blocks) |b| {
                if (others.contains(b.name)) continue;
                const info = s.have.get(b.name) orelse continue;
                const ino = s.inodes.get(info.ino) orelse continue;
                var all_mine = true;
                for (ino.names.items) |nm| if (others.contains(nm)) {
                    all_mine = false;
                };
                if (!all_mine) continue;
                if ((inos.getOrPut(gpa, info.ino) catch return n).found_existing) continue;
                n += ino.bytes;
            }
        }
        return n;
    }

    /// A write of `p` whose conversation holds more than its quota: its own old turns and forks go first.
    fn trimOwn(s: *Store, p: *Point, need: u64) void {
        if (s.cfg.quota <= 0 or p.shared or s.points.count() == 0) return;
        const gpa = s.gpa;
        var t = s.tree() catch return;
        defer t.deinit(gpa);
        // the conversation `p` would join: its parent's (below a shared point), or its group's
        var cv: ?usize = null;
        if (s.parentOf(p)) |up| {
            const q = s.points.get(up).?;
            if (!q.shared) cv = t.conv[t.index.get(up).?];
        }
        if (cv == null and p.group != 0) {
            for (t.keys, 0..) |k, j| if (s.points.get(k).?.group == p.group) {
                cv = t.conv[j];
                break;
            };
        }
        const conv = cv orelse return;
        var keys: std.ArrayList(Digest) = .empty;
        defer keys.deinit(gpa);
        for (t.keys, 0..) |k, j| if (t.conv[j] == conv) keys.append(gpa, k) catch return;
        const allowed: u64 = @intFromFloat(s.cfg.quota * @as(f64, @floatFromInt(s.cfg.cap_bytes)));
        const held = s.convBytes(keys.items) + need;
        if (held <= allowed) return;
        s.counters.quota_trims += 1;
        const target = s.stored_bytes -| (held - allowed);
        var v = s.victims(conv) catch return;
        defer v.deinit(gpa);
        for (v.items) |x| {
            if (s.stored_bytes <= target) break;
            if (s.reading.contains(x.key) or !s.points.contains(x.key)) continue;
            s.removeWhy(x.key, x.why);
        }
    }

    // -- writes (engine thread) ----------------------------------------------------------------------------------
    /// At an eligible keep: queue the point's missing blocks (left in device memory; `stage` asks for them between
    /// rounds), reserve its small state's slot (and a tail block's), or say why not. Ids stored or on their way are
    /// not written again; kept again as a shared prefix their point becomes shared, durably.
    pub fn keep(s: *Store, k: Keep) KeepResult {
        const gpa = s.gpa;
        const l = s.cfg.limits;
        const at: u32 = @intCast(@min(k.keyed.len, std.math.maxInt(u32)));
        if (at < s.cfg.min_tokens or at == 0) return s.refuse(.short);
        if (!s.validKeep(k)) return s.refuse(.invalid);
        // hashing runs before the lock (the writer and readers never wait on it), one pass over the ids: the chain,
        // from which the key and the block names derive; none when the caller passes the chain it hashed
        const ch = (if (k.chain) |given| (if (mut(.keep_rehashes)) chainOf(gpa, k.keyed, l.block_tokens) else gpa.dupe(Digest, given)) else chainOf(gpa, k.keyed, l.block_tokens)) catch return s.refuse(.no_staging);
        var ch_owned = true;
        defer if (ch_owned) gpa.free(ch);
        const key = format.pointKeyChained(ch, k.keyed, l.block_tokens, k.media);
        if (s.alreadyKept(key, k.shared)) return .stored;
        var lineage = k.lineage.clone(gpa) catch return s.refuse(.no_staging);
        const p = s.makePoint(k.keyed, ch, k.media, lineage, .{
            .key = key,
            .at = at,
            .shared = k.shared,
            .group = k.group,
            .created_ns = 0,
            .has_raw = k.raw != null,
            .mutable_tail = at % l.block_tokens != 0,
            .origin = k.origin,
        }) catch {
            lineage.deinit(gpa);
            return s.refuse(.no_staging);
        };
        ch_owned = false;
        var ok = false;
        defer if (!ok) p.deinit(gpa);
        // sections, laid out on 256-byte boundaries. A format constant, not a setting: the offsets it gives are
        // written into the manifest and decide the .st file's bytes (validKeep sizes a state the same way); 256 is
        // the alignment CUDA gives its allocations (a family's copy of each section starts aligned)
        const secs = gpa.alloc(format.Section, k.sections.len) catch return s.refuse(.no_staging);
        p.sections = secs;
        var off: u64 = 0;
        for (k.sections, secs) |spec, *sec| {
            off = std.mem.alignForward(u64, off, 256);
            sec.* = .{ .name = format.Name.of(spec.name) catch unreachable, .offset = off, .len = spec.len, .crc = 0 };
            off += spec.len;
        }
        p.small_bytes = off;
        p.ids_bytes = @as(u64, at) * 4 * (if (k.raw != null) @as(u64, 2) else 1);
        const has_tail = at % l.block_tokens != 0;
        const full = l.fullBlockBytes(k.lineage.part).?; // (validKeep: a part where the store needs one)
        for (p.blocks, 0..) |*b, i| b.bytes = if (has_tail and i + 1 == p.blocks.len) k.tail_bytes else full;
        const ids = format.encodeIds(gpa, k.keyed, k.raw) catch return s.refuse(.no_staging);
        var ids_owned = true;
        defer if (ids_owned) gpa.free(ids);
        // the disk as it is now (one fstatvfs, before the lock): the floor must see what other writers took since the
        // store's own last write
        const free_now: ?u64 = if (mut(.floor_stale)) null else fsio.diskFree(&s.hooks, s.blocks_fd);
        s.lockS();
        defer s.unlockS();
        if (free_now) |f| s.disk_free = f;
        s.counters.keeps += 1;
        if (s.points.contains(key) or s.waiting.contains(key)) { // (only the engine thread keeps: never, in practice)
            s.counters.stored_skips += 1;
            return .stored;
        }
        p.created_ns = @intCast(s.now());
        var size: u64 = 0;
        var new: std.ArrayList(u32) = .empty;
        defer if (!ok) new.deinit(gpa);
        var tail: ?u32 = null;
        for (p.blocks, 0..) |b, i| {
            if (s.have.contains(b.name) or s.inflight.contains(b.name)) continue;
            size += b.bytes;
            if (has_tail and i + 1 == p.blocks.len and !mut(.tail_staged_later)) tail = @intCast(i) else new.append(gpa, @intCast(i)) catch return s.refuseOom();
        }
        // the bytes the write adds: new blocks, small state, ids and the manifest's header and block list (80:
        // format's block_ref_bytes); the manifest's forks, media and sections are not counted
        const need = size + p.small_bytes + p.ids_bytes + format.header_bytes + 80 * p.blocks.len;
        // GLM 0110's pin: the point holds its references before room is made, so making room never takes a block it
        // is about to reuse (a resumed least recently used conversation would evict its own parent's blocks, and its
        // write fail on them). Every refusal below gives them back.
        var pinned = false;
        defer if (!ok and pinned) {
            s.ref(p.blocks, -1);
            s.gc(p.blocks);
        };
        if (!mut(.keep_unpinned)) {
            s.ref(p.blocks, 1);
            pinned = true;
        }
        s.trimOwn(p, need);
        const room = s.floorShort(need);
        const over = (s.stored_bytes + s.queued_bytes + need) -| s.cfg.cap_bytes;
        if (@max(room, over) > s.stored_bytes) {
            s.counters.no_space += 1;
            return .no_space;
        }
        if (room > 0 or over > 0) {
            s.makeRoom(@max(room, over), key);
            if (s.floorShort(need) > 0 or s.stored_bytes + s.queued_bytes + need > s.cfg.cap_bytes) {
                s.counters.no_space += 1;
                return .no_space;
            }
        }
        if (s.queued_bytes + size > s.cfg.backlog_bytes) {
            s.counters.backlog_refused += 1;
            return .backlog;
        }
        const id = s.next_id;
        const small_slot = s.slots.take(.small, id) orelse {
            s.counters.no_staging += 1;
            return .no_staging;
        };
        var tail_slot: ?Slot = null;
        if (tail != null) {
            tail_slot = s.slots.take(.write, id) orelse {
                s.slots.give(small_slot);
                s.counters.no_staging += 1;
                return .no_staging;
            };
        }
        const j = gpa.create(WriteJob) catch {
            s.slots.give(small_slot);
            if (tail_slot) |t| s.slots.give(t);
            return s.refuseOom();
        };
        j.* = .{ .id = id, .point = p, .source = k.source, .new = new, .tail = tail, .tail_slot = tail_slot, .small_slot = small_slot, .size = need, .ids = ids };
        if (!pinned) { // (the keep_unpinned mutant: references taken only now)
            s.ref(p.blocks, 1);
            pinned = true;
        }
        s.registerJob(j) catch {
            gpa.destroy(j);
            s.slots.give(small_slot);
            if (tail_slot) |t| s.slots.give(t);
            return s.refuseOom();
        };
        ok = true;
        ids_owned = false;
        s.next_id += 1;
        s.counters.queued += 1;
        const small = s.slots.bytes(small_slot);
        return .{ .queued = .{
            .job = id,
            .small = small[0..@intCast(p.small_bytes)],
            .sections = p.sections,
            .tail = if (tail_slot) |t| s.slots.bytes(t)[0..@intCast(k.tail_bytes)] else null,
        } };
    }

    fn refuseOom(s: *Store) KeepResult {
        s.counters.no_staging += 1;
        return .no_staging;
    }

    /// A refusal decided before the lock (counted under it).
    fn refuse(s: *Store, why: std.meta.Tag(KeepResult)) KeepResult {
        s.lockS();
        defer s.unlockS();
        s.counters.keeps += 1;
        switch (why) {
            .short => s.counters.short += 1,
            .invalid => s.counters.invalid += 1,
            else => s.counters.no_staging += 1,
        }
        return switch (why) {
            .short => .short,
            .invalid => .invalid,
            else => .no_staging,
        };
    }

    /// Ids stored or on their way: not written again. Kept again as a shared prefix their point becomes shared,
    /// durably (its manifest rewritten on the writer, or written so if it is still waiting).
    fn alreadyKept(s: *Store, key: Digest, shared: bool) bool {
        s.lockS();
        defer s.unlockS();
        if (s.points.get(key)) |p| {
            s.counters.keeps += 1;
            p.used = s.now();
            s.queue(.{ .touch = .{ .key = key, .ns = p.used } });
            if (shared and !p.shared and !mut(.no_upgrade)) {
                p.shared = true;
                s.counters.shared_marked += 1;
                if (!mut(.upgrade_not_durable)) s.queue(.{ .shared = key });
            }
            s.counters.stored_skips += 1;
            return true;
        }
        if (s.waiting.get(key)) |j| {
            s.counters.keeps += 1;
            if (shared and !j.point.shared and !mut(.waiting_not_flagged) and !mut(.no_upgrade)) {
                j.point.shared = true; // its manifest takes the flag (or is rewritten after, if already written)
                s.counters.shared_marked += 1;
            }
            s.counters.stored_skips += 1;
            return true;
        }
        return false;
    }

    fn validKeep(s: *Store, k: Keep) bool {
        const l = s.cfg.limits;
        if (k.keyed.len > l.max_tokens) return false;
        if (k.raw) |r| if (r.len != k.keyed.len) return false;
        if (k.chain) |ch| if (ch.len != k.keyed.len / l.block_tokens) return false; // one digest a whole block
        if (k.media.len > l.max_media or k.sections.len > l.max_sections or k.lineage.forks.len > l.max_forks) return false;
        if (l.fullBlockBytes(k.lineage.part) == null) return false; // (owner-sharded: every point names its part)
        // pictures: in order, inside the point, the keyed ids made from their digests and the raw ids beside them
        if (!format.mediaIds(k.keyed, k.raw, k.media)) return false;
        var total: u64 = 0;
        for (k.sections) |sec| {
            if (sec.name.len == 0 or sec.name.len > 32) return false;
            total = std.mem.alignForward(u64, total, 256) + sec.len;
        }
        if (total > l.max_small_bytes or total > s.small_slot) return false;
        const has_tail = k.keyed.len % l.block_tokens != 0;
        if (has_tail and (k.tail_bytes == 0 or k.tail_bytes > l.block_bytes)) return false;
        if (!has_tail and k.tail_bytes != 0) return false;
        return true;
    }

    /// Bytes the floor would be short by after writing `need` more (queued writes count as written).
    fn floorShort(s: *Store, need: u64) u64 {
        if (mut(.floor_ignored)) return 0;
        const want = s.cfg.floor_bytes + need + s.queued_bytes;
        return want -| (s.disk_free + s.pending_free);
    }

    fn registerJob(s: *Store, j: *WriteJob) !void {
        const gpa = s.gpa;
        try s.jobs_by_id.ensureUnusedCapacity(gpa, 1);
        try s.waiting.ensureUnusedCapacity(gpa, 1);
        try s.fifo.ensureUnusedCapacity(gpa, 1);
        try s.inflight.ensureUnusedCapacity(gpa, @intCast(j.new.items.len + 1));
        s.jobs_by_id.putAssumeCapacity(j.id, j);
        s.waiting.putAssumeCapacity(j.point.key, j);
        for (j.new.items) |i| s.inflight.putAssumeCapacity(j.point.blocks[i].name, j.id);
        if (j.tail) |i| s.inflight.putAssumeCapacity(j.point.blocks[i].name, j.id);
        s.queued_bytes += j.size; // (its references were taken by `keep` before it made room)
        s.fifo.appendAssumeCapacity(j);
    }

    /// The engine's capture of `job`'s small state (and tail) completed on the device (`ok`), or failed.
    pub fn captured(s: *Store, job: JobId, ok: bool) void {
        s.lockS();
        defer s.unlockS();
        const j = s.jobs_by_id.get(job) orelse return;
        if (j.capture != .pending) return;
        j.capture = if (ok) .ok else .failed;
        if (!ok and j.state == .open) {
            j.state = .failed;
            j.fail = .capture;
        }
        if (j.tail) |ti| {
            const slot = j.tail_slot.?;
            j.tail_slot = null;
            if (j.state == .open) {
                j.outstanding += 1;
                s.queue(.{ .block = .{ .job = j, .index = ti, .slot = slot } });
            } else s.slots.give(slot);
        }
        s.settle(j);
    }

    /// Between rounds: the waiting points' next blocks, oldest point first, into free write slots (at most
    /// `out.len`). The engine packs each one into `dst` from `source`'s rows (checking the source's generation)
    /// and calls `staged` once its copy completed. Never waits.
    pub fn stage(s: *Store, out: []StageReq) usize {
        s.lockS();
        defer s.unlockS();
        var n: usize = 0;
        var i: usize = 0;
        while (i < s.fifo.items.len and n < out.len) {
            const j = s.fifo.items[i];
            if (j.state != .open) {
                i += 1;
                continue;
            }
            while (j.next < j.new.items.len and n < out.len) {
                const slot = s.slots.take(.write, j.id) orelse return n;
                out[n] = s.req(j, slot);
                n += 1;
            }
            if (j.next >= j.new.items.len) {
                j.listed = false;
                _ = s.fifo.orderedRemove(i);
            } else i += 1;
        }
        return n;
    }

    fn req(s: *Store, j: *WriteJob, slot: Slot) StageReq {
        const idx = j.new.items[j.next];
        j.next += 1;
        j.outstanding += 1;
        j.engine += 1;
        s.counters.staged_blocks += 1;
        const b = j.point.blocks[idx];
        return .{
            .ticket = .{ .job = j.id, .index = idx, .slot = slot },
            .source = j.source,
            .first = idx * s.cfg.limits.block_tokens,
            .tokens = b.tokens,
            .dst = s.slots.bytes(slot)[0..@intCast(b.bytes)],
        };
    }

    /// The engine's copy into a staged slot completed (`ok`), or could not be made (the source changed): the
    /// block goes to the writer, or the point is not written.
    pub fn staged(s: *Store, t: Ticket, ok: bool) void {
        s.lockS();
        defer s.unlockS();
        const j = s.jobs_by_id.get(t.job) orelse {
            s.slots.give(t.slot);
            return;
        };
        j.engine -= 1;
        if (!ok and j.state == .open) {
            j.state = .failed;
            j.fail = .stage;
            s.release(j);
        }
        if (j.state == .open or (ok and s.stillNamed(j, t.index))) {
            s.queue(.{ .block = .{ .job = j, .index = t.index, .slot = t.slot } });
        } else {
            s.slots.give(t.slot);
            j.outstanding -= 1;
            s.dropInflight(j, t.index);
            s.settle(j);
        }
    }

    /// Whether another point or waiting write names block `index` of `j` (its bytes are worth writing anyway).
    fn stillNamed(s: *Store, j: *WriteJob, index: u32) bool {
        return (s.refs.get(j.point.blocks[index].name) orelse 0) > 1;
    }

    fn dropInflight(s: *Store, j: *WriteJob, index: u32) void {
        const name = j.point.blocks[index].name;
        if (s.inflight.get(name)) |owner| if (owner == j.id) {
            _ = s.inflight.remove(name);
            s.recheckParked();
        };
    }

    /// Source rows `source.id` from token `from` on are about to change: waiting points still to stage blocks
    /// there have them staged now into `out` (the engine queues those copies before the change, on its stream),
    /// or, past the free slots or `out`, are cancelled. Points of an older generation of the source are cancelled.
    pub fn guard(s: *Store, source: Source, from: u32, out: []StageReq) GuardResult {
        s.lockS();
        defer s.unlockS();
        var r: GuardResult = .{};
        var i: usize = 0;
        while (i < s.fifo.items.len) {
            const j = s.fifo.items[i];
            i += 1;
            if (j.state != .open or j.source.id != source.id) continue;
            if (j.source.generation != source.generation) {
                s.cancelJob(j, .source_gone);
                r.cancelled += 1;
                i = 0;
                continue;
            }
            // the blocks left reaching past `from`
            var needed: usize = 0;
            for (j.new.items[j.next..]) |idx| {
                const b = j.point.blocks[idx];
                if (idx * s.cfg.limits.block_tokens + b.tokens > from) needed += 1;
            }
            if (needed == 0) continue;
            const left = j.new.items.len - j.next;
            if (left > out.len - r.staged or left > s.slots.available(.write) or mut(.guard_cancels_only)) {
                s.cancelJob(j, .guard);
                r.cancelled += 1;
                i = 0;
                continue;
            }
            while (j.next < j.new.items.len) {
                out[r.staged] = s.req(j, s.slots.take(.write, j.id).?);
                r.staged += 1;
                s.counters.guard_staged += 1;
            }
            j.listed = false;
            for (s.fifo.items, 0..) |x, k| if (x == j) {
                _ = s.fifo.orderedRemove(k);
                break;
            };
            i = 0;
        }
        return r;
    }

    /// Cancel an unpublished point (true), or false when it is past cancelling (already being published).
    pub fn cancel(s: *Store, job: JobId) bool {
        s.lockS();
        defer s.unlockS();
        const j = s.jobs_by_id.get(job) orelse return false;
        if (j.state != .open) return j.state != .publishing;
        s.cancelJob(j, .explicit);
        return true;
    }

    fn cancelJob(s: *Store, j: *WriteJob, why: CancelReason) void {
        j.state = .cancelled;
        j.cancel = why;
        switch (why) {
            .guard => s.counters.cancelled_guard += 1,
            .source_gone => s.counters.cancelled_source += 1,
            else => {},
        }
        s.release(j);
        s.settle(j);
    }

    /// A job that will not publish gives its unstaged blocks to another open job that names them (their rows are
    /// the same bytes there, and their reservation moves with them), else lets their names go.
    fn release(s: *Store, j: *WriteJob) void {
        if (j.listed) {
            for (s.fifo.items, 0..) |x, k| if (x == j) {
                _ = s.fifo.orderedRemove(k);
                break;
            };
            j.listed = false;
        }
        for (j.new.items[j.next..]) |idx| {
            const name = j.point.blocks[idx].name;
            if (!s.handOver(j, name)) {
                if (s.inflight.get(name)) |owner| if (owner == j.id) {
                    _ = s.inflight.remove(name);
                };
            }
        }
        j.next = j.new.items.len;
        if (j.tail) |ti| if (j.tail_slot != null and j.capture != .pending) {
            s.slots.give(j.tail_slot.?);
            j.tail_slot = null;
            s.dropInflight(j, ti);
        };
        s.recheckParked();
    }

    fn handOver(s: *Store, from: *WriteJob, name: Digest) bool {
        var it = s.jobs_by_id.valueIterator();
        while (it.next()) |kk| {
            const k = kk.*;
            if (k == from or k.state != .open) continue;
            for (k.point.blocks, 0..) |b, idx| {
                if (!std.mem.eql(u8, &b.name, &name) or b.tokens != s.cfg.limits.block_tokens) continue;
                s.fifo.ensureUnusedCapacity(s.gpa, 1) catch return false;
                k.new.append(s.gpa, @intCast(idx)) catch return false;
                s.inflight.put(s.gpa, name, k.id) catch {
                    _ = k.new.pop();
                    return false;
                };
                if (!k.listed) {
                    s.fifo.appendAssumeCapacity(k);
                    k.listed = true;
                }
                // its bytes are now `k`'s to write: reserved with `k` (the giver's reservation ends with it)
                if (!mut(.handover_unreserved)) {
                    k.size += b.bytes;
                    s.queued_bytes += b.bytes;
                }
                return true;
            }
        }
        return false;
    }

    /// Lock held: a job ready to publish goes to the writer; a failed or cancelled one with nothing outstanding
    /// ends and frees its staging.
    fn settle(s: *Store, j: *WriteJob) void {
        switch (j.state) {
            .open => if (j.next >= j.new.items.len and j.outstanding == 0 and j.capture == .ok and j.tail_slot == null) {
                j.state = .publishing;
                j.shared_at_write = j.point.shared;
                s.queue(.{ .point = j });
            },
            .publishing => {},
            .failed, .cancelled => if (j.outstanding == 0 and j.capture != .pending) s.endJob(j, false),
        }
    }

    /// Lock held: `j` leaves (published or not); its references, names in flight, reservation and slots go.
    fn endJob(s: *Store, j: *WriteJob, published: bool) void {
        if (j.listed) {
            for (s.fifo.items, 0..) |x, k| if (x == j) {
                _ = s.fifo.orderedRemove(k);
                break;
            };
        }
        for (s.parked.items, 0..) |x, k| if (x == j) {
            _ = s.parked.swapRemove(k);
            break;
        };
        if (j.tail_slot) |t| s.slots.give(t);
        s.slots.give(j.small_slot);
        var it = s.inflight.iterator();
        var drop: std.ArrayList(Digest) = .empty;
        defer drop.deinit(s.gpa);
        while (it.next()) |e| if (e.value_ptr.* == j.id) drop.append(s.gpa, e.key_ptr.*) catch {};
        for (drop.items) |d| _ = s.inflight.remove(d);
        _ = s.waiting.remove(j.point.key);
        _ = s.jobs_by_id.remove(j.id);
        s.queued_bytes -= j.size;
        s.ref(j.point.blocks, -1);
        s.gc(j.point.blocks);
        const key = j.point.key;
        if (published) {
            s.complete(.{ .published = .{ .job = j.id, .key = key, .at = j.point.at } });
        } else if (j.state == .cancelled) {
            s.counters.cancelled += 1;
            s.complete(.{ .cancelled = .{ .job = j.id, .key = key, .why = j.cancel } });
        } else {
            s.counters.write_failed += 1;
            s.complete(.{ .failed = .{ .job = j.id, .key = key, .why = j.fail } });
        }
        freeJob(s.gpa, j, !published);
        s.recheckParked();
        s.done.broadcast(s.io);
    }

    fn recheckParked(s: *Store) void {
        var i: usize = 0;
        while (i < s.parked.items.len) {
            const j = s.parked.items[i];
            if (s.blocksSettled(j)) {
                _ = s.parked.swapRemove(i);
                j.parked = false;
                s.queue(.{ .point = j });
            } else i += 1;
        }
    }

    /// Every block `j` names is on disk or no longer being written by anyone.
    fn blocksSettled(s: *Store, j: *WriteJob) bool {
        for (j.point.blocks) |b| if (!s.have.contains(b.name) and s.inflight.contains(b.name)) return false;
        return true;
    }

    /// Completions since the last poll (published, failed, cancelled points), oldest first.
    pub fn poll(s: *Store, out: []Completion) usize {
        s.lockS();
        defer s.unlockS();
        const n = @min(out.len, s.completions.items.len);
        @memcpy(out[0..n], s.completions.items[0..n]);
        std.mem.copyForwards(Completion, s.completions.items, s.completions.items[n..]);
        s.completions.shrinkRetainingCapacity(s.completions.items.len - n);
        return n;
    }

    /// Unpublished points still waiting for the engine (blocks to stage or a capture to report).
    pub fn pendingStage(s: *Store) usize {
        s.lockS();
        defer s.unlockS();
        var n: usize = 0;
        var it = s.jobs_by_id.valueIterator();
        while (it.next()) |j| {
            if (waitsOnEngine(j.*)) n += 1;
        }
        return n;
    }

    pub const Drain = struct { done: bool, unpublished: usize };

    fn waitsOnEngine(j: *const WriteJob) bool {
        if (j.engine > 0 or j.capture == .pending) return true;
        return j.state == .open and j.next < j.new.items.len;
    }

    /// A clean stop or an idle engine: waits, at most `timeout_ns`, until every write the engine has handed over is
    /// published and every queued metadata rewrite and deletion ran (a sync behind them). Points still waiting for
    /// the engine (staging) are not waited for: they count in `unpublished`, as do those the deadline cut off.
    pub fn drain(s: *Store, timeout_ns: u64) Drain {
        const deadline = std.Io.Clock.awake.now(s.io).toNanoseconds() + timeout_ns;
        s.lockS();
        defer s.unlockS();
        while (true) {
            var engine_wait: usize = 0;
            var writer_wait: usize = 0;
            var it = s.jobs_by_id.valueIterator();
            while (it.next()) |jj| {
                const j = jj.*;
                if (waitsOnEngine(j)) engine_wait += 1 else writer_wait += 1;
            }
            if (writer_wait == 0 and mut(.drain_skips_rewrites)) return .{ .done = engine_wait == 0, .unpublished = engine_wait };
            if (writer_wait == 0) {
                s.sync_sent += 1;
                const want = s.sync_sent;
                s.queue(.{ .sync = want });
                while (s.sync_done < want) {
                    const left = deadline - std.Io.Clock.awake.now(s.io).toNanoseconds();
                    if (left <= 0 or s.crashed) return .{ .done = false, .unpublished = engine_wait };
                    s.condWaitFor(&s.done, @min(left, wait_slice_ns));
                }
                return .{ .done = engine_wait == 0, .unpublished = engine_wait };
            }
            const left = deadline - std.Io.Clock.awake.now(s.io).toNanoseconds();
            if (left <= 0 or s.crashed) return .{ .done = false, .unpublished = engine_wait + writer_wait };
            s.condWaitFor(&s.done, @min(left, wait_slice_ns));
        }
    }

    /// Whether unpublished points still read `source`'s rows from token `from` on (they pin those rows).
    pub fn pins(s: *Store, source: Source, from: u32) bool {
        s.lockS();
        defer s.unlockS();
        for (s.fifo.items) |j| {
            if (j.state != .open or j.source.id != source.id) continue;
            for (j.new.items[j.next..]) |idx| if (idx * s.cfg.limits.block_tokens + j.point.blocks[idx].tokens > from) return true;
        }
        return false;
    }

    /// pins(.{ .id = ids[i] }, 0) for every id under one lock and one walk of the write fifo: `out[i]` whether an
    /// unpublished point still reads source `ids[i]`'s rows (any unstaged block). `out.len == ids.len`.
    pub fn pinsAny(s: *Store, ids: []const u64, out: []bool) void {
        std.debug.assert(out.len == ids.len);
        @memset(out, false);
        if (ids.len == 0) return;
        s.lockS();
        defer s.unlockS();
        for (s.fifo.items) |j| {
            if (j.state != .open) continue;
            const i = std.mem.indexOfScalar(u64, ids, j.source.id) orelse continue;
            if (out[i]) continue;
            for (j.new.items[j.next..]) |idx| if (idx * s.cfg.limits.block_tokens + j.point.blocks[idx].tokens > 0) {
                out[i] = true;
                break;
            };
        }
    }

    // -- the writer thread ---------------------------------------------------------------------------------------
    /// At a worker's start (lock held): its I/O priority set (a refusal counted), its kernel id recorded.
    fn threadStart(s: *Store) void {
        if (builtin.os.tag == .linux) if (s.cfg.io_priority) |v| {
            const L = std.os.linux;
            const rc = L.syscall3(.ioprio_set, 1, 0, v); // IOPRIO_WHO_PROCESS, 0: this thread
            if (L.errno(rc) != .SUCCESS) s.counters.ioprio_refused += 1;
        };
        s.thread_ids.appendAssumeCapacity(std.Thread.getCurrentId());
    }

    fn writerLoop(s: *Store) void {
        s.lockS();
        defer s.unlockS();
        s.threadStart();
        while (true) {
            while (s.task_head >= s.tasks.items.len and !s.stopping) s.condWait(&s.work);
            if (s.task_head >= s.tasks.items.len) break;
            const t = s.tasks.items[s.task_head];
            s.task_head += 1;
            if (s.task_head == s.tasks.items.len) {
                s.tasks.clearRetainingCapacity();
                s.task_head = 0;
            }
            if (s.stopping) {
                if (s.abandon) { // past closeWithin's deadline nothing more runs
                    s.counters.close_dropped += 1;
                    s.dropTask(t);
                    continue;
                }
                switch (t) {
                    .block, .point => {
                        s.dropTask(t);
                        continue;
                    },
                    else => {},
                }
            }
            if (s.crashed) {
                s.dropTask(t);
                continue;
            }
            s.busy += 1;
            s.runTask(t);
            s.busy -= 1;
            s.done.broadcast(s.io);
        }
        s.writer_exited = true;
        s.done.broadcast(s.io);
    }

    /// Lock held: a task that will not run (a stop, or a simulated crash) gives back what it holds.
    fn dropTask(s: *Store, t: Task) void {
        switch (t) {
            .block => |b| {
                s.slots.give(b.slot);
                b.job.outstanding -= 1;
                if (b.job.state == .open) {
                    b.job.state = .failed;
                    b.job.fail = .crashed;
                }
                s.dropInflight(b.job, b.index);
                if (!s.crashed) s.settle(b.job);
            },
            .point => |j| if (!s.crashed) {
                j.state = .failed;
                j.fail = .crashed;
                s.endJob(j, false);
            },
            .sync => |g| s.sync_done = @max(s.sync_done, g),
            else => {},
        }
        s.done.broadcast(s.io);
    }

    /// Lock held on entry and exit; released around the disk work.
    fn runTask(s: *Store, t: Task) void {
        switch (t) {
            .block => |b| s.writeBlock(b.job, b.index, b.slot),
            .point => |j| s.writePoint(j),
            .shared => |k| s.rewriteShared(k),
            .touch => |x| {
                var nb: [80]u8 = undefined;
                const name = pointName(&nb, &x.key, ".man");
                s.unlockS();
                fsio.touchAt(s.points_fd, name, x.ns);
                s.lockS();
            },
            .unlink_block => |x| {
                var nb: [80]u8 = undefined;
                const name = blockName(&nb, &x.name);
                s.unlockS();
                fsio.unlinkAt(s.blocks_fd, name);
                const free = fsio.diskFree(&s.hooks, s.blocks_fd);
                s.lockS();
                s.disk_free = free;
                s.pending_free -|= x.bytes;
            },
            .unlink_point => |x| {
                s.unlockS();
                s.unlinkPointFiles(x.key);
                const free = fsio.diskFree(&s.hooks, s.blocks_fd);
                s.lockS();
                s.disk_free = free;
                s.pending_free -|= x.bytes;
            },
            .sync => |g| s.sync_done = @max(s.sync_done, g),
            .unmark => |key| s.unlinkMarker(key),
        }
    }

    /// The writer: `key`'s marker unlinked, unless a new restore of the key holds it again. No fsync: an unlink a
    /// crash loses only makes the next open forget a good point. Lock held on entry and exit.
    fn unlinkMarker(s: *Store, key: Digest) void {
        const m = s.marks.getPtr(key) orelse return;
        m.queued = false;
        if (m.restores > 0 or m.writing > 0) return; // (its new restore unmarks it later)
        m.unlinking = true;
        var nb: [80]u8 = undefined;
        const name = pointName(&nb, &key, ".rst");
        s.unlockS();
        fsio.unlinkAt(s.points_fd, name);
        s.lockS();
        const mm = s.marks.getPtr(key).?; // (unlinking kept it in the books)
        mm.unlinking = false;
        mm.on_disk = false;
        if (mm.restores == 0 and mm.writing == 0) _ = s.marks.remove(key);
        s.done.broadcast(s.io); // (a reader waits for the unlink before it writes a new marker)
    }

    fn unlinkPointFiles(s: *Store, key: Digest) void {
        var nb: [80]u8 = undefined;
        fsio.unlinkAt(s.points_fd, pointName(&nb, &key, ".man")); // the manifest first: the point is gone at once
        fsio.unlinkAt(s.points_fd, pointName(&nb, &key, ".st"));
        fsio.unlinkAt(s.points_fd, pointName(&nb, &key, ".ids"));
    }

    fn crash(s: *Store, step: fsio.Step, name: []const u8) bool {
        if (!fsio.crashAt(&s.hooks, step, name)) return false;
        s.lockS();
        s.crashed = true;
        s.unlockS();
        return true;
    }

    /// The writer: one block (CRC-32 and SHA-256 of its bytes), or a hard link to a block on disk with the same
    /// bytes. Lock held on entry and exit.
    fn writeBlock(s: *Store, j: *WriteJob, index: u32, slot: Slot) void {
        if (j.state != .open and !s.stillNamed(j, index)) { // cancelled meanwhile and nobody else names it
            s.slots.give(slot);
            j.outstanding -= 1;
            s.dropInflight(j, index);
            s.settle(j);
            return;
        }
        const b = j.point.blocks[index];
        const data = s.slots.bytes(slot);
        const len: usize = @intCast(b.bytes);
        s.unlockS();
        const crc = format.crc32(data[0..len]);
        const sha = format.sha256(data[0..len]);
        s.lockS();
        var same: ?Digest = null;
        if (s.content.get(sha)) |other| {
            if (s.have.get(other)) |info| if (info.crc == crc and info.bytes == b.bytes) {
                same = other;
            };
        }
        if (same == null and mut(.dedup_without_sha)) {
            var hit = s.have.iterator();
            while (hit.next()) |e| if (e.value_ptr.crc == crc and e.value_ptr.bytes == b.bytes) {
                same = e.key_ptr.*;
                break;
            };
        }
        s.unlockS();
        var nb: [80]u8 = undefined;
        const name = blockName(&nb, &b.name);
        var linked = false;
        var failed: ?WriteFailure = null;
        var crashed = false;
        if (same) |other| {
            var ob: [80]u8 = undefined;
            var tb: [160]u8 = undefined;
            const tmp = s.writer.tmpName(&tb, name);
            if (fsio.linkAt(s.blocks_fd, blockName(&ob, &other), tmp)) {
                if (s.crash(.linked, name)) {
                    crashed = true;
                } else if (fsio.renameAt(&s.hooks, s.blocks_fd, tmp, name)) {
                    linked = true;
                } else |_| fsio.unlinkAt(s.blocks_fd, tmp);
            } else |_| {} // gone meanwhile: written below instead
        }
        if (!linked and !crashed) {
            _ = s.writer.put(s.blocks_fd, name, .{ .aligned = .{ .buf = data, .len = len } }) catch |e| {
                if (e == error.Crashed) {
                    s.lockS();
                    s.crashed = true;
                    s.unlockS();
                    crashed = true;
                } else failed = if (e == error.NoSpace) .no_space else .io;
            };
        }
        var ino: u64 = 0;
        if (failed == null and !crashed) {
            if (fsio.statAt(s.blocks_fd, name)) |st| ino = st.ino else |_| failed = .io;
        }
        const free = fsio.diskFree(&s.hooks, s.blocks_fd);
        s.lockS();
        s.disk_free = free;
        s.slots.give(slot);
        j.outstanding -= 1;
        if (crashed) return;
        if (failed) |why| {
            if (j.state == .open) {
                j.state = .failed;
                j.fail = why;
                s.release(j);
            }
        } else {
            s.addBlock(b.name, crc, sha, ino, b.bytes) catch {};
            if (linked) {
                s.counters.dedup_blocks += 1;
                s.counters.dedup_bytes += b.bytes;
            } else {
                s.counters.written_blocks += 1;
                s.counters.written_bytes += b.bytes;
            }
            // on disk now (in stored_bytes, its inode once): its reservation ends here, not when its point publishes,
            // or room is made for these bytes twice (the cap, the floor) between the two; a link adds no bytes
            if (!mut(.written_still_reserved)) {
                j.size -|= b.bytes;
                s.queued_bytes -|= b.bytes;
            }
        }
        s.dropInflight(j, index);
        s.settle(j);
    }

    /// The writer: a point whose blocks are all handed over. Its blocks durable first (the directory fsynced),
    /// then its ids and small state, the directory again, then the manifest, then the directory: only now is the
    /// point readable. Lock held on entry and exit.
    fn writePoint(s: *Store, j: *WriteJob) void {
        const gpa = s.gpa;
        const p = j.point;
        if (j.state != .publishing) {
            s.settle(j);
            return;
        }
        if (!s.blocksSettled(j)) { // a block another write owns is still on its way
            j.parked = true;
            s.parked.append(gpa, j) catch {};
            return;
        }
        for (p.blocks) |*b| {
            const info = s.have.get(b.name) orelse return s.failPoint(j, .missing_block);
            b.crc = info.crc;
            b.sha = info.sha;
        }
        const shared = p.shared;
        const small = s.slots.bytes(j.small_slot);
        s.unlockS();
        var why: ?WriteFailure = null;
        var crashed = false;
        var nb: [80]u8 = undefined;
        var nb2: [80]u8 = undefined;
        var nb3: [80]u8 = undefined;
        var man_bytes: u64 = 0;
        p.ids_crc = format.crc32(j.ids);
        for (p.sections) |*sec| sec.crc = format.crc32(small[@intCast(sec.offset)..][0..@intCast(sec.len)]);
        p.small_crc = format.crc32(small[0..@intCast(p.small_bytes)]);
        publish: {
            fsio.syncDir(&s.hooks, s.blocks_fd, "blocks") catch {
                why = .io;
                break :publish;
            };
            if (s.crash(.dir_synced, "blocks")) {
                crashed = true;
                break :publish;
            }
            var m = manifestOf(p, shared, s.cfg.limits.block_tokens, s.cfg.owner_layout);
            const raw = m.encode(gpa, s.cfg.compat) catch {
                why = .io;
                break :publish;
            };
            defer gpa.free(raw);
            man_bytes = raw.len;
            const files = [_]struct { name: [:0]const u8, src: fsio.Source }{
                .{ .name = pointName(&nb, &p.key, ".ids"), .src = .{ .parts = &.{j.ids} } },
                .{ .name = pointName(&nb2, &p.key, ".st"), .src = .{ .aligned = .{ .buf = small, .len = @intCast(p.small_bytes) } } },
                .{ .name = pointName(&nb3, &p.key, ".man"), .src = .{ .parts = &.{raw} } },
            };
            const order: [3]usize = if (mut(.manifest_not_last)) .{ 2, 0, 1 } else .{ 0, 1, 2 };
            for (order) |fi| {
                if (fi == 2) { // the manifest last: the files it names durable first
                    fsio.syncDir(&s.hooks, s.points_fd, "points") catch {
                        why = .io;
                        break :publish;
                    };
                    if (s.crash(.dir_synced, "points")) {
                        crashed = true;
                        break :publish;
                    }
                }
                _ = s.writer.put(s.points_fd, files[fi].name, files[fi].src) catch |e| {
                    if (e == error.Crashed) crashed = true else why = if (e == error.NoSpace) .no_space else .io;
                    break :publish;
                };
            }
            fsio.syncDir(&s.hooks, s.points_fd, "points") catch {
                why = .io;
                break :publish;
            };
            if (s.crash(.published, "points")) {
                crashed = true;
                break :publish;
            }
        }
        const free = fsio.diskFree(&s.hooks, s.blocks_fd);
        s.lockS();
        s.disk_free = free;
        if (crashed) {
            s.crashed = true;
            return;
        }
        if (why) |w| {
            s.unlockS();
            s.unlinkPointFiles(p.key);
            s.lockS();
            return s.failPoint(j, w);
        }
        for (p.blocks) |b| if (!s.have.contains(b.name)) {
            s.queue(.{ .unlink_point = .{ .key = p.key, .bytes = 0 } }); // a block it names was dropped while it was written
            return s.failPoint(j, .missing_block);
        };
        p.files = man_bytes + p.ids_bytes + p.small_bytes;
        p.used = s.now();
        s.points.put(gpa, p.key, p) catch {
            s.queue(.{ .unlink_point = .{ .key = p.key, .bytes = 0 } });
            return s.failPoint(j, .io);
        };
        s.ref(p.blocks, 1); // its own reference; the waiting one goes in endJob
        s.adopt(p);
        s.stored_bytes += p.files;
        s.counters.committed += 1;
        s.counters.written_small_bytes += p.small_bytes;
        s.endJob(j, true);
        if (p.shared and !shared and !mut(.written_no_rewrite)) s.queue(.{ .shared = p.key }); // made shared while it was written
        s.trimPath(p);
    }

    fn failPoint(s: *Store, j: *WriteJob, why: WriteFailure) void {
        j.state = .failed;
        j.fail = why;
        s.endJob(j, false);
    }

    fn manifestOf(p: *const Point, shared: bool, block_tokens: u32, layout: format.OwnerLayout) format.Manifest {
        return .{
            .key = p.key,
            .at = p.at,
            .block_tokens = block_tokens,
            .owner_layout = layout,
            .flags = .{ .shared = shared, .has_raw = p.has_raw, .mutable_tail = p.mutable_tail },
            .group = p.group,
            .created_ns = p.created_ns,
            .lineage = p.lineage,
            .origin = p.origin,
            .ids_bytes = p.ids_bytes,
            .ids_crc = p.ids_crc,
            .small_bytes = p.small_bytes,
            .small_crc = p.small_crc,
            .media = p.media,
            .blocks = p.blocks,
            .sections = p.sections,
        };
    }

    /// The writer: point `key`'s manifest again with its shared flag (temporary name, fsync, rename, directory
    /// fsync, like its first write). A point dropped before is left alone; one dropped while this ran loses the
    /// manifest this wrote back (no resurrection). Lock held on entry and exit.
    fn rewriteShared(s: *Store, key: Digest) void {
        const gpa = s.gpa;
        const p = s.points.get(key) orelse return;
        var m = manifestOf(p, p.shared, s.cfg.limits.block_tokens, s.cfg.owner_layout);
        const raw = m.encode(gpa, s.cfg.compat) catch return;
        defer gpa.free(raw);
        var nb: [80]u8 = undefined;
        const name = pointName(&nb, &key, ".man");
        s.unlockS();
        var crashed = false;
        const ok = blk: {
            if (mut(.rewrite_not_put)) {
                const fd = c.openat(s.points_fd, name, .{ .ACCMODE = .WRONLY, .TRUNC = true, .CLOEXEC = true }, @as(c.mode_t, 0));
                if (fd < 0) break :blk false;
                fsio.pwriteAll(&s.hooks, fd, raw, 0) catch {};
                _ = c.close(fd);
            } else _ = s.writer.put(s.points_fd, name, .{ .parts = &.{raw} }) catch |e| {
                crashed = e == error.Crashed;
                break :blk false;
            };
            if (!mut(.rewrite_no_dir_fsync)) fsio.syncDir(&s.hooks, s.points_fd, "points") catch break :blk false;
            if (s.crash(.rewritten, name)) {
                crashed = true;
                break :blk false;
            }
            break :blk true;
        };
        s.lockS();
        if (crashed) {
            s.crashed = true;
            return;
        }
        if (ok) s.counters.shared_rewrites += 1;
        if (ok and s.points.get(key) == p) s.queue(.{ .touch = .{ .key = key, .ns = p.used } }); // its last use, kept
        if (s.points.get(key) != p and !mut(.rewrite_resurrects)) {
            s.unlockS();
            fsio.unlinkAt(s.points_fd, name);
            s.lockS();
        }
    }

    // -- lookups (engine thread) ---------------------------------------------------------------------------------
    /// The longest complete point `q.keyed` extends with the same media before its cut, within `q`'s bounds, or
    /// null. Points being written do not count. The prompt's block chain is hashed outside the lock, only as far as
    /// the longest candidate needs; the comparison under it is digests and a tail of ids.
    pub fn lookup(s: *Store, q: Query) ?Hit {
        const gpa = s.gpa;
        const block = s.cfg.limits.block_tokens;
        var need: usize = 0;
        {
            s.lockS();
            defer s.unlockS();
            s.counters.disk_lookups += 1;
            var it = s.points.valueIterator();
            while (it.next()) |pp| if (eligible(pp.*, q)) {
                need = @max(need, pp.*.chain.len);
            };
        }
        const chain = gpa.alloc(Digest, need) catch return null;
        defer gpa.free(chain);
        var prev: Digest = undefined;
        std.crypto.hash.sha2.Sha256.hash("tf-spill/ids", &prev, .{});
        for (chain, 0..) |*d, i| {
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            h.update(&prev);
            h.update(std.mem.sliceAsBytes(q.keyed[i * block .. (i + 1) * block]));
            h.final(&prev);
            d.* = prev;
        }
        s.lockS();
        defer s.unlockS();
        var best: ?*Point = null;
        var it = s.points.valueIterator();
        while (it.next()) |pp| {
            const p = pp.*;
            if (!eligible(p, q) or p.chain.len > chain.len) continue;
            if (best != null and (p.at < best.?.at or (p.at == best.?.at and p.used <= best.?.used))) continue;
            var same = true;
            for (p.chain, chain[0..p.chain.len]) |a, b| if (!std.mem.eql(u8, &a, &b)) {
                same = false;
                break;
            };
            if (!same) continue;
            const whole = p.chain.len * block;
            if (!std.mem.eql(u32, p.tail, q.keyed[whole..p.at])) continue;
            if (!mediaMatch(p, q.media)) continue;
            if (!s.complete_(p)) continue;
            best = p;
        }
        const b = best orelse return null;
        s.counters.disk_hits += 1;
        return .{ .key = b.key, .at = b.at, .shared = b.shared, .group = b.group, .part = b.lineage.part };
    }

    fn eligible(p: *const Point, q: Query) bool {
        if (p.at <= q.longer_than or p.at > q.max_at or p.at > q.keyed.len) return false;
        return q.grid == 0 or p.at % q.grid == 0;
    }

    fn mediaMatch(p: *const Point, media: []const Media) bool {
        var i: usize = 0;
        for (media) |m| {
            if (m.start >= p.at) continue;
            if (i >= p.media.len or !p.media[i].eql(m)) return false;
            i += 1;
        }
        return i == p.media.len;
    }

    fn complete_(s: *Store, p: *const Point) bool {
        for (p.blocks) |b| if (!s.have.contains(b.name)) return false;
        return true;
    }

    /// Whether this rank holds `key` complete (manifest, ids, small state, every block): what a rank answers to
    /// the all-rank presence check.
    /// Rank 0 asks rank_wire's `has` with the point digest in the ordered stream; every rank answers with this, and
    /// nobody reads unless all hold it.
    pub fn presence(s: *Store, key: Digest) bool {
        s.lockS();
        defer s.unlockS();
        const p = s.points.get(key) orelse return false;
        return s.complete_(p);
    }

    /// Damages point `key` on disk: one byte of each of its block files is flipped (the fault drill's corrupt fault), so a read of it fails
    /// its check. The store's own records stay as written. Returns how many block files were damaged.
    pub fn damage(s: *Store, key: Digest) usize {
        s.lockS();
        defer s.unlockS();
        const p = s.points.get(key) orelse return 0;
        var n: usize = 0;
        for (p.blocks) |b| {
            var nb: [80]u8 = undefined;
            fsio.flipByteAt(&s.hooks, s.blocks_fd, blockName(&nb, &b.name), 7) catch continue;
            n += 1;
        }
        return n;
    }

    /// A digest of the bytes point `key` holds on this rank: its length, every block's SHA-256 and size, and the small
    /// state's size and CRC. Two ranks that wrote the same rows and the same small state give the same digest, whatever
    /// the point's lineage names. Null when the point is not whole here.
    pub fn pointDigest(s: *Store, key: Digest) ?Digest {
        s.lockS();
        defer s.unlockS();
        const p = s.points.get(key) orelse return null;
        if (p.removed or !s.complete_(p)) return null;
        return digestOf(p);
    }

    fn digestOf(p: *const Point) Digest {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(std.mem.asBytes(&std.mem.nativeToLittle(u32, p.at)));
        for (p.blocks) |b| {
            h.update(&b.sha);
            h.update(std.mem.asBytes(&std.mem.nativeToLittle(u64, b.bytes)));
        }
        h.update(std.mem.asBytes(&std.mem.nativeToLittle(u64, p.small_bytes)));
        h.update(std.mem.asBytes(&std.mem.nativeToLittle(u32, p.small_crc)));
        var d: Digest = undefined;
        h.final(&d);
        return d;
    }

    /// Every whole point this rank holds with its digest (pointDigest), appended to `out`.
    pub fn pointList(s: *Store, gpa: Allocator, out: *std.ArrayList(PointEntry)) !void {
        s.lockS();
        defer s.unlockS();
        var it = s.points.iterator();
        while (it.next()) |e| {
            const p = e.value_ptr.*;
            if (p.removed or !s.complete_(p)) continue;
            try out.append(gpa, .{ .key = p.key, .digest = digestOf(p) });
        }
    }

    /// The part salt of the point `key` this rank holds (null inside: a part-less point), or null when it holds none.
    /// An owner-sharded rank checks it against the part a restore names: the same ids kept from another part are
    /// other bytes here.
    pub fn partOf(s: *Store, key: Digest) ?struct { part: ?u32 } {
        s.lockS();
        defer s.unlockS();
        const p = s.points.get(key) orelse return null;
        return .{ .part = p.lineage.part };
    }

    /// `key` was used now (its manifest's time follows, on the writer).
    pub fn touch(s: *Store, key: Digest) void {
        s.lockS();
        defer s.unlockS();
        const p = s.points.get(key) orelse return;
        p.used = s.now();
        s.queue(.{ .touch = .{ .key = key, .ns = p.used } });
    }

    /// Makes a stored point a shared prefix, durably (its manifest rewritten on the writer).
    pub fn markShared(s: *Store, key: Digest) void {
        s.lockS();
        defer s.unlockS();
        if (s.points.get(key)) |p| {
            if (!p.shared) {
                p.shared = true;
                s.counters.shared_marked += 1;
                s.queue(.{ .shared = key });
            }
        } else if (s.waiting.get(key)) |j| j.point.shared = true;
    }

    // -- reads (engine thread, readers) --------------------------------------------------------------------------
    pub const BeginError = error{ Missing, NoSlot, TooMany, OutOfMemory };

    /// Starts reading point `key`: its small state into a restore slot and its blocks, in order, into read slots as
    /// they come free (several restores share the slots fairly). The point is pinned against eviction until
    /// `endRead`. NoSlot / TooMany: defer the restore.
    pub fn beginRead(s: *Store, key: Digest) BeginError!ReadId {
        const gpa = s.gpa;
        s.lockS();
        defer s.unlockS();
        const p = s.points.get(key) orelse return error.Missing;
        if (!s.complete_(p)) return error.Missing;
        if (s.liveReads() >= s.cfg.max_reads) {
            s.counters.read_deferred += 1;
            return error.TooMany;
        }
        const id = s.next_id;
        const small_slot = s.slots.take(.restore, id) orelse {
            s.counters.read_deferred += 1;
            return error.NoSlot;
        };
        errdefer s.slots.give(small_slot);
        const r = try gpa.create(ReadJob);
        errdefer gpa.destroy(r);
        const n: u32 = @intCast(p.blocks.len);
        r.* = .{ .id = id, .point = p, .total = n, .state = try gpa.alloc(ReadJob.BState, n), .slot = undefined, .small_slot = small_slot, .t0 = std.Io.Clock.awake.now(s.io).toNanoseconds() };
        errdefer gpa.free(r.state);
        r.slot = try gpa.alloc(Slot, n);
        errdefer gpa.free(r.slot);
        @memset(r.state, .idle);
        try s.reads.append(gpa, r);
        errdefer _ = s.reads.pop();
        try s.marks.ensureUnusedCapacity(gpa, 1);
        const gop = try s.reading.getOrPut(gpa, key);
        gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
        // the restore holds the key's quarantine marker (its reader writes it before the small state)
        const mg = s.marks.getOrPutAssumeCapacity(key);
        if (!mg.found_existing) mg.value_ptr.* = .{};
        mg.value_ptr.restores += 1;
        p.pins += 1;
        s.next_id += 1;
        p.used = s.now();
        s.queue(.{ .touch = .{ .key = key, .ns = p.used } });
        r.disk += 1;
        s.rtasks.append(gpa, .{ .job = r, .index = null, .slot = small_slot }) catch {
            r.disk -= 1;
            r.small = .failed;
            r.err = .io;
        };
        s.assign();
        s.rwork.broadcast(s.io);
        return id;
    }

    fn readJob(s: *Store, id: ReadId) ?*ReadJob {
        for (s.reads.items) |r| if (r.id == id and !r.ended) return r;
        return null;
    }

    /// Whether beginRead would refuse for capacity now (TooMany: max_reads in flight; NoSlot: no restore slot free),
    /// changing nothing: a tier asks before the prompt cache makes room for a restore (Tier.VTable.full).
    pub fn readsFull(s: *Store) bool {
        s.lockS();
        defer s.unlockS();
        return s.liveReads() >= s.cfg.max_reads or s.slots.available(.restore) == 0;
    }

    fn liveReads(s: *Store) usize {
        var n: usize = 0;
        for (s.reads.items) |r| n += @intFromBool(!r.ended);
        return n;
    }

    /// Lock held: free read slots go to the next blocks of the restores in flight, each to the restore holding the
    /// fewest slots (ties in turn), so one large restore cannot keep another waiting for the whole ring.
    fn assign(s: *Store) void {
        const n = s.reads.items.len;
        while (s.slots.available(.read) > 0 and n > 0) {
            var pick: ?*ReadJob = null;
            for (0..n) |k| {
                const r = s.reads.items[(s.read_turn + k) % n];
                if (r.ended or r.err != null or r.next_assign >= r.total) continue;
                if (pick == null or r.held < pick.?.held) pick = r;
            }
            const r = pick orelse break;
            s.read_turn +%= 1;
            const slot = s.slots.take(.read, r.id) orelse break;
            const i = r.next_assign;
            r.next_assign += 1;
            r.state[i] = .reading;
            r.slot[i] = slot;
            r.disk += 1;
            r.held += 1;
            s.rtasks.append(s.gpa, .{ .job = r, .index = i, .slot = slot }) catch {
                r.disk -= 1;
                r.held -= 1;
                r.state[i] = .failed;
                r.err = .io;
                s.slots.give(slot);
                continue;
            };
        }
        s.rwork.broadcast(s.io);
    }

    fn giveRead(s: *Store, r: *ReadJob, slot: Slot) void {
        s.slots.give(slot);
        r.held -= 1;
    }

    fn readerLoop(s: *Store) void {
        s.lockS();
        defer s.unlockS();
        s.threadStart();
        defer {
            s.readers_exited += 1;
            s.done.broadcast(s.io);
        }
        while (true) {
            while (s.rtask_head >= s.rtasks.items.len and !s.stopping) s.condWait(&s.rwork);
            if (s.stopping) break;
            const t = s.rtasks.items[s.rtask_head];
            s.rtask_head += 1;
            if (s.rtask_head == s.rtasks.items.len) {
                s.rtasks.clearRetainingCapacity();
                s.rtask_head = 0;
            }
            const r = t.job;
            if (r.ended or (r.err != null and t.index != null)) {
                if (t.index) |i| {
                    s.giveRead(r, t.slot);
                    r.state[i] = .failed;
                } else if (r.ended) s.slots.give(t.slot); // the small slot: its reader gives it back once ended
                r.disk -= 1;
                s.maybeFreeRead(r);
                s.assign();
                continue;
            }
            if (t.index) |i| {
                const b = r.point.blocks[i];
                const buf = s.slots.bytes(t.slot);
                s.unlockS();
                var times: BlockTimes = .{};
                const got = s.readBlock(b, buf, &times);
                s.lockS();
                s.noteBlock(times);
                r.disk -= 1;
                if (r.ended) {
                    s.giveRead(r, t.slot);
                    r.state[i] = .failed;
                } else if (got) |_| {
                    r.state[i] = .ready;
                    r.read += 1;
                    if (r.read == r.total) s.counters.read_wall.note(@intCast(@max(0, monoNow(s.io) - r.t0)));
                } else |e| {
                    r.state[i] = .failed;
                    s.giveRead(r, t.slot); // (never handed: its slot comes back now, or the job could never end)
                    s.readFailed(r, b.name, e);
                }
            } else {
                const buf = s.slots.bytes(t.slot);
                const p = r.point;
                // the quarantine marker first, durable before the small state can read as ready
                var marked = false;
                var small: BlockReadError!void = error.Io;
                if (!r.ended) {
                    marked = mut(.marker_skipped) or s.mark(p.key);
                    if (marked and !r.ended) {
                        s.unlockS();
                        small = s.readSmall(p, buf);
                        s.lockS();
                    }
                }
                r.disk -= 1;
                if (r.ended) {
                    s.slots.give(t.slot);
                } else if (!marked) {
                    r.small = .failed;
                    if (r.err == null) r.err = .io; // (not damage: the point stays)
                } else if (small) |_| r.small = .ready else |e| {
                    r.small = .failed;
                    switch (e) {
                        error.Io => if (r.err == null) {
                            r.err = .io; // (not damage: the point stays)
                        },
                        error.Missing => {
                            s.counters.missing_errors += 1;
                            if (r.err == null) r.err = .missing;
                        },
                        error.WrongSize => {
                            s.counters.size_errors += 1;
                            if (r.err == null) r.err = .small;
                        },
                        error.Damaged => {
                            s.counters.crc_errors += 1;
                            if (r.err == null) r.err = .small;
                        },
                    }
                }
            }
            if (r.ended) {
                s.maybeFreeRead(r);
                s.assign();
            }
            s.done.broadcast(s.io);
        }
    }

    /// A reader: `key`'s quarantine marker, points/<key>.rst (the key's bytes), written and durable (fsync, rename, the
    /// points folder fsynced) before the restore reads its small state. Lock held on entry and exit, released around
    /// the disk work. False when it could not be made (counted in marker_failed: the restore fails as an I/O error,
    /// never as damage), or when the key's restores ended meanwhile.
    fn mark(s: *Store, key: Digest) bool {
        while (true) { // the writer may be unlinking this key's previous marker: after it (else it could take this one)
            const m = s.marks.getPtr(key) orelse return false;
            if (!m.unlinking) break;
            s.condWait(&s.done);
        }
        const m = s.marks.getPtr(key).?;
        m.writing += 1;
        m.on_disk = true;
        var nb: [80]u8 = undefined;
        const name = pointName(&nb, &key, ".rst");
        s.unlockS();
        var ok = true;
        if (mut(.marker_not_durable)) {
            const fd = c.openat(s.points_fd, name, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
            if (fd >= 0) {
                _ = c.pwrite(fd, &key, key.len, 0);
                _ = c.close(fd);
            } else ok = false;
        } else {
            _ = s.writer.put(s.points_fd, name, .{ .parts = &.{&key} }) catch {
                ok = false;
            };
            if (ok) fsio.syncDir(&s.hooks, s.points_fd, "points") catch {
                ok = false;
            };
        }
        if (ok) _ = fsio.crashAt(&s.hooks, .marked, name);
        s.lockS();
        const mm = s.marks.getPtr(key).?; // (writing > 0 kept it in the books)
        mm.writing -= 1;
        if (!ok) s.counters.marker_failed += 1;
        s.unmarkIfDone(key); // (its restores may have ended meanwhile)
        return ok;
    }

    /// Lock held: `key`'s marker is no restore's any more and no reader writes it: the writer unlinks it (once), or,
    /// with no file made, it leaves the books.
    fn unmarkIfDone(s: *Store, key: Digest) void {
        const m = s.marks.getPtr(key) orelse return;
        if (m.restores > 0 or m.writing > 0) return;
        if (!m.on_disk) {
            _ = s.marks.remove(key);
            return;
        }
        if (m.queued or m.unlinking) return;
        m.queued = true;
        s.queue(.{ .unmark = key });
    }

    /// Lock held: a restore of `key` lets its marker go (an abort, a cancel, or restored(key) after a commit).
    fn unmark(s: *Store, key: Digest) void {
        const m = s.marks.getPtr(key) orelse return;
        if (m.restores == 0) return;
        m.restores -= 1;
        s.unmarkIfDone(key);
    }

    /// The tier's word that a committed restore of `key` is in use (its first round ran on the restored state, or the
    /// state was dropped unused): the restore's quarantine marker goes (unlinked on the writer). One call for every
    /// commit; until it, a crash makes the next open forget the point (it may be what crashed the engine).
    pub fn restored(s: *Store, key: Digest) void {
        s.lockS();
        defer s.unlockS();
        s.unmark(key);
    }

    /// A file of a point read back: gone (Missing), its bytes damaged (Damaged: short, CRC), of the wrong size
    /// (WrongSize: damage, counted in size_errors), or a read that failed for a reason the file is not to blame for (Io:
    /// EIO, EMFILE, a permission: the point stays).
    const BlockReadError = error{ Missing, Damaged, WrongSize, Io };

    /// One block read's stages, each in nanoseconds; null: the read ended before it.
    const BlockTimes = struct { open: ?u64 = null, read: ?u64 = null, crc: ?u64 = null, close: ?u64 = null };

    /// Lock held: a block read's stages into the counters.
    fn noteBlock(s: *Store, t: BlockTimes) void {
        if (t.open) |ns| s.counters.block_open.note(ns);
        if (t.read) |ns| s.counters.block_read.note(ns);
        if (t.crc) |ns| s.counters.block_crc.note(ns);
        if (t.close) |ns| s.counters.block_close.note(ns);
    }

    /// A reader: block `b` into `buf` (O_DIRECT where it works), its size and CRC-32 checked, each stage timed into
    /// `t`. Missing: its name is gone (links to its bytes are not in question). Damaged: wrong size, short, or CRC
    /// mismatch.
    fn readBlock(s: *Store, b: format.BlockRef, buf: slots_mod.Buffer, t: *BlockTimes) BlockReadError!void {
        var nb: [80]u8 = undefined;
        var t0 = monoNow(s.io);
        var f = s.writer.openRead(s.blocks_fd, blockName(&nb, &b.name)) catch |e| return if (e == error.Missing) error.Missing else error.Io;
        defer {
            const c0 = monoNow(s.io);
            f.close();
            t.close = elapsed(s.io, c0);
        }
        const st = fsio.statFd(f.fd) catch return error.Io;
        t.open = elapsed(s.io, t0);
        if (st.size != b.bytes) return error.WrongSize;
        const len: usize = @intCast(b.bytes);
        t0 = monoNow(s.io);
        const got = f.read(buf, 0, len) catch |e| return if (e == error.EndOfFile) error.Damaged else error.Io;
        std.debug.assert(got.ptr == buf.ptr);
        if (!f.direct) fsio.dropPages(&s.hooks, f.fd, "block");
        t.read = elapsed(s.io, t0);
        t0 = monoNow(s.io);
        const crc = format.crc32(buf[0..len]);
        t.crc = elapsed(s.io, t0);
        if (crc != b.crc) return error.Damaged;
    }

    fn elapsed(io: std.Io, t0: i96) u64 {
        return @intCast(@max(0, monoNow(io) - t0));
    }

    /// A point's small state read and checked into `buf`, its failure classed as readBlock's (an I/O error is not
    /// damage).
    fn readSmall(s: *Store, p: *const Point, buf: slots_mod.Buffer) BlockReadError!void {
        var nb: [80]u8 = undefined;
        var f = s.writer.openRead(s.points_fd, pointName(&nb, &p.key, ".st")) catch |e| return if (e == error.Missing) error.Missing else error.Io;
        defer f.close();
        const st = fsio.statFd(f.fd) catch return error.Io;
        if (st.size != p.small_bytes) return error.WrongSize;
        const len: usize = @intCast(p.small_bytes);
        if (len == 0) return if (p.small_crc == format.crc32(&.{})) {} else error.Damaged;
        _ = f.read(buf, 0, len) catch |e| return if (e == error.EndOfFile) error.Damaged else error.Io;
        if (!f.direct) fsio.dropPages(&s.hooks, f.fd, "small");
        if (format.crc32(buf[0..len]) != p.small_crc) return error.Damaged;
        for (p.sections) |sec| if (format.crc32(buf[@intCast(sec.offset)..][0..@intCast(sec.len)]) != sec.crc) return error.Damaged;
    }

    fn readFailed(s: *Store, r: *ReadJob, name: Digest, e0: BlockReadError) void {
        const e = if (mut(.lost_is_damage) and e0 == error.Missing) error.Damaged else e0;
        switch (e) {
            error.Missing => {
                r.lost.append(s.gpa, name) catch {};
                s.counters.missing_errors += 1;
                if (r.err == null) r.err = .missing;
            },
            error.Damaged, error.WrongSize => {
                r.bad.append(s.gpa, name) catch {};
                if (e == error.WrongSize) s.counters.size_errors += 1 else s.counters.crc_errors += 1;
                if (r.err == null) r.err = .damaged;
            },
            error.Io => if (r.err == null) {
                r.err = .io;
            },
        }
    }

    /// Between rounds: `id`'s next blocks read and checked, in order, before block `upto` (null: all), at most
    /// `out.len`. The engine copies each into the destination and calls `readRelease` once that copy completed
    /// (a slot goes back to the readers then). `.waiting`: the next block is not read yet.
    pub fn readReady(s: *Store, id: ReadId, upto: ?u32, out: []ReadyBlock) struct { n: usize, status: ReadStatus } {
        s.lockS();
        defer s.unlockS();
        const r = s.readJob(id) orelse return .{ .n = 0, .status = .failed };
        const stop = if (mut(.loadpart_past_upto)) r.total else @min(upto orelse r.total, r.total);
        var n: usize = 0;
        while (r.next_hand < stop and n < out.len) {
            const i = r.next_hand;
            switch (r.state[i]) {
                .ready => {},
                .failed => return .{ .n = n, .status = .failed },
                else => break,
            }
            r.state[i] = .handed;
            r.next_hand += 1;
            const b = r.point.blocks[i];
            out[n] = .{ .index = i, .first = i * s.cfg.limits.block_tokens, .tokens = b.tokens, .data = s.slots.bytes(r.slot[i])[0..@intCast(b.bytes)], .slot = r.slot[i] };
            s.counters.loaded_bytes += b.bytes;
            n += 1;
        }
        if (r.err != null and r.next_hand < stop and r.state[r.next_hand] != .ready) return .{ .n = n, .status = .failed };
        if (r.next_hand >= r.total) return .{ .n = n, .status = .done };
        if (r.next_hand >= stop) return .{ .n = n, .status = .done };
        return .{ .n = n, .status = if (r.state[r.next_hand] == .ready) .more else .waiting };
    }

    /// The engine's copy out of `slot` completed: it goes back to the readers.
    pub fn readRelease(s: *Store, slot: Slot) void {
        s.lockS();
        defer s.unlockS();
        const holder = s.slots.holder(slot);
        for (s.reads.items) |r| if (r.id == holder) {
            r.held -= 1;
            break;
        };
        s.slots.give(slot);
        if (holder != 0) for (s.reads.items) |r| if (r.id == holder) {
            s.maybeFreeRead(r);
            break;
        };
        s.assign();
    }

    /// The engine's way to recycle slots while it waits: it polls its copies' events and calls `readRelease`.
    pub const Recycler = struct {
        ctx: ?*anyopaque = null,
        recycle: ?*const fn (ctx: ?*anyopaque) void = null,
    };

    /// The follower copies up to rank 0's copied-block boundary with this, its slots recycled by its events.
    /// A follower (or an idle engine) waiting for block `index` of `id`, at most `timeout_ns`: it gives back the read
    /// slots its own copies held before each wait (PR 124: a rank whose peer failed early, with more blocks left
    /// than read slots, waited for ever on a block the reader had no slot for). True once it is read, false when
    /// the read failed, was cancelled, or the time ran out.
    pub fn readAwait(s: *Store, id: ReadId, index: u32, timeout_ns: u64, rc: Recycler) bool {
        const deadline = std.Io.Clock.awake.now(s.io).toNanoseconds() + timeout_ns;
        while (true) {
            if (rc.recycle) |f| if (!mut(.hang_no_recycle_in_wait)) f(rc.ctx);
            s.lockS();
            const r = s.readJob(id) orelse {
                s.unlockS();
                return false;
            };
            if (index >= r.total) {
                s.unlockS();
                return false;
            }
            const st = r.state[index];
            if (st == .ready or st == .handed) {
                s.unlockS();
                return true;
            }
            if (st == .failed or r.ended or (r.err != null and st == .idle)) {
                s.unlockS();
                return false;
            }
            const left = deadline - std.Io.Clock.awake.now(s.io).toNanoseconds();
            if (left <= 0) {
                s.unlockS();
                return false;
            }
            // A constant, not a setting: the longest this wait sleeps before it recycles its copies' slots again (a
            // reader wakes it sooner on `done`), so it bounds how long a slot whose copy finished stays out of the ring
            // while a follower waits.
            s.condWaitFor(&s.done, @min(left, 20 * std.time.ns_per_ms));
            s.unlockS();
        }
    }

    /// The small state once read and checked (null: not yet, or failed: see `readFailure`).
    pub fn readSmallView(s: *Store, id: ReadId) ?SmallView {
        s.lockS();
        defer s.unlockS();
        const r = s.readJob(id) orelse return null;
        if (r.small != .ready) return null;
        return .{ .bytes = s.slots.bytes(r.small_slot)[0..@intCast(r.point.small_bytes)], .sections = r.point.sections };
    }

    pub const Failure = struct { err: ?ReadError, bad: []const Digest, lost: []const Digest };

    /// Why `id` failed: the blocks whose bytes are damaged (`bad`) or whose names are gone (`lost`). Valid until
    /// `endRead`.
    pub fn readFailure(s: *Store, id: ReadId) Failure {
        s.lockS();
        defer s.unlockS();
        const r = s.readJob(id) orelse return .{ .err = .cancelled, .bad = &.{}, .lost = &.{} };
        return .{ .err = r.err, .bad = r.bad.items, .lost = r.lost.items };
    }

    pub const EndError = error{ Unknown, Incomplete, Failed, OutOfMemory };

    /// How a restore ends: every rank agreed and copied it (`commit`), it failed on some rank (`abort`: counted as
    /// a failed restore), or its request went away (`cancel`).
    pub const Outcome = enum { commit, abort, cancel };

    /// `.commit` only after every rank reported validation for the transaction (rank 0 publishes the entry then); a
    /// link or rank failure ends the pair instead (no local fallback).
    /// Ends restore `id`. `.commit`: every block was handed and copied and every rank agreed; the restore counts and
    /// its point's lineage comes back (gpa-owned); an incomplete or failed one refuses to commit and stays open for
    /// the caller to abort. Otherwise (`.abort`, `.cancel`) its reads stop: slots of blocks
    /// still being read come back once their reads end. The engine must have released every handed slot's copy
    /// (or will, through `readRelease`) and finished with the small state before calling this.
    pub fn endRead(s: *Store, id: ReadId, outcome: Outcome) EndError!?ReadDone {
        s.lockS();
        defer s.unlockS();
        const r = s.readJob(id) orelse return error.Unknown;
        var done: ?ReadDone = null;
        var result: EndError!?ReadDone = null;
        if (outcome == .commit) {
            if (r.err != null) {
                result = error.Failed;
            } else if (r.next_hand < r.total or r.small != .ready) {
                result = error.Incomplete;
            } else {
                const p = r.point;
                const lineage = p.lineage.clone(s.gpa) catch return error.OutOfMemory;
                done = .{ .key = p.key, .at = p.at, .shared = p.shared, .group = p.group, .has_raw = p.has_raw, .lineage = lineage, .origin = p.origin };
                s.counters.loaded += 1;
                s.counters.loaded_tokens += p.at;
                result = done;
            }
            if (done == null) return result; // nothing ended: the caller aborts it next
        } else if (outcome == .abort) s.counters.load_failed += 1;
        if (outcome != .commit) s.unmark(r.point.key); // (a commit keeps it until restored(key))
        r.ended = true;
        if (r.small != .queued) s.slots.give(r.small_slot); // (else its reader gives it back)
        if (s.reading.getPtr(r.point.key)) |v| {
            v.* -= 1;
            if (v.* == 0) _ = s.reading.remove(r.point.key);
        }
        // blocks read but never handed: their slots back now (a block still reading gives its own back)
        for (r.state, 0..) |st, i| if (st == .ready) {
            s.giveRead(r, r.slot[i]);
            r.state[i] = .failed;
        };
        s.maybeFreeRead(r);
        s.assign();
        return done;
    }

    fn maybeFreeRead(s: *Store, r: *ReadJob) void {
        if (!r.ended or r.disk != 0 or r.held != 0) return;
        for (s.reads.items, 0..) |x, k| if (x == r) {
            _ = s.reads.orderedRemove(k);
            break;
        };
        const p = r.point;
        p.pins -= 1;
        if (p.pins == 0 and p.removed) p.deinit(s.gpa);
        freeRead(s.gpa, r);
    }

    /// Every rank, when a restore of `key` failed on some rank (each rank passes its own `readFailure` lists: the
    /// abort carries only that the point is forgotten, spill_tier rxEnd): the point goes, and so do the blocks in `bad`
    /// (damaged bytes: every name linked to their inode too) and in `lost` (a name whose file is gone: that name
    /// only), with every point naming one of them.
    pub fn forget(s: *Store, key: Digest, bad: []const Digest, lost: []const Digest) void {
        const gpa = s.gpa;
        s.lockS();
        defer s.unlockS();
        var gone: std.AutoHashMapUnmanaged(Digest, void) = .empty;
        defer gone.deinit(gpa);
        for (bad) |b| {
            gone.put(gpa, b, {}) catch {};
            if (mut(.damage_not_widened)) continue;
            if (s.have.get(b)) |info| if (s.inodes.get(info.ino)) |ino| {
                for (ino.names.items) |nm| gone.put(gpa, nm, {}) catch {};
            };
        }
        if (!mut(.lost_names_kept)) for (lost) |b| gone.put(gpa, b, {}) catch {};
        var drop: std.ArrayList(Digest) = .empty;
        defer drop.deinit(gpa);
        drop.append(gpa, key) catch {};
        var it = s.points.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr, &key)) continue;
            for (e.value_ptr.*.blocks) |b| if (gone.contains(b.name)) {
                drop.append(gpa, e.key_ptr.*) catch {};
                break;
            };
        }
        // every one goes, read or not: a point being read is unlisted now and freed when its last read ends (those
        // reads fail on the names dropped below); left listed, it would stay incomplete, hidden from lookups, holding
        // its files, and claim its key as stored for ever
        for (drop.items) |k| {
            if (mut(.forget_skips_reading) and s.reading.contains(k) and !std.mem.eql(u8, &k, &key)) continue;
            s.remove(k, "dropped");
        }
        var git = gone.keyIterator();
        while (git.next()) |b| s.dropBlock(b.*);
    }

    /// A read-side fairness helper (0110's step between rounds): one time budget shared by every restore in flight,
    /// each pause starting with the next restore in turn. `copy` takes one ready block (the engine's H2D copy and
    /// event) and returns false to stop. Blocks copied.
    pub fn stepReads(s: *Store, ids: []const ReadId, turn: *usize, budget_ns: u64, ctx: ?*anyopaque, copy: *const fn (ctx: ?*anyopaque, id: ReadId, b: ReadyBlock) bool) usize {
        if (ids.len == 0) return 0;
        const deadline = std.Io.Clock.awake.now(s.io).toNanoseconds() + budget_ns;
        const k = if (mut(.budget_same_first)) 0 else turn.* % ids.len;
        turn.* +%= 1;
        var n: usize = 0;
        for (0..ids.len) |x| {
            const id = ids[(k + x) % ids.len];
            const own = std.Io.Clock.awake.now(s.io).toNanoseconds() + budget_ns;
            while (std.Io.Clock.awake.now(s.io).toNanoseconds() < (if (mut(.budget_each_read)) own else deadline)) {
                var one: [1]ReadyBlock = undefined;
                const got = s.readReady(id, null, &one);
                if (got.n == 0) break;
                n += 1;
                if (!copy(ctx, id, one[0])) return n;
            }
        }
        return n;
    }

    // -- status ----------------------------------------------------------------------------------------------------
    pub fn counts(s: *Store) Counters {
        s.lockS();
        defer s.unlockS();
        var x = s.counters;
        x.points = s.points.count();
        x.blocks = s.have.count();
        x.stored_bytes = s.stored_bytes;
        x.queued_bytes = s.queued_bytes;
        x.pending_jobs = s.jobs_by_id.count();
        x.reads = s.reads.items.len;
        x.staging_bytes = s.slots.heldBytes();
        x.staging_peak_bytes = s.slots.peakBytes();
        x.disk_free_bytes = s.disk_free;
        x.direct_refused = s.writer.direct_refused.load(.monotonic);
        x.owner_failed = s.writer.owner_failed.load(.monotonic);
        return x;
    }

    /// Reads the file system's free bytes now (the writer refreshes them after every write and deletion).
    pub fn refreshDiskFree(s: *Store) void {
        const free = fsio.diskFree(&s.hooks, s.blocks_fd);
        s.lockS();
        defer s.unlockS();
        s.disk_free = free;
    }

    /// Whether a test's simulated crash stopped the writer.
    pub fn isCrashed(s: *Store) bool {
        s.lockS();
        defer s.unlockS();
        return s.crashed;
    }

    pub const Summary = struct {
        conversations: u64 = 0,
        shared_states: u64 = 0,
        forks: u64 = 0,
        turns_kept_max: u64 = 0,
        conversation_bytes_max: u64 = 0,
        conversation_tokens_max: u64 = 0,
    };

    /// What the store holds by conversation (for /health): conversations, shared states, abandoned forks, the most
    /// states on one active path, the largest conversation's own bytes and tokens.
    pub fn summary(s: *Store) Summary {
        const gpa = s.gpa;
        s.lockS();
        defer s.unlockS();
        var out: Summary = .{};
        var t = s.tree() catch return out;
        defer t.deinit(gpa);
        var v = s.victims(null) catch return out;
        defer v.deinit(gpa);
        for (v.items) |x| switch (x.why) {
            .pruned_forks => out.forks += 1,
            .pruned_shared => out.shared_states += 1,
            else => {},
        };
        var convs: std.AutoHashMapUnmanaged(usize, void) = .empty;
        defer convs.deinit(gpa);
        for (t.conv) |cv| if (cv) |x| {
            convs.put(gpa, x, {}) catch {};
        };
        out.conversations = convs.count();
        var cit = convs.keyIterator();
        var keys: std.ArrayList(Digest) = .empty;
        defer keys.deinit(gpa);
        while (cit.next()) |cv| {
            keys.clearRetainingCapacity();
            var tokens: u64 = 0;
            for (t.keys, 0..) |k, j| if (t.conv[j] == cv.*) {
                keys.append(gpa, k) catch {};
                tokens = @max(tokens, s.points.get(k).?.at);
            };
            out.conversation_bytes_max = @max(out.conversation_bytes_max, s.convBytes(keys.items));
            out.conversation_tokens_max = @max(out.conversation_tokens_max, tokens);
            // its active path's length (from its newest leaf)
            var newest: ?Digest = null;
            var used: i96 = std.math.minInt(i96);
            for (keys.items) |k| {
                const p = s.points.get(k).?;
                if (p.used > used) {
                    used = p.used;
                    newest = k;
                }
            }
            var len: u64 = 0;
            var cur = newest;
            while (cur) |k| {
                const p = s.points.get(k) orelse break;
                if (p.shared) break;
                len += 1;
                cur = s.parent.get(k) orelse null;
            }
            out.turns_kept_max = @max(out.turns_kept_max, len);
        }
        return out;
    }
};
