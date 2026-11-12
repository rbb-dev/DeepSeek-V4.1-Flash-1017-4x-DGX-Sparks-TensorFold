//! Port of wojo's GLM 0088 (PR #78) and 0110 (PR #124; BadAd84's block layout, lineage and fsync order) spill tier:
//! the native spill v2 on-disk format. Fixed-schema binary records only (no JSON, no pickles, no process pointers):
//! every count is checked against limits derived from the model and context, and every length against the file,
//! before anything is allocated. A foreign or damaged record is a cache miss, never an error that stops the engine.
//!
//! Layout of one rank's store: `<root>/native-spill-<compat:16 hex>/rank<N>/{owner.lock, blocks/, points/}`.
//! - `blocks/<name>.blk`: one block's packed per-rank planes, exactly `bytes` long. Its name is a digest chained over
//!   the canonical key ids up to the block's end and salted with the computation lineage (`blockNames`), so a later
//!   turn or a fork names the blocks its earlier turns wrote. Equal bytes under another name are a hard link.
//! - `points/<key>.ids` (keyed ids, then raw ids when they differ), `points/<key>.st` (named small-state sections),
//!   then `points/<key>.man` (this file's `Manifest`), written last: a point with a manifest is complete.
//! - `points/<key>.rst`: a restore's quarantine marker (the key's 32 bytes), durable before the restore's small state
//!   is read and removed once the restored state served its first round (or the restore ended without it): an open
//!   that finds one forgets that point, which may be what stopped the engine.
//! Every file goes to a unique temporary name starting with '.', is fsynced and renamed; a restart deletes the rest.
const std = @import("std");
const builtin = @import("builtin");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Crc32 = @import("crc.zig").Crc32; // the Arm instructions where the CPU has them
const Allocator = std.mem.Allocator;

comptime {
    // ids and fields are hashed and stored as their little-endian bytes
    std.debug.assert(builtin.cpu.arch.endian() == .little);
}

pub const Digest = [32]u8;
pub const zero_digest: Digest = @splat(0);

/// The native spill format's manifest version (bumped on any byte change below). Version 3 (pictures): a point with
/// media writes 48-byte media entries naming the picture-id scheme its keyed ids were made with (`pictureId`); a text
/// point is still written, byte for byte, as version 2, and version 2 manifests without media stay valid text points.
pub const version: u32 = 3;
/// The version a point without media is written as (and the oldest one read).
pub const text_version: u32 = 2;
/// The store namespace's version, in the compat digest: unchanged by version 3, so a store's text points survive it.
/// (A change that cannot read the older points bumps it: the directory name changes and the old store is pruned.)
pub const namespace_version: u32 = 2;
pub const magic = "TFSPILL\x02".*;
/// The manifest's fixed header.
pub const header_bytes = 256;
const fork_bytes = 20;
const media_bytes = 48; // version 3: start, len, kind, scheme, SHA-256
const media_v2_bytes = 40; // version 2 (never written with media: such a manifest is foreign)
/// A version-3 media entry's kind: a picture (an image's span of the prompt).
pub const media_picture: u32 = 1;
/// The scheme a picture's keyed ids are made with: `pictureId` (9014's ids, seeded by the digest's first 8 bytes).
pub const picture_scheme: u32 = 1;
const block_ref_bytes = 80;
const section_bytes = 56;
const trailer_bytes = 8;

/// How a rank's planes are laid out on disk. Replicated and owner-sharded stores never share a namespace (the compat
/// digest carries it too); only `replicated` is written in round 1.
pub const OwnerLayout = enum(u32) { replicated = 0, owner_sharded = 1 };

/// Bounds every parse checks before it allocates, derived from the context window and the model descriptor.
pub const Limits = struct {
    /// Most tokens a point may hold (the context window).
    max_tokens: u32,
    /// Tokens a block (the engine's chunk rows: 2048).
    block_tokens: u32,
    /// A full block's bytes on this rank (exact: a full block of any other length is damaged).
    block_bytes: u64,
    /// Most small-state bytes a point carries.
    max_small_bytes: u64,
    /// `owner_sharded` (0118): a full block's bytes on this rank for a point whose part (`Lineage.part`) another rank
    /// owns: that part's owner alone holds (and stores) the sharded planes, the other ranks only the replicated ones.
    /// 0: every full block is `block_bytes` (a replicated store).
    foreign_block_bytes: u64 = 0,
    /// `owner_sharded`: the part this rank owns (a point of that part has full blocks of `block_bytes`).
    own_part: u32 = 0,
    // Format limits, not settings: the most pictures, small-state sections and lineage forks a point may name. They
    // bound a manifest before anything is allocated (`maxManifest`, the open's read size), far above what a point
    // holds; lowering one would make points written under the larger bound read as corrupt.
    max_media: u32 = 4096,
    max_sections: u32 = 256,
    max_forks: u32 = 4096,

    /// A full block's exact bytes for a point of part `part` (its lineage's salt): `block_bytes`, or on an
    /// owner-sharded store `block_bytes` in this rank's own part and `foreign_block_bytes` in another; null for a
    /// part-less point there (every owner-sharded point names its part).
    pub fn fullBlockBytes(l: Limits, part: ?u32) ?u64 {
        if (l.foreign_block_bytes == 0) return l.block_bytes;
        const p = part orelse return null;
        return if (p == l.own_part) l.block_bytes else l.foreign_block_bytes;
    }

    pub fn maxBlocks(l: Limits) u32 {
        return std.math.divCeil(u32, l.max_tokens, l.block_tokens) catch 0;
    }

    /// The largest manifest these limits admit.
    pub fn maxManifest(l: Limits) u64 {
        return header_bytes + @as(u64, l.max_forks) * fork_bytes + @as(u64, l.max_media) * media_bytes +
            @as(u64, l.maxBlocks()) * block_ref_bytes + @as(u64, l.max_sections) * section_bytes + trailer_bytes;
    }
};

/// A picture inside a point: its prompt positions and the full SHA-256 of its payload (9014's per-position
/// keys are shortened; disk identity takes the whole digest and the span). A picture may run past the point's cut
/// (`start < at < start + len`). (extern: the same bytes as core/prompt_cache's Media, which the tier hands over.)
pub const Media = extern struct {
    start: u32,
    len: u32,
    digest: Digest,

    pub fn eql(a: Media, b: Media) bool {
        return a.start == b.start and a.len == b.len and std.mem.eql(u8, &a.digest, &b.digest);
    }
};

/// Keyed ids at or above this are pictures' (`pictureId`); vocabulary ids stay below it.
pub const picture_ids: u32 = 0x8000_0000;

/// Scheme 1's keyed id for position `j` of a picture whose payload's SHA-256 is `digest` (equal to
/// core/prompt_cache.pictureId, which the server keys prompts with; the tier's tests check the two agree).
pub fn pictureId(digest: *const Digest, j: u32) u32 {
    const seed = std.mem.readInt(u64, digest[0..8], .little);
    return picture_ids | (@as(u32, @truncate(std.hash.Wyhash.hash(seed, std.mem.asBytes(&j)))) & 0x7fff_ffff);
}

/// Whether a point's ids agree with its media: spans in order, not overlapping, not empty, each starting inside the
/// ids; every keyed id in a span is `pictureId(digest, offset)` and every other one a vocabulary id; with media the raw
/// ids are required, equal the keyed ones outside the spans and hold model ids (below `picture_ids`) inside them.
/// Bounded by the ids' length; what keep and recovery check before a point with pictures is trusted.
pub fn mediaIds(keyed: []const u32, raw: ?[]const u32, media: []const Media) bool {
    if (raw) |r| if (r.len != keyed.len) return false;
    if (media.len > 0 and raw == null) return false;
    var at: usize = 0;
    for (media) |m| {
        if (m.len == 0 or m.start < at or m.start >= keyed.len) return false;
        const stop: usize = @intCast(@min(@as(u64, m.start) + m.len, keyed.len));
        if (!textIds(keyed[at..m.start], if (raw) |r| r[at..m.start] else null)) return false;
        for (keyed[m.start..stop], 0..) |id, j| if (id != pictureId(&m.digest, @intCast(j))) return false;
        if (raw) |r| for (r[m.start..stop]) |id| if (id >= picture_ids) return false;
        at = @intCast(@as(u64, m.start) + m.len); // (past the ids when the picture straddles the cut: the loop ends)
    }
    if (at < keyed.len) return textIds(keyed[at..], if (raw) |r| r[at..] else null);
    return true;
}

fn textIds(keyed: []const u32, raw: ?[]const u32) bool {
    for (keyed) |id| if (id >= picture_ids) return false;
    if (raw) |r| if (!std.mem.eql(u32, keyed, r)) return false;
    return true;
}

/// One fork of a lineage: a computation that resumed at `at` and computed its own rows from there.
pub const Fork = struct { name: [16]u8, at: u32 };

/// The computation a state's rows belong to. Two prefills of the same ids may differ in their numerical history, so
/// only equal lineages (through a block) may share that block by name. A fresh prefill has its own root; a resume
/// inherits the source's lineage and forks it at the cut (the same names before the block holding the cut, its own
/// from there). `part` is 0118's part salt (owner-sharded pools): replaced, never stacked.
pub const Lineage = struct {
    root: [16]u8,
    part: ?u32 = null,
    forks: []const Fork = &.{},

    /// A fresh prefill's lineage: `boot` (this process's nonce) and its own sequence number.
    pub fn fresh(boot: [16]u8, seq: u64) Lineage {
        var h = Sha256.init(.{});
        h.update("tf-spill/lineage-root");
        h.update(&boot);
        h.update(std.mem.asBytes(&seq));
        const d = h.finalResult();
        return .{ .root = d[0..16].* };
    }

    /// `l` resumed at `at` by computation `name` (gpa-owned forks; free with `deinit`).
    pub fn fork(l: Lineage, gpa: Allocator, name: [16]u8, at: u32) !Lineage {
        const forks = try gpa.alloc(Fork, l.forks.len + 1);
        @memcpy(forks[0..l.forks.len], l.forks);
        forks[l.forks.len] = .{ .name = name, .at = at };
        return .{ .root = l.root, .part = l.part, .forks = forks };
    }

    /// `l` for rows that live in part `part` of an owner-sharded pool: the part replaces any earlier one (a lineage
    /// read back from disk or forked from a kept state already has one), forks untouched. Shares `l.forks`.
    pub fn inPart(l: Lineage, part: u32) Lineage {
        return .{ .root = l.root, .part = part, .forks = l.forks };
    }

    pub fn clone(l: Lineage, gpa: Allocator) !Lineage {
        return .{ .root = l.root, .part = l.part, .forks = try gpa.dupe(Fork, l.forks) };
    }

    pub fn deinit(l: *Lineage, gpa: Allocator) void {
        gpa.free(l.forks);
        l.forks = &.{};
    }

    pub fn eql(a: Lineage, b: Lineage) bool {
        if (!std.mem.eql(u8, &a.root, &b.root) or a.part != b.part or a.forks.len != b.forks.len) return false;
        for (a.forks, b.forks) |x, y| if (x.at != y.at or !std.mem.eql(u8, &x.name, &y.name)) return false;
        return true;
    }
};

/// One block of a point: tokens [start, end) and its chained name.
pub const Span = struct { start: u32, end: u32, name: Digest };

fn idsBytes(ids: []const u32) []const u8 {
    return std.mem.sliceAsBytes(ids);
}

/// Each block of `ids` under `lineage`: whole blocks, then the tail [k * block, n) if any. A name covers the lineage
/// (its root and part, then each fork whose position lies in the block), every id up to the block's end and, for a
/// tail, its length: equal names mean equal prefixes of one computation. A whole block's ids enter through `ids_chain`
/// (`chain(ids, block)`, which already commits to every id before the block's end), chained through the previous whole
/// block's name; only a tail's ids are hashed here. So naming every block costs one small hash a block on top of the
/// chain, not another pass over the ids. `out` holds ceil(n / block) spans.
pub fn blockNames(lineage: Lineage, ids_chain: []const Digest, ids: []const u32, block: u32, out: []Span) void {
    std.debug.assert(block > 0 and out.len == std.math.divCeil(usize, ids.len, block) catch unreachable);
    std.debug.assert(ids_chain.len == ids.len / block);
    var head: Digest = undefined;
    {
        var h = Sha256.init(.{});
        h.update("tf-spill/lineage");
        h.update(&lineage.root);
        if (lineage.part) |p| {
            h.update("#part");
            h.update(std.mem.asBytes(&p));
        }
        h.final(&head);
    }
    var s: usize = 0;
    var i: usize = 0;
    while (s < ids.len) : (i += 1) {
        const e = @min(ids.len, s + block);
        var h = Sha256.init(.{});
        h.update("tf-spill/block");
        h.update(&head);
        for (lineage.forks) |f| {
            if (s <= f.at and f.at < e) {
                h.update("|fork");
                h.update(&f.name);
                h.update(std.mem.asBytes(&f.at));
            }
        }
        const whole = e - s == block;
        if (whole) {
            h.update("|ids");
            h.update(&ids_chain[i]);
        } else {
            const end: u32 = @intCast(e);
            h.update("|tail");
            h.update(idsBytes(ids[s..e]));
            h.update(std.mem.asBytes(&end));
        }
        var d: Digest = undefined;
        h.final(&d);
        out[i] = .{ .start = @intCast(s), .end = @intCast(e), .name = d };
        if (whole) head = d;
        s = e;
    }
}

fn chainStep(prev: *Digest, ids: []const u32) void {
    var h = Sha256.init(.{});
    h.update(prev);
    h.update(idsBytes(ids));
    h.final(prev);
}

fn chainSeed() Digest {
    var d: Digest = undefined;
    Sha256.hash("tf-spill/ids", &d, .{});
    return d;
}

/// Tests: the passes over a state's ids `chain` and `pointKey` made, counted in test builds only.
pub var test_id_passes: std.atomic.Value(u64) = .init(0);

fn countPass() void {
    if (builtin.is_test) _ = test_id_passes.fetchAdd(1, .monotonic);
}

/// The id chain of `ids` (no lineage) at each whole block's end: two id lists share a prefix of k whole blocks
/// exactly when their first k chain digests agree. Appends to `out`. The one pass over a state's ids: its block
/// names and its key derive from this chain.
pub fn chain(gpa: Allocator, ids: []const u32, block: u32, out: *std.ArrayList(Digest)) !void {
    countPass();
    try out.ensureUnusedCapacity(gpa, ids.len / block);
    var prev = chainSeed();
    var s: usize = 0;
    while (s + block <= ids.len) : (s += block) {
        chainStep(&prev, ids[s .. s + block]);
        out.appendAssumeCapacity(prev);
    }
}

/// A point's key from its id chain (`chain(keyed, block)`): its length, the block size, the last whole block's chain
/// digest (every id before it), the tail's ids and the media identities (whichever computation produced it).
pub fn pointKeyChained(ids_chain: []const Digest, keyed: []const u32, block: u32, media: []const Media) Digest {
    std.debug.assert(block > 0 and ids_chain.len == keyed.len / block);
    const last = if (ids_chain.len > 0) ids_chain[ids_chain.len - 1] else chainSeed();
    return keyOf(last, keyed, ids_chain.len * block, block, media);
}

/// A point's key from its keyed ids alone (the chain computed here, nothing allocated): equals `pointKeyChained`.
pub fn pointKey(keyed: []const u32, block: u32, media: []const Media) Digest {
    countPass();
    std.debug.assert(block > 0);
    var last = chainSeed();
    const whole = keyed.len / block;
    for (0..whole) |i| chainStep(&last, keyed[i * block ..][0..block]);
    return keyOf(last, keyed, whole * block, block, media);
}

fn keyOf(last: Digest, keyed: []const u32, whole: usize, block: u32, media: []const Media) Digest {
    var h = Sha256.init(.{});
    h.update("tf-spill/point");
    const n: u32 = @intCast(keyed.len);
    h.update(std.mem.asBytes(&n));
    h.update(std.mem.asBytes(&block));
    h.update(&last);
    h.update(idsBytes(keyed[whole..]));
    const m: u32 = @intCast(media.len);
    h.update(std.mem.asBytes(&m));
    if (m > 0) { // (version 3: the scheme the keyed ids were made with; a text point's key is unchanged)
        h.update("|pictures");
        h.update(std.mem.asBytes(&picture_scheme));
    }
    for (media) |x| {
        h.update(std.mem.asBytes(&x.start));
        h.update(std.mem.asBytes(&x.len));
        h.update(&x.digest);
    }
    return h.finalResult();
}

/// The compat digest a store's directory is named by: the caller's common digest (build, kit, model, tokenizer,
/// geometry, TP plan) and this format's own fields. The caller exchanges and agrees it.
pub fn compatDigest(common: Digest, layout: OwnerLayout, block_tokens: u32, block_bytes: u64) Digest {
    var h = Sha256.init(.{});
    h.update("tf-spill/compat");
    h.update(std.mem.asBytes(&namespace_version));
    h.update(&common);
    const l: u32 = @backingInt(layout);
    h.update(std.mem.asBytes(&l));
    h.update(std.mem.asBytes(&block_tokens));
    h.update(std.mem.asBytes(&block_bytes));
    return h.finalResult();
}

pub fn hex(d: *const Digest) [64]u8 {
    return std.fmt.bytesToHex(d.*, .lower);
}

pub fn parseHex(s: []const u8) ?Digest {
    if (s.len != 64) return null;
    var d: Digest = undefined;
    _ = std.fmt.hexToBytes(&d, s) catch return null;
    return d;
}

pub fn crc32(bytes: []const u8) u32 {
    return Crc32.hash(bytes);
}

pub fn sha256(bytes: []const u8) Digest {
    var d: Digest = undefined;
    Sha256.hash(bytes, &d, .{});
    return d;
}

/// A small-state section's name: at most 32 bytes.
pub const Name = struct {
    bytes: [32]u8 = @splat(0),
    len: u8 = 0,

    pub fn of(s: []const u8) !Name {
        if (s.len == 0 or s.len > 32) return error.BadSectionName;
        var n: Name = .{ .len = @intCast(s.len) };
        @memcpy(n.bytes[0..s.len], s);
        return n;
    }

    pub fn slice(n: *const Name) []const u8 {
        return n.bytes[0..n.len];
    }
};

/// One block a point names: its chained name, its tokens and bytes, CRC-32 (checked on every read) and content
/// SHA-256 (what dedup compares; CRC alone never authorizes a link).
pub const BlockRef = struct { name: Digest, tokens: u32, bytes: u64, crc: u32, sha: Digest };

/// A named small-state section inside the point's `.st` file.
pub const Section = struct { name: Name, offset: u64, len: u64, crc: u32 };

pub const Flags = packed struct(u32) {
    shared: bool = false,
    has_raw: bool = false,
    /// The last block was captured from rows still being written (a mutable tail; native points are chunk-aligned).
    mutable_tail: bool = false,
    _: u29 = 0,
};

/// Where a point's computation came from: the point it resumed (informational; eviction uses the id tree).
pub const Origin = struct { key: Digest, at: u32 };

/// A point's manifest: the last file written, the only one that makes a point readable.
pub const Manifest = struct {
    key: Digest,
    at: u32,
    block_tokens: u32,
    flags: Flags = .{},
    group: u64 = 0,
    created_ns: i64 = 0,
    owner_layout: OwnerLayout = .replicated,
    owner_epoch: u32 = 0,
    lineage: Lineage,
    origin: ?Origin = null,
    ids_bytes: u64,
    ids_crc: u32,
    small_bytes: u64,
    small_crc: u32,
    media: []const Media = &.{},
    blocks: []const BlockRef = &.{},
    sections: []const Section = &.{},

    pub fn deinit(m: *Manifest, gpa: Allocator) void {
        gpa.free(m.media);
        gpa.free(m.blocks);
        gpa.free(m.sections);
        gpa.free(m.lineage.forks);
        m.* = undefined;
    }

    /// The version this manifest is written as: 3 with media, else 2 (a text point's bytes as before).
    pub fn versionOf(m: *const Manifest) u32 {
        return if (m.media.len > 0) version else text_version;
    }

    pub fn encodedLen(m: *const Manifest) u64 {
        return header_bytes + m.lineage.forks.len * fork_bytes + m.media.len * media_bytes +
            m.blocks.len * block_ref_bytes + m.sections.len * section_bytes + trailer_bytes;
    }

    /// The manifest's bytes (gpa-owned).
    pub fn encode(m: *const Manifest, gpa: Allocator, compat: Digest) ![]u8 {
        const out = try gpa.alloc(u8, @intCast(m.encodedLen()));
        @memset(out, 0);
        var w: Cursor = .{ .buf = out };
        w.bytes(&magic);
        w.int(u32, m.versionOf());
        w.int(u32, header_bytes);
        w.bytes(&compat);
        w.bytes(&m.key);
        w.int(u32, m.at);
        w.int(u32, m.block_tokens);
        w.int(u32, @bitCast(m.flags));
        w.int(u32, @intCast(m.media.len));
        w.int(u32, @intCast(m.blocks.len));
        w.int(u32, @intCast(m.sections.len));
        w.int(u32, @intCast(m.lineage.forks.len));
        w.int(u32, @backingInt(m.owner_layout));
        w.int(u32, m.owner_epoch);
        w.int(u64, m.group);
        w.int(i64, m.created_ns);
        w.bytes(&m.lineage.root);
        w.int(u32, @intFromBool(m.lineage.part != null));
        w.int(u32, m.lineage.part orelse 0);
        w.int(u32, @intFromBool(m.origin != null));
        w.int(u32, if (m.origin) |o| o.at else 0);
        w.bytes(if (m.origin) |*o| &o.key else &zero_digest);
        w.int(u64, m.ids_bytes);
        w.int(u32, m.ids_crc);
        w.int(u32, m.small_crc);
        w.int(u64, m.small_bytes);
        std.debug.assert(w.at <= header_bytes);
        w.at = header_bytes;
        for (m.lineage.forks) |f| {
            w.bytes(&f.name);
            w.int(u32, f.at);
        }
        for (m.media) |x| {
            w.int(u32, x.start);
            w.int(u32, x.len);
            w.int(u32, media_picture);
            w.int(u32, picture_scheme);
            w.bytes(&x.digest);
        }
        for (m.blocks) |b| {
            w.bytes(&b.name);
            w.int(u32, b.tokens);
            w.int(u32, b.crc);
            w.int(u64, b.bytes);
            w.bytes(&b.sha);
        }
        for (m.sections) |s| {
            w.bytes(&s.name.bytes);
            w.int(u64, s.offset);
            w.int(u64, s.len);
            w.int(u32, s.crc);
            w.int(u32, s.name.len);
        }
        w.int(u32, crc32(out[0..w.at]));
        w.int(u32, 0);
        std.debug.assert(w.at == out.len);
        return out;
    }

    pub const ParseError = error{ Foreign, Corrupt, OutOfMemory };

    /// The manifest in `raw` (a whole file), checked against `compat` and `limits`: sizes and counts before any
    /// allocation, the trailer CRC, then every field's consistency. Foreign: another format, version, store or
    /// geometry. Corrupt: anything else that does not add up.
    pub fn parse(gpa: Allocator, raw: []const u8, compat: Digest, limits: Limits) ParseError!Manifest {
        if (raw.len < header_bytes + trailer_bytes or raw.len > limits.maxManifest()) return error.Corrupt;
        var r: Cursor = .{ .buf = @constCast(raw) };
        if (!std.mem.eql(u8, r.take(8), &magic)) return error.Foreign;
        const ver = r.get(u32);
        if ((ver != version and ver != text_version) or r.get(u32) != header_bytes) return error.Foreign;
        if (!std.mem.eql(u8, r.take(32), &compat)) return error.Foreign;
        var m: Manifest = .{ .key = r.take(32)[0..32].*, .at = 0, .block_tokens = 0, .lineage = .{ .root = @splat(0) }, .ids_bytes = 0, .ids_crc = 0, .small_bytes = 0, .small_crc = 0 };
        m.at = r.get(u32);
        m.block_tokens = r.get(u32);
        m.flags = @bitCast(r.get(u32));
        const n_media = r.get(u32);
        const n_blocks = r.get(u32);
        const n_sections = r.get(u32);
        const n_forks = r.get(u32);
        const layout = r.get(u32);
        m.owner_epoch = r.get(u32);
        m.group = r.get(u64);
        m.created_ns = r.get(i64);
        m.lineage.root = r.take(16)[0..16].*;
        const has_part = r.get(u32);
        const part = r.get(u32);
        const has_origin = r.get(u32);
        const origin_at = r.get(u32);
        const origin_key = r.take(32)[0..32].*;
        m.ids_bytes = r.get(u64);
        m.ids_crc = r.get(u32);
        m.small_crc = r.get(u32);
        m.small_bytes = r.get(u64);
        if (m.block_tokens != limits.block_tokens) return error.Foreign;
        m.owner_layout = std.enums.fromInt(OwnerLayout, layout) orelse return error.Foreign;
        if (m.flags._ != 0 or has_part > 1 or has_origin > 1) return error.Corrupt;
        if (m.at == 0 or m.at > limits.max_tokens) return error.Corrupt;
        if (n_blocks != std.math.divCeil(u32, m.at, m.block_tokens) catch unreachable) return error.Corrupt;
        if (n_media > limits.max_media or n_sections > limits.max_sections or n_forks > limits.max_forks) return error.Corrupt;
        // version 2 points are text points (no build wrote one with media; its ids name no scheme: foreign), version 3
        // points hold pictures and their raw ids
        if (ver == text_version and n_media > 0) return error.Foreign;
        if (ver == version and (n_media == 0 or !m.flags.has_raw)) return error.Corrupt;
        const ids_want = @as(u64, m.at) * 4 * (if (m.flags.has_raw) @as(u64, 2) else 1);
        if (m.ids_bytes != ids_want or m.small_bytes > limits.max_small_bytes) return error.Corrupt;
        const total = header_bytes + @as(u64, n_forks) * fork_bytes + @as(u64, n_media) * (if (ver == version) @as(u64, media_bytes) else media_v2_bytes) +
            @as(u64, n_blocks) * block_ref_bytes + @as(u64, n_sections) * section_bytes + trailer_bytes;
        if (raw.len != total) return error.Corrupt;
        const body = raw.len - trailer_bytes;
        if (std.mem.readInt(u32, raw[body..][0..4], .little) != crc32(raw[0..body])) return error.Corrupt;
        if (std.mem.readInt(u32, raw[body + 4 ..][0..4], .little) != 0) return error.Corrupt;
        if (has_part == 1) m.lineage.part = part;
        if (has_origin == 1) m.origin = .{ .key = origin_key, .at = origin_at };
        r.at = header_bytes;

        const forks = try gpa.alloc(Fork, n_forks);
        errdefer gpa.free(forks);
        for (forks) |*f| {
            f.name = r.take(16)[0..16].*;
            f.at = r.get(u32);
            if (f.at > limits.max_tokens) return error.Corrupt;
        }
        const media = try gpa.alloc(Media, n_media);
        errdefer gpa.free(media);
        var media_end: u64 = 0;
        for (media) |*x| {
            x.start = r.get(u32);
            x.len = r.get(u32);
            const kind = r.get(u32);
            const scheme = r.get(u32);
            x.digest = r.take(32)[0..32].*;
            if (kind != media_picture or scheme != picture_scheme) return error.Foreign;
            // in order, not overlapping, starting inside the point (one may run past its cut), inside the window
            if (x.len == 0 or x.start < media_end or x.start >= m.at or @as(u64, x.start) + x.len > limits.max_tokens) return error.Corrupt;
            media_end = @as(u64, x.start) + x.len;
        }
        // (a part-less point in an owner-sharded store, or a foreign block size where none is kept: damaged)
        const full = limits.fullBlockBytes(m.lineage.part) orelse return error.Corrupt;
        if (full > limits.block_bytes) return error.Corrupt;
        const blocks = try gpa.alloc(BlockRef, n_blocks);
        errdefer gpa.free(blocks);
        for (blocks, 0..) |*b, i| {
            b.name = r.take(32)[0..32].*;
            b.tokens = r.get(u32);
            b.crc = r.get(u32);
            b.bytes = r.get(u64);
            b.sha = r.take(32)[0..32].*;
            const last = i + 1 == n_blocks;
            const want = if (!last) m.block_tokens else m.at - @as(u32, @intCast(i)) * m.block_tokens;
            if (b.tokens != want) return error.Corrupt;
            if (b.tokens == m.block_tokens) {
                if (b.bytes != full) return error.Corrupt;
            } else if (b.bytes == 0 or b.bytes > limits.block_bytes) return error.Corrupt;
        }
        const sections = try gpa.alloc(Section, n_sections);
        errdefer gpa.free(sections);
        var end: u64 = 0;
        for (sections) |*s| {
            s.name.bytes = r.take(32)[0..32].*;
            s.offset = r.get(u64);
            s.len = r.get(u64);
            s.crc = r.get(u32);
            const nl = r.get(u32);
            if (nl == 0 or nl > 32) return error.Corrupt;
            s.name.len = @intCast(nl);
            for (s.name.bytes[nl..]) |c| if (c != 0) return error.Corrupt;
            const stop = std.math.add(u64, s.offset, s.len) catch return error.Corrupt;
            if (s.offset < end or stop > m.small_bytes) return error.Corrupt;
            end = stop;
        }
        m.lineage.forks = forks;
        m.media = media;
        m.blocks = blocks;
        m.sections = sections;
        return m;
    }
};

/// A bounded little-endian cursor over a buffer whose size the caller checked first.
const Cursor = struct {
    buf: []u8,
    at: usize = 0,

    fn bytes(c: *Cursor, b: []const u8) void {
        @memcpy(c.buf[c.at..][0..b.len], b);
        c.at += b.len;
    }

    fn int(c: *Cursor, comptime T: type, v: T) void {
        std.mem.writeInt(T, c.buf[c.at..][0..@sizeOf(T)], v, .little);
        c.at += @sizeOf(T);
    }

    fn take(c: *Cursor, n: usize) []const u8 {
        defer c.at += n;
        return c.buf[c.at..][0..n];
    }

    fn get(c: *Cursor, comptime T: type) T {
        defer c.at += @sizeOf(T);
        return std.mem.readInt(T, c.buf[c.at..][0..@sizeOf(T)], .little);
    }
};

/// A point's `.ids` file: its keyed ids, then its raw ids when they differ (pictures), little-endian u32.
pub fn encodeIds(gpa: Allocator, keyed: []const u32, raw: ?[]const u32) ![]u8 {
    const n = keyed.len * 4 * (if (raw != null) @as(usize, 2) else 1);
    const out = try gpa.alloc(u8, n);
    @memcpy(out[0 .. keyed.len * 4], idsBytes(keyed));
    if (raw) |r| {
        std.debug.assert(r.len == keyed.len);
        @memcpy(out[keyed.len * 4 ..], idsBytes(r));
    }
    return out;
}

/// A recovered point's ids checked against its manifest (`CheckedIds.keyed` and its chain, gpa-owned).
pub const CheckedIds = struct {
    keyed: []u32,
    chain: []Digest,

    pub fn deinit(x: *CheckedIds, gpa: Allocator) void {
        gpa.free(x.keyed);
        gpa.free(x.chain);
        x.* = undefined;
    }
};

/// What recovery trusts of a point's .ids file, given its parsed manifest `m`: `raw` (the whole file) has the
/// manifest's length and CRC, its keyed ids' chain gives the manifest's key, and with pictures its keyed and raw ids
/// agree with the media (`mediaIds`). Bounded by `m.at`, which `Manifest.parse` checked against the limits. The keyed
/// ids and their chain, else Corrupt. A text point's raw ids, when it has them, are not compared with its keyed ids.
pub fn checkIds(gpa: Allocator, raw: []const u8, m: *const Manifest) error{ Corrupt, OutOfMemory }!CheckedIds {
    if (raw.len != m.ids_bytes or crc32(raw) != m.ids_crc) return error.Corrupt;
    const n: usize = m.at;
    const sets: usize = if (m.media.len > 0) 2 else 1;
    if (m.block_tokens == 0 or raw.len < sets * n * 4) return error.Corrupt; // parse rules both out; defensive
    const keyed = try gpa.alloc(u32, n);
    errdefer gpa.free(keyed);
    @memcpy(std.mem.sliceAsBytes(keyed), raw[0 .. n * 4]);
    var ch: std.ArrayList(Digest) = .empty;
    errdefer ch.deinit(gpa);
    try chain(gpa, keyed, m.block_tokens, &ch);
    if (!std.mem.eql(u8, &pointKeyChained(ch.items, keyed, m.block_tokens, m.media), &m.key)) return error.Corrupt;
    if (m.media.len > 0) { // a point with pictures: its keyed and raw ids agree with its media (version 3)
        const raw_ids = try gpa.alloc(u32, n);
        defer gpa.free(raw_ids);
        @memcpy(std.mem.sliceAsBytes(raw_ids), raw[n * 4 ..][0 .. n * 4]);
        if (!mediaIds(keyed, raw_ids, m.media)) return error.Corrupt;
    }
    return .{ .keyed = keyed, .chain = try ch.toOwnedSlice(gpa) };
}

// -- tests ---------------------------------------------------------------------------------------------------------
const testing = std.testing;

fn testNames(lineage: Lineage, ids: []const u32, out: []Span) !void {
    var ch: std.ArrayList(Digest) = .empty;
    defer ch.deinit(testing.allocator);
    try chain(testing.allocator, ids, 16, &ch);
    blockNames(lineage, ch.items, ids, 16, out);
}

fn testIds(gpa: Allocator, n: usize, seed: u64) ![]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const ids = try gpa.alloc(u32, n);
    for (ids) |*x| x.* = prng.random().intRangeAtMost(u32, 1, 150000);
    return ids;
}

test "block names: shared prefixes of one computation share names, forks diverge at the cut's block" {
    const gpa = testing.allocator;
    const ids = try testIds(gpa, 3 * 16 + 5, 1);
    defer gpa.free(ids);
    const root: Lineage = .{ .root = @splat(7) };
    var a: [4]Span = undefined;
    try testNames(root, ids, &a);
    var b: [3]Span = undefined;
    try testNames(root, ids[0..48], &b);
    for (0..3) |i| try testing.expectEqualSlices(u8, &a[i].name, &b[i].name);
    try testing.expectEqual(@as(u32, 48), a[3].start);
    try testing.expectEqual(@as(u32, 53), a[3].end);
    // a tail's name covers its length: the same ids one shorter is another name
    var c: [4]Span = undefined;
    try testNames(root, ids[0..52], &c);
    try testing.expect(!std.mem.eql(u8, &a[3].name, &c[3].name));
    // a fork at 32: blocks before it keep their names, the block holding it and every later one are new
    var forked = try root.fork(gpa, @splat(9), 32);
    defer forked.deinit(gpa);
    var f: [4]Span = undefined;
    try testNames(forked, ids, &f);
    try testing.expectEqualSlices(u8, &a[0].name, &f[0].name);
    try testing.expectEqualSlices(u8, &a[1].name, &f[1].name);
    try testing.expect(!std.mem.eql(u8, &a[2].name, &f[2].name));
    try testing.expect(!std.mem.eql(u8, &a[3].name, &f[3].name));
    // another computation of the same ids shares nothing by name
    var o: [4]Span = undefined;
    try testNames(.{ .root = @splat(8) }, ids, &o);
    for (0..4) |i| try testing.expect(!std.mem.eql(u8, &a[i].name, &o[i].name));
}

test "full block sizes: a replicated store's every point block_bytes; an owner-sharded one's by its part" {
    const rep: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    for ([_]?u32{ null, 0, 3 }) |p| try std.testing.expectEqual(@as(?u64, 4096), rep.fullBlockBytes(p));
    var sh = rep;
    sh.foreign_block_bytes = 1024;
    sh.own_part = 1;
    try std.testing.expectEqual(@as(?u64, 4096), sh.fullBlockBytes(1));
    try std.testing.expectEqual(@as(?u64, 1024), sh.fullBlockBytes(0));
    try std.testing.expectEqual(@as(?u64, null), sh.fullBlockBytes(null));
}

test "part salt: replaced, never stacked; forks untouched" {
    const gpa = testing.allocator;
    const ids = try testIds(gpa, 32, 2);
    defer gpa.free(ids);
    var l = try (Lineage{ .root = @splat(1) }).fork(gpa, @splat(2), 16);
    defer l.deinit(gpa);
    const p1 = l.inPart(1);
    const p1again = p1.inPart(1);
    const p0 = p1.inPart(0);
    try testing.expect(p1.eql(p1again));
    try testing.expect(!p1.eql(p0));
    try testing.expect(p0.eql(l.inPart(0)));
    var x: [2]Span = undefined;
    var y: [2]Span = undefined;
    var z: [2]Span = undefined;
    try testNames(p1, ids, &x);
    try testNames(p1again, ids, &y);
    try testNames(l, ids, &z);
    try testing.expectEqualSlices(u8, &x[0].name, &y[0].name);
    try testing.expect(!std.mem.eql(u8, &x[0].name, &z[0].name)); // an unsharded store's names are unchanged
}

test "chain and point key" {
    const gpa = testing.allocator;
    const ids = try testIds(gpa, 40, 3);
    defer gpa.free(ids);
    var c1: std.ArrayList(Digest) = .empty;
    defer c1.deinit(gpa);
    try chain(gpa, ids, 16, &c1);
    try testing.expectEqual(@as(usize, 2), c1.items.len);
    var c2: std.ArrayList(Digest) = .empty;
    defer c2.deinit(gpa);
    try chain(gpa, ids[0..32], 16, &c2);
    try testing.expectEqualSlices(u8, &c1.items[1], &c2.items[1]);
    const m: Media = .{ .start = 3, .len = 4, .digest = @splat(5) };
    const k1 = pointKey(ids, 16, &.{});
    const k2 = pointKey(ids, 16, &.{m});
    try testing.expect(!std.mem.eql(u8, &k1, &k2));
    var m2 = m;
    m2.digest[0] = 6;
    try testing.expect(!std.mem.eql(u8, &k2, &pointKey(ids, 16, &.{m2})));
    try testing.expectEqualSlices(u8, &k2, &pointKeyChained(c1.items, ids, 16, &.{m}));
    try testing.expectEqualSlices(u8, &k1, &pointKeyChained(c1.items, ids, 16, &.{}));
    // a tail's ids count: one id changed after the last whole block is another key
    var changed = try gpa.dupe(u32, ids);
    defer gpa.free(changed);
    changed[39] +%= 1;
    try testing.expect(!std.mem.eql(u8, &k1, &pointKey(changed, 16, &.{})));
    changed[39] -%= 1;
    changed[3] +%= 1;
    try testing.expect(!std.mem.eql(u8, &k1, &pointKey(changed, 16, &.{})));
    try testing.expect(!std.mem.eql(u8, &k1, &pointKey(ids[0..39], 16, &.{})));
}

fn sampleManifest(gpa: Allocator, l: Limits) !Manifest {
    const blocks = try gpa.alloc(BlockRef, 3);
    for (blocks, 0..) |*b, i| b.* = .{ .name = @splat(@intCast(i)), .tokens = if (i < 2) 16 else 5, .bytes = if (i < 2) l.block_bytes else 100, .crc = 7, .sha = @splat(3) };
    const sections = try gpa.alloc(Section, 2);
    sections[0] = .{ .name = try Name.of("ring"), .offset = 0, .len = 64, .crc = 1 };
    sections[1] = .{ .name = try Name.of("raw"), .offset = 64, .len = 32, .crc = 2 };
    const media = try gpa.alloc(Media, 1);
    media[0] = .{ .start = 4, .len = 3, .digest = @splat(9) };
    const forks = try gpa.alloc(Fork, 1);
    forks[0] = .{ .name = @splat(4), .at = 16 };
    return .{
        .key = @splat(1),
        .at = 37,
        .block_tokens = 16,
        .flags = .{ .shared = true, .has_raw = true },
        .group = 99,
        .created_ns = 12345,
        .lineage = .{ .root = @splat(2), .part = 1, .forks = forks },
        .origin = .{ .key = @splat(8), .at = 32 },
        .ids_bytes = 37 * 8,
        .ids_crc = 5,
        .small_bytes = 96,
        .small_crc = 6,
        .media = media,
        .blocks = blocks,
        .sections = sections,
    };
}

test "manifest round trip and bounded parse" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    const compat: Digest = @splat(0xab);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    var back = try Manifest.parse(gpa, raw, compat, l);
    defer back.deinit(gpa);
    try testing.expectEqual(m.at, back.at);
    try testing.expectEqual(m.group, back.group);
    try testing.expect(back.flags.shared and back.flags.has_raw);
    try testing.expect(back.lineage.eql(m.lineage));
    try testing.expectEqual(@as(u32, 32), back.origin.?.at);
    try testing.expectEqualStrings("raw", back.sections[1].name.slice());
    try testing.expect(back.media[0].eql(m.media[0]));
    try testing.expectEqual(@as(u64, 100), back.blocks[2].bytes);

    // another store, another geometry: foreign; any flipped byte: corrupt (the trailer CRC), never a crash
    try testing.expectError(error.Foreign, Manifest.parse(gpa, raw, @splat(0xac), l));
    var l2 = l;
    l2.block_tokens = 32;
    try testing.expectError(error.Foreign, Manifest.parse(gpa, raw, compat, l2));
    const bad = try gpa.dupe(u8, raw);
    defer gpa.free(bad);
    for (0..bad.len) |i| {
        bad[i] ^= 0x41;
        if (Manifest.parse(gpa, bad, compat, l)) |*ok| {
            var x = ok.*;
            x.deinit(gpa);
            return error.TestUnexpectedResult;
        } else |_| {}
        bad[i] ^= 0x41;
    }
    // truncation at every length is refused before allocation
    for (0..raw.len) |n| try testing.expect(std.meta.isError(Manifest.parse(gpa, raw[0..n], compat, l)));
    // a count past the limits is refused even when the bytes would add up
    var l3 = l;
    l3.max_media = 0;
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, raw, compat, l3));
    // a full block of another size is damage
    var l4 = l;
    l4.block_bytes = 8192;
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, raw, compat, l4));
}

test "a huge count in a forged header with a valid CRC is refused before allocating" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    const compat: Digest = @splat(1);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    // n_media at offset 8+4+4+32+32+4+4+4 = 92
    std.mem.writeInt(u32, raw[92..96], 0xffff_fff0, .little);
    const body = raw.len - trailer_bytes;
    std.mem.writeInt(u32, raw[body..][0..4], crc32(raw[0..body]), .little);
    var failing = testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    try testing.expectError(error.Corrupt, Manifest.parse(failing.allocator(), raw, compat, l));
}

// -- pictures (version 3) --------------------------------------------------------------------------------------------

/// `sampleManifest` without media: a text point (version 2).
fn textManifest(gpa: Allocator, l: Limits) !Manifest {
    var m = try sampleManifest(gpa, l);
    gpa.free(m.media);
    m.media = &.{};
    m.flags.has_raw = false;
    m.ids_bytes = 37 * 4;
    return m;
}

test "manifest versions: a text point is written and read as version 2, a point with pictures as version 3" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5a);
    // text: version 2, 40-byte media entries never written, the bytes those of the format before version 3
    var t = try textManifest(gpa, l);
    defer t.deinit(gpa);
    const traw = try t.encode(gpa, compat);
    defer gpa.free(traw);
    try testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, traw[8..12], .little));
    var tback = try Manifest.parse(gpa, traw, compat, l);
    tback.deinit(gpa);
    // pictures: version 3, 48-byte entries (kind, scheme)
    var p = try sampleManifest(gpa, l);
    defer p.deinit(gpa);
    const praw = try p.encode(gpa, compat);
    defer gpa.free(praw);
    try testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, praw[8..12], .little));
    try testing.expectEqual(traw.len + media_bytes, praw.len);
    var pback = try Manifest.parse(gpa, praw, compat, l);
    defer pback.deinit(gpa);
    try testing.expect(pback.media[0].eql(p.media[0]));
    // the compat digest names the namespace, not the manifest version: version 3 keeps a store's text points
    try testing.expectEqual(@as(u32, 2), namespace_version);

    const Fix = struct {
        fn crc(raw: []u8) void {
            const body = raw.len - trailer_bytes;
            std.mem.writeInt(u32, raw[body..][0..4], crc32(raw[0..body]), .little);
        }
    };
    const bad = try gpa.dupe(u8, praw);
    defer gpa.free(bad);
    // a version 2 manifest with media (the scheme of its ids unknown): foreign, even with its CRC made good
    std.mem.writeInt(u32, bad[8..12], 2, .little);
    Fix.crc(bad);
    try testing.expectError(error.Foreign, Manifest.parse(gpa, bad, compat, l));
    // a version 3 manifest without media, or without raw ids: corrupt
    const t3 = try gpa.dupe(u8, traw);
    defer gpa.free(t3);
    std.mem.writeInt(u32, t3[8..12], 3, .little);
    Fix.crc(t3);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, t3, compat, l));
    @memcpy(bad, praw);
    const flags_at = 8 + 4 + 4 + 32 + 32 + 4 + 4;
    var fl: Flags = @bitCast(std.mem.readInt(u32, bad[flags_at..][0..4], .little));
    fl.has_raw = false;
    std.mem.writeInt(u32, bad[flags_at..][0..4], @bitCast(fl), .little);
    Fix.crc(bad);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, bad, compat, l));
    // another kind or scheme: foreign; a span out of order or past the window: corrupt
    const media_at = header_bytes + p.lineage.forks.len * fork_bytes;
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at + 8 ..][0..4], 2, .little);
    Fix.crc(bad);
    try testing.expectError(error.Foreign, Manifest.parse(gpa, bad, compat, l));
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at + 12 ..][0..4], 2, .little);
    Fix.crc(bad);
    try testing.expectError(error.Foreign, Manifest.parse(gpa, bad, compat, l));
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at + 4 ..][0..4], 2000, .little);
    Fix.crc(bad);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, bad, compat, l));
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at..][0..4], 37, .little); // at the cut, not before it
    Fix.crc(bad);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, bad, compat, l));
}

test "manifest with pictures: two overlapping spans are refused, a span straddling the cut is not" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5b);
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    gpa.free(m.media);
    const media = try gpa.alloc(Media, 2);
    m.media = media;
    media[0] = .{ .start = 4, .len = 3, .digest = @splat(9) };
    media[1] = .{ .start = 30, .len = 20, .digest = @splat(8) }; // [30, 50) runs past the cut at 37
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    var back = try Manifest.parse(gpa, raw, compat, l);
    back.deinit(gpa);
    media[1].start = 5; // inside the first span
    const raw2 = try m.encode(gpa, compat);
    defer gpa.free(raw2);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, raw2, compat, l));
}

test "manifest fuzz: random edits of a version 3 manifest never crash and never parse into an inconsistent point" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5c);
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    var prng = std.Random.DefaultPrng.init(0x7069637475726573);
    const rnd = prng.random();
    const buf = try gpa.alloc(u8, raw.len + 64);
    defer gpa.free(buf);
    var parsed: usize = 0;
    for (0..20_000) |_| {
        // a copy, a few random bytes changed (in the media entries half the time), a random length, the CRC made good
        // half the time so the field checks run, not only the trailer's
        var n = raw.len;
        @memcpy(buf[0..n], raw);
        const media_at = header_bytes + m.lineage.forks.len * fork_bytes;
        for (0..rnd.intRangeAtMost(usize, 1, 4)) |_| {
            const i = if (rnd.boolean()) media_at + rnd.uintLessThan(usize, media_bytes) else rnd.uintLessThan(usize, n);
            buf[i] = rnd.int(u8);
        }
        if (rnd.uintLessThan(u8, 8) == 0) n = rnd.uintLessThan(usize, raw.len + 64);
        if (n >= trailer_bytes and rnd.boolean()) {
            const body = n - trailer_bytes;
            std.mem.writeInt(u32, buf[body..][0..4], crc32(buf[0..body]), .little);
            std.mem.writeInt(u32, buf[body + 4 ..][0..4], 0, .little);
        }
        var got = Manifest.parse(gpa, buf[0..n], compat, l) catch continue;
        defer got.deinit(gpa);
        parsed += 1;
        // whatever parses is consistent: spans in order inside the point and the window, version and media agree
        var end: u64 = 0;
        for (got.media) |x| {
            try testing.expect(x.len > 0 and x.start >= end and x.start < got.at and @as(u64, x.start) + x.len <= l.max_tokens);
            end = @as(u64, x.start) + x.len;
        }
        try testing.expect(got.media.len == 0 or got.flags.has_raw);
        try testing.expect(got.at > 0 and got.at <= l.max_tokens and got.blocks.len == std.math.divCeil(u32, got.at, 16) catch unreachable);
        try expectConsistent(&got, l); // the whole property, blocks and sections too
    }
    try testing.expect(parsed > 0); // (the CRC-made-good edits of unchecked bytes, e.g. a digest, still parse)
}

/// The pinned points' ids: 40 ids (two whole blocks of 16 and a tail of 8), from a formula (no PRNG: its stream may
/// change with the toolchain, these bytes must not).
fn pinnedIds() [40]u32 {
    var ids: [40]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast((@as(u64, i) * 2654435761 + 12345) % 150000 + 1);
    return ids;
}

/// A fixed point's manifest (gpa-owned blocks), its point key and its blocks' spans: a text point, or with
/// `picture` a version 3 point whose ids 3..6 are a picture's (raw ids 99 there).
fn pinnedManifest(gpa: Allocator, picture: bool, keyed: *[40]u32, raw: *[40]u32, spans: *[3]Span) !Manifest {
    keyed.* = pinnedIds();
    raw.* = keyed.*;
    const media: []const Media = if (picture) &pinned_media else &.{};
    if (picture) for (3..7) |i| {
        keyed[i] = pictureId(&pinned_media[0].digest, @intCast(i - 3));
        raw[i] = 99;
    };
    var ch: std.ArrayList(Digest) = .empty;
    defer ch.deinit(gpa);
    try chain(gpa, keyed, 16, &ch);
    blockNames(pinned_lineage, ch.items, keyed, 16, spans);
    const blocks = try gpa.alloc(BlockRef, 3);
    for (blocks, spans) |*b, sp| b.* = .{ .name = sp.name, .tokens = sp.end - sp.start, .bytes = if (sp.end - sp.start == 16) 4096 else 100, .crc = 7, .sha = @splat(3) };
    const ids = try encodeIds(gpa, keyed, if (picture) raw else null);
    defer gpa.free(ids);
    return .{
        .key = pointKeyChained(ch.items, keyed, 16, media),
        .at = 40,
        .block_tokens = 16,
        .flags = .{ .has_raw = picture },
        .group = 99,
        .created_ns = 12345,
        .lineage = pinned_lineage,
        .origin = .{ .key = @splat(8), .at = 32 },
        .ids_bytes = ids.len,
        .ids_crc = crc32(ids),
        .small_bytes = 96,
        .small_crc = 6,
        .media = media,
        .blocks = blocks,
        .sections = &pinned_sections,
    };
}

const pinned_forks = [_]Fork{.{ .name = @splat(4), .at = 16 }};
const pinned_lineage: Lineage = .{ .root = @splat(2), .forks = &pinned_forks };
const pinned_media = [_]Media{.{ .start = 3, .len = 4, .digest = @splat(1) }};
const pinned_sections = [_]Section{
    .{ .name = .{ .bytes = "ring".* ++ @as([28]u8, @splat(0)), .len = 4 }, .offset = 0, .len = 64, .crc = 1 },
    .{ .name = .{ .bytes = "raw".* ++ @as([29]u8, @splat(0)), .len = 3 }, .offset = 64, .len = 32, .crc = 2 },
};

test "format bytes pinned: manifests, ids, keys, block names and the compat digest" {
    // a change to the store that moves a byte of these fails here: a store already on disk would read differently
    const gpa = testing.allocator;
    const want = [2]struct { manifest: []const u8, ids: []const u8, key: []const u8, block0: []const u8 }{
        .{ // text (version 2)
            .manifest = "b4df82d84520843e84309e2ce8da1871e3132431581f07cc578ef71a12858703",
            .ids = "bfb8411904d0956673254c00b137cce9f5a0c39934b180df086513c1e5a8d4c2",
            .key = "f125c5d64691f63df79dab84357e2ca2ff38e35c3e117c25f0ba5c5a2e8e2581",
            .block0 = "d6c341b34036c09514092df46872d186ffc2478ca4ae02792b78cb221d3ffec5",
        },
        .{ // a picture (version 3)
            .manifest = "83a6645e1c8e01e0ddae8f9b9c0fb9cc31ad567dfd38ddd53f17138b3745a0d9",
            .ids = "c474263454f651211fb0e285823d06005ff4f0b9bf507ff394570328c0053044",
            .key = "a873aef5c9dac2c69d06d113b2006dbd4a5f88ed504b4f06dcc1bb78c111ac55",
            .block0 = "3389d57e4e396a407e9ab580e3529123107a7ce18db71dcbf498cedeb65a60f5",
        },
    };
    const compat: Digest = @splat(0x5d);
    var bad: usize = 0;
    for (want, 0..) |w, i| {
        var keyed: [40]u32 = undefined;
        var raw: [40]u32 = undefined;
        var spans: [3]Span = undefined;
        var m = try pinnedManifest(gpa, i == 1, &keyed, &raw, &spans);
        defer gpa.free(m.blocks);
        const bytes = try m.encode(gpa, compat);
        defer gpa.free(bytes);
        const ids = try encodeIds(gpa, &keyed, if (i == 1) &raw else null);
        defer gpa.free(ids);
        const got = [4][64]u8{ hex(&sha256(bytes)), hex(&sha256(ids)), hex(&m.key), hex(&spans[0].name) };
        const exp = [4][]const u8{ w.manifest, w.ids, w.key, w.block0 };
        for (got, exp, [_][]const u8{ "manifest", "ids", "key", "block0" }) |g, e, what| if (!std.mem.eql(u8, &g, e)) {
            std.debug.print("pinned {s} of point {d}: {s}\n", .{ what, i, &g });
            bad += 1;
        };
    }
    const cd = hex(&compatDigest(@splat(7), .replicated, 2048, 1_822_720));
    if (!std.mem.eql(u8, &cd, "4ab7a065452b2a95b32073a0941924cc3d90e80a32bf57db26f838d058007acb")) {
        std.debug.print("pinned compat digest: {s}\n", .{&cd});
        bad += 1;
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

/// What a parsed manifest satisfies whatever bytes it came from: its length and blocks agree, each block's tokens and
/// bytes are a block's, its ids' size is its length's, its sections lie in order inside its small state, its forks
/// inside the window, its picture spans in order inside the point and the window with raw ids.
fn expectConsistent(m: *const Manifest, l: Limits) !void {
    try testing.expect(m.at > 0 and m.at <= l.max_tokens and m.block_tokens == l.block_tokens);
    try testing.expect(m.blocks.len == std.math.divCeil(u32, m.at, m.block_tokens) catch unreachable);
    for (m.blocks, 0..) |b, i| {
        const want = if (i + 1 < m.blocks.len) m.block_tokens else m.at - @as(u32, @intCast(i)) * m.block_tokens;
        try testing.expect(b.tokens == want);
        if (b.tokens == m.block_tokens) try testing.expect(b.bytes == l.fullBlockBytes(m.lineage.part).?) else try testing.expect(b.bytes > 0 and b.bytes <= l.block_bytes);
    }
    try testing.expect(m.ids_bytes == @as(u64, m.at) * 4 * (if (m.flags.has_raw) @as(u64, 2) else 1));
    try testing.expect(m.small_bytes <= l.max_small_bytes and m.sections.len <= l.max_sections);
    var end: u64 = 0;
    for (m.sections) |sec| {
        try testing.expect(sec.name.len > 0 and sec.name.len <= 32 and sec.offset >= end and sec.offset + sec.len <= m.small_bytes);
        end = sec.offset + sec.len;
    }
    try testing.expect(m.lineage.forks.len <= l.max_forks);
    for (m.lineage.forks) |f| try testing.expect(f.at <= l.max_tokens);
    var mend: u64 = 0;
    try testing.expect(m.media.len <= l.max_media and (m.media.len == 0 or m.flags.has_raw));
    for (m.media) |x| {
        try testing.expect(x.len > 0 and x.start >= mend and x.start < m.at and @as(u64, x.start) + x.len <= l.max_tokens);
        mend = @as(u64, x.start) + x.len;
    }
}

test "manifest fuzz: random edits of a version 2 (text) manifest never crash and never parse into an inconsistent point" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5e);
    var m = try textManifest(gpa, l);
    defer m.deinit(gpa);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    try testing.expectEqual(text_version, std.mem.readInt(u32, raw[8..12], .little));
    var prng = std.Random.DefaultPrng.init(0x7465787432); // "text2"
    const rnd = prng.random();
    const buf = try gpa.alloc(u8, raw.len + 64);
    defer gpa.free(buf);
    var parsed: usize = 0;
    for (0..20_000) |_| {
        // a copy, a few random bytes changed (in the fixed header half the time: its counts, lengths and flags), a
        // random length now and then, the CRC made good half the time so the field checks run, not only the trailer's
        var n = raw.len;
        @memcpy(buf[0..n], raw);
        for (0..rnd.intRangeAtMost(usize, 1, 4)) |_| {
            const i = if (rnd.boolean()) rnd.uintLessThan(usize, header_bytes) else rnd.uintLessThan(usize, n);
            buf[i] = rnd.int(u8);
        }
        if (rnd.uintLessThan(u8, 8) == 0) n = rnd.uintLessThan(usize, raw.len + 64);
        if (n >= trailer_bytes and rnd.boolean()) {
            const body = n - trailer_bytes;
            std.mem.writeInt(u32, buf[body..][0..4], crc32(buf[0..body]), .little);
            std.mem.writeInt(u32, buf[body + 4 ..][0..4], 0, .little);
        }
        var got = Manifest.parse(gpa, buf[0..n], compat, l) catch continue;
        defer got.deinit(gpa);
        parsed += 1;
        try expectConsistent(&got, l);
        try testing.expect(got.media.len == 0); // (version 2 never carries pictures)
    }
    try testing.expect(parsed > 0); // (the CRC-made-good edits of unchecked bytes, e.g. a digest, still parse)
}

test "ids fuzz: random edits of a point's .ids never crash checkIds, and what it accepts gives the key and agrees with the media" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6964732e66757a7a); // "ids.fuzz"
    const rnd = prng.random();
    var accepted: [2]usize = .{ 0, 0 };
    var changed: [2]usize = .{ 0, 0 }; // accepted with bytes other than the point's own
    for (0..2) |picture| {
        var keyed: [40]u32 = undefined;
        var raw_ids: [40]u32 = undefined;
        var spans: [3]Span = undefined;
        var m = try pinnedManifest(gpa, picture == 1, &keyed, &raw_ids, &spans);
        defer gpa.free(m.blocks);
        const ids = try encodeIds(gpa, &keyed, if (picture == 1) &raw_ids else null);
        defer gpa.free(ids);
        var whole = try checkIds(gpa, ids, &m); // (the point as written: accepted)
        whole.deinit(gpa);
        const buf = try gpa.alloc(u8, ids.len + 16);
        defer gpa.free(buf);
        for (0..10_000) |_| {
            var n = ids.len;
            @memcpy(buf[0..n], ids);
            const ids_n = n / 4;
            for (0..rnd.intRangeAtMost(usize, 1, 3)) |_| {
                const w: []align(1) u32 = std.mem.bytesAsSlice(u32, buf[0 .. ids_n * 4]);
                switch (rnd.uintLessThan(u8, 5)) {
                    0 => buf[rnd.uintLessThan(usize, n)] = rnd.int(u8), // a byte
                    1 => w[rnd.uintLessThan(usize, w.len)] = picture_ids | rnd.int(u32), // a picture id anywhere
                    2 => w[rnd.uintLessThan(usize, w.len)] = pictureId(&pinned_media[0].digest, rnd.uintLessThan(u32, 8)),
                    3 => if (picture == 1) { // a raw id inside the picture: another model id, or a picture id
                        w[40 + rnd.intRangeAtMost(usize, 3, 6)] = if (rnd.boolean()) rnd.intRangeAtMost(u32, 1, 150000) else picture_ids | 5;
                    },
                    else => w[rnd.uintLessThan(usize, w.len)] = rnd.intRangeAtMost(u32, 1, 150000), // a model id
                }
            }
            if (rnd.uintLessThan(u8, 8) == 0) n = rnd.uintLessThan(usize, ids.len + 16); // another length
            var mm = m;
            if (rnd.uintLessThan(u8, 4) != 0) mm.ids_crc = crc32(buf[0..n]); // (made good: the deeper checks run)
            var got = checkIds(gpa, buf[0..n], &mm) catch |e| {
                try testing.expect(e == error.Corrupt);
                continue;
            };
            defer got.deinit(gpa);
            accepted[picture] += 1;
            if (n != ids.len or !std.mem.eql(u8, buf[0..n], ids)) changed[picture] += 1;
            // accepted: the ids' chain gives the manifest's key, the media agree, the chain is the ids'
            try testing.expectEqualSlices(u8, &m.key, &pointKey(got.keyed, 16, m.media));
            if (picture == 1) {
                const r: []align(1) const u32 = std.mem.bytesAsSlice(u32, buf[40 * 4 .. 80 * 4]);
                var rr: [40]u32 = undefined;
                for (&rr, r) |*d, x| d.* = x;
                try testing.expect(mediaIds(got.keyed, &rr, m.media));
            }
            try testing.expectEqualSlices(u32, &keyed, got.keyed); // (the key covers every keyed id)
        }
    }
    // a text point accepts only its own ids (edits that changed nothing); a picture point also raw ids changed
    // inside the picture to other model ids (raw ids are not in the key, and recovery's media check allows them)
    try testing.expect(accepted[0] > 0 and changed[0] == 0 and changed[1] > 0);
}

test "media ids: keyed ids derive from the digest, raw ids hold model ids there and equal the keyed ones elsewhere" {
    const d: Digest = @splat(3);
    var keyed = [_]u32{ 5, 6, 0, 0, 0, 7, 0, 0 };
    var raw = [_]u32{ 5, 6, 99, 99, 99, 7, 99, 99 };
    for (2..5) |i| keyed[i] = pictureId(&d, @intCast(i - 2));
    for (6..8) |i| keyed[i] = pictureId(&d, @intCast(i - 6)); // a second picture straddling the end
    const m = [_]Media{ .{ .start = 2, .len = 3, .digest = d }, .{ .start = 6, .len = 10, .digest = d } };
    try testing.expect(mediaIds(&keyed, &raw, &m));
    try testing.expect(!mediaIds(&keyed, null, &m)); // pictures need their raw ids
    try testing.expect(!mediaIds(&keyed, &raw, m[0..1])); // a picture the media does not name
    try testing.expect(!mediaIds(&keyed, &raw, &.{ m[1], m[0] })); // out of order
    try testing.expect(!mediaIds(&keyed, &raw, &.{ m[0], .{ .start = 4, .len = 1, .digest = d } })); // overlapping
    var r2 = raw;
    r2[0] = 4;
    try testing.expect(!mediaIds(&keyed, &r2, &m)); // raw differs outside a picture
    r2 = raw;
    r2[3] = keyed[3];
    try testing.expect(!mediaIds(&keyed, &r2, &m)); // a picture id where a model id belongs
    var k2 = keyed;
    k2[3] +%= 1;
    try testing.expect(!mediaIds(&k2, &raw, &m)); // an id not made from the digest
    try testing.expect(mediaIds(&.{ 1, 2, 3 }, null, &.{})); // text
    try testing.expect(!mediaIds(&.{ 1, picture_ids, 3 }, null, &.{})); // a picture id without its media
}
