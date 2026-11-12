//! The RDMA proxies' nap between exchanges (TF_DS_RDMA_WFE, rdma.zig's proxy loop). A proxy with nothing in flight
//! waits for its doorbell with the load-exclusive monitor armed on it and WFET: the core sleeps until the staging
//! kernel's write to the doorbell clears the monitor, or a short deadline on the virtual counter passes, instead of
//! reading the doorbell in a loop. Nothing waits on a napping proxy: every write it posted has completed, every peer's
//! part of every sequence it posted is published, and a peer's part of the next sequence that lands during the nap is
//! waited for only after this rank's own next doorbell, which ends the nap (the staging kernel rings it before the
//! collecting kernel waits on any flag). The deadline bounds a nap nothing ends: a ring parked or stopped.
const std = @import("std");
const builtin = @import("builtin");

/// A nap's longest length in microseconds, unless the ring's settings give another.
pub const default_us: u32 = 50;

/// AT_HWCAP2's HWCAP2_WFXT: the CPU has WFET and WFIT (FEAT_WFxT).
const hwcap2_wfxt: usize = 1 << 31;

const arm = builtin.cpu.arch == .aarch64;

/// Whether TF_DS_RDMA_WFE asks the proxies to nap: set and not "0" (an empty value is on, as the family's switches).
pub fn wanted() bool {
    const v = std.c.getenv("TF_DS_RDMA_WFE") orelse return wantedText(null);
    return wantedText(std.mem.span(v));
}

pub fn wantedText(text: ?[]const u8) bool {
    const t = text orelse return false;
    return !std.mem.eql(u8, t, "0");
}

/// The longest nap TF_DS_RDMA_NAP_US may set (1 s). start() caps the effective length at `teardown_us` (2.5 ms) in
/// production, so the knob's usable range there is 1..2,500 us; the tests' rings take their length as given.
pub const max_nap_us: u32 = 1_000_000;

/// A nap's longest length when the environment or the settings choose it, so a stop's join waits at most this long per
/// ring (2.5 ms: the three 2D rings stop within a few milliseconds). The tests' rings take their length as given.
pub const teardown_us: u32 = 2_500;

/// A nap's longest length from its text: a positive integer of microseconds, clamped to 1 s; `fallback` (the settings')
/// when unset, unreadable or 0 (a zero deadline is a hot spin, not a nap).
pub fn napUsFrom(text: ?[]const u8, fallback: u32) u32 {
    const t = text orelse return fallback;
    const v = std.fmt.parseInt(u32, std.mem.trim(u8, t, " \t"), 10) catch return fallback;
    if (v == 0) return fallback;
    return @min(v, max_nap_us);
}

/// The nap's longest length: TF_DS_RDMA_NAP_US when set, else `fallback` (Settings.nap_us, the baseline 50 us).
pub fn napUs(fallback: u32) u32 {
    const e = std.c.getenv("TF_DS_RDMA_NAP_US") orelse return napUsFrom(null, fallback);
    return napUsFrom(std.mem.span(e), fallback);
}

/// An entry of this process's auxiliary vector (0: none). Through libc's getauxval where libc is linked, as in the
/// engine's binaries: there libc's start code runs, not Zig's, and Zig's own reader finds no vector (it returns 0).
pub fn auxval(t: usize) usize {
    if (builtin.link_libc) return @intCast(std.c.getauxval(@intCast(t)));
    return std.os.linux.getauxval(t);
}

/// Whether a CPU can nap, from its auxiliary vector: WFET (FEAT_WFxT), none, or a vector that reads nothing (its page
/// size 0), which cannot tell.
pub const Cpu = enum { wfet, no_wfet, unread };

pub fn cpuFrom(pagesz: usize, hwcap2: usize) Cpu {
    if (pagesz == 0) return .unread;
    return if (hwcap2 & hwcap2_wfxt != 0) .wfet else .no_wfet;
}

/// This CPU's, on aarch64. Elsewhere only the host tests nap, on a read loop that ends the ways a nap ends.
pub fn cpu() Cpu {
    if (arm) return cpuFrom(auxval(std.elf.AT.PAGESZ), auxval(std.elf.AT.HWCAP2));
    return if (builtin.is_test) .wfet else .no_wfet;
}

pub fn available() bool {
    return cpu() == .wfet;
}

/// Whether an arm64 /proc/cpuinfo's Features list has `wfxt`: an authority independent of the auxiliary vector cpu()
/// reads, so a wrong key, bit or argument order in that read cannot agree with itself. Lines other than Features (and
/// the x86 cpuinfo, whose capability list is `flags`) are not it.
pub fn featuresWfxt(text: []const u8) bool {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |row| {
        const colon = std.mem.indexOfScalar(u8, row, ':') orelse continue;
        if (!std.mem.eql(u8, std.mem.trim(u8, row[0..colon], " \t"), "Features")) continue;
        var it = std.mem.tokenizeAny(u8, row[colon + 1 ..], " \t");
        while (it.next()) |f| if (std.mem.eql(u8, f, "wfxt")) return true;
    }
    return false;
}

/// /proc/cpuinfo's wfxt on this machine, when the file reads; null when it does not (a non-arm cpuinfo under
/// qemu-user has no Features line, so callers must not treat false there as "no WFET": use null for that).
pub fn cpuinfoWfxt(io: std.Io, gpa: std.mem.Allocator) ?bool {
    const text = std.Io.Dir.cwd().readFileAlloc(io, "/proc/cpuinfo", gpa, .limited(1 << 20)) catch return null;
    defer gpa.free(text);
    if (std.mem.indexOf(u8, text, "Features") == null) return null;
    return featuresWfxt(text);
}

/// Host tests only: how many times now() and cpuNow() have run. A proxy with TF_DS_RDMA_WFE unset must run its
/// pre-nap loop without a clock read, so the switch-off test asserts these deltas are 0. `builtin.is_test` is comptime
/// false in production, so the increments compile out there.
pub var now_calls: std.atomic.Value(u64) = .init(0);
pub var cpu_now_calls: std.atomic.Value(u64) = .init(0);

/// The count a nap's deadline is on: the virtual counter on aarch64 (the count WFET compares with), else the monotonic
/// clock in nanoseconds.
pub fn now() u64 {
    if (builtin.is_test) _ = now_calls.fetchAdd(1, .monotonic);
    if (arm) return asm volatile ("mrs %[v], cntvct_el0"
        : [v] "=r" (-> u64),
    );
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The calling thread's CPU time in nanoseconds (CLOCK_THREAD_CPUTIME_ID): a syscall (no vDSO path), so never inside a nap
/// loop. It cannot tell a nap from a read loop: WFET does not deschedule the thread, so Linux bills a sleeping nap as CPU
/// time; a nap is judged by its own counts.
pub fn cpuNow() u64 {
    if (builtin.is_test) _ = cpu_now_calls.fetchAdd(1, .monotonic);
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.THREAD_CPUTIME_ID, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The count's ticks a microsecond, from its frequency: 0 when the rate reads nothing (no trustworthy deadline). A
/// pure function of the frequency, so its divisor is tested against known rates (1 GHz is 1000 ticks/us), not against
/// itself.
pub fn ticksPerUsFrom(freq: u64) u64 {
    return (freq + std.time.us_per_s / 2) / std.time.us_per_s; // nearest tick, not always down
}

fn cntfrq() u64 {
    return asm volatile ("mrs %[f], cntfrq_el0"
        : [f] "=r" (-> u64),
    );
}

/// The count's ticks a microsecond (at least one; start() refuses to nap when the rate reads nothing, see Clock).
pub fn ticksPerUs() u64 {
    if (arm) return @max(1, ticksPerUsFrom(cntfrq()));
    return std.time.ns_per_us;
}

/// Whether the deadline's rate is readable. CNTFRQ_EL0 0 would make ticksPerUs 1 (a deadline already past, a hot
/// spin) and a wrong divisor a 1000x mis-sized nap; a ring whose counter reads nothing spins and says so.
pub const Clock = enum { fine, unread };

pub fn clockFrom(freq: u64) Clock {
    return if (freq == 0) .unread else .fine;
}

pub fn clock() Clock {
    if (arm) return clockFrom(cntfrq());
    return .fine;
}

/// One nap on `word` while it holds `seen`: the word loaded with the exclusive monitor armed on its granule (LDAXR)
/// and, when it still holds `seen`, WFET until the count reaches `deadline`. Nothing between the load and the WFET
/// touches memory (a store there could clear the monitor). The core wakes when another agent writes the word's
/// granule (the GPU's doorbell, or a word beside it), on an event or an interrupt, or at the deadline. False when the
/// word had moved before the nap; the monitor is cleared on that path alone, so an unpaired store-exclusive on this core
/// cannot fail once. Not after the WFET: there the next pass's load-exclusive arms the monitor again (with this
/// placement and the rest of the nap, the proxies napped 93.7-98.9 % of the time on four GB10 nodes, 16 streams
/// decoding). An event from
/// before the nap can end it early: the stale-event prologue bought only that, and its untimed WFE with the monitor
/// still unarmed could not be ended by the doorbell's store.
pub fn napOnce(word: *const u64, seen: u64, deadline: u64) bool {
    if (arm) {
        const v = asm volatile (
            \\ ldaxr %[v], [%[p]]
            \\ cmp %[v], %[seen]
            \\ b.ne 2f
            \\ msr s0_3_c1_c0_0, %[dl]
            \\ b 3f
            \\ 2: clrex
            \\ 3:
            : [v] "=&r" (-> u64),
            : [p] "r" (word),
              [seen] "r" (seen),
              [dl] "r" (deadline),
            : .{ .memory = true, .nzcv = true });
        return v == seen;
    }
    // the host tests' stand-in: reads until the word moves or the deadline passes, the two ways a nap ends
    if (@atomicLoad(u64, word, .acquire) != seen) return false;
    var beside_seen: [8]u64 = undefined;
    const nb = @min(stand_in_beside.len, beside_seen.len);
    for (stand_in_beside[0..nb], 0..) |w, i| beside_seen[i] = @atomicLoad(u64, w, .acquire);
    const t0 = now();
    while (@atomicLoad(u64, word, .acquire) == seen and now() < deadline) {
        // a stand-in for the event stream and interrupts, which end a real WFET nap early: the tests must tolerate them
        if (stand_in_wake_after != 0 and now() - t0 >= stand_in_wake_after) return true;
        // and for a write to another word of the doorbell's monitor granule (the GPU's sizes[] before its doorbell)
        for (stand_in_beside[0..nb], 0..) |w, i| if (@atomicLoad(u64, w, .acquire) != beside_seen[i]) return true;
        std.atomic.spinLoopHint();
    }
    return true;
}

/// Host tests only: after this many count ticks the stand-in ends the nap as an event or an interrupt does (0: never).
/// The target measures ~10 kHz of them (39 % of naps), so the proxy's tests run with the same wake mix rather than a
/// cleanness no CPU has. Written by the test before its proxy starts.
pub var stand_in_wake_after: u64 = 0;

/// Host tests only: words beside the doorbell whose change ends the stand-in's nap, as a store to any word of the
/// doorbell's monitor granule does on the target. Empty outside tests.
pub var stand_in_beside: []const *const u64 = &.{};

/// How a nap ended, from the word and the count when it returned.
pub const End = enum {
    skipped, // the word had moved before the nap began
    doorbell, // the word moved, before the deadline
    late, // the word moved, the count at or past the deadline (the deadline may have ended it, not the write)
    deadline, // the word held and the deadline passed
    other, // the word held before the deadline: an event or an interrupt
    neighbour, // the word held, but a word beside it (the GPU's sizes[] write before its doorbell) moved
};

pub fn classify(moved: bool, at: u64, deadline: u64) End {
    if (moved) return if (at >= deadline) .late else .doorbell;
    return if (at >= deadline) .deadline else .other;
}

/// classify(), with the doorbell's neighbour known: a wake that ended with the doorbell still holding but a word
/// beside it changed is the GPU's sizes[] write (fence mode 1, block 0, before the copy and the doorbell), not an
/// event or an interrupt.
pub fn classifyWith(moved: bool, at: u64, deadline: u64, neighbour: bool) End {
    const e = classify(moved, at, deadline);
    return if (e == .other and neighbour) .neighbour else e;
}

/// A proxy's naps since its ring started, as numbers: `naps` begun, each one's end, the count's ticks spent napping.
pub const Counts = struct {
    naps: u64 = 0,
    skipped: u64 = 0,
    doorbell: u64 = 0,
    late: u64 = 0,
    deadline: u64 = 0,
    other: u64 = 0,
    neighbour: u64 = 0,
    ticks: u64 = 0,

    /// The naps that have ended (one fewer than begun while a nap is on).
    pub fn ended(c: Counts) u64 {
        return c.skipped + c.doorbell + c.late + c.deadline + c.other + c.neighbour;
    }

    /// The naps since `base`, an earlier reading of the same proxy.
    pub fn since(c: Counts, base: Counts) Counts {
        var d: Counts = .{};
        inline for (@typeInfo(Counts).@"struct".field_names) |n| @field(d, n) = @field(c, n) -% @field(base, n);
        return d;
    }
};

/// Host tests only: runs between the reads `counts` makes (null in production), so a reader can be preempted at the
/// boundary the invariant depends on. Written by the test before its reader starts.
pub var counts_between: ?*const fn () void = null;

/// A proxy's naps since its ring started: written by the proxy thread alone (a load and a store, no read-modify-write),
/// read by anyone.
pub const Stats = struct {
    naps: std.atomic.Value(u64) = .init(0),
    skipped: std.atomic.Value(u64) = .init(0),
    doorbell: std.atomic.Value(u64) = .init(0),
    late: std.atomic.Value(u64) = .init(0),
    deadline: std.atomic.Value(u64) = .init(0),
    other: std.atomic.Value(u64) = .init(0),
    neighbour: std.atomic.Value(u64) = .init(0),
    ticks: std.atomic.Value(u64) = .init(0),

    /// A nap begins.
    pub fn begin(s: *Stats) void {
        add(&s.naps, 1);
    }

    /// The nap begun last ended as `how`, after `ticks` of the count.
    pub fn end(s: *Stats, how: End, ticks: u64) void {
        add(switch (how) {
            .skipped => &s.skipped,
            .doorbell => &s.doorbell,
            .late => &s.late,
            .deadline => &s.deadline,
            .other => &s.other,
            .neighbour => &s.neighbour,
        }, 1);
        add(&s.ticks, ticks);
    }

    /// A reading of every counter. The causes and ticks are read first and `naps` last, so a nap that begins or ends
    /// between the reads still leaves `naps - 1 <= ended <= naps`: reading `naps` first could show more ends than
    /// begins. The stores are release and the loads acquire, so an end seen implies its begin is seen (Arm's weak
    /// ordering would otherwise allow the end's store to land first in a reader).
    pub fn counts(s: *const Stats) Counts {
        var c: Counts = .{};
        inline for (.{ "skipped", "doorbell", "late", "deadline", "other", "neighbour", "ticks" }) |n| @field(c, n) = @field(s, n).load(.acquire);
        if (counts_between) |f| f();
        c.naps = s.naps.load(.acquire);
        return c;
    }

    fn add(v: *std.atomic.Value(u64), n: u64) void {
        v.store(v.load(.monotonic) +% n, .release);
    }
};

/// Whether a proxy may nap: every sequence the doorbell rang (`rung`) posted, none of its writes still to complete,
/// and every peer's part of every posted sequence published (`published[p]` for each rank p but this one, `rank`).
/// Whatever the completion queues still hold then is a peer's part of a later sequence: a peer already a sequence
/// ahead does not hold a nap off, as this rank waits for that part only after its own next doorbell.
pub fn quiet(posted: u64, rung: u64, inflight: u64, published: []const u64, rank: usize) bool {
    if (posted < rung or inflight != 0) return false;
    for (published, 0..) |q, p| if (p != rank and q < posted) return false;
    return true;
}

/// The line a proxy logs when its ring parks: its naps `c` over the `elapsed` ticks since its last wake (`per_us`
/// ticks a microsecond; `cpu_elapsed` nanoseconds on the CPU, which a truly sleeping nap does not spend). The doorbell
/// bucket says only what is known: the doorbell had moved by the time the nap returned before its deadline (another
/// wake may have ended it first, with the GPU's write having landed just before).
/// The name a line prints: caller text is truncated to a fixed prefix so a long name cannot fill the buffer and drop
/// the whole line, leaving the operator with an empty string.
fn shownName(name: []const u8, max: usize) []const u8 {
    return name[0..@min(name.len, max)];
}

pub fn line(buf: []u8, name: []const u8, c: Counts, elapsed: u64, per_us: u64, cpu_elapsed: u64) []const u8 {
    const nm = shownName(name, 128);
    const secs = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(@max(1, per_us))) / 1e6;
    const awake = @as(f64, @floatFromInt(elapsed -| c.ticks)) / @as(f64, @floatFromInt(@max(1, per_us))) / 1e6;
    const on_cpu = @as(f64, @floatFromInt(cpu_elapsed)) / 1e9;
    const share = if (elapsed == 0) 0 else 100 * @as(f64, @floatFromInt(c.ticks)) / @as(f64, @floatFromInt(elapsed));
    return std.fmt.bufPrint(buf, "RDMA proxy naps on {s} over the {d:.3} s since its wake ({d:.3} s awake, {d:.3} s on CPU): {d:.1} % of the time in {d} naps; ended with the doorbell moved before the deadline {d} ({d} at or past it), by the deadline {d}, by another wake {d} ({d} beside the doorbell), skipped {d} (the doorbell had rung)", .{ nm, secs, awake, on_cpu, share, c.naps, c.doorbell + c.late, c.late, c.deadline, c.other + c.neighbour, c.neighbour, c.skipped }) catch buf[0..0];
}

/// Whether a ring's nap meets what the switch asked: when TF_DS_RDMA_WFE is set, a napping ring that has begun naps and
/// slept, and whose naps are not more than a quarter ending at the deadline with the doorbell moved (
/// the GPU store not clearing the monitor would show as a late share above a quarter); nothing to check when it is
/// not. `how` is the decision start() made: a ring that correctly spins because its routes relay or its CPU cannot nap
/// is not judged; only a ring whose start said it naps must have napped. tf-dsv41-ring2d-check adds this to its exit
/// code.
pub fn napOk(asked: bool, how: ?Start, c: ?Counts) bool {
    if (!asked) return true;
    const h = how orelse return false; // asked, but the ring made no decision: not a nap to judge
    return switch (h) {
        .naps => {
            const x = c orelse return false;
            return x.naps > 0 and x.ticks > 0 and lateOk(asked, c);
        },
        // a ring whose start said it spins for a reason that is not a defect: relays, no WFET, an unread auxv, no clock
        .relays, .no_wfet, .unread, .no_clock => true,
        // it was asked to nap, did, and then stopped because its wakes kept missing the deadline: a broken monitor
        .backoff => false,
    };
}

/// The ring check's per-case fold: whether the cases so far all matched and this one did too. `bad` is the words the
/// case's gather mismatched; null is a case that returned an error and did not run, which fails the check as a wrong
/// gather does. The main applies it to every case (`all_ok = gatherOk(all_ok, bad)`) and on its error path
/// (`all_ok = gatherOk(all_ok, null)`); a test calls it, so a fold `and` -> `or` fails there.
pub fn gatherOk(so_far: bool, bad: ?usize) bool {
    if (!so_far) return false;
    const n = bad orelse return false;
    return n == 0;
}

/// The ring check's exit code: 0 only when every gather matched and every ring's nap met the switch's request.
pub fn checkExit(all_equal: bool, nap_ok: bool) u8 {
    return if (all_equal and nap_ok) 0 else 1;
}

/// One ring as the ring check reads it: the Start decision start() made, the counts it read, and why its proxy
/// stopped (`Ring.failure()`: null when it did not fail).
pub const RingNap = struct { how: ?Start, c: ?Counts, failed: ?[]const u8 = null };

/// Whether every ring's nap met the switch's request: the ring check's per-ring fold. A ring that was asked to nap
/// and did not fails the check as a wrong gather does, and so does a ring whose proxy failed after the last gather
/// (whatever the switch asked: a failed run is a failed run).
pub fn ringsOk(asked: bool, rings: []const RingNap) bool {
    for (rings) |r| {
        if (r.failed != null) return false;
        if (!napOk(asked, r.how, r.c)) return false;
    }
    return true;
}

/// The ring check's exit code from its two folds: 0 only when every gather matched and every ring's nap met the
/// switch's request. The main returns this; a test calls it, so a `return 0` mutant fails.
pub fn runCheck(all_equal: bool, asked: bool, rings: []const RingNap) u8 {
    return checkExit(all_equal, ringsOk(asked, rings));
}

/// Whether a napping ring's naps are being ended promptly where gathers run: a nap that ends `.late` means the
/// doorbell moved, but the count was at or past the deadline — the nap timed out first. A ring whose naps are more
/// than a quarter late is one whose GPU store does not clear this core's global monitor, where every exchange waits
/// out the nap. The ring check warns and fails on it, and the proxy also stops napping after such a
/// window (`Ring.backOff`), so a broken monitor costs awake time rather than a silently bad gate; the switch stays the
/// operator's control, and the next start() decides again. Fewer than 16 naps is too few to judge, and a ring whose
/// switch did not ask is not judged.
pub fn lateOk(asked: bool, c: ?Counts) bool {
    if (!asked) return true;
    const x = c orelse return true;
    if (x.naps < 16) return true;
    return x.late * 4 <= x.naps; // at most a quarter of the naps ended late
}

/// One ring's failure line in tf-dsv41-ring2d-check's JSON: why its proxy stopped (`Ring.failure()`'s reason), printed
/// beside the ring's nap line when the failure came after the last gather. The name is truncated as every line here.
pub fn failureLine(buf: []u8, rank: u32, name: []const u8, why: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{{\"rank\": {d}, \"ring\": \"{s}\", \"failure\": \"{s}\"}}", .{ rank, shownName(name, 128), why }) catch buf[0..0];
}

/// One ring's line in tf-dsv41-ring2d-check's JSON: each count by name, the interval since the proxy last woke in
/// milliseconds (elapsed, awake) and the cumulative time napped, its thread CPU time, and whether the switch's
/// request is met. `elapsed` and the counts are count ticks (`per_us` a microsecond); `cpu_elapsed` is nanoseconds. A
/// ring that does not nap says why in `"why"`: its Start decision (`"relays"`, `"no_wfet"`, `"unread"`,
/// `"no_clock"`, `"backoff"`, or `"naps"` for a ring that was to nap but is not).
pub fn jsonLine(buf: []u8, rank: u32, name: []const u8, how: ?Start, c: ?Counts, elapsed: u64, per_us: u64, cpu_elapsed: u64, asked: bool) []const u8 {
    const nm = shownName(name, 128);
    const us = @as(f64, @floatFromInt(@max(1, per_us)));
    const ok: []const u8 = if (napOk(asked, how, c)) "true" else "false";
    const x = c orelse {
        const h = how orelse return std.fmt.bufPrint(buf, "{{\"rank\": {d}, \"ring\": \"{s}\", \"rdma_wfe\": false, \"rdma_wfe_ok\": {s}}}", .{ rank, nm, ok }) catch buf[0..0];
        return std.fmt.bufPrint(buf, "{{\"rank\": {d}, \"ring\": \"{s}\", \"rdma_wfe\": false, \"rdma_wfe_ok\": {s}, \"why\": \"{s}\"}}", .{ rank, nm, ok, @tagName(h) }) catch buf[0..0];
    };
    const napped_ms = @as(f64, @floatFromInt(x.ticks)) / us / 1e3;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed)) / us / 1e3;
    const awake_ms = @as(f64, @floatFromInt(elapsed -| x.ticks)) / us / 1e3;
    const cpu_ms = @as(f64, @floatFromInt(cpu_elapsed)) / 1e6;
    return std.fmt.bufPrint(buf, "{{\"rank\": {d}, \"ring\": \"{s}\", \"rdma_wfe\": true, \"rdma_wfe_ok\": {s}, \"naps\": {d}, \"doorbell\": {d}, \"late\": {d}, \"deadline\": {d}, \"other\": {d}, \"neighbour\": {d}, \"skipped\": {d}, \"napped_ms\": {d:.3}, \"elapsed_ms\": {d:.3}, \"awake_ms\": {d:.3}, \"cpu_ms\": {d:.3}}}", .{ rank, nm, ok, x.naps, x.doorbell, x.late, x.deadline, x.other, x.neighbour, x.skipped, napped_ms, elapsed_ms, awake_ms, cpu_ms }) catch buf[0..0];
}

/// What a ring whose proxy TF_DS_RDMA_WFE asked to nap does: it naps, or why it spins. `.backoff` is a ring that
/// napped and then stopped because its wakes kept arriving at or past the deadline (the GPU store is not clearing the
/// core's monitor): it spins from then on, and the ring check refuses it.
pub const Start = enum { naps, relays, no_wfet, unread, no_clock, backoff };

/// The naps a proxy judges its late share over: the last `late_window` naps, whatever ended them. A sliding window,
/// not a run: an idle deadline nap or another wake is not evidence that the monitor clears, so it neither resets the
/// window nor is it ignored. 64 naps at the measured 10 kHz event mix (60.8 % late) is over a quarter late; a healthy
/// ring's false-late share is ~1 %.
pub const late_window: u32 = 64;

/// A proxy's last `late_window` naps, one bit each (1: ended `.late`), and how many of them are late. The window
/// slides one nap at a time, so a broken monitor is judged within `late_window` naps of its first late end.
pub const LateWindow = struct {
    bits: u64 = 0, // the last 64 naps, newest in bit 0
    late: u32 = 0, // the 1 bits in `bits`
    naps: u32 = 0, // how many naps the window holds (up to late_window)

    /// One nap ended: `late` or not. Returns whether the window is full and more than a quarter of it is late.
    pub fn add(w: *LateWindow, late: bool) bool {
        if (w.naps == late_window) {
            if (w.bits >> 63 != 0) w.late -= 1; // the oldest nap leaves the window
        } else w.naps += 1;
        w.bits = (w.bits << 1) | @as(u64, @intFromBool(late));
        if (late) w.late += 1;
        return w.over();
    }

    /// Whether the window is full and more than a quarter of its naps ended `.late`: the monitor is not clearing.
    pub fn over(w: LateWindow) bool {
        return w.naps == late_window and w.late * 4 > w.naps;
    }
};

/// Host tests only: overrides the end classifyWith() would give, so a proxy's ends can be driven to `.late` without a
/// CPU whose monitor behaves that way. Null in production.
pub var end_for_test: ?End = null;

/// A ring naps where its routes have no relays, its CPU has WFET, and its counter rate is readable.
pub fn startOf(relays: bool, c: Cpu, clk: Clock) Start {
    if (relays) return .relays;
    return switch (c) {
        .wfet => switch (clk) {
            .fine => .naps,
            .unread => .no_clock,
        },
        .no_wfet => .no_wfet,
        .unread => .unread,
    };
}

/// The start line of a ring whose proxy TF_DS_RDMA_WFE asked to nap.
pub fn startLine(buf: []u8, name: []const u8, how: Start, us: u32) []const u8 {
    const nm = shownName(name, 80);
    const t = switch (how) {
        .naps => std.fmt.bufPrint(buf, "TF_DS_RDMA_WFE: {s}'s RDMA proxy naps on its doorbell between exchanges (WFET, at most {d} us a nap)", .{ nm, us }),
        .relays => std.fmt.bufPrint(buf, "TF_DS_RDMA_WFE: {s} relays (a ring without a switch), so its RDMA proxy spins", .{nm}),
        .no_wfet => std.fmt.bufPrint(buf, "TF_DS_RDMA_WFE: this CPU has no WFET (HWCAP2_WFXT), so {s}'s RDMA proxy spins", .{nm}),
        .unread => std.fmt.bufPrint(buf, "TF_DS_RDMA_WFE: this process reads no auxiliary vector (AT_PAGESZ 0), so it cannot tell whether the CPU has WFET: {s}'s RDMA proxy spins", .{nm}),
        .no_clock => std.fmt.bufPrint(buf, "TF_DS_RDMA_WFE: this CPU's counter rate reads 0 (CNTFRQ_EL0), so {s}'s RDMA proxy spins", .{nm}),
        .backoff => std.fmt.bufPrint(buf, "TF_DS_RDMA_WFE: {s}'s RDMA proxy stopped napping: its doorbell's stores keep arriving at or past the deadline, so it spins from now on", .{nm}),
    };
    return t catch buf[0..0];
}

test "the start line says whether a proxy naps, and why not" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("TF_DS_RDMA_WFE: the column pair's ring's RDMA proxy naps on its doorbell between exchanges (WFET, at most 50 us a nap)", startLine(&buf, "the column pair's ring", .naps, 50));
    try std.testing.expectEqualStrings("TF_DS_RDMA_WFE: the four nodes' ring relays (a ring without a switch), so its RDMA proxy spins", startLine(&buf, "the four nodes' ring", .relays, 50));
    try std.testing.expectEqualStrings("TF_DS_RDMA_WFE: this CPU has no WFET (HWCAP2_WFXT), so the row pair's ring's RDMA proxy spins", startLine(&buf, "the row pair's ring", .no_wfet, 50));
    try std.testing.expectEqualStrings("TF_DS_RDMA_WFE: this process reads no auxiliary vector (AT_PAGESZ 0), so it cannot tell whether the CPU has WFET: the four nodes' ring's RDMA proxy spins", startLine(&buf, "the four nodes' ring", .unread, 50));
    try std.testing.expectEqualStrings("TF_DS_RDMA_WFE: this CPU's counter rate reads 0 (CNTFRQ_EL0), so the row pair's ring's RDMA proxy spins", startLine(&buf, "the row pair's ring", .no_clock, 50));
    try std.testing.expectEqualStrings("TF_DS_RDMA_WFE: the row pair's ring's RDMA proxy stopped napping: its doorbell's stores keep arriving at or past the deadline, so it spins from now on", startLine(&buf, "the row pair's ring", .backoff, 50));
    // a long caller name is truncated, not dropped along with the whole line
    const long: [400]u8 = @splat('x');
    try std.testing.expect(startLine(&buf, &long, .naps, 50).len > 0);
}

test "the late window backs off at the measured 10 kHz event mix" {
    // the measured mix: 60.8 % late, 17.9 % doorbell, 21.3 % other. A run of 64 consecutive late naps
    // never happens (0.608^64 = 1.5e-14); the window must judge the share, not a run.
    var w: LateWindow = .{};
    var backed_off = false;
    for (0..1000) |i| {
        // 608 late spread evenly over 1000 naps; the rest are doorbell/other ends, both non-late
        const late = (i * 608) / 1000 != ((i + 1) * 608) / 1000;
        if (w.add(late)) backed_off = true;
    }
    try std.testing.expect(backed_off);
    // the same counts fail the ring check
    try std.testing.expect(!lateOk(true, .{ .naps = 1000, .late = 608, .doorbell = 179, .other = 213, .ticks = 1_000_000 }));
    // an idle deadline nap is not evidence the monitor clears: 16 late and 48 deadline is a quarter, ok; 17 late
    // and 47 deadline is more than a quarter, refused.
    var q: LateWindow = .{};
    for (0..16) |_| _ = q.add(true);
    for (0..48) |_| _ = q.add(false);
    try std.testing.expect(!q.over());
    var r: LateWindow = .{};
    for (0..17) |_| _ = r.add(true);
    for (0..47) |_| _ = r.add(false);
    try std.testing.expect(r.over());
    // the window slides: 17 late naps, then 64 non-late ones push every bit out, so the last 64 naps are not late. A
    // window that never forgets (add()'s full branch without the decrement at `if (w.bits >> 63 != 0) w.late -= 1`)
    // keeps late == 17 over 64 naps, and 17 * 4 > 64 is over: this case fails under that mutant.
    var v: LateWindow = .{};
    for (0..17) |_| _ = v.add(true);
    for (0..64) |_| _ = v.add(false);
    try std.testing.expectEqual(@as(u32, 0), v.late);
    try std.testing.expectEqual(late_window, v.naps);
    try std.testing.expect(!v.over());
    // a run of 63 late naps is not enough to judge: the window is not full
    var s: LateWindow = .{};
    for (0..63) |_| try std.testing.expect(!s.add(true));
    try std.testing.expect(s.add(true)); // the 64th late nap fills the window: over
}

test "TF_DS_RDMA_WFE: set and not 0" {
    try std.testing.expect(!wantedText(null));
    try std.testing.expect(!wantedText("0"));
    try std.testing.expect(wantedText("1"));
    try std.testing.expect(wantedText(""));
    try std.testing.expect(wantedText("yes"));
}

test "TF_DS_RDMA_NAP_US: a positive length, clamped, else the setting" {
    try std.testing.expectEqual(@as(u32, 50), napUsFrom(null, 50));
    try std.testing.expectEqual(@as(u32, 200), napUsFrom("200", 50));
    try std.testing.expectEqual(@as(u32, 7), napUsFrom(" 7\t", 50));
    try std.testing.expectEqual(@as(u32, 50), napUsFrom("0", 50));
    try std.testing.expectEqual(@as(u32, 50), napUsFrom("-1", 50));
    try std.testing.expectEqual(@as(u32, 50), napUsFrom("x", 50));
    try std.testing.expectEqual(@as(u32, 50), napUsFrom("", 50));
    try std.testing.expectEqual(max_nap_us, napUsFrom("99999999", 50));
}

test "a proxy naps only with nothing in flight" {
    // rank 2 of four, sequence 5 posted and every peer's part of it in
    try std.testing.expect(quiet(5, 5, 0, &.{ 5, 5, 0, 5 }, 2));
    // the doorbell rang a sequence not yet posted
    try std.testing.expect(!quiet(5, 6, 0, &.{ 5, 5, 0, 5 }, 2));
    // a write still to complete
    try std.testing.expect(!quiet(5, 5, 1, &.{ 5, 5, 0, 5 }, 2));
    // a peer's part of sequence 5 not yet in
    try std.testing.expect(!quiet(5, 5, 0, &.{ 5, 4, 0, 5 }, 2));
    try std.testing.expect(!quiet(5, 5, 0, &.{ 4, 5, 0, 5 }, 2));
    try std.testing.expect(!quiet(5, 5, 0, &.{ 5, 5, 0, 4 }, 2));
    // a peer a sequence ahead: its part of 6 landed before this rank's doorbell of 6
    try std.testing.expect(quiet(5, 5, 0, &.{ 6, 5, 0, 6 }, 2));
    // this rank's own entry is not a peer's
    try std.testing.expect(quiet(5, 5, 0, &.{ 5, 5, 9, 5 }, 2));
    // before the first doorbell
    try std.testing.expect(quiet(0, 0, 0, &.{ 0, 0 }, 0));
}

test "how a nap ended" {
    try std.testing.expectEqual(End.doorbell, classify(true, 99, 100));
    try std.testing.expectEqual(End.late, classify(true, 100, 100));
    try std.testing.expectEqual(End.late, classify(true, 250, 100));
    try std.testing.expectEqual(End.deadline, classify(false, 100, 100));
    try std.testing.expectEqual(End.deadline, classify(false, 101, 100));
    try std.testing.expectEqual(End.other, classify(false, 99, 100));
    // a word beside the doorbell moved (the GPU's sizes[] write): not the doorbell's, not an event's
    try std.testing.expectEqual(End.neighbour, classifyWith(false, 99, 100, true));
    try std.testing.expectEqual(End.doorbell, classifyWith(true, 99, 100, true)); // the doorbell moved: not a neighbour
    try std.testing.expectEqual(End.deadline, classifyWith(false, 100, 100, true));
    try std.testing.expectEqual(End.other, classifyWith(false, 99, 100, false));
}

test "a proxy's counts: naps begun, each one's end, the ticks napped, and the counts since an earlier reading" {
    var s: Stats = .{};
    s.begin();
    try std.testing.expectEqual(@as(u64, 1), s.counts().naps);
    try std.testing.expectEqual(@as(u64, 0), s.counts().ended());
    s.end(.doorbell, 40);
    const base = s.counts();
    // one end of each kind, the last a neighbour's (a word beside the doorbell moved): ended() must count it, or the
    // invariant below and the exact counts fail.
    for ([_]End{ .skipped, .doorbell, .late, .deadline, .other, .neighbour }, [_]u64{ 0, 10, 50, 50, 3, 4 }) |e, t| {
        s.begin();
        s.end(e, t);
    }
    s.begin(); // one on
    const c = s.counts();
    try std.testing.expectEqual(Counts{ .naps = 8, .skipped = 1, .doorbell = 2, .late = 1, .deadline = 1, .other = 1, .neighbour = 1, .ticks = 157 }, c);
    try std.testing.expectEqual(c.naps - 1, c.ended());
    try std.testing.expectEqual(Counts{ .naps = 7, .skipped = 1, .doorbell = 1, .late = 1, .deadline = 1, .other = 1, .neighbour = 1, .ticks = 117 }, c.since(base));
}

test "counts reads never show a nap that ended before it began" {
    // The invariant counts documents: while a nap is on, one fewer nap has ended than begun. A reader preempted between
    // the first and the last load must not see more ends than begins. The hook below stands in for that preemption and
    // lets the writer finish exactly one begin+end between the two groups of loads.
    const Sync = struct {
        var ready = std.atomic.Value(bool).init(false);
        var go = std.atomic.Value(bool).init(false);
        var stop = std.atomic.Value(bool).init(false);
        fn writer(st: *Stats) void {
            while (!stop.load(.acquire)) {
                while (!ready.load(.acquire)) {
                    if (stop.load(.acquire)) return;
                    std.Thread.yield() catch {};
                }
                st.begin();
                st.end(.doorbell, 1);
                go.store(true, .release);
                // (stop too: a writer left waiting here when the reads end would hold the join forever)
                while (go.load(.acquire)) {
                    if (stop.load(.acquire)) return;
                    std.Thread.yield() catch {};
                }
            }
        }
        fn hook() void {
            ready.store(true, .release);
            while (!go.load(.acquire)) std.Thread.yield() catch {};
            // ready first: a writer released by go must find ready cleared, or it runs one more begin+end unasked and leaves
            // go set behind the last read
            ready.store(false, .release);
            go.store(false, .release);
        }
    };
    var s: Stats = .{};
    const t = try std.Thread.spawn(.{}, Sync.writer, .{&s});
    defer {
        Sync.stop.store(true, .release);
        t.join();
    }
    counts_between = Sync.hook;
    defer counts_between = null;
    for (0..1000) |_| {
        const c = s.counts();
        try std.testing.expect(c.ended() <= c.naps);
        try std.testing.expect(c.naps - c.ended() <= 1);
    }
}

test "the line a proxy logs when its ring parks" {
    var buf: [512]u8 = undefined;
    const c: Counts = .{ .naps = 1004, .skipped = 2, .doorbell = 950, .late = 3, .deadline = 40, .other = 9, .ticks = 1_624_000_000 };
    const t = line(&buf, "the column pair's ring", c, 2_000_000_000, 1000, 120_000_000);
    try std.testing.expectEqualStrings("RDMA proxy naps on the column pair's ring over the 2.000 s since its wake (0.376 s awake, 0.120 s on CPU): 81.2 % of the time in 1004 naps; ended with the doorbell moved before the deadline 953 (3 at or past it), by the deadline 40, by another wake 9 (0 beside the doorbell), skipped 2 (the doorbell had rung)", t);
    // nothing napped: no division by zero
    try std.testing.expect(std.mem.indexOf(u8, line(&buf, "the four nodes' ring", .{}, 0, 1000, 0), "0.0 % of the time in 0 naps") != null);
    // a long caller name is truncated, not dropped along with the whole line
    const long: [400]u8 = @splat('x');
    try std.testing.expect(line(&buf, &long, .{}, 0, 1000, 0).len > 0);
}

test "a WFET nap sleeps: over 5 ms the nap loop goes round at the wake rate, not the read rate" {
    // gated on the hardware capability, not the test build: the host's stand-in reads in a loop by design and qemu's
    // WFET is a no-op, so only a real WFET can show the core sleeps.
    // The instrument is the nap loop's own pass count, never the thread's CPU time: WFET does not deschedule the thread, so
    // Linux bills a sleeping nap as CPU time (ENGINE-FACTS: "judge a nap by its own report ... never by CPU %"; this test
    // asserted CPU < 20 % of wall and failed on every GB10). A real nap ends at the event stream or an interrupt (~11 K
    // wakes a second measured on a GB10): ~55 passes in 5 ms. A nap that returns at once (a no-op WFET, or an event
    // always pending, as when the loop entered the kernel: 2.6 M passes a second) goes round >= 13,000 times.
    // Nothing in the loop enters the kernel: now() reads the count register.
    if (builtin.cpu.arch != .aarch64 or cpu() != .wfet) return error.SkipZigTest;
    var word: u64 = 7;
    const w0 = now();
    const deadline = w0 + 5_000 * ticksPerUs(); // 5 ms
    var passes: u64 = 0;
    while (now() < deadline) : (passes += 1) try std.testing.expect(napOnce(&word, 7, deadline));
    const wall_us = (now() - w0) / @max(1, ticksPerUs());
    try std.testing.expect(wall_us >= 5_000);
    // a mean nap of at least 20 us (<= 250 passes in 5 ms): 4.5x the measured wake rate's ~55, 50x under a nap that does not sleep
    if (passes > wall_us / 20) {
        std.debug.print("WFET nap loop: {d} passes in {d} us (mean nap {d} ns): the core is not sleeping\n", .{ passes, wall_us, wall_us * 1000 / @max(1, passes) });
        return error.TestUnexpectedResult;
    }
}

test "the nap test's bound: a sleeping nap's pass count is under it, a nap that returns at once is far over it" {
    // the bound above, on the numbers it was set from (pure arithmetic, every platform): 11 K wakes a second, 2.6 M a second
    const wall_us: u64 = 5_000;
    try std.testing.expect(11_000 * wall_us / 1_000_000 <= wall_us / 20); // ~55 passes: under 250
    try std.testing.expect(2_600_000 * wall_us / 1_000_000 > wall_us / 20); // ~13,000 passes: over 250
}

test "a nap on a word that has moved does not begin" {
    // the WFET instruction is undefined without the capability: skip where the CPU cannot nap, as the ring does
    if (!available()) return error.SkipZigTest;
    var word: u64 = 7;
    try std.testing.expect(!napOnce(&word, 6, now() + 1000 * ticksPerUs()));
}

test "a nap on a word nothing writes ends by the deadline" {
    // the WFET instruction is undefined without the capability: skip where the CPU cannot nap, as the ring does
    if (!available()) return error.SkipZigTest;
    var word: u64 = 7;
    const t0 = now();
    const deadline = t0 + 2000 * ticksPerUs(); // 2 ms
    var end: End = .other;
    var naps: usize = 0;
    while (end == .other) : (naps += 1) {
        // (each WFET may end early, on an event or an interrupt; a proxy naps again then, as here)
        if (!napOnce(&word, 7, deadline)) return error.TestUnexpectedResult;
        end = classify(@atomicLoad(u64, &word, .acquire) != 7, now(), deadline);
    }
    try std.testing.expectEqual(End.deadline, end);
    try std.testing.expect(naps >= 1);
    try std.testing.expect(now() - t0 >= 2000 * ticksPerUs());
}

test "another thread's write to the word ends a nap before its deadline" {
    // the WFET instruction is undefined without the capability: skip where the CPU cannot nap, as the ring does
    if (!available()) return error.SkipZigTest;
    const Writer = struct {
        fn run(w: *u64) void {
            std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
            @atomicStore(u64, w, 8, .release);
        }
    };
    var word: u64 = 7;
    const t0 = now();
    const deadline = t0 + 5_000_000 * ticksPerUs(); // 5 s
    const th = try std.Thread.spawn(.{}, Writer.run, .{&word});
    defer th.join();
    var end: End = .other;
    while (end == .other) {
        if (!napOnce(&word, 7, deadline)) {
            end = .skipped; // the write came between two naps
            break;
        }
        end = classify(@atomicLoad(u64, &word, .acquire) != 7, now(), deadline);
    }
    try std.testing.expect(end == .doorbell or end == .skipped);
    try std.testing.expectEqual(@as(u64, 8), @atomicLoad(u64, &word, .acquire));
    try std.testing.expect(now() < deadline);
}

test "the auxiliary vector reads as the engine's binaries read it: AT_PAGESZ is this process's page size" {
    // the family's tests link libc, as tensorfold-native and tf-dsv41-lanes do (CUDA): the start code is libc's
    try std.testing.expectEqual(std.heap.pageSize(), auxval(std.elf.AT.PAGESZ));
    try std.testing.expect(auxval(std.elf.AT.HWCAP) != 0); // every Linux CPU reports some capability
    if (arm) try std.testing.expect(cpu() != .unread) else try std.testing.expectEqual(Cpu.wfet, cpu()); // (the stand-in)
    try std.testing.expect(ticksPerUs() >= 1);
    const a = now();
    try std.testing.expect(now() >= a);
}

test "a ring naps with routes without relays on a CPU with WFET, and says why it spins otherwise" {
    try std.testing.expectEqual(Start.naps, startOf(false, .wfet, .fine));
    try std.testing.expectEqual(Start.relays, startOf(true, .wfet, .fine));
    try std.testing.expectEqual(Start.relays, startOf(true, .unread, .fine));
    try std.testing.expectEqual(Start.no_wfet, startOf(false, .no_wfet, .fine));
    try std.testing.expectEqual(Start.unread, startOf(false, .unread, .fine));
}

test "ticksPerUs pins the counter's rate against known frequencies, not against itself" {
    // 1 GHz and 24 MHz are the rates GB10-class counters report; a / 1000, * us_per_s or / 1e9 divisor fails here
    try std.testing.expectEqual(@as(u64, 1000), ticksPerUsFrom(1_000_000_000));
    try std.testing.expectEqual(@as(u64, 24), ticksPerUsFrom(24_000_000));
    try std.testing.expectEqual(@as(u64, 1), ticksPerUsFrom(1_000_000));
    try std.testing.expectEqual(@as(u64, 0), ticksPerUsFrom(0)); // a rate that reads nothing: start() must not nap
    try std.testing.expectEqual(@as(u64, 1), ticksPerUsFrom(999_999)); // rounds up to the nearest tick
    // a fractional-MHz rate is rounded to the nearest tick, not always down: 24.5 MHz is 25 ticks/us (the mutant
    // that truncates gives 24 and fails here)
    try std.testing.expectEqual(@as(u64, 25), ticksPerUsFrom(24_500_000));
    try std.testing.expectEqual(@as(u64, 24), ticksPerUsFrom(24_499_999));
    try std.testing.expect(ticksPerUs() >= 1);
}

test "now()'s rate is the wall clock's, and ticksPerUs converts between them" {
    var ts: std.os.linux.timespec = undefined;
    const wall = struct {
        fn ns(t: *std.os.linux.timespec) u64 {
            return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
        }
    }.ns;
    const n0 = now();
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    const w0 = wall(&ts);
    std.Io.sleep(std.testing.io, .fromMilliseconds(10), .awake) catch {};
    const n1 = now();
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    const w1 = wall(&ts);
    // a tick is 1000 / ticksPerUs nanoseconds: the same interval in both counts, within 20 % (under qemu's and the
    // target's counters alike; on the x86 stand-in they are the same clock)
    const want = (w1 - w0) * ticksPerUs() / std.time.ns_per_us;
    const got = n1 - n0;
    try std.testing.expect(got + got / 5 >= want and want + want / 5 >= got);
}

test "a CPU whose counter rate reads 0 does not nap" {
    try std.testing.expectEqual(Clock.fine, clockFrom(1_000_000_000));
    try std.testing.expectEqual(Clock.unread, clockFrom(0));
    try std.testing.expectEqual(Start.naps, startOf(false, .wfet, .fine));
    try std.testing.expectEqual(Start.no_clock, startOf(false, .wfet, .unread));
    try std.testing.expectEqual(Start.relays, startOf(true, .wfet, .unread)); // relays first: it says why it relays
    try std.testing.expectEqual(Start.no_wfet, startOf(false, .no_wfet, .unread));
}

test "the doorbell word's WFXt bit is HWCAP2's, not HWCAP's PACG bit" {
    // Zig's std names both (os/linux.zig, arm64): HWCAP.PACG and HWCAP2.WFXT are both bit 31. A key mix-up in cpu()
    // would therefore report WFET on every pointer-authentication CPU and execute an UNDEFINED WFET; this pins the
    // constants, and the /proc/cpuinfo test above pins the read. (arm-only: the x86 HWCAP has no @"2".)
    if (comptime arm) {
        try std.testing.expectEqual(std.os.linux.HWCAP.@"2".WFXT, hwcap2_wfxt);
        try std.testing.expectEqual(std.os.linux.HWCAP.PACG, hwcap2_wfxt);
    } else return error.SkipZigTest;
}

test "a CPU naps with WFET; without it, or with a vector that reads nothing, it spins" {
    try std.testing.expectEqual(Cpu.wfet, cpuFrom(4096, 1 << 31));
    try std.testing.expectEqual(Cpu.wfet, cpuFrom(65536, (1 << 31) | 0xffff));
    try std.testing.expectEqual(Cpu.no_wfet, cpuFrom(4096, 0));
    try std.testing.expectEqual(Cpu.no_wfet, cpuFrom(4096, (1 << 31) - 1));
    try std.testing.expectEqual(Cpu.unread, cpuFrom(0, 0));
    try std.testing.expectEqual(Cpu.unread, cpuFrom(0, 1 << 31));
}

test "the Features list names wfxt, and nothing else" {
    // an arm64 /proc/cpuinfo line, as GB10's
    try std.testing.expect(featuresWfxt("processor\t: 0\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics wfxt\n"));
    // the last feature, and the only line
    try std.testing.expect(featuresWfxt("Features: wfxt\n"));
    // no wfxt, a prefix, another capitalization, and x86's `flags`
    try std.testing.expect(!featuresWfxt("Features\t: fp asimd evtstrm aes wfxt2\n"));
    try std.testing.expect(!featuresWfxt("Features\t: fp asimd evtstrm aes\nflags\t\t: fpu vme de pse wfxt\n"));
    try std.testing.expect(!featuresWfxt("features\t: wfxt\n"));
    try std.testing.expect(!featuresWfxt(""));
}

test "cpu() agrees with /proc/cpuinfo's wfxt on this machine" {
    // The auxiliary vector is one authority; /proc/cpuinfo is another, and they must agree. This kills a wrong HWCAP2
    // key (AT.HWCAP, whose bit 31 is HWCAP_PACG), a swapped cpuFrom argument and a wrong WFXT bit on the target. It
    // cannot run under qemu-user: /proc/cpuinfo there is the x86 host's, which has no Features line (null).
    if (!arm) return error.SkipZigTest;
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    const wfxt = cpuinfoWfxt(io, gpa) orelse return error.SkipZigTest;
    try std.testing.expect(cpu() != .unread);
    try std.testing.expectEqual(wfxt, cpu() == .wfet);
}

test "a non-arm /proc/cpuinfo has no Features line, so the capability is unknown rather than absent" {
    if (arm) return error.SkipZigTest;
    try std.testing.expectEqual(@as(?bool, null), cpuinfoWfxt(std.testing.io, std.testing.allocator));
}

test "the ring check's nap JSON names each count, and a wanted ring that did not nap fails" {
    var buf: [512]u8 = undefined;
    const c: Counts = .{ .naps = 1004, .skipped = 2, .doorbell = 950, .late = 3, .deadline = 40, .other = 9, .ticks = 1_624_000_000 };
    const json = jsonLine(&buf, 2, "the four nodes' ring", .naps, c, 2_000_000_000, 1000, 120_000_000, true);
    // every field by name: a swapped late/deadline, a wrong napped_ms/elapsed_ms or a missing ok all fail here
    try std.testing.expectEqualStrings("{\"rank\": 2, \"ring\": \"the four nodes' ring\", \"rdma_wfe\": true, \"rdma_wfe_ok\": true, \"naps\": 1004, \"doorbell\": 950, \"late\": 3, \"deadline\": 40, \"other\": 9, \"neighbour\": 0, \"skipped\": 2, \"napped_ms\": 1624.000, \"elapsed_ms\": 2000.000, \"awake_ms\": 376.000, \"cpu_ms\": 120.000}", json);
    // a ring that naps but has not begun one: not ok when the switch asked for naps
    try std.testing.expect(std.mem.indexOf(u8, jsonLine(&buf, 0, "the ring", .naps, .{}, 1000, 1000, 0, true), "\"rdma_wfe_ok\": false") != null);
    // a ring that does not nap: false, and ok only when the switch did not ask
    try std.testing.expectEqualStrings("{\"rank\": 1, \"ring\": \"the ring\", \"rdma_wfe\": false, \"rdma_wfe_ok\": false, \"why\": \"naps\"}", jsonLine(&buf, 1, "the ring", .naps, null, 0, 1000, 0, true));
    try std.testing.expect(std.mem.indexOf(u8, jsonLine(&buf, 1, "the ring", .naps, null, 0, 1000, 0, false), "\"rdma_wfe_ok\": true") != null);
    // the criterion the check's exit code takes
    try std.testing.expect(napOk(true, .naps, c));
    try std.testing.expect(!napOk(true, .naps, .{}));
    try std.testing.expect(!napOk(true, .naps, null));
    // an all-skipped ring began naps but spent no tick napping: the ticks clause alone refuses it (the mutant that
    // drops `and x.ticks > 0` would pass this case)
    try std.testing.expect(!napOk(true, .naps, .{ .naps = 8, .skipped = 8 }));
    // and the JSON line carries the same refusal, not just the predicate (a mutant in jsonLine's ok expression fails)
    try std.testing.expect(std.mem.indexOf(u8, jsonLine(&buf, 0, "the ring", .naps, .{ .naps = 8, .skipped = 8 }, 1000, 1000, 0, true), "\"rdma_wfe_ok\": false") != null);
    try std.testing.expect(napOk(false, .naps, null));
    // a ring asked to nap but correctly spinning because its routes relay or its CPU
    // cannot nap is not a broken ring; only a ring whose start said it naps and then did not is. The JSON says why.
    try std.testing.expect(napOk(true, .relays, null));
    try std.testing.expect(napOk(true, .no_wfet, null));
    try std.testing.expect(napOk(true, .unread, null));
    try std.testing.expect(napOk(true, .no_clock, null));
    try std.testing.expect(!napOk(true, .backoff, null)); // asked to nap, then its wakes kept missing the deadline
    const why = jsonLine(&buf, 0, "the four nodes' ring", .relays, null, 0, 1000, 0, true);
    try std.testing.expect(std.mem.indexOf(u8, why, "\"rdma_wfe\": false") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "\"rdma_wfe_ok\": true") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "\"why\": \"relays\"") != null);
    // a ring whose naps all end late is one whose GPU store does not clear the monitor — the doorbell
    // moved, but the count was at or past the deadline every time, so the nap timed out first. The check must refuse
    // it when the switch asked: the JSON says so, and so does the exit criterion.
    const all_late: Counts = .{ .naps = 64, .late = 64, .ticks = 64_000 };
    try std.testing.expect(std.mem.indexOf(u8, jsonLine(&buf, 0, "the ring", .naps, all_late, 1000, 1000, 0, true), "\"rdma_wfe_ok\": false") != null);
    try std.testing.expect(!napOk(true, .naps, all_late));
    try std.testing.expect(napOk(false, .naps, all_late)); // the switch did not ask: the late share means nothing
    // the threshold is a quarter, not three quarters: the measured 10 kHz mix (60.8 % late) must be
    // refused, and a healthy ring's ~1 % false-late share must pass.
    try std.testing.expect(lateOk(true, .{ .naps = 64, .late = 16, .ticks = 64_000 })); // a quarter late: ok
    try std.testing.expect(!lateOk(true, .{ .naps = 64, .late = 17, .ticks = 64_000 })); // more than a quarter: refused
    try std.testing.expect(!lateOk(true, .{ .naps = 1000, .late = 608, .doorbell = 179, .other = 213, .ticks = 1_000_000 }));
    try std.testing.expect(lateOk(true, .{ .naps = 1000, .late = 10, .doorbell = 700, .other = 290, .ticks = 1_000_000 })); // ~1 % false-late
    try std.testing.expect(lateOk(true, .{ .naps = 4, .late = 4, .ticks = 4_000 })); // too few naps to judge
    try std.testing.expect(lateOk(true, null));
    // the "fewer than 16 naps" edge: 15 all-late naps stays under it; 16 all-late naps is refused (the mutant
    // `< 16` -> `<= 16` would pass the 16 case and fail here).
    try std.testing.expect(lateOk(true, .{ .naps = 15, .late = 15, .ticks = 15_000 }));
    try std.testing.expect(!lateOk(true, .{ .naps = 16, .late = 16, .ticks = 16_000 }));
    // a long caller name is truncated, not dropped along with the whole JSON line
    const long: [400]u8 = @splat('x');
    try std.testing.expect(jsonLine(&buf, 0, &long, .naps, c, 1000, 1000, 0, true).len > 0);
}

test "the ring check's per-case fold: a wrong word or a case that did not run fails it" {
    // the main's application line (`all_ok = gatherOk(all_ok, bad)`) and its error path (`gatherOk(all_ok, null)`):
    // a fold `and` -> `or` passes a mismatched case (true and false) and a case after a failure (false and true), so
    // both cases below fail under it. The exit the main returns is runCheck's (tests A's `return 0` mutant fails the
    // exit test below, which calls it on a failed fold).
    try std.testing.expect(gatherOk(true, 0)); // a case whose words all matched
    try std.testing.expect(!gatherOk(true, 1)); // one wrong word
    try std.testing.expect(!gatherOk(false, 0)); // a later case after an earlier failure
    try std.testing.expect(!gatherOk(true, null)); // a case that returned an error and did not run
    try std.testing.expect(!gatherOk(false, null));
    try std.testing.expectEqual(@as(u8, 0), runCheck(gatherOk(true, 0), false, &.{}));
    try std.testing.expectEqual(@as(u8, 1), runCheck(gatherOk(true, 3), false, &.{}));
    try std.testing.expectEqual(@as(u8, 1), runCheck(gatherOk(true, null), false, &.{}));
}

test "the ring check's main tail: a failed case or a failed ring exits 1" {
    // main's two application lines: `all_ok = gatherOk(all_ok, bad)` for every case (null on its error path) and
    // `return runCheck(all_ok, asked, naps[0..n_naps])`. Both decisions are here, so this test pins the exit the
    // main applies, including the empty ring slice a run that opened none builds (main itself needs CUDA and is not
    // runnable on a host).
    var matched = gatherOk(true, 0);
    try std.testing.expectEqual(@as(u8, 0), runCheck(matched, false, &.{}));
    matched = gatherOk(matched, 4); // the second case's gather had four wrong words
    try std.testing.expectEqual(@as(u8, 1), runCheck(matched, false, &.{}));
    try std.testing.expectEqual(@as(u8, 1), runCheck(gatherOk(true, null), false, &.{})); // the error path's case
    const failed: RingNap = .{ .how = .naps, .c = .{ .naps = 64, .late = 1, .ticks = 64_000 }, .failed = "ibv_poll_cq failed" };
    try std.testing.expectEqual(@as(u8, 1), runCheck(true, true, &.{failed}));
}

test "the ring check's exit is 0 only when every gather matched and every ring's nap met the switch" {
    // tf-dsv41-ring2d-check's automated ship gate: 0 only when the words all matched and no ring's nap was refused.
    try std.testing.expectEqual(@as(u8, 0), checkExit(true, true));
    try std.testing.expectEqual(@as(u8, 1), checkExit(false, true));
    try std.testing.expectEqual(@as(u8, 1), checkExit(true, false));
    try std.testing.expectEqual(@as(u8, 1), checkExit(false, false));
    // the per-ring fold: a correctly spinning relay ring beside a napping one is still ok; one ring that was to nap
    // and did not fails the check as a wrong gather does.
    const one_good = napOk(true, .relays, null) and napOk(true, .naps, .{ .naps = 1, .ticks = 1 });
    const one_bad = napOk(true, .relays, null) and napOk(true, .naps, null);
    try std.testing.expectEqual(@as(u8, 0), checkExit(true, one_good));
    try std.testing.expectEqual(@as(u8, 1), checkExit(true, one_bad));
}

test "the ring check's per-ring fold and exit: one refused ring fails the check" {
    // the fold the ring check applies to its rings: a relay ring that correctly spins is not a failure; a ring that
    // was to nap and did not, or that stopped napping, is. The exit is 0 only when every gather matched and every
    // ring's nap met the switch. A mutant `and` -> `or` in the fold, or `return 0` in the exit, fails here.
    const good: RingNap = .{ .how = .naps, .c = .{ .naps = 64, .late = 1, .ticks = 64_000 } };
    const relay: RingNap = .{ .how = .relays, .c = null };
    const bad: RingNap = .{ .how = .naps, .c = null };
    const backoff: RingNap = .{ .how = .backoff, .c = null };
    // a ring whose proxy failed after the last gather: its nap half is healthy, but the failure alone refuses it, and
    // a failed run fails the check whatever the switch asked
    const failed: RingNap = .{ .how = .naps, .c = .{ .naps = 64, .late = 1, .ticks = 64_000 }, .failed = "ibv_poll_cq failed" };
    try std.testing.expect(ringsOk(true, &.{ good, relay }));
    try std.testing.expect(!ringsOk(true, &.{ good, bad }));
    try std.testing.expect(!ringsOk(true, &.{ good, backoff }));
    try std.testing.expect(ringsOk(false, &.{ bad, backoff })); // the switch did not ask: not judged
    try std.testing.expect(!ringsOk(true, &.{ good, failed }));
    try std.testing.expect(!ringsOk(false, &.{ good, failed })); // a failed proxy fails the check even when the switch did not ask
    try std.testing.expectEqual(@as(u8, 0), runCheck(true, true, &.{ good, relay }));
    try std.testing.expectEqual(@as(u8, 1), runCheck(false, true, &.{ good, relay }));
    try std.testing.expectEqual(@as(u8, 1), runCheck(true, true, &.{ good, bad }));
    try std.testing.expectEqual(@as(u8, 0), runCheck(true, false, &.{ good, bad })); // the switch did not ask: not judged
    try std.testing.expectEqual(@as(u8, 1), runCheck(false, false, &.{ good, bad }));
    try std.testing.expectEqual(@as(u8, 1), runCheck(true, true, &.{ good, failed }));
    try std.testing.expectEqual(@as(u8, 1), runCheck(true, false, &.{ good, failed }));
}

test "the ring check's failure line names the ring and the reason" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("{\"rank\": 3, \"ring\": \"the four nodes' ring\", \"failure\": \"ibv_poll_cq failed\"}", failureLine(&buf, 3, "the four nodes' ring", "ibv_poll_cq failed"));
    // a long caller name is truncated, not dropped along with the whole line
    const long: [400]u8 = @splat('x');
    try std.testing.expect(failureLine(&buf, 0, &long, "PostFailed").len > 0);
}
