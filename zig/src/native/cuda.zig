//! CUDA family registration, device setup, memory admission and the native lane host.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const dsv41 = @import("dsv41");
const Allocator = std.mem.Allocator;
const budget = @import("cuda_memory.zig");
const Pool = budget.Pool;

/// CUDA families provide metadata, open their lane backend and explain their refusals.
const registry = .{ nemotron.native, dsv41.native };

pub const backends: []const []const u8 = &.{"cuda"};
pub const families: []const api.Family = blk: {
    var out: [registry.len]api.Family = undefined;
    for (registry, 0..) |F, i| out[i] = .{ .model_type = F.model_type, .formats = F.formats };
    const final = out;
    break :blk &final;
};

const gib: f64 = 1 << 30;

/// The chip class gate entries name ("nvidia-sm121" for a GB10) of the device TF_CUDA_DEVICE picks; null without one.
pub fn chip(a: Allocator) ?[]const u8 {
    const ordinal = envNumber(getenv("TF_CUDA_DEVICE")) catch return null;
    var driver = cuda.Driver.open() catch return null;
    defer driver.close();
    if (ordinal orelse 0 >= driver.deviceCount() catch return null) return null;
    return chipClass(a, driver.capability(@intCast(ordinal orelse 0)) catch return null);
}

fn chipClass(a: Allocator, capability: u32) ?[]const u8 {
    return std.fmt.allocPrint(a, "nvidia-sm{d}", .{capability}) catch null;
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// A variable's whole number: null when unset or empty, an error for anything else that is not one.
fn envNumber(text: ?[]const u8) error{Invalid}!?u32 {
    const t = std.mem.trim(u8, text orelse return null, " ");
    if (t.len == 0) return null;
    return std.fmt.parseInt(u32, t, 10) catch error.Invalid;
}

/// The flag's value, else the variable's, else `default`; `problem` names a variable that is not a whole number.
fn setting(a: Allocator, flag: ?u32, name: [:0]const u8, default: u32, problem: *[]const u8) !?u32 {
    if (flag) |v| return v;
    const v = envNumber(getenv(name)) catch {
        problem.* = try std.fmt.allocPrint(a, "{s}={s}: a whole number", .{ name, getenv(name).? });
        return null;
    };
    return v orelse default;
}

/// The model's window (config.json's max_position_embeddings, text_config's first), 0 when it names none.
fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

/// Prompt plus reply tokens a request may use: --context (0: the model's window), else the family default within it.
fn contextWindow(requested: ?i64, native: i64, default: i64) error{ Negative, NoNative, PastNative }!i64 {
    const r = requested orelse return if (native > 0) @min(default, native) else default;
    if (r < 0) return error.Negative;
    if (r == 0) return if (native > 0) native else error.NoNative;
    if (native > 0 and r > native) return error.PastNative;
    return r;
}

/// The kernel set: TENSORFOLD_CUDA_KERNELS, else share/tensorfold/cuda/sm<capability> beside the binary.
fn kernelDir(a: Allocator, io: std.Io, capability: u32) ![]const u8 {
    if (getenv("TENSORFOLD_CUDA_KERNELS")) |dir| return a.dupe(u8, dir);
    const exe = try std.process.executableDirPathAlloc(io, a);
    return std.fs.path.join(a, &.{ exe, "..", "share", "tensorfold", "cuda", try std.fmt.allocPrint(a, "sm{d}", .{capability}) });
}

/// The bytes of the checkpoint's safetensors files: what its weights need on the device, near enough to refuse early.
fn weightBytes(io: std.Io, dir: []const u8) u64 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const st = d.statFile(io, e.name, .{}) catch continue;
        total += st.size;
    }
    return total;
}

/// A /proc file's text, streamed: procfs reports size 0, and a positional read (readFileAlloc) stops there.
fn procText(a: Allocator, io: std.Io, path: []const u8) ?[]u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &buf);
    return r.interface.allocRemaining(a, .limited(1 << 20)) catch null;
}

/// The pool now, read on the thread whose context is current; `problem` names a bad variable.
fn pool(a: Allocator, io: std.Io, ctx: *const cuda.Context, problem: *[]const u8) !?Pool {
    const unified = (try ctx.attribute(.integrated)) != 0;
    const card = try ctx.memInfo();
    const text = if (unified) procText(a, io, "/proc/meminfo") else null;
    const available = budget.counts(unified, .{ .total = card.total, .available = card.free }, text) catch {
        problem.* = "cannot read or parse /proc/meminfo's MemTotal and MemAvailable; refusing CUDA unified-memory admission";
        return null;
    };
    const free = available.available;
    const total = available.total;
    const reserve = budget.reserveBytes(getenv("TENSORFOLD_MEMORY_RESERVE_GIB"), total, unified) catch {
        problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_MEMORY_RESERVE_GIB={s}: a number of GiB from 2 to the memory's size", .{getenv("TENSORFOLD_MEMORY_RESERVE_GIB").?});
        return null;
    };
    const limit = budget.limitBytes(getenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB")) catch {
        problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_CUDA_MEMORY_LIMIT_GB={s}: a positive number of GiB whose byte count fits in a 64-bit size", .{getenv("TENSORFOLD_CUDA_MEMORY_LIMIT_GB").?});
        return null;
    };
    return .{ .free = free, .total = total, .reserve = reserve, .limit = limit, .unified = unified };
}

/// The lane host's health source: no sticky fault on the GPU context (read without a driver call).
fn contextHealthy(ctx: *anyopaque) bool {
    const c: *const cuda.Context = @ptrCast(@alignCast(ctx));
    return c.fault() == 0;
}

fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const u = cuda.usage(reset_peak);
    return .{ .active = u.device, .cache = 0, .peak = u.peak };
}

/// A lone driver returns false at completion or true when yield hands it to the lane core.
const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// The context a lane thread needs current: the lane host steps rounds on its own thread, CUDA binds per thread.
threadlocal var bound: ?*const cuda.Context = null;

const Gpu = struct { driver: cuda.Driver, ctx: cuda.Context };

/// One loaded model behind the lane host: everything the engine thread reads lives here.
const Host = struct {
    gpa: Allocator,
    gpu: ?*Gpu, // null in host tests: no context to bind
    family: *anyopaque,
    release: *const fn (*anyopaque) void,
    inner: lanes.backend.Backend,
    vtable: lanes.backend.Backend.VTable,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []u8 = &.{},
    lone: ?LoneRun = null, // the family's driver for a lone drafted stream, called with `family`
    cache: ?api.prompt_cache.Store = null, // kept prompt states, when the family keeps them
    snap_vt: api.prompt_cache.Snapshots.VTable = undefined,
    graph_fallbacks: ?*const std.atomic.Value(u64) = null, // the family's failed graph captures, when it counts them
    head_rounds: ?*const [6]std.atomic.Value(u64) = null, // the family's decode rounds by the head's path, when counted
    draft_vocab: ?*const api.DraftVocabCounters = null, // the family's drafter on a pruned vocabulary, when it runs one
    finish: ?*const fn (*anyopaque) void = null, // the family's orderly close (dsv41: its followers hear done), once
    finished: bool = false,
    quiet: ?*const fn (*anyopaque) bool = null, // the family's check for the tier's service fence (dsv41: no graph capture)
    spill: ?api.Spill = null, // the tier's settings, when the family's tier was asked for: the lane host's configure
    spill_start: ?api.SpillState = null, // the family's tier at its start (off: the reason status shows)

    /// api.Opened.quiesce: every request ended and the engine thread joined (its last work done), then the family's
    /// finish; nothing freed, so the server's connection threads may still call the engine until close.
    fn quiesce(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.quiesce();
        h.finishOnce();
    }

    /// The family's finish, at most once (quiesce's, else close's), with the context current on the calling thread.
    fn finishOnce(h: *Host) void {
        if (h.finished) return;
        h.finished = true;
        const f = h.finish orelse return;
        if (h.gpu) |g| g.ctx.makeCurrent() catch {};
        f(h.family);
    }

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.finishOnce(); // after quiesce: nothing; else the family's finish before its kept states and its release
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        if (h.gpu) |g| g.ctx.makeCurrent() catch {};
        if (h.cache) |*store| store.deinit(); // the family's kept states go before the family
        h.release(h.family);
        if (h.gpu) |g| {
            g.ctx.deinit();
            g.driver.close();
            h.gpa.destroy(g);
        }
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }

    /// The tier's service fence asks (the lane host's engine thread): the family's quiet, with the context bound.
    fn quietCheck(p: *anyopaque) bool {
        const x = bind(p);
        return x.quiet.?(x.family);
    }

    fn bind(p: *anyopaque) *Host {
        const h: *Host = @ptrCast(@alignCast(p));
        if (h.gpu) |g| if (bound != &g.ctx) {
            g.ctx.makeCurrent() catch |e| std.log.err("cuCtxSetCurrent on the lane thread: {s}", .{@errorName(e)});
            bound = &g.ctx;
        };
        return h;
    }

    /// The round loop and the lane host over the family's backend, its thread started.
    fn serve(h: *Host, io: std.Io, facts: lanes.Model, rows: u32, info: api.Info, explain: ?api.Explain) !void {
        h.cfg = try lanes.Config.init(h.gpa, facts, rows, rows - 1);
        errdefer h.cfg.deinit(h.gpa);
        h.clock = .{ .io = io };
        h.core = lanes.Engine.init(h.gpa, &h.cfg, h.backend(), h.clock.clock());
        errdefer h.core.deinit();
        h.host = api.LaneHost.init(h.gpa, io, &h.core, info);
        h.host.memory = if (h.gpu != null) .{ .read = readMemory } else null;
        h.host.health = if (h.gpu) |g| .{ .ctx = &g.ctx, .read = contextHealthy } else null;
        h.host.explain = explain;
        if (h.cache) |*store| h.host.cache = store;
        if (h.quiet != null) h.host.quiet = .{ .ctx = h, .check = quietCheck };
        // the tier asked for: its settings configure the host (after the store and its tier: configure reads them), the
        // family's own off reason from its start (the open, the probe, an identity) shown as the settings' would be
        if (h.spill) |given| {
            var sp = given;
            if (h.spill_start) |*start| if (start.state == .off and sp.off == null) {
                sp.off = start.reason();
            };
            h.host.configure(sp);
        }
        if (h.lone != null) h.host.lone = .{ .ctx = h, .run = loneRun, .sampled = true };
        h.host.graph_fallbacks = h.graph_fallbacks;
        h.host.head_rounds = h.head_rounds;
        h.host.draft_vocab = h.draft_vocab;
        try h.host.start();
    }

    /// The family's lone driver with the context current on the lane thread.
    fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const x = bind(p);
        return x.lone.?(x.family, s, hooks.ctx, hooks.committed, hooks.yield);
    }

    /// The family's backend, each call made with the context current on the calling thread.
    fn backend(h: *Host) lanes.backend.Backend {
        const v = h.inner.vtable;
        h.vtable = .{
            .prefill = struct {
                fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.prefill(x.inner.ptr, s);
                }
            }.f,
            .prefill_step = if (v.prefill_step != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream) anyerror!bool {
                    const x = bind(p);
                    return x.inner.vtable.prefill_step.?(x.inner.ptr, s);
                }
            }.f else null,
            .live_cfg = if (v.live_cfg != null) struct {
                fn f(p: *anyopaque, version: u64, items: []const lanes.backend.CfgItem) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.live_cfg.?(x.inner.ptr, version, items);
                }
            }.f else null,
            .first = struct {
                fn f(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
                    const x = bind(p);
                    return x.inner.vtable.first(x.inner.ptr, s, position);
                }
            }.f,
            .queue = struct {
                fn f(p: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
                    const x = bind(p);
                    return x.inner.vtable.queue(x.inner.ptr, s, feed, position);
                }
            }.f,
            .read = struct {
                fn f(p: *anyopaque, handle: u64) anyerror!u32 {
                    const x = bind(p);
                    return x.inner.vtable.read(x.inner.ptr, handle);
                }
            }.f,
            .verify = struct {
                fn f(p: *anyopaque, w: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.verify(x.inner.ptr, w, out);
                }
            }.f,
            .keep = struct {
                fn f(p: *anyopaque, w: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.keep(x.inner.ptr, w, paths);
                }
            }.f,
            .draft = struct {
                fn f(p: *anyopaque, r: []const lanes.backend.DraftRequest) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.draft(x.inner.ptr, r);
                }
            }.f,
            .unspeculate = if (v.unspeculate != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.unspeculate.?(x.inner.ptr, s);
                }
            }.f else null,
            .probabilities = if (v.probabilities != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
                    const x = bind(p);
                    return x.inner.vtable.probabilities.?(x.inner.ptr, s, out);
                }
            }.f else null,
            .tree = if (v.tree != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
                    const x = bind(p);
                    return x.inner.vtable.tree.?(x.inner.ptr, s, gpa);
                }
            }.f else null,
            .alternatives = if (v.alternatives != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, out: []lanes.backend.Alternative) anyerror!usize {
                    const x = bind(p);
                    return x.inner.vtable.alternatives.?(x.inner.ptr, s, out);
                }
            }.f else null,
            .first_masked = if (v.first_masked != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
                    const x = bind(p);
                    return x.inner.vtable.first_masked.?(x.inner.ptr, s, position, mask);
                }
            }.f else null,
            .held = if (v.held != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.held.?(x.inner.ptr, s, out);
                }
            }.f else null,
            .release = struct {
                fn f(p: *anyopaque, s: *lanes.Stream) void {
                    const x = bind(p);
                    x.inner.vtable.release(x.inner.ptr, s);
                }
            }.f,
        };
        return .{ .ptr = h, .vtable = &h.vtable };
    }
};

/// The engine for `o.dir`, or null with `problem` set when no CUDA family reads the checkpoint.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    inline for (registry) |F| {
        if (std.mem.eql(u8, o.model_type, F.model_type)) return openWith(F, a, gpa, io, o, problem);
    }
    problem.* = try std.fmt.allocPrint(a, "the native CUDA engine has no backend for {s} checkpoints yet", .{o.model_type});
    return null;
}

fn openWith(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    // a family whose window depends on its own settings (DeepSeek-V4.1: the YaRN factor of its RoPE tables) reports it
    const native = if (@hasDecl(F, "nativeWindow")) F.nativeWindow(a, io, o.dir, modelContext(a, io, o.dir)) else modelContext(a, io, o.dir);
    const window = contextWindow(o.context, native, F.default_context) catch |e| {
        problem.* = switch (e) {
            error.Negative => try std.fmt.allocPrint(a, "--context {d}: a token count, or 0 for the model's window", .{o.context.?}),
            error.NoNative => "--context 0 asks for the model's window, and its config.json names none: give a token count",
            error.PastNative => try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ o.context.?, native }),
        };
        return null;
    };
    const device = try setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    const segments = try setting(a, o.segments, "TF_CUDA_SEGMENTS", 1, problem) orelse return null;
    if (segments < 1 or segments > F.max_segments) {
        const named = if (o.segments != null) "--segments " else "TF_CUDA_SEGMENTS=";
        problem.* = try std.fmt.allocPrint(a, "{s}{d}: whole prompt chunks a call, 1 to {d}", .{ named, segments, F.max_segments });
        return null;
    }
    const g = try gpa.create(Gpu);
    var opened = false;
    defer if (!opened) gpa.destroy(g);
    // a family's own process settings (the DeepSeek engine's sleeping waits and 16 hardware queues), before the context reads them
    if (@hasDecl(F, "applyFamilyDefaults")) F.applyFamilyDefaults();
    g.driver = cuda.Driver.open() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(e)});
        return null;
    };
    defer if (!opened) g.driver.close();
    const count = try g.driver.deviceCount();
    if (device >= count) {
        const named = if (o.device != null) "--device " else "TF_CUDA_DEVICE=";
        problem.* = try std.fmt.allocPrint(a, "{s}{d}: this machine has {d} CUDA device{s} (0 to {d})", .{ named, device, count, if (count == 1) "" else "s", @max(count, 1) - 1 });
        return null;
    }
    g.ctx = try cuda.Context.init(&g.driver, @intCast(device));
    defer if (!opened) g.ctx.deinit();
    bound = &g.ctx;
    const capability = try g.ctx.capability();
    var name_buf: [256]u8 = undefined;
    const name = g.ctx.name(&name_buf) catch "GPU";
    const kernels = try kernelDir(a, io, capability);
    std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ kernels, "aot.json" }), .{}) catch {
        problem.* = try std.fmt.allocPrint(a, "no CUDA kernel set for sm_{d} at {s}: set TENSORFOLD_CUDA_KERNELS to a folder from aot_pack.py", .{ capability, kernels });
        return null;
    };
    const before = try pool(a, io, &g.ctx, problem) orelse return null;
    // a family that splits the checkpoint over several GPUs reports its own share
    const weights = if (@hasDecl(F, "weightBytes")) F.weightBytes(io, o.dir) else weightBytes(io, o.dir);
    const held0 = cuda.usage(false).device;
    if (weights > before.room(held0)) {
        problem.* = try std.fmt.allocPrint(a, "the checkpoint's {d:.1} GiB of weights do not fit the {d:.1} GiB the CUDA memory budget grants ({d:.1} GiB free less a {d:.1} GiB reserve{s}); free device memory or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB", .{ toGib(weights), toGib(before.room(held0)), toGib(before.free), toGib(before.reserve), if (before.limit != null) ", under TENSORFOLD_CUDA_MEMORY_LIMIT_GB" else "" });
        return null;
    }
    const cache_want: u64 = @intFromFloat((o.prompt_cache_gib orelse 0.5) * gib); // the prompt cache's budget
    var fo: F.Options = .{ .context = @intCast(window), .drafts = o.drafts, .segments = segments };
    if (@hasField(F.Options, "spill")) fo.spill = familySpill(o.spill, cache_want); // a family that binds a tier
    const loaded = F.open(gpa, io, &g.ctx, o.dir, kernels, fo) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native CUDA engine cannot load {s} with kernels {s} ({s})", .{ o.dir, kernels, @errorName(e) });
        return null;
    };
    defer if (!opened) loaded.deinit(loaded.ctx);
    const model = cuda.usage(false).device - held0;
    const after = try pool(a, io, &g.ctx, problem) orelse return null;
    const room = after.room(model);
    const streams = budget.admit(room, loaded.stream_bytes, if (@hasField(@TypeOf(loaded), "free_streams")) loaded.free_streams else 0, o.lanes, o.lanes_fixed) catch |e| {
        problem.* = switch (e) {
            error.NoStream => try std.fmt.allocPrint(a, "the CUDA memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve; lower --context, or free device memory", .{ window, toGib(loaded.stream_bytes), toGib(room), toGib(model), toGib(after.reserve) }),
            error.TooMany => try std.fmt.allocPrint(a, "--parallel {d} needs {d:.2} GiB for its streams at a {d}-token window, and the CUDA memory budget leaves {d:.2} GiB: serve --parallel {d}, or lower --context", .{ o.lanes, toGib(loaded.stream_bytes * o.lanes), window, toGib(room), room / loaded.stream_bytes }),
        };
        return null;
    };
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = g, .family = loaded.ctx, .release = loaded.deinit, .inner = loaded.backend, .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined, .lone = loaded.lone };
    h.startup = try std.fmt.allocPrint(gpa, "CUDA sm_{d} device {d} ({s}{s}): model {d:.2} GiB; {d} stream{s} at once, {d:.2} GiB each at a {d}-token window, of {d:.1} GiB left after a {d:.1} GiB reserve; prompts in {d}-row chunks{s}", .{
        capability, device, name, if (after.unified) ", memory shared with the host" else "", toGib(model), streams, if (streams == 1) "" else "s", toGib(loaded.stream_bytes), window, toGib(room), toGib(after.reserve), F.prompt_rows, if (segments > 1) try std.fmt.allocPrint(a, ", {d} staggered segments a call", .{segments}) else "",
    });
    errdefer gpa.free(h.startup);
    if (@hasDecl(F, "graphFallbacks")) h.graph_fallbacks = F.graphFallbacks(loaded.ctx);
    if (@hasDecl(F, "headRounds")) h.head_rounds = F.headRounds(loaded.ctx);
    if (@hasDecl(F, "draftVocab")) h.draft_vocab = F.draftVocab(loaded.ctx);
    if (@hasField(@TypeOf(loaded), "finish")) h.finish = loaded.finish;
    // a family that keeps prompt states (its snapshots) gets the prompt cache: states at its chunk ends
    if (@hasField(@TypeOf(loaded), "snaps")) if (loaded.snaps) |sn| {
        const want = cache_want;
        if (want > 0) {
            h.snap_vt = .{ .bytes = sn.bytes, .save = sn.save, .restore = sn.restore, .drop = sn.drop };
            h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = sn.ptr, .vtable = &h.snap_vt }, .{ .window = if (@hasField(@TypeOf(sn), "tail")) sn.tail else 0 }, want);
            if (@hasField(@TypeOf(loaded), "tier")) if (loaded.tier) |t| bindTier(&h.cache.?, &h.snap_vt, t, loaded.grid, loaded.replay, if (@hasField(@TypeOf(sn), "usable")) sn.usable else null, io);
        }
    };
    // the SSD tier's settings and state for /health and /metrics (the family's, when it was asked for); the lane host
    // gets them, the family's start state and its fence check at serve
    const spill_info: ?api.SpillInfo = if (@hasField(@TypeOf(loaded), "spill_info")) loaded.spill_info else null;
    if (spill_info) |si| if (si.settings.asked) {
        h.spill = si.settings;
        h.spill_start = if (@hasField(@TypeOf(loaded), "spill_state")) loaded.spill_state else null;
    };
    // what /health and the start-up line show: the family's own off reason from its start too (h holds the text)
    const shown_spill = startInfo(spill_info, if (h.spill_start) |*st| st else null);
    // the speed report, once, now that the cache and the tier are decided (a family with no tier has none to report)
    if (@hasField(@TypeOf(loaded), "spill_info")) api.speed_report.emit(.{
        .cache_bytes = if (h.cache != null) cache_want else 0,
        .spill_asked = if (shown_spill) |si| si.settings.asked else false,
        .spill_on = if (shown_spill) |si| si.settings.asked and si.settings.off == null and si.settings.dir != null and si.settings.gib > 0 else false,
        .spill_off = if (shown_spill) |si| si.settings.off else null,
        .spill_gib = if (shown_spill) |si| si.settings.gib else 0,
    });
    if (@hasField(@TypeOf(loaded), "quiet")) h.quiet = loaded.quiet;
    // the family cuts its own prompt grid from position 0, as `tensorfold run` does: prefill_step 0
    const vision: ?api.LaneVision = if (@hasField(@TypeOf(loaded), "lane_vision")) loaded.lane_vision else null;
    // the server's window is what a stream holds when the family says (its lanes add a round's rows to every
    // stream); else the configured window, as before
    const usable: i64 = if (@hasField(@TypeOf(loaded), "usable_window")) (if (loaded.usable_window) |u| @min(window, @as(i64, u)) else window) else window;
    // the family's resume grid: a prompt step's rows between two rounds at the default cadence (the scheduling facts)
    const resume_rows: u32 = if (@hasField(@TypeOf(loaded), "grid")) loaded.grid else 0;
    // the rows a prompt step may run between two rounds while a stream decodes (0: the resume rows)
    const beside_rows: u32 = if (@hasField(@TypeOf(loaded), "beside")) loaded.beside else 0;
    // the rounds that follow each long prompt step while a stream decodes (0: unset)
    const rounds_beside: u32 = if (@hasField(@TypeOf(loaded), "rounds")) loaded.rounds else 0;
    // the family's prompt-step order while several prompts fill together (a family that schedules none: the rotation)
    const fill: api.fill_order.Policy = if (@hasField(@TypeOf(loaded), "fill")) loaded.fill else .{};
    try h.serve(io, loaded.facts, loaded.rows, .{ .lanes = streams, .context_window = @intCast(usable), .prompt_cache = h.cache != null, .startup = h.startup, .efforts = familyEfforts(F), .lane_vision = vision, .structures = loaded.backend.vtable.first_masked != null, .spill = shown_spill, .resume_rows = resume_rows, .beside_rows = beside_rows, .rounds_beside = rounds_beside, .fill = fill }, .{ .ctx = loaded.ctx, .text = F.explain });
    opened = true;
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h, .quiesce = Host.quiesce };
}

/// Binds the SSD tier to the prompt cache: the family's resume grid (its prompt chunk) and replay window, its usable
/// rule (stale and loading states passed over), the tier with this process's seed for conversations and lineages, and
/// the token arrays charged to the budget (only with a tier).
fn bindTier(store: *api.prompt_cache.Store, vt: *api.prompt_cache.Snapshots.VTable, tier: api.prompt_cache.Tier, grid: u32, replay: u32, usable: ?*const fn (*anyopaque, *anyopaque, u32) bool, io: std.Io) void {
    vt.usable = usable;
    store.rules.grid = grid;
    store.rules.window = replay;
    store.charge_tokens = true;
    var seed: u32 = undefined;
    io.random(std.mem.asBytes(&seed));
    store.setTier(tier, seed | 1);
}

/// The tier's settings the family opens with: as given, but off from the start, named, with no prompt cache
/// (--prompt-cache-gib 0: nothing to keep or restore), so the family opens no store, staging, landing or probe.
fn familySpill(sp: api.Spill, cache_bytes: u64) api.Spill {
    var out = sp;
    if (out.asked and out.off == null and cache_bytes == 0) out.off = "the prompt cache is off (--prompt-cache-gib 0): the SSD tier has nothing to keep or restore";
    return out;
}

/// /health's and the start-up line's tier, with the family's own off reason from its start (the open, the probe, an
/// identity) when the settings had none: neither says "on" while it is off. `start` (the Host's) outlives the server.
fn startInfo(si: ?api.SpillInfo, start: ?*const api.SpillState) ?api.SpillInfo {
    var out = si orelse return null;
    if (start) |st| if (st.state == .off and out.settings.off == null) {
        out.settings.off = st.reason();
    };
    return out;
}

/// A family's reasoning efforts as numbers its chat template takes (F.efforts: name and value), else none.
fn familyEfforts(comptime F: type) []const api.Effort {
    if (!@hasDecl(F, "efforts")) return &.{};
    const table = comptime blk: {
        var out: [F.efforts.len]api.Effort = undefined;
        for (F.efforts, &out) |e, *o| o.* = .{ .name = e.name, .value = e.value };
        break :blk out;
    };
    return &table;
}

fn toGib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / gib;
}

test "this host's /proc/meminfo reads whole and parses (Linux)" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const text = procText(a, std.testing.io, "/proc/meminfo") orelse return error.Unreadable;
    defer a.free(text);
    try std.testing.expect(budget.meminfo(text) != null);
}

test "chip classes name the compute capability as gate entries do" {
    const a = std.testing.allocator;
    const name = chipClass(a, 121).?;
    defer a.free(name);
    try std.testing.expectEqualStrings("nvidia-sm121", name);
}

test "every registered family is listed for capabilities" {
    try std.testing.expectEqual(@as(usize, registry.len), families.len);
    try std.testing.expectEqualStrings("nemotron_h", families[0].model_type);
}

test "variables: unset or empty means the default, anything but a whole number is refused" {
    try std.testing.expectEqual(@as(?u32, null), try envNumber(null));
    try std.testing.expectEqual(@as(?u32, null), try envNumber(""));
    try std.testing.expectEqual(@as(?u32, 3), try envNumber(" 3"));
    try std.testing.expectError(error.Invalid, envNumber("one"));
    try std.testing.expectError(error.Invalid, envNumber("-1"));
}

test "context windows: the family default inside the model's, 0 for the model's, never past it" {
    try std.testing.expectEqual(@as(i64, 16384), try contextWindow(null, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 4096), try contextWindow(null, 4096, 16384));
    try std.testing.expectEqual(@as(i64, 262144), try contextWindow(0, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 65536), try contextWindow(65536, 262144, 16384));
    try std.testing.expectError(error.PastNative, contextWindow(300000, 262144, 16384));
    try std.testing.expectError(error.Negative, contextWindow(-5, 262144, 16384));
    try std.testing.expectError(error.NoNative, contextWindow(0, 0, 16384));
}

/// A family for the orderly-close tests: counts its finish and release calls, and keeps a Fake target's prompt passes.
const CloseFamily = struct {
    mutex: std.Io.Mutex = .init,
    finished: u32 = 0,
    released: u32 = 0,
    chunks: usize = 0, // prompt chunks begun
    after_finish: usize = 0, // chunks begun after finish ran (its followers would wait for frames that never come)

    fn finish(p: *anyopaque) void {
        const f: *CloseFamily = @ptrCast(@alignCast(p));
        f.mutex.lockUncancelable(std.testing.io);
        defer f.mutex.unlock(std.testing.io);
        f.finished += 1;
    }

    fn release(p: *anyopaque) void {
        const f: *CloseFamily = @ptrCast(@alignCast(p));
        f.released += 1;
    }

    /// A prompt chunk of the target: 10 ms each.
    fn chunk(p: *anyopaque, _: *lanes.Stream, _: usize) void {
        const f: *CloseFamily = @ptrCast(@alignCast(p));
        f.mutex.lockUncancelable(std.testing.io);
        f.chunks += 1;
        if (f.finished > 0) f.after_finish += 1;
        f.mutex.unlock(std.testing.io);
        std.Io.sleep(std.testing.io, .fromMilliseconds(10), .awake) catch {};
    }

    fn state(f: *CloseFamily) struct { u32, usize, usize } {
        f.mutex.lockUncancelable(std.testing.io);
        defer f.mutex.unlock(std.testing.io);
        return .{ f.finished, f.chunks, f.after_finish };
    }

    /// A Host over `target` with no device, served with one lane, as openWith leaves it.
    fn host(f: *CloseFamily, gpa: Allocator, target: *lanes.fake.Fake) !*Host {
        const h = try gpa.create(Host);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .gpu = null, .family = f, .release = release, .finish = finish, .inner = target.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
        try h.serve(std.testing.io, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 7, .hidden_rows = true }, 8, .{ .lanes = 1 }, null);
        return h;
    }
};

/// A request's sink for the orderly-close tests.
const CloseBox = struct {
    mutex: std.Io.Mutex = .init,
    tokens: usize = 0,
    done: ?api.Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (ev.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn state(b: *@This()) struct { ?api.Reason, usize } {
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        return .{ b.done, b.tokens };
    }
};

test "the orderly close ends the requests and runs the family's finish once; close does not run it again" {
    const gpa = std.testing.allocator;
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var fam: CloseFamily = .{};
    const h = try fam.host(gpa, &target);
    var closed = false;
    defer if (!closed) Host.close(h);
    const e = h.host.engine();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var box: CloseBox = .{};
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 1 << 30 }, .{ .ctx = &box, .event = CloseBox.event });
    while (box.state()[1] == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    Host.quiesce(h);
    try std.testing.expectEqual(@as(?api.Reason, .failed), box.state()[0]); // cut by the stop: the client sends it again
    try std.testing.expectEqual(@as(u32, 1), fam.state()[0]);
    try std.testing.expectEqual(@as(u32, 0), fam.released);
    var late: CloseBox = .{};
    try std.testing.expectError(error.Closed, e.submit(2, &.{ .prompt = &prompt, .max_tokens = 8 }, .{ .ctx = &late, .event = CloseBox.event }));
    Host.quiesce(h); // a second one does nothing
    try std.testing.expectEqual(@as(u32, 1), fam.state()[0]);
    closed = true;
    Host.close(h);
    try std.testing.expectEqual(@as(u32, 1), fam.state()[0]);
    try std.testing.expectEqual(@as(u32, 1), fam.released);
}

test "the DeepSeek family hands the host its orderly close (the field openWith reads)" {
    // openWith takes it by @hasField: a rename on either side would leave the followers without done at a stop
    try std.testing.expect(@hasField(dsv41.native.Loaded, "finish"));
    try std.testing.expect(@FieldType(dsv41.native.Loaded, "finish") == @FieldType(Host, "finish"));
}

test "the DeepSeek family hands the host its decode rounds by the head's path (the declaration openWith reads)" {
    // openWith takes it by @hasDecl: a rename on either side would leave /metrics without the counts, silently
    try std.testing.expect(@hasDecl(dsv41.native, "headRounds"));
    try std.testing.expect(@typeInfo(@TypeOf(dsv41.native.headRounds)).@"fn".return_type.? == @FieldType(Host, "head_rounds"));
    const src = @embedFile("cuda.zig");
    const read = "if (@hasDecl(F, \"headRounds\")) h.head_rounds = F.headRounds(loaded.ctx);";
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, read)); // openWith's (this text escapes its quotes)
}

test "serve: the family's decode rounds by the head's path reach status; a family without them reports none" {
    const gpa = std.testing.allocator;
    const Fam = struct {
        fn release(_: *anyopaque) void {}
    };
    var rounds: [6]std.atomic.Value(u64) = .{ .init(1), .init(2), .init(3), .init(4), .init(5), .init(6) };
    for ([_]bool{ false, true }) |counted| {
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        const h = try gpa.create(Host);
        defer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .gpu = null, .family = &target, .release = Fam.release, .inner = target.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
        if (counted) h.head_rounds = &rounds;
        try h.serve(std.testing.io, .{ .exact_width = 8 }, 8, .{ .lanes = 2 }, null);
        defer {
            h.host.stop();
            h.core.deinit();
            h.cfg.deinit(gpa);
        }
        var st: api.Status = .{};
        h.host.engine().status(&st, &.{});
        try std.testing.expectEqual(if (counted) @as(?[6]u64, .{ 1, 2, 3, 4, 5, 6 }) else null, st.head_rounds);
    }
}

test "a close without the orderly close still runs the family's finish, once, before its release" {
    const gpa = std.testing.allocator;
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var fam: CloseFamily = .{};
    const h = try fam.host(gpa, &target);
    Host.close(h);
    try std.testing.expectEqual(@as(u32, 1), fam.state()[0]);
    try std.testing.expectEqual(@as(u32, 1), fam.released);
}

test "a stop in a 1M-token prompt pass: the pass ends at its next chunk, then the family's finish, well inside the close's bound" {
    const gpa = std.testing.allocator;
    var fam: CloseFamily = .{};
    // 512 chunks of 2048 tokens (a 1M-token prompt), 10 ms each here: 5 s whole
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 512, .prefill_hook = CloseFamily.chunk, .prefill_hook_ctx = &fam };
    defer target.deinit();
    const h = try fam.host(gpa, &target);
    defer Host.close(h);
    const e = h.host.engine();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var box: CloseBox = .{};
    try e.submit(1, &.{ .prompt = &prompt, .max_tokens = 64 }, .{ .ctx = &box, .event = CloseBox.event });
    while (fam.state()[1] < 3) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    Host.quiesce(h);
    const took_ms = @divTrunc(std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - began, std.time.ns_per_ms);
    const seen = fam.state();
    try std.testing.expectEqual(@as(?api.Reason, .failed), box.state()[0]); // its pass cut by the stop
    try std.testing.expectEqual(@as(u32, 1), seen[0]);
    try std.testing.expect(seen[1] < 16); // the pass stopped at a chunk end, not after its 512
    try std.testing.expectEqual(@as(usize, 0), seen[2]); // finish (dsv41: done to the followers) after the pass ended
    try std.testing.expect(took_ms < 1000); // TENSORFOLD_CLOSE_MS is 20,000 by default
}

test "a cancel during a long prompt stops it, and the next request's reply is unchanged (#291)" {
    const gpa = std.testing.allocator;
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 10 };
    defer target.deinit();
    const Nothing = struct {
        fn release(_: *anyopaque) void {}
    };
    const h = try gpa.create(Host);
    defer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = null, .family = &target, .release = Nothing.release, .inner = target.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    try h.serve(std.testing.io, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 7, .hidden_rows = true }, 8, .{ .lanes = 2 }, null);
    defer {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(gpa);
    }
    const e = h.host.engine();
    try std.testing.expect(e.memory(false) == null); // no device to count
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?api.Reason = null,
        fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (ev.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) api.Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const Breaker = struct {
        engine: api.Engine,
        id: api.Id,
        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == 2 and c.id != 0) c.engine.cancel(c.id);
        }
    };
    const prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 40 };
    var breaker: Breaker = .{ .engine = e, .id = 1 };
    target.prefill_hook = Breaker.call;
    target.prefill_hook_ctx = &breaker;
    var broken: Box = .{};
    defer broken.tokens.deinit(gpa);
    try e.submit(1, &request, .{ .ctx = &broken, .event = Box.event });
    try std.testing.expectEqual(api.Reason.cancelled, broken.wait());
    try std.testing.expect(target.prefill_count <= 3); // stopped at the next chunk, not after all ten
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released
    try std.testing.expectEqual(@as(usize, 0), broken.tokens.items.len);
    breaker.id = 0;
    var restored: Box = .{};
    defer restored.tokens.deinit(gpa);
    try e.submit(2, &request, .{ .ctx = &restored, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, restored.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (restored.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 40), restored.tokens.items.len);
}

test "the CUDA context wrapper forwards bounded prefill and keeps its capability optional" {
    const gpa = std.testing.allocator;
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var h: Host = undefined;
    h.gpu = null;
    h.inner = target.backend();
    try std.testing.expect(h.backend().vtable.prefill_step == null);
    h.inner = target.chunkedBackend();
    const be = h.backend();
    var s = try lanes.Stream.init(gpa, .{ .id = "partial", .prompt = &.{ 1, 2, 3, 4, 5 }, .max_new = 1 });
    defer s.deinit(gpa);
    defer be.release(&s);
    try std.testing.expect(!try be.vtable.prefill_step.?(be.ptr, &s));
    try std.testing.expect(try be.vtable.prefill_step.?(be.ptr, &s));
}

test "bindTier: the tier on the prompt cache with the family's grid, replay window and usable rule; a store without one is the RAM cache alone" {
    const gpa = std.testing.allocator;
    const pc = api.prompt_cache;
    const Snap = struct {
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 1;
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
            return error.Unused;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: pc.Saved) void {}
        fn usable(_: *anyopaque, _: *anyopaque, _: u32) bool {
            return true;
        }
    };
    const Stub = struct { // a tier that answers nothing: only its presence matters here
        fn lookup(_: *anyopaque, _: *const pc.Want) ?pc.Point {
            return null;
        }
        fn begin(_: *anyopaque, _: *const pc.Point) anyerror!u64 {
            return error.Unused;
        }
        fn poll(_: *anyopaque, _: u64, _: u64) pc.Poll {
            return .{ .pending = .{} };
        }
        fn cancel(_: *anyopaque, _: u64) void {}
        fn keep(_: *anyopaque, _: *const pc.Kept) ?u64 {
            return null;
        }
        fn forget(_: *anyopaque, _: u64) void {}
        fn flush(_: *anyopaque, _: u64) u64 {
            return 0;
        }
        fn counts(_: *anyopaque, _: []pc.TierCounts) usize {
            return 0;
        }
        const vt: pc.Tier.VTable = .{ .lookup = lookup, .begin = begin, .poll = poll, .cancel = cancel, .keep = keep, .forget = forget, .flush = flush, .counts = counts };
    };
    var family: u8 = 0;
    var tier_state: u8 = 0;
    var vt: pc.Snapshots.VTable = .{ .bytes = Snap.bytes, .save = Snap.save, .restore = Snap.restore, .drop = Snap.drop };
    // without a tier (TF_DS_SPILL_GIB=0): the window from the family's tail, no grid, no usable rule, no tier, no charge
    var plain = pc.Store.init(gpa, .{ .ptr = &family, .vtable = &vt }, .{ .window = 128 }, 1 << 20);
    defer plain.deinit();
    try std.testing.expectEqual(@as(u32, 128), plain.rules.window);
    try std.testing.expectEqual(@as(u32, 0), plain.rules.grid);
    try std.testing.expect(vt.usable == null and plain.tier == null and !plain.groups and !plain.charge_tokens);
    // with one: exactly those change (the replay window is the family's, whatever the store started with)
    var store = pc.Store.init(gpa, .{ .ptr = &family, .vtable = &vt }, .{ .window = 0 }, 1 << 20);
    defer store.deinit();
    bindTier(&store, &vt, .{ .ptr = &tier_state, .vtable = &Stub.vt }, 2048, 128, Snap.usable, std.testing.io);
    try std.testing.expectEqual(@as(u32, 2048), store.rules.grid);
    try std.testing.expectEqual(@as(u32, 128), store.rules.window);
    try std.testing.expect(vt.usable == Snap.usable);
    try std.testing.expect(store.tier != null and store.tier.?.ptr == @as(*anyopaque, &tier_state));
    try std.testing.expect(store.charge_tokens);
    // conversations and lineages numbered from this process's seed (odd: never 0), in the high half
    try std.testing.expect(store.groups);
    try std.testing.expect(store.next_group != 0 and store.next_group & 0xffff_ffff == 0 and (store.next_group >> 32) & 1 == 1);
    try std.testing.expectEqual(store.next_group, store.next_lineage);
}

test "serve: the family's quiet hook and the tier's settings reach the lane host, its start's off reason reaches status; a family without them leaves the host unchanged" {
    const gpa = std.testing.allocator;
    const Fam = struct {
        quiet_calls: u32 = 0,
        fn release(_: *anyopaque) void {}
        fn quiet(p: *anyopaque) bool {
            const t: *lanes.fake.Fake = @ptrCast(@alignCast(p));
            _ = t;
            calls += 1;
            return true;
        }
        var calls: u32 = 0;
    };
    var buf: [256]u8 = undefined;
    const Env = struct {
        pairs: []const [2][]const u8,
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            for (e.pairs) |q| if (std.mem.eql(u8, q[0], name)) return q[1];
            return null;
        }
    };
    for ([_]bool{ false, true }) |with_tier| {
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        const h = try gpa.create(Host);
        defer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .gpu = null, .family = &target, .release = Fam.release, .inner = target.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
        if (with_tier) {
            h.quiet = Fam.quiet;
            h.spill = api.Spill.fromEnv(Env{ .pairs = &.{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "/spill" }, .{ "TF_SPILL_LOAD_MS", "7" } } }, &buf);
            h.spill_start = api.SpillState.offFor("the store did not open within TF_DS_SPILL_OPEN_MS=300000");
        }
        try h.serve(std.testing.io, .{ .exact_width = 8 }, 8, .{ .lanes = 2 }, null);
        defer {
            h.host.stop();
            h.core.deinit();
            h.cfg.deinit(gpa);
        }
        var st: api.Status = .{};
        h.host.engine().status(&st, &.{});
        if (!with_tier) {
            // SPILL_GIB=0 (or another family): no fence hook, no tier settings, nothing under cache or spill
            try std.testing.expect(h.host.quiet == null and h.host.tele == null and !h.host.spill_asked);
            try std.testing.expect(st.cache == null and st.spill == null);
            continue;
        }
        try std.testing.expect(h.host.spill_asked);
        try std.testing.expectEqual(@as(u64, 7 * std.time.ns_per_ms), h.host.tier.service_ns); // TF_SPILL_LOAD_MS as given
        try std.testing.expectEqual(api.SpillState.State.off, st.spill.?.state);
        try std.testing.expectEqualStrings("the store did not open within TF_DS_SPILL_OPEN_MS=300000", st.spill.?.reason());
        // the fence's quiet check reaches the family's, with the host bound
        const q = h.host.quiet.?;
        const before = Fam.calls;
        try std.testing.expect(q.check(q.ctx));
        try std.testing.expectEqual(before + 1, Fam.calls);
    }
}

test "the tier with no prompt cache is off from the start, named, and the family opens nothing" {
    var buf: [256]u8 = undefined;
    const Env = struct {
        pairs: []const [2][]const u8,
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            for (e.pairs) |q| if (std.mem.eql(u8, q[0], name)) return q[1];
            return null;
        }
    };
    const sp = api.Spill.fromEnv(Env{ .pairs = &.{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "/spill" } } }, &buf);
    const none = familySpill(sp, 0); // --prompt-cache-gib 0
    try std.testing.expect(none.asked and none.off != null);
    try std.testing.expect(std.mem.indexOf(u8, none.off.?, "--prompt-cache-gib 0") != null);
    const some = familySpill(sp, 1 << 30); // a prompt cache: the settings as given
    try std.testing.expect(some.off == null);
    const bad = api.Spill.fromEnv(Env{ .pairs = &.{.{ "TF_DS_SPILL_GIB", "x" }} }, &buf);
    try std.testing.expectEqualStrings(bad.off.?, familySpill(bad, 0).off.?); // a bad setting keeps its own reason
    const unasked = familySpill(.{}, 0); // not asked: nothing changes
    try std.testing.expect(!unasked.asked and unasked.off == null);
}

test "the start-up line and /health carry the family's own off reason from its start" {
    var buf: [256]u8 = undefined;
    const Env = struct {
        pairs: []const [2][]const u8,
        pub fn get(e: @This(), name: [:0]const u8) ?[]const u8 {
            for (e.pairs) |q| if (std.mem.eql(u8, q[0], name)) return q[1];
            return null;
        }
    };
    const sp = api.Spill.fromEnv(Env{ .pairs = &.{ .{ "TF_DS_SPILL_GIB", "256" }, .{ "TF_DS_SPILL_DIR", "/spill" } } }, &buf);
    const si: api.SpillInfo = .{ .settings = sp };
    const off = api.SpillState.offFor("the start probe: rank 2's bytes differ from rank 0's");
    const shown = startInfo(si, &off).?;
    try std.testing.expectEqualStrings("the start probe: rank 2's bytes differ from rank 0's", shown.settings.off.?);
    var line: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&line);
    try shown.settings.describe(&w);
    try std.testing.expect(std.mem.indexOf(u8, w.buffered(), "spill: on") == null); // never "on" while it is off
    const on_state: api.SpillState = .{ .state = .on };
    try std.testing.expect(startInfo(si, &on_state).?.settings.off == null); // on: the settings as they were
    try std.testing.expect(startInfo(null, &off) == null); // not asked: none
}
