//! The speed settings the engine ships ON: every one measured on the four Sparks as a win with equal replies (LEDGER,
//! ACHIEVEMENTS), so an install without a settings file gets the speed the measured heads run.
//!
//! `apply` runs once at start, after engine.cfg and before anything reads the environment: a name the operator left unset
//! (environment and engine.cfg alike) gets the value below, in the process's own environment (which the parsers read) and in
//! the settings map; a name the operator set, to anything, is left as set (`TF_DS_X=0` is how a switch is turned off). Each
//! parser keeps its own fallbacks: a switch that cannot run says so at the start and the engine goes on without it.
//!
//! A switch whose precondition can fail (the stagger needs Rocinante, 4,096-row passes and memory; the four-way drafter needs the kit's
//! extra set; WOA_W8 the checkpoint's FP8 rows; the Engram partner split whole tables and packed shards; keep-ahead asynchronous
//! Engram reads) is here too, and `verdict` makes its failure a loud fallback when the operator did not set
//! it (`defaulted`), a refusal by name when the operator set it to 1. What cannot be undone once the weights load is the one exception:
//! ranks that loaded different drafters (a kit that differs between Sparks) still end the start, naming the cause. Not here: the
//! proxies' CPU pins (they need the machine's core map).

const std = @import("std");

pub const Default = struct {
    /// The environment name (the schema row's env).
    env: []const u8,
    /// The value the engine runs when the name is unset (the schema row's default).
    value: []const u8,
    /// The feature in plain words, for the start-up report and /status.
    what: []const u8,
    /// What the engine loses without it, as measured (the source is in the schema row's doc).
    cost: []const u8,
    /// Other values of the name that also run the feature (a width, a variant): the report lists them as ON, not as another choice.
    also_on: []const []const u8 = &.{},
};

pub const table = [_]Default{
    .{ .env = "TF_DS_PASS_ROWS", .value = "4096", .what = "4,096-row prompt passes", .cost = "long prompts run about 12 % slower (x1.139 measured; about 1.5 GiB more memory a node with it)" },
    .{ .env = "TF_DS_PASS_FROM", .value = "0", .what = "larger prompt passes from the first row", .cost = "fresh 4,000 to 19,000-token prompts start in small passes and finish slower" },
    .{ .env = "TF_DS_PASS_FILLING", .value = "1", .what = "whole passes while several prompts fill", .cost = "several long prompts at once take about 14 % longer to their first token (x1.159 aggregate measured)" },
    .{ .env = "TF_DS_PF_SPLIT", .value = "1", .what = "split FP8 prompt GEMMs on the restructured loop", .cost = "prompts about 1 % slower (x1.010 measured)" },
    .{ .env = "TF_DS_QUANT_ONCE", .value = "1", .what = "attention input quantized once", .cost = "prompts about 1 % slower with the next four kernel switches (x1.012 together)" },
    .{ .env = "TF_DS_GROUP_TILES", .value = "1", .what = "expert tile list built in one launch", .cost = "see TF_DS_QUANT_ONCE" },
    .{ .env = "TF_DS_TOPK_TILES", .value = "1", .what = "indexer top-k in one launch", .cost = "see TF_DS_QUANT_ONCE" },
    .{ .env = "TF_DS_BF16_ROWS", .value = "1", .what = "bf16 GEMMs by 16-row tile", .cost = "see TF_DS_QUANT_ONCE" },
    .{ .env = "TF_DS_PF_RUN128", .value = "1", .what = "expert tiles read 128-byte weight rows", .cost = "prompts about 0.7 % slower (x1.007 measured)" },
    .{ .env = "TF_DS_MMA_SF_EXACT", .value = "1", .what = "block-scaled FP8 GEMMs, exact arithmetic", .cost = "prompts about 2 % slower (x1.022 measured)" },
    .{ .env = "TF_DS_IDX_SHARD", .value = "1", .what = "indexer scored on each node's own quarter of the rows", .cost = "long prompts slower (not measured alone: the recipe's switches were measured as a bundle); several later switches need it" },
    .{ .env = "TF_DS_IDX_RB", .value = "2", .what = "indexer kernel at two rows a block", .cost = "prompts slower (not measured alone: the setting of every measured head)" },
    .{ .env = "TF_DS_IDX_PROJ_OWNED", .value = "1", .what = "indexer projections on owned rows", .cost = "see TF_DS_QUANT_ONCE" },
    .{ .env = "TF_DS_ATTN_U", .value = "1", .what = "prompt attention from packed latents (the kernel kit's extra set)", .cost = "prompts about 1 % slower (x1.011 measured)" },
    .{ .env = "TF_DS_ATTN_EPI_FUSED", .value = "1", .what = "prompt attention epilogue (inverse RoPE, wo_a quant) in the attention kernel (the kernel kit's extra set; needs TF_DS_ATTN_EPILOGUE)", .cost = "prompts slower by the epilogue launch and a pass over the attention output (estimated 10-12 ms a 4,096-row pass; not measured)" },
    .{ .env = "TF_DS_HC_2S", .value = "1", .what = "prompt mHC mix on the two-block sums, four streams in one program (the kernel kit's extra set)", .cost = "the mix reads the previous streams and sums four times from L2 (estimated 4-10 ms a 4,096-row pass; not measured)" },
    .{ .env = "TF_DS_IDX_U", .value = "1", .what = "indexer scores from packed keys (the kernel kit's extra set)", .cost = "prompts about 1.6 % slower (x1.016 measured)" },
    .{ .env = "TF_DS_SAMPLE_CAND", .value = "1", .what = "candidate sampling on the GPU", .cost = "sampled decode slower (not measured alone: the recipe's switches were measured as a bundle)" },
    .{ .env = "TF_DS_EXPERT_EXIT", .value = "1", .what = "decode expert kernels skip idle rows early", .cost = "decode slower (not measured alone: the recipe's switches were measured as a bundle)" },
    .{ .env = "TF_DS_PREFILL_YIELD", .value = "1", .what = "other requests get a round between a long prompt's chunks", .cost = "a short request during a long prompt waits for the whole prompt (admission latency; its speed cost was not measured alone)" },
    .{ .env = "TF_DS_ENGRAM_AHEAD_PASS", .value = "1", .what = "Engram rows read a layer ahead for a prompt pass", .cost = "the first prompt after a start is 21 % slower (5,457 against 4,330 tokens/s measured)" },
    .{ .env = "TF_DS_ENGRAM_PREFAULT", .value = "1", .what = "Engram row cache faulted in at the start", .cost = "the first prompt after a start pays for the page faults" },
    .{ .env = "TF_DS_ENGRAM_PINNED", .value = "1", .what = "Engram rows staged in pinned memory", .cost = "the host thread waits on every Engram upload; the two-pass stagger cannot run" },
    .{ .env = "TF_DS_ENGRAM_AIO", .value = "1", .what = "Engram reads by asynchronous disk I/O", .cost = "Engram reads block the engine thread (not measured alone: the setting of every measured head)" },
    .{ .env = "TF_DS_ISSUE_LEAD", .value = "4", .what = "the engine thread sleeps instead of spinning in a full launch queue", .cost = "one core spins at 100 % on every Spark during prompts (speed unchanged)" },
    .{ .env = "TF_DS_DRAFT_VOCAB", .value = "1", .what = "drafter proposes from a list of 32,768 token ids (8,192 a node)", .cost = "a token outside the list is never drafted (misses 0.1 to 0.4 % of tokens held out, ESTIMATED offline; a guard returns to the whole head past 2 %); the speed was not measured alone" },
    .{ .env = "TF_DS_DRAFT_HEAD", .value = "fp4", .what = "drafter output head in FP4", .cost = "one-stream decode about 1 to 2 % slower (+1.9 % prose, +1.0 % code measured)" },
    .{ .env = "TF_DS_QKV_PLAIN", .value = "1", .what = "a decode window's q norm and window KV in one launch (the kernel kit's extra set)", .cost = "one launch more a layer (about 0.1 to 0.15 ms of a 28 ms decode round, estimated; not measured alone)" },
    .{ .env = "TF_DS_COPY_MATCH", .value = "16", .what = "copied continuations need 16 matching tokens", .cost = "decode slower on repeated text (part of the x1.036 / x1.046 release)" },
    .{ .env = "TF_DS_COPY_ROWS", .value = "17", .what = "copied continuation up to 17 rows in one round", .cost = "copy-heavy text decodes up to 34 % slower (x1.34 measured)" },
    .{ .env = "TF_DS_RDMA_COLLECT_PEERS", .value = "1", .what = "decode exchanges read every peer at once", .cost = "one-stream decode about 1.3 % slower with the next three switches (x1.013 together)" },
    .{ .env = "TF_DS_CAND_FIRST", .value = "1", .what = "candidates read back before the drafter's absorb", .cost = "see TF_DS_RDMA_COLLECT_PEERS" },
    .{ .env = "TF_DS_HEAD_CAND", .value = "1", .what = "each node ranks its quarter of the vocabulary (sampled)", .cost = "sampled decode about 0.7 % slower (x1.0074 measured)" },
    .{ .env = "TF_DS_HEAD_KEY", .value = "1", .what = "each node picks its quarter's best token (greedy)", .cost = "greedy code decode about 1.4 % slower (x1.0135 measured)" },
    .{ .env = "TF_DS_DEC_GROUP2", .value = "1", .what = "decode expert grouping by a parallel scan", .cost = "each decode round about 0.1 ms longer (estimated, not measured; the grouping bytes are equal)" },
    .{ .env = "TF_DS_MERGE_D", .value = "3", .what = "decode attention merge in column blocks (the kernel kit's extra set)", .cost = "each decode round about 0.06 to 0.08 ms longer with 128-column blocks (TF_DS_MERGE_D=1) and 0.15 to 0.25 ms longer with the single block (estimated from kernel timings, not measured in a round; the bytes are equal)", .also_on = &.{ "1", "2" } },
    .{ .env = "TF_DS_ATTN_U_DECODE", .value = "1", .what = "decode attention from packed latents (the kernel kit's extra set)", .cost = "the decode half of TF_DS_ATTN_TWIN=z does not run (alone x0.997 / x1.001 measured, under the noise floor; the same bytes)" },
    .{ .env = "TF_DS_ATTN_TWIN", .value = "z", .what = "sparse attention from the kit's z twin (fewer instructions; decode picks prefetched into L2)", .cost = "the standard u kernel (not measured on the Sparks; the same bytes, kernels replay)" },
    .{ .env = "TF_DS_ROPE_W", .value = "1", .what = "RoPE on 32-bit bf16 pairs (the kit's _rope_heads_w)", .cost = "the standard _rope_heads (not measured on the Sparks; the same bytes, kernels replay)" },
    .{ .env = "TF_DS_IDX_TPP", .value = "4", .what = "indexer score at 4 key tiles a program, the next tile prefetched into L1 (the kernel kit's batch 8)", .cost = "the indexer score kernel runs about 3 % longer (t/u 0.958 to 0.980 at the 9 production shapes of a 4,096-row pass, kernel time measured on the Spark; about 0.8 ms a pass, estimated; the same bytes)" },
    .{ .env = "TF_DS_WOA_MERGE", .value = "1", .what = "wo_a's two groups of a prompt chunk's layer in one launch (grid.z = groups)", .cost = "the wo_a launches run about 5.5 % longer a layer (merged / two launches 0.945 per layer, null 0.998, event time measured on the Spark, byte-identical; about 2.1 ms a 4,096-row pass, the wave estimate of 3.9 was high: the merge trims the tail wave, it does not remove it)" },
    .{ .env = "TF_DS_AQ_VEC", .value = "1", .what = "activation quantizer on four lanes a 32-value group (16-byte loads)", .cost = "the prompt pass's quantizer launches run about 11 % longer on the big shapes and up to 3.7 times on the 1,024-row ones (kernel times measured on the Spark, the same bytes; not measured in a pass)" },
    .{ .env = "TF_DS_SPLIT64N", .value = "1", .what = "wkv's and wq_a's exact split GEMMs of a prompt chunk on a 64 x 64 tile (two blocks an SM)", .cost = "the two launches take about 27 % / 22 % more time at a 4,096-row pass's 1,024 owned rows (64 x 128 tile over 64 x 64: wkv 0.1080 / 0.0849 ms, wq_a 0.2170 / 0.1781 ms a launch, null control 0.0 %, kernel time measured on the Spark, the same bytes); about 2.5 ms a 4,096-row pass at one launch a layer" },
    .{ .env = "TF_DS_PACK_EPILOGUE", .value = "1", .what = "gate/up writes the expert pack's codes itself", .cost = "a 4,096-row prompt pass about 5 to 6 ms longer (estimated, not measured; the quantizing launch runs again)" },
    .{ .env = "TF_DS_ROCI_STAGGER", .value = "1", .what = "two prompt passes in flight (stagger)", .cost = "long prompts about 12 % slower (x1.134 to x1.152 measured)" },
    .{ .env = "TF_DS_ROCI_P2_FIRST", .value = "1", .what = "Rocinante: second part sent first", .cost = "prompts a few percent slower (the Rocinante orderings together x1.054 to x1.069 measured, not one by one)" },
    .{ .env = "TF_DS_ROCI_ENGRAM_AHEAD", .value = "1", .what = "Rocinante: Engram a layer ahead", .cost = "see TF_DS_ROCI_P2_FIRST" },
    .{ .env = "TF_DS_ROCI_COMBINE_DOOR", .value = "1", .what = "Rocinante: sum and signal in one launch", .cost = "see TF_DS_ROCI_P2_FIRST" },
    .{ .env = "TF_DS_DRAFT_4WAY", .value = "1", .what = "drafter split four ways (the kernel kit's extra set)", .cost = "decode about 3 to 4 % slower (x1.029 one stream, x1.040 16 streams measured); about 2 GB more a Spark" },
    .{ .env = "TF_DS_DRAFT_MAIN_4WAY", .value = "1", .what = "drafter main projection four ways", .cost = "one-stream decode slower (part of the x1.013 decode release)" },
    .{ .env = "TF_DS_WOA_W8", .value = "1", .what = "attention output from FP8 rows", .cost = "about 0.67 GB more memory a Spark (speed unchanged)" },
    .{ .env = "TF_DS_RDMA_WFE", .value = "1", .what = "RDMA proxy threads nap instead of polling", .cost = "proxy threads busy at 100 % of a core each (speed unchanged, 3 to 7 C hotter)" },
    .{ .env = "TF_DS_ATTN_EPILOGUE", .value = "1", .what = "attention output's inverse RoPE and quant in one launch", .cost = "one launch and one memory pass more per layer (no speed difference measured: net -0.3 ms of a ~600 ms pass)" },
    .{ .env = "TF_DS_HC_STRIDE", .value = "1", .what = "quarter-HC mixes read the gathered planes in place", .cost = "a repack copy more where owner sums do not already avoid it (no speed difference measured)" },
    .{ .env = "TF_DS_RING_WC", .value = "1", .what = "decode rings on write-combined slots", .cost = "the in-place ring cannot run (no speed difference measured: decode x1.000 one stream, x0.994 at 16)" },
    .{ .env = "TF_DS_RING_INPLACE", .value = "1", .what = "decode exchanges write straight into the ring's send slots", .cost = "one staging copy more per decode exchange (no speed difference measured: decode x0.9965 one stream, x0.9985 at 16)" },
    .{ .env = "TF_DS_ENGRAM_KEEP_AHEAD", .value = "1", .what = "the Engram reader stays across decode rounds between a prompt's chunks", .cost = "prompts beside generating requests about 1 % slower (+0.5 to +1.5 % measured)" },
    .{ .env = "TF_DS_ENGRAM_PARTNER", .value = "1", .what = "column partners split each other's Engram disk reads", .cost = "every Spark reads all its own Engram rows from disk: twice the reads (speed equal measured)" },
    .{ .env = "TF_DS_ROCI_IDLE_PARK_US", .value = "2000", .what = "idle Rocinante proxies sleep in the kernel", .cost = "idle proxies busy-wait on their doorbells (speed and first-request latency unchanged measured; hotter)" },
};

/// Which of `table` this process applied (set at start, read by the speed report).
pub var applied: [table.len]bool = @splat(false);

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// Gives every unset name in `table` its value, in `map` and, with `libc`, in the process's own environment. A name that is set
/// (even to an empty or an unknown value) is left alone. Returns how many it applied.
pub fn apply(gpa: std.mem.Allocator, map: *std.process.Environ.Map, comptime libc: bool) usize {
    var n: usize = 0;
    for (table, 0..) |d, i| {
        if (map.get(d.env) != null) continue;
        map.put(d.env, d.value) catch continue;
        if (libc) {
            const name = gpa.dupeSentinel(u8, d.env, 0) catch continue;
            const val = gpa.dupeSentinel(u8, d.value, 0) catch continue;
            _ = setenv(name.ptr, val.ptr, 0);
        }
        applied[i] = true;
        n += 1;
    }
    return n;
}

/// Whether the operator left `env` unset, so the default gave it its value (a switch whose precondition fails then falls back).
pub fn defaulted(env: []const u8) bool {
    const i = find(env) orelse return false;
    return applied[i];
}

pub const Verdict = enum { run, fallback, refuse };

/// What a failed precondition does: nothing to say (`why` null) runs; the default falls back; a switch the operator set refuses by name.
pub fn verdict(why: ?[]const u8, is_default: bool) Verdict {
    if (why == null) return .run;
    return if (is_default) .fallback else .refuse;
}

/// What the engine loses without `env`'s feature, for the fallback's one line.
pub fn costOf(env: []const u8) []const u8 {
    const i = find(env) orelse return "";
    return table[i].cost;
}

/// The DeepSeek engine's own process settings, applied by its start (not the shared CUDA context's, which other families use): host
/// threads and event waits sleep for the GPU (heat, speed unchanged), and 16 hardware queues, which the stagger's second pass needs.
pub const family_table = [_]struct { env: []const u8, value: []const u8 }{
    .{ .env = "TF_DS_CUDA_SCHED", .value = "blocking" },
    .{ .env = "TF_DS_CUDA_EVENTS", .value = "blocking" },
    .{ .env = "CUDA_DEVICE_MAX_CONNECTIONS", .value = "16" },
};

/// Gives the unset names of `family_table` their values in the process's own environment (before the CUDA context is made).
pub fn applyFamily() void {
    for (family_table) |d| {
        var nb: [64:0]u8 = undefined;
        var vb: [32:0]u8 = undefined;
        if (d.env.len >= nb.len or d.value.len >= vb.len) continue;
        @memcpy(nb[0..d.env.len], d.env);
        nb[d.env.len] = 0;
        @memcpy(vb[0..d.value.len], d.value);
        vb[d.value.len] = 0;
        _ = setenv(&nb, &vb, 0);
    }
}

/// The row for an environment name, or null.
pub fn find(env: []const u8) ?usize {
    for (table, 0..) |d, i| if (std.mem.eql(u8, d.env, env)) return i;
    return null;
}

test "every name here has a schema row whose default is this value" {
    const settings = @import("settings.zig");
    for (table) |d| {
        const o = settings.byEnv(d.env) orelse return error.NoSchemaRow;
        std.testing.expectEqualStrings(d.value, o.default) catch |e| {
            std.debug.print("schema default of {s} is '{s}', speed_defaults says '{s}'\n", .{ d.env, o.default, d.value });
            return e;
        };
    }
}

test "apply gives an unset name its value and leaves a set one alone, an empty value and an off value included" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var map = std.process.Environ.Map.init(a);
    defer map.deinit();
    try map.put("TF_DS_PF_SPLIT", "0");
    try map.put("TF_DS_PASS_ROWS", "");
    const n = apply(arena.allocator(), &map, false);
    try std.testing.expectEqual(table.len - 2, n);
    try std.testing.expectEqualStrings("0", map.get("TF_DS_PF_SPLIT").?);
    try std.testing.expectEqualStrings("", map.get("TF_DS_PASS_ROWS").?);
    try std.testing.expectEqualStrings("1", map.get("TF_DS_QUANT_ONCE").?);
    try std.testing.expectEqualStrings("fp4", map.get("TF_DS_DRAFT_HEAD").?);
    try std.testing.expectEqualStrings("4", map.get("TF_DS_ISSUE_LEAD").?);
    try std.testing.expect(!applied[find("TF_DS_PF_SPLIT").?]);
    try std.testing.expect(applied[find("TF_DS_QUANT_ONCE").?]);
}

test "apply with the process's own environment compiles and, with every name set, changes nothing" {
    const a = std.testing.allocator;
    var map = std.process.Environ.Map.init(a);
    defer map.deinit();
    for (table) |d| try map.put(d.env, "x");
    try std.testing.expectEqual(@as(usize, 0), apply(a, &map, true));
}

test "a failed precondition falls back when the operator set nothing and refuses by name when the operator set the switch" {
    try std.testing.expectEqual(Verdict.run, verdict(null, true));
    try std.testing.expectEqual(Verdict.run, verdict(null, false));
    try std.testing.expectEqual(Verdict.fallback, verdict("needs the rails' devices", true));
    try std.testing.expectEqual(Verdict.refuse, verdict("needs the rails' devices", false));
}

test "the switches that would refuse are defaulted, the shared context's are not" {
    for ([_][]const u8{ "TF_DS_ROCI_STAGGER", "TF_DS_ROCI_P2_FIRST", "TF_DS_ROCI_ENGRAM_AHEAD", "TF_DS_ROCI_COMBINE_DOOR", "TF_DS_DRAFT_4WAY", "TF_DS_DRAFT_MAIN_4WAY", "TF_DS_WOA_W8" }) |n|
        try std.testing.expect(find(n) != null);
    for ([_][]const u8{ "TF_DS_CUDA_SCHED", "TF_DS_CUDA_EVENTS" }) |n| try std.testing.expect(find(n) == null);
    try std.testing.expect(!defaulted("TF_DS_ROCI_STAGGER") or applied[find("TF_DS_ROCI_STAGGER").?]);
    try std.testing.expect(costOf("TF_DS_ROCI_STAGGER").len > 0);
}

test "the kit's bit-identical attention twins are on by default: TF_DS_ATTN_TWIN z, TF_DS_ROPE_W, and TF_DS_ATTN_U_DECODE (z's decode half runs only in its calls)" {
    try std.testing.expectEqualStrings("z", table[find("TF_DS_ATTN_TWIN") orelse return error.NotDefaulted].value);
    try std.testing.expectEqualStrings("1", table[find("TF_DS_ROPE_W") orelse return error.NotDefaulted].value);
    try std.testing.expectEqualStrings("1", table[find("TF_DS_ATTN_U_DECODE") orelse return error.NotDefaulted].value);
}

test "the bit-identical switches the heads ran by the protocol are defaulted, with their fallbacks named" {
    for ([_][]const u8{ "TF_DS_ATTN_EPILOGUE", "TF_DS_HC_STRIDE", "TF_DS_RING_WC", "TF_DS_RING_INPLACE", "TF_DS_ENGRAM_KEEP_AHEAD", "TF_DS_ENGRAM_PARTNER", "TF_DS_ROCI_IDLE_PARK_US" }) |n| {
        try std.testing.expect(find(n) != null);
        try std.testing.expect(costOf(n).len > 0);
    }
    // the bit-changing switches stay opt-in, and so do the ones measured without effect or slower
    for ([_][]const u8{ "TF_DS_MMA_SF", "TF_DS_DECODE_KSPLIT", "TF_DS_HC_FUSED", "TF_DS_DECODE_FUSE", "TF_DS_ROCI_SENT_FIRST" }) |n| try std.testing.expect(find(n) == null);
}

test "the four-lane act_quant (the same bytes, proven at kernel level) is on by default with its cost named" {
    try std.testing.expectEqualStrings("1", table[find("TF_DS_AQ_VEC") orelse return error.NotDefaulted].value);
    try std.testing.expect(costOf("TF_DS_AQ_VEC").len > 0);
}

test "the indexer's four-tile score twin (the same bytes, measured faster at kernel level) is on by default with its cost named" {
    try std.testing.expectEqualStrings("4", table[find("TF_DS_IDX_TPP") orelse return error.NotDefaulted].value);
    try std.testing.expect(costOf("TF_DS_IDX_TPP").len > 0);
}

test "the 64-column split tile (byte-identical, measured 17.9 to 21.4 % less kernel time) is on by default with its cost named" {
    try std.testing.expectEqualStrings("1", table[find("TF_DS_SPLIT64N") orelse return error.NotDefaulted].value);
    try std.testing.expect(costOf("TF_DS_SPLIT64N").len > 0);
}

test "the merged wo_a launch (byte-identical, measured x0.945 a layer) is on by default with its cost named" {
    try std.testing.expectEqualStrings("1", table[find("TF_DS_WOA_MERGE") orelse return error.NotDefaulted].value);
    try std.testing.expect(costOf("TF_DS_WOA_MERGE").len > 0);
}
