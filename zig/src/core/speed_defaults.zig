    /// Other values of the name that also run the feature (a width, a variant): the report lists them as ON, not as another choice.
    also_on: []const []const u8 = &.{},
    .{ .env = "TF_DS_ATTN_EPI_FUSED", .value = "1", .what = "prompt attention epilogue (inverse RoPE, wo_a quant) in the attention kernel (the kernel kit's extra set; needs TF_DS_ATTN_EPILOGUE)", .cost = "prompts slower by the epilogue launch and a pass over the attention output (estimated 10-12 ms a 4,096-row pass; not measured)" },
    .{ .env = "TF_DS_HC_2S", .value = "1", .what = "prompt mHC mix on the two-block sums, four streams in one program (the kernel kit's extra set)", .cost = "the mix reads the previous streams and sums four times from L2 (estimated 4-10 ms a 4,096-row pass; not measured)" },
    .{ .env = "TF_DS_QKV_PLAIN", .value = "1", .what = "a decode window's q norm and window KV in one launch (the kernel kit's extra set)", .cost = "one launch more a layer (about 0.1 to 0.15 ms of a 28 ms decode round, estimated; not measured alone)" },
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
    .{ .env = "TF_DS_ROCI_P2_FIRST", .value = "1", .what = "Rocinante: second part sent first", .cost = "prompts a few percent slower (the Rocinante orderings together x1.054 to x1.069 measured, not one by one)" },
    .{ .env = "TF_DS_ROCI_COMBINE_DOOR", .value = "1", .what = "Rocinante: sum and signal in one launch", .cost = "see TF_DS_ROCI_P2_FIRST" },
    .{ .env = "TF_DS_DRAFT_4WAY", .value = "1", .what = "drafter split four ways (the kernel kit's extra set)", .cost = "decode about 3 to 4 % slower (x1.029 one stream, x1.040 16 streams measured); about 2 GB more a Spark" },
    .{ .env = "TF_DS_WOA_W8", .value = "1", .what = "attention output from FP8 rows", .cost = "about 0.67 GB more memory a Spark (speed unchanged)" },
    .{ .env = "TF_DS_ATTN_EPILOGUE", .value = "1", .what = "attention output's inverse RoPE and quant in one launch", .cost = "one launch and one memory pass more per layer (no speed difference measured: net -0.3 ms of a ~600 ms pass)" },
    .{ .env = "TF_DS_RING_WC", .value = "1", .what = "decode rings on write-combined slots", .cost = "the in-place ring cannot run (no speed difference measured: decode x1.000 one stream, x0.994 at 16)" },
    .{ .env = "TF_DS_RING_INPLACE", .value = "1", .what = "decode exchanges write straight into the ring's send slots", .cost = "one staging copy more per decode exchange (no speed difference measured: decode x0.9965 one stream, x0.9985 at 16)" },
    .{ .env = "TF_DS_ROCI_IDLE_PARK_US", .value = "2000", .what = "idle Rocinante proxies sleep in the kernel", .cost = "idle proxies busy-wait on their doorbells (speed and first-request latency unchanged measured; hotter)" },
pub const Verdict = enum { run, fallback, refuse };

pub fn verdict(why: ?[]const u8, is_default: bool) Verdict {
    if (why == null) return .run;
/// What the engine loses without `env`'s feature, for the fallback's one line.
    return table[i].cost;
}

/// The DeepSeek engine's own process settings, applied by its start (not the shared CUDA context's, which other families use): host
    .{ .env = "TF_DS_CUDA_SCHED", .value = "blocking" },
        var nb: [64:0]u8 = undefined;
        var vb: [32:0]u8 = undefined;
        if (d.env.len >= nb.len or d.value.len >= vb.len) continue;
    try std.testing.expectEqual(Verdict.run, verdict(null, false));
    try std.testing.expectEqual(Verdict.fallback, verdict("needs the rails' devices", true));
    try std.testing.expectEqual(Verdict.refuse, verdict("needs the rails' devices", false));
}


test "the kit's bit-identical attention twins are on by default: TF_DS_ATTN_TWIN z, TF_DS_ROPE_W, and TF_DS_ATTN_U_DECODE (z's decode half runs only in its calls)" {
    try std.testing.expectEqualStrings("z", table[find("TF_DS_ATTN_TWIN") orelse return error.NotDefaulted].value);
    try std.testing.expectEqualStrings("1", table[find("TF_DS_ROPE_W") orelse return error.NotDefaulted].value);
    try std.testing.expectEqualStrings("1", table[find("TF_DS_ATTN_U_DECODE") orelse return error.NotDefaulted].value);
}

test "the bit-identical switches the heads ran by the protocol are defaulted, with their fallbacks named" {

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
