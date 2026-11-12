//! Prometheus text for GET /metrics, family for family as the Python server writes it.
const std = @import("std");
const process = @import("process_memory.zig");
const api = @import("engine_api");

const prefix = "tensorfold:";
/// Upper edges shared by the request and time-to-first-token histograms; +Inf is added when rendered.
pub const buckets = [_]f64{ 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0 };
pub const tpot_buckets = [_]f64{ 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0 };

fn HistogramType(comptime edges: []const f64) type {
    return struct {
        pub const bucket_edges = edges;
        counts: [edges.len + 1]u64 = @splat(0),
        total: f64 = 0,
        n: u64 = 0,

        pub fn observe(h: *@This(), raw: f64) void {
            const value = @max(0, raw);
            h.n += 1;
            h.total += value;
            for (edges, 0..) |edge, i| if (value <= edge) {
                h.counts[i] += 1;
                return;
            };
            h.counts[edges.len] += 1;
        }
    };
}

pub const Histogram = HistogramType(&buckets);
pub const TpotHistogram = HistogramType(&tpot_buckets);

/// One depth of the draft head's chains over finished requests (api.DraftDepth's counts).
pub const Depth = struct { drafted: u64 = 0, reached: u64 = 0, kept: u64 = 0 };

/// What the stats line compares: finished requests' counts, now and at the last line.
const Totals = struct {
    replies: u64 = 0,
    generation: u64 = 0,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    depths: [api.draft_depths]Depth = @splat(.{}),

    fn since(t: Totals, base: Totals) Totals {
        var out: Totals = .{ .replies = t.replies - base.replies, .generation = t.generation - base.generation, .rounds = t.rounds - base.rounds, .drafted = t.drafted - base.drafted, .accepted = t.accepted - base.accepted };
        for (&out.depths, t.depths, base.depths) |*o, d, b| o.* = .{ .drafted = d.drafted - b.drafted, .reached = d.reached - b.reached, .kept = d.kept - b.kept };
        return out;
    }
};

/// Counters of finished requests; gauges are read from the engine at scrape time.
pub const Metrics = struct {
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    prompt: u64 = 0,
    cached: u64 = 0,
    generation: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    depths: [api.draft_depths]Depth = @splat(.{}),
    rounds: u64 = 0,
    plans: api.Plans = .{}, // those rounds, drafted and kept drafts by the window's plan
    disconnects: u64 = 0,
    latency: Histogram = .{},
    ttft: Histogram = .{},
    decode: Histogram = .{},
    prefill: Histogram = .{},
    tpot: TpotHistogram = .{},
    prepare: TpotHistogram = .{}, // each request's render and tokenization (its pictures prepared too)
    requests: std.ArrayList(struct { key: []const u8, status: u16, count: u64 }) = .empty,
    /// The stats line (TENSORFOLD_STATS_EVERY): when the last one went out (null: lines off), and the totals then.
    line_at: ?i96 = null,
    line_base: Totals = .{},

    pub fn deinit(m: *Metrics) void {
        for (m.requests.items) |r| m.gpa.free(r.key);
        m.requests.deinit(m.gpa);
    }

    pub fn note(m: *Metrics, io: std.Io, prompt: usize, cached: usize, generation: usize, drafted: u64, accepted: u64, rounds: u64, plans: api.Plans, latency: f64, ttft: ?f64, decode: ?f64, prefill: ?f64, tpot: ?f64) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.prompt += prompt;
        m.cached += cached;
        m.generation += generation;
        m.drafted += drafted;
        m.accepted += accepted;
        m.rounds += rounds;
        inline for (.{ "copy", "head" }) |k| {
            @field(m.plans, k).rounds += @field(plans, k).rounds;
            @field(m.plans, k).drafted += @field(plans, k).drafted;
            @field(m.plans, k).accepted += @field(plans, k).accepted;
        }
        m.latency.observe(latency);
        if (ttft) |t| m.ttft.observe(t);
        if (decode) |d| m.decode.observe(d);
        if (prefill) |p| m.prefill.observe(p);
        if (tpot) |t| m.tpot.observe(t);
    }

    /// A request's prompt rendered and tokenized, its pictures prepared, in `took` seconds.
    pub fn notePrepare(m: *Metrics, io: std.Io, took: f64) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.prepare.observe(took);
    }

    /// A finished request's head chains by depth (api.Stats.depths).
    pub fn noteDepths(m: *Metrics, io: std.Io, depths: []const api.DraftDepth) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        for (depths, 0..) |d, j| {
            const t = &m.depths[@min(j, m.depths.len - 1)];
            t.drafted += d.drafted;
            t.reached += d.reached;
            t.kept += d.kept;
        }
    }

    /// Starts the stats line's first window at `now` (nanoseconds, the reply timings' clock).
    pub fn startLines(m: *Metrics, io: std.Io, now: i96) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.line_at = now;
        m.line_base = m.totals();
    }

    /// The stats line once `every` nanoseconds have passed since the last (null before, or with lines off): what the
    /// requests finished since then did, in `buf`.
    pub fn statsLine(m: *Metrics, io: std.Io, now: i96, every: i96, buf: []u8) ?[]const u8 {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        const at = m.line_at orelse return null;
        if (now - at < every) return null;
        const t = m.totals();
        const text = lineText(buf, t.since(m.line_base), now - at) catch null;
        m.line_at = now;
        m.line_base = t;
        return text;
    }

    fn totals(m: *const Metrics) Totals {
        return .{ .replies = m.latency.n, .generation = m.generation, .rounds = m.rounds, .drafted = m.drafted, .accepted = m.accepted, .depths = m.depths };
    }

    pub fn tpotValue(_: *const Metrics, first: ?i96, last: ?i96, generated: usize) ?f64 {
        if (generated < 2 or first == null or last == null) return null;
        return @max(0, @as(f64, @floatFromInt(last.? - first.?)) / 1e9) / @as(f64, @floatFromInt(generated - 1));
    }

    /// One HTTP reply under its key label, counted once per request.
    pub fn httpRequest(m: *Metrics, io: std.Io, key: []const u8, status: u16) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        for (m.requests.items) |*r| if (r.status == status and std.mem.eql(u8, r.key, key)) {
            r.count += 1;
            return;
        };
        const owned = m.gpa.dupe(u8, key) catch return;
        m.requests.append(m.gpa, .{ .key = owned, .status = status, .count = 1 }) catch m.gpa.free(owned);
    }

    pub fn disconnected(m: *Metrics, io: std.Io) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.disconnects += 1;
    }

    /// The scrape body, ending in a newline.
    pub fn render(m: *Metrics, io: std.Io, w: *std.Io.Writer, engine: api.Engine, window: u32) !void {
        var status: api.Status = .{};
        var streams: [512]u32 = undefined;
        engine.status(&status, &streams);
        var snap: Metrics = undefined;
        const requests = blk: {
            m.mutex.lockUncancelable(io);
            defer m.mutex.unlock(io);
            snap = m.*;
            break :blk try m.gpa.dupe(@TypeOf(m.requests.items[0]), m.requests.items);
        };
        defer m.gpa.free(requests);
        std.mem.sort(@TypeOf(requests[0]), requests, {}, struct {
            fn less(_: void, x: @TypeOf(requests[0]), y: @TypeOf(requests[0])) bool {
                const o = std.mem.order(u8, x.key, y.key);
                return o == .lt or (o == .eq and x.status < y.status);
            }
        }.less);
        if (requests.len > 0) {
            try family(w, "requests_total", "counter", "HTTP replies by API key label and status.");
            for (requests) |r| try w.print("{s}requests_total{{key=\"{s}\",status=\"{d}\"}} {d}\n", .{ prefix, r.key, r.status, r.count });
        }
        try gauge(w, "requests_running", "gauge", "Requests in prefill or decode.", status.running);
        try gauge(w, "requests_waiting", "gauge", "Requests queued or held until a lane is free.", status.waiting);
        try gauge(w, "generation_tokens_running", "gauge", "Generated tokens held by live streams.", status.generation_tokens);
        try gauge(w, "prompt_tokens_total", "counter", "Prompt tokens of finished requests.", snap.prompt);
        const keeps = engine.info().prompt_cache;
        if (keeps) try gauge(w, "prompt_tokens_cached_total", "counter", "Prompt tokens of finished requests restored from kept prompt states.", snap.cached);
        try gauge(w, "generation_tokens_total", "counter", "Generated tokens of finished requests.", snap.generation);
        const live = streams[0..@min(status.streams, streams.len)];
        try family(w, "kv_cache_usage_ratio", "gauge", "Tokens in a stream cache divided by that stream's context window.");
        try pools(w, "kv_cache_usage_ratio", "pool", live, window);
        try gauge(w, "mtp_drafted_total", "counter", "Draft tokens verified on finished requests.", snap.drafted);
        try gauge(w, "mtp_accepted_total", "counter", "Draft tokens kept on finished requests.", snap.accepted);
        try depthFamilies(w, &snap.depths);
        try histogram(w, "request_latency_seconds", "Seconds from arrival to the reply leaving.", snap.latency);
        try histogram(w, "time_to_first_token_seconds", "Seconds from arrival to the first generated token.", snap.ttft);
        try histogram(w, "request_decode_seconds", "Seconds a finished request spent decoding. Its sum over generation_tokens_total is the decode rate.", snap.decode);
        try histogram(w, "request_time_per_output_token_seconds", "Seconds from the first generated token to the last, divided by the tokens less one.", snap.tpot);
        try gauge(w, "decode_rounds_total", "counter", "Decode rounds used by finished requests.", snap.rounds);
        try byPlan(w, "decode_rounds_by_plan_total", "Decode rounds of finished requests by the plan their window ran: a copied continuation of the stream's context (copy) or the drafter's drafts (head); forced and plain rounds count in neither.", snap.plans, "rounds");
        try byPlan(w, "mtp_drafted_by_plan_total", "Draft tokens verified on finished requests, by the plan of their round's window (copy, head).", snap.plans, "drafted");
        try byPlan(w, "mtp_accepted_by_plan_total", "Draft tokens kept on finished requests, by the plan of their round's window (copy, head).", snap.plans, "accepted");
        try histogram(w, "request_prefill_seconds", "Seconds a finished request's prompt pass took.", snap.prefill);
        try histogram(w, "request_prepare_seconds", "Seconds a request's prompt took to render and tokenize, its pictures prepared too, before it reached the engine.", snap.prepare);
        try processGauges(w, process.read());
        if (engine.memory(false)) |mem| {
            try gauge(w, "device_memory_bytes", "gauge", "Device memory the engine holds now: weights, caches and live streams; only where the backend counts it (CUDA).", mem.active);
            try gauge(w, "device_memory_peak_bytes", "gauge", "The most device memory the engine has held since start or the last /health?reset_peak=1.", mem.peak);
            try gauge(w, "pinned_memory_bytes", "gauge", "Pinned host memory the engine holds now: staging buffers and windows; only where the backend counts it (CUDA).", mem.pinned);
        }
        try gauge(w, "num_requests_running", "gauge", "Requests in prefill or decode. A mirror of tensorfold:requests_running.", status.running);
        try gauge(w, "num_requests_waiting", "gauge", "Requests queued or held until a lane is free. A mirror of tensorfold:requests_waiting.", status.waiting);
        try family(w, "kv_cache_usage_perc", "gauge", "A stream's cache occupancy under vLLM's name; same streams and ratios as tensorfold:kv_cache_usage_ratio.");
        try pools(w, "kv_cache_usage_perc", "stream", live, window);
        try gauge(w, "spec_decode_num_draft_tokens_total", "counter", "Draft tokens verified on finished requests, this server's single draft counter.", snap.drafted);
        try gauge(w, "spec_decode_num_accepted_tokens_total", "counter", "Draft tokens kept on finished requests.", snap.accepted);
        try perPosition(w, &snap.depths);
        if (keeps) {
            try gauge(w, "prefix_cache_queries_total", "counter", "Prompt tokens of finished requests, looked up in the kept prompt states, under vLLM's name.", snap.prompt);
            try gauge(w, "prefix_cache_hits_total", "counter", "Prompt tokens of finished requests restored from kept prompt states, under vLLM's name.", snap.cached);
        }
        try histogram(w, "e2e_request_latency_seconds", "Seconds from arrival to the reply leaving, under vLLM's name.", snap.latency);
        try histogram(w, "request_decode_time_seconds", "Seconds a finished request spent decoding, under vLLM's name.", snap.decode);
        try histogram(w, "request_prefill_time_seconds", "Seconds a finished request's prompt pass took, under vLLM's name.", snap.prefill);
        try gauge(w, "client_disconnections_total", "counter", "Requests a client left before the reply left the server.", snap.disconnects);
        if (status.preemptions) |p| try gauge(w, "preemptions_total", "counter", "Requests that had to give a lane up to a later one.", p);
        if (status.graph_fallbacks) |n| try gauge(w, "graph_fallbacks_total", "counter", "CUDA graph captures that failed on a rank; every rank ran that work eagerly instead.", n);
        if (status.head_rounds) |hr| {
            try family(w, "decode_rounds_by_head_path_total", "counter", "Decode rounds by the path the output head's logits took to the draws: the device's argmax (every row greedy, none masked), the tokens merged from the nodes' bests (TF_DS_HEAD_KEY), the GPU's candidates, the candidates merged from the nodes' own (TF_DS_HEAD_CAND), candidates a flag sent on to whole rows, or whole rows to the host.");
            for ([_][]const u8{ "argmax", "keys", "candidates", "merged", "flagged", "whole" }, hr) |name, n| try w.print("{s}decode_rounds_by_head_path_total{{path=\"{s}\"}} {d}\n", .{ prefix, name, n });
        }
        if (status.draft_vocab) |v| try draftVocabFamilies(w, v);
        if (status.cache) |c| try cacheFamilies(w, c);
        if (engine.info().spill) |si| if (si.settings.asked) try spillFamilies(w, si, status.spill);
    }
};

/// The disk tier asked for: tensorfold:spill_on (1 while on), tensorfold:spill_state{state,reason} 1 (the cause when
/// off; /metrics stays open when API keys are on, so the dashboards read the cause here), and every setting as
/// tensorfold:spill_setting{name} with the derived room_gib.
pub fn spillFamilies(w: *std.Io.Writer, si: api.SpillInfo, now: ?api.SpillState) !void {
    const s = si.settings;
    const st = s.stateOf(now);
    try gauge(w, "spill_on", "gauge", "1 while the prompt cache's disk tier is on; 0 while it opens or is off (spill_state names the cause).", @intFromBool(st.state == .on));
    try family(w, "spill_state", "gauge", "The disk tier's state (opening, on, off) and, when off, its cause; always 1.");
    try w.print("{s}spill_state{{state=\"{s}\",reason=\"", .{ prefix, @tagName(st.state) });
    try labelText(w, st.reason());
    try w.writeAll("\"} 1\n");
    try family(w, "spill_setting", "gauge", "The disk tier's settings as the engine took them (TF_DS_SPILL_*, TF_SPILL_*), and room_gib = (1 + keep_builds) x gib + min_free_gib.");
    try w.print("{s}spill_setting{{name=\"gib\"}} {d}\n", .{ prefix, s.gib });
    inline for (api.Spill.ints) |n| try w.print("{s}spill_setting{{name=\"{s}\"}} {d}\n", .{ prefix, n.field, @field(s, n.field) });
    inline for (api.Spill.floats) |n| try w.print("{s}spill_setting{{name=\"{s}\"}} {d}\n", .{ prefix, n.field, @field(s, n.field) });
    try w.print("{s}spill_setting{{name=\"room_gib\"}} {d}\n", .{ prefix, s.roomGib() });
    if (si.staging_min_bytes) |b| try w.print("{s}spill_setting{{name=\"staging_min_bytes\"}} {d}\n", .{ prefix, b });
}

/// A Prometheus label value: backslash, double quote and newline escaped.
fn labelText(w: *std.Io.Writer, t: []const u8) !void {
    for (t) |c| switch (c) {
        '\\' => try w.writeAll("\\\\"),
        '"' => try w.writeAll("\\\""),
        '\n' => try w.writeAll("\\n"),
        else => try w.writeByte(c),
    };
}

/// The prompt cache's counters (its tier or grouping on): the GLM engine's names for continuation, then
/// tensorfold:prompt_cache_* and the tier's tensorfold:spill_*{rank="N"}.
pub fn cacheFamilies(w: *std.Io.Writer, c: api.CacheStats) !void {
    const H = api.CacheStats.help;
    try gauge(w, "chats_continued", "counter", H.chats_continued, c.chats_continued);
    try gauge(w, "chats_continued_prefill_tokens", "counter", H.chats_continued_prefill_tokens, c.chats_continued_prefill_tokens);
    const info = @typeInfo(api.CacheStats).@"struct";
    inline for (info.field_names, info.field_types) |name, T| if (T == u64) {
        try gauge(w, "prompt_cache_" ++ name, comptime kindOf(&api.CacheStats.gauges, name), @field(H, name), @field(c, name));
    };
    if (c.spill) |rc| { // every rank's, labelled (the disk tiers are per rank)
        const T = api.prompt_cache.TierCounts;
        inline for (@typeInfo(T).@"struct".field_names) |name| {
            try family(w, "spill_" ++ name, comptime kindOf(&T.gauges, name), @field(T.help, name));
            for (rc.slice(), 0..) |t, r| try w.print("{s}spill_{s}{{rank=\"{d}\"}} {d}\n", .{ prefix, name, r, @field(t, name) });
        }
    }
    try timeFamilies(w, c.times);
}

/// The tier's call timings: tensorfold:spill_time_seconds{call} (a histogram on the log2 bounds of
/// prompt_cache.Timing), and the p50, p99 and max of each call as gauges (estimated within one bucket).
fn timeFamilies(w: *std.Io.Writer, t: api.prompt_cache.TierTimes) !void {
    const H = api.prompt_cache.TierTimes.help;
    const fields = @typeInfo(api.prompt_cache.TierTimes).@"struct".field_names;
    try w.print("# HELP {s}spill_time_seconds Seconds the disk tier's calls took, by call:", .{prefix});
    inline for (fields) |name| try w.print(" {s}: {s}", .{ name, @field(H, name) });
    try w.print("\n# TYPE {s}spill_time_seconds histogram\n", .{prefix});
    var buf: [40]u8 = undefined;
    inline for (fields) |name| {
        const x = @field(t, name);
        var cumulative: u64 = 0;
        for (x.buckets[0..api.Timing.edges], 0..) |b, i| {
            cumulative += b;
            try w.print("{s}spill_time_seconds_bucket{{call=\"{s}\",le=\"{s}\"}} {d}\n", .{ prefix, name, seconds(&buf, api.Timing.bound(i)), cumulative });
        }
        try w.print("{s}spill_time_seconds_bucket{{call=\"{s}\",le=\"+Inf\"}} {d}\n", .{ prefix, name, x.count });
        try w.print("{s}spill_time_seconds_sum{{call=\"{s}\"}} {s}\n", .{ prefix, name, seconds(&buf, x.sum_ns) });
        try w.print("{s}spill_time_seconds_count{{call=\"{s}\"}} {d}\n", .{ prefix, name, x.count });
    }
    inline for (.{ .{ "p50", 0.5 }, .{ "p99", 0.99 } }) |q| {
        try family(w, "spill_time_" ++ q[0] ++ "_seconds", "gauge", "The disk tier's calls: the " ++ q[0] ++ " of spill_time_seconds, estimated as its bucket's bound (at most the max).");
        inline for (fields) |name| try w.print("{s}spill_time_{s}_seconds{{call=\"{s}\"}} {s}\n", .{ prefix, q[0], name, seconds(&buf, @field(t, name).quantile(q[1])) });
    }
    try family(w, "spill_time_max_seconds", "gauge", "The disk tier's calls: the longest of spill_time_seconds.");
    inline for (fields) |name| try w.print("{s}spill_time_max_seconds{{call=\"{s}\"}} {s}\n", .{ prefix, name, seconds(&buf, @field(t, name).max_ns) });
}

/// Nanoseconds as seconds, to the nanosecond, without trailing zeros.
fn seconds(buf: []u8, ns: u64) []const u8 {
    if (ns == 0) return "0";
    return trimmed(buf, @as(f64, @floatFromInt(ns)) / 1e9, 9);
}

fn kindOf(comptime gauges: []const []const u8, comptime name: []const u8) []const u8 {
    @setEvalBranchQuota(100_000);
    for (gauges) |g| if (std.mem.eql(u8, g, name)) return "gauge";
    return "counter";
}

fn family(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} {s}\n", .{ prefix, name, help, prefix, name, kind });
}

fn gauge(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8, value: u64) !void {
    try family(w, name, kind, help);
    try w.print("{s}{s} {d}\n", .{ prefix, name, value });
}

/// The drafter on a pruned vocabulary (api.DraftVocab): its passes by head, its guard's committed tokens and those
/// outside the list, and whether the passes take the whole head.
fn draftVocabFamilies(w: *std.Io.Writer, v: api.DraftVocab) !void {
    try family(w, "draft_vocab_passes_total", "counter", "Drafter passes on the pruned list (head=\"list\") and, while its guard asked for it, on the whole head (head=\"full\").");
    try w.print("{s}draft_vocab_passes_total{{head=\"list\"}} {d}\n{s}draft_vocab_passes_total{{head=\"full\"}} {d}\n", .{ prefix, v.list_passes, prefix, v.full_passes });
    try gauge(w, "draft_vocab_tokens_total", "counter", "Committed tokens the pruned list's guard counted.", v.tokens);
    try gauge(w, "draft_vocab_misses_total", "counter", "Of the counted committed tokens, those outside the pruned list.", v.misses);
    try gauge(w, "draft_vocab_full", "gauge", "1 while the guard has the drafter's passes take the whole head (over 2% of the last 4,096 committed tokens outside the list).", @intFromBool(v.full));
}

/// A counter family of the rounds by plan (api.Plans): one line a plan, its `count` (rounds, drafted or accepted).
fn byPlan(w: *std.Io.Writer, name: []const u8, help: []const u8, plans: api.Plans, comptime count: []const u8) !void {
    try family(w, name, "counter", help);
    inline for (.{ "copy", "head" }) |k| try w.print("{s}{s}{{plan=\"{s}\"}} {d}\n", .{ prefix, name, k, @field(@field(plans, k), count) });
}

/// Depths 1 to the deepest any counted chain drafted (none: no chain counted).
fn counted(depths: []const Depth) []const Depth {
    var n: usize = 0;
    for (depths, 0..) |d, j| {
        if (d.drafted > 0) n = j + 1;
    }
    return depths[0..n];
}

/// The draft head's chains by depth, once one is counted: an engine that does not count them shows none.
fn depthFamilies(w: *std.Io.Writer, depths: []const Depth) !void {
    const ds = counted(depths);
    if (ds.len == 0) return;
    const sets = .{
        .{ "mtp_depth_drafted_total", "Draft head chains of finished requests that drafted this deep (depth 1: the first draft).", "drafted" },
        .{ "mtp_depth_reached_total", "Of those, the chains that kept every draft above this depth.", "reached" },
        .{ "mtp_depth_accepted_total", "Of those, the chains that kept this depth's draft too. Over mtp_depth_drafted_total: the accept rate at this depth; over mtp_depth_reached_total: the rate of a draft whose drafts above were kept.", "kept" },
    };
    inline for (sets) |set| {
        try family(w, set[0], "counter", set[1]);
        for (ds, 1..) |d, n| try w.print("{s}{s}{{depth=\"{d}\"}} {d}\n", .{ prefix, set[0], n, @field(d, set[2]) });
    }
}

/// The same chains' kept drafts under vLLM's per-position name (position 0: the first draft).
fn perPosition(w: *std.Io.Writer, depths: []const Depth) !void {
    const ds = counted(depths);
    if (ds.len == 0) return;
    try family(w, "spec_decode_num_accepted_tokens_per_pos_total", "counter", "Draft head tokens kept on finished requests by position in the chain, under vLLM's name; the counts of tensorfold:mtp_depth_accepted_total.");
    for (ds, 0..) |d, i| try w.print("{s}spec_decode_num_accepted_tokens_per_pos_total{{position=\"{d}\"}} {d}\n", .{ prefix, i, d.kept });
}

/// The stats line's text: the requests finished in a window of `ns`, in the terms of DeepInfer's stats line (the
/// accept rate overall and by draft position, drafts verified and kept; deepinfer/engine/stats_logger.py:297-309).
fn lineText(buf: []u8, t: Totals, ns: i96) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    try w.print("stats over {d:.1}s: {d} {s}, {d} tokens in {d} rounds ({d:.2} a round)", .{ @as(f64, @floatFromInt(ns)) / 1e9, t.replies, if (t.replies == 1) "reply" else "replies", t.generation, t.rounds, share(t.generation, t.rounds) });
    if (t.drafted > 0) try w.print(" | drafts kept {d:.2}% ({d} of {d})", .{ 100 * share(t.accepted, t.drafted), t.accepted, t.drafted });
    const ds = counted(&t.depths);
    if (ds.len > 0) {
        try w.writeAll(" | head chains by depth: kept [");
        for (ds, 0..) |d, j| try w.print("{s}{d:.2}%", .{ sep(j), 100 * share(d.kept, d.drafted) });
        try w.writeAll("] of drafted [");
        for (ds, 0..) |d, j| try w.print("{s}{d}", .{ sep(j), d.drafted });
        try w.writeAll("], kept when reached [");
        for (ds, 0..) |d, j| try w.print("{s}{d:.2}%", .{ sep(j), 100 * share(d.kept, d.reached) });
        try w.writeAll("]");
    }
    return w.buffered();
}

fn share(part: u64, whole: u64) f64 {
    return if (whole == 0) 0 else @as(f64, @floatFromInt(part)) / @as(f64, @floatFromInt(whole));
}

fn sep(j: usize) []const u8 {
    return if (j > 0) ", " else "";
}

fn processGauges(w: *std.Io.Writer, sample: ?process.Snapshot) !void {
    const p = sample orelse return;
    try gauge(w, "process_footprint_bytes", "gauge", "Current OS physical footprint including Metal buffers (macOS).", p.physical_footprint_bytes);
    try gauge(w, "process_footprint_peak_bytes", "gauge", "Kernel lifetime peak physical footprint including Metal buffers; never reset by health (macOS).", p.lifetime_peak_physical_footprint_bytes);
}

test "process gauges omit unknown samples and keep current and lifetime peak distinct" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try processGauges(&out.writer, null);
    try std.testing.expectEqual(@as(usize, 0), out.written().len);
    try processGauges(&out.writer, .{ .physical_footprint_bytes = 0, .lifetime_peak_physical_footprint_bytes = 12 });
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "tensorfold:process_footprint_bytes 0\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "tensorfold:process_footprint_peak_bytes 12\n") != null);
}

fn pools(w: *std.Io.Writer, name: []const u8, label: []const u8, lengths: []const u32, window: u32) !void {
    if (lengths.len == 0) return w.print("{s}{s}{{{s}=\"0\"}} 0\n", .{ prefix, name, label });
    for (lengths, 0..) |n, i| {
        const ratio: f64 = if (window == 0) 0 else @min(1.0, @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(window)));
        var buf: [40]u8 = undefined;
        try w.print("{s}{s}{{{s}=\"{d}\"}} {s}\n", .{ prefix, name, label, i, num(&buf, ratio) });
    }
}

fn histogram(w: *std.Io.Writer, name: []const u8, help: []const u8, h: anytype) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} histogram\n", .{ prefix, name, help, prefix, name });
    var cumulative: u64 = 0;
    const edges = @TypeOf(h).bucket_edges;
    for (edges, 0..) |edge, i| {
        cumulative += h.counts[i];
        var buf: [40]u8 = undefined;
        try w.print("{s}{s}_bucket{{le=\"{s}\"}} {d}\n", .{ prefix, name, trimmed(&buf, edge, 4), cumulative });
    }
    try w.print("{s}{s}_bucket{{le=\"+Inf\"}} {d}\n", .{ prefix, name, cumulative + h.counts[edges.len] });
    var buf: [40]u8 = undefined;
    try w.print("{s}{s}_sum {s}\n{s}{s}_count {d}\n", .{ prefix, name, num(&buf, h.total), prefix, name, h.n });
}

/// ``_num``: an integral value as an int, else six decimals without trailing zeros.
fn num(buf: []u8, value: f64) []const u8 {
    if (value == @trunc(value) and @abs(value) < 1e18) return std.fmt.bufPrint(buf, "{d}", .{@as(i64, @intFromFloat(value))}) catch "0";
    return trimmed(buf, value, 6);
}

fn trimmed(buf: []u8, value: f64, comptime places: u8) []const u8 {
    const text = std.fmt.bufPrint(buf, "{d:." ++ std.fmt.comptimePrint("{d}", .{places}) ++ "}", .{value}) catch return "0";
    return std.mem.trimEnd(u8, std.mem.trimEnd(u8, text, "0"), ".");
}

test "edges and numbers" {
    var buf: [40]u8 = undefined;
    try std.testing.expectEqualStrings("0.01", trimmed(&buf, 0.01, 4));
    try std.testing.expectEqualStrings("120", trimmed(&buf, 120.0, 4));
    try std.testing.expectEqualStrings("0.333333", num(&buf, 1.0 / 3.0));
    try std.testing.expectEqualStrings("2", num(&buf, 2.0));
}

test "metrics record rounds prefill and per-token timing" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.note(std.testing.io, 10, 0, 3, 0, 0, 2, .{}, 1.0, null, null, 0.25, m.tpotValue(1_000_000_000, 1_024_000_000, 3));
    try std.testing.expectEqual(@as(u64, 2), m.rounds);
    try std.testing.expectEqual(@as(u64, 1), m.prefill.n);
    try std.testing.expectEqual(@as(u64, 1), m.tpot.n);
    try std.testing.expectEqual(@as(u64, 1), m.prefill.counts[3]);
    try std.testing.expectEqual(@as(u64, 1), m.tpot.counts[4]);
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try histogram(&out.writer, "request_prefill_seconds", "prefill", m.prefill);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "request_prefill_seconds_sum 0.25") != null);
}

test "a request's front: the seconds its prompt took to render and tokenize, on the per-token scale" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.notePrepare(std.testing.io, 0.012);
    m.notePrepare(std.testing.io, 0.9);
    var stub: RenderStub = .{};
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:request_prepare_seconds_count 2\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:request_prepare_seconds_bucket{le=\"0.015\"} 1\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:request_prepare_seconds_sum 0.912") != null);
}

const RenderStub = struct {
    pub fn info(ctx: *anyopaque) api.Info {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        return s.info_value;
    }
    pub fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        unreachable;
    }
    pub fn cancel(_: *anyopaque, _: api.Id) void {}
    pub fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        out.* = s.status_value;
    }
    pub fn memory(ctx: *anyopaque, _: bool) ?api.Memory {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        return s.memory_value;
    }
    info_value: api.Info = .{},
    status_value: api.Status = .{ .running = 1, .waiting = 2, .generation_tokens = 3 },
    memory_value: ?api.Memory = null,

    pub fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
};

test "render exposes live counters, rounds, prefill and TPOT without cache family" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.note(std.testing.io, 10, 0, 3, 0, 0, 2, .{}, 1.0, null, null, 0.25, m.tpotValue(1_000_000_000, 3_000_000_000, 3));
    var stub: RenderStub = .{};
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:generation_tokens_running 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:decode_rounds_total 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:request_prefill_seconds_sum 0.25") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:request_time_per_output_token_seconds_sum 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "prompt_tokens_cached_total") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "prefix_cache_hits_total") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "device_memory_bytes") == null); // unknown memory: no gauge
    try std.testing.expect(std.mem.indexOf(u8, body, "pinned_memory_bytes") == null);
    stub.memory_value = .{ .active = 7 << 30, .peak = 8 << 30, .pinned = 1 << 30 };
    var known: std.Io.Writer.Allocating = .init(gpa);
    defer known.deinit();
    try m.render(std.testing.io, &known.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, known.written(), "tensorfold:device_memory_bytes 7516192768\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, known.written(), "tensorfold:device_memory_peak_bytes 8589934592\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, known.written(), "tensorfold:pinned_memory_bytes 1073741824\n") != null);
}

test "render counts restored prompt tokens where the engine keeps prompt states" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.note(std.testing.io, 100, 0, 3, 0, 0, 2, .{}, 1.0, null, null, null, null);
    m.note(std.testing.io, 120, 96, 3, 0, 0, 2, .{}, 1.0, null, null, null, null);
    var stub: RenderStub = .{ .info_value = .{ .prompt_cache = true } };
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:prompt_tokens_total 220\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:prompt_tokens_cached_total 96\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:prefix_cache_queries_total 220\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "tensorfold:prefix_cache_hits_total 96\n") != null);
}

test "render shows the head's chains by depth once one is counted, and their kept drafts under vLLM's per-position name" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    var stub: RenderStub = .{};
    var none: std.Io.Writer.Allocating = .init(gpa);
    defer none.deinit();
    try m.render(std.testing.io, &none.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, none.written(), "mtp_depth_") == null); // nothing counted: no family
    try std.testing.expect(std.mem.indexOf(u8, none.written(), "per_pos") == null);
    var depths: [api.draft_depths]api.DraftDepth = @splat(.{});
    depths[0] = .{ .drafted = 10, .reached = 10, .kept = 8 };
    depths[1] = .{ .drafted = 9, .reached = 7, .kept = 5 };
    depths[2] = .{ .drafted = 4, .reached = 2, .kept = 1 };
    m.noteDepths(std.testing.io, &depths);
    m.noteDepths(std.testing.io, &depths);
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    for ([_][]const u8{
        "tensorfold:mtp_depth_drafted_total{depth=\"1\"} 20\n",
        "tensorfold:mtp_depth_drafted_total{depth=\"3\"} 8\n",
        "tensorfold:mtp_depth_reached_total{depth=\"2\"} 14\n",
        "tensorfold:mtp_depth_accepted_total{depth=\"1\"} 16\n",
        "tensorfold:mtp_depth_accepted_total{depth=\"3\"} 2\n",
        "# TYPE tensorfold:mtp_depth_accepted_total counter\n",
        "tensorfold:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"} 16\n",
        "tensorfold:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"} 2\n",
    }) |line| if (std.mem.indexOf(u8, body, line) == null) {
        std.debug.print("missing: {s}", .{line});
        return error.TestUnexpectedResult;
    };
    // depths past the deepest drafted stay out
    try std.testing.expect(std.mem.indexOf(u8, body, "depth=\"4\"") == null and std.mem.indexOf(u8, body, "position=\"3\"") == null);
}

test "the stats line covers the requests finished since the last, once its interval has passed" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var m: Metrics = .{ .gpa = gpa };
    var buf: [1024]u8 = undefined;
    const s_ns: i96 = std.time.ns_per_s;
    try std.testing.expectEqual(@as(?[]const u8, null), m.statsLine(io, 99 * s_ns, 10 * s_ns, &buf)); // lines off
    m.note(io, 10, 0, 30, 20, 9, 11, .{}, 1.0, null, null, null, null); // before the lines start: in no window
    m.startLines(io, 0);
    m.note(io, 10, 0, 40, 30, 15, 16, .{}, 1.0, null, null, null, null);
    var depths: [api.draft_depths]api.DraftDepth = @splat(.{});
    depths[0] = .{ .drafted = 16, .reached = 16, .kept = 12 };
    depths[1] = .{ .drafted = 14, .reached = 12, .kept = 3 };
    m.noteDepths(io, &depths);
    try std.testing.expectEqual(@as(?[]const u8, null), m.statsLine(io, 5 * s_ns, 10 * s_ns, &buf));
    const first = m.statsLine(io, 12 * s_ns, 10 * s_ns, &buf) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("stats over 12.0s: 1 reply, 40 tokens in 16 rounds (2.50 a round) | drafts kept 50.00% (15 of 30) | head chains by depth: kept [75.00%, 21.43%] of drafted [16, 14], kept when reached [75.00%, 25.00%]", first);
    try std.testing.expectEqual(@as(?[]const u8, null), m.statsLine(io, 13 * s_ns, 10 * s_ns, &buf));
    m.note(io, 10, 0, 7, 0, 0, 7, .{}, 1.0, null, null, null, null); // no drafts: the line says none
    const second = m.statsLine(io, 22 * s_ns, 10 * s_ns, &buf) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("stats over 10.0s: 1 reply, 7 tokens in 7 rounds (1.00 a round)", second);
}

test "metrics snapshot allocation failure releases the mutex" {
    const io = std.testing.io;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var m: Metrics = .{ .gpa = gpa };
    defer m.deinit();
    m.httpRequest(io, "client", 200);
    try std.testing.expectEqual(@as(usize, 1), m.requests.items.len);
    var stub: RenderStub = .{};
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    failing.fail_index = failing.alloc_index;
    try std.testing.expectError(error.OutOfMemory, m.render(io, &out.writer, stub.engine(), 4096));
    const available = m.mutex.tryLock();
    // No other thread uses m: either tryLock acquired it, or render retained it.
    // Release before asserting so the unfixed regression terminates without blocking.
    m.mutex.unlock(io);
    try std.testing.expect(available);
    try std.testing.expect(failing.has_induced_failure);
    failing.fail_index = std.math.maxInt(usize);
    m.httpRequest(io, "client", 200);
    m.note(io, 10, 0, 3, 0, 0, 2, .{}, 1.0, null, null, 0.25, null);
    try m.render(io, &out.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "requests_total{key=\"client\",status=\"200\"} 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "prompt_tokens_total 10") != null);
}

test "request counter append failure frees its label" {
    var bytes: [1024]u8 = undefined;
    var buffer = std.heap.FixedBufferAllocator.init(&bytes);
    var failing = std.testing.FailingAllocator.init(buffer.allocator(), .{ .fail_index = 1 });
    const gpa = failing.allocator();
    var m: Metrics = .{ .gpa = gpa };
    defer m.deinit();
    m.httpRequest(std.testing.io, "client", 200);
    try std.testing.expect(failing.has_induced_failure);
    try std.testing.expectEqual(@as(usize, 6), failing.allocated_bytes);
    try std.testing.expectEqual(@as(usize, 0), m.requests.items.len);
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);

    failing.fail_index = std.math.maxInt(usize);
    m.httpRequest(std.testing.io, "client", 200);
    try std.testing.expectEqual(@as(usize, 1), m.requests.items.len);
    try std.testing.expectEqualStrings("client", m.requests.items[0].key);
    const allocations = failing.allocations;
    m.httpRequest(std.testing.io, "client", 200);
    try std.testing.expectEqual(@as(u64, 2), m.requests.items[0].count);
    try std.testing.expectEqual(allocations, failing.allocations);
}

test "render counts failed graph captures only for an engine that counts them" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    var stub: RenderStub = .{};
    var none: std.Io.Writer.Allocating = .init(gpa);
    defer none.deinit();
    try m.render(std.testing.io, &none.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, none.written(), "graph_fallbacks") == null);
    stub.status_value.graph_fallbacks = 3;
    var some: std.Io.Writer.Allocating = .init(gpa);
    defer some.deinit();
    try m.render(std.testing.io, &some.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, some.written(), "# TYPE tensorfold:graph_fallbacks_total counter\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, some.written(), "tensorfold:graph_fallbacks_total 3\n") != null);
}

test "render counts decode rounds by the head's path only for an engine that counts them" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    var stub: RenderStub = .{};
    var none: std.Io.Writer.Allocating = .init(gpa);
    defer none.deinit();
    try m.render(std.testing.io, &none.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, none.written(), "decode_rounds_by_head_path") == null);
    stub.status_value.head_rounds = .{ 11, 5, 22, 7, 3, 4 };
    var some: std.Io.Writer.Allocating = .init(gpa);
    defer some.deinit();
    try m.render(std.testing.io, &some.writer, stub.engine(), 4096);
    const want =
        "# TYPE tensorfold:decode_rounds_by_head_path_total counter\n" ++
        "tensorfold:decode_rounds_by_head_path_total{path=\"argmax\"} 11\n" ++
        "tensorfold:decode_rounds_by_head_path_total{path=\"keys\"} 5\n" ++
        "tensorfold:decode_rounds_by_head_path_total{path=\"candidates\"} 22\n" ++
        "tensorfold:decode_rounds_by_head_path_total{path=\"merged\"} 7\n" ++
        "tensorfold:decode_rounds_by_head_path_total{path=\"flagged\"} 3\n" ++
        "tensorfold:decode_rounds_by_head_path_total{path=\"whole\"} 4\n";
    try std.testing.expect(std.mem.indexOf(u8, some.written(), want) != null);
}

test "render counts the drafter's pruned-list passes, its guard's tokens and misses only for an engine that runs the list" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    var stub: RenderStub = .{};
    var none: std.Io.Writer.Allocating = .init(gpa);
    defer none.deinit();
    try m.render(std.testing.io, &none.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, none.written(), "draft_vocab") == null);
    stub.status_value.draft_vocab = .{ .list_passes = 120, .full_passes = 7, .tokens = 9000, .misses = 41, .full = true };
    var some: std.Io.Writer.Allocating = .init(gpa);
    defer some.deinit();
    try m.render(std.testing.io, &some.writer, stub.engine(), 4096);
    const out = some.written();
    for ([_][]const u8{
        "# TYPE tensorfold:draft_vocab_passes_total counter\n",
        "tensorfold:draft_vocab_passes_total{head=\"list\"} 120\n",
        "tensorfold:draft_vocab_passes_total{head=\"full\"} 7\n",
        "# TYPE tensorfold:draft_vocab_tokens_total counter\n",
        "tensorfold:draft_vocab_tokens_total 9000\n",
        "# TYPE tensorfold:draft_vocab_misses_total counter\n",
        "tensorfold:draft_vocab_misses_total 41\n",
        "# TYPE tensorfold:draft_vocab_full gauge\n",
        "tensorfold:draft_vocab_full 1\n",
    }) |want| if (std.mem.indexOf(u8, out, want) == null) {
        std.debug.print("missing: {s}", .{want});
        return error.TestUnexpectedResult;
    };
    // every family has its help
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, out, "# HELP tensorfold:draft_vocab_"));
}

test "cache families: present only with the cache's counters, every field with its kind and help" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    var stub: RenderStub = .{};
    var plain: std.Io.Writer.Allocating = .init(gpa);
    defer plain.deinit();
    try m.render(std.testing.io, &plain.writer, stub.engine(), 4096);
    for ([_][]const u8{ "chats_continued", "prompt_cache_", "spill_" }) |name| try std.testing.expect(std.mem.indexOf(u8, plain.written(), name) == null);
    var ranks: api.prompt_cache.RankCounts = .{ .n = 2 };
    ranks.ranks[0] = .{ .writes_committed = 9, .staging_peak = 7 };
    ranks.ranks[1] = .{ .writes_committed = 8, .crc_errors = 1 };
    stub.status_value.cache = .{ .chats_continued = 1, .restores_active = 2, .service_refused = 4, .spill = ranks };
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    for ([_][]const u8{
        "tensorfold:chats_continued 1\n",
        "# TYPE tensorfold:prompt_cache_restores_active gauge\n",
        "tensorfold:prompt_cache_restores_active 2\n",
        "# TYPE tensorfold:prompt_cache_disk_hits_committed counter\n",
        "# TYPE tensorfold:prompt_cache_service_refused counter\n",
        "tensorfold:prompt_cache_service_refused 4\n",
        "tensorfold:spill_writes_committed{rank=\"0\"} 9\n",
        "tensorfold:spill_writes_committed{rank=\"1\"} 8\n",
        "tensorfold:spill_crc_errors{rank=\"1\"} 1\n",
        "# TYPE tensorfold:spill_staging_peak gauge\n",
    }) |line| try std.testing.expect(std.mem.indexOf(u8, body, line) != null);
}

test "spill families: the tier's state with its cause and its settings, only when the tier was asked for" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    var stub: RenderStub = .{};
    var plain: std.Io.Writer.Allocating = .init(gpa);
    defer plain.deinit();
    try m.render(std.testing.io, &plain.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, plain.written(), "spill_") == null);
    stub.info_value.spill = .{ .settings = .{ .asked = true, .off = "TF_DS_SPILL_GIB=\"x\": GiB" } };
    var off: std.Io.Writer.Allocating = .init(gpa);
    defer off.deinit();
    try m.render(std.testing.io, &off.writer, stub.engine(), 4096);
    for ([_][]const u8{
        "# TYPE tensorfold:spill_on gauge\n",
        "tensorfold:spill_on 0\n",
        "tensorfold:spill_state{state=\"off\",reason=\"TF_DS_SPILL_GIB=\\\"x\\\": GiB\"} 1\n",
    }) |line| std.testing.expect(std.mem.indexOf(u8, off.written(), line) != null) catch |e| {
        std.debug.print("missing {s} in:\n{s}\n", .{ line, off.written() });
        return e;
    };
    stub.info_value.spill = .{ .settings = .{ .asked = true, .dir = "/spill", .gib = 256 } };
    stub.status_value.spill = .{ .state = .on };
    var on: std.Io.Writer.Allocating = .init(gpa);
    defer on.deinit();
    try m.render(std.testing.io, &on.writer, stub.engine(), 4096);
    for ([_][]const u8{
        "tensorfold:spill_on 1\n",
        "tensorfold:spill_state{state=\"on\",reason=\"\"} 1\n",
        "tensorfold:spill_setting{name=\"gib\"} 256\n",
        "tensorfold:spill_setting{name=\"turns\"} 4\n",
        "tensorfold:spill_setting{name=\"room_gib\"} 562\n",
    }) |line| std.testing.expect(std.mem.indexOf(u8, on.written(), line) != null) catch |e| {
        std.debug.print("missing {s} in:\n{s}\n", .{ line, on.written() });
        return e;
    };
}

test "the one store's tier counters render under rank 0, v3's landing, probe and direct-I/O counters with them" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    var stub: RenderStub = .{};
    var ranks: api.prompt_cache.RankCounts = .{ .n = 1 };
    ranks.ranks[0] = .{ .direct_refused = 2, .landings_failed = 1, .probe_failed = 1, .landing_bytes = 64 };
    stub.status_value.cache = .{ .spill = ranks };
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    for ([_][]const u8{
        "tensorfold:spill_direct_refused{rank=\"0\"} 2\n",
        "tensorfold:spill_landings_failed{rank=\"0\"} 1\n",
        "tensorfold:spill_probe_failed{rank=\"0\"} 1\n",
        "tensorfold:spill_landing_bytes{rank=\"0\"} 64\n",
        "# TYPE tensorfold:spill_landings counter\n",
    }) |line| std.testing.expect(std.mem.indexOf(u8, out.written(), line) != null) catch |e| {
        std.debug.print("missing {s}\n", .{line});
        return e;
    };
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "rank=\"1\"") == null);
}

test "the tier's call timings render as one histogram labelled by call, with p50, p99 and max gauges" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    var stub: RenderStub = .{};
    var c: api.CacheStats = .{};
    c.times.lookup.observe(5000);
    c.times.land_piece.observe(3 << 20);
    stub.status_value.cache = c;
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    for ([_][]const u8{
        "# TYPE tensorfold:spill_time_seconds histogram\n",
        "tensorfold:spill_time_seconds_bucket{call=\"lookup\",le=\"0.000004096\"} 0\n",
        "tensorfold:spill_time_seconds_bucket{call=\"lookup\",le=\"0.000008192\"} 1\n",
        "tensorfold:spill_time_seconds_bucket{call=\"lookup\",le=\"+Inf\"} 1\n",
        "tensorfold:spill_time_seconds_sum{call=\"lookup\"} 0.000005\n",
        "tensorfold:spill_time_seconds_count{call=\"lookup\"} 1\n",
        "tensorfold:spill_time_seconds_count{call=\"probe\"} 0\n",
        "tensorfold:spill_time_p50_seconds{call=\"land_piece\"} 0.003145728\n",
        "tensorfold:spill_time_p99_seconds{call=\"lookup\"} 0.000005\n",
        "tensorfold:spill_time_max_seconds{call=\"lookup\"} 0.000005\n",
    }) |line| std.testing.expect(std.mem.indexOf(u8, out.written(), line) != null) catch |e| {
        std.debug.print("missing {s}\n", .{line});
        return e;
    };
}

test "render counts finished requests' rounds by plan: copied continuations and the head's drafts, verified and kept" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    m.note(std.testing.io, 10, 0, 30, 21, 17, 9, .{ .copy = .{ .rounds = 3, .drafted = 15, .accepted = 13 }, .head = .{ .rounds = 2, .drafted = 6, .accepted = 4 } }, 1.0, null, null, null, null);
    m.note(std.testing.io, 10, 0, 10, 5, 2, 6, .{ .head = .{ .rounds = 1, .drafted = 5, .accepted = 2 } }, 1.0, null, null, null, null);
    var stub: RenderStub = .{};
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.render(std.testing.io, &out.writer, stub.engine(), 4096);
    const body = out.written();
    for ([_][]const u8{
        "# TYPE tensorfold:decode_rounds_by_plan_total counter\n",
        "tensorfold:decode_rounds_by_plan_total{plan=\"copy\"} 3\n",
        "tensorfold:decode_rounds_by_plan_total{plan=\"head\"} 3\n",
        "# TYPE tensorfold:mtp_drafted_by_plan_total counter\n",
        "tensorfold:mtp_drafted_by_plan_total{plan=\"copy\"} 15\n",
        "tensorfold:mtp_drafted_by_plan_total{plan=\"head\"} 11\n",
        "# TYPE tensorfold:mtp_accepted_by_plan_total counter\n",
        "tensorfold:mtp_accepted_by_plan_total{plan=\"copy\"} 13\n",
        "tensorfold:mtp_accepted_by_plan_total{plan=\"head\"} 6\n",
        "tensorfold:mtp_drafted_total 26\n", // the plans split it (other plans none here)
        "tensorfold:decode_rounds_total 15\n",
    }) |line| {
        if (std.mem.indexOf(u8, body, line) == null) std.debug.print("missing: {s}", .{line});
        try std.testing.expect(std.mem.indexOf(u8, body, line) != null);
    }
    // nothing noted: each plan's line at 0
    var empty: Metrics = .{ .gpa = gpa };
    var none: std.Io.Writer.Allocating = .init(gpa);
    defer none.deinit();
    try empty.render(std.testing.io, &none.writer, stub.engine(), 4096);
    try std.testing.expect(std.mem.indexOf(u8, none.written(), "tensorfold:mtp_accepted_by_plan_total{plan=\"head\"} 0\n") != null);
}
