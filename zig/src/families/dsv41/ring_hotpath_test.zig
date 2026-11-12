//! What the decode rings' kernels do per exchange, as source contracts (the scan runs from the engine's root, as no_nccl_test does):
//! nothing on an exchange's critical path that the exchange does not need. Each contract names what was removed and why it was waste;
//! the device behaviour itself is checked by the ring's replay on the Sparks (kernels_replay_main, rdma_bench_main), the bytes by
//! the replies. A contract test cannot prove a kernel fast; it keeps a removed wait from coming back.
const std = @import("std");

const kernel_path = "zig/kernels/cuda/dsv41/rdma_gather.cu";

fn load(gpa: std.mem.Allocator) !?[]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, kernel_path, gpa, .limited(1 << 22)) catch null;
}

/// The text from `header` to the first closing brace in column 0 (a kernel's or helper's whole body).
fn body(src: []const u8, header: []const u8) ?[]const u8 {
    const at = std.mem.indexOf(u8, src, header) orelse return null;
    const end = std.mem.indexOfPos(u8, src, at, "\n}\n") orelse return null;
    return src[at..end];
}

fn has(b: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, b, needle) != null;
}

test "control: body() finds a function and stops at its closing brace" {
    const src = "int f(int a) {\n  return a;\n}\nint g() {\n  return 2;\n}\n";
    try std.testing.expect(has(body(src, "int f(").?, "return a"));
    try std.testing.expect(!has(body(src, "int f(").?, "return 2"));
    try std.testing.expect(body(src, "int h(") == null);
}

test "the generation-word wait ends in a block vote, not a host-memory load of the abort word by every thread" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest; // (not run from the engine's root)
    defer gpa.free(src);
    const w = body(src, "int roci_waited(") orelse return error.TestUnexpectedResult;
    // a failed lane is every thread's through the barrier's vote: __syncthreads_or, one barrier, no sys-scope load after it
    try std.testing.expect(has(w, "__syncthreads_or("));
    try std.testing.expect(!has(w, "return load_sys(abort_word)"));
    // the deadline's clock is read once a lane has had to wait, not for every lane of every exchange
    try std.testing.expect(!has(w, "const uint64_t t0 = now_ns();"));
    // the copy-out's clock is the probe's too
    for ([_][]const u8{ "dsv41_rdma_collect_roci(", "dsv41_rdma_collect_into_roci(" }) |h| {
        const k = body(src, h) orelse return error.TestUnexpectedResult;
        try std.testing.expect(!has(k, "const uint64_t t2 = now_ns();"));
    }
    // the peers kernel's single waiter: the same vote, and no abort-word load per lane
    const p = body(src, "dsv41_rdma_collect_peers_roci(") orelse return error.TestUnexpectedResult;
    try std.testing.expect(has(p, "__syncthreads_or("));
    try std.testing.expect(!has(p, "if (load_sys(abort_word) != 0) return;"));
    try std.testing.expect(!has(p, "if (load_sys(abort_word) != 0) break;"));
}

test "the card-WAIT kernels write no word that only the proxy path reads, and read no clock the probe does not use" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    // sizes[], send_q[], send_slot[] are read by the proxy loop (rdma.zig proxyLoop, napOn) only; with the queue pairs waiting in the card
    // nothing reads them. send_last IS read (dsv41_rdma_await_send_wait).
    const pub_w = body(src, "dsv41_rdma_publish_wait(") orelse return error.TestUnexpectedResult;
    for ([_][]const u8{ "store_relaxed_sys(sizes", "store_relaxed_sys(send_q", "store_relaxed_sys(send_slot", "store_sys(sizes" }) |n| try std.testing.expect(!has(pub_w, n));
    try std.testing.expect(has(pub_w, "store_relaxed_sys(send_last"));
    const stg_w = body(src, "dsv41_rdma_stage_wait(") orelse return error.TestUnexpectedResult;
    for ([_][]const u8{ "store_relaxed_sys(sizes", "store_sys(sizes" }) |n| try std.testing.expect(!has(stg_w, n));
    // t0 feeds the probe only (null in a served engine): read under the probe's condition
    try std.testing.expect(!has(pub_w, "const uint64_t t0 = now_ns();"));
    try std.testing.expect(!has(stg_w, "const uint64_t t0 = now_ns();"));
    const aw = body(src, "dsv41_rdma_await_send_wait(") orelse return error.TestUnexpectedResult;
    try std.testing.expect(!has(aw, "now_ns()"));
}

test "the user's last published sequence lives in device memory: the await kernel reads it from L2, not from pinned host memory over C2C" {
    const src = @embedFile("rdma.zig");
    // sendLast()'s host words stay in the metadata block (its layout is the in-place ring's contract) but no kernel is given them
    try std.testing.expect(std.mem.indexOf(u8, src, "r.metaDev(r.sendLast())") == null);
    try std.testing.expect(std.mem.indexOf(u8, src, "fn sendLastDev(") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "cuda.DeviceBuffer.alloc(d, scalars_bytes)") != null);
    // both the publish and the await take it (the proxy path's and the card-WAIT path's kernels alike)
    try std.testing.expect(std.mem.count(u8, src, "r.sendLastDev()") >= 3);
}

test "the peers kernel polls a peer's lanes together, one thread a lane, not one lane after another" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    const p = body(src, "dsv41_rdma_collect_peers_roci(") orelse return error.TestUnexpectedResult;
    try std.testing.expect(has(p, "threadIdx.x < nl"));
    try std.testing.expect(!has(p, "for (int k = 0; k < (int)lanes[p]; ++k)"));
}

fn fnBody(src: []const u8, header: []const u8) ?[]const u8 {
    const at = std.mem.indexOf(u8, src, header) orelse return null;
    const end = std.mem.indexOfPos(u8, src, at, "\n    }\n") orelse return null;
    return src[at..end];
}

test "Ring.fail never writes a generation word: a failed rank poisons the abort word and fences its queue pairs, it does not release a peer's wait" {
    const src = @embedFile("rdma.zig");
    // (fail() hands the reason to failLocal for the ring and for the ring whose failure it follows: the writes are failLocal's)
    const f = fnBody(src, "    fn failLocal(r: *Ring, why: []const u8) void {") orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, fnBody(src, "    fn fail(r: *Ring, why: []const u8) void {") orelse return error.TestUnexpectedResult, "gen") == null);
    // the vote that replaced the abort-word load is safe only because a failure reaches a waiting kernel as the abort word (or a missing word), never as a word that satisfies it
    try std.testing.expect(std.mem.indexOf(u8, f, "gen") == null);
    try std.testing.expect(std.mem.indexOf(u8, f, "@atomicStore(u64, r.abortWord(), 1, .release)") != null);
}

test "the card-WAIT kernels run only on a ring whose queue pairs wait in the card: a ring that fell back to the proxies launches the proxy kernels" {
    const src = @embedFile("rdma.zig");
    // every choice between a _wait kernel and its proxy twin is made on r.cd (the card-WAIT backend), which a proxy ring does not have
    try std.testing.expect(std.mem.indexOf(u8, src, "const waiting = r.cd != null;") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "(if (waiting) pw.f else r.k.publish)") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "if (r.cd != null) return r.stageWait(") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "if (r.cd != null) {\n            const fw = r.k.await_send_wait") != null);
    // and the _wait staging kernel is launched from stageWait alone
    const sw = fnBody(src, "    fn stageWait(") orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, sw, "Kernels.pick(r.k.stage_wait, r.k.stage_wait_x)") != null);
    const st = fnBody(src, "    fn stage(r: *Ring") orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, st, "r.k.stage_wait") == null);
}

test "the fault record has one writer: the first failing lane claims it with a compare-and-swap, a later one only stores the abort word" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    const r = body(src, "void roci_record(") orelse return error.TestUnexpectedResult;
    const cas = std.mem.indexOf(u8, r, "atomicCAS(f, 0ull,") orelse return error.TestUnexpectedResult;
    const first_field = std.mem.indexOf(u8, r, "fault[1] =") orelse return error.TestUnexpectedResult;
    try std.testing.expect(cas < first_field); // every field store is inside the claim
    try std.testing.expect(std.mem.indexOf(u8, r, "== 0ull) {") != null);
}

test "the wait loops poll relaxed and take one acquire load after them; the peers kernel's slot and row arithmetic is done before the wait" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    for ([_][]const u8{ "int roci_waited(", "dsv41_rdma_collect_peers_roci(" }) |h| {
        const k = body(src, h) orelse return error.TestUnexpectedResult;
        try std.testing.expect(has(k, "(seen = load_relaxed_sys(gen"));
        try std.testing.expect(!has(k, "(seen = load_sys(gen"));
        try std.testing.expect(has(k, "if (!failed) seen = load_sys(gen"));
    }
    const p = body(src, "dsv41_rdma_collect_peers_roci(") orelse return error.TestUnexpectedResult;
    const slot_at = std.mem.indexOf(u8, p, "const int slot = (int)(q % slots);") orelse return error.TestUnexpectedResult;
    const wait_at = std.mem.indexOf(u8, p, "__syncthreads_or(failed)") orelse return error.TestUnexpectedResult;
    try std.testing.expect(slot_at < wait_at);
    try std.testing.expect(has(p, "per_row == 1 ? u : u / per_row"));
}

test "with the generation-word peers kernel the copy-out copies this rank's own slice and the posting kernel does not" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    const p = body(src, "dsv41_rdma_collect_peers_roci(") orelse return error.TestUnexpectedResult;
    const own_at = std.mem.indexOf(u8, p, "own_dst != nullptr && blockIdx.y == 0") orelse return error.TestUnexpectedResult;
    const early = std.mem.indexOf(u8, p, "if (p < 0) return;") orelse return error.TestUnexpectedResult;
    try std.testing.expect(own_at < early); // a grid of one with no peer still copies it
    const host = @embedFile("rdma.zig");
    try std.testing.expect(std.mem.indexOf(u8, host, "const post_own: DevicePtr = if (fold_own) 0 else mine.ptr;") != null);
    try std.testing.expect(std.mem.indexOf(u8, host, "r.s.roci.on() and r.s.collect_peers and r.k.collect_peers_roci != null") != null);
    try std.testing.expect(std.mem.indexOf(u8, host, "if (b.count != 19) return error.BadInfo;") != null);
}

test "the exchange log layout the kernel writes is the layout rdma.zig's xlogDecode reads" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    const p = body(src, "dsv41_rdma_collect_peers_roci_x(") orelse return error.TestUnexpectedResult;
    try std.testing.expect(has(p, "e[0] = (q << 3) | (uint64_t)p;"));
    try std.testing.expect(has(p, "fault + 8 + ((q * 8 + (uint64_t)p) & 65535) * 4"));
    try std.testing.expect(has(p, "e[2] = t_enter;"));
    try std.testing.expect(has(p, "((q * 8 + (uint64_t)p) & 65535) * 4 + 3] = now_ns();"));
    try std.testing.expectEqual(@as(usize, 8192), @import("rdma.zig").xlog_records);
    // the receive window is 8,192 sequences x 8 peer slots: the kernel's mask is the record count less one, and the doorbell log sits behind it
    try std.testing.expectEqual(@as(usize, 65536), @import("rdma.zig").xlog_recv_records);
    try std.testing.expectEqual(@as(usize, 8192), @import("rdma.zig").xlog_recv_records / 8);
}

test "the expert-count log layout the kernel writes is the layout mx.zig's nactDecode reads" {
    const gpa = std.testing.allocator;
    const src = std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/cuda/dsv41/native.cu", gpa, .limited(1 << 25)) catch return error.SkipZigTest;
    defer gpa.free(src);
    const at = std.mem.indexOf(u8, src, "tf_dsn_nact_log_kernel(") orelse return error.TestUnexpectedResult;
    const k = src[at .. at + 1400];
    try std.testing.expect(has(k, "ring + 8 + (i & 4095) * 4"));
    try std.testing.expect(has(k, "r[0] = i + 1;"));
    try std.testing.expect(has(k, "r[1] = ((unsigned long long)rows << 32) | (unsigned)n;"));
    try std.testing.expect(has(k, "r[2] = ((unsigned long long)sh << 32) | (unsigned)owned;"));
    try std.testing.expectEqual(@as(usize, 4096), @import("mx.zig").nact_records);
}

test "the doorbell log the card-WAIT kernels write is the layout rdma.zig's xlogPubDecode reads" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    for ([_][]const u8{ "dsv41_rdma_stage_wait_x(", "dsv41_rdma_publish_wait_x(" }) |h| {
        const k = body(src, h) orelse return error.TestUnexpectedResult;
        try std.testing.expect(has(k, "xlog_pub[(q & 8191) * 2] = q;"));
        try std.testing.expect(has(k, "xlog_pub[(q & 8191) * 2 + 1] = now_ns();"));
        try std.testing.expect(has(k, "if (xlog_pub) probe = nullptr;"));
    }
}

test "with TF_DS_ROCI_XLOG off the exchange runs the instructions it ran before the log: the plain kernels hold no logging, the _x kernels are the plain ones plus their xlog lines" {
    const gpa = std.testing.allocator;
    const src = (try load(gpa)) orelse return error.SkipZigTest;
    defer gpa.free(src);
    for ([_][]const u8{ "dsv41_rdma_stage_wait", "dsv41_rdma_publish_wait", "dsv41_rdma_collect_peers_roci" }) |name| {
        var hb: [96]u8 = undefined;
        const plain = body(src, std.fmt.bufPrint(&hb, "{s}(", .{name}) catch unreachable) orelse return error.TestUnexpectedResult;
        var xb: [96]u8 = undefined;
        const x = body(src, std.fmt.bufPrint(&xb, "{s}_x(", .{name}) catch unreachable) orelse return error.TestUnexpectedResult;
        try std.testing.expect(!has(plain, "xlog"));
        // the _x body without the lines that mention the log is the plain body (its name apart)
        var stripped: std.ArrayList(u8) = .empty;
        defer stripped.deinit(gpa);
        var it = std.mem.splitScalar(u8, x, '\n');
        while (it.next()) |line| {
            if (std.ascii.findIgnoreCase(line, "xlog") != null) continue;
            try stripped.appendSlice(gpa, line);
            try stripped.append(gpa, '\n');
        }
        var expect: std.ArrayList(u8) = .empty;
        defer expect.deinit(gpa);
        try expect.appendSlice(gpa, plain);
        try expect.append(gpa, '\n');
        const renamed = try std.mem.replaceOwned(u8, gpa, stripped.items, std.fmt.bufPrint(&xb, "{s}_x(", .{name}) catch unreachable, std.fmt.bufPrint(&hb, "{s}(", .{name}) catch unreachable);
        defer gpa.free(renamed);
        try std.testing.expectEqualStrings(expect.items, renamed);
    }
}
