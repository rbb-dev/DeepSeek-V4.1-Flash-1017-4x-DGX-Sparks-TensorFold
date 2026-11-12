//! DeepSeek-V4.1-Flash on CUDA in Zig, tensor parallel over two GB10s: the served Python family's kernels, host first.

pub const Config = @import("config.zig").Config;
pub const config = @import("config.zig");
pub const native = @import("native.zig");
pub const rank_cache = @import("rank_cache.zig");
pub const plan = @import("plan.zig");
pub const engram = @import("engram.zig");
pub const picture = @import("picture.zig");
pub const cublaslt = @import("cublaslt.zig");
pub const fmha = @import("fmha.zig");
pub const vit = @import("vit.zig");
pub const vision = @import("vision.zig");
pub const pil = @import("pil.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const weights = @import("weights.zig");
pub const engram_io = @import("engram_io.zig");
pub const engram_aio = @import("engram_aio.zig");
pub const engram_pack = @import("engram_pack.zig");
pub const engram_cache = @import("engram_cache.zig");
pub const engram_gpu = @import("engram_gpu.zig");
pub const engram_rows = @import("engram_rows.zig");
pub const link = @import("link.zig");
pub const comm = @import("comm.zig");
pub const tri = @import("tri.zig");
pub const tri_basic = @import("tri_basic.zig");
pub const cublas = @import("cublas.zig");
pub const elf = @import("elf.zig");
pub const exl3_linear = @import("exl3_linear.zig");
pub const mx = @import("mx.zig");
pub const idx_spread = @import("idx_spread.zig");
pub const tri_norm = @import("tri_norm.zig");
pub const tri_index = @import("tri_index.zig");
pub const idx_split = @import("idx_split.zig");
pub const tri_attn = @import("tri_attn.zig");
pub const tri_markov = @import("tri_markov.zig");
pub const tri_hc = @import("tri_hc.zig");
pub const exl3_prefill = @import("exl3_prefill.zig");
pub const ops = @import("ops.zig");
pub const exl3_experts = @import("exl3_experts.zig");
pub const exl3_experts_rec = @import("exl3_experts_rec.zig");
pub const prompt = @import("prompt.zig");
pub const exact = @import("exact.zig");
pub const rdma_seq = @import("rdma_seq.zig");
pub const rdma = @import("rdma.zig");
pub const exl3_experts2d = @import("exl3_experts2d.zig");
pub const exl3_experts2d_rec = @import("exl3_experts2d_rec.zig");
pub const prompt2d = @import("prompt2d.zig");
pub const wc_verdict = @import("wc_verdict.zig");
pub const ring2d = @import("ring2d.zig");
pub const neighbor = @import("neighbor.zig");
pub const text = @import("text.zig");
pub const round = @import("round.zig");
pub const round2d = @import("round2d.zig");
pub const draft2d = @import("draft2d.zig");
pub const draft_vocab = @import("draft_vocab.zig");
pub const markov_tokens = @import("markov_tokens.zig");
pub const draft = @import("draft.zig");
pub const sampling = @import("sampling.zig");
/// TF_DS_HEAD_CAND: a sampled round's candidates from each node's columns of the head, merged on rank 0.
pub const head_cand = @import("head_cand.zig");
pub const model = @import("model.zig");
pub const lanes = @import("lanes.zig");
pub const spill_start = @import("spill_start.zig");
/// The spill tier's state layout and packing.
pub const spill_state = @import("spill_state.zig");
/// The SSD tier on rank 0: the prompt cache's Tier over core/spill's store.
pub const spill_tier = @import("spill_tier.zig");
/// tf-dsv41-rows-replay's pass sizes and, at each, the nodes' quarters, the planes' offsets and a control's row.
/// A decode ring's progress watchdog.
pub const ring_watch = @import("ring_watch.zig");
/// Where this rank's host is, for a failure that names it.
pub const where = @import("where.zig");
pub const pass_replay = @import("pass_replay.zig");
pub const qgemv = @import("qgemv.zig");
/// The owned-rows plan of a chunk of any row count.
pub const slab_plan = @import("slab_plan.zig");
pub const any_check = @import("any_check.zig");
pub const replay_runner = @import("replay_runner.zig");
/// Rocinante: the zero-copy two-rail transport.
pub const rocinante = @import("rocinante.zig");
/// Rocinante on ibverbs.
pub const rocinante_verbs = @import("rocinante_verbs.zig");
/// The ConnectX's DEVX interface: a queue pair the card waits on a word with, its entries written by hand.
pub const mlx5_devx = @import("mlx5_devx.zig");
/// Rocinante on queue pairs that wait in the card.
pub const rocinante_cd = @import("rocinante_cd.zig");
/// The decode rings' queue pairs that wait in the card (TF_DS_RING_WAIT).
pub const ring_wait = @import("ring_wait.zig");
/// Rocinante inside the engine.
pub const roci_engine = @import("roci_engine.zig");
/// The grouped experts' tiles in an order that starts on a node's own rows.
pub const local_first = @import("local_first.zig");
/// Collectives of the ranks' host threads over the engine's own links.
pub const star = @import("star.zig");

/// A sleeping wait on a counter another thread advances (the Engram worker's and the engine thread's).
pub const wait_bell = @import("wait_bell.zig");

/// The SSD landing's broadcast as the WRITEs Rocinante would post.
pub const land_writes = @import("land_writes.zig");

test {
    // every declaration of every file, so functions no test calls are still compiled
    const refAll = @import("std").testing.refAllDecls;
    refAll(@import("config.zig"));
    refAll(native);
    refAll(rank_cache);
    refAll(plan);
    refAll(engram);
    refAll(checkpoint);
    refAll(weights);
    refAll(engram_io);
    refAll(engram_aio);
    refAll(engram_pack);
    refAll(engram_cache);
    refAll(engram_gpu);
    refAll(engram_rows);
    refAll(link);
    refAll(comm);
    refAll(tri);
    refAll(tri_basic);
    refAll(cublas);
    refAll(elf);
    refAll(exl3_linear);
    refAll(mx);
    refAll(tri_norm);
    refAll(tri_index);
    refAll(idx_split);
    refAll(tri_attn);
    refAll(tri_markov);
    refAll(tri_hc);
    refAll(exl3_prefill);
    refAll(ops);
    refAll(exl3_experts);
    refAll(exl3_experts_rec);
    refAll(prompt);
    refAll(exact);
    refAll(rdma_seq);
    refAll(rdma);
    refAll(exl3_experts2d);
    refAll(exl3_experts2d_rec);
    refAll(prompt2d);
    refAll(ring2d);
    refAll(text);
    refAll(round);
    refAll(@import("graph_cache.zig"));
    refAll(@import("graph_fallback.zig"));
    refAll(round2d);
    refAll(draft2d);
    refAll(draft_vocab);
    refAll(markov_tokens);
    refAll(draft);
    refAll(sampling);
    refAll(head_cand);
    refAll(model);
    refAll(lanes);
    refAll(ring_watch);
    refAll(where);
    refAll(pass_replay);
    refAll(qgemv);
    refAll(slab_plan);
    refAll(rocinante);
    refAll(rocinante_verbs);
    refAll(mlx5_devx);
    refAll(rocinante_cd);
    refAll(ring_wait);
    refAll(roci_engine);
    refAll(local_first);
    refAll(star);
    refAll(land_writes);
    refAll(wait_bell);
    _ = @import("roci_pair.zig");
    _ = @import("pack_epilogue.zig");
    _ = @import("wc_verdict.zig");
    _ = @import("pair_spans.zig");
    refAll(@import("keep_ahead.zig"));
    _ = @import("pass_meter.zig");
    _ = @import("engram_partner.zig");
    _ = @import("rocinante_test.zig");
    _ = @import("rocinante_cd_test.zig");
    _ = @import("kset_test.zig");
    _ = @import("any_check.zig");
    refAll(replay_runner);
    refAll(spill_state);
    refAll(model.Model.SpillDev);
    refAll(spill_state.Packer(model.Model.SpillDev));
    _ = &model.Model.spillLayout;
    _ = &model.Model.spillGenerations;
    _ = &model.Model.spillPacker;
    _ = &model.Model.resetEngramAhead;
    _ = &model.Model.spillCheckResumed;
    _ = &model.Model.spillRefPlanes;
    _ = @import("spill_state_native_test.zig");
    refAll(@import("land.zig"));
    _ = @import("land_test.zig");
    _ = @import("ring_hotpath_test.zig");
    _ = @import("kv_vote_test.zig");
    refAll(@import("kv_vmm.zig"));
    refAll(@import("spill_identity.zig"));
    _ = @import("spill_identity_test.zig");
    refAll(@import("probe.zig"));
    refAll(spill_tier);
    refAll(spill_tier.Leader(model.Model.SpillDev));
    _ = @import("spill_tier_test.zig");
    refAll(@import("spill_open.zig"));
    _ = @import("spill_open_test.zig");
    refAll(@import("spill_start.zig"));
    _ = @import("spill_start_test.zig");
    _ = @import("probe_test.zig");
    refAll(@import("spill_sync.zig"));
    _ = @import("spill_sync_test.zig");
    refAll(@import("spill_wire.zig"));
    refAll(@import("spill_mirror.zig"));
    refAll(@import("spill_mirror.zig").Mirror(model.Model.SpillDev));
    _ = @import("spill_mirror_test.zig");
    _ = &native.followerTier;
}
