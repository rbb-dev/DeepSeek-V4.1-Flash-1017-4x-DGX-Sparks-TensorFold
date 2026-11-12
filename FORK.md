# This repository's features

This repository is a git fork of BertholomusAI's TensorFold branch `deepseek-v41-zig-tp4`, which adds DeepSeek-V4.1 to
[TensorFold](https://github.com/ashhart/TensorFold) (Ash Hart and the TensorFold contributors). The engine work listed
here sits on top of that branch as its own git commits. The four-Spark deployment began as
[MiaAI-Lab's recipe](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold). The engine image is
built from this checkout ([`zig/Dockerfile`](zig/Dockerfile)); there is no patch series.

[`README.md`](README.md) gives the measured speeds of the whole system. This file lists the features one by one.

## How to read an entry

- **Settings**: the `TF_DS_...` names or script settings that control the feature. [`scripts/served.sh`](scripts/served.sh)
  sets the ones this repository runs with, and [`scripts/config.sh`](scripts/config.sh) documents the script settings. A
  feature the engine always applies has no switch; its name is refused at start, with the reason.
- **Written by**: "this repository" means the commits of `rbb-dev`, developed with Claude Code. Work by others is named.
- **Measured**: on four Dell Pro Max GB10 Sparks, tensor-parallel over all four, the GPU clock limited to 2,200 MHz, each
  change against a control run of the same engine without it, the replies compared byte for byte. A prompt is a cold
  prompt of real text. "Beside" means the same prompt while four requests write replies. "13.7 % faster" means 13.7 % more
  speed. Run-to-run noise on these Sparks is about 4 to 5 percent, so one run does not prove a ratio inside that; such
  changes are kept because each gives the same output at no measured loss. Where a line says otherwise (an NCCL
  build as the control, one Spark, kernel timings), that line states its own conditions.
- **Same output**: whether the engine's output is byte for byte the same with and without the feature.

## Transport: Rocinante

Rocinante is this repository's RDMA transport between the Sparks. It replaces NCCL. [`ROCINANTE.md`](ROCINANTE.md)
explains it in full.

- **Rocinante, the transport.** Every exchange between the Sparks runs on it. Prompt exchanges use its own queue pairs.
  Decode exchanges use BertholomusAI's RDMA rings, which it drives the same way. The exchanged tensors live in pinned host
  memory mapped into the GPU (zero-copy memory) and registered with both ConnectX-7 rails, so the producing kernel writes
  its output once and the network card sends it from there, with no staging copy. NCCL is not loaded, and the image's
  build fails if any engine binary links it.
  - Settings: none; it is the only transport.
  - Written by: this repository, on BertholomusAI's rings.
  - Measured, against a build of the engine that used NCCL: prompts 8 to 29 % faster from 4K to 256K tokens; replies 4 % slower
    to 2 % faster.
  - Measured, energy: the same prompt work used 5 percent less GPU energy (28.6 J against 30.2 J per 1,000 prompt tokens;
    the GPUs' power as `nvidia-smi` reports it, sampled every 200 ms, one run of three prompts each; not the power at
    the wall).
    The four Sparks drew more power (184.4 W against 164.7 W) because the work finished sooner.
  - Measured, memory: on one Spark the GPU streams zero-copy memory at 168 GB/s read and 111 GB/s write, against 252 and
    194 GB/s for ordinary device memory.
  - Same output: yes, checked at exactly 4,096 and 4,097 tokens, at about 2K, 32K and 131K tokens, and with an 8K-token
    image prompt.
- **GPU-triggered sends that wait in the card.** Each transfer is handed to the network card in advance. The GPU's stream
  triggers it when the data is complete, and the ConnectX-7 runs the send itself, so no CPU thread polls, posts or reaps
  in the transfer path. Decode's rings do the same.
  - Settings: `TF_DS_ROCI_WAIT` and `TF_DS_RING_WAIT`, both on by default; `0` falls back to proxy threads.
  - Written by: this repository.
  - Measured: between two Sparks over 1,000 round trips, the card's pre-posted wait released a write 6.21 microseconds
    after the trigger (99th percentile 6.82), against 6.64 (99th percentile 6.88) for a CPU thread posting it. At idle no
    engine thread on any Spark reaches 2 percent of a core.
  - Same output: yes.
- **Completion flags.** After the data, the card adds 1 to a flag in the receiver's memory. The receiving GPU's next
  kernel is queued behind a stream wait on that flag, so it starts when the flag lands, holding no multiprocessor and no
  CPU thread.
  - Settings: `TF_DS_ROCI_SIGNAL` (`atomic` by default), `TF_DS_RING_WORD` for the rings.
  - Written by: this repository.
- **Two prompt passes in flight.** A second set of buffers and queue pairs lets one pass compute while the other's data
  is on the wire. NCCL's waiting kernels hold the GPU, so the same overlap gained nothing there.
  - Settings: `TF_DS_ROCI_STAGGER`.
  - Written by: this repository.
  - Measured: a cold prompt of about 147K tokens went from 5,792 to 6,324 tokens a second (9.2 % faster); on NCCL the same
    overlap measured the same speed.
  - Same output: yes.
- **Send-order options for prompt passes.** An owner sum's second part is sent first; an Engram layer's rows, projection
  and first send run a layer ahead; the routed sum and its trigger run in one launch.
  - Settings: `TF_DS_ROCI_P2_FIRST`, `TF_DS_ROCI_ENGRAM_AHEAD`, `TF_DS_ROCI_COMBINE_DOOR`, `TF_DS_ROCI_SENT_FIRST`.
  - Written by: this repository.
  - Same output: yes. No separate speed figure is recorded for these.
- **Decode rings: write-combined slots, in-place writes, vector adds.** Decode's exchanges write straight into the rings'
  send slots, which are write-combined; the owner sums add four elements a thread with 16-byte loads.
  - Settings: `TF_DS_RING_WC`, `TF_DS_RING_INPLACE`, `TF_DS_ADD2_VEC`.
  - Written by: this repository, on BertholomusAI's rings.
  - Same output: yes.
- **A watchdog on the transport.** A peer whose completion flags stop advancing for 5 seconds fails the transport on that
  Spark. The queue pairs are taken from the card first, every queued wait is released, and the engine exits, logging the
  Spark, the reason, the exchange class and the peer.
  - Written by: this repository.

## Prompt path

The exchange entries below are always on, so they have no setting. All of these give the same output.

- **Owner sums.** In a layer whose work is split by rows, each Spark receives only the partial sums for its own rows: a
  gather of 13.1 MB instead of 62.9 MB.
  - Written by: this repository.
  - Measured: prompts 21.2 % faster alone (3,034 to 3,676 tokens a second), 19.5 % faster beside.
- **Exchanges that carry only owned rows or FP8 codes.** The input of `wo_b` from the column partner, the attention front,
  routing and the experts' input as FP8 codes, and the experts' pack as `down`'s FP8 codes. About 806 MB less a chunk on
  each Spark. Codes are sent only where the receiving Spark would compute exactly the same codes.
  - Written by: this repository.
  - Measured, on top of the owner sums: prompts 13.3 % faster alone and 11.5 % faster beside; the requests writing replies beside a
    prompt went from 74.9 to 79.0 tokens a second.
- **Further owned-row cuts.** Engram's layer sums at owned rows, an encoder-only chunk's boundary on owned rows, and
  `wo_b`'s partner input as FP8 codes.
  - Written by: this repository.
  - Measured, one after the other: prompts 2.8 %, 2.3 % and 1.7 % faster.
- **Prompt passes of 4,096 rows.** Where nothing else decodes, fills or waits, a prompt is read in passes of up to 4,096
  rows instead of 2,048. It costs about 1.5 GiB more memory a Spark.
  - Settings: `TF_DS_PASS_ROWS=4096`.
  - Written by: this repository.
  - Measured: 13.7 % faster alone (4,732 to 5,379 tokens a second).
  - Same output: yes.
- **Whole passes in turn while several prompts fill.**
  - Settings: `TF_DS_PASS_FILLING`, with `TF_DS_PASS_FROM`, `TF_DS_PASS_BESIDE` (the largest pass while a request
    decodes), `TF_DS_PROMPT_ORDER` and `TENSORFOLD_PREFILL_SHARE`.
  - Written by: this repository.
  - Measured: two and three cold prompts of about 147K tokens at once, 15.9 % faster in total (5,518 and 5,548 against 4,761 and
    4,786 tokens a second), first tokens 14 percent sooner, no extra memory.
  - Same output: yes.

Kept-state rules for prompts that end near a kept state, and for blank lines next to an image, are under
[Serving fixes](#serving-fixes).

## Kernels and the kernel kit's extra set

The engine loads its GPU kernels ready-built from a kernel kit. This repository uses
[its fork of BertholomusAI's kit](https://github.com/rbb-dev/deepseek-v4.1-tensorfold-tp4-4xgb10) (branch `extra-set`):
BertholomusAI's kit with an extra kernel set and its own README added. The extra set was written for this repository.
`./setup.sh` fetches the kit at a pinned commit and checks every file against its manifest. With a kit that lacks the
extra set, each setting that needs it says at start that it is off. Every kernel below is written by this repository and
gives the same output; the kernel replay compares each with the kernel or reference it replaces.

- **Block-scaled FP8 matrix multiplies, exact.** A prompt chunk's FP8 GEMMs run on the tensor cores' block-scaled
  multiply, and a split's K blocks accumulate in the tensor core with the exact accumulation.
  - Settings: `TF_DS_MMA_SF`, `TF_DS_MMA_SF_EXACT`.
  - Measured: prompts 2.2 % faster alone, 2.1 % faster beside.
- **Prompt GEMM and quantization work.**

  | Setting | What it does | Measured |
  |---|---|---|
  | `TF_DS_PF_SPLIT` | split FP8 GEMMs on a restructured loop | prompts 1.0 % faster |
  | `TF_DS_QUANT_ONCE`, `TF_DS_GROUP_TILES`, `TF_DS_TOPK_TILES`, `TF_DS_IDX_PROJ_OWNED`, `TF_DS_BF16_ROWS` | quantize once for two projections, one expert tile list, one-launch top-k, indexer projections on owned rows only, 16-row bf16 tiles | prompts 1.2 % faster together |
  | `TF_DS_PF_RUN128` | MXFP4 expert tiles staged 128 bytes at a time | prompts 0.7 % faster |
  | `TF_DS_WOA_W8` | `wo_a` from its FP8 rows | about 0.67 GB less memory a Spark, speed unchanged |
  | `TF_DS_PF_GEMM5` | dense FP8 GEMMs without a K split | no figure recorded |
  | `TF_DS_PF_T64` | the router's GEMM on a 64-row tile | no figure recorded |
- **Sparse attention and the indexer from FP4 unpacked in hardware.** They need the extra set.
  - Settings: `TF_DS_ATTN_U` (prompts 1.1 % faster), `TF_DS_IDX_U` (prompts 1.6 % faster), `TF_DS_ATTN_U_DECODE` (decode rounds).
- **Kernels that do the same work in fewer launches or cycles.** The figures are kernel timings on one Spark against a null
  control, not end-to-end speeds.

  | Setting | What it does | Measured |
  |---|---|---|
  | `TF_DS_ATTN_EPI_TWIN=we` | the fused attention epilogue's kernel | prompts 2.0 % faster |
  | `TF_DS_IDX_TPP=4` | the indexer takes four key tiles a program | kernel time 2.0 to 4.2 % shorter than the control's across the shapes of a 4,096-row pass |
  | `TF_DS_AQ_VEC` | the activation quantizer on four lanes a group | 11 % less time on the large prompt shapes |
  | `TF_DS_WOA_MERGE` | `wo_a`'s two groups in one launch | 5.5 % less time than the two launches |
  | `TF_DS_SPLIT64N` | `wkv` and `wq_a` on a 64 x 64 tile | 21.4 and 17.9 percent less kernel time |
  | `TF_DS_L2_NEXT_EARLY` | the next layer's L2 prefetch behind gate/up | in decode, the dense linears ran 2x faster when their prefetch had 80 microseconds of lead |
  | `TF_DS_ATTN_TWIN=z`, `TF_DS_ATTN_EPILOGUE`, `TF_DS_ATTN_EPI_FUSED`, `TF_DS_HC_2S`, `TF_DS_HC_STRIDE`, `TF_DS_PACK_EPILOGUE`, `TF_DS_ROPE_W`, `TF_DS_WINDOW_ONCE`, `TF_DS_QKV_PLAIN`, `TF_DS_DEC_GROUP2`, `TF_DS_MERGE_D` | fused or vectorised variants of existing kernels | no figure recorded |

## Decode

All of these give the same output; the full model checks every drafted token.

- **The drafter split four ways.** The drafter's stages are split four ways like the backbone's layers, and its
  `main_proj` is cut over each Spark's quarter of the output columns.
  - Settings: `TF_DS_DRAFT_4WAY` (needs the extra set; less memory a Spark: each holds a quarter of the drafter's stages instead of half), `TF_DS_DRAFT_MAIN_4WAY`.
  - Written by: this repository.
  - Measured: decode 2.9 % faster for one request and 4.0 % faster for 16.
- **The drafter's output head as MXFP4**, on by default here.
  - Settings: `TF_DS_DRAFT_HEAD=fp4`.
  - Written by: MiaAI-Lab, in the recipe's engine patch `0001`; this repository measured it and turned it on by default.
  - Measured: one request, prose 1.9 % faster, code 1.0 % faster.
- **A short vocabulary for the drafter.** The drafter proposes from 32,768 token ids (8,192 on each Spark), so its head
  reads about a sixteenth of the bytes. A guard on the first Spark returns to the whole head while more than 2 percent of
  the last 4,096 committed tokens fall outside the list. Replies are the model's own whatever the list holds.
  - Settings: `TF_DS_DRAFT_VOCAB`.
  - Written by: this repository.
- **Each Spark ranks its own quarter of the vocabulary.** A sampled round's candidates and a greedy round's tokens come
  from each Spark's own columns of the head, merged on the first Spark, instead of every Spark exchanging whole rows of
  logits.
  - Settings: `TF_DS_HEAD_CAND`, `TF_DS_HEAD_KEY`.
  - Written by: this repository.
  - Measured: sampled prose 0.74 % faster (16 requests) and 0.61 % faster (one); greedy code 1.35 % faster (16).
- **Decode-round work.** The rings' copy-out reads every peer's slot at once, a round's first candidates come before the
  drafter's absorb, a lone request's copied continuations span more rows, and a decode round's routed sum runs in the
  `down` kernel's launch.
  - Settings: `TF_DS_RDMA_COLLECT_PEERS`, `TF_DS_CAND_FIRST`, `TF_DS_COPY_ROWS`, `TF_DS_COPY_MATCH`, `TF_DS_DECODE_FUSE`,
    `TF_DS_HC_LATE_FORK`.
  - Written by: this repository.
  - Measured: copy-out from every peer, candidates first, the drafter's main projection four ways and wider copy windows
    together, decode for one request 1.3 % faster. Sparse attention from the kit's twin, the four-way drafter, the pre-faulted
    Engram cache and the copy windows together, decode 3.6 % faster (one request) and 4.6 % faster (16), prompts 1.1 % faster.
- **Decode rounds counted by plan.** Rounds, copied continuations and the head's drafts are counted on `/metrics` and in
  each request's done line, and a round is counted by the path of the head it took.
  - Written by: this repository.

## Engram

Engram is the model's n-gram memory, two tables of about 384 million rows.

- **Packed rows, the whole tables on every Spark.** Engram's packed rows (one read per row) are MiaAI-Lab's, patch `0003`
  of MiaAI-Lab's recipe. This repository runs them with the whole tables on every Spark, one of the options, and adds the
  scripts that pack, check and copy the shards.
  - Settings: `ENGRAM_PACKED=1`, `ENGRAM_OWNED=0`, `ENGRAM_CACHE_MIB`.
  - Written by: MiaAI-Lab (the packed rows); this repository (the placement, `scripts/engram-pack.sh` and
    `scripts/engram-verify.py`).
  - Measured: prompts 1.9 % faster alone and 44.8 % faster beside four requests writing replies (1,896 to 2,746 tokens a second);
    those requests 26.1 % faster.
  - Same output: yes.
- **A shard checked against its tables.** `scripts/engram-verify.py` compares every record, in order, with its table row's
  values and scales and every header field, and a shard is used only when its verified line matches the file. Shards are
  cut at the exact sizes the tables' headers give.
  - Written by: this repository.
- **The row cache faulted in at start**, in 2 MiB pages where the kernel gives them.
  - Settings: `TF_DS_ENGRAM_PREFAULT`.
  - Written by: this repository.
- **Read-ahead for prompt passes.** The Engram worker takes each pass's size from the pass rule and reads a layer ahead.
  - Settings: `TF_DS_ENGRAM_AHEAD_PASS`, `TF_DS_ENGRAM_KEEP_AHEAD`, `TF_DS_ENGRAM_PARTNER` (column partners split each
    other's disk reads), `TF_DS_ENGRAM_PINNED` (rows staged in pinned memory).
  - Written by: this repository.
  - Measured: the first prompt after a start 5,457 tokens a second against 4,330 without; warm prompts 2.1 % faster.
  - Same output: yes.
- **The reader sleeps.** The engine thread sleeps 100 microseconds between looks for Engram's rows
  (`TF_DS_ENGRAM_WAIT_US`), and the disk reader sleeps between batches; see [Heat](#heat-no-spinning-threads).

## Long context: 2M and 4M windows

- **Windows past the trained 1M.** DeepSeek trained V4.1-Flash to 1,048,576 tokens with YaRN factor 16. This repository
  can stretch the positions to 2,097,152 tokens (factor 32) or 4,128,768 (factor 64). The 4M window is 63 x 65,536
  tokens: the decode indexer launches one GPU block per 64 earlier tokens on a grid axis limited to 65,535 blocks, and its
  ready-built kernels need that count to be a multiple of 16. At start the engine refuses a window that breaks either
  rule and names the nearest one that works.
  - Settings: `CONTEXT` with `TF_DS_YARN_FACTOR` (32 or 64); the kit's rotary tables for the factor.
  - Written by: this repository.
  - Same output: no. Factors 32 and 64 are not the checkpoint's own position encoding, so both are off by default.
- **Needle tests.** `tools/needle_bench.py`, `tools/multi_needle.py` and `tools/needle_test.sh` hide codes in filler text
  and ask for them.
  - Measured: 10 of 10 codes at 1,995,164 tokens (2M window); 10 of 10 at 3,990,287 tokens (4M window, cold, 21 minutes);
    20 of 20 at 3,990,312 tokens after an engine restart, from the SSD tier, in 7.2 s.
  - These tests show that the model finds and quotes facts anywhere in 4M tokens; they do not show that it reasons across
    that much text.
  - Written by: this repository; the checks in `needle_bench.py` that fail on bad references are SvangenStudios's (see
    below).

## The SSD tier

- **Kept conversations on every Spark's NVMe.** A conversation's kept prompt state is also written to disk, and restored
  when the conversation returns after its state left memory or after an engine restart. The tier is named by every rank's
  weight digest and the build's identity, so a state never restores into other weights or another build. A problem turns
  the tier off, named on `/health` and `/metrics`; the engine keeps serving from memory. With `TF_DS_PREFILL_YIELD=1`,
  restores, landings and writes are serviced between a prompt's chunks.
  - Settings: `SPILL_GIB` (the store's limit on each Spark; 0 is off), `SPILL_FLOOR_GIB`, `SPILL_TURNS`,
    `SPILL_STAGING_MIB`, and the `TF_DS_SPILL_*` family (for example `TF_DS_SPILL_RESTORES`, `TF_DS_SPILL_DIR`).
  - Written by: rbb-dev, with Robert Wojciechowski (wojo) and Coinupbtc. It is built on wojo's tier for MiaAI-Lab's
    two-Spark recipe (its pull request #30); the between-chunks test follows Coinupbtc's pull request #5.
  - Measured: after an engine restart, the state of a 3,989,504-token prompt restored in 664 ms and the whole request
    answered in 7.2 s, against 21 minutes cold.

## The KV pool and Turbo

- **A KV pool sized from free memory.** The pool lives in a CUDA virtual-memory reservation and takes the memory above a
  free target on each Spark; it is resized while serving. At the 1M window the pool holds about 18.3M positions.
  - Settings: `TF_DS_KV_VMM=1`, `TF_DS_KV_FREE_TARGET_GIB=12`.
  - Written by: this repository.
- **Turbo.** A prompt is read in passes of 8,192 rows, with the dense matrix multiplies launched as 4,096-row halves. The
  larger buffers come out of the KV pool, never out of the memory the Sparks keep free. The dashboard's button sets both
  settings on all four Sparks.
  - Settings: `TF_DS_PASS_ROWS=8192`, `TF_DS_GEMM_HALVES=1`.
  - Written by: this repository.
  - Measured: a cold prompt of real text of about 152K tokens, 7,468 against 6,729 tokens a second (11.0 % faster); the KV pool
    10,420,224 positions against 18,350,080 (43.2 percent less); free memory unchanged.
  - Same output: yes.

## The built-in dashboard and live settings

- **A dashboard in the engine.** The engine serves a page at `/admin/ui/` and a JSON API under `/admin/api/v1` on the
  API's own port. They show the four Sparks (temperatures, load, memory, NVMe wear, each rail's traffic, the GPU clock
  limit), each request in flight with its stage and speed, the KV pool and the SSD tier, Turbo, and every setting the
  engine reads. It is off unless the engine is given an admin key file. A key's label sets its scope (`admin` or
  `admin-...` may do everything; any other label reads only); the admin keys are separate from the API keys; every refusal
  of a caller without a valid key is the 404 any unknown path gets.
  - Settings: `ADMIN_KEY_FILE`, `ADMIN_POLICY_FILE`, `ENGINE_CFG`.
  - Written by: this repository.
- **Live settings.** The settings table gives every setting its default, range, label and documentation, whether it takes
  effect live, and whether it needs every Spark to agree. An admin key sets one on all four Sparks, reverts it, or keeps
  it across restarts in `engine.cfg`. A setting name the engine does not have is refused at start, with the reason. The
  page has a test of its own (`zig/tests/admin_ui/check.mjs`).
  - Written by: this repository.
- **Goals.** An alert-only tab for what the engine is asked to keep, decode speed for each user and prompt speed: it
  measures and reports and changes nothing.
  - Written by: this repository.

## Heat: no spinning threads

A thread that waits by checking a value in a loop keeps its core at 100 percent while doing nothing. On a GB10 the CPU
and the GPU share one power budget and one cooler. Each wait below sleeps or is handed to the network card. Every entry
is written by this repository and gave equal replies byte for byte.

Temperatures are the Sparks' own thermal zones, read every 5 seconds on all four under the same load (16 requests at
once), each arm starting from the same idle temperature (52 °C or less).

- **CUDA blocking sync.** A host thread waiting on the GPU sleeps until it signals. The idea is TensorFold's
  `TF_CUDA_SCHED` (Ash Hart).
  - Settings: `TF_DS_CUDA_SCHED=blocking`.
  - Measured: the engine thread's 100 percent spin on a fast core is gone on the three workers; decode 0.3 % slower and 0.1 % slower
    (prose, code, one request), 0.15 % slower (16); prompts 1.5 % faster alone, 0.3 % faster beside; the hottest zone 3.1 to 6.7 °C cooler,
    and the cluster that had spun 10.1 to 13.9 °C cooler.
- **Event waits sleep.** Every CUDA event is made with the blocking-sync flag, because a wait on an event made without it
  busy-waits whatever the context's scheduling says.
  - Settings: `TF_DS_CUDA_EVENTS=blocking`.
  - Measured: the first Spark's engine thread went from 91 to 4 percent of a fast core; decode 0.15 % faster and 0.13 % slower (one
    request), 0.5 % slower and 0.1 % slower (16); prompts 0.2 % faster and 0.05 % slower; the hottest zone 6.6 °C cooler and the fast cluster
    11.3 °C cooler.
- **The RDMA proxies nap.** When the card's wait is off, each ring's proxy thread waits for the GPU's trigger with the Arm
  CPU's timed wait-for-event (WFET) instead of polling. It works beside the idle park by Hironori Uchikawa (uchkw) in
  MiaAI-Lab's recipe.
  - Settings: `TF_DS_RDMA_WFE=1`.
  - Measured: the proxies nap 93.7 to 98.9 percent of the time; decode 0.15 % faster and 0.04 % faster (one request), 0.02 % faster and
    0.53 % slower (16); small-core zones 3.2 to 6.6 °C cooler.
- **The engine thread sleeps while issuing a prompt.** It stays at most four layers ahead of the GPU and sleeps, instead
  of spinning in the driver's full launch queue.
  - Settings: `TF_DS_ISSUE_LEAD=4`.
  - Measured: the thread's CPU time over wall time went from 1.00 to 0.01-0.05 on all four Sparks; a cold prompt of about
    147K tokens 6,488 to 6,495-6,497 tokens a second (unchanged); fast-core zones 3.8 °C cooler on average over the
    window, 7.9 °C at the peak.
- **Idle and wait tuning.**
  - Settings: `TF_DS_ENGRAM_WAIT_US` (100 microseconds), `TF_DS_HYBRID_WAIT` (a decode round's reads are waited for asleep
    until about 250 microseconds before their learned end), `TF_DS_ROCI_IDLE_PARK_US` (idle transport threads sleep in the
    kernel).
- **NCCL's polling thread on small cores.** In a build that used NCCL, `TF_DS_NCCL_CPUS` ran its polling thread on four
  small cores: the fast cluster where it had spun measured 11.5 to 12.5 °C cooler, with no speed lost. The engine has no
  NCCL, so the setting is refused at start.

## The network and the setup

All written by this repository.

- **RoCE with priority flow control on a MikroTik CRS812.** PFC on the RoCE class at the switch and on priority 3 at
  every card. [`NETWORK.md`](NETWORK.md) and [`network/`](network/) give the switch's configuration from the factory
  state and each Spark's files (`51-cx7-roce-static.yaml`, `roce-pfc.service`). Measured: 0 drops and 0 resends under
  overload.
- **One-command setup.** `./setup.sh` runs five steps from one config file: config, access (ssh, and sudo: passwordless, or a rule for only the start's two root commands),
  network, data and start. The network step keeps a copy of every netplan file, stops if another one configures a rail,
  then sets each Spark's two rails and RoCE flow control and checks every rail: up, MTU, PCIe link at x4, PFC on, and a
  full-size unfragmentable ping between every pair on both rails. `scripts/setup-data.sh` builds the image from the
  checkout, downloads the checkpoint, fetches the kit and makes each Spark's weight file and Engram shards; then every
  worker gets its share over the fabric. The start step makes an API key and an admin key unless the config names
  yours, starts the engine and prints the addresses and keys. The long steps print per cent done and the time left.
- **`tools/fabric-xfer`, the setup's copy.** RDMA over both rails, in C with libibverbs (nothing to install on a
  DGX Spark): each file is read once on the first Spark and written to every worker that needs it, files routed to
  different workers are read interleaved so every worker's disk writes at once, and every chunk's CRC-32C is compared on
  both ends before a file is renamed into place. The receivers accept one authenticated connection from the sender's
  addresses only and write nothing outside their folder. Measured on four Sparks: the three 80 GB weight files from the
  first Spark to the other three in 44.5 s (5.41 GB/s: the first Spark's disk read is the limit); Engram's 405.5 GB
  packed shards to all three at once in 207 s.
- **`scripts/fabric-copy.sh`.** The fallback copy where RDMA is not available: files go to a worker as `tar | nc` streams
  spread over its fabric addresses (ssh for control only); only what the worker lacks or holds at another size is sent,
  and every file's SHA-256 is compared on both ends.
- **`scripts/served.sh`.** Every setting the engine runs with, each with a comment. `scripts/local.sh` lines win, and the
  environment wins over both. Settings are numbered per worker (`WORKER1` to `WORKER3`).
- **A start that drops page caches and limits the GPU clock.** Every start drops each Spark's page cache (`DROP_CACHES`),
  limits the GPU clock (`GPU_CLOCK`, in MHz) and checks the rails (`RAIL_CHECK`). `./stop.sh` waits for running requests
  up to `DRAIN_TIMEOUT`.
- **Weights read from each Spark's own disk.** Each rank reads its weight file past the page cache (O_DIRECT), each byte
  once, in 256 MiB windows, and an expert's pack is read in the rank cache's order. Setting: `TF_DS_DIRECT_WEIGHTS`.
  Measured: the weights load in 27 to 47 seconds.
- **API keys.** `API_KEY_FILE` names a file of `label: key` lines readable only by its owner; `METRICS_OPEN` keeps
  `/metrics` open. With keys on, `/health` answers `{"status": "ok"}` to a request without a key and in full to one with a
  key.

## Serving fixes

All written by this repository unless a name is given; the carried fixes of others are in the next section.

- A request without `max_tokens` is served with every token left, and one asking for more than fits gets a 400 naming
  the most it may ask. The server is told the window a stream really holds.
- A prompt pass in flight shows on `/health` and `/metrics`: its tokens and speed so far, and a counter of the prompt
  tokens every pass computed. `/health` adds `cached_tokens_total`, the process start (`started_at`, `uptime_s`) and, on
  `/metrics`, `process_start_time_seconds`.
- After a sticky CUDA fault `/health` answers 503 and `/metrics` reports `tensorfold:healthy 0`; new requests are refused.
- A cancelled prompt pass keeps the state its finished chunks reached, so the same prompt resumes there.
- A request whose grammar cannot start fails alone; a finished request leaves the admitted list before it is freed.
- The reasoning split ends at the model's own close token, not the first `</think>` spelled in the text; a DSML block that
  `max_tokens` cuts reaches a stream as text, as the whole reply has it; a `<tool_call>` whose payload only opens a DSML
  block is no call and no crash.
- Polls answered 200 print no access line (`TENSORFOLD_LOG_POLLS=1` prints them).
- A failed CUDA graph capture runs that work eagerly on every rank (the key stays eager for a while); a drafter pass's
  graph is launched before it is kept; the graph cache keeps at most `TF_DS_GRAPH_CAP` stretches.
- Every rank compares the frame kinds its image speaks at start, so ranks from different images refuse to start. The
  check of settings against the first rank's is BertholomusAI's (see below).
- Blank lines go only next to an image in a request with images, so an image does not change how the conversation's
  other multi-part messages render and their kept states match.

## Tests and the kernel replay

- **Every feature carries its tests** in the engine's own suites, each seen to fail without its change. Scripts have
  offline tests that need no Spark (`tests/test_scripts.py`, `tests/test_validation_tools.py`: Python's standard library).
- **The kernel replay** (`tf-dsv41-kernels-replay`, on a GB10) runs each GPU kernel on recorded inputs against a reference
  and against the kernel it replaces. It gates each section's line count and their sum, so a section that stops running
  or printing fails it; `TF_DS_REPLAY_ONLY` runs chosen sections alone, and `TF_DS_REPLAY_SAN` and
  `TF_DS_REPLAY_CONTROLS` shorten the soaks for a run under compute-sanitizer.
- **Byte-for-byte checks of replies** with and without a change, by `tools/validate.py` and the replies' SHA-256.
- **A build check**: the image's build fails if any engine binary links NCCL.
- **The dashboard's page** is checked by `zig/tests/admin_ui/check.mjs` against a fixture that serves the workers' reports
  as the engine does.

## Fixes by others carried here

Each is in the tree under its author's name, from the source shown. Links are to the pull request or issue. "Two-Spark
recipe" is [MiaAI-Lab's two-Spark DeepSeek recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks);
"MiaAI-Lab's recipe" is the four-Spark one this repository began as. Numbered pull requests and issues are on
[bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) unless said otherwise.

| Author | Change | Source |
|---|---|---|
| Nathan McBride | The chat template's string appends grow in place, so rendering takes memory proportional to the conversation, not its square | [Manrique's pull request #24 to BertholomusAI's TensorFold](https://github.com/bertholomus/TensorFold/pull/24); upstream TensorFold's template fix `8c725d2f` |
| Jay Leaton | The JSON writer does not slice past a string that ends inside a multi-byte character | bertholomus/TensorFold `deepseek-v41-zig` commit `0937cc13f` |
| Robert Wojciechowski (wojo) | A prompt's render and tokenization give their scratch back at once; concurrent preparations wait while free memory is under a floor (`TENSORFOLD_PREPARE_FLOOR_GIB`) | Two-Spark recipe, its patch `0023` |
| Manrique | DeepSeek's own tool-call blocks are read first, from their own tags | [#25](https://github.com/bertholomus/TensorFold/pull/25) |
| Manrique | A conversation that shares another's prefix does not stale the first one's kept states | [#26](https://github.com/bertholomus/TensorFold/pull/26) |
| Manrique | Prompts with images keep and resume their states; a prompt pass in pieces between other requests' rounds | [#28](https://github.com/bertholomus/TensorFold/pull/28), [#29](https://github.com/bertholomus/TensorFold/pull/29) |
| Manrique (hashd1ve), carried by Manuel (manuel1618) | The chat template writes each piece out as it is made | [Two-Spark recipe issue #1](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/1) (`fd2cdb2` there); [MiaAI-Lab's recipe PR #18](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/18) |
| Blackwellboy | A streamed reply of only tool calls sends no leading newline as text | [#32](https://github.com/bertholomus/TensorFold/pull/32) |
| Blackwellboy | A refused Engram row id does not leak the read batch's buffers | [#38](https://github.com/bertholomus/TensorFold/pull/38) |
| Blackwellboy | `TF_DS_NO_COPIES` drops a drafted request's suffix-lookup copy windows, so the draft head proposes alone. BertholomusAI's `ATTRIBUTION.md` also lists #33 (`TF_DS_SOLO_OVERHEAD`); this tree does not carry that setting | [#34](https://github.com/bertholomus/TensorFold/pull/34), [#33](https://github.com/bertholomus/TensorFold/pull/33) |
| evilpsycho42 | A prompt ending within the window past a kept state resumes an earlier state | [#35](https://github.com/bertholomus/TensorFold/pull/35) |
| LeetJoe | An image placeholder in a message's own text is escaped, not refused | [#17](https://github.com/bertholomus/TensorFold/pull/17) |
| BertholomusAI | The Python child that decodes images other than PNG is restarted when it dies | bertholomus/TensorFold `deepseek-v41-zig` commit `24bbde8e5` |
| BertholomusAI | The regression test for kept extents, for a bug reported by sayyidfareed | [PR #10](https://github.com/bertholomus/TensorFold/pull/10); commit `9f820999f` |
| BertholomusAI | The replay floor (`TF_DS_REPLAY_FLOOR`) | `deepseek-v41-zig` commit `bdcfd10a2` |
| BertholomusAI | Every rank's settings are checked against the first rank's at start | `deepseek-v41-zig-tp4` commit `2e07790e79` |
| BertholomusAI, on a memo by plotarmordev | The tokenizer keeps each text segment's ids, so a chat's earlier messages are tokenized once; the memo's test is Ash Hart's | bertholomus/TensorFold `16c8d30e08` on upstream TensorFold `2d2930d8ba`; test `ff05511460` |
| MiaAI-Lab | A conversation with images resumes and keeps prompt states (`TF_DS_IMAGE_CACHE`) | Two-Spark recipe, its patch `0006` |
| Coinupbtc | A control token's spelling in a request's own text stays text; the text before tool calls is the reply's content | Two-Spark recipe, its patches `0028` and `0027` ([issue #21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/21)) |
| ezoushen | The CUDA graph cache keeps at most `TF_DS_GRAPH_CAP` stretches | [Two-Spark recipe PR #70](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/70) (issue #69) |
| nuk3s | A CUDA graph capture that fails on one rank runs that work eagerly on every rank | [TensorFold PR #45](https://github.com/bertholomus/TensorFold/pull/45) |
| sayyidfareed | The direct-I/O test compares against a buffered descriptor of its own | [TensorFold PR #9](https://github.com/bertholomus/TensorFold/pull/9) |
| akol1 | Accepted HTTP connections turn off Nagle (`TCP_NODELAY`) | [upstream TensorFold PR #929](https://github.com/ashhart/TensorFold/pull/929) |
| Christopher A. Williamson (cawilliamson) | A GET or DELETE request's body is read and dropped before routing | [MiaAI-Lab's recipe PR #37](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/37) |
| SvangenStudios | A JSON answer stays content when tools are offered; a reply of only orphan DSML closers is flagged | MiaAI-Lab's recipe, its patches `0042` and `0043` |
| SvangenStudios | The structured-output grammar tests compile against the current API | [MiaAI-Lab's recipe PR #30](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/30) |
| SvangenStudios | `tools/validate.py` and `tools/needle_bench.py` fail on bad references, error replies and malformed tool calls; `stop.sh` exits 1 when a rank may still run; the prepared marker covers the cache, the settings and the chat template; a failed weight-file index stops `prepare.sh` instead of deleting the weight file; the start's smoke test passes only on HTTP 200 | MiaAI-Lab's recipe pull requests [#4](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/4), [#5](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/5), [#33](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/33); commits `08893f0`, `44343fc` |
| Hector Isaza (hecisaza) | A DSML call whose value holds the block's own closer stays a call, whole or streamed | MiaAI-Lab's recipe, its patch `0052` |
| Tyler Folkman (tfolkman) | The launchers' lock descriptors stay out of persistent children; the preparation utilities run on the runtime image | MiaAI-Lab's recipe commits `462bf1b`, `75d3c6b` |
| Thomas Kempka (thkempka) | The prompt cache keys a kept state by the request's images as well as its tokens | [MiaAI-Lab's recipe PR #10](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/10) |
| Mauvis Ledford (krunkosaurus) | Vision work skipped for restored images; requests admitted between a prompt's chunks; the backports of keepalives and the golden harness's stdout drain | MiaAI-Lab's recipe pull requests [#13](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/13), [#15](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/15) |
| Manuel (manuel1618) | Cached prompt tokens on `/metrics` | [MiaAI-Lab's recipe PR #8](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/8) |
| Hironori Uchikawa (uchkw) | Idle RDMA proxies are quiesced and park in a futex wait | [MiaAI-Lab's recipe PR #2](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/2) |
| knapcio | The RDMA queue pairs' path MTU comes from the port's active MTU | MiaAI-Lab's recipe, its patch `0026`, after [bertholomus/TensorFold #12](https://github.com/bertholomus/TensorFold/pull/12) |
| Jay Leaton (jayleaton), Ash Hart | Pre-token SSE keepalives | [ashhart/TensorFold #663](https://github.com/ashhart/TensorFold/pull/663); Ash Hart's `cc857503` |
| BobClawblaw, Ash Hart, BertholomusAI | Structured output and required tool calls: BobClawblaw's lane-core grammar, Ash Hart's `tool_choice` grammar, BertholomusAI's DeepSeek-V4.1 lane | [ashhart/TensorFold #596](https://github.com/ashhart/TensorFold/pull/596); MiaAI-Lab's recipe, its patch `0036`, and the commits it lists |
| Lucas Peters, lepetitprince716-prog, Scott Woolsey Biggart, mrsergiopascal, tonyd2wild | Contributions to BertholomusAI's branch, kept under their names | BertholomusAI's `ATTRIBUTION.md` |

[`CREDITS.md`](CREDITS.md) says who wrote what at length.

## How a feature is added

A feature lands as its own commits, with its tests, its replay section when it adds a GPU kernel, and a measurement
against its own control with the replies compared byte for byte. BertholomusAI's branch is merged into this one.
`git log --first-parent` shows this repository's own line, and each commit under the merged branch keeps its author.
