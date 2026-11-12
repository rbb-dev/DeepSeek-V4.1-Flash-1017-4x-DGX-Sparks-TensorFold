# Credits

This repository is a git fork of BertholomusAI's TensorFold branch `deepseek-v41-zig-tp4`, with its own engine work added
on top as its own commits, and a deployment for four NVIDIA DGX Sparks that began as MiaAI-Lab's recipe. Much of what
makes it work was built by others, and each of them is named here. This repository's own work is licensed under the
Apache License 2.0 ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE) carries the notices that go with it. [`FORK.md`](FORK.md)
lists this repository's features and the fixes by others that it carries.

## Model

- **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** by
  [DeepSeek](https://huggingface.co/deepseek-ai): the model, its weights (MIT, Copyright (c) 2023 DeepSeek) and its
  reference inference code. The weights are not part of this repository; `scripts/prepare.sh` downloads them from
  Hugging Face and serves them as published.
- The native FP8 and MXFP4 kernels compute what DeepSeek's reference inference code computes (`inference/model.py`'s
  `linear`, `Expert` and `MoE`; `inference/kernel.py`'s `act_quant`, `fp8_gemm` and `fp4_gemm`; `inference/convert.py`'s
  bf16 `wo_a`). The arithmetic rules were read there and written anew in CUDA; no DeepSeek code is included.
- **[Keys abliteration overlay for DeepSeek-V4.1-Flash-1017](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated)**
  by drowzeys (MIT, gated): the `attn.wo_b` weights of layers 10 to 35 that `ABLIT=1` grafts onto the checkpoint. It is not
  part of this repository; `scripts/prepare.sh` downloads the sidecar and builds the overlaid checkpoint view when the
  setting is on.

## The engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors (Apache-2.0 from TensorFold 0.6.0; code written before 0.6.0 keeps its MIT notice in
  `LICENSES/MIT.txt`): the native Zig engine, its OpenAI-compatible server, its lane core, CUDA runtime, Triton AOT loader
  and EXL3 module.
- **DeepSeek-V4.1 on TensorFold** was written by [BertholomusAI (Albert Lee, @bertholomus)](https://github.com/bertholomus)
  in the branch [bertholomus/TensorFold `deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4)
  (Apache-2.0), which this repository forks: the `dsv41` family, its exact four-node 2D split with the balanced experts
  split, the Engram reader, DSpark drafting, the vision tower, the RDMA rings that carry decode's exchanges, and the tools
  that write each Spark's weight file and Engram's token map. The branch's own [`ATTRIBUTION.md`](ATTRIBUTION.md) lists
  every outside source the family uses.
- **The kernel kit** (the family's Triton kernels as captured binaries, Engram's constants, the RoPE table generator, the
  vision tower's attention kernel) is the `kit/` of BertholomusAI's four-node recipe
  [bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10)
  (Apache-2.0). `scripts/prepare.sh` fetches [this repository's fork of it](https://github.com/rbb-dev/deepseek-v4.1-tensorfold-tp4-4xgb10)
  at a pinned commit and checks it against its manifest. The fork is BertholomusAI's kit with an extra kernel set and its
  own README added; the extra set was written for this repository. The kit's own licenses (`kit/KIT-LICENSES.md`) cover the parts it carries: Triton (MIT),
  PyTorch's attention kernel (BSD-3-Clause, built on NVIDIA CUTLASS, BSD-3-Clause) and the EXL3 format of
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- The device code of `zig/kernels/cuda/dsv41/experts_par.cu` (the helpers of `grouped_cp_kernel`, arithmetic unchanged) is
  copied from TensorFold's EXL3 `experts_grouped.cuh` (MIT, Copyright (c) 2026 TensorFold contributors), which follows
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT, Copyright (c) 2025 Turboderp).
- The served settings that `scripts/config.sh` and `scripts/served.sh` start from in part follow BertholomusAI's recipe
  (`TF_DS_2D_GU=parity`, 48-row rounds, CUDA graphs, the side stream, the paced L2 prefetch, Engram by AIO, the lane core's
  round costs), from its `config.env.example` and README ("The served settings").

### Contributions to BertholomusAI's branch, kept under their authors' names

From the branch's `ATTRIBUTION.md` and `NOTICE`:

- **Lucas Peters** ([@lucas-peters](https://github.com/lucas-peters)), pull request #4: the prompt indexer passes each row
  block's `vis` 16-byte aligned.
- **lepetitprince716-prog** ([@lepetitprince716-prog](https://github.com/lepetitprince716-prog)), pull request #18: the
  loader reads each layer's expert run from the rank cache whole, with O_DIRECT.
- **Manrique** ([@hashd1ve](https://github.com/hashd1ve)), pull requests #26, #28 and #29: a shared prefix does not stale
  kept states; images' digests key kept states; a prompt pass in pieces between other requests' rounds.
- **Scott Woolsey Biggart** ([@swbiggart](https://github.com/swbiggart)), pull request #30: the optional neighbor transport
  for a physical four-node ring.
- **mrsergiopascal** ([@mrsergiopascal](https://github.com/mrsergiopascal)), pull request #31: a CPU test and a server
  reproducer for the replay cut's position alignment.
- **Blackwellboy** ([@Blackwellboy](https://github.com/Blackwellboy)), pull requests #33, #34 and #38: a setting that
  prices a lone request's head windows (#33, `TF_DS_SOLO_OVERHEAD`, not carried in this tree), `TF_DS_NO_COPIES` for a
  drafted request (#34), and the fix of the Engram AIO leak (#38).
- **LeetJoe** ([pull request #17](https://github.com/bertholomus/TensorFold/pull/17)): an image placeholder in a message's
  own text is escaped.
- **Nathan McBride** ([@nmcbride](https://github.com/nmcbride)): upstream TensorFold's template fix `8c725d2f`.
- **tonyd2wild** (MIT, Copyright (c) 2026 Tech2wild, `LICENSES/tonyd2wild-MIT.txt`): `TF_RDMA_GID`, and `TF_SEED_RANDOM` with
  `TF_TOP_K_MAX`.
- Reports that shaped the branch: [@Capicua25x](https://github.com/Capicua25x) (the loop guard's novelty signal) and
  [@den13501](https://github.com/den13501) (streamed keepalives for long prompts).

## MiaAI-Lab's recipe

The four-Spark deployment this repository began as is
[MiaAI-Lab's recipe](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold) (Apache-2.0, MiaAI-Lab
and its contributors). It supplies:

- **The scripts' structure**: the layered settings (`scripts/config.sh`), the Spark discovery, the rail detection and
  `stop.sh`, adapted from MiaAI-Lab's
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (branch `zig-engine`, Apache-2.0), with code those contributors wrote there: **webzone** (both PCIe links of a cabled
  QSFP port as rails, `d5afeb0`), **[Alexbob0](https://github.com/Alexbob0)** (workers by host name and `FABRIC_PEER`
  resolution, `5e1bf27`) and **john** (each rank given its link's address of the head as its rendezvous, `58eaed0`).
- **The chat template**, `model/chat_template.jinja`, written by MiaAI-Lab ([Mia-AiLab](https://huggingface.co/Mia-AiLab))
  for [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw): a
  Jinja port of the checkpoint's own `encoding/encoding.py` (DeepSeek, MIT). One change, by Manrique
  ([hashd1ve](https://github.com/hashd1ve)) (the two-Spark recipe's issue #1, `fd2cdb2` there), writes each piece of the
  prompt out as it is made instead of adding it to one growing string. Manuel ([manuel1618](https://github.com/manuel1618))
  carried it over in [PR #18](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/18) and
  re-checked that the rendered prompt is unchanged.
- **`tools/recipe_bench.py`**, MiaAI-Lab's benchmark from the
  [two-Spark DeepSeek recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) (`tools/bench.py`
  there), unchanged. The README's performance tables were measured with it.
- **Engine changes**, MiaAI-Lab's own unless a name is given. MiaAI-Lab's recipe numbers them as patches, and the labels
  below are that four-Spark recipe's (the two-Spark recipe's labels are named as such):
  - `0001` to `0007`, except `0003` (below): the checkpoint format on the native engine (FP8 E4M3 linears with UE8M0 scales on 32 x 32 blocks,
    MXFP4 routed experts, bf16 `wo_a`, router, head, compressor and indexer projections, checked against DeepSeek's
    reference arithmetic on real weights); the Spark order and links for a ring; video input (`video_url` parts become
    timestamped frames, decoded by PyAV); tool-call arguments sent as JSON strings parsed before the template renders
    them; kernel-variant fallbacks; alignment fixes; the drafter's output head as MXFP4 (`TF_DS_DRAFT_HEAD=fp4`, on by
    default here).
  - **`0003`: Engram's packed rows** (`ENGRAM_PACKED`), one read per row. This repository runs them with the whole
    tables on every Spark and adds the scripts that pack, check and copy the shards.
  - `0008` to `0016`: the prefill plan. It started from **Hector Isaza's** ([hecisaza](https://github.com/hecisaza)) measured
    prefill profile and his pair-sum gathers (`0008`, not carried here: this repository's engine has no NCCL); the work was
    reviewed twice with Codex (OpenAI). `0009` to `0016` are
    MiaAI-Lab's: tensor dumps at several layers, replay tools, the K-split GEMM tiling (from Hector's measured tuning), the
    indexer at RB=2 and quarter-HC (each Spark computes the hidden-state mix for its own quarter). The quarter-HC defect
    that three attempts missed was diagnosed by Codex (OpenAI) and Grok (xAI), consulted independently on the same bug.
  - `0017` to `0024`: the decode plan, written by MiaAI-Lab with Codex (OpenAI) and Grok (xAI) as a council over several
    rounds: the drafter's width past 64 rows (the failure was found by Codex from the source), the PDL controls, graph
    captures agreed on every rank (the incident was reported by Hector), and candidate sampling, expert exit and prefill
    yield (`0020` and `0024`, the opt-in pre-pass and the drafter row cost, are not carried here). Codex reviewed each of
    these patches read-only before it was built: its findings shaped the agreement protocol of `0019`, and it found the
    defects in the cancellation paths fixed in `0023`. The design of the Triton PDL control in `0018` is Codex's.
  - `0025`: the indexer's own row-sharding. An outside reviewer's observation that the indexer was still run on every Spark
    led to it.
  - `0026`: **knapcio** ([knapcio](https://github.com/knapcio)): the RDMA queue pairs' path MTU from the port's active MTU,
    after BertholomusAI's
    [TensorFold#12](https://github.com/bertholomus/TensorFold/pull/12).
  - `0027`: **[Thomas Kempka](https://github.com/thkempka)**, [PR #10](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/10):
    the prompt cache keys a kept state by the images inside its prefix as well as its tokens.
  - `0028` and `0029`: upstream TensorFold fixes backported by **[Mauvis Ledford](https://github.com/krunkosaurus)**
    ([PR #13](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/13)): pre-token SSE
    keepalives, introduced by **Jay Leaton ([jayleaton](https://github.com/jayleaton))** in
    [TensorFold #663](https://github.com/ashhart/TensorFold/pull/663) and integrated by **Ash Hart** in
    [`cc857503`](https://github.com/ashhart/TensorFold/commit/cc8575039b191df20d7739713b77744f462221c4); and the drain of
    the golden server's stdout pipe, written by **BobClawblaw** ([BobClawblaw](https://github.com/BobClawblaw)) in
    [`0baccf36`](https://github.com/ashhart/TensorFold/commit/0baccf3687972fb83bb8e95a4a4cb222600b800b).
  - `0030` and `0031`: **Mauvis Ledford**, [PR #15](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/15):
    vision work skipped for restored images, and requests admitted between a prompt's prefill chunks. `0032` is
    MiaAI-Lab's: `0031` behind `TF_DS_PREFILL_YIELD`.
  - `0033`: **Manuel** ([manuel1618](https://github.com/manuel1618)), [PR #8](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/8):
    the cached prompt-token counter on `/metrics`: a backport of commit `ce0d9053` in BertholomusAI's TensorFold by
    juliankang4; eleata ports it to the two-node branch in [TensorFold#44](https://github.com/bertholomus/TensorFold/pull/44).
  - `0034`: **Hironori Uchikawa** ([uchkw](https://github.com/uchkw)), Codex-assisted,
    [PR #2](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/2): idle RDMA proxies are
    quiesced by a READY/COMMIT handshake and park in a futex wait; the next compute frame wakes them. Its design, handshake
    and tests are uchkw's, as are the image build's CPU-only gate tests (`zig/Dockerfile`) and the validation records in
    `docs/`. MiaAI-Lab ported it onto `0031`.
  - `0035`: MiaAI-Lab's startup rendezvous, for a failure reported by [tlacerdap](https://github.com/tlacerdap) in
    [#3](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/3) and diagnosed there by
    [shanedertrain](https://github.com/shanedertrain).
  - `0036`: structured output and required tool calls, ported by MiaAI-Lab from TensorFold and BertholomusAI's fork
    ([#6](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/6)), each commit kept under
    its author. **BobClawblaw**: [`6bccf644`](https://github.com/ashhart/TensorFold/commit/6bccf644), structured output in
    the lane core and libtfgrammar, from [TensorFold #596](https://github.com/ashhart/TensorFold/pull/596). **Ash Hart**
    ([ashhart](https://github.com/ashhart)): [`bd116970`](https://github.com/ashhart/TensorFold/commit/bd116970),
    [`4145b25d`](https://github.com/ashhart/TensorFold/commit/4145b25d), [`bcc7457a`](https://github.com/ashhart/TensorFold/commit/bcc7457a),
    [`4d4ee324`](https://github.com/ashhart/TensorFold/commit/4d4ee324), [`fd16680e`](https://github.com/ashhart/TensorFold/commit/fd16680e),
    [`08652e67`](https://github.com/ashhart/TensorFold/commit/08652e67), [`6c151c40`](https://github.com/ashhart/TensorFold/commit/6c151c40)
    and the CUDA host part of [`9c1d35e9`](https://github.com/ashhart/TensorFold/commit/9c1d35e9). **bertholomus**:
    [`d4a4b4e6`](https://github.com/bertholomus/TensorFold/commit/d4a4b4e6), [`0a7e0b41`](https://github.com/bertholomus/TensorFold/commit/0a7e0b41),
    [`9f147937`](https://github.com/bertholomus/TensorFold/commit/9f147937) and [`3821abd0`](https://github.com/bertholomus/TensorFold/commit/3821abd0).
  - `0042` and `0043` (**[SvangenStudios](https://github.com/SvangenStudios)**) and `0052` (**Hector Isaza**, [hecisaza](https://github.com/hecisaza)):
    a JSON answer stays JSON when tools are offered; a reply of only orphan DSML closers is flagged; a DSML value that holds
    its block's closer stays in the call, whole or streamed.
- **Scripts contributed to MiaAI-Lab's recipe**: by **[SvangenStudios](https://github.com/SvangenStudios)**,
  [PR #4](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/4) (`stop.sh` exits 1 when a
  rank may still run; `DRY_RUN=1` writes no NCCL graph files; the prepared marker covers `HF_CACHE`, the NFS settings and the
  chat template; with `tests/test_scripts.py`), [PR #5](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/5)
  (`tools/validate.py` and `tools/needle_bench.py` fail on bad references, error replies, malformed tool calls and bad
  calibrations; with `tests/test_validation_tools.py`), [PR #33](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/33)
  (a failed weight-file index stops `prepare.sh` instead of deleting the weight file) and `start.sh`'s smoke test that
  passes only on HTTP 200 with no error object ([`08893f0`](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/commit/08893f0),
  [`44343fc`](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/commit/44343fc)); and by
  **Tyler Folkman** ([tfolkman](https://github.com/tfolkman)): the launchers' lock descriptors stay out of persistent
  children ([`462bf1b`](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/commit/462bf1b)) and the
  preparation utilities run on the runtime image ([`75d3c6b`](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/commit/75d3c6b),
  with MiaAI-Lab's [`443618f`](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/commit/443618f)).

The pair-sum and owner-gather settings of `0008` and `0016` are refused at start, because this repository's engine has no
NCCL.

## Fixes by others carried in this repository

Each is in the tree under its author's name. The sources, and what each fixes, are in the table in [`FORK.md`](FORK.md).

- [Nathan McBride](https://github.com/nmcbride): the chat template's appends grow in place (carried to BertholomusAI's TensorFold in Manrique's pull request #24).
- [Jay Leaton](https://github.com/jayleaton): the JSON writer and a multi-byte character cut at a string's end.
- [Robert Wojciechowski](https://github.com/wojo) (wojo): a prompt's preparation gives its memory back (the two-Spark
  recipe's `0023`); the SSD tier this repository's tier is built on (the two-Spark recipe's PR #30; its code comments cite
  wojo's GLM pull requests #78 and #124 on
  [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)).
- [Manrique](https://github.com/hashd1ve): DSML calls from their own tags (#25); a shared prefix (#26).
- [Blackwellboy](https://github.com/Blackwellboy): tool-only streams without framing whitespace (#32); the Engram AIO leak (#38); `TF_DS_NO_COPIES` (#34).
- [BertholomusAI](https://github.com/bertholomus): the kept-extent regression test (PR #10, reported by
  [sayyidfareed](https://github.com/sayyidfareed)); the Pillow child's restart; every rank's settings checked against rank
  0's at start (`deepseek-v41-zig-tp4` `2e07790e79`); the replay floor (`deepseek-v41-zig` `bdcfd10a2`); the tokenizer memo's
  port (`16c8d30e08`), on upstream TensorFold's memo by plotarmordev (`2d2930d8ba`), with [Ash Hart](https://github.com/ashhart)'s
  test (`ff05511460`).
- [evilpsycho42](https://github.com/evilpsycho42): a prompt ending within the window past a kept state (#35).
- [LeetJoe](https://github.com/LeetJoe): the image placeholder in a message's own text (#17), co-authored by Claude Code.
- [Coinupbtc](https://github.com/Coinupbtc): a request's own text never becomes a control token; the text before tool calls
  is the reply's content (the two-Spark recipe's `0028` and `0027`). The SSD tier's between-chunks test follows
  Coinupbtc's PR #5.
- [ezoushen](https://github.com/ezoushen): the CUDA graph cache's cap (the two-Spark recipe's PR #70, for its issue #69).
- [nuk3s](https://github.com/nuk3s): a failed CUDA graph capture runs that work eagerly (TensorFold PR #45).
- [sayyidfareed](https://github.com/sayyidfareed): the direct-I/O test's buffered descriptor (TensorFold PR #9).
- [akol1](https://github.com/akol1): Nagle off on accepted HTTP streams (upstream TensorFold PR #929).
- [Christopher A. Williamson](https://github.com/cawilliamson) (cawilliamson): GET and DELETE request bodies drained
  (MiaAI-Lab's recipe PR #37); the regression test is this repository's.
- [SvangenStudios](https://github.com/SvangenStudios): the grammar tests against the current API (MiaAI-Lab's recipe PR #30)
  and the items listed above.
- [Hector Isaza](https://github.com/hecisaza) (hecisaza): a DSML value that holds its block's closer stays in the call
  (MiaAI-Lab's recipe `0052`).
- plotarmordev: upstream TensorFold's tokenizer memo.
- This repository's tier cites upstream TensorFold's own work on the same things: the cancelled-prefill retention (#1228,
  Sapphire-Rapids), the sticky-fault readiness (#1116, #1216), the fixes for DSML slicing and the reasoning split (#1324,
  #1180) and the tier's read-ahead (#1464). Each commit says which.

## Tools and platform

- [Zig](https://ziglang.org) 0.17.0 (MIT), the engine's compiler.
- NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`, the image's base: its CUDA toolchain, cuBLAS, libibverbs
  and the Python and Pillow the engine calls for image formats other than PNG (NVIDIA's container license and the
  components' own). Its NCCL is not used by the engine.
- [PyAV](https://github.com/PyAV-Org/PyAV) 18.1.0 by the PyAV contributors (BSD-3-Clause; its wheel's FFmpeg libraries are
  LGPL-3.0-or-later), installed into the image for video input.
- [huggingface_hub](https://github.com/huggingface/huggingface_hub) (Apache-2.0), the `hf` download.
- [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 by the XGrammar contributors (Apache-2.0; its `NOTICE`: "XGrammar
  Copyright (c) 2024 by XGrammar Contributors"), with the [dlpack](https://github.com/dmlc/dlpack) headers (Apache-2.0)
  and [picojson](https://github.com/kazuho/picojson) (BSD-2-Clause, Cybozu Labs and Kazuho Oku) it bundles: compiled with
  TensorFold's `tf_grammar.cc` into the image's `libtfgrammar.so` for structured output. Their licenses are in the image's
  `/opt/tensorfold/share/doc/tensorfold/LICENSES`, picojson's in its `THIRD_PARTY_NOTICES.md`.
- For Rocinante's queue pairs, the layouts of the device interface (facts, no code linked), with the notices kept in the
  file headers: NVIDIA DOCA GPUNetIO (BSD-3-Clause, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES), DPDK's mlx5 driver
  (BSD-3-Clause, Copyright 2016 and 2021 6WIND S.A. and Mellanox Technologies, Ltd) and rdma-core's mlx5 provider
  (OpenIB.org BSD license beside the GPL-2.0, Copyright (c) 2017 Mellanox Technologies, Inc.). The license texts are in
  `LICENSES/`.
- Ideas read, no code copied: [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) (Apache-2.0), as
  BertholomusAI's `ATTRIBUTION.md` lists.

## This repository

By [rbb-dev](https://github.com/rbb-dev), developed with the help of [Claude Code](https://claude.com/claude-code). This
repository's own work is Apache-2.0:

- Rocinante, the RDMA transport that replaces NCCL, and its use of BertholomusAI's decode rings.
- The prompt path: owner sums, fewer bytes between the Sparks, passes of 4,096 rows, whole passes in turn while several
  prompts fill.
- The GPU kernels of the kernel kit's extra set, and the settings and call sites that use them.
- Faster decode: the drafter split four ways, each Spark ranking its own part of the vocabulary, and the work on the
  rings.
- Engram's placement on every Spark, the scripts that pack, check and copy the shards, and the read-ahead.
- Windows of 2M and 4M tokens, and the needle tests that check them.
- The SSD tier, by rbb-dev with Robert Wojciechowski (wojo) and Coinupbtc, built on wojo's tier for MiaAI-Lab's two-Spark
  recipe (its pull request #30); the between-chunks test follows Coinupbtc's pull request #5.
- The KV pool sized from free memory, and Turbo (prompt passes of 8,192 rows; see [`FORK.md`](FORK.md)).
- The dashboard built into the engine, its live settings and scoped admin keys.
- The removal of every spinning thread.
- The network configuration for a MikroTik CRS812, the setup (`setup.sh`, `scripts/setup-data.sh`,
  `scripts/fabric-copy.sh`, `scripts/served.sh`, the start banner), `tools/bench.py` and the offline tests.
- Serving fixes, the test suites and the kernel replay.

"DeepSeek" belongs to DeepSeek; "DGX Spark" and "GB10" belong to NVIDIA. This repository is not affiliated with or endorsed
by DeepSeek, NVIDIA, Dell, MikroTik, Anthropic, the TensorFold authors, BertholomusAI or MiaAI-Lab.
