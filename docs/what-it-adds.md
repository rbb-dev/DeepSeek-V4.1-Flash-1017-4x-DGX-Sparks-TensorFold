# What this repository adds, in detail

Each change below was measured on the four Sparks against the same engine without it. The conditions of each (a cold
real-text prompt, alone or beside four replies) are in [`FORK.md`](../FORK.md); the Rocinante figures use the
[README's Performance section](../README.md#performance). "13.7 % faster" means 13.7 % more speed than without the change.
Run-to-run noise on these Sparks is about ±4-5 %; gains smaller than that are listed as measured, and they are kept because
each one is bit-identical at no measured loss, not because a single run proves them.

**Every speed-up here is bit-identical: the engine's output is the same, byte for byte, with and without it. Nothing trades
accuracy for speed.** [`FORK.md`](../FORK.md) lists each feature with its setting and its author.

## The transport between the Sparks

- **Rocinante replaces NCCL.** The tensors that cross between the Sparks live in pinned host memory mapped into the GPU
  (zero-copy memory), registered with the network card. The producing kernel writes its output once; each transfer is a
  GPU-triggered send that the card starts the moment the data is complete, writing straight into the buffer the receiving
  kernel reads; a completion flag behind the data releases that kernel. No GPU multiprocessor and no CPU thread waits for
  the network. Two passes of a prompt are in flight at once, one computing while the other's data is on the wire. Prompts
  **8 to 29 % faster** than with NCCL from 4K to 256K tokens, replies 4 % slower to 2 % faster, outputs byte-identical
  ([`ROCINANTE.md`](../ROCINANTE.md)). The decode exchanges run on BertholomusAI's RDMA rings, which Rocinante drives the
  same way.
- **Less traffic between the Sparks, with the same bits.** Where the four Sparks split a prompt's rows, each Spark receives
  only the sums for its own rows: a gather of 13.1 MB instead of 62.9 MB, which made prompts **21.2 % faster**. Where the
  next matrix multiply converts its input to FP8 anyway, the sender converts first and sends the 8-bit codes instead of
  16-bit values. The FP8 conversion works on groups of 32 values, so converting each Spark's part gives exactly the codes
  and scales of converting the whole row, and the receiving multiply reads the very numbers it would have made. Four
  exchanges that carry only a Spark's own rows or FP8 codes added **13.3 %** to the speed on top, and further cuts 2.8 %, 2.3 % and
  1.7 %. No rounding is added anywhere.

## Reading a prompt

- **Block-scaled FP8 matrix multiplies** in which the hardware applies each block's scale, accumulating exactly as the plain multiply does,
  so every result is the same number (2.2 % faster).
- **Prompt passes of 4,096 rows** instead of 2,048 (**13.7 % faster**).
- **The sparse attention and the indexer read their FP4 data unpacked in hardware** (1.1 % and 1.6 % faster).
- Split and grouped matrix-multiply tiles, and a top-k selection in one launch.
- **A kernel kit with an extra set**: [this repository's fork of BertholomusAI's kit](https://github.com/rbb-dev/deepseek-v4.1-tensorfold-tp4-4xgb10)
  (branch `extra-set`) keeps BertholomusAI's kit unchanged (apart from its own README) and adds the ready-built kernel
  variants these paths need: the
  hardware-unpacked attention and indexer, the four-way drafter, Rocinante's in-place readers, and the tuned prompt and
  decode paths.

## Writing replies

- **The drafter split four ways** like the rest of the model (2.9 % faster for one request, 4.0 % faster for 16).
- **The decode exchanges collect from every Spark at once**, and each Spark ranks next-token candidates from its own quarter
  of the vocabulary.
- **The drafter's copy of the output head in MXFP4** (1.9 % faster on prose). The full model still checks every drafted token.

## Engram

- **MiaAI-Lab's packed Engram rows** (patch `0003` of MiaAI-Lab's recipe: each row's values and scales side by side, so a
  row is one disk read), run with the whole tables on every Spark, which is one of that patch's options. Against the same engine without it, a
  prompt alone reads 1.9 % faster, and a prompt read while four replies are written **44.8 % faster**.
- **A read-ahead that follows each prompt pass** a layer at a time (`TF_DS_ENGRAM_AHEAD_PASS`): the first prompt after a
  start reads at 5,457 tokens a second against 4,330 without it (26 % faster).
- **The row cache faulted in at start** (`TF_DS_ENGRAM_PREFAULT`), so the first prompts do not pay for touching its pages.
- The scripts that pack the shards, check them, and copy each Spark its share over the fabric.

## Memory and long context

- **A KV pool sized from free memory**: every Spark keeps 12 GiB free and the pool takes the rest, about 18.3M positions at
  the 1M window, resized while serving. Each conversation takes only its own length from it.
- **Windows of 2M and 4M tokens** past the trained 1M, opt-in at start, with the engine fixed to serve them
  ([long context](long-context.md)).
- **The SSD tier**: kept conversation states on every Spark's NVMe (each keeps its own share), restored instead of read again,
  also after a restart: a 4M-token prompt's state in 664 ms.
- **Turbo**: 8,192-row prompt passes paid for from the KV pool, on a button: 11 % faster on a 152K-token real-text prompt,
  for 43 % of the KV pool at the 1M window ([README](../README.md#turbo)).

## Heat

- **No CPU core left spinning.** Every wait the engine has, for the GPU, the network or the disk, sleeps in the CUDA driver
  or is handed to the network card, at the same speed ([heat and cores](heat-and-cores.md)).

## The network and the setup

- **A RoCE network with priority flow control (PFC) on one switch** ([`NETWORK.md`](../NETWORK.md)): PFC is on the RoCE
  traffic class on the switch and on every card. 0 drops and 0 resends under a three-into-one overload (three Sparks
  sending to one at once) and under real load; without PFC, the same overload dropped 1.45 million packets in 20 s.
- **A setup from one config file**: [`./setup.sh`](../setup.sh) checks the config, ssh and sudo, and sets up the network's
  two rails on every Spark. On the first Spark it builds the image, checks the downloaded checkpoint, fetches the kernel
  kit and cuts the weight files and Engram's shards; it sends each of the others its share over both rails by RDMA (every
  chunk checked by CRC-32C on both ends), starts the engine twice (a first start right after the copy comes up slower),
  and prints the addresses and keys. Measured on four Sparks from a fresh config, with the checkpoint already downloaded:
  about 50 minutes.

## Serving

Some of these fixes were written by others; [`FORK.md`](../FORK.md) lists each under its author's name.

- A request without `max_tokens` is served, and one asking for more than fits gets a 400 naming the limit.
- Tool calls are read from their own tags, and the thinking ends at the model's own close token.
- A CUDA graph capture that fails on one Spark falls back on all four, and the requests carry on.
- A window the indexer's kernels cannot serve is refused at start, with the nearest one that works.
- Each Spark reads its own weight file from its own disk, past the page cache: the weights load in 27 to 47 s.
- The dashboard and its live settings ([README](../README.md#the-built-in-dashboard)).

## Tests

- **A test suite and a kernel replay** check every change's output byte for byte before it ships. The replay runs each
  kernel switch on the real kernels with random inputs against the kernels it replaces: every launch must give the same
  bytes, and a control with one input changed must differ, so the check is shown to be able to fail.
