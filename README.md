<h1 align="center">DeepSeek-V4.1-Flash-1017 on four DGX Sparks</h1>

<p align="center"><b>A fork of BertholomusAI's DeepSeek-V4.1 branch of TensorFold, with its own engine work added</b></p>

This repository serves DeepSeek's **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**
from four NVIDIA DGX Sparks over their ConnectX-7 ports, through an OpenAI-compatible API: text, image and video input,
tool calls, structured output, thinking on or off, and speculative decoding from the checkpoint's own draft layers. The
weights are DeepSeek's as published (FP8 attention and shared experts, MXFP4 routed experts), with one derived tensor: the
drafter's copy of the output head is converted to MXFP4. The full model checks every drafted token, so a greedy reply is
the same with or without drafting.

The engine is [TensorFold](https://github.com/ashhart/TensorFold), by Ash Hart and the TensorFold contributors. Support for
DeepSeek-V4.1 on TensorFold (the four-node split, Engram, the drafter, the vision tower, the decode RDMA rings and the kernel
kit) is [BertholomusAI's](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4). The four-Spark setup began
as [MiaAI-Lab's recipe](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold). This repository is a
fork of BertholomusAI's branch with its own engine work added on top; [Built on](#built-on) lists everyone. BertholomusAI's
README is kept unchanged as [`README-BertholomusAI.md`](README-BertholomusAI.md).

## At a glance

Measured on four Dell Pro Max GB10 on one MikroTik CRS812 switch, with the GPU clock limited to 2,200 MHz. Every prompt is
cold unless it says "from the SSD" ([Performance](#performance) has the conditions; [Words used here](#words-used-here)
explains the terms).

| | |
|---|---|
| **Reading a prompt** | **7,716 tokens a second** for a 64K prompt, **6,186** for a 1M prompt (**6,500** with Turbo) |
| **Writing replies** | **65.3 tokens a second** for one request, **227.3** in total for 16 (an essay, greedy); code **115.8** for one, **249.7** in total for 16 |
| **Long context** | windows of 1M, 2M and 4M tokens; needle tests with prompts up to 3.99M tokens: every hidden code found |
| **A 4M prompt asked again** | **7.2 s** after an engine restart, from the SSD tier (21 minutes to read it cold) |
| **The transport** | **Rocinante**, this repository's RDMA transport, in place of NCCL: every transfer is a GPU-triggered send, with no CPU thread in the path. Prompts 8 to 29 % faster than with NCCL (4K to 256K tokens), replies within 4 % slower to 2 % faster |
| **The CPU** | no engine thread spins; at idle none reaches 2 % of a core |
| **The network** | one **MikroTik CRS812**, RoCE with priority flow control: 0 drops and 0 resends under overload. Its configuration from the factory state, each Spark's side and the steps are in [`NETWORK.md`](NETWORK.md) and [`network/`](network/) |
| **Turbo** | one button on the dashboard: long prompts 3-11 % faster, paid for with 43 % of the KV pool |
| **The dashboard** | built into the engine at `/admin/ui/`: the four Sparks, the requests, the KV pool, the current speed, Turbo, and every setting, changeable live |

## Contents

- [Getting started](#getting-started)
- [What this repository adds](#what-this-repository-adds)
- [Performance](#performance)
- [Rocinante](#rocinante)
- [Long context: 1M, 2M and 4M tokens](#long-context-1m-2m-and-4m-tokens)
- [Turbo](#turbo)
- [The built-in dashboard](#the-built-in-dashboard)
- [Heat and cores](#heat-and-cores)
- [Settings](#settings)
- [Using it](#using-it)
- [Built on](#built-on)
- [Words used here](#words-used-here)
- [How this repository relates to its upstreams](#how-this-repository-relates-to-its-upstreams)
- [Repository layout](#repository-layout)
- [License](#license)

## Getting started

**You need**

- Four DGX Sparks on one RoCE switch, one ConnectX-7 port each (prepare the switch as [`NETWORK.md`](NETWORK.md)
  describes).
- The same user on all four, with key-based ssh from the first Spark to the other three (`ssh-copy-id user@spark2` for
  each), and sudo: passwordless sudo, or the setup installs a rule for only the start's two root commands on a Spark
  without it (`./setup.sh check` shows which); Docker with the NVIDIA runtime (DGX
  OS has both); the Hugging Face CLI (`hf`) on the first Spark.
- Disk: the first Spark holds the checkpoint, the weight files and Engram's packed shards, about 1 TB, plus `SPILL_GIB`
  for its SSD store; each of the others needs about 490 GB (its 80 GB weight file, 203 GB of Engram's tables and 203 GB of their packed
  copy), plus `SPILL_GIB`. Make the Spark with the most room the first.
- About 100 GB of free memory on each Spark.

**Set up** (on the first Spark):

```bash
git clone <this repository> dsv41 && cd dsv41
cp scripts/local.sh.example scripts/local.sh     # then set WORKER1, WORKER2, WORKER3: the three other Sparks,
                                                 # and HF_CACHE if the model is already downloaded elsewhere
./setup.sh
```

`./setup.sh` runs five steps and can be run again at any time: each step checks first and changes only what differs.
The long ones print their progress (per cent done and the time left).

1. **config**: reads `scripts/local.sh` and names what you still have to change (the three workers at least).
2. **access**: checks key-based ssh to the three others (printing the `ssh-copy-id` command when it fails) and sudo
   on all four. A Spark with passwordless sudo keeps it and gets nothing. On any other, the setup asks that Spark's
   password once (it goes to that Spark's sudo over ssh, never to a file or a command line) and installs a sudo rule for
   only the start's two root commands, the GPU clock limit and the page-cache drop: the rule lets you run
   `/usr/local/sbin/dsv41-root` and nothing else (`/etc/sudoers.d/90-dsv41-start`, checked with `visudo` before it is
   installed). The network step uses the same password for the rails. `./setup.sh remove` takes the rule away (and the
   full rule an earlier version of the setup wrote to `/etc/sudoers.d/90-dsv41`, only if it holds exactly that line) and
   says what else the setup left (the rails, the flow control, the data, the image).
3. **network**: sets up each Spark's two fabric rails (static addresses, MTU 9000: `/etc/netplan/51-cx7-roce-static.yaml`)
   and its RoCE flow control (`/etc/systemd/system/roce-pfc.service`), kept across reboots. Then it checks each
   rail: that it is up, its MTU, its PCIe link at x4 and PFC on. Last, a full-size ping that may not be split runs
   between every pair of Sparks, on both rails. Before it writes, it keeps a copy of every file in `/etc/netplan`
   (`/var/backups/dsv41-setup/`). An earlier fabric file that configures only the ConnectX-7 ports, such as the one
   NVIDIA's two-Spark playbook writes, moves into that copy; any other netplan file that also configures a rail stops
   the step, because that file is yours to change.
4. **data**: on the first Spark, builds the engine image from this checkout, downloads the checkpoint, fetches and checks
   the kernel kit, and makes the four weight files (one per Spark, ~80 GB each) and Engram's packed shards. Then each of
   the others gets its share over both rails by RDMA ([`tools/fabric-xfer`](tools/fabric-xfer/)): each file is read
   once and written to every Spark that needs it, every chunk checked by CRC-32C on both ends. Where it cannot be
   built or finds no rails, a TCP copy does the same, every file checked by SHA-256.
5. **start**: makes an API key and an admin key (in `~/.config/dsv41`, or `KEY_DIR`) unless `API_KEY_FILE` and
   `ADMIN_KEY_FILE` already name yours, adds them to `scripts/local.sh`, starts the engine, starts it once more
   (`./start.sh restart`) and prints the API's address, the dashboard's address and both keys. The second start is there
   because an engine started once, right after the data copy, was measured answering every prompt about 0.9 s late,
   and every restart after it at full speed.

`./setup.sh network` and `./setup.sh data` run one step, with the config and access checks before it; `./setup.sh check`
runs only those two checks. After the setup the engine is running.

**Start and stop:**

```bash
./start.sh          # starts the four ranks (after a stop; the setup starts them the first time)
./stop.sh
python3 tools/validate.py http://localhost:8000 --model deepseek-v4.1-flash-1017 --key-file ~/.config/dsv41/api-keys
                                     # text, thinking, tools, an image, a long prompt
```

A start drops every Spark's page cache and compacts its memory, limits the GPU clocks to `GPU_CLOCK` (until a reboot),
starts the four ranks and runs a smoke test; the weights load in 27 to 47 s. `./start.sh restart` restarts with changed
settings; it is also what a start tells you to run when a Spark's memory was still in small pieces. `./stop.sh` waits
for running requests to finish (up to `DRAIN_TIMEOUT`) before it stops all four.

## What this repository adds

On top of BertholomusAI's TensorFold branch, this repository adds:

- **Rocinante**, its own RDMA transport, in place of NCCL. Every prompt exchange runs on its queue pairs, and it drives
  BertholomusAI's decode rings the same way. Every transfer is triggered by the GPU and needs no CPU thread.
- **Less traffic between the Sparks, with the same bits**: each Spark receives only the sums for its own rows, and FP8
  codes are sent where the receiving Spark would compute exactly the same codes.
- **Faster prompt kernels, with the same bits**: block-scaled FP8 matrix multiplies with exact accumulation, prompt
  passes of 4,096 rows instead of 2,048 (13.7 % faster), and FP4 data unpacked in hardware. They come from an extra kernel set in
  [this repository's fork of the kernel kit](https://github.com/rbb-dev/deepseek-v4.1-tensorfold-tp4-4xgb10).
- **Faster decode**: the drafter split four ways like the rest of the model, each Spark ranking next-token candidates from
  its own quarter of the vocabulary, and the decode rings writing straight into write-combined send slots.
- **No busy-waiting**: every wait sleeps in the CUDA driver or is handed to the network card ([Heat and cores](#heat-and-cores)).
- **Engram on every Spark**: MiaAI-Lab's packed Engram rows (patch `0003` of MiaAI-Lab's recipe), run with the whole
  tables on every Spark, which is one of its options (44.8 % faster for a prompt read while four replies are written). This repository adds the scripts that
  pack, check and copy the shards, and a read-ahead that follows each prompt pass a layer at a time.
- **Windows of 2M and 4M tokens** past the trained 1M, with the engine fixed to serve them, and needle tests to check them.
- **An SSD tier**: each Spark saves its own copy of a kept conversation's KV state on its own NVMe, restored after a restart.
- **A KV pool sized from free memory**, resized while serving, and **Turbo**.
- **A dashboard built into the engine**, with live settings on all four Sparks and scoped admin keys.
- **A RoCE network with priority flow control** on one MikroTik CRS812, with its configuration included.
- **A one-command setup** of four Sparks from one config file.
- **A test suite and a kernel replay** that check every change's output byte for byte before it ships.
- **Serving fixes**, some written by others and listed under their names in [`FORK.md`](FORK.md): a request without
  `max_tokens` is served, one asking for more than fits gets a 400 naming the limit, tool calls are read from their own
  tags, and a CUDA graph capture that fails on one Spark falls back on all four.

Every speed-up is bit-identical: the engine's output is the same, byte for byte, with and without it.
[`docs/what-it-adds.md`](docs/what-it-adds.md) gives each change with what it measured against its own control.

## Performance

Measured with [`tools/recipe_bench.py`](tools/recipe_bench.py), the benchmark of MiaAI-Lab's two-Spark DeepSeek recipe,
unchanged, so the numbers compare directly with others from that tool. The conditions, for every table in this README:

- four Dell Pro Max GB10 (DGX OS 7.6, kernel 7.0, driver 580.178.04), one ConnectX-7 200G port each on one MikroTik
  CRS812 (RoCE v2, priority flow control on the RoCE class);
- the GPU clock limited to 2,200 MHz (`GPU_CLOCK`, set at every start);
- every setting in [`scripts/served.sh`](scripts/served.sh), this repository's kernel kit, the drafter on;
- the prompt cache and the SSD tier off, so every prompt is cold;
- each cell the median of three runs. Run-to-run noise on these Sparks is about ±4-5 %;
- the four Sparks set up from scratch with this repository's `./setup.sh`, the kernel kit fetched from its published
  default, stock weights.

Every request of these runs is in [`evidence/native-8c6e8e88d875/`](evidence/native-8c6e8e88d875/), named by the image
tag of the build it ran on, with the kit, the model revision and the software it ran on; `tools/readme_tables.py` rebuilds the
three tables from those files. The real-text figures beside them (the ~147K and ~152K-token prompts) come from separate
runs and are not in it.

**Reading a prompt** (one request; tokens a second = prompt tokens / time to the first token; the sizes are the ones the
benchmark asks for, and the prompts it builds come out a few per cent shorter; Turbo is explained [below](#turbo))

| Prompt | Tokens a second | First token | With Turbo | First token | Turbo speed change |
|---:|---:|---:|---:|---:|---:|
| 4K | 5,341 | 739 ms | 5,296 | 744 ms | -0.9 % |
| 8K | 6,021 | 1.31 s | 5,895 | 1.33 s | -2.1 % |
| 16K | 6,884 | 2.28 s | 6,502 | 2.41 s | -5.5 % |
| 32K | 7,455 | 4.21 s | 7,280 | 4.32 s | -2.3 % |
| 64K | **7,716** | 8.14 s | 7,735 | 8.12 s | +0.2 % |
| 128K | 7,684 | 16.3 s | 7,903 | 15.9 s | +2.8 % |
| 256K | 7,428 | 33.8 s | 7,766 | 32.3 s | +4.6 % |
| 512K | 6,933 | 72.4 s | 7,268 | 69.1 s | +4.8 % |
| 1M | 6,186 | 154.9 s | 6,500 | 147.4 s | +5.1 % |

The benchmark's prompts are generated text, which reads faster than real text: a cold prompt of real text of about
147K tokens reads at 6,773 tokens a second (the mean of three such prompts; a different text from the ~152K-token
prompt of the Turbo table). The 8K row varies: over eight separate runs without Turbo its
first token took 1.29 to 2.18 s (an 8K prompt is two passes; with Turbo it is one pass, about 1.33 s every time); the
run in the table landed at the fast end. Beyond 64K the speed falls with the length
([Long context](#long-context-1m-2m-and-4m-tokens) explains why).

**Writing replies** (greedy, thinking off, the benchmark's essay prompt, 256 tokens; total = the requests' tokens after
their first, over the time from the first token of any request to the last token of any; each = one request's own rate;
first token = the mean of the requests' first tokens). Each request's prompt carries a random tag, so no cache serves
it and every run writes different essays; one request's speed follows how well the drafter guesses its essay, and its
three repetitions here ran at 64.0, 65.3 and 70.0 tokens a second.

| Requests at once | Total | Each | First token |
|---:|---:|---:|---:|
| 1 | **65.3** | 65.3 | 131 ms |
| 2 | 98.7 | 50.7 | 213 ms |
| 4 | 137.1 | 36.3 | 384 ms |
| 8 | 181.6 | 24.6 | 731 ms |
| 16 | **227.3** | 16.0 | 1.53 s |

**Writing code** ([`tools/bench.py`](tools/bench.py): code prompts, greedy, thinking off, replies up to 1,024 tokens;
total = all the requests' tokens over the time from the first request sent to the last token, reading the prompts
included; each = one request's own rate, their mean; first token = their mean). Code is more predictable than prose, so the drafter's guesses are accepted more often and each
step yields more tokens. This tool's prompts and its total differ from the benchmark's, so compare the two tables only
roughly.

| Requests at once | Total | Each | First token |
|---:|---:|---:|---:|
| 1 | **113.3** | 115.8 | 135 ms |
| 2 | 138.5 | 83.2 | 204 ms |
| 4 | 174.3 | 54.4 | 363 ms |
| 8 | 215.6 | 34.0 | 759 ms |
| 16 | **249.7** | 19.8 | 1.54 s |

The commands (with keys on, `recipe_bench.py` reads `API_KEY` or `API_KEY_FILE`, and `bench.py` takes `--key-file FILE`):

```bash
PROMPT_CACHE_GIB=0 SPILL_GIB=0 ./start.sh restart        # every prompt cold
python3 tools/recipe_bench.py --url http://localhost:8000 --model deepseek-v4.1-flash-1017 --suite prefill,decode,multi \
  --prefill-sizes 4096,8192,16384,32768,65536,131072,262144,524288,1000000 --streams 1,2,4,8,16 --reps 3
python3 tools/bench.py http://localhost:8000 decode --kind code --conc 1,2,4,8,16 --reps 3
PROMPT_CACHE_GIB=0 SPILL_GIB=0 TF_DS_PASS_ROWS=8192 TF_DS_GEMM_HALVES=1 ./start.sh restart     # the Turbo columns
python3 tools/recipe_bench.py --url http://localhost:8000 --model deepseek-v4.1-flash-1017 --suite prefill \
  --prefill-sizes 4096,8192,16384,32768,65536,131072,262144,524288,1000000 --reps 3
```

## Rocinante

Rocinante ("Roci", after the ship in *The Expanse*) is this repository's network transport. It carries every exchange
between the Sparks, for reading prompts and for writing replies: prompt exchanges on its own queue pairs, and decode
exchanges on BertholomusAI's RDMA rings, which it drives the same way. NCCL is not used.

NCCL had three costs on the Sparks. GPUDirect RDMA is not supported on the DGX Spark
([NVIDIA's DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/cuda.html)), so NCCL
staged every prompt-sized exchange through extra copies. Its kernels held GPU multiprocessors while they waited for the
network. And its progress thread ran at 100 % of a fast CPU core. Rocinante instead:

- **keeps the exchanged tensors in pinned host memory mapped into the GPU** (zero-copy memory, the kind of buffer NVIDIA's
  guide recommends for RDMA on the Spark), registered with both ConnectX-7 rails. The producing kernel writes its output there
  once, and the network card sends it from there, with no staging copy;
- **makes every transfer a GPU-triggered send**: each transfer is handed to the network card in advance, and the card sends
  it the moment the GPU finishes writing the data, with no CPU thread involved;
- **ends each transfer with a completion flag** that the receiving GPU's next kernel waits on, without holding a thread or
  any of the GPU's multiprocessors;
- **keeps two passes of a prompt in flight**, one computing while the other's data is on the wire.

Measured on the same four Sparks with the same benchmark, against a build of the engine that used NCCL: prompts **8 to
29 % faster** from 4K to 256K tokens, replies 4 % slower to 2 % faster, outputs byte-identical. The same prompt work used 5 % less GPU energy
(28.6 J against 30.2 J per 1,000 prompt tokens: the GPUs' power as `nvidia-smi` reports it, sampled every 200 ms on all
four Sparks, one run of three prompts each, so within reach of the run-to-run noise; not the power drawn at the wall). The four GPUs drew
more power while doing it (184.4 W against 164.7 W), because the work finished sooner. [`ROCINANTE.md`](ROCINANTE.md) explains how it works and lists its settings.

## Long context: 1M, 2M and 4M tokens

DeepSeek trained V4.1-Flash up to 1,048,576 tokens, stretching its positions with YaRN (factor 16 over the 65,536 the base
model saw). This repository can stretch them further at start:

| Window | Start with | KV pool | Longest request |
|---|---|---:|---:|
| 1M (the checkpoint's own) | `./start.sh` | about 18.3M positions | 1,048,526 tokens |
| 2M | `CONTEXT=2097152 TF_DS_YARN_FACTOR=32 ./start.sh restart` | about 16.8M | 2,097,102 |
| 4M | `CONTEXT=4128768 TF_DS_YARN_FACTOR=64 ./start.sh restart` | about 13.5M | 4,128,718 |

The longer windows need the kernel kit's rotary-position tables for their factor. A conversation takes pool space for its
own length, not the window's. By the pool's arithmetic, about 17 conversations of the full 1M length fit at once, 8 of
2M, or 3 of 4M; several long conversations at once have not been tested. The 4M window is 4,128,768 tokens
(63 x 65,536), not 4,194,304: when writing a reply, the indexer launches one GPU block per 64 earlier tokens along a grid
axis limited to 65,535 blocks, and its ready-built kernels need that block count to be a multiple of 16. At start the
engine refuses a window that breaks either rule and names the nearest one that works.

**What was tested** (greedy, thinking off, one request at a time): codes tied to city names were hidden through a filler
text, and the question at the end asked for them. Every code came back right at every length and depth tried:

| Window | Prompt tokens | Test | Found |
|---|---:|---|---:|
| 2M | 1,995,164 | ten codes in one prompt | 10 / 10 |
| 4M | 3,990,287 | twenty codes planted, ten asked, cold (21 minutes) | 10 / 10 |
| 4M | 3,990,312 | the same prompt, all twenty asked, after an engine restart, from the SSD tier (7.2 s) | 20 / 20 |

For the last row, the SSD tier had written the first request's state to disk within 5 s of its answer. After the restart,
it restored the state of the first 3,989,504 tokens in 664 ms; the engine read only the 808 new tokens of the different
question, and the model answered all twenty.

A long prompt slows as it goes: the first half-million tokens read at about 7,300 tokens a second, and the last stretch of
a 4M prompt at about 2,000. Each token's attention uses only 512 earlier tokens, but, as we read the model's design,
DeepSeek's indexer scores every earlier token to choose them, so that work grows with the position. Writing a reply slows the same way: about 15.5 steps
a second 4M tokens deep, against about 35 at short lengths.

These tests show that the model finds and quotes facts anywhere in 4M tokens. Keeping near-identical passages in order is
harder: on Google's MRCR v2 (two passages, 80 questions) the model scores about the noise rate with thinking off at 1M
(the served window) and below it at 4M (the 4M window); with thinking on, the same 20 questions at 4M went from 42.7 % to
80.3 %. On LongBench v2's
108 long items (THUDM's set: real documents, multiple choice; the authors' protocol, thinking off, the 1M window) it answers 50.9 %
right, where a random letter scores 25 % ([`docs/long-context.md`](docs/long-context.md#does-it-reason-over-it)). Factors 32 and 64 are not the checkpoint's own position encoding at any length, so both longer windows are off
by default. [`docs/long-context.md`](docs/long-context.md) has every needle test, the speed curves and how to run the
tests yourself.

## Turbo

Turbo reads a prompt in passes of 8,192 rows instead of 4,096, with the dense matrix multiplies launched as 4,096-row
halves (`TF_DS_PASS_ROWS=8192 TF_DS_GEMM_HALVES=1`; the dashboard's button sets both on all four Sparks). Its larger
buffers come out of the KV pool, never out of the memory the Sparks keep free:

| | Turbo off | Turbo on | |
|---|---:|---:|---:|
| A cold ~152K-token prompt of real text | 6,729 tokens a second | **7,468** | **+11.0 %** |
| KV pool at the 1M window | 18,350,080 positions | 10,420,224 | **-43.2 %** |
| Free memory after the prompt (first Spark / the others) | 13.1 / 14.9 GiB | 13.0 / 15.5 GiB | unchanged |
| Replies | | byte-identical | |

On the benchmark's generated prompts (the Turbo columns of the [prompt table](#performance)), Turbo reads prompts of 128K
tokens and more 3 to 5 % faster, and prompts of 4K to 32K 1 to 6 % slower; on the real-text prompt above it gains 11 %.
Past about 1.25M tokens both modes read at the same speed, because the indexer's work grows with the position, not with
the pass; at the 2M window Turbo costs 46 % of the pool for no gain beyond the noise. Use it for long prompts under 1M when
you can spare the pool. Replies are byte-identical with and without it.

## The built-in dashboard

The engine serves a dashboard at **`/admin/ui/`** on the API's own port on the first Spark, and a JSON API under
`/admin/api/v1`. It shows:

- **the four Sparks**: CPU and GPU temperatures, load, memory, the NVMe's temperature and wear, each fabric rail's traffic,
  and the GPU clock limit;
- **the requests**: each one in flight, its stage, its tokens and its speed;
- **the current speed**: prompt and reply tokens a second over the last few seconds;
- **the KV pool and the SSD tier**: positions and GiB in use, kept states, restores;
- **Turbo**: its button, with its measured gain and cost;
- **settings**: every setting the engine reads, with its default, its range and whether it changes live. An admin key sets
  one on all four Sparks, reverts it, or keeps it across restarts (`ENGINE_CFG`).

### Opening the dashboard

The dashboard is off until you give the engine an admin key file; without it, every `/admin` path answers like an unknown
one. You make the keys yourself:

1. **Make a key file** on the first Spark, readable only by you. Each line is `label: key`; a key must be at least 32
   bytes, and must not also be in the API key file.

   ```bash
   mkdir -p ~/.config/dsv41
   (umask 077; printf 'admin: %s\n' "$(openssl rand -hex 32)" > ~/.config/dsv41/admin-keys)
   ```

   The label sets what the key may do: `admin` or `admin-...` may do everything; `read-...`, `agent-...` and any other
   label may only read. Add a line per person or tool; the engine rereads the file when it changes.
2. **Point the engine at it** in `scripts/local.sh`, and restart:

   ```bash
   ADMIN_KEY_FILE=$HOME/.config/dsv41/admin-keys
   ENGINE_CFG=$HOME/.config/dsv41/engine-cfg/engine.cfg     # optional: where settings kept across restarts live
   ```

   `./start.sh restart`. The start refuses a key file that others can read.
3. **Open `http://<first Spark>:8000/admin/ui/`** and paste the key (`cat ~/.config/dsv41/admin-keys` shows it) into the
   **Admin key** box. The page keeps the key in that browser tab only: a refresh keeps you signed in, and closing the tab,
   signing out, or 30 minutes without use forgets it.

The page and its API use plain HTTP on the API's port. Outside a network you trust, reach it through an ssh tunnel
(`ssh -L 8000:localhost:8000 user@<first Spark>`, then `http://localhost:8000/admin/ui/`) or put a TLS proxy in front.
`ADMIN_POLICY_FILE` can give agent keys their own limits (priority, thinking budget, wait).

## Heat and cores

A thread that waits by checking a value in a loop ("spinning") keeps its core at 100 % while doing nothing. On a GB10 the
CPU and the GPU are one chip, with one power budget and one cooler, so every watt a spinning core burns is heat the cooler
must also carry for the GPU. No engine thread spins, at the same speed: the engine's waits for the GPU sleep in the CUDA
driver, network transfers are triggered by the GPU and handled by the network card, and Engram's disk reader sleeps
between batches. The cores that would otherwise spin run 3.8 to 14 °C cooler, depending on the thread, and the chip's hottest zone
3 to 7 °C cooler, under 16 requests at once. [`docs/heat-and-cores.md`](docs/heat-and-cores.md) has each change, what it
measured, and which core runs what.

## Settings

[`scripts/served.sh`](scripts/served.sh) holds the settings this repository runs with, each described in a comment.
The first line of `scripts/local.sh` (copied from `scripts/local.sh.example`) loads it, so your own lines below it win, and
the environment wins over both. [`scripts/config.sh`](scripts/config.sh) documents every setting; its own defaults differ in places (port 8888),
and every example here assumes `served.sh`. The main ones:

| Setting | In `served.sh` (or the example) | |
| --- | --- | --- |
| `WORKER1`, `WORKER2`, `WORKER3` | (yours) | the three other Sparks, as ssh targets (`user@address`) |
| `RAIL_A_ADDRS`, `RAIL_B_ADDRS` | `"10.255.0.1 ... 10.255.0.4"`, `"10.255.1.1 ... 10.255.1.4"` | the rails' four addresses each, the first Spark's first |
| `RAIL_A_IF`, `RAIL_B_IF` | `enp1s0f0np0`, `enP2p1s0f0np0` | the two rails' interfaces (`ip -br link` lists them) |
| `HF_CACHE` | `~/.cache/huggingface` | the Hugging Face cache on the first Spark that holds (or will hold) the model, beside the weight files and the kit |
| `CONTEXT` | 1048576 | a request's window in tokens (a request may use `CONTEXT` less 50); 2097152 or 4128768 with `TF_DS_YARN_FACTOR` 32 or 64 |
| `TF_DS_KV_VMM`, `TF_DS_KV_FREE_TARGET_GIB` | 1, 12 | the KV pool takes the memory above 12 GiB free on each Spark and is resized while serving |
| `PARALLEL` | 16 | requests at once |
| `PORT` / `SERVED_NAME` | 8000 / `deepseek-v4.1-flash-1017` | the API's port and the model name clients ask for |
| `ABLIT` | 0 | `0`: DeepSeek's weights as published. `1` adds [drowzeys's Keys overlay](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated) to the attention output of layers 10-35, which weakens the model's refusals; you are then responsible for any guardrails. Its ~1.1 GB download needs a Hugging Face account that accepted the overlay's terms (`hf auth login`) |
| `GPU_CLOCK` | 2200 | the GPU clock limit in MHz, set at every start; empty leaves the clocks alone |
| `PROMPT_CACHE_GIB` / `SPILL_GIB` | 2 / 256 | kept conversation states in memory / the SSD tier's limit on each Spark, in GiB (0: off) |
| `API_KEY_FILE` / `METRICS_OPEN` | (none) / 1 | [API keys](#api-keys) |
| `ADMIN_KEY_FILE`, `ADMIN_POLICY_FILE`, `ENGINE_CFG` | (none) | [the dashboard](#opening-the-dashboard)'s keys and policy, and the settings it keeps |
| `TF_DS_PASS_ROWS`, `TF_DS_GEMM_HALVES` | 4096, unset | [Turbo](#turbo) sets 8192 and 1 |
| `DRAIN_TIMEOUT` | 120 | seconds `./stop.sh` lets running requests finish |

**The kernel kit.** The engine loads its GPU kernels ready-built from a kernel kit. The default (`KIT_REPO` / `KIT_REF` in
`scripts/config.sh`) is this repository's fork of BertholomusAI's kit,
[rbb-dev/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/rbb-dev/deepseek-v4.1-tensorfold-tp4-4xgb10) (branch
`extra-set`): BertholomusAI's kit, plus an extra kernel set and its own README. `./setup.sh` fetches it at a pinned commit and checks
every file against its manifest. A kit's rotary-position (RoPE) tables are cut for one YaRN factor, so the default kit
follows `TF_DS_YARN_FACTOR`: three commits of the fork that differ only in those tables, for the 1M, 2M and 4M windows.
With a kit that has no extra set, the settings that need it say at start that they are off.

## Using it

```bash
KEY=$(sed -n '1s/^[^:]*: *//p' ~/.config/dsv41/api-keys)     # the API key the setup made
curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model": "deepseek-v4.1-flash-1017", "messages": [{"role": "user", "content": "Hello!"}], "chat_template_kwargs": {"enable_thinking": false}}'
```

- **Thinking** is on unless a request sends `enable_thinking: false`; the reasoning comes back in
  `message.reasoning_content`, the answer in `message.content`. `reasoning_effort` takes `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`.
- **Images and video** go in as base64 `data:` URLs in `image_url` and `video_url` parts. A video is sampled into frames
  (2 a second, at most 32, each at most 768 pixels a side), each costing prompt tokens like an image.
- **Structured output**: `response_format` with a JSON schema or `json_object`, and `tool_choice: "required"` or a named
  function; every token is drawn among those the grammar allows. With thinking on, the grammar starts after the reasoning.
- **Log probabilities** are not served: a request that asks for them (`logprobs: true`) gets a 400, "logprobs are not
  supported by this model or backend".
- **Kept conversations**: a conversation that comes back with the same beginning resumes where its kept state ends, from
  memory or from the SSD tier (also after a restart), instead of reading its whole prompt again.
- **`/health` and `/metrics`**: the engine's state, the prompt being read and its current speed, kept tokens, the SSD tier.

### API keys

`./setup.sh` makes an API key and names it in `scripts/local.sh`; without `API_KEY_FILE` the API has no keys. It listens
on every address of the first Spark (`HOST=0.0.0.0`). `API_KEY_FILE` names a file of
`label: key` lines that only its owner may read (`chmod 600`: the start refuses it otherwise; the engine rereads it when it
changes). Every route then needs a key except `/health` and, by default, `/metrics` (`METRICS_OPEN=0` closes it too), as
`Authorization: Bearer <key>` or `x-api-key: <key>`; `/health` answers `{"status": "ok"}` without one and in full with one.
A key of this file opens nothing under `/admin`; the dashboard has [its own keys](#opening-the-dashboard).

## Built on

| What | By | License |
|---|---|---|
| The engine: [TensorFold](https://github.com/ashhart/TensorFold), an inference engine in Zig and CUDA: the server, the scheduler, the tools | [Ash Hart](https://github.com/ashhart) and the TensorFold contributors | Apache-2.0 |
| DeepSeek-V4.1 on TensorFold: the four-node split, Engram, the drafter, the vision tower, the decode RDMA rings, the weight-file and token-map tools, the kernel kit ([bertholomus/TensorFold](https://github.com/bertholomus/TensorFold), [the four-node recipe and kit](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10)) | [BertholomusAI](https://github.com/bertholomus) | Apache-2.0 |
| The four-Spark recipe this repository's setup began as: its engine changes (Engram's packed rows among them), the scripts, the chat template | [MiaAI-Lab](https://github.com/MiaAI-Lab) and its contributors | Apache-2.0 |
| Fixes by others, each under its author's name | the authors listed in [`CREDITS.md`](CREDITS.md) and [`FORK.md`](FORK.md) | their own |
| The model | [DeepSeek](https://huggingface.co/deepseek-ai) | MIT |
| The optional abliteration overlay (`ABLIT=1`) | [drowzeys (Keys)](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated) | MIT (gated) |
| This repository's engine work, scripts and the kernel kit's extra set ([its kit fork](https://github.com/rbb-dev/deepseek-v4.1-tensorfold-tp4-4xgb10)) | [rbb-dev](https://github.com/rbb-dev) | Apache-2.0 |

This repository's work was developed with the help of [Claude Code](https://claude.com/claude-code). [`NOTICE`](NOTICE) and
[`CREDITS.md`](CREDITS.md) keep every author's notice and say who wrote what.

## Words used here

- **The first Spark**: the one you run `./setup.sh` and `./start.sh` on; it serves the API (rank 0). **Workers**: the
  other three. **Ranks**: the four engine processes, one per Spark.
- **Fabric** and **rails**: the Sparks' private network; each ConnectX-7 port shows up as two network devices, two rails
  of ~100 Gb/s each ([`NETWORK.md`](NETWORK.md)).
- **RDMA**: one machine writing into another's memory without its CPU; **RoCE**: RDMA over Ethernet; **PFC**: priority flow
  control, which makes the switch pause a sender instead of dropping its packets.
- **GPU-triggered send**: a transfer handed to the network card in advance, which the card starts the moment the GPU
  finishes writing the data. **Completion flag**: the value the card writes after the data, which the receiving GPU waits on.
- **Engram**: the model's n-gram memory, two tables of ~384 million rows (~203 GB together), read from disk row by row.
- **The drafter**: the checkpoint's own draft layers, which guess tokens that the model then checks.
- **The indexer**: the part of each attention layer that scores earlier tokens to choose the 512 the layer attends to.
- **Cold**: a prompt no kept state serves, read in full. **13.7 % faster**: 13.7 % more speed than without the change.

## How this repository relates to its upstreams

This repository is a git fork of BertholomusAI's TensorFold branch `deepseek-v41-zig-tp4`. The history below this
repository's own work is his and TensorFold's, each commit under its author's name. This repository's work sits on top as
its own commits, and his branch is merged in as it moves (`git log --first-parent` shows this repository's own line).
[`FORK.md`](FORK.md) lists the features, their authors and their settings. The engine image is built from this checkout
([`zig/Dockerfile`](zig/Dockerfile)), so what runs is what is here. A `TF_DS_...` setting is off unless set;
[`scripts/served.sh`](scripts/served.sh) turns on the ones this repository runs with.

## Repository layout

```
setup.sh                  set up four Sparks from scripts/local.sh: config, access, network, data
start.sh, stop.sh         start and stop all four ranks on a switch
scripts/local.sh.example  your config file's template: copy it to scripts/local.sh
scripts/served.sh         the settings this repository runs with
scripts/config.sh         every setting and its default
scripts/prepare.sh        the image, checkpoint, kit, weight files, Engram's shards and each Spark's checks
scripts/setup-data.sh     the data step: this Spark's data, then each worker's share over the fabric
scripts/setup-lib.sh      the setup's helpers: the copy's routes, the keys, the netplan check
scripts/fabric-copy.sh    the fallback copy (TCP), SHA-256 on both ends
tools/fabric-xfer/        the setup's copy: RDMA over both rails, each file read once, CRC-32C on both ends
zig/                      the engine: the server, the scheduler, the dashboard, the DeepSeek family
                          (zig/src/families/dsv41) with Rocinante, its CUDA kernels and the tests
zig/Dockerfile            the engine image, built from this checkout
model/chat_template.jinja the chat template the server renders
network/                  each Spark's netplan file and flow-control unit, the switch's configuration
tools/                    validate.py, recipe_bench.py, bench.py, needle_bench.py, multi_needle.py, needle_test.sh,
                          longbench2.py, reason4m/ (MRCR, BABILong), readme_tables.py
tests/                    the scripts' and tools' offline tests (Python's standard library, no Spark needed)
evidence/                 the runs behind the published numbers, one folder per engine image
docs/                     the details: what each change measured, long context, heat and cores; TensorFold's own docs
FORK.md                   this repository's features: what each does, who wrote it, its setting
ROCINANTE.md              Rocinante: how it works, what it gives, its settings
NETWORK.md                preparing the network: any RoCE switch, the MikroTik's configuration, each Spark
CHANGELOG.md              the changes, newest first; TensorFold's own below
CREDITS.md, NOTICE, LICENSE
README-BertholomusAI.md   BertholomusAI's README for the engine branch
```

## License

This repository's code, scripts and documentation: Apache License 2.0 ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE) lists the
third-party work and its licenses. The model's weights are DeepSeek's, under the MIT license, and are not part of this
repository. All product and company names (among them DeepSeek, NVIDIA, DGX Spark, GB10, ConnectX, CUDA, NCCL, Dell Pro
Max, MikroTik, RouterOS and Claude Code) are trademarks of their owners, used here for identification only. This
repository is not affiliated with or endorsed by DeepSeek, NVIDIA, Dell, MikroTik, Anthropic, the TensorFold authors,
BertholomusAI or MiaAI-Lab.
