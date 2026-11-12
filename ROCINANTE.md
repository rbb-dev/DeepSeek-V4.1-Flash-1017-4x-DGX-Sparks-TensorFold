# Rocinante: the transport between the Sparks

Rocinante ("Roci", after the ship in *The Expanse*) is the network transport this repository adds to TensorFold's
DeepSeek-V4.1 engine (`zig/src/families/dsv41/rocinante*.zig`), in place of NVIDIA NCCL, between four DGX Sparks. It is
built for the GB10, where the CPU and the GPU share one memory: the producing kernels write their outputs directly into
the buffers the network cards send from, and the consuming kernels read them where the remote card placed them.

**Rocinante carries every exchange between the Sparks, for reading prompts and for writing replies, and NCCL is not
loaded.** Every rank says so in its start log ("no NCCL is loaded or made"). The prompt exchanges run on Rocinante's own queue pairs. The decode exchanges
run on BertholomusAI's RDMA rings (`rdma.zig`, `ring2d.zig`), which Rocinante drives with the same GPU-triggered sends,
completion flags and card-side waits. The first Spark (rank 0) sends each step's small control frames to the others over
TCP.

## Why

GPUDirect RDMA is not supported on the DGX Spark: memory the GPU allocates the usual way (`cudaMalloc`) cannot be read by
the network card, and NVIDIA recommends pinned host buffers registered with the card instead
([NVIDIA's DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/porting/cuda.html)). On the
Sparks, NCCL therefore staged every prompt-sized exchange through its own copies; its kernels held GPU multiprocessors
while they waited for the network; and its progress thread ran at 100 % of a fast CPU core. Rocinante is built around the
memory NVIDIA recommends, from the buffers up.

## How it works

- **Zero-copy, one-sided RDMA from the producing kernel to the consuming kernel.** The tensors that cross between Sparks
  are allocated in pinned host memory mapped into the GPU (zero-copy memory, `cuMemHostRegister`), registered as memory
  regions on both ConnectX-7 rails. Measured on one Spark, the GPU streams this memory at 168 GB/s read and 111 GB/s
  write, against 252 and 194 GB/s for `cudaMalloc` memory in the same test. The producing kernel writes its output there
  once, and the sender's card writes it into the buffer the receiver's next kernel reads, half of each part on each rail.
  There is no staging copy on either side.
- **GPU-triggered sends: the card waits, not a thread** (`TF_DS_ROCI_WAIT`, on by default; `rocinante_cd.zig`). In short:
  the network card is given the whole transfer in advance and waits for the GPU to say the data is ready, then sends it
  with no CPU thread involved. In detail: each queue
  pair is made as a Core Direct master through DEVX (Core Direct: the ConnectX-7 mode in which a send waits on a value in
  memory; DEVX: the driver interface that creates such queue pairs). When the engine reserves a send, it writes the whole chain into the
  card's send queue in advance, behind a WAIT entry on a trigger value in memory. The GPU's stream writes that value when
  the data is complete, and the ConnectX-7 itself runs the WAIT, the data writes and the completion flag. No CPU thread
  polls, posts or reaps in the transfer path. Measured between two Sparks over 1,000 round trips: the card's pre-posted
  WAIT releases a write 6.21 µs after the trigger is written (99th percentile 6.82 µs). A CPU thread posting the write
  takes 6.64 µs (99th percentile 6.88 µs). The decode rings wait in the card the same way (`TF_DS_RING_WAIT`, on by default): each of their queue pairs
  waits on its own trigger.
- **Completion flags behind the data.** The completion flag is an RDMA atomic fetch-and-add of 1 on the receiver's flag
  (`TF_DS_ROCI_SIGNAL=atomic`, the default), which the card places after the writes before it on the same queue pair. The
  receiver's consuming kernel is already queued behind `cuStreamWaitValue64` on that flag (`>= N`, compared cyclically),
  so it starts when the flag lands. The wait is a stream memory operation: it holds no multiprocessor and no CPU thread,
  and the count tells one use of a buffer from the next.
- **Failure.** If a peer's flags stop advancing for 5 s, a watchdog fails the transport on that Spark: the queue pairs are
  taken from the card before the trigger can hold the abort value (which would release every armed WAIT), every queued wait
  is released, and the engine exits, logging the Spark, the reason, the exchange and the peer.
- **Two lanes.** A second, independent set of buffers and queue pairs keeps two prompt passes in flight: one computes
  while the other's exchange is on the wire (`TF_DS_ROCI_STAGGER`).
- **Any pass size.** The exchanges of a pass of any row count run on Rocinante, each Spark computing a quarter of the rows
  (rounded up to 16).
- **Where it runs.** Every exchange: when reading a prompt, the owner sums, gathers, FP8 codes and the decoder half of a
  prompt's last pass on Rocinante's queue pairs; when writing replies, the decode exchanges, the drafter's gathers among
  them, on the rings. The start-up agreements and the SSD tier's digests go over TCP links from the first Spark.

Outputs are byte-identical to NCCL's: checked at exactly 4,096 and 4,097 tokens, at about 2K, 32K and 131K tokens, and
with an 8K-token image prompt.

## What it gives

**Compute while the data is on the wire.** Rocinante waits for incoming data without holding the GPU, so while one pass
of a prompt waits for its data, the next pass computes. NCCL's waiting kernels hold the GPU, so the same arrangement on
NCCL gained nothing (the same speed, a cold ~147K-token prompt). On Rocinante it takes that prompt from 5,792 to 6,324 tokens a
second (9.2 % faster). Without it, a 4,096-row pass of 692 ms spends 109 ms waiting for the wire (profiled); with it, the next
pass's compute runs in those waits.

**Against NCCL**, measured on the same four Sparks with the same benchmark (MiaAI-Lab's): reading prompts 8 to 29 % faster
from 4K to 256K tokens, writing replies 4 % slower to 2 % faster, outputs byte-identical. The current numbers are in the
[README](README.md#performance).

**No busy-waiting thread.** With the card waiting, the transport keeps no thread in the transfer path. Measured 30 s after
the last request, no engine thread on any of the four Sparks uses 2 % of a core or more (about 50 M cycles a
second). Engram's reader on the prompt path keeps at most 256 reads in flight, where the drive reaches its
ceiling of ~293,000 reads a second, and sleeps between batches of finished reads. While it issues a prompt, the engine's
thread stays at most 4 layers ahead of the GPU and then sleeps (`TF_DS_ISSUE_LEAD=4`).

**Energy.** On three cold ~147K-token prompts processed alone, reading the prompts ran 18 % faster than with NCCL and used 5 % less
GPU energy for the same work: 28.6 J against 30.2 J per 1,000 prompt tokens. The four GPUs together drew more power with
Rocinante (184.4 W against 164.7 W with NCCL), and the work finished sooner. The power is the GPUs' as `nvidia-smi` reports
it, sampled every 200 ms on all four Sparks, not the power drawn at the wall; one run of three prompts each, so the 5 % is within reach of
the run-to-run noise.

## Its settings

Rocinante is on in every start; it is the only transport. Its main settings follow; `scripts/served.sh` sets the ones the
engine runs with.

| Setting | What it does |
|---|---|
| `TF_DS_ROCI_WAIT` | on (unset or 1, the default): every chain waits in the card behind its trigger, with no proxy threads; `0`: per-rail proxy threads post the writes, napping on the Arm CPU's timed wait-for-event between sends |
| `TF_DS_RING_WAIT` | the same for the decode rings: on by default; `0`: the rings' proxy threads |
| `TF_DS_ROCI_SIGNAL` | `atomic` (the default): the completion flag is an RDMA atomic add placed after the writes |
| `TF_DS_ROCI_STAGGER` | two whole passes of a prompt in flight on two lanes |
| `TF_DS_ROCI_P2_FIRST` | an owner sum's second part is sent as soon as its add has made it, before the add of this Spark's own block |
| `TF_DS_ROCI_ENGRAM_AHEAD` | an Engram layer's rows, projection and first-part send run at the start of the layer before it |
| `TF_DS_ROCI_COMBINE_DOOR` | the MoE's routed sum runs in one launch whose first blocks are the row quarters the first part sends, and the last of those triggers the part's send |
| `TF_DS_MMA_SF_EXACT` | a prompt pass's FP8 matrix multiplies on the block-scaled MMA with exact accumulation (the same bits) |
| `TF_DS_ENGRAM_PINNED` | the Engram worker stages its rows in pinned host memory |
| `TF_DS_ROCI_SENT_FIRST` | `wo_b`'s sum runs in the launch that sends it, and that launch triggers the send |
| `TF_DS_ROCI_IDLE_PARK_US` | idle transport threads sleep in the kernel after this many microseconds (2,000 in `served.sh`) |
| `TF_DS_ROCI_CPUS` | the cores of the first lane's proxy threads when the card's wait is off (small cores 3 and 4; the second lane's on 1 and 2) |

`CUDA_DEVICE_MAX_CONNECTIONS=16` goes with them. Each Spark restores the SSD tier from its own store.
