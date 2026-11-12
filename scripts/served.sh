# Every setting the engine runs with on four DGX Sparks on one RoCE switch, as its own start report lists them. Loaded by
# the first line of scripts/local.sh (copied from local.sh.example), so a line of yours below it wins; the environment wins
# over both. Each TF_DS_... switch here was measured on the four Sparks against the engine without it, with replies equal
# byte for byte, and all of them run together; some are also on by default, and are named here so the list is complete.

# ---------------------------------------------------------------- serving
CONTEXT=1048576                  # the model's whole window for every stream
POOL=16777216                    # the pool's size only without TF_DS_KV_FREE_TARGET_GIB (below), which sizes it from free memory
PROMPT_CACHE_GIB=2               # kept prompt states in memory (agent sessions return to the same long prefix)
MAX_TOKENS=1048576               # a request without max_tokens may use the whole window
HOST=0.0.0.0                     # the API on every address of this Spark
PORT=8000                        # (any free port)
SERVED_NAME=deepseek-v4.1-flash-1017   # the name clients ask for (any name; the engine answers whatever name a request sends)
DRAIN_TIMEOUT=120                # ./stop.sh lets running requests finish for up to 120 s
WAIT_TIMEOUT=600                 # ./start.sh waits up to 600 s for the API
GPU_CLOCK=2200                   # this setup's GPU clock limit (MHz): the unlocked clock measured no faster
                                 # and an unlocked GB10 once cut its own power under a long prompt

# ---------------------------------------------------------------- MiaAI-Lab's switches, on
export TF_DS_IDX_RB=2            # the indexer's kit kernel at RB=2
export TF_DS_IDX_SHARD=1         # the indexer scores only its own quarter of rows
export TF_DS_SAMPLE_CAND=1       # candidate sampling on the GPU
export TF_DS_EXPERT_EXIT=1       # decode's expert kernels skip inactive rows early
export TF_DS_PREFILL_YIELD=1     # other requests get a round between a long prompt's chunks
export TF_DS_RDMA_CPUS=0,10,11   # the RDMA proxies on small Cortex-A725 cores (MiaAI-Lab's issue #40)

# ---------------------------------------------------------------- the SSD tier: kept prompt states on every Spark's NVMe (each its quarter)
SPILL_GIB=256                    # the store's cap; 0 turns the tier off
SPILL_FLOOR_GIB=50               # free disk always left
SPILL_TURNS=4                    # older turns a conversation keeps on disk past its newest
export TF_DS_SPILL_RESTORES=8    # restores at once (a burst of returning agent sessions does not queue behind two)
SPILL_STAGING_MIB=192
REPLAY_FLOOR=1                   # a short chunk at the replay cut reads its window from the floor

# ---------------------------------------------------------------- Engram's rows
ENGRAM_PACKED=1                  # one read a row
ENGRAM_OWNED=0                   # the whole tables on every Spark (~203 GB on every Spark): no exchange a layer;
                                 # prompts beside four decoders +44.8 %
ENGRAM_CACHE_MIB=2048
ENGRAM_INSTREAM=0
export TF_DS_ENGRAM_PREFAULT=1   # the row cache faulted in at the start

# ---------------------------------------------------------------- prompt speed: fewer bytes between the Sparks
# each Spark receives only its own rows' sums, and FP8 codes go where the receiver would make the same codes:
# always on in this engine (Rocinante carries them), so nothing to set
export TF_DS_PASS_ROWS=4096      # prompt passes of 4,096 rows where nothing else waits: prompts 13.7 % faster
                                 # (about 1.5 GiB more memory a Spark)

# ---------------------------------------------------------------- prompt speed: kernels
export TF_DS_PF_SPLIT=1          # split FP8 GEMMs on a restructured loop: prompts 1.0 % faster
export TF_DS_QUANT_ONCE=1        # several small kernel changes: prompts 1.2 % faster together
export TF_DS_GROUP_TILES=1
export TF_DS_TOPK_TILES=1
export TF_DS_IDX_PROJ_OWNED=1
export TF_DS_BF16_ROWS=1
export TF_DS_PF_RUN128=1         # the MXFP4 expert tiles stage each weight row 128 bytes at a time: 0.7 % faster
export TF_DS_WOA_W8=1            # wo_a from its FP8 rows (~0.67 GB less memory a Spark; speed unchanged)
export TF_DS_MMA_SF_EXACT=1      # FP8 GEMMs on the block-scaled tensor-core multiply, the same bits: 2.2 % faster

# ---------------------------------------------------------------- these need the kernel kit's extra set (aot-extra/)
# ATTN_U, IDX_U and the four-way drafter run the kit's twins with FP4 unpacked in hardware: the extra set in this
# repository's kit (the default KIT_REPO / KIT_REF in scripts/config.sh: BertholomusAI's kit plus that set). With a kit
# that has no extra set each says at the start that it is off, prepare.sh names it, and the engine runs without them.
export TF_DS_ATTN_U=1            # prompts 1.1 % faster
export TF_DS_IDX_U=1             # prompts 1.6 % faster
export TF_DS_DRAFT_4WAY=1        # the drafter's stages split four ways like the backbone's layers:
                                 # decode one stream 2.9 % faster, 16 streams 4.0 % faster (less memory a Spark: each holds a quarter of the drafter's stages, not half)

# ---------------------------------------------------------------- decode speed
export TF_DS_DRAFT_HEAD=fp4      # the drafter's head as MXFP4: one stream prose 1.9 % faster, code 1.0 % faster
export TF_DS_COPY_MATCH=16
export TF_DS_RDMA_COLLECT_PEERS=1 # one stream 1.3 % faster together with the lines above
export TF_DS_CAND_FIRST=1
export TF_DS_DRAFT_MAIN_4WAY=1
export TF_DS_COPY_ROWS=17
export TF_DS_HEAD_CAND=1         # each Spark ranks its own quarter of the vocabulary
export TF_DS_HEAD_KEY=1          #   (sampled 0.74 % faster at 16 streams; greedy code 1.35 % faster)

# ---------------------------------------------------------------- heat: no busy-waiting CPU thread (no speed lost)
export TF_DS_CUDA_SCHED=blocking # a host thread waiting on the GPU sleeps: the spinning cluster 10-14 °C cooler
export TF_DS_CUDA_EVENTS=blocking # every event wait sleeps: rank 0's engine thread 91 % -> 4 % of a core
export TF_DS_RDMA_WFE=1          # the RDMA proxies nap with Arm WFET (94-99 % of the time) when the card's wait is off
export TF_DS_ISSUE_LEAD=4         # the engine thread stays 4 layers ahead of the GPU and sleeps, not spinning in the launch queue
export TF_DS_ENGRAM_WAIT_US=100   # the engine thread sleeps 100 us between looks for Engram's rows
export TF_DS_HYBRID_WAIT=1        # a decode round's reads waited for asleep until ~250 us before their learned time
export TF_DS_ROCI_IDLE_PARK_US=2000 # idle transport threads sleep in the kernel
# (NCCL is not used: Rocinante carries every exchange, each transfer a GPU-triggered send that the network card starts
#  itself; TF_DS_ROCI_WAIT and TF_DS_RING_WAIT are on by default, so no proxy thread runs in the transfer path)

# ---------------------------------------------------------------- the KV pool
export TF_DS_KV_VMM=1             # the pool in a CUDA virtual-memory reservation, resized while serving
export TF_DS_KV_FREE_TARGET_GIB=12 # the pool takes what fits while every Spark keeps 12 GiB free (about 18.3M positions at the 1M window)

# ---------------------------------------------------------------- Rocinante: every exchange, for prompts and for replies
export TF_DS_ROCI_STAGGER=1       # two prompt passes in flight on two lanes, one computing while the other's data is on the wire (9.2 % faster)
export TF_DS_ROCI_P2_FIRST=1      # an owner sum's second part sent first
export TF_DS_ROCI_ENGRAM_AHEAD=1  # an Engram layer's rows, projection and first send a layer ahead
export TF_DS_ROCI_COMBINE_DOOR=1  # the routed sum and its send trigger in one launch
export TF_DS_ROCI_SENT_FIRST=1    # wo_b's sum in the sending launch, its send triggered from inside it
export TF_DS_ROCI_CPUS=3,4        # the first lane's proxy cores (small cores), when the card's wait is off
export CUDA_DEVICE_MAX_CONNECTIONS=16 # 16 GPU hardware queues (the driver's default is 8)
export TF_DS_RING_WORD=1          # the decode rings end each transfer with a completion flag (the card's wait needs it)
export TF_DS_RING_WC=1            # decode's rings on write-combined slots
export TF_DS_RING_INPLACE=1       # decode's exchanges write straight into the ring's send slots
export TF_DS_ADD2_VEC=1           # the owner sums' add four elements a thread with 16-byte loads (the same bits)

# ---------------------------------------------------------------- prompt passes, and prompts beside decoding
export TF_DS_PASS_FROM=0          # larger prompt passes from the first row
export TF_DS_PASS_FILLING=1       # whole passes in turn while several prompts fill: 2 and 3 prompts at once 15.9 % faster
export TF_DS_PASS_BESIDE=4096     # the largest prompt pass while a stream decodes
export TENSORFOLD_PREFILL_SHARE=0.83 # a long prompt's share of the time while other streams decode
export TF_DS_PROMPT_ORDER=serial  # several prompts: one finished before the next

# ---------------------------------------------------------------- Engram's reader
export TF_DS_ENGRAM_AHEAD_PASS=1  # rows read a layer ahead for a prompt pass: the first prompt after a start 5,457 vs 4,330 tok/s
export TF_DS_ENGRAM_PINNED=1      # rows staged in pinned memory
export TF_DS_ENGRAM_KEEP_AHEAD=1  # the reader stays across decode rounds between a prompt's chunks
export TF_DS_ENGRAM_PARTNER=1     # column partners split each other's disk reads

# ---------------------------------------------------------------- kernels (most from the kit's extra set)
export TF_DS_SPLIT64N=1           # wkv's and wq_a's exact split GEMMs on a 64 x 64 tile, two blocks an SM
export TF_DS_WOA_MERGE=1          # wo_a's two groups in one launch
export TF_DS_AQ_VEC=1             # the activation quantizer on four lanes a group, 16-byte loads
export TF_DS_IDX_TPP=4            # the indexer's scores at 4 key tiles a program, the next tile prefetched into L1
export TF_DS_ATTN_EPILOGUE=1      # the attention output's inverse RoPE and quantization in one launch
export TF_DS_ATTN_EPI_FUSED=1     # the same inside the prompt's attention kernel
export TF_DS_ATTN_EPI_TWIN=we     # which kit kernel the fused epilogue runs
export TF_DS_ATTN_TWIN=z          # sparse attention from the kit's z twin (fewer instructions; decode's picks prefetched into L2)
export TF_DS_HC_STRIDE=1          # the quarter mixes read the gathered planes in place
export TF_DS_HC_2S=1              # the prompt's mix on the two-block sums, four streams a program
export TF_DS_PACK_EPILOGUE=1      # gate/up writes the expert pack's codes itself
export TF_DS_ROPE_W=1             # RoPE on 32-bit bf16 pairs
export TF_DS_WINDOW_ONCE=1        # a chunk's window indices uploaded once
export TF_DS_L2_NEXT_EARLY=1      # the next layer's L2 prefetch behind gate/up, beside the exchange

# ---------------------------------------------------------------- decode
export TF_DS_DEC_GROUP2=1         # decode's expert grouping by a parallel scan
export TF_DS_DRAFT_VOCAB=1        # the drafter proposes from 32,768 token ids (8,192 a Spark)
export TF_DS_MERGE_D=3            # decode attention's merge in column blocks
export TF_DS_QKV_PLAIN=1          # a decode window's q norm and window KV in one launch
