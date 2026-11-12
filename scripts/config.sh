# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`
#   2. scripts/local.sh (this setup's own values, above all WORKER1 .. WORKER3; sourced as bash), then ./.env (KEY=value
#      lines, read, never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  source "$_cfg_root/scripts/local.sh"
  # the environment wins over local.sh: put back any variable it had that local.sh changed
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
if [[ -f "$_cfg_root/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$_cfg_root/.env"
fi
unset _n _line _key _value

# ---------------------------------------------------------------- the four Sparks
# This machine serves rank 0 and the API; WORKER1, WORKER2, WORKER3 (ssh targets, key-based) are ranks 1-3.
# The four share one switch (NETWORK.md); the workers can be listed in any order.
TP=4
# WORKER_HF_CACHE1, WORKER_HF_CACHE2, WORKER_HF_CACHE3: a path on that worker holding its own copy of what its rank reads,
# laid out as HF_CACHE is here (scripts/nodes.sh: check_worker_files); unset, it reads this Spark's over NFS.
WORKER1="${WORKER1:-}"; FABRIC_PEER1="${FABRIC_PEER1:-}"; WORKER_HF_CACHE1="${WORKER_HF_CACHE1:-}"
WORKER2="${WORKER2:-}"; FABRIC_PEER2="${FABRIC_PEER2:-}"; WORKER_HF_CACHE2="${WORKER_HF_CACHE2:-}"
WORKER3="${WORKER3:-}"; FABRIC_PEER3="${FABRIC_PEER3:-}"; WORKER_HF_CACHE3="${WORKER_HF_CACHE3:-}"
MASTER_PORT="${MASTER_PORT:-29760}"  # the ranks' rendezvous port (keep it on the private links)
# The fabric rails (setup.sh's network step, NETWORK.md): each Spark's two CX-7 functions, one subnet each, MTU 9000;
# the addresses this Spark's first, then WORKER1's, WORKER2's and WORKER3's (the fabric's own, on its isolated VLAN)
RAIL_A_IF="${RAIL_A_IF:-enp1s0f0np0}"; RAIL_B_IF="${RAIL_B_IF:-enP2p1s0f0np0}"
RAIL_A_ADDRS="${RAIL_A_ADDRS:-10.255.0.1 10.255.0.2 10.255.0.3 10.255.0.4}"
RAIL_B_ADDRS="${RAIL_B_ADDRS:-10.255.1.1 10.255.1.2 10.255.1.3 10.255.1.4}"
# The rendezvous address: this node's address every worker reaches. When this Spark holds rail A's first address (the
# network step puts it there), rail A: the ranks' start-up exchange then stays off the LAN, as the published numbers ran.
# Else what its hostname resolves to, else its first address. A value you set wins.
if [[ -z "${MASTER_ADDR:-}" ]]; then
  _ra=${RAIL_A_ADDRS%% *}
  if ip -o -4 addr show dev "$RAIL_A_IF" 2>/dev/null | grep -q -F " $_ra/"; then
    MASTER_ADDR=$_ra
  else
    _ma=$(getent ahostsv4 "$(hostname)" 2>/dev/null | awk '$1 !~ /^127\./ {print $1; exit}' || true)
    [[ -n "$_ma" ]] || _ma=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
    MASTER_ADDR=$_ma
  fi
fi
RAIL_PREFIX="${RAIL_PREFIX:-24}"; RAIL_MTU="${RAIL_MTU:-9000}"
# At three or four Sparks every rail between this node and a worker carries a full-size ping both ways before the
# ranks start (scripts/nodes.sh probe_links): a rail that does not answer stops the start, named. 0 skips the check.
RAIL_CHECK="${RAIL_CHECK:-1}"
# The 2D split's node (role) each rank serves (the engine's TF_DS_2D_ORDER), comma separated; auto: 0,1,2,3.
ORDER="${ORDER:-auto}"

# ---------------------------------------------------------------- the checkpoint, the engine, the kernel kit
# DeepSeek's own release, served as published: FP8 E4M3 linears (UE8M0 scales on 32 x 32 blocks), MXFP4 routed
# experts, bf16 elsewhere, the Engram tables in FP8 read from disk. ~510 GB, downloaded once on this Spark; a worker
# reads what its rank needs from its own copy (WORKER_HF_CACHE*) or from here over NFS.
# DERISKED=1 selects Blackfrost-AI's checkpoint; explicit MODEL_ID / MODEL_REVISION still win.
DERISKED="${DERISKED:-0}"
[[ "$DERISKED" =~ ^[01]$ ]] || { echo "DERISKED must be 0 or 1" >&2; exit 1; }
_default_model=deepseek-ai/DeepSeek-V4.1-Flash-1017
[[ "$DERISKED" != 1 ]] || _default_model=Blackfrost-AI/DeepSeek-V4.1-Flash-1017-DERISKED-FP8
MODEL_ID="${MODEL_ID:-$_default_model}"
_rev=""
case "$MODEL_ID" in
  deepseek-ai/DeepSeek-V4.1-Flash-1017) _rev=1bae0e1088993b9b82de08f59de5be5a34e42170 ;;
  Blackfrost-AI/DeepSeek-V4.1-Flash-1017-DERISKED-FP8) _rev=4bb7df57c6bf26caf40e8da905e4062aed088ea6 ;;
esac
unset _default_model
MODEL_REVISION="${MODEL_REVISION-$_rev}"
# ABLIT=1 grafts drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated's native-FP8 sidecar (attn.wo_b, layers 10-35,
# the overlay's "recipe A") onto the checkpoint before the rank caches are baked. Gated, free, automatic approval
# after accepting its terms at https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated while
# logged in; export HF_TOKEN or `hf auth login` with that account. The vendor validates this exact sidecar against
# SGLang/vLLM, not TensorFold -- its own TensorFold-validated recipe needs a different, EXL3-quantized checkpoint
# this repo does not serve. Validate live (tools/validate.py, tools/needle_bench.py) before trusting a build of it.
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" =~ ^[01]$ ]] || { echo "ABLIT must be 0 or 1" >&2; exit 1; }
ABLIT_ID=drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated
ABLIT_REVISION="${ABLIT_REVISION:-de412d2249632903dace23399526e00a145aada5}"
ABLIT_SIDECAR=wo_b_l10_35_1017.safetensors
# The engine is this repository (zig/, built by zig/Dockerfile from the commit checked out here). ENGINE_PATHS are
# what the image is built from: a commit that changes none of them reuses the image.
ENGINE_PATHS=(zig build.zig build.zig.zon .zig-version tools/dsv41-zig LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES)
# The family's kernel kit (its Triton kernels as captured binaries, Engram's constants, the vision tower's attention
# kernel): the kit/ of BertholomusAI's four-node recipe at v0.1.1, checked against its MANIFEST; the RoPE tables and
# the image routing bias are made from the checkpoint by its verify_kit.sh. The default is this repository's fork of
# that kit: BertholomusAI's kit unchanged, plus the extra set (kit/aot-extra/) that this repository's kernel switches load
# (its kit/README.md, "The extra set"). A kit's RoPE tables are cut for one YaRN factor (its rope.json) and the engine
# refuses any other, so the default follows TF_DS_YARN_FACTOR: three commits of the fork that differ only in those tables,
# for the 1M window (the checkpoint's factor 16), the 2M (32) and the 4M (64). A KIT_REF you set wins.
# BertholomusAI's own kit (that repository at 6033ea48a27004b2715db82190e2a61f33411d92) runs without them.
KIT_REPO="${KIT_REPO:-https://github.com/rbb-dev/deepseek-v4.1-tensorfold-tp4-4xgb10.git}"
case "${TF_DS_YARN_FACTOR:-16}" in
  32) _kit=30cb665fe6cd15ed113c198f4ba34b5da944923d ;;   # RoPE tables cut for factor 32 (the 2M window)
  64) _kit=3abe1d5ecb7a718b1ef16ee425de5d9ab5542c17 ;;   # RoPE tables cut for factor 64 (the 4M window)
  *)  _kit=b88aca850cb2e473069a4cd5d225e7a4b188fe8b ;;   # the checkpoint's factor 16 (the 1M window)
esac
KIT_REF="${KIT_REF:-$_kit}"; unset _kit
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"   # zig-aarch64-linux-0.17.0.tar.xz
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c}"   # by its digest: the exact image every number was built on
# The commit the image is built from (empty in a copy without git's history: then the files themselves are used).
source_commit() { git -C "$_cfg_root" rev-parse -q --verify HEAD 2>/dev/null || true; }
# The image's tag: git's own hashes of ENGINE_PATHS at that commit (or, without git, the files' SHA-256), the
# Dockerfile, Zig and the base image.
image_hash() {
  ( cd "$_cfg_root" || exit 1
    if [[ -n "$(source_commit)" ]]; then git ls-tree -r HEAD -- "${ENGINE_PATHS[@]}"
    else find "${ENGINE_PATHS[@]}" -type f -print0 2>/dev/null | LC_ALL=C sort -z | xargs -0 sha256sum; fi
    cat zig/Dockerfile
    echo "zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE"
  ) | sha256sum | cut -c1-12
}
IMAGE="${IMAGE:-tensorfold-dsv41:native-$(image_hash)}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-1017-tf}"   # the same name on every Spark

# ---------------------------------------------------------------- serving
SERVED_NAME="${SERVED_NAME:-deepseek-v4.1-flash-1017}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# A request's positions at most (prompt and reply), and the positions all requests share (the streams' extents in one
# pool). The model's window is 1,048,576 tokens.
CONTEXT="${CONTEXT:-1048576}"
POOL="${POOL:-2097152}"
PARALLEL="${PARALLEL:-16}"             # requests at once (the engine's streams)
ROUND_ROWS="${ROUND_ROWS:-48}"         # a decode round's rows at most (drafts of every stream together)
MAX_TOKENS="${MAX_TOKENS:-32768}"      # a reply's budget when a request sets none
THINKING="${THINKING:-1}"              # think first unless a request turns it off
# The reasoning effort when a request names none (the chat template's levels: low 50, high 75, max 100 on its 1-100
# scale; a request's own reasoning_effort wins). Without it the template's own default is high.
REASONING_EFFORT="${REASONING_EFFORT:-max}"
# The model card's recommended sampling (temperature 1.0, top_p 1.0) as the server's defaults, from the 20 likeliest
# tokens: the family's exact sampler needs a top-k (top_k 0, the nucleus alone, is not in it; BertholomusAI serve 20).
# A request's own values win.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
DRAFTS="${DRAFTS:-1}"                  # the checkpoint's own DSpark draft blocks (exact: drafted replies equal plain)
# The engine's settings as BertholomusAI's served launch has them (their README "The served settings"): CUDA graphs,
# the mixes' side stream, the paced L2 prefetch, Engram reads by Linux AIO, the lane core's measured round costs.
# The ranks' exchanges over RDMA (RoCE), on each node's RoCE devices (scripts/nodes.sh finds them).
RDMA="${RDMA:-1}"
GRAPHS="${GRAPHS:-1}"
# CUDA graphs kept at most (TensorFold issue #69): past it the least recently replayed stretch is
# destroyed and captured again when it comes back; every rank keeps the same ones. 0 keeps every graph, as before,
# and their device and host memory grows for days as conversations reach new sizes. Sent only with GRAPHS=1.
GRAPH_CAP="${GRAPH_CAP:-128}"
# The prompt cache's budget in GiB (--prompt-cache-gib; empty: the engine's 0.5 GiB, about 79 kept states). A kept
# state costs about 6.4 MiB on every Spark whatever the conversation's length (its window rings and positional
# stores; its rows stay in the pool), so the budget counts conversations, not tokens.
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"
# The API's keys (the engine's --api-key-file): a file on this Spark of "label: key" lines, readable by its owner only
# (chmod 600: the engine refuses it otherwise; it rereads the file when it changes). Every route but /health then needs
# a key (Authorization: Bearer, or x-api-key), /metrics too unless METRICS_OPEN=1 (counts, no text: dashboards keep
# reading it); /metrics counts each key's requests by its label. start.sh's own checks send the file's first key.
# Empty: no key (the engine warns when it listens beyond loopback).
API_KEY_FILE="${API_KEY_FILE:-}"
METRICS_OPEN="${METRICS_OPEN:-1}"
# ADMIN_KEY_FILE: the admin surface's own keys (/admin/api/v1 and the page at /admin/ui), the API key file's format, chmod
# 600, each key at least 32 bytes and in no other key file, its label's prefix its scope (admin*, agent-*, read-*; any
# other label reads only). Mounted for rank 0 alone. Empty: every /admin path answers like an unknown one.
ADMIN_KEY_FILE="${ADMIN_KEY_FILE:-}"
# ADMIN_POLICY_FILE: the admin surface's policy (admin-policy.json: agent profiles, request_caps per key label), mounted
# read-only for rank 0 alone; not writable by others. Empty: no policy (per-request fields dropped, no agent profiles).
ADMIN_POLICY_FILE="${ADMIN_POLICY_FILE:-}"
# ENGINE_CFG: this setup's engine.cfg (`key = value` lines; the engine's own doc: setting tf_engine_cfg). Every rank applies
# the same file (rank 0 reads it from its folder, mounted writable: the engine keeps engine.history.jsonl beside it; each
# worker gets a copy of the file, read-only); a start where the ranks applied different files refuses. Keep it in a folder
# of its own. Empty: no file, the environment alone.
ENGINE_CFG="${ENGINE_CFG:-}"
# stop.sh waits up to DRAIN_TIMEOUT seconds for the requests running or waiting (/metrics) to finish before it stops the
# ranks; 0 stops at once, cutting them off. Either way it says how many it cuts off.
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-0}"
# Then each rank gets STOP_TIMEOUT seconds to end (docker stop -t) before docker kills it. Rank 0's orderly close (its
# requests ended, the SSD tier's flush, the followers told "done") is bounded by TENSORFOLD_CLOSE_MS, the engine's own
# setting, derived from STOP_TIMEOUT: STOP_TIMEOUT x 2000 / 3 ms (20,000 at 30 s), so the close and the server's 2 s
# wait for its connections end inside the stop; the tier's flush (SPILL_FLUSH_MS, below) inside the close. Rank 0's.
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
_st=30; [[ ! "$STOP_TIMEOUT" =~ ^[1-9][0-9]{0,5}$ ]] || _st=$STOP_TIMEOUT   # (start.sh refuses another value)
TENSORFOLD_CLOSE_MS="${TENSORFOLD_CLOSE_MS:-$(( _st * 2000 / 3 ))}"
SIDE="${SIDE:-1}"
PREFETCH="${PREFETCH:-1}"
ENGRAM_AIO="${ENGRAM_AIO:-1}"
# Engram's rows from packed shards (engram_pack.zig, off by default): each Spark keeps the tables' rows as records (a
# row's FP8 values and its scales side by side, so a row is one read instead of two), written once by
# tf-dsv41-engram-pack, with a row cache in memory and the FP8 decode on the GPU; the rows a layer reads are the same
# bytes. ENGRAM_OWNED=1: each Spark keeps and reads only its quarter of the rows (~51 GB a Spark), and one exchange a
# layer fills in the rest; 0: every Spark keeps the whole tables (~203 GB). ENGRAM_CACHE_MIB: the row cache a Spark (0:
# none). ENGRAM_INSTREAM=1: a round's wait for its rows runs as a host step on the GPU stream instead of holding the host
# (needs ENGRAM_AIO=1).
ENGRAM_PACKED="${ENGRAM_PACKED:-0}"
ENGRAM_OWNED="${ENGRAM_OWNED:-1}"
ENGRAM_CACHE_MIB="${ENGRAM_CACHE_MIB:-2048}"
ENGRAM_INSTREAM="${ENGRAM_INSTREAM:-0}"
# A decode round's ms by its rows (1 .. ROUND_ROWS), the lane core's price of a row when it chooses how many drafts
# each stream verifies (TF_DS_ROUND_MS): measured on this build: a round
# of N streams x 1 row costs 21.5, 36.0, 53.4, 83.1 ms at N = 1, 4, 8, 16 (host steps included), linear between, past
# 16 the bandwidth model's falling slope (+3.3 ms a row to 32, +2.7 to 48). The EXL3-era table this replaces (14.5 ...
# 81.7) priced rows at half their cost, so the lane over-drafted at 4-16 streams.
ROUND_MS="${ROUND_MS:-21.5,26.3,31.1,36.0,40.3,44.7,49.0,53.4,57.1,60.8,64.5,68.2,72.0,75.7,79.4,83.1,86.4,89.7,93.0,96.3,99.6,102.9,106.2,109.5,112.8,116.1,119.4,122.7,126.0,129.3,132.6,135.9,138.6,141.3,144.0,146.7,149.4,152.1,154.8,157.5,160.2,162.9,165.6,168.3,171.0,173.7,176.4,179.1}"
# A round's ms past its forward that the one-stream chooser adds to every choice of drafts (TF_DS_DRAFT_MS): with the
# table above, 25 keeps one stream's choice where the old table had it (the measured A/B: 12 and 18 lost 0.5-3 % at c1).
DRAFT_MS="${DRAFT_MS:-25}"
# The drafts a stream verifies at most (the engine's TF_DS_SERVED_K): solo (the served lane's own choice of k, from
# ROUND_MS and its draft ms, when one stream is live; more streams keep the lane core's allocation), all (that choice at
# any number of streams), off (the lane core's allocation alone). Replies are the same either way; only speed changes.
SERVED_K="${SERVED_K:-solo}"
# bertholomus's replay floor (TF_DS_REPLAY_FLOOR): with 1, a prompt chunk of at most 16 rows whose window reaches below
# the cut a request resumes from reads that window by the gather path from the cut, instead of the ring the previous
# request left. Its two classes of prompts (2,048 k + 112 .. 127 tokens, cut to 1 .. 16 rows; last chunks of 2 .. 16
# rows) change bits; nothing else does. Every rank gets the same value (the ranks' start-up settings exchange refuses a
# difference). 0 (off) keeps the engine's replies as before. An exported TF_DS_REPLAY_FLOOR is taken when REPLAY_FLOOR
# is not set; REPLAY_FLOOR wins over it.
REPLAY_FLOOR="${REPLAY_FLOOR:-${TF_DS_REPLAY_FLOOR:-0}}"

# ---------------------------------------------------------------- files
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# What the ranks read besides the checkpoint (each worker from its own copy or over NFS): the four weight files, Engram's
# token map and the kernel kit, under HF_CACHE so that one export or one copy's layout serves them all.
DATA_SUB="tensorfold-dsv41/${MODEL_REVISION:0:8}"
[[ "$ABLIT" != 1 ]] || DATA_SUB+="-ablit-${ABLIT_REVISION:0:8}"
DATA_DIR="$HF_CACHE/$DATA_SUB"
KIT_SUB="tensorfold-dsv41/kit-${KIT_REF:0:8}"
# A worker without WORKER_HF_CACHE<i> reads this Spark's HF_CACHE over NFS (one copy of ~510 GB, on spark1): a read-only
# docker volume NFS_VOLUME on each worker that prepare.sh creates. This Spark must export NFS_PATH (the path of HF_CACHE
# in its export, default /huggingface) to every worker's address on its link. NFS_SERVER1, NFS_SERVER2, NFS_SERVER3: the
# head's address each worker mounts from (default: the head's address on its link, found by scripts/nodes.sh).
NFS_PATH="${NFS_PATH:-/huggingface}"
NFS_SERVER1="${NFS_SERVER1:-}"; NFS_SERVER2="${NFS_SERVER2:-}"; NFS_SERVER3="${NFS_SERVER3:-}"
NFS_VOLUME="${NFS_VOLUME:-dsv41-hf}"
# Engram's tables (the checkpoint's shards 47 and 48, ~203 GB) on each worker's own disk: a decode round reads ~70
# random 4 KiB rows from them, which over NFS took ~10 ms each and most of the round. prepare.sh copies them there once
# (from the NFS volume); the main weights and the weight files still load over NFS. A path on the worker.
WORKER_ENGRAM_DIR="${WORKER_ENGRAM_DIR:-\$HOME/.cache/dsv41-tensorfold/engram-${MODEL_REVISION:0:8}}"
# With ENGRAM_PACKED=1 the packed shards replace that copy, on every Spark's own disk (this one's too): a path on each.
ENGRAM_PACKED_DIR="${ENGRAM_PACKED_DIR:-\$HOME/.cache/dsv41-tensorfold/engram-packed-${MODEL_REVISION:0:8}}"
packed_dir_here() { echo "${ENGRAM_PACKED_DIR//\$HOME/$HOME}"; }
# The nodes the packed shards are cut for: the 2D split's four (ENGRAM_OWNED=1), or one (the whole tables).
packed_nodes() { if [[ "$ENGRAM_OWNED" == 1 ]]; then echo 4; else echo 1; fi; }
engram_shards() { (cd "$(model_cache_dir)/snapshots/$(snapshot_rev)" && python3 -c "
import json; m = json.load(open('model.safetensors.index.json'))['weight_map']
print(' '.join(sorted({f for k, f in m.items() if '.engram.' in k})))"); }
# packed_files <node> <nodes> [<table file>...]: the shards node <node> of <nodes> reads, one a layer, each at its exact
# size, from the Engram tables' own headers (scripts/engram-verify.py plan; default: the checkpoint's Engram files):
# "<file> <bytes> <layer> <rows> <lo> <hi> <row_w> <row_s>" a line. The checkpoint's Engram files also hold other
# tensors, so their sizes over the nodes are no measure of a node's shards.
packed_files() {
  local node=$1 nodes=$2 snap f
  shift 2
  local -a tables=("$@")
  if (( ${#tables[@]} == 0 )); then
    snap="$(model_cache_dir)/snapshots/$(snapshot_rev)"
    for f in $(engram_shards); do tables+=("$snap/$f"); done
  fi
  python3 -I "$_cfg_root/scripts/engram-verify.py" plan "$node" "$nodes" "${tables[@]}"
}
STATE_DIR="${STATE_DIR:-$HOME/.local/state/dsv41-tensorfold}"   # locks, the prepared marker
LOG_DIR="${LOG_DIR:-$HOME/.cache/dsv41-tensorfold/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------- the SSD tier (every Spark)
# Kept prompt states written to a Spark's own disk when they are kept, in shared 2,048-token blocks, and read back
# instead of a new prefill, also after a restart (wojo's tier). Every Spark keeps a store of its own, on its own disk:
# the first Spark (rank 0) gets these settings by the engine's name (scripts/nodes.sh spill_env0) and each worker
# gets the same values, the store's identity and its own weight digest (spill_rank_env). An exported engine name
# (TF_DS_SPILL_GIB, ...) is taken when its setting is not set. Values are not checked here: the engine's one parser checks every one, and a bad
# one turns the tier off, named on /health (the tier never refuses a start). Every other TF_DS_SPILL_* / TF_SPILL_*
# you export reaches every rank as it is (TF_SPILL_LOAD_MS, TF_DS_SPILL_RESTORES, ...; the engine logs every one at its start).
#   SPILL_GIB          the store's cap in GiB (TF_DS_SPILL_GIB). 0: the tier off, the engine as without it
#   SPILL_FLOOR_GIB    free disk the tier always leaves on its folder's file system (TF_DS_SPILL_MIN_FREE_GIB)
#   SPILL_TURNS        old turns a conversation keeps on disk past its newest (TF_SPILL_TURNS)
#   SPILL_QUOTA        the share of the cap one conversation may hold before its own old turns go (TF_DS_SPILL_QUOTA;
#                      0: none)
#   SPILL_MIN_TOKENS   shorter kept states are not written (TF_DS_SPILL_MIN_TOKENS)
#   SPILL_STAGING_MIB  the pinned memory the tier stages its copies in (TF_DS_SPILL_STAGING_MIB)
#   SPILL_FLUSH_MS     what a stop gives the tier to write what still waits (TF_SPILL_FLUSH_MS): STOP_TIMEOUT x 500 ms
#                      (15,000 at 30 s), inside rank 0's close (TENSORFOLD_CLOSE_MS, above)
#   SPILL_DIR          the store's folder on this Spark's own disk, mounted at /spill in every rank's container (a
#                      worker mounts the same path on its own disk; default: beside HF_CACHE, never inside it).
#                      start.sh creates it as you on this Spark, and on each worker at its start, and checks this Spark's before
#                      anything is stopped: a local disk (not a network or memory file system), not inside an NFS
#                      export, no overlap with a folder of SPILL_FORBID, HF_CACHE, the weight files, Engram's, STATE_DIR
#                      or LOG_DIR, not / or your home, not under an archive folder, not root's, at least SPILL_FLOOR_GIB
#                      free (under (1 + its kept builds, 1) x SPILL_GIB + SPILL_FLOOR_GIB it only warns). A check that
#                      fails, or does not end within SPILL_CHECK_TIMEOUT seconds, starts the engine without the tier,
#                      the reason named on /health (TF_DS_SPILL_OFF_REASON): never a refused start
#   SPILL_FORBID       folders the tier's may not equal, sit inside or hold (space-separated): other TensorFold tiers'.
#                      The store deletes the folders of other builds in its root, so two tiers never share one. Default:
#                      the GLM TensorFold recipes' ($HOME/.cache/tensorfold-spill, their default; -blocks, one setup's)
#   SPILL_CHECK_TIMEOUT seconds each of start.sh's disk checks for the tier may take (the folder, the weight files'
#                      digests): a disk that hangs turns the tier off for the start instead of holding it
SPILL_GIB="${SPILL_GIB:-${TF_DS_SPILL_GIB:-0}}"
SPILL_DIR="${SPILL_DIR:-$(dirname "$HF_CACHE")/dsv41-tensorfold-spill}"
SPILL_FORBID="${SPILL_FORBID-$HOME/.cache/tensorfold-spill $HOME/.cache/tensorfold-spill-blocks}"
SPILL_CHECK_TIMEOUT="${SPILL_CHECK_TIMEOUT:-30}"
# Each Spark's weight file's SHA-256 names the store with the rest of its identity (scripts/nodes.sh): prepare.sh reads
# each Spark's own file once with O_DIRECT, all four at once (about a minute for a new 80 GB file, estimated; cached by
# the file's size, times, inode and device in that Spark's state folder), each under WEIGHTS_SHA_TIMEOUT seconds.
# start.sh only reads the caches: a rank without its digest starts the engine with the tier off, named.
WEIGHTS_SHA_TIMEOUT="${WEIGHTS_SHA_TIMEOUT:-1800}"
SPILL_FLOOR_GIB="${SPILL_FLOOR_GIB:-${TF_DS_SPILL_MIN_FREE_GIB:-50}}"
SPILL_TURNS="${SPILL_TURNS:-${TF_SPILL_TURNS:-4}}"
SPILL_QUOTA="${SPILL_QUOTA:-${TF_DS_SPILL_QUOTA:-0}}"
SPILL_MIN_TOKENS="${SPILL_MIN_TOKENS:-${TF_DS_SPILL_MIN_TOKENS:-8192}}"
SPILL_STAGING_MIB="${SPILL_STAGING_MIB:-${TF_DS_SPILL_STAGING_MIB:-128}}"
SPILL_FLUSH_MS="${SPILL_FLUSH_MS:-${TF_SPILL_FLUSH_MS:-$(( _st * 500 ))}}"
unset _st

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
# A line's tag: the setup's step when ./setup.sh runs this script (SETUP_TAG, "4/5 data"), else the script's name.
_tag() { printf '%s' "${SETUP_TAG:-$(basename "$0")}"; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(_tag)" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(_tag)" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(_tag)" "$(_c 2 0)" "$*" >&2; exit 1; }
# detail: an internal line (sizes, devices, hashes, the commands run): under ./setup.sh into its log file (SETUP_LOG)
# only, else printed like log.
detail() { if [[ -n "${SETUP_LOG:-}" ]]; then printf '[%s] %s\n' "$(_tag)" "$*" >> "$SETUP_LOG"; else log "$@"; fi; }
# The recipe's unnumbered names for the first worker's settings (a local.sh written for the recipe) still reach it, with
# a warning: this repository numbers the workers 1 to 3, and a numbered name wins.
for _n in WORKER FABRIC_PEER WORKER_HF_CACHE NFS_SERVER; do
  [[ -n "${!_n:-}" ]] || continue
  _m="${_n}1"
  [[ -n "${!_m:-}" ]] || printf -v "$_m" '%s' "${!_n}"
  warn "$_n is the recipe's name: this repository numbers the workers, so it is $_m (rename it in scripts/local.sh)"
done
unset _n _m
# api_header_file: with API_KEY_FILE, STATE_DIR/api-header holding "Authorization: Bearer <the file's first key>" (mode
# 600, rewritten each call) for `curl -H @file`, so the key never shows on a command line; prints its path. Without
# API_KEY_FILE it prints nothing; it fails when the file holds no key. (A line is "label: key" or a bare key; the
# label ends at the first colon, as the engine reads it.)
api_header_file() {
  [[ -n "$API_KEY_FILE" ]] || return 0
  local line key=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    line=${line#"${line%%[![:space:]]*}"}; line=${line%"${line##*[![:space:]]}"}
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" == *:* ]]; then key=${line#*:}; else key=$line; fi
    key=${key#"${key%%[![:space:]]*}"}
    break
  done < "$API_KEY_FILE"
  [[ -n "$key" ]] || return 1
  mkdir -p "$STATE_DIR"
  (umask 077; printf 'Authorization: Bearer %s\n' "$key" > "$STATE_DIR/api-header") || return 1
  echo "$STATE_DIR/api-header"
}

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
# The snapshot this setup serves: the pin, else what refs/main names here.
snapshot_rev() { local rev=$MODEL_REVISION; [[ -n "$rev" ]] || rev=$(cat "$(model_cache_dir)/refs/main" 2>/dev/null); echo "$rev"; }
# The checkpoint as the containers see it (HF_CACHE mounted at /hf on every Spark). ABLIT=1: the overlaid view
# prepare.sh builds at $DATA_SUB/checkpoint (every stock file linked, plus the Keys sidecar).
model_in_container() {
  if [[ "$ABLIT" == 1 ]]; then echo "/hf/$DATA_SUB/checkpoint"
  else echo "/hf/hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"; fi
}
# The folder rank 0 serves (prepare.sh): the checkpoint's files linked, with the recipe's chat template.
served_in_container() { echo "/hf/$DATA_SUB/model"; }

# What scripts/prepare.sh last left ready (start.sh runs it again when this line changes): with where the checkpoint is
# (HF_CACHE), how the workers mount it (NFS_PATH, NFS_VOLUME, NFS_SERVER*) and the chat template it copies into the
# served folder, each of which needs prepare.sh's steps again when it changes. Each field is shell-quoted (%q, in the C
# locale so the line does not depend on it), so no two settings give the same line.
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local i tpl f
  tpl=$(sha256sum "$_cfg_root/model/chat_template.jinja" | cut -c1-64) && [[ -n "$tpl" ]] ||
    die "cannot hash model/chat_template.jinja (sha256sum)"
  f=("model=$MODEL_ID@$(snapshot_rev)" "image=$(image_ident "$IMAGE")" "kit=$KIT_REF" "order=$ORDER" "engram=$WORKER_ENGRAM_DIR" "ablit=$ABLIT")
  [[ "$ENGRAM_PACKED" != 1 ]] || f+=("packed=$ENGRAM_PACKED_DIR/$(packed_nodes)")
  f+=("hf=$HF_CACHE" "nfs_path=$NFS_PATH" "nfs_volume=$NFS_VOLUME" "template=$tpl")
  for i in 1 2 3; do f+=("w$i=$(worker_host "$i")" "nfs$i=$(wval NFS_SERVER "$i")" "hf$i=$(wval WORKER_HF_CACHE "$i")"); done
  (export LC_ALL=C; printf '%q ' "${f[@]}"; echo)   # (a subshell: the caller's locale is left as it was)
}
# prepare.sh's last step: computed first and swapped in whole, so a failure leaves the previous marker as it was
write_prepared_marker() {
  local state
  state=$(prepared_state) || return 1
  printf '%s\n' "$state" > "$PREPARED_MARKER.new" && mv -f "$PREPARED_MARKER.new" "$PREPARED_MARKER"
}
