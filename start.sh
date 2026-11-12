#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 (deepseek-ai's own FP8 / MXFP4 release) on four DGX Sparks on a switch with TensorFold's
# native engine, end to end: runs scripts/prepare.sh when the image, the checkpoint, the weight files, the kit or the workers'
# copies or NFS volumes are not ready yet (the first run, or after the patches change), starts ranks 3, 2 and 1 on the
# workers and rank 0 here, which serves the API on port 8888, waits until the OpenAI API answers, then runs a smoke test.
# A worker reads its own copy of what its rank reads (WORKER_HF_CACHE*), else this Spark's over NFS. Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                  # scripts/config.sh's defaults (16 requests at once, a 1,048,576-token window, drafts,
#                               # images); if the server already runs, says so and leaves it alone
#   ./start.sh restart          # stop every rank (./stop.sh), then start again (e.g. after changing settings)
#   CONTEXT=262144 ./start.sh restart
#   DRY_RUN=1 ./start.sh        # print the links found and every rank's docker command, change nothing
# Extra arguments go to rank 0's `tensorfold-native serve` after the defaults, so they win.
# Setup: WORKER1, WORKER2, WORKER3 (ssh targets, key-based) in scripts/local.sh.
# Settings, from the environment, scripts/local.sh or ./.env (defaults and their notes in scripts/config.sh):
#   serving  GPU_CLOCK, DROP_CACHES, CONTEXT, POOL, PARALLEL, ROUND_ROWS, MAX_TOKENS, THINKING, REASONING_EFFORT, TEMPERATURE, TOP_P, TOP_K, DRAFTS, SERVED_NAME,
#            HOST, PORT, API_KEY_FILE, ADMIN_KEY_FILE, ENGINE_CFG, METRICS_OPEN, RDMA, GRAPHS, SIDE, PREFETCH, ENGRAM_AIO, ENGRAM_PACKED, ENGRAM_OWNED,
#            ENGRAM_CACHE_MIB, ENGRAM_INSTREAM, ROUND_MS, DRAFT_MS, SERVED_K, REPLAY_FLOOR
#   nodes    WORKER1, WORKER2, WORKER3, WORKER_HF_CACHE*, FABRIC_PEER*, MASTER_ADDR, MASTER_PORT, ORDER,
#            RAIL_CHECK, RAIL_COUNT
#   files    DERISKED, MODEL_ID, MODEL_REVISION, HF_CACHE, NFS_PATH, NFS_SERVER*, NFS_VOLUME, STATE_DIR, LOG_DIR, LOG_KEEP
#   engine   KIT_REPO, KIT_REF, ZIG_VERSION, BASE_IMAGE, IMAGE, CONTAINER_NAME (the engine is this repository)
#   SSD tier SPILL_GIB (0: off), SPILL_DIR, SPILL_FLOOR_GIB, SPILL_TURNS, SPILL_QUOTA, SPILL_MIN_TOKENS, SPILL_STAGING_MIB,
#            SPILL_FLUSH_MS, SPILL_FORBID, SPILL_CHECK_TIMEOUT: every Spark keeps a store of its own, and every rank gets these
#            and every TF_DS_SPILL_* and TF_SPILL_* variable you set (this Spark's folder is checked before anything is stopped; a failed
#            check starts without the tier)
#   stop     STOP_TIMEOUT, TENSORFOLD_CLOSE_MS (rank 0's orderly close), DRAIN_TIMEOUT
#   setup    PREPARE (auto | 1 | 0), WAIT_TIMEOUT (seconds, default 2400), DRY_RUN=1
#   ranks    every other TENSORFOLD_* and TF_DS_* variable you set goes to every rank
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
# GPU_CLOCK (MHz): every Spark's GPU clock limit for this start (scripts/served.sh: 2200, as the numbers ran); empty: unchanged
GPU_CLOCK="${GPU_CLOCK:-}"
[[ -z "$GPU_CLOCK" || "$GPU_CLOCK" =~ ^[0-9]+$ ]] || { echo "GPU_CLOCK must be a clock in MHz (e.g. 2200) or empty, not '$GPU_CLOCK'" >&2; exit 1; }
# DROP_CACHES (1, as every published number's start): every Spark's page cache dropped before the ranks start; 0 leaves it
DROP_CACHES="${DROP_CACHES:-1}"
[[ "$DROP_CACHES" =~ ^[01]$ ]] || { echo "DROP_CACHES must be 1 (drop every Spark's page cache before the ranks start) or 0, not '$DROP_CACHES'" >&2; exit 1; }
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help|-h|--help) usage; exit 0 ;;
esac
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1
for v in CONTEXT POOL PARALLEL ROUND_ROWS MAX_TOKENS; do [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v is a positive count, not ${!v}"; done
# the window the kit's RoPE tables reach: 65,536 x TF_DS_YARN_FACTOR (16, the checkpoint's own, is 1,048,576; the engine refuses a factor its kit was not cut for)
YARN="${TF_DS_YARN_FACTOR:-16}"
[[ "$YARN" =~ ^[1-9][0-9]*$ ]] || die "TF_DS_YARN_FACTOR is a whole number (16 is the checkpoint's own), not $YARN"
(( CONTEXT <= 65536 * YARN )) || die "CONTEXT is at most $((65536 * YARN)) (the model's window at YaRN factor $YARN), not $CONTEXT"
# TF_DS_KV_FREE_TARGET_GIB (the engine sizes the pool from the free memory each Spark keeps, voted over the ranks, never below
# CONTEXT) replaces POOL: then no rank is given a fixed pool
KV_TARGET="${TF_DS_KV_FREE_TARGET_GIB:-}"
if [[ -n "$KV_TARGET" ]]; then
  [[ "$KV_TARGET" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "TF_DS_KV_FREE_TARGET_GIB is the GiB each Spark keeps free, a number, not $KV_TARGET"
else
  (( POOL >= CONTEXT )) || die "POOL ($POOL) holds every stream's positions together: at least CONTEXT ($CONTEXT)"
fi
(( PARALLEL <= 16 )) || die "PARALLEL is 1 to 16, not $PARALLEL"
[[ "$SERVED_K" =~ ^(solo|all|off)$ ]] || die "SERVED_K is solo, all or off, not $SERVED_K"
for v in THINKING DRAFTS METRICS_OPEN RDMA GRAPHS SIDE PREFETCH ENGRAM_AIO ENGRAM_PACKED ENGRAM_OWNED ENGRAM_INSTREAM REPLAY_FLOOR; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$GRAPH_CAP" =~ ^[0-9]+$ ]] || die "GRAPH_CAP is a count of graphs (0: all), not $GRAPH_CAP"
[[ -z "$PROMPT_CACHE_GIB" || "$PROMPT_CACHE_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "PROMPT_CACHE_GIB is GiB (empty: the engine's 0.5), not $PROMPT_CACHE_GIB"
[[ "$ENGRAM_CACHE_MIB" =~ ^[0-9]+$ ]] || die "ENGRAM_CACHE_MIB is a count of MiB, not $ENGRAM_CACHE_MIB"
[[ "$ENGRAM_INSTREAM" == 0 || ( "$ENGRAM_PACKED" == 1 && "$ENGRAM_AIO" == 1 ) ]] || die "ENGRAM_INSTREAM=1 needs ENGRAM_PACKED=1 and ENGRAM_AIO=1"
[[ "$STOP_TIMEOUT" =~ ^[1-9][0-9]{0,5}$ ]] || die "STOP_TIMEOUT is a positive count of seconds (each rank's docker stop -t), not $STOP_TIMEOUT"
[[ "$TENSORFOLD_CLOSE_MS" =~ ^[0-9]{1,9}$ ]] || die "TENSORFOLD_CLOSE_MS is milliseconds (rank 0's orderly close at a stop), not $TENSORFOLD_CLOSE_MS"
# the SSD tier's settings reach rank 0 as settings, never as its extra serve arguments (refused before anything stops)
for a in "$@"; do
  [[ "$a" != --spill-* ]] || die "rank 0's extra argument $a: the SSD tier's settings are SPILL_GIB, SPILL_DIR, SPILL_FLOOR_GIB, SPILL_TURNS, SPILL_QUOTA, SPILL_MIN_TOKENS, SPILL_STAGING_MIB and SPILL_FLUSH_MS (scripts/config.sh) in scripts/local.sh or the environment, and every TF_DS_SPILL_* / TF_SPILL_* you export: they reach every rank's engine, by its own names. Nothing was stopped"
done
if [[ -n "$API_KEY_FILE" ]]; then
  [[ -f "$API_KEY_FILE" && -r "$API_KEY_FILE" ]] || die "API_KEY_FILE $API_KEY_FILE is not a readable file"
  if (( 8#$(stat -c %a "$API_KEY_FILE") & 8#044 )); then die "API_KEY_FILE $API_KEY_FILE can be read by others: chmod 600 it (the engine refuses it otherwise)"; fi
fi
if [[ -n "$ENGINE_CFG" ]]; then
  [[ -f "$ENGINE_CFG" && -r "$ENGINE_CFG" ]] || die "ENGINE_CFG $ENGINE_CFG is not a readable file"
  [[ ! -L "$ENGINE_CFG" ]] || die "ENGINE_CFG $ENGINE_CFG is a symbolic link: name the file itself"
  (( 8#$(stat -c %a "$ENGINE_CFG") & 8#066 )) && die "ENGINE_CFG $ENGINE_CFG can be read or written by others: chmod 600 it"
  [[ $(stat -c %u "$ENGINE_CFG") == $(id -u) ]] || die "ENGINE_CFG $ENGINE_CFG is not yours: rank 0 runs it as the setup's own file"
  CFG_DIR=$(cd "$(dirname "$ENGINE_CFG")" && pwd -P); CFG_NAME=$(basename "$ENGINE_CFG")
  [[ $(stat -c %u "$CFG_DIR") == $(id -u) ]] && ! (( 8#$(stat -c %a "$CFG_DIR") & 8#022 )) || die "ENGINE_CFG's folder $CFG_DIR must be yours and writable by you alone (rank 0 mounts it writable)"
  [[ ! -L "$CFG_DIR/engine.history.jsonl" ]] || die "ENGINE_CFG's folder holds engine.history.jsonl as a symbolic link: remove it"
  # rank 0 mounts this folder writable (engine.history.jsonl lands beside the file): it must be a folder of the file's own
  [[ "$CFG_DIR" != "$HOME" && "$CFG_DIR" != / ]] || die "ENGINE_CFG $ENGINE_CFG: keep it in a folder of its own (rank 0 mounts that folder writable), not $CFG_DIR"
  other=$(find "$CFG_DIR" -mindepth 1 -maxdepth 1 ! -name "$CFG_NAME" ! -name engine.history.jsonl ! -name 'engine.history.jsonl.*' -printf '%f ' | head -c 200)
  [[ -z "$other" ]] || die "ENGINE_CFG $ENGINE_CFG: its folder $CFG_DIR holds more than the file and its history ($other): rank 0 mounts that folder writable, so keep engine.cfg in a folder of its own"
fi
if [[ -n "$ADMIN_POLICY_FILE" ]]; then
  [[ -f "$ADMIN_POLICY_FILE" && -r "$ADMIN_POLICY_FILE" ]] || die "ADMIN_POLICY_FILE $ADMIN_POLICY_FILE is not a readable file"
  (( 8#$(stat -c %a "$ADMIN_POLICY_FILE") & 8#022 )) && die "ADMIN_POLICY_FILE $ADMIN_POLICY_FILE can be written by others: chmod 600 or 644 it"
fi
if [[ -n "$ADMIN_KEY_FILE" ]]; then
  [[ -f "$ADMIN_KEY_FILE" && -r "$ADMIN_KEY_FILE" ]] || die "ADMIN_KEY_FILE $ADMIN_KEY_FILE is not a readable file"
  if (( 8#$(stat -c %a "$ADMIN_KEY_FILE") & 8#044 )); then die "ADMIN_KEY_FILE $ADMIN_KEY_FILE can be read by others: chmod 600 it (the engine refuses it otherwise)"; fi
fi
check_workers

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
STEPS=5
# Under ./setup.sh (its step 5) a few plain lines reach the screen (show) and every other line goes to the setup's log
# file (log is its detail there); on its own, this start prints its banner, its five steps and its lines as they come.
if [[ -n "${SETUP_TAG:-}" && -n "${SETUP_LOG:-}" ]]; then
  UNDER_SETUP=1
  eval "show() $(declare -f log | tail -n +2)"
  log() { detail "$@"; }
  step() { detail "[$1/$STEPS] $2"; }
else
  UNDER_SETUP=0
  show() { :; }
  echo
  banner
  printf '\n%s  Starting the engine%s\n' "$M" "$R"
  printf '%s  %s · 4 x DGX Spark · %s at once · %s-token window · native FP8 / MXFP4 · port %s%s\n\n' "$D" "$MODEL_ID" "$PARALLEL" "$CONTEXT" "$PORT" "$R"
  step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }
fi

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_workers
(( DRY )) && log "DRY_RUN=1: printing the docker commands; nothing is stopped, started or prepared"
# with a key file the API answers only with a key: the checks below send the file's first one (never on a command line)
auth=()
if [[ -n "$API_KEY_FILE" ]]; then hdr=$(api_header_file) || die "API_KEY_FILE $API_KEY_FILE holds no key"; auth=(-H "@$hdr"); fi

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
URL="http://$API_HOST:$PORT"
running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker "$1" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
all_running() { local i; running_here || return 1; for i in 1 2 3; do running_worker "$i" || return 1; done; }
if (( ! DRY )) && [[ "$MODE" == start ]] && all_running; then
  log "$CONTAINER_NAME is already running on all 4 Sparks (port $PORT): nothing to do. ./start.sh restart restarts it, ./stop.sh stops it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
step 1 "Setup: image, checkpoint, weight files and kit on spark1; the workers' copies (or NFS volumes)"
need_prepare=0
case "${PREPARE:-auto}" in
  1) need_prepare=1 ;;
  0) ;;
  *) # (computed on its own: a failure inside $(...) would otherwise read as an empty line equal to an empty marker)
     state=$(prepared_state) || die "cannot tell whether the setup is ready: see the error above"
     [[ "$state" == "$(cat "$PREPARED_MARKER" 2>/dev/null)" ]] || need_prepare=1 ;;
esac
if (( need_prepare )); then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now (not ready yet)"
  else
    log "Not ready yet: running scripts/prepare.sh (the first time this builds the image, downloads ~510 GB and writes four ~80 GB weight files)"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE, $MODEL_ID @ $(snapshot_rev | cut -c1-8), its weight files and kit"
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || (( DRY )) || die "image $IMAGE missing: scripts/prepare.sh did not build it"
MODEL_ARG=$(model_in_container)

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: links, previous server, port, memory"
detect_links
NODE_ORDER=$(node_order)
log "Rendezvous $MASTER_ADDR:$MASTER_PORT; 2D nodes by rank: $NODE_ORDER"
for r in 0 1 2 3; do
  log "  rank $r ($( (( r == 0 )) && echo this node || worker_host "$r")): RoCE ${NODE_HCAS[r]} (GID ${NODE_GID[r]:-per device})$( (( r == 0 )) || { [[ -n "$(worker_hf_cache "$r")" ]] && echo ", its own copy at $(worker_hf_cache "$r")" || echo ", NFS from ${LINK_HEAD_ADDR[r]:-?}"; })"
done
# the SSD tier (a store on every Spark): whether it runs this start, decided before anything is stopped (scripts/nodes.sh)
spill_start
if (( ! DRY )); then
  if running_here || running_worker 1 || running_worker 2 || running_worker 3; then show "  stopping the engine that was running"; ./stop.sh; fi
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  for i in 1 2 3; do worker "$i" "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1; true"; done
  if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"; fi
fi
# every Spark's page cache dropped once the old ranks are gone, before the memory check: on a GB10 the GPU's memory is
# the same RAM, and a cache a copy or a weight-file bake left full leaves the engine that much less. Then the kernel
# compacts the memory (/proc/sys/vm/compact_memory: free memory in contiguous blocks): memory a data copy left in small
# pieces gave a rank too few large pages, and that engine (measured) answered every prompt ~0.9 s late until a restart,
# which frees the old ranks' memory in large blocks. A Spark with only the setup's sudo rule does it through the setup's
# helper (/usr/local/sbin/dsv41-root drop); a Spark that refuses both is named and the start goes on.
DROP="{ sudo -n /usr/local/sbin/dsv41-root drop 2>/dev/null || sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches && if [ -e /proc/sys/vm/compact_memory ]; then echo 1 > /proc/sys/vm/compact_memory; fi'; } && awk '/^Cached:/ {printf \"%.1f\", \$2 / 1048576}' /proc/meminfo"
if [[ "$DROP_CACHES" != 1 ]]; then log "DROP_CACHES=0: every Spark's page cache and memory are left as they are"
elif (( DRY )); then printf '[dry-run] every Spark: %s\n' "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory (sudo -n)"
else
  cached=""
  if c=$(bash -c "$DROP" 2>/dev/null); then cached="$c GiB here"
  else warn "could not drop this Spark's page cache (needs passwordless sudo or the setup's sudo rule: ./setup.sh access): the engine gets what is free"; fi
  for i in 1 2 3; do
    if c=$(worker "$i" "$DROP" 2>/dev/null); then cached+="${cached:+, }$c GiB on $(worker_host "$i")"
    else warn "could not drop $(worker_host "$i")'s page cache (needs passwordless sudo or the setup's sudo rule: ./setup.sh access): the engine gets what is free"; fi
  done
  log "Page caches dropped and memory compacted (cached now: ${cached:-none read})"
  show "  page caches dropped and memory compacted on all four Sparks"
fi
if (( ! DRY )); then
  here_gb=$(free -g | awk '/^Mem:/ {print $7}')
  (( here_gb >= 100 )) || warn "only ${here_gb} GiB memory available here: the weights take ~75 GiB a Spark, the caches the rest"
  for i in 1 2 3; do
    g=$(worker "$i" "free -g | awk '/^Mem:/ {print \$7}'" 2>/dev/null || echo 0)
    (( g >= 100 )) || warn "only ${g} GiB memory available on $(worker_host "$i"): stop other GPU work there"
  done
  show "  checks: fabric links, port $PORT and memory: OK"
fi
# the GPU clock limit on every Spark before any rank starts (the driver keeps it until a reboot), through the setup's
# helper on a Spark with only its sudo rule; a Spark that refuses both is named and serves unlimited
if [[ -n "$GPU_CLOCK" ]]; then
  clk="sudo -n /usr/local/sbin/dsv41-root clock $GPU_CLOCK 2>/dev/null || sudo -n nvidia-smi -lgc 0,$GPU_CLOCK"
  if (( DRY )); then printf '[dry-run] every Spark: %s\n' "$clk"
  else
    limited=()
    if bash -c "$clk" >/dev/null 2>&1; then limited+=("this one")
    else warn "could not limit this Spark's GPU clock to $GPU_CLOCK MHz (needs passwordless sudo or the setup's sudo rule: ./setup.sh access): it runs unlimited"; fi
    for i in 1 2 3; do
      if worker "$i" "$clk" >/dev/null 2>&1; then limited+=("$(worker_host "$i")")
      else warn "could not limit $(worker_host "$i")'s GPU clock to $GPU_CLOCK MHz (needs passwordless sudo or the setup's sudo rule: ./setup.sh access): it runs unlimited"; fi
    done
    if (( ${#limited[@]} == 4 )); then log "GPU clock limited to $GPU_CLOCK MHz on all 4 Sparks"; show "  GPU clock limited to $GPU_CLOCK MHz on all four Sparks"
    elif (( ${#limited[@]} )); then log "GPU clock limited to $GPU_CLOCK MHz on ${#limited[@]} of 4 Sparks ($(IFS=,; echo "${limited[*]}" | sed 's/,/, /g'))"
      show "  GPU clock limited to $GPU_CLOCK MHz on ${#limited[@]} of the four Sparks"
    fi
  fi
fi

# ---------------------------------------------------------------- 3. launch
# What every rank's engine reads (HF_CACHE at /hf: this Spark's own; a worker's own copy, or this Spark's over NFS). Never
# the SSD tier's settings (TF_DS_SPILL_*, TF_SPILL_*: sent separately, below), its identity (derived: rank 0's names, each
# rank's own TF_DS_RANK_CACHE_SHA) nor rank 0's TENSORFOLD_CLOSE_MS.
# (REPLAY_FLOOR: every rank the same, as the ranks' start-up settings exchange requires)
ENGINE_ENV=(TF_DS_2D_GU=parity "TF_DS_2D_ORDER=$NODE_ORDER" "TF_DS_ROUND_ROWS=$ROUND_ROWS" "TF_DS_STREAMS=$PARALLEL" "TF_DS_SERVED_K=$SERVED_K"
            "TF_DS_WINDOW=$POOL" "TF_DS_GRAPHS=$GRAPHS" "TF_DS_HC_SIDE=$SIDE" "TF_DS_L2_PREFETCH=$PREFETCH" "TF_DS_ENGRAM_AIO=$ENGRAM_AIO"
            "TF_DS_ROUND_MS=$ROUND_MS" "TF_DS_DRAFT_MS=$DRAFT_MS" "TF_DS_REPLAY_FLOOR=$REPLAY_FLOOR" "HF_HUB_OFFLINE=$HF_HUB_OFFLINE")
if [[ -n "$KV_TARGET" ]]; then keep=(); for x in "${ENGINE_ENV[@]}"; do [[ $x == TF_DS_WINDOW=* ]] || keep+=("$x"); done; ENGINE_ENV=("${keep[@]}"); fi
# the graph cap with graphs only: rank 0 reads TF_DS_GRAPH_CAP, the followers take --graph-cap (below)
[[ "$GRAPHS" != 1 ]] || ENGINE_ENV+=("TF_DS_GRAPH_CAP=$GRAPH_CAP")
# the driver's queue count, when local.sh sets it (TF_DS_OVERLAP needs 16 on every rank: read at context creation)
[[ -z "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]] || ENGINE_ENV+=("CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS")
# Engram's packed shards (ENGRAM_PACKED=1, engram_pack.zig): every rank reads its own at /engram-packed
[[ "$ENGRAM_PACKED" != 1 ]] || ENGINE_ENV+=(TF_DS_ENGRAM_PACKED=/engram-packed "TF_DS_ENGRAM_OWNED=$ENGRAM_OWNED"
                                            "TF_DS_ENGRAM_CACHE_MIB=$ENGRAM_CACHE_MIB" "TF_DS_ENGRAM_INSTREAM=$ENGRAM_INSTREAM")
while IFS='=' read -r name _; do ENGINE_ENV+=("$name=${!name}"); done < <(env | grep -E '^(TENSORFOLD|TF_DS|TF_SSE)_[A-Z0-9_]+=' | grep -vE '^(TF_DS_(2D_GU|2D_ORDER|ROUND_ROWS|STREAMS|SERVED_K|WINDOW|GRAPHS|HC_SIDE|L2_PREFETCH|ENGRAM_AIO|ENGRAM_PACKED|ENGRAM_OWNED|ENGRAM_CACHE_MIB|ENGRAM_INSTREAM|ROUND_MS|DRAFT_MS|GRAPH_CAP|REPLAY_FLOOR|SPILL_[A-Z0-9_]+|ENGINE_ID|MODEL_REVISION|ABLATION_ID|ENGRAM_REVISION|KIT_ID|RANK_CACHE_SHA)|TENSORFOLD_CLOSE_MS|TENSORFOLD_ADMIN_KEY_FILE|TENSORFOLD_ADMIN_POLICY)=' || true)
envs() { local x; for x in "$@"; do printf -- '-e %q ' "$x"; done; }
RUN_ARGS="--gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband"
DATA=/hf/$DATA_SUB
KIT=/hf/$KIT_SUB/kit
# the followers take the round's settings as flags only (tf-dsv41-lanes reads no TF_DS_GRAPHS & co.): without them they
# run every round ungraphed, with no side stream, prefetch or AIO, and rank 0 waits on them each round
FOLLOW="$([[ -n "$KV_TARGET" ]] || echo "--pool $POOL ")--context $CONTEXT --engram $MODEL_ARG --token-map $DATA/dsv41_token_map.json --drafts $DRAFTS --round-rows $ROUND_ROWS --streams $PARALLEL"
FOLLOW+=" --graphs $GRAPHS --side $SIDE --prefetch $PREFETCH --aio $ENGRAM_AIO"
[[ "$GRAPHS" != 1 ]] || FOLLOW+=" --graph-cap $GRAPH_CAP"
step 3 "Launch: container $CONTAINER_NAME, ranks 3 to 1 on the workers, then rank 0 here"
# the SSD tier's environment before the workers start: with the tier on, each worker keeps a store of its own and reads the same settings
spill_env0
# From here until the server answers, a start that fails stops every container it launched, on every Spark (rank 0 included): a rank left running
# holds its GPU memory (75 GB) with no API, and ./stop.sh is the one place that knows how to remove it and keep its log.
STARTED=0; LOGS_PID=""
on_exit() {
  local rc=$?
  trap - EXIT
  [[ -z "$LOGS_PID" ]] || kill "$LOGS_PID" 2>/dev/null || true
  if (( rc != 0 && STARTED )); then
    warn "the start failed (exit $rc): stopping every container it launched, on every Spark"
    DRAIN_TIMEOUT=0 ./stop.sh || warn "./stop.sh could not stop every rank: docker ps on each Spark says what is left"
  fi
  exit "$rc"
}
trap on_exit EXIT
(( DRY )) || STARTED=1
pids=()
for i in 3 2 1; do
  # Engram's tables from the worker's own disk (prepare.sh), the rest over NFS; with packed shards the rows come from
  # the shards on the worker's disk and the tables' headers from its own copy of the tables when it has its own copy of
  # the model (WORKER_HF_CACHE<i>: no NFS, and that copy holds no checkpoint shard), else over NFS
  if [[ "$ENGRAM_PACKED" == 1 && -n "$(worker_hf_cache "$i")" ]]; then
    cmd="tf-dsv41-lanes $MODEL_ARG $DATA/rank-cache $i 4 ${LINK_HEAD_ADDR[i]:-$MASTER_ADDR} $MASTER_PORT $KIT ${FOLLOW/--engram $MODEL_ARG/--engram /engram}"
    engram_mount="-v $WORKER_ENGRAM_DIR:/engram:ro -v $ENGRAM_PACKED_DIR:/engram-packed:ro"
  elif [[ "$ENGRAM_PACKED" == 1 ]]; then
    cmd="tf-dsv41-lanes $MODEL_ARG $DATA/rank-cache $i 4 ${LINK_HEAD_ADDR[i]:-$MASTER_ADDR} $MASTER_PORT $KIT $FOLLOW"
    engram_mount="-v $ENGRAM_PACKED_DIR:/engram-packed:ro"
  else
    cmd="tf-dsv41-lanes $MODEL_ARG $DATA/rank-cache $i 4 ${LINK_HEAD_ADDR[i]:-$MASTER_ADDR} $MASTER_PORT $KIT ${FOLLOW/--engram $MODEL_ARG/--engram /engram}"
    engram_mount="-v $WORKER_ENGRAM_DIR:/engram:ro"
  fi
  [[ "$RDMA" == 0 ]] || cmd+=" --rdma ${NODE_HCAS[i]}"   # this node's RoCE devices
  # (with the SSD tier asked, the rank's own weight file's digest: rank 0 names the store from all four)
  cfg_mount=""
  if [[ -n "$ENGINE_CFG" ]]; then   # the worker's private copy of engine.cfg (sent on ssh's input, never on a command line), read-only inside
    if (( ! DRY )); then worker "$i" "umask 077 && mkdir -p \$HOME/.cache/dsv41-tensorfold/cfg && cat > \$HOME/.cache/dsv41-tensorfold/cfg/$CFG_NAME" < "$ENGINE_CFG" || die "could not copy ENGINE_CFG to $(worker_host "$i")"; fi
    cfg_mount=" -v \$HOME/.cache/dsv41-tensorfold/cfg:/run/tensorfold/cfg:ro -e TF_ENGINE_CFG=/run/tensorfold/cfg/$CFG_NAME"
  fi
  remote="mkdir -p \$HOME/.cache/dsv41-tensorfold$(spill_rank_mkdir) && docker run -d --name $CONTAINER_NAME $RUN_ARGS $(hf_mount "$i") $engram_mount$(spill_rank_mount)$cfg_mount $(envs "${ENGINE_ENV[@]}") $(spill_rank_env "$i") $IMAGE $cmd"
  if (( DRY )); then printf '[dry-run] rank %s on %s:\n  %s\n' "$i" "$(worker_host "$i")" "$remote"; continue; fi
  log "Rank $i on $(worker_host "$i"): $cmd"
  worker "$i" "$remote" >/dev/null & pids[i]=$!
done
# (the three at once; rank 0 only once all three run)
for i in 3 2 1; do [[ -z "${pids[i]:-}" ]] || wait "${pids[i]}" || die "could not start rank $i on $(worker_host "$i")"; done
# rank 0 listens for each worker at the address that worker was given (its link's address of this node)
LISTEN="${LINK_HEAD_ADDR[1]:-$MASTER_ADDR},${LINK_HEAD_ADDR[2]:-$MASTER_ADDR},${LINK_HEAD_ADDR[3]:-$MASTER_ADDR}"
ENV0=(TF_TP_WORLD=4 "TF_TP_MASTER=$MASTER_ADDR" "TF_TP_LISTEN=$LISTEN" "TF_TP_PORT=$MASTER_PORT" "TF_DS_RANK_CACHE=$DATA/rank-cache" "TF_DS_KIT=$KIT"
      "TENSORFOLD_CUDA_KERNELS=$KIT/aot" "TF_DS_ENGRAM=$MODEL_ARG" "TF_DS_TOKEN_MAP=$DATA/dsv41_token_map.json"
      "TENSORFOLD_CLOSE_MS=$TENSORFOLD_CLOSE_MS")
# the SSD tier's settings for rank 0 (the workers' are in their docker lines: spill_rank_env); scripts/nodes.sh spill_env0, called above; TF_DS_SPILL_GIB=0 when it is off
ENV0+=("${SPILL_ENV0[@]}")
SERVE=(tensorfold-native serve "$(served_in_container)" --backend cuda --host "$HOST" --port "$PORT" --name "$SERVED_NAME"
       --context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE+=(--thinking); else SERVE+=(--no-thinking); fi
[[ -z "$REASONING_EFFORT" ]] || SERVE+=(--reasoning-effort "$REASONING_EFFORT")
# drafts off on rank 0 as on the followers (a follower without the drafter refuses a drafted round)
[[ "$DRAFTS" == 1 ]] || SERVE+=(--no-drafts)
[[ -z "$PROMPT_CACHE_GIB" ]] || SERVE+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
# the key file, mounted where rank 0 reads it (the engine rereads it when it changes)
if [[ -n "$API_KEY_FILE" ]]; then SERVE+=(--api-key-file /run/tensorfold/api-keys); [[ "$METRICS_OPEN" != 1 ]] || SERVE+=(--metrics-open); fi
SERVE+=("$@")
here=(docker run -d --name "$CONTAINER_NAME" $RUN_ARGS -v "$HF_CACHE:/hf:ro")
[[ -z "$API_KEY_FILE" ]] || here+=(-v "$API_KEY_FILE:/run/tensorfold/api-keys:ro")
[[ -z "$ADMIN_KEY_FILE" ]] || { here+=(-v "$ADMIN_KEY_FILE:/run/tensorfold/admin-keys:ro"); ENV0+=("TENSORFOLD_ADMIN_KEY_FILE=/run/tensorfold/admin-keys"); }
[[ -z "$ADMIN_POLICY_FILE" ]] || { here+=(-v "$ADMIN_POLICY_FILE:/run/tensorfold/admin-policy.json:ro"); ENV0+=("TENSORFOLD_ADMIN_POLICY=/run/tensorfold/admin-policy.json"); }
[[ -z "$ENGINE_CFG" ]] || { here+=(-v "$CFG_DIR:/run/tensorfold/cfg"); ENV0+=("TF_ENGINE_CFG=/run/tensorfold/cfg/$CFG_NAME"); }
# the SSD tier's folder on this Spark (spill_check: checked, created as you); each worker mounts its own at the same path
[[ -z "$SPILL_REAL" ]] || here+=(-v "$SPILL_REAL:$SPILL_MOUNT")
[[ "$ENGRAM_PACKED" != 1 ]] || here+=(-v "$(packed_dir_here):/engram-packed:ro")
[[ "$RDMA" == 0 ]] || ENV0+=("TF_RDMA_DEVICES=${NODE_HCAS[0]}")
for x in "${ENV0[@]}" "${ENGINE_ENV[@]}"; do here+=(-e "$x"); done
here+=("$IMAGE" "${SERVE[@]}")
if (( DRY )); then printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${here[@]}")"; exit 0; fi
log "Rank 0 here: ${SERVE[*]}"
"${here[@]}" >/dev/null
show "  engine started on all four Sparks; loading the weights (about 1 min)"

# ---------------------------------------------------------------- 4. load
step 4 "Loading: ~75 GiB of weights on each Spark, then the caches (a few minutes)"
# neither log follower may keep the launcher's lock after it exits
if (( UNDER_SETUP )); then   # the engine's own log: the setup's log file
  docker logs -f "$CONTAINER_NAME" 8>&- >> "$SETUP_LOG" 2>&1 &
else
  docker logs -f "$CONTAINER_NAME" 8>&- 2>&1 | sed -u "s/^/  ${D}│${R} /" 8>&- &
fi
LOGS_PID=$!
fail() {
  kill $LOGS_PID 2>/dev/null || true
  if (( UNDER_SETUP )); then printf '%s── this Spark: last log lines ──%s\n' "$D" "$R"; docker logs --tail 15 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'; fi
  for i in 1 2 3; do printf '%s── rank %s (%s): last log lines ──%s\n' "$D" "$i" "$(worker_host "$i")" "$R"; worker "$i" docker logs --tail 15 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'; done
  die "$1"
}
start=$SECONDS
polls=0; next_show=$(( start + 15 ))
until curl -sf --max-time 5 "${auth[@]}" "$URL/v1/models" >/dev/null 2>&1; do
  running_here || fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  if (( polls++ % 5 == 0 )); then   # the workers' ranks every fifth second (an ssh each)
    for i in 1 2 3; do running_worker "$i" || fail "rank $i on $(worker_host "$i") exited before the server was ready"; done
  fi
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); the ranks were stopped"
  if (( SECONDS >= next_show )); then show "  loading: $(( SECONDS - start )) s"; next_show=$(( SECONDS + 15 )); fi
  sleep 1
done
kill $LOGS_PID 2>/dev/null || true
STARTED=0   # the server is up: a smoke test that fails from here leaves the ranks running to be looked at (stop.sh stops them)
log "Server answered after $((SECONDS - start))s"
show "  answering after $(( SECONDS - start )) s"
# each rank's Engram row cache asks for 2 MiB pages: a Spark whose memory was in small pieces when its rank started gets
# fewer (measured: 316 of 2048 MiB on the one engine a setup started once; it answered every prompt ~0.9 s late until a
# restart). awk reads the whole log: no early exit to break the pipe under pipefail.
row_short() {
  awk '!done && /row cache.s [0-9]+ MiB faulted in/ && /MiB of it in 2 MiB pages/ {
         want = $0; sub(/.*row cache.s /, "", want); sub(/ MiB.*/, "", want)
         got = $0; sub(/ MiB of it in 2 MiB pages.*/, "", got); sub(/.* /, "", got)
         if (got + 0 < want + 0) print got " of " want " MiB"; done = 1 }'
}
short=()
s=$(docker logs "$CONTAINER_NAME" 2>&1 | row_short) || true
[[ -z "$s" ]] || short+=("this Spark ($s)")
for i in 1 2 3; do
  s=$(worker "$i" "docker logs '$CONTAINER_NAME' 2>&1" 2>/dev/null | row_short) || true
  [[ -z "$s" ]] || short+=("$(worker_host "$i") ($s)")
done
if (( ${#short[@]} )); then
  list=$(printf '%s, ' "${short[@]}"); list=${list%, }
  warn "memory was in small pieces when the engine started on $list: its Engram row cache got fewer large (2 MiB) pages than it asked for. An engine started like this was measured answering every prompt about 0.9 s late; ./start.sh restart (page caches dropped, memory compacted) starts it whole."
fi

# ---------------------------------------------------------------- 5. smoke test
step 5 "Smoke test: one chat completion through all 4 ranks"
# a pass needs HTTP 200, no error object and the assistant's reply text (curl -w adds the status as the last line)
if smoke=$(curl -s --max-time 300 -w '\n%{http_code}' "${auth[@]}" "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 16, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" |
           python3 -c 'import json,sys
def no(why): sys.exit(why)
body, _, code = sys.stdin.read().rpartition("\n")
code == "200" or no("HTTP " + code)
r = json.loads(body)
r.get("error") is None or no("an error reply")
m = r["choices"][0]["message"]
m.get("role", "assistant") == "assistant" or no("not the assistant")
c = (m.get("content") or "").strip()
c or no("no reply text")
len(c) < 8 or len(set(c)) > 2 or no("one repeated character")
print(repr(c[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  log "OK: $smoke"
  show "  a first request through all four Sparks: OK"
else
  fail "the smoke test request failed (an HTTP error, an error reply, no reply text, or one character repeated); the ranks are still running"
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
(( ! UNDER_SETUP )) || exit 0   # the setup prints the addresses and keys itself
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED_NAME" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED_NAME)
    Window   $CONTEXT tokens · $PARALLEL at once · native FP8 / MXFP4 · drafts $( [[ "$DRAFTS" == 1 ]] && echo on || echo off) · images on · API key $( [[ -n "$API_KEY_FILE" ]] && echo required || echo none)
    Logs     docker logs -f $CONTAINER_NAME$(for i in 1 2 3; do printf '   (rank %s: ssh %s docker logs -f %s)' "$i" "$(worker_host "$i")" "$CONTAINER_NAME"; done)
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
