#!/usr/bin/env bash
# Engram's packed shards of every node, cut on this Spark (rank 0) from its own checkpoint, for workers that read their
# own copy of the model (WORKER_HF_CACHE*): they mount no checkpoint over NFS, so they cannot pack their own. Run here,
# in the recipe's folder, with the settings start.sh will read (ENGRAM_PACKED=1, and ENGRAM_OWNED):
#   scripts/engram-pack.sh [--mbps N] [--verify all|own|none]
# 1. each node's shards (ENGRAM_OWNED=1: the 2D split's four nodes', 0: the whole tables, one set every node reads) at
#    their exact sizes from the tables' own headers (packed_files): a file of that name and size stays; the nodes that
#    lack one are packed by the image's tf-dsv41-engram-pack (it keeps a shard that checks out), all at once, into this
#    Spark's ENGRAM_PACKED_DIR, which holds every node's: the source of the workers' copies
# 2. each record checked against its table row, in order (scripts/engram-verify.py verify; --verify all: every node's
#    shards here (the default), own: this Spark's node's, none: each node's checked on its own Spark later), merged into
#    MANIFEST.sha256 beside them and this Spark's engram-packed.ok
# 3. PACKED.json beside them: the checkpoint, its Engram files' blobs, the image, each shard's size
# Then each worker needs its node's files, MANIFEST.sha256 and PACKED.json in its own ENGRAM_PACKED_DIR (printed at the
# end); its prepare.sh reads each file once against the manifest, unless a check there wrote its verified line.
# --mbps N: at most N MB/s (10^6 bytes a second) on this Spark's disks, shared by the packers (docker's --device-read-bps
# on the disk that holds HF_CACHE, --device-write-bps on the folder's) and by the checks (their own pacing); 0 (the
# default): unpaced. Under the none I/O scheduler too: the limits are the cgroup's (io.max), on whole disks.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh

MBPS=0; VERIFY=all
while (( $# )); do
  case "$1" in
    --mbps) MBPS=${2:-}; shift 2 || die "--mbps needs a value" ;;
    --verify) VERIFY=${2:-}; shift 2 || die "--verify needs a value" ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$MBPS" =~ ^[0-9]+$ ]] || die "--mbps is a whole number of MB/s (0: unpaced), not $MBPS"
[[ "$VERIFY" =~ ^(all|own|none)$ ]] || die "--verify is all, own or none, not $VERIFY"
[[ "$ENGRAM_PACKED" == 1 ]] ||
  die "ENGRAM_PACKED is not 1 in this folder's settings (scripts/local.sh or the environment): this packs the shards a start with packed rows reads"
mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "a prepare.sh or engram-pack.sh runs: wait for it"
command -v docker >/dev/null || die "docker is not installed"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing: scripts/prepare.sh builds it"

SNAP="$(model_cache_dir)/snapshots/$(snapshot_rev)"
SNAP_C=$(model_in_container)
read -r -a shards <<<"$(engram_shards)"
(( ${#shards[@]} )) || die "the checkpoint's index names no Engram tables"
tables=(); for f in "${shards[@]}"; do tables+=("$SNAP/$f"); done
mapfile -t sources < <(engram_sources)
nodes=$(packed_nodes)
dir=$(packed_dir_here)
mkdir -p "$dir"
work=$(mktemp -d "$STATE_DIR/engram-pack.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
ts() { TZ=Australia/Sydney date '+%F %T %Z'; }

# ---------------------------------------------------------------- 1. what is here, the missing nodes packed
todo=(); specs=(); need=0; have=0
for (( g = 0; g < nodes; g++ )); do
  spec=$(packed_files "$g" "$nodes") || die "cannot read Engram's tables' headers in $SNAP"
  whole=1
  while read -r line; do
    read -r name bytes _ <<<"$line"
    specs+=("${line// /:}")
    need=$(( need + bytes ))
    if [[ -f "$dir/$name" && "$(stat -c %s "$dir/$name")" == "$bytes" ]]; then have=$(( have + bytes )); else whole=0; fi
  done <<<"$spec"
  (( whole )) || todo+=("$g")
done
detail "$(ts): $nodes node(s)' shards in $dir: $(( ${#todo[@]} )) to pack ($(( (need - have) / 1000000000 )) GB), $(( nodes - ${#todo[@]} )) here at their sizes"
if (( ${#todo[@]} )); then log "  $(( (need - have) / 1000000000 )) GB to pack$( (( ${#todo[@]} < nodes )) && echo ", the rest already here")"
else log "  already packed"; fi
if (( ${#todo[@]} )); then
  free=$(df -B1 --output=avail "$dir" | tail -1)
  (( free + have >= need + 10 * 1024 ** 3 )) ||
    die "$dir has $(( free / 1024 ** 3 )) GiB free: every node's packed shards need $(( (need - have) / 1024 ** 3 )) GiB more and 10 GiB to spare"
  throttle=()
  if (( MBPS > 0 )); then
    # the whole disk under a path (io.max takes no partition)
    disk_of() {
      local src pk
      src=$(findmnt -n -o SOURCE -T "$1" 2>/dev/null) || return 1
      [[ "$src" == /dev/* ]] || return 1
      pk=$(lsblk -no PKNAME "$src" 2>/dev/null | head -1)
      echo "/dev/${pk:-${src#/dev/}}"
    }
    rd=$(disk_of "$HF_CACHE") || die "cannot pace the packers: $HF_CACHE is on no local disk"
    wr=$(disk_of "$dir") || die "cannot pace the packers: $dir is on no local disk"
    per=$(( MBPS * 1000000 / ${#todo[@]} ))
    throttle=(--device-read-bps "$rd:$per" --device-write-bps "$wr:$per")
    log "  packing at most $MBPS MB/s in all"; detail "each of the ${#todo[@]} packer(s) at most $per bytes a second read from $rd and written to $wr"
  fi
  pids=()
  for g in "${todo[@]}"; do
    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp "${throttle[@]}" -v "$HF_CACHE:/hf:ro" \
      -v "$dir:/out" --entrypoint tf-dsv41-engram-pack "$IMAGE" "$SNAP_C" /out "$g" "$nodes" > "$work/pack-$g.log" 2>&1 &
    pids+=($!)
  done
  bad=""
  for k in "${!pids[@]}"; do wait "${pids[k]}" || bad+=" ${todo[k]}"; done
  [[ -z "$bad" ]] || die "packing node(s)$bad of $nodes failed: $(for g in $bad; do tail -n 3 "$work/pack-$g.log"; done | tr '\n' ' ')"
  for s in "${specs[@]}"; do
    IFS=: read -r name bytes _ <<<"$s"
    [[ "$(stat -c %s "$dir/$name" 2>/dev/null)" == "$bytes" ]] || die "$dir/$name is not $bytes bytes after packing"
  done
  detail "$(ts): packed node(s) ${todo[*]} of $nodes"; log "  packed"
fi

# ---------------------------------------------------------------- 2. each record against its table row
case "$VERIFY" in
  all) mapfile -t check < <(seq 0 $(( nodes - 1 ))) ;;
  own) check=("$(engram_node 0)") ;;
  none) check=() ;;
esac
if (( ${#check[@]} )); then
  detail "$(ts): checking node(s) ${check[*]}'s shards against the tables, record by record$( (( MBPS == 0 )) || echo ", at most $MBPS MB/s in all")"
  log "  checking every record against the model's tables (about 3 min for each 100 GB)"
  share=$(awk -v m="$MBPS" -v k="${#check[@]}" 'BEGIN { printf "%.3f", m / k }')
  pids=()
  for g in "${check[@]}"; do
    python3 -I scripts/engram-verify.py verify "$g" "$nodes" "$dir" "${tables[@]}" --mbps "$share" \
      > "$work/verify-$g.json" 2> "$work/verify-$g.err" &
    pids+=($!)
  done
  bad=""
  for k in "${!pids[@]}"; do wait "${pids[k]}" || bad+=" ${check[k]}"; done
  [[ -z "$bad" ]] || die "node(s)$bad: a shard differs from the checkpoint's tables: $(for g in $bad; do grep -h '^defect:' "$work/verify-$g.json" "$work/verify-$g.err"; done | tr '\n' ' ')"
  cat "$work"/verify-*.json | python3 -I scripts/engram-verify.py merge --manifest "$dir/MANIFEST.sha256" --ok "$(engram_ok 0)"
  detail "$(ts): $(cat "$work"/verify-*.json | grep -c '^{') shard(s) equal to their tables; their lines in $dir/MANIFEST.sha256 and $(engram_ok 0)"
  log "  checked: every packed table equal to the model's tables"
fi

# ---------------------------------------------------------------- 3. PACKED.json
python3 -I scripts/engram-verify.py record "$dir" "$MODEL_ID@$(snapshot_rev)" "$nodes" "$(image_ident "$IMAGE")" \
  "${sources[@]}" "${specs[@]}"
detail "$(ts): $dir/PACKED.json: $MODEL_ID@$(snapshot_rev), $nodes node(s), image $(image_ident "$IMAGE" | cut -c1-12)"
for r in 1 2 3; do
  g=$(engram_node "$r")
  files=$(for s in "${specs[@]}"; do IFS=: read -r name _ <<<"$s"; [[ "$name" != *"-n${g}of${nodes}.tfe" ]] || printf ' %s' "$name"; done)
  detail "rank $r ($(worker_host "$r")) needs in its $ENGRAM_PACKED_DIR:$files MANIFEST.sha256 PACKED.json"
done
