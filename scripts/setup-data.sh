#!/usr/bin/env bash
# Every Spark's data, made on this one (rank 0) and sent to the workers over the fabric: ./setup.sh's data step, or run
# by itself after a change of model, kit, engine or settings. Each worker reads its own copy (WORKER_HF_CACHE*):
#   1. scripts/prepare.sh --here: on this Spark the engine image (built), the checkpoint (downloaded once, ~510 GB), the
#      kernel kit, Engram's token map, the four nodes' weight files (~80 GB each) and the served folder
#   2. the image on every worker that lacks it: docker save | nc over the fabric, nc | docker load there, the three at
#      once from one read of the image here
#   3. with ENGRAM_PACKED=1, Engram's packed shards, cut here from the checkpoint and checked against its tables record
#      by record (scripts/engram-pack.sh)
#   4. each worker's share over the fabric, after the rails are set (./setup.sh network): tools/fabric-xfer, RDMA over both
#      rails, one run a group (each file read once here and written to every worker that needs it, every chunk checked by
#      CRC-32C on both ends), else scripts/fabric-copy.sh (TCP, every file checked by SHA-256) where it cannot be built or
#      finds no rails. Only what a worker lacks is sent: what its rank reads into its WORKER_HF_CACHE<i> (the checkpoint's
#      small files, the weight file of the node it serves, the token map, the kit), Engram's tables into WORKER_ENGRAM_DIR,
#      and its node's packed shards with MANIFEST.sha256 and PACKED.json into ENGRAM_PACKED_DIR
#   5. scripts/prepare.sh: every worker's copy checked file by file, the packed shards read once against the manifest on
#      each worker, each Spark's weight file's digest. Then ./start.sh.
# A worker whose WORKER_HF_CACHE<i> is empty reads this Spark's HF_CACHE over NFS instead (scripts/config.sh): nothing is
# sent to it. Safe to re-run: each step keeps what is already done.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh
source ./scripts/setup-lib.sh
t0=$SECONDS
took() { local s=$(( SECONDS - ${1:-t0} )); printf '%d min %02d s' $(( s / 60 )) $(( s % 60 )); }

# ---------------------------------------------------------------- 1. this Spark's data
log "This Spark first: the engine image, the model, the kernel kit, the weight files"
./scripts/prepare.sh --here
check_workers
need_workers
detect_links
# fabric_addrs <i>: worker i's fabric addresses, comma-separated: one a rail between this Spark and it, else its address
# on their link
fabric_addrs() {
  local line j hdev hip wdev wip
  local -a a=()
  for line in "${RAILS[@]}"; do
    read -r j hdev hip wdev wip <<<"$line"
    (( j == $1 )) && a+=("$wip")
  done
  (( ${#a[@]} )) || a=("${LINK_WORKER_ADDR[$1]:-}")
  [[ -n "${a[0]}" ]] || die "no fabric address of $(worker_host "$1") (rank $1): is its link up? (./setup.sh network)"
  (IFS=,; echo "${a[*]}")
}
own=()
for i in 1 2 3; do
  if [[ -n "$(worker_hf_cache "$i")" ]]; then own+=("$i")
  else log "  $(worker_host "$i") reads this Spark's copy over NFS ($(wvar WORKER_HF_CACHE "$i") is empty): nothing is sent to it"; fi
done
for i in "${own[@]}"; do detail "  $(worker_host "$i"): fabric addresses $(fabric_addrs "$i")"; done
work=$(mktemp -d "$STATE_DIR/setup-data.XXXXXX")
trap 'rm -rf -- "$work"' EXIT

# ---------------------------------------------------------------- 2. the image on each worker
ident=$(image_ident "$IMAGE")
[[ $ident != missing ]] || die "$IMAGE is not built here"
need=()
for i in 1 2 3; do
  if [[ "$(worker_image_ident "$i" "$IMAGE")" == "$ident" ]]; then detail "  $(worker_host "$i") has $IMAGE"
  else need+=("$i"); fi
done
(( ${#need[@]} )) || log "Engine image: already on the other three Sparks"
if (( ${#need[@]} )); then
  t=$SECONDS; port=5390
  names=(); for i in "${need[@]}"; do names+=("$(worker_host "$i")"); done
  log "Engine image to $(sparks_phrase 3 "${names[@]}") over the fabric"
  for i in "${need[@]}"; do
    a=$(fabric_addrs "$i" | cut -d, -f1)
    worker "$i" "rm -f /tmp/dsv41-load.rc; ss -ltnH | grep -q ':$port\$' && { echo 'port $port is in use' >&2; exit 1; }
      nohup bash -c 'timeout 3600 nc -l $a $port < /dev/null | docker load > /tmp/dsv41-load.log 2>&1; echo \${PIPESTATUS[*]} > /tmp/dsv41-load.rc' > /dev/null 2>&1 < /dev/null &
      for t in \$(seq 50); do ss -ltnH | grep -q '$a:$port' && exit 0; sleep 0.1; done; echo 'nc did not listen' >&2; exit 1" < /dev/null ||
      die "$(worker_host "$i") cannot receive the engine image on $a:$port"
    mkfifo "$work/image.$i"
    ( nc -N "$a" "$port" < "$work/image.$i" > /dev/null ) &
  done
  fifos=(); for i in "${need[@]}"; do fifos+=("$work/image.$i"); done
  docker save "$IMAGE" | tee "${fifos[@]}" > /dev/null || die "the engine image could not be sent over the fabric"
  wait
  for i in "${need[@]}"; do
    for _ in $(seq 600); do worker "$i" "test -s /tmp/dsv41-load.rc" < /dev/null && break; sleep 1; done
    [[ "$(worker_image_ident "$i" "$IMAGE")" == "$ident" ]] ||
      die "$(worker_host "$i"): its engine image differs from this Spark's after the copy: $(worker "$i" 'cat /tmp/dsv41-load.rc; tail -2 /tmp/dsv41-load.log' < /dev/null | tr '\n' ' ')"
    detail "  $(worker_host "$i"): $IMAGE identical to this Spark's"
  done
  log "  done in $(took "$t"); the same image on all four"
fi

# ---------------------------------------------------------------- 3. Engram's packed shards, cut here
if [[ "$ENGRAM_PACKED" == 1 ]]; then
  log "Engram's packed tables, cut from the model"
  engram_pack_progress
else
  detail "ENGRAM_PACKED is not 1: no packed tables"
fi

# ---------------------------------------------------------------- 4. each worker's share over the fabric
SNAP="$(model_cache_dir)/snapshots/$(snapshot_rev)"
roots=("hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)" "$DATA_SUB" "$KIT_SUB/kit")
read -r -a shards <<<"$(engram_shards)"
nodes=$(packed_nodes)
# each worker's groups: "<name> <streams> <port> <this Spark's folder> <the worker's folder>", the paths in $work/<name>.<i>
groups_of() {
  local i=$1 node
  node=$(cut -d, -f$((i + 1)) <<<"$(node_order)")
  (cd "$HF_CACHE" && listing "${roots[@]}" | rank_reads "$node") | awk '{ print substr($0, 1, length($0) - length($NF) - 1) }' > "$work/model.$i"
  echo "model 4 5400 $HF_CACHE $(worker_hf_cache "$i")"
  if (( ${#shards[@]} )); then
    printf '%s\n' "${shards[@]}" > "$work/engram.$i"
    echo "engram 8 5420 $SNAP $WORKER_ENGRAM_DIR"
  fi
  if [[ "$ENGRAM_PACKED" == 1 ]]; then
    { packed_files "$(engram_node "$i")" "$nodes" | awk '{ print $1 }'; echo MANIFEST.sha256; echo PACKED.json; } > "$work/packed.$i"
    echo "packed 8 5440 $(packed_dir_here) $ENGRAM_PACKED_DIR"
  fi
}
log "Copying each Spark's share over the fabric"
for i in "${own[@]}"; do
  groups_of "$i" > "$work/groups.$i"
  total=0
  while read -r g n p src dst; do
    b=$(scripts/fabric-copy.sh -s "$(worker_host "$i")" "$(fabric_addrs "$i")" "$src" "$dst" < "$work/$g.$i")
    total=$(( total + b ))
    detail "  $(worker_host "$i") $g: $(wc -l < "$work/$g.$i") files, $(( b / 1000000000 )) GB to send, into $dst"
  done < "$work/groups.$i"
  # the share's free room: the smallest of its folders' free space against all of it (conservative when they are on
  # different disks)
  free=$(while read -r g n p src dst; do
           worker "$i" "mkdir -p \"$dst\" && df -B1 --output=avail \"$dst\" | tail -1" < /dev/null
         done < "$work/groups.$i" | sort -n | head -1)
  (( free > total + 10 * 1024 ** 3 )) ||
    die "$(worker_host "$i") has $(( free / 1024 ** 3 )) GiB free: its share needs $(( total / 1024 ** 3 )) GiB, and 10 GiB to spare"
done
# the copy: tools/fabric-xfer when it builds here (gcc, make, libibverbs-dev) and finds this Spark's two rails; else the TCP copy
read -r -a _ra <<<"$RAIL_A_ADDRS"; read -r -a _rb <<<"$RAIL_B_ADDRS"
FX=tools/fabric-xfer
if ! make -s -C "$FX" fabric-xfer > "$work/fx-build.log" 2>&1; then
  copier=tcp; why=$(grep -m 1 -i -E 'error|not found' "$work/fx-build.log" | cut -c1-160)
elif ! "$FX/fabric-xfer" selftest --addr "${_ra[0]}" --addr "${_rb[0]}" > "$work/fx-self.log" 2>&1 || grep -q '^FAIL' "$work/fx-self.log"; then
  copier=tcp; why=$(grep -m 1 -E '^FAIL|ERROR' "$work/fx-self.log" | cut -c1-160)
else copier=fx; fi
t=$SECONDS
if [[ $copier == fx ]]; then
  log "  RDMA over both rails; each file read once here, every chunk checked on both ends"
  for g in $(awk '{ print $1 }' "$work/groups.${own[0]}"); do
    # the workers whose folder for this group is the same share one run; a worker with its own folder gets its own
    declare -A by_dst=(); declare -A meta=()
    for i in "${own[@]}"; do
      read -r _ n p src dst < <(grep "^$g " "$work/groups.$i")
      by_dst[$dst]+="$i "; meta[$dst]="$n $p $src"
    done
    for dst in "${!by_dst[@]}"; do
      read -r -a ws <<<"${by_dst[$dst]}"; read -r n p src <<<"${meta[$dst]}"
      lists=(); for i in "${ws[@]}"; do lists+=("$work/$g.$i"); done
      fx_routes "${lists[@]}" > "$work/$g.routes"
      targets=$(for i in "${ws[@]}"; do worker_host "$i"; done | paste -sd,)
      addrs=$(for i in "${ws[@]}"; do fabric_addrs "$i"; done | paste -sd';')
      case $g in model) gname="model files" ;; engram) gname="Engram tables" ;; packed) gname="Engram's packed tables" ;; *) gname=$g ;; esac
      detail "  $g: $(wc -l < "$work/$g.routes") files to $targets, into $dst"
      FX_OPTS="--progress-s ${FX_PROGRESS_S:-10}" "$FX/fabric-xfer.sh" -n "$n" -p "$p" -l "$g" "$targets" "$addrs" "$src" "$dst" \
        < "$work/$g.routes" 2>&1 | fx_lines "$gname" || die "the copy of the $gname failed (above). Run ./setup.sh data again: what arrived whole is kept"
    done
    unset by_dst meta
  done
else
  warn "the RDMA copy is not available here (${why:-see $work}): copying over TCP instead, which is slower"
  pids=(); names=()
  for i in "${own[@]}"; do
    while read -r g n p src dst; do
      ( scripts/fabric-copy.sh -n "$n" -p "$p" -l "rank $i $g" "$(worker_host "$i")" "$(fabric_addrs "$i")" "$src" "$dst" \
          < "$work/$g.$i" 2>&1 | tee "$work/copy.$g.$i.log"; exit "${PIPESTATUS[0]}" ) &
      pids+=($!); names+=("rank $i $g")
    done < "$work/groups.$i"
  done
  bad=""
  for k in "${!pids[@]}"; do wait "${pids[k]}" || bad+=" ${names[k]};"; done
  [[ -z "$bad" ]] || die "a copy failed:$bad (above). Run ./setup.sh data again: what arrived whole is kept"
fi
log "  every share sent and checked in $(took "$t")"

# ---------------------------------------------------------------- 5. every copy checked, the digests
log "Final check of every Spark's copy"
./scripts/prepare.sh
log "All data in place on the four Sparks, in $(took)$( [[ ${SETUP_STARTS:-0} == 1 ]] || echo ". Start the engine with ./start.sh")"
