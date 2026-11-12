#!/usr/bin/env bash
# Prepare the four Sparks to serve deepseek-ai/DeepSeek-V4.1-Flash-1017 with TensorFold's native engine. Everything
# lands on this Spark (spark1); a worker reads it over NFS, or its own copy (WORKER_HF_CACHE*: checked in step 6):
#   1. preflight: docker and the GPU on every Spark, key-based ssh, the links (every pair on one RoCE subnet)
#   2. the engine image on this Spark (zig/Dockerfile: this repository's engine at the commit checked out, on
#      BASE_IMAGE), then on each worker (docker save | docker load: only what it lacks is stored)
#   3. the checkpoint into HF_CACHE (~510 GB, resumable), at its pinned revision
#   4. the kernel kit: BertholomusAI's kit/ at KIT_REF, its RoPE tables and image bias made from the checkpoint, every
#      file checked against its MANIFEST (the image bias is 1017's own: the one file that differs from theirs)
#   5. Engram's token map and the four nodes' weight files (one each, ~80 GB) from the checkpoint, no GPU, and the
#      folder rank 0 serves: the checkpoint's files linked, beside model/chat_template.jinja (the release ships its
#      chat format as encoding/encoding.py, which the server cannot run)
#   6. each worker's own copy (WORKER_HF_CACHE*), or a read-only NFS volume of this Spark's HF_CACHE, checked: every
#      file its rank reads, by size
#   7. Engram's tables on each worker's disk, or with ENGRAM_PACKED=1 each Spark's packed shards (tf-dsv41-engram-pack):
#      each at its exact size from the tables' headers, checked against the tables record by record (or read once
#      against the manifest of that check), and the workers' own copies of the tables beside them
#   8. each Spark's weight file's SHA-256 (O_DIRECT, cached in its state folder): the SSD tier's identity
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch; --here to stop after step 5, with nothing copied to the workers (the
# first step of scripts/setup-data.sh, which then sends each worker its copy over the fabric and runs this again).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
source ./scripts/nodes.sh
source ./scripts/setup-lib.sh

REBUILD=0; HERE=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    --here) HERE=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it: pgrep -af prepare.sh"
detail "Preflight checks on all 4 Sparks"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
nvidia-smi -L >/dev/null 2>&1 || warn "nvidia-smi failed on this node: is the NVIDIA driver working?"
check_workers
for i in 1 2 3; do
  need_worker "$i"
  worker "$i" 'docker info >/dev/null 2>&1' || die "$(worker_host "$i") cannot talk to its docker daemon (docker group?)"
  worker "$i" 'nvidia-smi -L >/dev/null 2>&1' || warn "nvidia-smi failed on $(worker_host "$i")"
done
detect_links
detail "2D nodes by rank: $(node_order)"
for i in 1 2 3; do detail "  worker $i ($(worker_host "$i")): link ${LINK_HEAD_ADDR[i]:-?} <-> ${LINK_WORKER_ADDR[i]:-?}; $( [[ -n "$(worker_hf_cache "$i")" ]] && echo "its own copy at $(worker_hf_cache "$i")" || echo "NFS from $(nfs_server "$i")")"; done

# ---------------------------------------------------------------- 2. the image
hash=$(image_hash)
built=$(docker image inspect -f '{{index .Config.Labels "tf.source"}}' "$IMAGE" 2>/dev/null || true)
if [[ $REBUILD -eq 1 || "$built" != "$hash" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || { log "  Downloading the base image ($BASE_IMAGE)"; docker pull "$BASE_IMAGE" >/dev/null; }
  commit=$(source_commit)
  ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/src"; cp zig/Dockerfile "$ctx/Dockerfile"
  if [[ -n "$commit" ]]; then
    # the engine as committed: what is not committed is not built (and is named here)
    [[ -z "$(git status --porcelain -- "${ENGINE_PATHS[@]}")" ]] ||
      log "  Note: the engine's files have uncommitted changes; the image is built from commit ${commit:0:12} without them"
    git archive "$commit" -- "${ENGINE_PATHS[@]}" | tar -x -C "$ctx/src"
  else
    tar -c "${ENGINE_PATHS[@]}" 2>/dev/null | tar -x -C "$ctx/src"
  fi
  mkdir -p "$LOG_DIR"; blog="$LOG_DIR/image-build-$(date +%Y%m%d-%H%M%S).log"
  detail "Building $IMAGE (the engine at ${commit:0:12}${commit:+, }$(du -sh "$ctx/src" | cut -f1) of source, Zig $ZIG_VERSION)"
  log "  Building the engine image (about 5 min; full log: $blog)"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  docker_build_steps "$blog" "${nocache[@]}" -t "$IMAGE" -f "$ctx/Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" \
    --build-arg SOURCE_HASH="$hash" --build-arg SOURCE_COMMIT="$commit" "$ctx" || die "the image did not build (above)"
  rm -rf -- "$ctx"; trap - EXIT
else
  log "  Engine image: already built"; detail "  $IMAGE (hash $hash; --rebuild builds it again)"
fi
ident=$(image_ident "$IMAGE")
for i in 1 2 3; do
  (( ! HERE )) || break
  [[ "$(worker_image_ident "$i" "$IMAGE")" != "$ident" ]] || continue
  log "  Engine image to $(worker_host "$i")"
  docker save "$IMAGE" | worker "$i" docker load >/dev/null
  [[ "$(worker_image_ident "$i" "$IMAGE")" == "$ident" ]] || die "$(worker_host "$i")'s $IMAGE differs from this Spark's"
done
if (( HERE )); then log "  Engine image ready"
else log "  Engine image: the same on all four Sparks"; fi

# ---------------------------------------------------------------- 3. the checkpoint
hf_on_path || die "the Hugging Face CLI ('hf') is not on this Spark (not on PATH, not in ~/.local/bin): pip install -U huggingface_hub"
log "  Model files: checking the download (510 GB the first time; an interrupted download resumes)"
detail "  $MODEL_ID${MODEL_REVISION:+ @ ${MODEL_REVISION:0:8}} into $HF_CACHE/hub"
if ! hf download "$MODEL_ID" ${MODEL_REVISION:+--revision "$MODEL_REVISION"} --cache-dir "$HF_CACHE/hub" >/dev/null; then
  [[ -n "$MODEL_REVISION" && -f "$(model_cache_dir)/snapshots/$MODEL_REVISION/model.safetensors.index.json" ]] ||
    die "$MODEL_ID: the download failed"
  warn "could not reach Hugging Face; using the model files already here"
fi
SNAP="$(model_cache_dir)/snapshots/$(snapshot_rev)"
[[ -f "$SNAP/model.safetensors.index.json" ]] || die "$MODEL_ID: no complete snapshot at $SNAP"
log "  Model files: complete ($(du -shL "$SNAP" | cut -f1)B)"; detail "  $SNAP"
# a container's view of HF_CACHE (the snapshot's files are links into ../../blobs)
in_hf() { docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HF_CACHE:/hf" "$@"; }

# ---------------------------------------------------------------- 4. the kit
KIT_DIR="$HF_CACHE/$KIT_SUB"
if [[ ! -d "$KIT_DIR/.git" ]]; then
  detail "Fetching the kernel kit: $KIT_REPO @ ${KIT_REF:0:8}"
  rm -rf "$KIT_DIR"; git init -q "$KIT_DIR"
  git -C "$KIT_DIR" remote add origin "$KIT_REPO"
  git -C "$KIT_DIR" fetch -q --depth 1 origin "$KIT_REF"
  git -C "$KIT_DIR" checkout -q --detach FETCH_HEAD
fi
[[ "$(git -C "$KIT_DIR" rev-parse HEAD)" == "$KIT_REF" ]] || die "$KIT_DIR is not at KIT_REF ${KIT_REF:0:8}: remove it and run this again"
detail "Checking the kit (its RoPE tables and image bias made from the checkpoint; CPU only)"
kit_out=$(IMAGE="$IMAGE" "$KIT_DIR/kit/verify_kit.sh" "$SNAP" "$SNAP" 2>&1 | tail -1) || true
# 1017's image routing bias is its own, not the original release's: the one file allowed to differ from MANIFEST
kit_n=0; kit_t=-1; [[ $kit_out =~ ([0-9]+)\ of\ ([0-9]+)\ files\ match ]] && { kit_n=${BASH_REMATCH[1]}; kit_t=${BASH_REMATCH[2]}; }
(( kit_n == kit_t )) || { (( kit_n == kit_t - 1 )) && [[ "$kit_out" == *"1 differ, 0 missing"* ]]; } ||
  die "the kit does not check out: $kit_out"
grep -q "DIFFERS (bias): vision/gate_bias_vl.safetensors" <(IMAGE="$IMAGE" "$KIT_DIR/kit/verify_kit.sh" "$SNAP" 2>&1) ||
  (( kit_n == kit_t )) || die "a kit file other than the image bias differs from MANIFEST"
detail "Kit: $kit_out (the image bias is 1017's own)"
if [[ $kit_out =~ ([0-9]+)\ of\ ([0-9]+)\ files ]]; then
  log "  Kernel kit ${KIT_REF:0:8}: fetched and checked (${BASH_REMATCH[1]} of ${BASH_REMATCH[2]} files as published; the image bias is this model's own)"
else log "  Kernel kit ${KIT_REF:0:8}: fetched and checked"; fi
kit_extra "$KIT_DIR" || warn "this kit ($KIT_REPO @ ${KIT_REF:0:8}) has no extra set (kit/aot-extra/): TF_DS_ATTN_U, TF_DS_IDX_U and TF_DS_DRAFT_4WAY will say at the start that they are off, and prompts run ~2.7 % and replies 3-4 % slower than the README's tables. The default KIT_REPO / KIT_REF (scripts/config.sh) name the kit that has it"

# ---------------------------------------------------------------- 4.5 the Keys abliteration overlay (ABLIT=1)
if [[ "$ABLIT" == 1 ]]; then
  warn "ABLIT=1: $ABLIT_ID is gated. You must have accepted its terms at https://huggingface.co/$ABLIT_ID while" \
       "logged in, with a Hugging Face read token for that account: export HF_TOKEN=hf_... or hf auth login."
  log "Downloading the Keys overlay's native-FP8 sidecar ($ABLIT_SIDECAR, ~1.1 GB) from $ABLIT_ID @ ${ABLIT_REVISION:0:8}"
  hf download "$ABLIT_ID" --revision "$ABLIT_REVISION" --include "$ABLIT_SIDECAR" --cache-dir "$HF_CACHE/hub" >/dev/null ||
    die "$ABLIT_ID: the download failed (is its gate accepted by this HF_TOKEN's account?)"
  ABLIT_DIR="$DATA_DIR/checkpoint"
  if [[ ! -e "$ABLIT_DIR/$ABLIT_SIDECAR" ]]; then
    log "Building the overlaid checkpoint view ($ABLIT_DIR): every stock file linked, plus $ABLIT_SIDECAR" \
        "(our engine never reads model.safetensors.index.json: checkpoint.zig's Shards.open scans every" \
        "*.safetensors file in a directory by its own header and keeps the last one seen, in sorted-path order," \
        "for a given tensor name -- \"wo_b_...\" sorts after every \"model-...\" shard, so the overlay's 26 layers" \
        "of attn.wo_b win)"
    rm -rf "$ABLIT_DIR.new" && mkdir -p "$ABLIT_DIR.new"
    for f in "$SNAP"/*; do ln -s "../../../hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)/$(basename "$f")" "$ABLIT_DIR.new/"; done
    ln -s "../../../hub/models--${ABLIT_ID//\//--}/snapshots/$ABLIT_REVISION/$ABLIT_SIDECAR" "$ABLIT_DIR.new/$ABLIT_SIDECAR"
    rm -rf "$ABLIT_DIR" && mv "$ABLIT_DIR.new" "$ABLIT_DIR"
  fi
  log "Checkpoint for baking: $ABLIT_DIR (stock + $ABLIT_SIDECAR grafted onto attn.wo_b, layers 10-35)"
fi

# ---------------------------------------------------------------- 5. token map and weight files
mkdir -p "$DATA_DIR/rank-cache"
SNAP_C=$(model_in_container)
if [[ ! -s "$DATA_DIR/dsv41_token_map.json" ]]; then
  detail "Engram's token map from the checkpoint's tokenizer"
  in_hf --entrypoint python3 "$IMAGE" /opt/tensorfold/tools/inputs/make_token_map.py "$SNAP_C" "/hf/$DATA_SUB/dsv41_token_map.json" >/dev/null
fi
log "  Weight files, one for each Spark, cut from the model"
for g in 0 1 2 3; do
  # the file the engine looks for: its name carries the index's hash (tf-dsv41-rank-cache --index-only). A failed or
  # empty index stops here: hashing nothing would match no weight file, and the loop would delete the good one before a
  # rebuild that fails the same way. The index also gives the file's size, for the progress lines.
  idx=$(mktemp "${TMPDIR:-/tmp}/dsv41-index$g.XXXXXX")
  { in_hf --entrypoint tf-dsv41-rank-cache "$IMAGE" "$SNAP_C" /tmp/x "$g" 4 --parity --index-only > "$idx" && [[ -s $idx ]]; } ||
    { rm -f "$idx"; die "cannot compute node $g's weight-file index from $SNAP (tf-dsv41-rank-cache --index-only): is the checkpoint complete?"; }
  want=$(sha256sum < "$idx" | cut -c1-20)
  if [[ -f "$DATA_DIR/rank-cache/rank${g}of4-$want.bin" ]]; then log "    weight file $((g + 1)) of 4: ready"; rm -f "$idx"; continue; fi
  rm -f "$DATA_DIR/rank-cache/rank${g}of4-"*.bin "$DATA_DIR/rank-cache/.rank${g}of4.partial"
  total=$(( $(index_data_bytes "$idx") + $(stat -c %s "$idx") + 16 )); rm -f "$idx"
  detail "Weight file of node $g from the checkpoint ($(( total / 1000000000 )) GB)"
  # the tool's own lines (its MiB counts) go to a log, shown when it fails; the progress here has the per cent and time left
  wlog=$(mktemp "${TMPDIR:-/tmp}/dsv41-weights$g.XXXXXX")
  in_hf --entrypoint tf-dsv41-rank-cache "$IMAGE" "$SNAP_C" "/hf/$DATA_SUB/rank-cache" "$g" 4 --parity >/dev/null 2>"$wlog" & pid=$!
  file_progress "$pid" "weight file $((g + 1)) of 4" "$total" "$DATA_DIR/rank-cache/rank${g}of4-$want.bin" "$DATA_DIR/rank-cache/.rank${g}of4.partial"
  wait "$pid" || { tail -n 20 "$wlog" >&2; rm -f "$wlog"; die "node $g's weight file was not built (tf-dsv41-rank-cache, its last lines above)"; }
  rm -f "$wlog"
done

# the served folder: every file of the snapshot (or, with ABLIT=1, the overlaid view above) as a relative link (it
# resolves under /hf in a container too), and the recipe's chat template
VIEW="$DATA_DIR/model"
rm -rf "$VIEW.new" && mkdir -p "$VIEW.new"
if [[ "$ABLIT" == 1 ]]; then
  for f in "$ABLIT_DIR"/*; do ln -s "../checkpoint/$(basename "$f")" "$VIEW.new/"; done
else
  for f in "$SNAP"/*; do ln -s "../../../hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)/$(basename "$f")" "$VIEW.new/"; done
fi
cp model/chat_template.jinja "$VIEW.new/chat_template.jinja"
rm -rf "$VIEW" && mv "$VIEW.new" "$VIEW"
[[ -s "$VIEW/config.json" ]] || die "the served folder $VIEW does not reach the checkpoint"
log "  Model folder ready (the model's files linked, the chat template beside them)"; detail "  $VIEW"
if (( HERE )); then log "  This Spark's data is ready"; exit 0; fi

# ---------------------------------------------------------------- 6. the workers' copies or NFS volumes
manifest() {  # the files a rank reads, with their sizes, as paths under HF_CACHE
  (cd "$HF_CACHE" && find -L "hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)" "$DATA_SUB" "$KIT_SUB/kit" \
     -path '*/.git' -prune -o -type f -printf '%p %s\n' | LC_ALL=C sort)
}
want=""   # (the NFS comparison's list, made for the first worker that reads over NFS)
for i in 1 2 3; do
  if [[ -n "$(worker_hf_cache "$i")" ]]; then
    diffs=$(check_worker_files "$i") ||
      die "$(worker_host "$i")'s own copy at $(worker_hf_cache "$i") does not hold every file rank $i reads at this Spark's size (< here, > there):
$diffs
    Copy them there first, at the same paths under $(worker_hf_cache "$i") (a weight file is ~80 GB: all workers at once)"
    detail "Worker $i ($(worker_host "$i")) reads its own copy at $(worker_hf_cache "$i"): every file rank $i reads, at this Spark's size"
    log "  $(worker_host "$i"): its copy of the model files and weight file: OK"
    continue
  fi
  [[ -n "$want" ]] || want=$(manifest)
  ensure_nfs_volume "$i"
  if ! out=$(worker_nfs "$i" true 2>&1); then
    die "$(worker_host "$i") cannot mount this Spark's :$NFS_PATH from $(nfs_server "$i") over NFS ($NFS_VOLUME): $(tail -1 <<<"$out")
    This Spark must export HF_CACHE (as $NFS_PATH) read-only to that worker's address on its link (README: NFS)"
  fi
  have=$(worker "$i" "docker run --rm --network none -v '$NFS_VOLUME:/hf:ro' -w /hf --entrypoint find '$IMAGE' -L 'hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)' '$DATA_SUB' '$KIT_SUB/kit' -path '*/.git' -prune -o -type f -printf '%p %s\n'" 2>/dev/null | LC_ALL=C sort || true)
  [[ "$have" == "$want" ]] || { diff <(printf '%s\n' "$want") <(printf '%s\n' "$have") | head -10 >&2
    die "$(worker_host "$i") does not see every file over NFS ($NFS_VOLUME)"; }
  log "  $(worker_host "$i") reads the model files, its weight file and the kit over NFS"
done

# ---------------------------------------------------------------- 7. Engram's tables on each worker's disk
SNAP_REL="hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"
read -r -a shards <<<"$(engram_shards)"
(( ${#shards[@]} )) || die "the checkpoint's index names no Engram tables"
need=0; for f in "${shards[@]}"; do need=$(( need + $(stat -L -c %s "$SNAP/$f") )); done
if [[ "$ENGRAM_PACKED" == 1 ]]; then
  # ENGRAM_PACKED=1: packed shards on every Spark's own disk instead (engram_pack.zig), cut for the node rank r serves
  # (TF_DS_2D_ORDER) or whole, each read as it is only when its exact size, its header, the manifest and PACKED.json
  # beside it and its verified line agree (scripts/nodes.sh engram_check). A shard missing or of another size is packed
  # here, or on a worker that reads over NFS, by the image's tf-dsv41-engram-pack (it keeps a shard that checks out, so a
  # re-run writes only what is missing), then checked against the tables record by record (scripts/engram-verify.py); a
  # worker with its own copy (WORKER_HF_CACHE<i>) reads no checkpoint to pack from: scripts/engram-pack.sh packs its node's
  # shards on rank 0, to be copied there.
  nodes=$(packed_nodes)
  tables=(); for f in "${shards[@]}"; do tables+=("$SNAP/$f"); done
  mapfile -t sources < <(engram_sources)
  # every rank's check at once, each on its own disk (a shard without a verified line is read once against the manifest,
  # ~100 s a 100 GB shard), then each rank's result read in rank order
  ck=$(mktemp -d "$STATE_DIR/engram-check.XXXXXX"); trap 'rm -rf -- "$ck"' EXIT
  log "  Engram's packed tables on all four Sparks, checked at once (about 2 min for each 100 GB not checked before)"
  cks=()
  for r in 0 1 2 3; do { rc=0; engram_check "$r" --hash > "$ck/$r" || rc=$?; echo "$rc" > "$ck/$r.rc"; } & cks+=($!); done
  wait "${cks[@]}"
  for r in 0 1 2 3; do
    node=$(engram_node "$r")
    where="this Spark"; (( r == 0 )) || where=$(worker_host "$r")
    rc=$(cat "$ck/$r.rc"); out=$(cat "$ck/$r")
    if (( rc == 0 )); then
      detail "Rank $r ($where): Engram's packed shards of node $node of $nodes on its disk, verified ($(paste -sd';' - <<<"$out" | sed 's/;/; /g'))"
      log "    $where: Engram's packed tables: OK"
      continue
    fi
    (( rc == 3 )) || die "rank $r ($where): its packed Engram shards in $(engram_dir "$r") cannot be read as they are: $out"
    (( r == 0 )) || [[ -z "$(worker_hf_cache "$r")" ]] ||
      die "rank $r ($where) lacks its packed shards in ENGRAM_PACKED_DIR: run scripts/engram-pack.sh on rank 0 and copy each worker its node's files, MANIFEST.sha256 and PACKED.json there (packing reads the checkpoint's Engram shards, which a worker with WORKER_HF_CACHE<i> does not mount over NFS): $out"
    spec=$(packed_files "$node" "$nodes") || die "cannot read Engram's tables' headers in $SNAP"
    pneed=$(awk '{s += $2} END {print s + 0}' <<<"$spec")
    names=$(awk '{print $1}' <<<"$spec" | xargs)
    pack="docker run --rm --network none --user \$(id -u):\$(id -g) -e HOME=/tmp"
    if (( r == 0 )); then
      dir=$(packed_dir_here)
      mkdir -p "$dir"
      have=0; for f in $names; do [[ -f "$dir/$f" ]] && have=$(( have + $(stat -c %s "$dir/$f") )); done
      free=$(df -B1 --output=avail "$dir" | tail -1)
      run() { bash -c "$pack -v '$HF_CACHE:/hf:ro' -v '$dir:/out' --entrypoint tf-dsv41-engram-pack '$IMAGE' '$SNAP_C' /out $node $nodes"; }
      verify() { python3 -I scripts/engram-verify.py verify "$node" "$nodes" "$dir" "${tables[@]}" --manifest "$dir/MANIFEST.sha256" --ok "$(engram_ok 0)"; }
    else
      have=$(worker "$r" "s=0; for f in $names; do [[ -f $ENGRAM_PACKED_DIR/\$f ]] && s=\$(( s + \$(stat -c %s $ENGRAM_PACKED_DIR/\$f) )); done; echo \$s")
      free=$(worker "$r" "mkdir -p $ENGRAM_PACKED_DIR && df -B1 --output=avail $ENGRAM_PACKED_DIR | tail -1")
      run() { worker "$r" "$pack -v '$NFS_VOLUME:/hf:ro' -v $ENGRAM_PACKED_DIR:/out --entrypoint tf-dsv41-engram-pack '$IMAGE' '$SNAP_C' /out $node $nodes"; }
      verify() { engram_verify_nfs "$r" "$node" "$nodes"; }
    fi
    (( free + have > pneed + 10 * 1024 ** 3 )) || die "$where has $(( free / 1024 ** 3 )) GiB free: Engram's packed shards need $(( pneed / 1024 ** 3 )) GiB (ENGRAM_PACKED_DIR) and 10 GiB to spare"
    log "    $where: packing Engram's tables ($(( pneed / 1024 ** 3 )) GiB, once), then checking every record"
    run >/dev/null || die "packing Engram's shards on $where failed"
    vout=$(verify) || die "the packed Engram shards on $where differ from the checkpoint's tables: $vout"
    engram_py "$r" record "$(engram_dir "$r")" "$MODEL_ID@$(snapshot_rev)" "$nodes" "$(image_ident "$IMAGE")" "${sources[@]}" \
      $(tr ' ' ':' <<<"$spec") || die "cannot write PACKED.json on $where"
    out=$(engram_check "$r") || die "rank $r ($where): its packed Engram shards cannot be read as they are after packing: $out"
    detail "Rank $r ($where): packed and verified ($(paste -sd';' - <<<"$out" | sed 's/;/; /g'))"
    log "    $where: Engram's packed tables: packed and checked"
  done
  rm -rf -- "$ck"; trap - EXIT
fi
for i in 1 2 3; do
  # with packed shards a worker that reads over NFS reads the tables' headers there; a worker with its own copy reads
  # them from its own copy of the tables (start.sh mounts it at /engram), which stays checked
  [[ "$ENGRAM_PACKED" != 1 || -n "$(worker_hf_cache "$i")" ]] || continue
  have=$(worker "$i" "d=$WORKER_ENGRAM_DIR; s=0; for f in ${shards[*]}; do [[ -f \$d/\$f ]] && s=\$(( s + \$(stat -c %s \$d/\$f) )); done; echo \$s")
  if [[ "$have" == "$need" ]]; then detail "Worker $i ($(worker_host "$i")): Engram tables on its disk"; continue; fi
  [[ -z "$(worker_hf_cache "$i")" ]] ||
    die "$(worker_host "$i") lacks Engram's tables in $WORKER_ENGRAM_DIR, and the copy reads them over NFS, which a worker with WORKER_HF_CACHE<i> does not mount: put them there first"
  free=$(worker "$i" "mkdir -p $WORKER_ENGRAM_DIR && df -B1 --output=avail $WORKER_ENGRAM_DIR | tail -1")
  (( free + have > need + 10 * 1024 ** 3 )) || die "$(worker_host "$i") has $(( free / 1024 ** 3 )) GiB free: Engram's tables need $(( need / 1024 ** 3 )) GiB on its disk (WORKER_ENGRAM_DIR) and 10 GiB to spare"
  log "  $(worker_host "$i"): copying Engram's tables to its disk ($(( need / 1024 ** 3 )) GiB, over NFS, once)"
  worker "$i" "docker run --rm --network none --user \$(id -u):\$(id -g) -v '$NFS_VOLUME:/hf:ro' -v $WORKER_ENGRAM_DIR:/out --entrypoint bash '$IMAGE' -c 'for f in ${shards[*]}; do cp /hf/$SNAP_REL/\$f /out/\$f.part && mv /out/\$f.part /out/\$f || exit 1; done'" ||
    die "copying Engram's tables to $(worker_host "$i") failed"
done

# ---------------------------------------------------------------- 8. each Spark's weight file's digest
# The SSD tier's store is named from every rank's weight file (scripts/nodes.sh): each Spark hashes its own (O_DIRECT,
# all four at once, each under WEIGHTS_SHA_TIMEOUT; cached, so a file hashed before is not read again) for start.sh to
# read. A digest that cannot be had is said, never a failure here: that start's tier is then off, named.
weights_hash_all

write_prepared_marker || die "cannot record the prepared state (the previous marker is left as it was): see the error above"
log "  All four Sparks are ready$( [[ ${SETUP_STARTS:-0} == 1 ]] || echo ". Start the engine with ./start.sh (port $PORT)")"
