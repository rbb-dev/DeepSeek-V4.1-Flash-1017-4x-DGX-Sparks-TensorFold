# The Sparks: this machine (rank 0), and the workers over ssh: WORKER1 (rank 1), WORKER2 (rank 2), WORKER3 (rank 3).
# Each node's links to the others are found from the routes and subnets between them: every pair on one subnet (a
# switch; see detect_links).
# Sourced by start.sh, stop.sh and prepare.sh after config.sh.
# From MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold's scripts/nodes.sh (MiaAI-Lab, with webzone's PCIe-twin
# rails and Alexbob0's worker host names; see CREDITS.md), with that engine's own exchange variables taken out.

# wvar <NAME> <i>: the name of worker i's own setting, NAME<i> for every worker: WORKER1, FABRIC_PEER1, NFS_SERVER1,
# WORKER_HF_CACHE1 for worker 1; WORKER2 ... for worker 2; WORKER3 ... for worker 3.
wvar() { echo "$1$2"; }
wval() { local v; v=$(wvar "$1" "$2"); echo "${!v:-}"; }
worker_host() { wval WORKER "$1"; }
# The workers this start uses (1 .. TP-1), and every worker configured at all (stop.sh stops them all).
worker_ids() { seq 1 $((TP - 1)); }
configured_workers() { local i; for i in 1 2 3; do [[ -z "$(worker_host "$i")" ]] || echo "$i"; done; }

# check_workers: TP is 2, 3 or 4, and WORKER1 .. WORKER<TP-1> are set and distinct (later ones are left out).
check_workers() {
  local i j h
  [[ "$TP" == 4 ]] || die "this recipe serves four Sparks (TP=4), not $TP"
  for i in $(worker_ids); do
    h=$(worker_host "$i")
    [[ -n "$h" ]] || die "TP=$TP needs $((TP - 1)) workers: set $(wvar WORKER "$i")=user@<address of rank $i> in scripts/local.sh (see scripts/local.sh.example)"
    for (( j = 1; j < i; j++ )); do
      [[ "$h" != "$(worker_host "$j")" ]] || die "$(wvar WORKER "$i") and $(wvar WORKER "$j") are the same node ($h)"
    done
  done
}

# kit_extra <kit folder>: whether the kit carries the extra set (kit/aot-extra/aot.json) that TF_DS_ATTN_U, TF_DS_IDX_U
# and TF_DS_DRAFT_4WAY load
kit_extra() { [[ -s "$1/kit/aot-extra/aot.json" ]]; }

# DRY_RUN=1 goes on without workers it cannot reach (a fictitious WORKER2): their values print as <WORKER2:...>.
declare -A WORKER_DOWN=()
# worker <i> <command...>: run a command on worker i (rank i). Key-based ssh only, no prompts. One connection a
# worker serves every command of a start (ControlMaster; it closes 60 s after the last): a start runs dozens, and
# each new connection is a key exchange.
worker() {
  local i=$1; shift
  [[ -z "${WORKER_DOWN[$i]:-}" ]] || return 255
  # the launchers' locks (start.lock on fd 8, prepare.lock on fd 9) stay with the launcher: the ssh connection's
  # master outlives this call (ControlPersist) and would otherwise hold them and refuse the next start
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 \
      -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%C" -o ControlPersist=60 "$(worker_host "$i")" "$@" 8>&- 9>&-
}

# save_log <dir> <rank> <container> <keep>: the container's log (stdout and stderr, with timestamps) gzipped into
# <dir>/<date>-<time>-rank<N>.log.gz before the container is removed (docker rm deletes it), keeping the newest <keep>
# of that rank; prints the file. Runs on the worker too (its definition is sent over ssh): no config.sh there.
save_log() {
  local dir=$1 rank=$2 c=$3 keep=$4 stamp f i=0
  [[ "$keep" =~ ^[0-9]+$ ]] && (( keep > 0 )) || return 0
  docker inspect "$c" >/dev/null 2>&1 || return 0
  mkdir -p "$dir" || return 1
  stamp=$(date +%Y%m%d-%H%M%S); f="$dir/$stamp-rank$rank.log.gz"
  while [[ -e "$f" ]]; do f="$dir/$stamp.$((++i))-rank$rank.log.gz"; done   # same second: still sorts as newer
  if ! docker logs --timestamps "$c" 2>&1 | gzip > "$f"; then rm -f "$f"; return 1; fi
  ls -1 "$dir"/*-rank"$rank".log.gz 2>/dev/null | { grep -vxF "$f" || true; } | LC_ALL=C sort -r | tail -n +"$keep" | xargs -r rm -f --
  echo "$f"
}
# worker_save_log <i>: the same, for rank i on worker i (into ~/.cache/dsv41-tensorfold/logs there)
worker_save_log() {
  worker "$1" "$(declare -f save_log); save_log \"\$HOME/.cache/dsv41-tensorfold/logs\" $1 '$CONTAINER_NAME' '$LOG_KEEP'"
}

need_worker() {
  local i=$1 v h
  v=$(wvar WORKER "$i"); h=$(worker_host "$i")
  [[ -n "$h" ]] || die "$v is not set: put $v=user@<worker address> in scripts/local.sh (see scripts/local.sh.example)"
  worker "$i" true 2>/dev/null && return 0
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    warn "DRY_RUN: cannot ssh to $h ($v): its values print as <$v:...>"
    WORKER_DOWN[$i]=1
    return 0
  fi
  die "cannot ssh to $h without a password: set up key-based ssh (ssh-copy-id $h)"
}
need_workers() { local i; for i in $(worker_ids); do need_worker "$i"; done; }

# Weights over NFS: rank i mounts the head's HF_CACHE read-only through the docker volume NFS_VOLUME on worker i, from
# nfs_server i: NFS_SERVER<i>, else the head's address on that worker's link (detect_links).
# ensure_nfs_volume i creates it (or checks the one there names the same export); worker_nfs i <cmd...> runs a command
# in a throwaway container on worker i with the volume at /hf.
nfs_server() { local s; s=$(wval NFS_SERVER "$1"); echo "${s:-${LINK_HEAD_ADDR[$1]:-}}"; }
ensure_nfs_volume() {
  local i=$1 server have h
  server=$(nfs_server "$i"); h=$(worker_host "$i")
  [[ -n "$server" ]] || die "no NFS server address for $h: set $(wvar NFS_SERVER "$i") to the head's address on that worker's link"
  have=$(worker "$i" "docker volume inspect -f '{{index .Options \"device\"}} {{index .Options \"o\"}}' '$NFS_VOLUME'" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    worker "$i" "docker volume create --driver local --opt type=nfs --opt device=':$NFS_PATH' \
      --opt o='addr=$server,nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600' '$NFS_VOLUME'" >/dev/null ||
      die "could not create the docker volume $NFS_VOLUME on $h"
  elif [[ "${have%% *}" != ":$NFS_PATH" || "$have" != *"addr=$server,"* ]]; then
    die "$h's docker volume $NFS_VOLUME is ${have%% *} from ${have#* }, not :$NFS_PATH from $server: remove it there (docker volume rm $NFS_VOLUME) or set NFS_VOLUME / NFS_PATH / $(wvar NFS_SERVER "$i")"
  fi
}
worker_nfs() { local i=$1; shift; worker "$i" "docker run --rm --network none --entrypoint '$1' -v '$NFS_VOLUME:/hf:ro' '$IMAGE' $(printf '%q ' "${@:2}")"; }

# Or from each worker's own disk: WORKER_HF_CACHE<i>, a path on worker i holding what rank
# i reads, laid out as HF_CACHE is here: the checkpoint's small files (its weights are in the weight file), the overlay's
# links, the weight file of the 2D node rank i serves, Engram's token map, the kit. Rank i binds it at /hf; no NFS.
worker_hf_cache() { wval WORKER_HF_CACHE "$1"; }
hf_mount() { local p; p=$(worker_hf_cache "$1"); if [[ -n "$p" ]]; then echo "-v $p:/hf:ro"; else echo "-v $NFS_VOLUME:/hf:ro"; fi; }
# listing <path...>: "path size" of every file under the paths, links followed (a link that leads nowhere is no file).
# Self-contained: it runs on the workers too (its definition is sent over ssh).
listing() { find -L "$@" -path '*/.git' -prune -o -type f -printf '%p %s\n' 2>/dev/null || true; }
# rank_reads <node> (stdin: listing lines): what a rank serving 2D node <node> reads: the checkpoint's files but its
# weight shards (*.safetensors outside the kit), that node's weight file and no other, the token map, the kit
rank_reads() {
  awk -v kit="$KIT_SUB/kit/" -v rc="$DATA_SUB/rank-cache/" -v mine="$DATA_SUB/rank-cache/rank${1}of4-" '
    { p = substr($0, 1, length($0) - length($NF) - 1) }
    index(p, kit) != 1 && p ~ /\.safetensors$/ { next }
    index(p, rc) == 1 && index(p, mine) != 1 { next }
    { print }' | LC_ALL=C sort
}
# check_worker_files <i>: worker i's own copy holds every file rank i reads, at this Spark's size, and no other weight
# file of its node; prints what differs ("<" here, ">" there) and fails when anything does
check_worker_files() {
  local i=$1 p node want have
  local -a roots=("hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)" "$DATA_SUB" "$KIT_SUB/kit")
  p=$(worker_hf_cache "$i")
  [[ -n "$p" ]] || return 0
  node=$(cut -d, -f$((i + 1)) <<<"$(node_order)")
  want=$(cd "$HF_CACHE" && listing "${roots[@]}" | rank_reads "$node") || return 1
  have=$( { worker "$i" "cd $(printf '%q' "$p") && $(declare -f listing); listing $(printf '%q ' "${roots[@]}")" || true; } | rank_reads "$node")
  [[ "$have" == "$want" ]] && return 0
  { diff <(printf '%s\n' "$want") <(printf '%s\n' "$have") | grep '^[<>]' | head -20; } || true
  return 1
}

# ---------------------------------------------------------------- Engram's packed shards (ENGRAM_PACKED=1)
# Each Spark keeps the shards of the node its rank reads (engram_pack.zig: one a layer, each row's FP8 values and
# scales side by side) in its ENGRAM_PACKED_DIR, beside MANIFEST.sha256 (every shard's sha256, from its check against the
# tables) and PACKED.json (the checkpoint they are cut from, its Engram files' blobs, each shard's size). A shard is read
# as it is only when that Spark's engram-packed.ok holds its verified line: the file's path, size, times, inode and
# device, and the manifest's sha256 (scripts/engram-verify.py writes it when it checks the file against the tables,
# record by record, or reads it once against the manifest).
# engram_py <rank> <args...>: scripts/engram-verify.py on rank <rank>'s Spark: here as a file, on a worker from stdin (a
# worker holds no copy of this recipe). A path that starts with a literal $HOME/ is that Spark's home.
engram_py() {
  local r=$1
  shift
  if (( r == 0 )); then python3 -I "$_cfg_root/scripts/engram-verify.py" "$@"
  else worker "$r" "python3 -I - $(printf '%q ' "$@")" < "$_cfg_root/scripts/engram-verify.py"; fi
}
# engram_node <rank>: the node whose shards rank <rank> reads: its 2D node (TF_DS_2D_ORDER) when they are cut for four,
# else 0 (the whole tables)
engram_node() { if [[ "$(packed_nodes)" == 1 ]]; then echo 0; else cut -d, -f$(($1 + 1)) <<<"$(node_order)"; fi; }
# engram_dir <rank>, engram_ok <rank>: that Spark's ENGRAM_PACKED_DIR and its verified lines (engram-packed.ok in its
# state folder: here STATE_DIR, on a worker its own ~/.local/state/dsv41-tensorfold)
engram_dir() { if (( $1 == 0 )); then packed_dir_here; else echo "$ENGRAM_PACKED_DIR"; fi; }
engram_ok() { if (( $1 == 0 )); then echo "$STATE_DIR/engram-packed.ok"; else echo '$HOME/.local/state/dsv41-tensorfold/engram-packed.ok'; fi; }
# engram_specs <node> <nodes>: packed_files' lines as engram-verify.py's shard specs (the fields joined by ':')
engram_specs() { packed_files "$1" "$2" | tr ' ' ':'; }
# engram_sources: "--source" and "<file>=<blob>" lines for each of the checkpoint's Engram files (the blob its snapshot
# link names: the file's content)
engram_sources() {
  local snap f
  snap="$(model_cache_dir)/snapshots/$(snapshot_rev)"
  for f in $(engram_shards); do printf -- '--source\n%s=%s\n' "$f" "$(basename "$(readlink -f "$snap/$f")")"; done
}
# engram_check <rank> [--hash]: whether rank <rank>'s packed shards may be read as they are (engram-verify.py check: each
# at its exact size from the tables' headers and with its header, MANIFEST.sha256 and PACKED.json beside them for this
# checkpoint, rank 0's from its blobs, each with its verified line; --hash: a file without one is read once against the
# manifest and its line written). Prints what it found; exit 0 ready, 3 a shard missing or of another size, else 1.
engram_check() {
  local r=$1 nodes specs
  shift
  nodes=$(packed_nodes)
  specs=$(engram_specs "$(engram_node "$r")" "$nodes") || { echo "cannot read Engram's tables' headers"; return 1; }
  local -a src=()
  (( r != 0 )) || mapfile -t src < <(engram_sources)
  engram_py "$r" check "$(engram_dir "$r")" "$(engram_ok "$r")" "$MODEL_ID@$(snapshot_rev)" "$nodes" "$@" "${src[@]}" $specs
}
# engram_verify_nfs <rank> <node> <nodes>: a worker that reads over NFS checks the shards it packed against the tables in
# the NFS volume, inside the image (no path on that worker holds the tables), merging the manifest beside them and its
# verified lines
engram_verify_nfs() {
  local r=$1 node=$2 nodes=$3 f rel
  rel="hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"
  local -a tables=()
  for f in $(engram_shards); do tables+=("/hf/$rel/$f"); done
  worker "$r" "mkdir -p \$HOME/.local/state/dsv41-tensorfold && docker run -i --rm --network none --user \$(id -u):\$(id -g) -e HOME=/tmp -v '$NFS_VOLUME:/hf:ro' -v $ENGRAM_PACKED_DIR:/out -v \$HOME/.local/state/dsv41-tensorfold:/state --entrypoint python3 '$IMAGE' -I - verify $node $nodes /out ${tables[*]} --manifest /out/MANIFEST.sha256 --ok /state/engram-packed.ok --host-dir $ENGRAM_PACKED_DIR" < "$_cfg_root/scripts/engram-verify.py"
}

# link_info <peer address>: this node's side of the link to <peer>: "<address> <netdev> <rdma device> <gid index>".
# The netdev and source address come from the route to the peer; the RDMA device from sysfs; the GID index is the
# RoCE v2 entry that holds this node's IPv4 address.
link_info() {
  local peer=$1 route dev src hca idx type gid want
  route=$(ip -o -4 route get "$peer" 2>/dev/null) || return 1
  dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  src=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  hca=$(ls /sys/class/net/"$dev"/device/infiniband 2>/dev/null | head -1)
  idx=""
  if [[ -n "$hca" ]]; then
    want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"$src"))
    for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      gid=$(cat "$f" 2>/dev/null) || continue
      [[ "$gid" == "$want" ]] || continue
      type=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)
      [[ "$type" == *"v2"* ]] && { idx=${f##*/}; break; }
    done
  fi
  echo "$src $dev ${hca:--} ${idx:--}"
}

# rails <netdev> <gid index>: the RoCE devices this node can run the link on - the link's own device first, then every
# other RoCE device whose RoCE v2 IPv4 GID sits at the same index: one on the link's subnet (a second CX7 port, when it
# is up and addressed there), and the second PCIe link of the same QSFP port (a DGX Spark's "twin", which lives in its
# own subnet, so the subnet scan below cannot see it).
rails() {
  local dev=$1 gid=$2 net cidr hcas other ip idx
  cidr=$(ip -o -4 addr show dev "$dev" | awk '{print $4}' | head -1)
  net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")
  hcas=$(ls /sys/class/net/"$dev"/device/infiniband | head -1)
  for n in /sys/class/net/*; do
    other=${n##*/}
    [[ "$other" == "$dev" || ! -d $n/device/infiniband || "$(cat $n/operstate 2>/dev/null)" != up ]] && continue
    ip=$(ip -o -4 addr show dev "$other" | awk '{print $4}' | head -1)
    [[ -n "$ip" ]] || continue
    python3 -c "import ipaddress,sys; sys.exit(ipaddress.ip_interface(sys.argv[1]) not in ipaddress.ip_network(sys.argv[2]) and ipaddress.ip_interface(sys.argv[1]).ip not in ipaddress.ip_network(sys.argv[2]))" "$ip" "$net" || continue
    idx=$(ls $n/device/infiniband | head -1)
    [[ "$(cat /sys/class/infiniband/$idx/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$idx"
  done
  # A DGX Spark's QSFP port reaches the GB10 over two independent PCIe Gen5 x4 links, so one cabled port shows up as
  # two netdevs and two RoCE devices (rocep1s0f0 and roceP2p1s0f0), and NVIDIA's own two-Spark playbook gives the two
  # twins different subnets - so the scan above never sees the twin and the rings get one x4 link: half the port.
  # Pair the twin by name instead of by subnet: same fN/npM tail, different PCIe prefix. Same conditions as above
  # (link up, a RoCE v2 GID at $gid), and never add a device twice.
  # (only for Spark-style names, <prefix>f<N>np<M>; the check below compares whole comma-separated names, so a twin the
  # subnet scan already added, as when both twins share the link's subnet, is not listed twice)
  local tail=${dev##*f} other2 ib2
  [[ "$dev" =~ f[0-9]+np[0-9]+$ ]] || { echo "$hcas"; return 0; }
  for n in /sys/class/net/*; do
    other2=${n##*/}
    [[ "$other2" == "$dev" || "$other2" != *f"$tail" ]] && continue
    [[ -d $n/device/infiniband && "$(cat $n/operstate 2>/dev/null)" == up ]] || continue
    ib2=$(ls $n/device/infiniband | head -1)
    [[ -n "$ib2" && ",$hcas," != *",$ib2,"* ]] || continue
    [[ "$(cat /sys/class/infiniband/$ib2/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$ib2"
  done
  echo "$hcas"
}

# rail_info <rdma device>: "<netdev> <IPv4 address> <MTU>" of the port a RoCE device sits on ("- - -" when it has none).
# Self-contained: it runs on the workers too (its definition is sent over ssh).
rail_info() {
  local n
  for n in /sys/class/infiniband/"$1"/device/net/*; do
    [[ -d $n ]] || continue
    echo "${n##*/} $(ip -o -4 addr show dev "${n##*/}" | awk '{print $4}' | head -1 | cut -d/ -f1) $(cat "$n/mtu" 2>/dev/null || echo 1500)"
    return 0
  done
  echo "- - -"
}

# rail_ping <local netdev> <peer address> <MTU>: one rail's traffic reaches the peer's twin - a full-size, unfragmentable
# ping bound to that rail's own port (-I <netdev>: an address alone would be routed out of whichever port of the same
# subnet the kernel picks, usually the cabled one), so an unaddressed, differently-subnetted or smaller-MTU twin fails
# here instead of as vendor_err 0x81 (transport retry exceeded) in the first all-gather (issue #66). Self-contained.
rail_ping() { ping -c2 -W2 -I "$1" -M do -s $(( $3 - 28 )) "$2" >/dev/null 2>&1; }

# net_mtu <netdev>: its MTU (1500 when sysfs has none). Self-contained.
net_mtu() { cat "/sys/class/net/$1/mtu" 2>/dev/null || echo 1500; }

# probe_links (TP>2): every rail between this node and a worker carries a full-size, unfragmentable ping both ways
# (rail_ping, at the smaller of the two ends' MTUs) before the engine's rings use it. pair_links pairs the
# ports by subnet alone, so an unaddressed or smaller-MTU port, a link that is down or a cable on another port would
# otherwise fail later, inside the start (as the second rail at two ranks did, issue #66). The rails are
# checked at once (a ping takes a second). A rail that does not answer stops the start, named; RAIL_CHECK=0 skips the
# check; a worker DRY_RUN cannot reach is skipped.
probe_links() {
  local line i hdev hip wdev wip hmtu dir k=0 bad mtus
  if [[ "${RAIL_CHECK:-1}" == 0 ]]; then log "RAIL_CHECK=0: the rails to the workers are not checked"; return 0; fi
  (( ${#RAILS[@]} > 0 )) || return 0
  dir=$(mktemp -d)
  for line in "${RAILS[@]}"; do
    read -r i hdev hip wdev wip <<<"$line"
    if [[ -n "${WORKER_DOWN[$i]:-}" ]]; then
      warn "DRY_RUN: the rail $hdev $hip <-> $wdev $wip to rank $i ($(worker_host "$i")) is not checked: the worker cannot be reached"
      continue
    fi
    k=$((k + 1)); hmtu=$(net_mtu "$hdev")
    (
      # the worker's way back first: it reads its own port's MTU, and the smaller of the two sizes both pings
      out=$(worker "$i" "$(declare -f net_mtu rail_ping); m=\$(net_mtu $wdev); echo \$m; (( m < $hmtu )) || m=$hmtu; rail_ping $wdev $hip \$m && echo back") || true
      wmtu=$(head -1 <<<"$out"); [[ "$wmtu" =~ ^[0-9]+$ ]] || wmtu=""
      mtu=$hmtu; [[ -z "$wmtu" ]] || (( wmtu >= hmtu )) || mtu=$wmtu
      if [[ -z "$out" ]]; then why="no ssh answer from $(worker_host "$i")"
      elif [[ "$out" != *back* ]]; then why="no answer from $wdev $wip to $hip"
      elif ! rail_ping "$hdev" "$wip" "$mtu"; then why="no answer from $hdev $hip to $wip"
      else echo "$mtu" > "$dir/$k.ok"; exit 0; fi
      echo "rank $i ($(worker_host "$i")): $hdev $hip (MTU $hmtu) <-> $wdev $wip (MTU ${wmtu:-?}): $why" > "$dir/$k.bad"
    ) &
  done
  wait
  bad=$(cat "$dir"/*.bad 2>/dev/null || true); mtus=$(sort -un "$dir"/*.ok 2>/dev/null | paste -sd/ - || true)
  rm -rf "$dir"
  [[ -z "$bad" ]] || die "a rail between this node and a worker does not carry a full-size ping both ways: $(paste -sd';' - <<<"$bad" | sed 's/;/; /g'). Check each port's address, subnet, MTU and link on both Sparks (ip -4 addr, ip link) and the cable, or set RAIL_CHECK=0 to start without this check"
  (( k == 0 )) || log "  Fabric links to the other Sparks: $k of $k answer full-size pings both ways (MTU $mtus)"
}

# The same function on worker i (its definition is sent over ssh).
worker_link_info() { worker "$1" "$(declare -f link_info); link_info $2"; }

# Per rank r, filled by detect_links: NODE_HCAS[r] (its RoCE devices toward its peers, the rings' devices), NODE_GID[r]
# (their RoCE v2 GID index, shown at the start; empty at TP>2 when they differ); LINK_HEAD_ADDR[i] / LINK_WORKER_ADDR[i]: the head's and worker i's addresses on their link (the
# head's is worker i's NFS server).
declare -a NODE_HCAS=() NODE_GID=() LINK_HEAD_ADDR=() LINK_WORKER_ADDR=()
# At TP>2, filled by detect_links from pair_links: RAILS, one "<i> <netdev> <address> <worker i's netdev> <its
# address>" a rail between this node and worker i (probe_links checks them).
declare -a RAILS=()

# An image's identity by content (its layers' diffIDs and runtime config), the same under Docker's overlay2 and
# containerd image stores: .Id is the config digest under one and the manifest digest under the other, so it never
# matches across a mixed pair (issue #8). The template holds no spaces: worker() passes it through ssh, which
# re-splits arguments. A missing image is "missing". worker_image_ident <i> <image>: on worker i.
IMAGE_IDENT='{{.RootFS.Layers}}{{.Config.Env}}{{.Config.Entrypoint}}{{.Config.Cmd}}{{.Config.WorkingDir}}'
image_ident() { local s; s=$(docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
worker_image_ident() {
  local s; s=$(worker "$1" docker image inspect -f "$IMAGE_IDENT" "$2" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing
}

# cx7_peer <i>: worker i's address on a CX7 port that one of this node's CX7 ports reaches directly (same subnet, no
# gateway), for a worker given by its LAN address (issue #9). Empty when there is none.
cx7_peer() {
  local a route dev
  for a in $(worker "$1" 'for d in /sys/class/net/*; do [ -d "$d/device/infiniband" ] && ip -o -4 addr show dev "${d##*/}"; done' 2>/dev/null |
             awk '{print $4}' | cut -d/ -f1); do
    route=$(ip -o -4 route get "$a" 2>/dev/null) || continue
    [[ "$route" == *" via "* ]] && continue
    dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
    [[ -d /sys/class/net/$dev/device/infiniband ]] && { echo "$a"; return 0; }
  done
  return 1
}

# probe_rails: before a second rail (the PCIe twin of the cabled port, or a second cabled port) is used, check that it
# reaches the peer. roce.py pairs the two nodes' devices by list position at two ranks, so a twin that is up and has a GID
# but no path to the peer's twin (no address, another subnet, another MTU, a cable on a different port) ends in vendor_err
# 0x81. Rail k of the head is probed against rail k of the worker, in both directions; a rail that fails is dropped from both
# lists with a warning (the first device, the link itself, is never probed: the launcher reached the worker over it).
probe_rails() {
  local -a h w keep_h=() keep_w=()
  local k hdev hip hmtu wdev wip wmtu mtu=0 ok
  IFS=, read -ra h <<<"$HEAD_HCAS"; IFS=, read -ra w <<<"$WORKER_HCAS"
  (( ${#h[@]} > 1 && ${#h[@]} == ${#w[@]} )) || return 0
  [[ -z "${WORKER_DOWN[1]:-}" ]] || { warn "DRY_RUN: the second rail ($HEAD_HCAS / $WORKER_HCAS) is not probed: the worker cannot be reached"; return 0; }
  keep_h=("${h[0]}"); keep_w=("${w[0]}")
  for (( k = 1; k < ${#h[@]}; k++ )); do
    read -r hdev hip hmtu <<<"$(rail_info "${h[k]}")"
    read -r wdev wip wmtu <<<"$(worker 1 "$(declare -f rail_info); rail_info ${w[k]}")"
    ok=0
    if [[ "$hip" =~ ^[0-9.]+$ && "$wip" =~ ^[0-9.]+$ && "$hmtu" =~ ^[0-9]+$ && "$wmtu" =~ ^[0-9]+$ ]]; then
      mtu=$(( hmtu < wmtu ? hmtu : wmtu ))
      if rail_ping "$hdev" "$wip" "$mtu" && worker 1 "$(declare -f rail_ping); rail_ping $wdev $hip $mtu"; then ok=1; fi
    fi
    if (( ok )); then
      keep_h+=("${h[k]}"); keep_w+=("${w[k]}")
      log "Rail ${h[k]} <-> ${w[k]}: $hip <-> $wip reaches the peer (MTU $mtu): used"
    else
      warn "second rail ${h[k]} (${hip:--}, MTU ${hmtu:--}) <-> ${w[k]} (${wip:--}, MTU ${wmtu:--}) does not reach the peer: NOT used, running on fewer rails. Give both twins an IPv4 address on one subnet per rail with the same MTU on both Sparks (README), or set RAIL_COUNT=1 to silence this"
    fi
  done
  HEAD_HCAS=$(IFS=,; echo "${keep_h[*]}"); WORKER_HCAS=$(IFS=,; echo "${keep_w[*]}")
}

# TP=2: HEAD_ADDR / HEAD_DEV / HEAD_HCA / HEAD_GID and WORKER_ADDR / WORKER_DEV / WORKER_HCA / WORKER_GID: the link the
# two ranks talk over (the rings and the rendezvous). FABRIC_PEER1 overrides the worker's link address when WORKER1 is reached
# over another network.
detect_link() {
  local peer=${FABRIC_PEER1:-${WORKER1#*@}} cx7
  # a host name (an /etc/hosts alias of the CX7 address, say) is resolved first: ``ip route get`` takes addresses only
  if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    peer=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
    [[ -n "$peer" ]] || die "cannot resolve ${FABRIC_PEER1:-${WORKER1#*@}} to an IPv4 address"
  fi
  read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  [[ -n "${HEAD_ADDR:-}" ]] || die "no route from this node to $peer"
  # WORKER1 given by a LAN address (the route goes out a port without RoCE): use the worker's CX7 address instead
  if [[ -z "${FABRIC_PEER1:-}" && "${HEAD_HCA:--}" == "-" ]] && cx7=$(cx7_peer 1); then
    log "$peer is reached over $HEAD_DEV, not a CX7 port: using the worker's CX7 address $cx7 for the link (FABRIC_PEER1)"
    peer=$cx7
    read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  fi
  read -r WORKER_ADDR WORKER_DEV WORKER_HCA WORKER_GID <<<"$(worker_link_info 1 "$HEAD_ADDR")" || true
  [[ -n "${WORKER_ADDR:-}" ]] || die "the worker has no route back to $HEAD_ADDR"
  # every rail (both PCIe twins of the cabled port, a second cabled port): a chunk's all-gather is ~1.8x faster on two
  HEAD_HCAS=$(rails "$HEAD_DEV" "$HEAD_GID")
  WORKER_HCAS=$(worker 1 "$(declare -f rails); rails $WORKER_DEV $WORKER_GID")
  [[ "${RAIL_COUNT:-2}" == 1 ]] && { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  [[ "${RAIL_COUNT:-2}" == 1 ]] || probe_rails
  [[ "$(tr ',' '\n' <<<"$HEAD_HCAS" | wc -l)" == "$(tr ',' '\n' <<<"$WORKER_HCAS" | wc -l)" ]] ||
    { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  for v in HEAD_HCA HEAD_GID WORKER_HCA WORKER_GID; do
    [[ "${!v}" != "-" ]] || die "no RoCE device or RoCE v2 GID for the link ($v): is $HEAD_DEV / $WORKER_DEV the CX7 port between the Sparks? Set FABRIC_PEER to the worker's CX7 address"
  done
  NODE_HCAS=("$HEAD_HCAS" "$WORKER_HCAS"); NODE_GID=("$HEAD_GID" "$WORKER_GID")
  LINK_HEAD_ADDR[1]=$HEAD_ADDR; LINK_WORKER_ADDR[1]=$WORKER_ADDR
  MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
}

# node_inventory [master]: this node's default-route netdev ("default <netdev>"), whether it routes to <master>
# ("master ok|none"), and one line per RoCE netdev that is up with an IPv4 address: "<netdev> <rdma device>
# <address/prefix> <RoCE v2 GID index of that address, or ->". Read-only (ip and sysfs).
node_inventory() {
  local n dev hca cidr want idx f
  echo "default $(ip -o -4 route show default 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)"
  [[ -z "${1:-}" ]] || { ip -o -4 route get "$1" >/dev/null 2>&1 && echo "master ok" || echo "master none"; }
  for n in /sys/class/net/*; do
    dev=${n##*/}
    [[ -d $n/device/infiniband && "$(cat "$n/operstate" 2>/dev/null)" == up ]] || continue
    hca=$(ls "$n/device/infiniband" | head -1)
    for cidr in $(ip -o -4 addr show dev "$dev" | awk '{print $4}'); do
      want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"${cidr%/*}"))
      idx=-
      for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
        [[ "$(cat "$f" 2>/dev/null)" == "$want" ]] || continue
        [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)" == *v2* ]] && { idx=${f##*/}; break; }
      done
      echo "$dev $hca $cidr $idx"
    done
  done
}

# pair_links <ranks...> (stdin: node_inventory lines, each prefixed with its rank): two nodes are linked where they
# have addresses in one subnet (a direct cable, or a switch). Prints "rank <r> <default netdev> <hcas> <gid or ->" (a
# node's RoCE devices toward all its peers, in its sysfs order; the GID index when it is one for all of them), "peer <i>
# <worker i's lowest address on its link to the head>", "missing <a> <b>" for two ranks without a common subnet, and
# "prefix <len>": the shortest prefix length
# (longer than the links' own) at which, on every link, each device's address and exactly one of the peer's share a
# subnet and no two of a node's devices do (a port and its PCIe twin addressed .N and .10N in one /24: 26); and "rail <i> <netdev> <address> <i's netdev> <i's address>" for
# each of this node's (rank 0's) addresses toward worker i, the peer's address its counterpart (on a port and its
# twin, the one in the same subnet at that prefix).
pair_links() {
  python3 -c '
import ipaddress, sys
nodes, default = {}, {}
for line in sys.stdin:
    f = line.split()
    if len(f) == 3 and f[1] == "default":
        default[int(f[0])] = f[2]
    elif len(f) == 5:
        nodes.setdefault(int(f[0]), []).append((f[1], f[2], ipaddress.ip_interface(f[3]), f[4]))
ranks = [int(x) for x in sys.argv[1:]]
linked = lambda e, others: any(e[2].network == o[2].network for o in others)
links = {r: set() for r in ranks}
rails = []
for r in ranks:
    used = set()
    for s in ranks:
        if s == r:
            continue
        mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
        if not mine:
            if r < s:
                print("missing", r, s)
            continue
        links[r].add(s)
        used.update(id(e) for e in mine)
        if r == 0:
            print("peer", s, min(o[2].ip for o in nodes[s] if linked(o, mine)))
            rails.append((s, mine))
    hcas, gids = [], set()
    for e in nodes.get(r, []):              # in the node sysfs order
        if id(e) in used and e[1] not in hcas:
            hcas.append(e[1]); gids.add(e[3])
    print("rank", r, default.get(r) or "-", ",".join(hcas) or "-", gids.pop() if len(gids) == 1 else "-")
def splits(n):
    for r in ranks:
        for s in links[r]:
            mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
            theirs = [o for o in nodes.get(s, []) if linked(o, mine)]
            nets = [ipaddress.ip_interface(f"{e[2].ip}/{n}").network for e in mine]
            if len(set(nets)) != len(nets) or any(sum(o[2].ip in net for o in theirs) != 1 for net in nets):
                return False
    return True
own = max((e[2].network.prefixlen for r in ranks for e in nodes.get(r, [])), default=32)
pre = next((n for n in range(own, 31) if splits(n)), None)
if pre is not None:
    print("prefix", pre)
for s, mine in rails:
    for e in mine:
        theirs = [o for o in nodes[s] if o[2].network == e[2].network]
        if pre is not None:
            theirs = [o for o in theirs if o[2].ip in ipaddress.ip_interface(f"{e[2].ip}/{pre}").network] or theirs
        o = min(theirs, key=lambda o: o[2].ip)
        print("rail", s, e[0], e[2].ip, o[0], o[2].ip)
' "$@"
}

# TP>2: every node's RoCE netdevs, paired by subnet (pair_links); a node's RoCE devices (the rings') are the union of
# its devices toward all its peers. The rendezvous is MASTER_ADDR (config.sh). Every pair of ranks must share a RoCE
# subnet: a switch, or at three Sparks a triangle of direct cables.
detect_links() {
  if (( TP == 2 )); then detect_link; return; fi
  local r i peer kind a b c d e pairs line missing="" names
  [[ -n "$MASTER_ADDR" ]] || die "no MASTER_ADDR: set it to an address of this node that every worker reaches (its LAN address)"
  local -a inv=()
  inv[0]=$(node_inventory)
  for i in $(worker_ids); do
    inv[i]=$(worker "$i" "$(declare -f node_inventory); node_inventory $MASTER_ADDR" 2>/dev/null || true)
    [[ "${inv[i]}" != *"master none"* ]] ||
      die "$(worker_host "$i") has no route to MASTER_ADDR $MASTER_ADDR: set MASTER_ADDR to an address of this node it reaches"
  done
  pairs=$(for r in "${!inv[@]}"; do sed "s/^/$r /" <<<"${inv[r]}"; done | pair_links 0 $(worker_ids)) ||
    die "could not pair the nodes' links"
  RAILS=()
  while read -r kind a b c d e; do
    case "$kind" in
      rank)
        NODE_HCAS[a]=$c; NODE_GID[a]=$d
        [[ "$d" != - ]] || NODE_GID[a]="" ;;
      peer)
        # the head's side of that link from its route to the worker's address there (FABRIC_PEER<i> overrides it)
        peer=$(wval FABRIC_PEER "$a"); peer=${peer:-$b}
        # a host name (an /etc/hosts alias of the CX7 address, say) is resolved first, as at TP=2
        if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
          c=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
          [[ -n "$c" ]] || die "cannot resolve $peer ($(wvar FABRIC_PEER "$a")) to an IPv4 address"
          peer=$c
        fi
        read -r LINK_HEAD_ADDR[a] _ <<<"$(link_info "$peer")" || true
        LINK_WORKER_ADDR[a]=$peer ;;
      missing) missing+=" $a-$b" ;;
      prefix) ;;   # (pair_links' subnet prefix length: nothing reads it any more)
      rail) RAILS+=("$a $b $c $d $e") ;;
    esac
  done <<<"$pairs"
  for r in 0 $(worker_ids); do
    if [[ -n "${WORKER_DOWN[$r]:-}" ]]; then
      names=$(wvar WORKER "$r")
      NODE_HCAS[r]="<$names:hcas>"; NODE_GID[r]="<$names:gid>"
      LINK_HEAD_ADDR[r]="<head address toward $names>"; LINK_WORKER_ADDR[r]="<$names:address>"
      continue
    fi
    names="this node"; (( r == 0 )) || names=$(worker_host "$r")
    [[ -n "${NODE_HCAS[r]:-}" && "${NODE_HCAS[r]}" != - ]] ||
      die "rank $r ($names) shares no RoCE subnet with any other rank: check the CX7 cabling and addresses (ip -4 addr)"
  done
  for line in $missing; do
    a=${line%-*}; b=${line#*-}
    if [[ -n "${WORKER_DOWN[$a]:-}${WORKER_DOWN[$b]:-}" ]]; then
      warn "DRY_RUN: the link between ranks $a and $b is unknown (a worker that cannot be reached)"
      continue
    fi
    die "ranks $a and $b share no RoCE subnet: every pair of the $TP Sparks needs one (a switch, or at three Sparks a triangle of direct cables); see NETWORK.md"
  done
  probe_links
  return 0
}

# The engine's TF_DS_2D_ORDER (config.sh ORDER): the 2D node each rank serves.
node_order() {
  case "$ORDER" in
    auto) echo 0,1,2,3 ;;
    *) echo "$ORDER" ;;
  esac
}

# ---------------------------------------------------------------- the SSD tier (every Spark)
# The tier keeps prompt states on each Spark's own disk: every rank keeps a store of its own, and the settings reach every
# rank (a worker gets them from spill_rank_env, with the tier asked; with it off, TF_DS_SPILL_GIB=0 alone).
# scripts/config.sh's settings go to rank 0 and the workers by the engine's names:
#   SPILL_GIB -> TF_DS_SPILL_GIB              SPILL_FLOOR_GIB -> TF_DS_SPILL_MIN_FREE_GIB   SPILL_QUOTA -> TF_DS_SPILL_QUOTA
#   SPILL_MIN_TOKENS -> TF_DS_SPILL_MIN_TOKENS SPILL_STAGING_MIB -> TF_DS_SPILL_STAGING_MIB SPILL_TURNS -> TF_SPILL_TURNS
#   SPILL_FLUSH_MS -> TF_SPILL_FLUSH_MS       the reason the recipe turned it off -> TF_DS_SPILL_OFF_REASON
# and every other TF_DS_SPILL_* / TF_SPILL_* exported passes through to every rank as it is. ENGINE_ENV (every rank, start.sh)
# never carries one. The values are the engine's to check: a bad one turns the tier off, named (never a refused start).
SPILL_MOUNT=/spill   # the tier's folder inside every rank's container (TF_DS_SPILL_DIR)
# the engine names start.sh sends itself: an exported one does not pass through beside them
SPILL_MAPPED='TF_DS_SPILL_(GIB|MIN_FREE_GIB|DIR|QUOTA|MIN_TOKENS|STAGING_MIB|OFF_REASON)|TF_SPILL_(TURNS|FLUSH_MS)'
SPILL_OFF_REASON=""   # why the recipe turned the tier off for this start (spill_start), sent as TF_DS_SPILL_OFF_REASON
# spill_asked: the tier is asked for (SPILL_GIB other than 0; a value that is no number is asked too: the engine names it)
spill_asked() { [[ ! "$SPILL_GIB" =~ ^0*([.]0*)?$ ]]; }
# spill_off <reason> [<what to do>]: the tier off for this start: SPILL_GIB=0, the reason kept for rank 0
# (TF_DS_SPILL_OFF_REASON, at most 120 bytes: the engine's reason field) and said loudly. The engine starts as without
# the tier.
spill_off() {
  SPILL_GIB=0
  SPILL_REAL=""
  SPILL_OFF_REASON=$(printf '%.120s' "$1")
  warn "SSD TIER OFF for this start: $1${2:+. $2}. The engine starts without it (kept prompt states stay in memory only, gone after a restart); /health names the reason"
}
# spill_start (start.sh, step 2, before anything is stopped): what decides the tier for this start, said once:
# TF_DS_SPILL_OFF_REASON exported (an outside watchdog after repeated restarts, say) turns it off with that reason; an
# exported engine name of a setting that differs from the setting is not used (said); the stop's bounds are compared
# (said, not refused).
spill_start() {
  local pair rname ename have
  [[ -z "${TF_DS_SPILL_OFF_REASON:-}" ]] || spill_off "$TF_DS_SPILL_OFF_REASON" "TF_DS_SPILL_OFF_REASON was exported"
  for pair in SPILL_GIB:TF_DS_SPILL_GIB SPILL_FLOOR_GIB:TF_DS_SPILL_MIN_FREE_GIB SPILL_QUOTA:TF_DS_SPILL_QUOTA \
              SPILL_MIN_TOKENS:TF_DS_SPILL_MIN_TOKENS SPILL_STAGING_MIB:TF_DS_SPILL_STAGING_MIB SPILL_TURNS:TF_SPILL_TURNS \
              SPILL_FLUSH_MS:TF_SPILL_FLUSH_MS; do
    rname=${pair%%:*}; ename=${pair#*:}
    have=$(printenv "$ename" 2>/dev/null) || continue
    [[ "$have" == "${!rname}" ]] || warn "$ename=$have (exported) is not used: $rname=${!rname} is the recipe's setting, sent as $ename"
  done
  have=$(printenv TF_DS_SPILL_DIR 2>/dev/null) && [[ "$have" != "$SPILL_MOUNT" ]] &&
    warn "TF_DS_SPILL_DIR=$have (exported) is not used: it is the folder inside each rank's container ($SPILL_MOUNT); set SPILL_DIR to this Spark's folder"
  for have in $(compgen -e | grep -E "^($SPILL_IDENT)\$" || true); do
    warn "$have (exported) is not used: the SSD tier's identity is derived (the image, the checkpoint, the overlay, Engram, the kit, each rank's weight file), never typed"
  done
  ! spill_asked || spill_check
  if spill_asked; then
    # the store's name needs the checkpoint's revision (two checkpoints must not share one identity)
    if [[ -z "$(snapshot_rev)" ]]; then spill_off "no checkpoint revision" "MODEL_REVISION is empty and $(model_cache_dir)/refs/main names none"
    else weights_digests; fi
  fi
  if [[ "$TENSORFOLD_CLOSE_MS" =~ ^[0-9]+$ ]] && (( 10#$TENSORFOLD_CLOSE_MS + 2000 >= STOP_TIMEOUT * 1000 )); then
    warn "TENSORFOLD_CLOSE_MS ($TENSORFOLD_CLOSE_MS) and the server's 2 s wait for its connections reach STOP_TIMEOUT ($STOP_TIMEOUT s): docker may kill rank 0 before its orderly close ends (the followers then end PeerDown)"
  fi
  if spill_asked && [[ "$SPILL_FLUSH_MS" =~ ^[0-9]+$ && "$TENSORFOLD_CLOSE_MS" =~ ^[0-9]+$ ]] && (( 10#$SPILL_FLUSH_MS >= 10#$TENSORFOLD_CLOSE_MS )); then
    warn "SPILL_FLUSH_MS ($SPILL_FLUSH_MS) is not below TENSORFOLD_CLOSE_MS ($TENSORFOLD_CLOSE_MS): a stop may end rank 0's close before the tier has written what waits"
  fi
  if spill_asked; then
    log "SSD tier (a store on every Spark): folder $SPILL_REAL, cap $SPILL_GIB GiB, floor $SPILL_FLOOR_GIB GiB free, $SPILL_TURNS old turns, quota $SPILL_QUOTA, from $SPILL_MIN_TOKENS tokens, staging $SPILL_STAGING_MIB MiB; a stop's flush $SPILL_FLUSH_MS ms inside rank 0's close $TENSORFOLD_CLOSE_MS ms"
  elif [[ -z "$SPILL_OFF_REASON" ]]; then
    log "SSD tier: off (SPILL_GIB=$SPILL_GIB)"
  fi
  return 0
}
# spill_env0: SPILL_ENV0, rank 0's tier environment (start.sh adds it to rank 0; spill_rank_env gives the workers the same values): TF_DS_SPILL_GIB always (0:
# off) and the recipe's reason when it turned the tier off; with the tier asked, every setting by the engine's name, the
# store's identity and rank 0's digest (below), and every other TF_DS_SPILL_* / TF_SPILL_* exported, as it is.
spill_env0() {
  local name
  SPILL_ENV0=("TF_DS_SPILL_GIB=$SPILL_GIB")
  [[ -z "$SPILL_OFF_REASON" ]] || SPILL_ENV0+=("TF_DS_SPILL_OFF_REASON=$SPILL_OFF_REASON")
  spill_asked || return 0
  SPILL_ENV0+=("TF_DS_SPILL_DIR=$SPILL_MOUNT" "TF_DS_SPILL_MIN_FREE_GIB=$SPILL_FLOOR_GIB" "TF_DS_SPILL_QUOTA=$SPILL_QUOTA"
               "TF_DS_SPILL_MIN_TOKENS=$SPILL_MIN_TOKENS" "TF_DS_SPILL_STAGING_MIB=$SPILL_STAGING_MIB"
               "TF_SPILL_TURNS=$SPILL_TURNS" "TF_SPILL_FLUSH_MS=$SPILL_FLUSH_MS")
  # the store's identity (scripts/nodes.sh spill_ident_env) and rank 0's own weight file's digest
  while IFS= read -r name; do SPILL_ENV0+=("$name"); done < <(spill_ident_env)
  SPILL_ENV0+=("TF_DS_RANK_CACHE_SHA=${RANK_SHA[0]:-}")
  while IFS='=' read -r name _; do SPILL_ENV0+=("$name=${!name}"); done < <(env | grep -E '^(TF_DS_SPILL|TF_SPILL)_[A-Z0-9_]+=' | grep -vE "^($SPILL_MAPPED)=" || true)
}

# spill_dir_probe <dir> [<folder it may not overlap>...]: one line about <dir> on this Spark, after MiaAI-Lab's two-Spark
# recipe's spill_dir_probe (MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks, scripts/port.sh), creating nothing: "OK"
# (it is a folder) or "NEW" (it is not there yet: its nearest existing parent is described), then <owner uid> <owner
# gid> <your uid> <fs type> <source device> <free bytes> <exported 0|1> <home-or-root 0|1> <canonical path> <the folder
# it equals, sits inside or holds, or ->; or "ERR <reason>". Exported: under a path the kernel's export table
# (/var/lib/nfs/etab), /etc/exports or /etc/exports.d/*.exports lists. Self-contained (start.sh runs it under timeout).
spill_dir_probe() {
  local d=$1 real p st=OK fs src avail own exp=0 e over=- top=0 f fr x
  shift
  real=$(readlink -m -- "$d")
  case "$real" in /*) ;; *) echo "ERR not an absolute path: $d"; return 0 ;; esac
  case "$real" in *[!A-Za-z0-9._/-]*) echo "ERR path has characters other than A-Za-z0-9._/-: $real"; return 0 ;; esac
  if [ -e "$real" ] && [ ! -d "$real" ]; then echo "ERR not a folder: $real"; return 0; fi
  [ -d "$real" ] || st=NEW
  p=$real; while [ ! -d "$p" ]; do p=$(dirname -- "$p"); done
  fs=$(findmnt -n -o FSTYPE -T "$p" 2>/dev/null | head -1); [ -n "$fs" ] || fs=$(stat -f -c %T -- "$p" 2>/dev/null)
  src=$(findmnt -n -o SOURCE -T "$p" 2>/dev/null | head -1 | tr -s ' \t' '__')
  avail=$(df -B1 --output=avail -- "$p" 2>/dev/null | tail -1 | tr -d ' ')
  for x in /var/lib/nfs/etab /etc/exports /etc/exports.d/*.exports; do
    [ -r "$x" ] || continue
    while read -r e _; do
      e=${e#\"}; e=${e%\"}
      case "$e" in /*) e=${e%/}; case "$real/" in "$e"/*) exp=1 ;; esac ;; esac
    done < "$x"
  done
  for f in "$@"; do
    [ -n "$f" ] || continue
    fr=$(readlink -m -- "$f")
    case "$real/" in "$fr"/*) over=$fr; break ;; esac
    case "$fr/" in "$real"/*) over=$fr; break ;; esac
  done
  [ "$real" != / ] && [ "$real" != "$(readlink -m -- "$HOME")" ] || top=1
  own=$(stat -c '%u %g' -- "$p" 2>/dev/null) || own=""
  case "$own" in *[0-9]" "*[0-9]) ;; *) echo "ERR cannot read the owner of $p"; return 0 ;; esac
  echo "$st $own $(id -u) ${fs:-unknown} ${src:--} ${avail:-0} $exp $top $real $over"
}

# spill_check (spill_start, with the tier asked): SPILL_DIR checked on this Spark before anything is stopped, every
# disk step under SPILL_CHECK_TIMEOUT: SPILL_REAL, the folder every rank mounts at /spill (created as you, mode 0700, when it
# is not there yet: docker would create it as root; DRY_RUN=1 creates nothing), or the tier off for this start
# (spill_off) naming what failed. The checks (each a reason): a local disk, not inside an NFS export, no overlap with
# SPILL_FORBID's folders (other tiers': the store deletes the folders of other builds in its root), HF_CACHE, the weight
# files, Engram's, STATE_DIR or LOG_DIR, not / or your home, not under an archive folder, not root's, at least
# SPILL_FLOOR_GIB free. Under (1 + TF_DS_SPILL_KEEP_BUILDS, 1) x SPILL_GIB + SPILL_FLOOR_GIB free it only warns: the
# tier stops writing at the floor before it holds its cap. No owner is changed afterwards: the engine gives each file
# the folder's owner itself.
SPILL_REAL=""
spill_check() {
  local t=$SPILL_CHECK_TIMEOUT rc=0 line st uid gid me fs src avail exp top real over lc floor_b room_b kb gib
  local -a forbid=()
  SPILL_REAL=""
  [[ "$t" =~ ^[1-9][0-9]{0,5}$ ]] ||
    { spill_off "SPILL_CHECK_TIMEOUT=$t is not a count of seconds" "it bounds start.sh's checks of the tier's folder"; return 0; }
  read -r -a forbid <<<"$SPILL_FORBID"
  line=$(timeout -k 5 "$t" bash -c "$(declare -f spill_dir_probe); spill_dir_probe \"\$@\"" spill_dir_probe "$SPILL_DIR" \
           ${forbid[@]+"${forbid[@]}"} "$HF_CACHE" "$DATA_DIR" "$(packed_dir_here)" "$STATE_DIR" "$LOG_DIR" 2>/dev/null) || rc=$?
  if (( rc == 124 || rc == 137 )); then
    spill_off "the spill folder check did not end within $t s" "SPILL_DIR $SPILL_DIR: its disk did not answer (SPILL_CHECK_TIMEOUT)"; return 0
  fi
  [[ -n "$line" ]] || { spill_off "the spill folder check gave no answer (exit $rc)" "SPILL_DIR $SPILL_DIR"; return 0; }
  [[ "$line" != ERR* ]] || { spill_off "spill folder: ${line#ERR }" "SPILL_DIR $SPILL_DIR"; return 0; }
  read -r st uid gid me fs src avail exp top real over <<<"$line"
  case "$fs" in
    nfs|nfs4|cifs|smb|smb2|smb3|smbfs|fuse.sshfs|sshfs|9p|ceph|fuse.ceph*|glusterfs|fuse.glusterfs|lustre|afs|davfs|fuse.rclone|fuse.s3fs|fuse.juicefs)
      spill_off "spill folder is on a network file system ($fs)" "SPILL_DIR $real: the tier needs this Spark's own local disk (NVMe)"; return 0 ;;
    tmpfs|ramfs|devtmpfs|overlay|overlayfs|squashfs|zram)
      spill_off "spill folder is on $fs (memory, a container layer or an image)" "SPILL_DIR $real: the tier needs this Spark's own local disk (NVMe)"; return 0 ;;
  esac
  [[ "$src" != /dev/zram* ]] || { spill_off "spill folder is on $src, a compressed RAM disk" "SPILL_DIR $real: the tier needs this Spark's own local disk (NVMe)"; return 0; }
  [[ "$exp" == 0 ]] || { spill_off "spill folder is inside an NFS export" "SPILL_DIR $real is under a path that /var/lib/nfs/etab, /etc/exports or /etc/exports.d lists: keep the tier out of what other machines read"; return 0; }
  if [[ "$over" != - ]]; then
    spill_off "spill folder overlaps $over" "SPILL_DIR $real equals, sits inside or holds $over (another tier's folder, SPILL_FORBID: ${SPILL_FORBID:-none}; or HF_CACHE, the weight files, Engram's, STATE_DIR or LOG_DIR): give the tier a folder of its own"
    return 0
  fi
  [[ "$top" == 0 ]] || { spill_off "spill folder is / or your home folder" "SPILL_DIR $real: give the tier a folder of its own"; return 0; }
  lc=${real,,}
  [[ "$lc/" != */archive*/* ]] || { spill_off "spill folder is under an archive folder" "SPILL_DIR $real: the tier writes and deletes there"; return 0; }
  if [[ "$st" == OK && "$uid" == 0 && "$me" != 0 ]]; then
    spill_off "spill folder belongs to root" "SPILL_DIR $real: its files would too (the engine gives each file the folder's owner); make it yours: sudo chown $me $real"; return 0
  fi
  [[ "$st" == NEW || "$uid" == "$me" ]] || warn "SPILL_DIR $real belongs to uid $uid, not to you ($me): the tier's files will be that user's"
  [[ "$avail" =~ ^[0-9]+$ ]] || avail=0
  if [[ "$SPILL_FLOOR_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    floor_b=$(awk -v g="$SPILL_FLOOR_GIB" 'BEGIN {printf "%.0f", g * 1073741824}')
    if awk -v a="$avail" -v f="$floor_b" 'BEGIN {exit !(a + 0 < f + 0)}'; then
      spill_off "spill folder: $(( avail / 1073741824 )) GiB free, under SPILL_FLOOR_GIB $SPILL_FLOOR_GIB" "SPILL_DIR $real: the tier would refuse every write; free the disk or lower SPILL_FLOOR_GIB"; return 0
    fi
    kb=${TF_DS_SPILL_KEEP_BUILDS:-1}; [[ "$kb" =~ ^[0-9]{1,3}$ ]] || kb=1
    if [[ "$SPILL_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      room_b=$(awk -v g="$SPILL_GIB" -v k="$kb" -v f="$floor_b" 'BEGIN {printf "%.0f", (1 + k) * g * 1073741824 + f}')
      gib=$(awk -v b="$room_b" 'BEGIN {printf "%.0f", b / 1073741824}')
      awk -v a="$avail" -v r="$room_b" 'BEGIN {exit !(a + 0 < r + 0)}' &&
        warn "SPILL_DIR $real has $(( avail / 1073741824 )) GiB free, under (1 + kept builds $kb) x SPILL_GIB $SPILL_GIB + SPILL_FLOOR_GIB $SPILL_FLOOR_GIB = $gib GiB: the tier stops writing at the floor before it holds its cap"
    fi
  fi
  if [[ "$st" == NEW ]]; then
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      log "DRY_RUN: SPILL_DIR $real does not exist yet: start.sh would create it (as you, mode 0700)"
    else
      rc=0; timeout -k 5 "$t" mkdir -p -m 0700 -- "$real" 2>/dev/null || rc=$?
      [[ "$rc" == 0 && -d "$real" ]] || { spill_off "spill folder: cannot create $real" "SPILL_DIR $real (exit $rc)"; return 0; }
      log "SPILL_DIR $real created (as you, mode 0700)"
    fi
  fi
  SPILL_REAL=$real
  log "SSD tier's folder: $real ($fs on $src, $(( avail / 1073741824 )) GiB free), mounted at $SPILL_MOUNT in rank 0's container alone"
}

# ---------------------------------------------------------------- the SSD tier's identity (rank 0 names one store)
# The store's name is rank 0's hash of the engine build, the checkpoint, the overlay, Engram, the kit and every rank's
# weight file (its SHA-256: the bytes each rank computes with), so a state is never restored into other weights or
# another build. start.sh sends the inputs; the engine hashes them (and the switches it agrees on itself):
#   TF_DS_ENGINE_ID        the image's identity by content (image_ident: its layers and config, the same on every Spark)
#   TF_DS_MODEL_REVISION   MODEL_ID@<the checkpoint's revision>
#   TF_DS_ABLATION_ID      keys@<ABLIT_REVISION> with ABLIT=1, else none
#   TF_DS_ENGRAM_REVISION  Engram's tables are the checkpoint's own shards: MODEL_ID@<the checkpoint's revision>
#   TF_DS_KIT_ID           KIT_REF
#   TF_DS_RANK_CACHE_SHA   each rank its own weight file's digest (64 hex; empty when it cannot be had: the engine then
#                          names that rank and turns the tier off), from the cache prepare.sh wrote on that Spark
# Derived, never typed: an exported one does not pass (said).
SPILL_IDENT='TF_DS_(ENGINE_ID|MODEL_REVISION|ABLATION_ID|ENGRAM_REVISION|KIT_ID|RANK_CACHE_SHA)'
ablation_id() { if [[ "$ABLIT" == 1 ]]; then echo "keys@$ABLIT_REVISION"; else echo none; fi; }
spill_ident_env() {
  local rev; rev=$(snapshot_rev)
  printf '%s\n' "TF_DS_ENGINE_ID=$(image_ident "$IMAGE")" "TF_DS_MODEL_REVISION=$MODEL_ID@$rev" "TF_DS_ABLATION_ID=$(ablation_id)" \
                "TF_DS_ENGRAM_REVISION=$MODEL_ID@$rev" "TF_DS_KIT_ID=$KIT_REF"
}

# weights_sha <seconds> <read|hash> <cache folder> <weight-file folder> <node> [buffered]: one line about this Spark's weight file
# of 2D node <node> (the one regular file rank<node>of4-*.bin in <weight-file folder>, as the engine takes it):
#   ok <64 hex> <file> (how)            its SHA-256, from the cache (<cache folder>/weights-sha/: an entry whose key
#                                       is the file's size, mtime_ns, ctime_ns, inode and device, python3's os.stat), or
#                                       with hash read once now: O_DIRECT, 16 MiB at a time into a page-aligned buffer
#                                       (the page cache stays empty; where the file system has no O_DIRECT, buffered
#                                       reads dropped behind with fadvise DONTNEED), and the entry written
#   none <file> | stale <file>          read: no entry for this file, or one of another state of it (it changed since)
#   err <reason>                        no such file, several, unreadable, changed while hashed, not done in <seconds>
# ("buffered" reads as on a file system without O_DIRECT: the tests' way to that path)
# A leading literal $HOME/ in either folder is this Spark's home. Self-contained (sent to the workers over ssh): bash,
# timeout and python3 there.
weights_sha() {
  local t=$1 out rc=0
  shift
  out=$(timeout -k 5 "$t" python3 -I -c '
import errno, hashlib, json, mmap, os, sys, time
def say(*words):
    print(" ".join(str(w) for w in words))
    sys.exit(0)
def home(p):
    return os.environ.get("HOME", "") + p[5:] if p.startswith("$HOME/") else p
def main():
    mode, cache, wdir, node = sys.argv[1], home(sys.argv[2]), home(sys.argv[3]), sys.argv[4]
    buffered = sys.argv[5:] == ["buffered"]
    pre = "rank%sof4-" % node
    try:
        names = sorted(e.name for e in os.scandir(wdir)
                       if e.name.startswith(pre) and e.name.endswith(".bin") and e.is_file(follow_symlinks=False))
    except OSError as e:
        say("err cannot list", wdir + ":", e.strerror)
    if len(names) != 1:
        say("err", len(names), "weight files %s*.bin in" % pre, wdir)
    f = os.path.join(wdir, names[0])
    def key():
        s = os.stat(f)
        return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev]
    k = key()
    entry = os.path.join(cache, "weights-sha",
                         hashlib.sha256(os.path.realpath(f).encode()).hexdigest()[:16] + "-" + names[0] + ".json")
    had = None
    try:
        with open(entry) as fh:
            had = json.load(fh)
    except (OSError, ValueError):
        pass
    sha = had.get("sha256") if isinstance(had, dict) else None
    if (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
            and had.get("file") == f and had.get("key") == k):
        say("ok", sha, f, "(from its cache)")
    if mode != "hash":
        say("stale" if had is not None else "none", f)
    t0 = time.monotonic()
    chunk = 16 << 20
    buf = mmap.mmap(-1, chunk)
    view = memoryview(buf)
    h = hashlib.sha256()
    size, off, direct = k[0], 0, hasattr(os, "O_DIRECT") and not buffered
    try:
        fd = os.open(f, os.O_RDONLY | (os.O_DIRECT if direct else 0))
    except OSError as e:
        if not direct or e.errno != errno.EINVAL:
            raise
        fd, direct = os.open(f, os.O_RDONLY), False
    while off < size:
        try:
            n = os.preadv(fd, [buf], off)
        except OSError as e:
            if not direct or e.errno != errno.EINVAL:
                raise
            os.close(fd)
            fd, direct = os.open(f, os.O_RDONLY), False
            continue
        if n <= 0:
            break
        h.update(view[:n])
        if not direct:
            os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
        off += n
    os.close(fd)
    if off != size:
        say("err short read of", f, "(%d of %d bytes)" % (off, size))
    if key() != k:
        say("err", f, "changed while it was hashed")
    os.makedirs(os.path.dirname(entry), exist_ok=True)
    tmp = "%s.%d" % (entry, os.getpid())
    secs = round(time.monotonic() - t0, 1)
    with open(tmp, "w") as fh:
        json.dump({"file": f, "key": k, "sha256": h.hexdigest(), "direct": direct, "seconds": secs}, fh)
    os.replace(tmp, entry)
    say("ok", h.hexdigest(), f, "(hashed in %s s, %s)" % (secs, "O_DIRECT" if direct else "buffered, dropped behind"))
try:
    main()
except SystemExit:
    raise
except BaseException as e:
    say("err", type(e).__name__ + ":", e)
' "$@" 2>&1) || rc=$?
  case "$rc" in
    0) printf '%s\n' "${out##*$'\n'}" ;;
    124|137) echo "err not done within $t s" ;;
    *) echo "err the digest failed (exit $rc): ${out##*$'\n'}" ;;
  esac
}

# rank_weights <rank>: RW_WHERE (here | worker), RW_CACHE, RW_DIR and RW_NODE: where rank <rank>'s weight file is
# hashed and its digest cached. A worker with its own copy (WORKER_HF_CACHE*) hashes its own file into its own state
# folder; a worker that reads this Spark's over NFS is hashed here, from this Spark's file of its node (the same bytes).
rank_weights() {
  local r=$1 p=""
  RW_NODE=$(cut -d, -f$((r + 1)) <<<"$(node_order)")
  (( r == 0 )) || p=$(worker_hf_cache "$r")
  if [[ -n "$p" ]]; then RW_WHERE=worker RW_CACHE='$HOME/.local/state/dsv41-tensorfold' RW_DIR="$p/$DATA_SUB/rank-cache"
  else RW_WHERE=here RW_CACHE=$STATE_DIR RW_DIR="$DATA_DIR/rank-cache"; fi
}

# worker_timed <seconds> <i> <command...>: worker i's command (as worker), its ssh ended after <seconds>
worker_timed() {
  local s=$1 i=$2
  shift 2
  [[ -z "${WORKER_DOWN[$i]:-}" ]] || return 255
  timeout -k 5 "$s" ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 \
      -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%C" -o ControlPersist=60 "$(worker_host "$i")" "$@"
}

# weights_run <read|hash> <seconds>: every rank's weights_sha line, all four at once, into WEIGHTS_LINE[0..3]
declare -a WEIGHTS_LINE=()
weights_run() {
  local mode=$1 t=$2 r
  local -a pids=() outs=() wheres=()
  WEIGHTS_LINE=()
  for r in 0 1 2 3; do
    rank_weights "$r"
    wheres[r]=$RW_WHERE
    outs[r]=$(mktemp "$STATE_DIR/weights-sha.XXXXXX") || { WEIGHTS_LINE[r]="err no temporary file in $STATE_DIR"; continue; }
    if [[ "$RW_WHERE" == here ]]; then
      weights_sha "$t" "$mode" "$RW_CACHE" "$RW_DIR" "$RW_NODE" > "${outs[r]}" 2>&1 &
    else
      worker_timed $(( t + 10 )) "$r" "$(declare -f weights_sha); weights_sha $(printf '%q ' "$t" "$mode" "$RW_CACHE" "$RW_DIR" "$RW_NODE")" > "${outs[r]}" 2>&1 &
    fi
    pids[r]=$!
  done
  for r in 0 1 2 3; do [[ -z "${pids[r]:-}" ]] || wait "${pids[r]}" || true; done
  for r in 0 1 2 3; do
    [[ -n "${outs[r]:-}" ]] || continue
    WEIGHTS_LINE[r]=$(tail -n 1 "${outs[r]}" 2>/dev/null || true); rm -f -- "${outs[r]}"
    [[ -n "${WEIGHTS_LINE[r]}" ]] || WEIGHTS_LINE[r]="err no answer$( [[ "${wheres[r]}" == here ]] || echo " from $(worker_host "$r")")"
  done
}

# weights_hash_all (prepare.sh): each Spark hashes its own weight file, all four at once, each under
# WEIGHTS_SHA_TIMEOUT (a file already hashed in that state is read from its cache, nothing read again). A rank whose
# digest cannot be had is said, and prepare.sh goes on: the engine then starts with the SSD tier off, naming the rank,
# until prepare.sh hashes it.
weights_hash_all() {
  local t=$WEIGHTS_SHA_TIMEOUT r kind sha rest
  [[ "$t" =~ ^[1-9][0-9]{0,6}$ ]] ||
    { warn "WEIGHTS_SHA_TIMEOUT=$t is not a count of seconds: the weight files are not hashed, and the SSD tier stays off until they are"; return 0; }
  log "  Weight files' fingerprints for the SSD cache, all four Sparks at once (about a minute for a new file)"
  weights_run hash "$t"
  for r in 0 1 2 3; do
    read -r kind sha rest <<<"${WEIGHTS_LINE[r]:-}"
    if [[ "$kind" == ok ]]; then detail "  rank $r: sha256 ${sha:0:16}... ($rest)"
    else warn "rank $r: no weight-file digest (${WEIGHTS_LINE[r]#err }): the SSD tier will be off at each start, naming rank $r, until prepare.sh hashes it"; fi
  done
  return 0
}

# weights_digests (spill_start, with the tier asked; start.sh never hashes): every rank's digest from its Spark's
# cache, all four at once, each under SPILL_CHECK_TIMEOUT: RANK_SHA[0..3], 64 hex or empty (no entry, the file changed
# since it was hashed, or no answer: the engine names that rank and starts with the tier off).
declare -a RANK_SHA=()
weights_digests() {
  local t=$SPILL_CHECK_TIMEOUT r kind sha rest missing=""
  RANK_SHA=()
  weights_run read "$t"
  for r in 0 1 2 3; do
    read -r kind sha rest <<<"${WEIGHTS_LINE[r]:-}"
    if [[ "$kind" == ok && "$sha" =~ ^[0-9a-f]{64}$ ]]; then RANK_SHA[r]=$sha; continue; fi
    RANK_SHA[r]=""; missing+=" $r"
    case "$kind" in
      none) warn "rank $r: no weight-file digest yet for ${sha}: prepare.sh hashes it" ;;
      stale) warn "rank $r: its weight file ${sha} changed since prepare.sh hashed it" ;;
      *) warn "rank $r: no weight-file digest (${WEIGHTS_LINE[r]#err })" ;;
    esac
  done
  if [[ -n "$missing" ]]; then
    warn "no weight-file digest for rank(s)$missing: the engine starts with the SSD tier off and names them on /health. Run scripts/prepare.sh (or PREPARE=1 ./start.sh restart) to hash the weight files"
  else
    log "Weight files' digests: $(for r in 0 1 2 3; do printf 'rank %s %s... ' "$r" "${RANK_SHA[r]:0:12}"; done)"
  fi
}
# spill_every_rank: the SSD tier is asked for, so every rank keeps a store of its own (the engine has no switch for it: always on)
spill_every_rank() { spill_asked; }
# spill_rank_env <rank>: a worker's docker -e for the SSD tier, with the tier asked (nothing otherwise): its own weight digest and, with
# a store on every rank, what that store needs: rank 0's settings and the store's identity as they are (SPILL_ENV0, spill_env0 has run;
# every rank's cap, quota, shortest point, turns and image cache must equal rank 0's or the ranks refuse each other), the kit and the token map
# (the store's name), the folder mounted at SPILL_MOUNT (spill_rank_mount)
spill_rank_env() {
  # TF_DS_SPILL_GIB on every rank, whether or not the tier is on: the ranks decide together from it whether there is a landing to
  # carry (a tier that is off has none), and ranks that read different values cannot start
  printf -- '-e %q ' "TF_DS_SPILL_GIB=$SPILL_GIB"
  spill_asked || return 0
  printf -- '-e %q ' "TF_DS_RANK_CACHE_SHA=${RANK_SHA[$1]:-}"
  spill_every_rank || return 0
  local x
  for x in "${SPILL_ENV0[@]}"; do [[ "$x" == TF_DS_RANK_CACHE_SHA=* || "$x" == TF_DS_SPILL_GIB=* ]] || printf -- '-e %q ' "$x"; done
  printf -- '-e %q ' "TF_DS_KIT=$KIT" "TF_DS_TOKEN_MAP=$DATA/dsv41_token_map.json"
}
# spill_rank_mkdir / spill_rank_mount: on a worker, the store's folder (the same path as this Spark's SPILL_REAL) made as the user before docker
# would make it as root, and mounted where the engine reads TF_DS_SPILL_DIR
spill_rank_mkdir() { if spill_every_rank && [[ -n "$SPILL_REAL" ]]; then printf ' && mkdir -p -m 0700 %q' "$SPILL_REAL"; fi; return 0; }
spill_rank_mount() { if spill_every_rank && [[ -n "$SPILL_REAL" ]]; then printf ' -v %q:%s' "$SPILL_REAL" "$SPILL_MOUNT"; fi; return 0; }
