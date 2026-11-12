#!/usr/bin/env bash
# fabric-copy.sh's command line, RDMA underneath: starts a receiver on each worker over ssh (the aarch64 binary is piped to a
# temp file there, so nothing needs installing on the workers), then runs the sender here.
#   fabric-xfer.sh [-n THREADS] [-p PORT] [-l LABEL] [-s] SSH_TARGET ADDRESSES SRC_DIR DEST_DIR < paths
# SSH_TARGET: one worker, or several comma-separated (ssh targets, such as user@192.0.2.2,user@192.0.2.3): each file is read once and sent to all of them.
# ADDRESSES: the worker's fabric addresses, comma-separated, one a rail (rail A first); for several workers one group per
#   worker, groups separated by ";" in the same order.
# paths: one a line, relative to SRC_DIR; each lands at the same path under DEST_DIR ($HOME/ allowed first), links followed
#   (make the list with find -L). A file the worker holds at the sender's size is not sent again.
# -n: disk I/O threads on each side (8). -p: control TCP port on each worker (5400). -s: print the bytes that would be sent.
# Environment: FX_BIN (the binary; default: fabric-xfer beside this script), FX_OPTS (extra options for sender AND receivers:
#   --ring-gib N --chunk-mib N --no-direct --readback --verify-existing --gid-index N --engine-pattern P --progress-s N),
#   FX_SEND_OPTS (sender only: --manifest PATH).
set -euo pipefail
N=8; PORT0=5400; LABEL=""; SIZE_ONLY=0
while getopts n:p:l:s o; do
  case $o in n) N=$OPTARG ;; p) PORT0=$OPTARG ;; l) LABEL=$OPTARG ;; s) SIZE_ONLY=1 ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
(( $# == 4 )) || { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
die() { printf '[fabric-xfer %s] ERROR: %s\n' "${LABEL:-${1:-}}" "$2" >&2; exit 1; }
IFS=, read -r -a TARGETS <<<"$1"
IFS=';' read -r -a GROUPS_ <<<"$2"
SRC=$3; DEST=$4
BIN=${FX_BIN:-$(dirname "$(readlink -f "$0")")/fabric-xfer}
[[ $N =~ ^[1-9][0-9]?$ ]] || die "" "THREADS is 1 to 99, not $N"
[[ $PORT0 =~ ^[0-9]+$ ]] && (( PORT0 >= 1024 && PORT0 <= 65535 )) || die "" "PORT is 1024 to 65535, not $PORT0"
(( ${#TARGETS[@]} >= 1 && ${#TARGETS[@]} == ${#GROUPS_[@]} )) || die "" "${#TARGETS[@]} targets but ${#GROUPS_[@]} address groups (separate groups with ';')"
for g in "${GROUPS_[@]}"; do
  for a in ${g//,/ }; do [[ $a =~ ^[0-9]+(\.[0-9]+){3}$ ]] || die "" "a fabric address is an IPv4 address, not '$a'"; done
done
for t in "${TARGETS[@]}"; do [[ $t =~ ^[A-Za-z0-9_.@-]+$ ]] || die "" "bad SSH_TARGET '$t'"; done
[[ -d $SRC ]] || die "" "SRC_DIR $SRC is no folder here"
[[ $DEST =~ ^(\$HOME/|/)[A-Za-z0-9_./+@=:-]*$ ]] || die "" "DEST_DIR is an absolute path, or one under a literal \$HOME/ (letters, digits and _ . / + @ = : - only), not $DEST"
[[ -x $BIN ]] || die "" "$BIN is not built: run make in $(dirname "$BIN") on a Spark"
FX_OPTS=${FX_OPTS:-}; FX_SEND_OPTS=${FX_SEND_OPTS:-}

# this side's address on each rail = the source address the routing table uses to reach the first worker's rail address
LOCAL=""
for a in ${GROUPS_[0]//,/ }; do
  s=$(ip -4 route get "$a" | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -1)
  [[ -n $s ]] || die "" "no route to $a"
  LOCAL+="${LOCAL:+,}$s"
done

# one-time token: 32 random bytes as hex. It travels on each ssh's stdin (first line, before the binary) and to the local sender on
# fd 3; it is on no command line, so ps cannot show it. A receiver accepts one connection and only from a sender that presents it.
TOKEN=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
[[ $TOKEN =~ ^[0-9a-f]{64}$ ]] || die "" "could not read 32 random bytes"

# the receivers accept connections only from this side's addresses: its rail addresses, and the LAN address ssh reaches each worker from
# (when the target is an IPv4 address; a hostname is skipped, the rail addresses are what the control connection comes from)
ALLOW=$LOCAL
for t in "${TARGETS[@]}"; do
  h=${t##*@}
  if [[ $h =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    l=$(ip -4 route get "$h" | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -1)
    [[ -n $l && ,$ALLOW, != *",$l,"* ]] && ALLOW+=",$l"
  fi
done

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15)
pids=()
cleanup() { local rc=$?; for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; exit $rc; }
trap cleanup EXIT
TARGS=()
for i in "${!TARGETS[@]}"; do
  t=${TARGETS[$i]}; addrs=${GROUPS_[$i]}; a0=${addrs%%,*}
  remote="IFS= read -r tok; f=\$(mktemp /tmp/fabric-xfer.XXXXXX) && cat >\"\$f\" && chmod 700 \"\$f\"; printf '%s\\n' \"\$tok\" | \"\$f\" recv --bind $a0 --port $PORT0 --addrs $addrs --dest '$DEST' -n $N --token-fd 0 --allow $ALLOW $FX_OPTS; rc=\$?; rm -f \"\$f\"; exit \$rc"
  { printf '%s\n' "$TOKEN"; cat "$BIN"; } | "${SSH[@]}" "$t" "$remote" > >(sed -u "s/^/[$t] /") 2>&1 &
  pids+=($!)
  TARGS+=(--target "$t=$addrs")
done

SARGS=(send -n "$N" -p "$PORT0" -l "${LABEL:-${TARGETS[0]}}")
(( SIZE_ONLY )) && SARGS+=(-s)
set +e
# shellcheck disable=SC2086
"$BIN" "${SARGS[@]}" --local "$LOCAL" --token-fd 3 "${TARGS[@]}" $FX_OPTS $FX_SEND_OPTS "$SRC" 3<<<"$TOKEN"
rc=$?
set -e
if (( rc == 0 )); then
  for p in "${pids[@]}"; do wait "$p" || rc=1; done     # the receivers exit by themselves after the sender's goodbye
fi
pids=()
exit $rc
