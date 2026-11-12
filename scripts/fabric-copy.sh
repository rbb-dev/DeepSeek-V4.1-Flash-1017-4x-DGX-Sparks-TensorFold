#!/usr/bin/env bash
# Copy files from this Spark to a worker over the fabric at its full speed: plain TCP streams (tar | nc) spread over the
# worker's fabric addresses, with no ssh in the data path, no encryption and no compression; ssh carries the control
# commands only. scripts/setup-data.sh sends each worker its share with it, the three workers at once.
#   scripts/fabric-copy.sh [-n STREAMS] [-p PORT] [-l LABEL] [-s] SSH_TARGET ADDRESSES SRC_DIR DEST_DIR < paths
# ADDRESSES: the worker's fabric addresses, comma-separated (one a rail); the streams take them in turn, and each
#   listener binds to its own address only.
# paths: one a line, relative to SRC_DIR. Each lands at the same path under DEST_DIR on the worker (DEST_DIR may start
#   with a literal $HOME/: the worker's home), links followed: every path arrives as a regular file.
# A path the worker holds at this Spark's size is not sent again; one it lacks, or holds at another size (a copy cut
# short), is sent whole. Every sent file is then read on both ends and its SHA-256 compared: a difference fails the copy,
# named. The worker's ports PORT .. PORT+STREAMS-1 (default 5400 ..) must be free; STREAMS defaults to 8. -l: the name
# its lines carry (default: SSH_TARGET). -s: print only how many bytes would be sent, and send nothing.
# Its first version measured 2.66 GB/s to one Spark with 8 streams, where ssh + rsync gave 0.81 GB/s: the receiver's
# disk, writing real data, is what limits a copy (1.8-2.7 GB/s a Spark).
set -euo pipefail
N=8; PORT0=5400; TMO=7200; LABEL=""; SIZE_ONLY=0
while getopts n:p:l:s o; do
  case $o in n) N=$OPTARG ;; p) PORT0=$OPTARG ;; l) LABEL=$OPTARG ;; s) SIZE_ONLY=1 ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
(( $# == 4 )) || { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
CTL=$1; SRC=$3; DEST=$4
IFS=, read -r -a ADDRS <<<"$2"
log() { (( SIZE_ONLY )) || printf '[fabric-copy %s] %s\n' "${LABEL:-$CTL}" "$*"; }
die() { printf '[fabric-copy %s] ERROR: %s\n' "${LABEL:-$CTL}" "$*" >&2; exit 1; }
[[ $N =~ ^[1-9][0-9]?$ ]] || die "STREAMS is 1 to 99, not $N"
[[ $PORT0 =~ ^[0-9]+$ ]] && (( PORT0 >= 1024 && PORT0 + N <= 65535 )) || die "PORT is 1024 or more, with room for $N ports, not $PORT0"
(( ${#ADDRS[@]} )) || die "no fabric address of the worker"
for a in "${ADDRS[@]}"; do [[ $a =~ ^[0-9]+(\.[0-9]+){3}$ ]] || die "a fabric address is an IPv4 address, not '$a'"; done
[[ -d $SRC ]] || die "SRC_DIR $SRC is no folder here"
[[ $DEST =~ ^(\$HOME/|/)[A-Za-z0-9_./+@=:-]*$ ]] ||
  die "DEST_DIR is an absolute path, or one under a literal \$HOME/ (letters, digits and _ . / + @ = : - only), not $DEST"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ControlMaster=auto
     -o "ControlPath=$HOME/.ssh/cm-%C" -o ControlPersist=60)
remote() { "${SSH[@]}" "$CTL" "$@"; }

S=$(mktemp -d "${TMPDIR:-/tmp}/fabric-copy.XXXXXX"); R=""
cleanup() {
  local rc=$?
  if [[ -n $R ]]; then   # receivers still running after a failure: stopped by the PIDs they wrote
    remote "[ -f $R/pids ] && for p in \$(cat $R/pids); do pkill -TERM -P \$p; kill \$p; done 2>/dev/null; rm -rf $R" \
      </dev/null || true
  fi
  rm -rf "$S"; exit $rc
}
trap cleanup EXIT
grep -v '^$' > "$S/paths" || true
[[ -s $S/paths ]] || { log "no paths given: nothing to send"; (( ! SIZE_ONLY )) || echo 0; exit 0; }

# 1. what to send: each path's size here (links followed) against the worker's
INV='import os, sys
d = sys.argv[1]
for p in sys.stdin.read().split("\n"):
    if p:
        try:
            print(f"{os.stat(os.path.join(d, p)).st_size}\t{p}")
        except OSError:
            pass'
(python3 -I -c "$INV" "$SRC" < "$S/paths") > "$S/here"
remote "python3 -I -c $(printf '%q' "$INV") \"$DEST\"" < "$S/paths" > "$S/there" 2>/dev/null || true
summary=$(python3 -I - "$S" "$N" "$SRC" <<'PY'
import os, sys
s, n, src = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def inv(name):
    out = {}
    for line in open(os.path.join(s, name)).read().splitlines():
        size, path = line.split("\t", 1)
        out[path] = int(size)
    return out
paths = [p for p in open(os.path.join(s, "paths")).read().split("\n") if p]
here, there = inv("here"), inv("there")
missing = [p for p in paths if p not in here or not os.path.isfile(os.path.join(src, p))]
if missing:
    print("not a file here (under %s): %s" % (src, ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")))
    sys.exit(1)
paths = list(dict.fromkeys(paths))
send = sorted((p for p in paths if there.get(p) != here[p]), key=lambda p: -here[p])
k = min(n, len(send))
groups = [[] for _ in range(k)]; load = [0] * k
for p in send:                                   # the biggest file to the lightest stream
    i = load.index(min(load)); groups[i].append(p); load[i] += here[p]
for i, g in enumerate(groups):
    open(os.path.join(s, f"list.{i}"), "wb").write(b"".join(p.encode() + b"\0" for p in g))
open(os.path.join(s, "sent"), "wb").write(b"".join(p.encode() + b"\0" for p in send))
open(os.path.join(s, "plan"), "w").write(f"{k} {sum(load)}\n")
print(f"{len(paths)} paths, {len(send)} to send ({sum(load) / 1e9:.1f} GB), {len(paths) - len(send)} there already at their size")
PY
) || die "$summary"
log "$summary"
read -r K BYTES < "$S/plan"
(( ! SIZE_ONLY )) || { echo "$BYTES"; exit 0; }
(( K > 0 )) || { log "nothing to send"; exit 0; }

# 2. the receivers: one listener a stream on its address, each piping into tar; their PIDs and exit codes kept there
addr() { echo "${ADDRS[$(( $1 % ${#ADDRS[@]} ))]}"; }
ports=""; for (( i = 0; i < K; i++ )); do ports+="|$((PORT0 + i))"; done; ports=${ports#|}
starts=""
for (( i = 0; i < K; i++ )); do
  starts+="nohup bash -c \"timeout $TMO nc -l $(addr $i) $((PORT0 + i)) </dev/null | tar -C \\\"\$d\\\" -xpf -; echo \\\${PIPESTATUS[*]} > \$R/rc.$i\" >/dev/null 2>&1 </dev/null & echo \$! >> \$R/pids
"
done
R=$(remote "set -e; d=\"$DEST\"
  busy=\$(ss -ltnH 2>/dev/null | awk '{print \$4}' | grep -cE ':($ports)\$' || true)
  [ \"\$busy\" = 0 ] || { echo \"a port of $PORT0..$((PORT0 + K - 1)) is in use on \$(hostname)\" >&2; exit 1; }
  R=\$(mktemp -d /tmp/fabric-copy.XXXXXX); mkdir -p \"\$d\"
  $starts
  for t in \$(seq 100); do [ \"\$(ss -ltnH 2>/dev/null | awk '{print \$4}' | grep -cE ':($ports)\$')\" -ge $K ] && break; sleep 0.1; done
  echo \$R" </dev/null) || die "could not start the receivers on the worker"
log "$K receivers listening ($(printf '%s\n' "${ADDRS[@]}" | sort -u | paste -sd' ' -), ports $PORT0..$((PORT0 + K - 1)))"

# 3. the senders, all at once
t0=$(date +%s.%N)
for (( i = 0; i < K; i++ )); do
  ( tar -C "$SRC" --dereference --hard-dereference --null -T "$S/list.$i" -cf - | timeout $TMO nc -N "$(addr $i)" $((PORT0 + i))
    echo "${PIPESTATUS[*]}" > "$S/rc.$i" ) </dev/null &
done
wait
for _ in $(seq 600); do [[ $(remote "ls $R/rc.* 2>/dev/null | wc -l" </dev/null) -ge $K ]] && break; sleep 0.5; done
t1=$(date +%s.%N)
bad=0
for (( i = 0; i < K; i++ )); do
  s=$(cat "$S/rc.$i" 2>/dev/null || echo none); r=$(remote "cat $R/rc.$i 2>/dev/null" </dev/null || true)
  [[ $s =~ ^0( 0)+$ && $r == "0 0" ]] || { log "stream $i: sender exit codes [$s], receiver [$r]"; bad=1; }
done
remote "rm -rf $R" </dev/null; R=""
log "sent $(awk -v b="$BYTES" -v a="$t0" -v c="$t1" 'BEGIN { printf "%.1f GB in %.1f s: %.2f GB/s", b / 1e9, c - a, b / (c - a + 1e-9) / 1e9 }')"
(( bad == 0 )) || die "a stream did not finish cleanly (above)"

# 4. the check: every sent file read on both ends, its SHA-256 compared
t0=$(date +%s.%N)
(cd "$SRC" && xargs -0 -r -P 16 -n 4 sha256sum -- < "$S/sent" | LC_ALL=C sort -k2) > "$S/sum.here"
remote "cd \"$DEST\" && xargs -0 -r -P 16 -n 4 sha256sum -- 2>/dev/null | LC_ALL=C sort -k2" < "$S/sent" > "$S/sum.there" || true
t1=$(date +%s.%N)
if cmp -s "$S/sum.here" "$S/sum.there"; then
  log "verified: $(wc -l < "$S/sum.here") files by SHA-256 on both ends ($(awk -v a="$t0" -v c="$t1" 'BEGIN { printf "%.1f", c - a }') s)"
else
  differ=$(python3 -I - "$S/sum.here" "$S/sum.there" <<'PY'
import sys
def sums(f):
    out = {}
    for line in open(f, errors="replace"):
        h, _, p = line.rstrip("\n").partition("  ")
        out[p] = h
    return out
a, b = sums(sys.argv[1]), sums(sys.argv[2])
print(" ".join(sorted(p for p in a if b.get(p) != a[p])))
PY
)
  die "SHA-256 differs between this Spark and the worker (or the file is missing there) for: $differ"
fi
