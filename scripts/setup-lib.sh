# Helpers of ./setup.sh and scripts/setup-data.sh: sourced, nothing runs on loading. Tests: tests/test_scripts.py, SetupHelpers.

# _say / _detail: a line through scripts/config.sh's log and detail (the setup's step tag, its log file) when those are
# loaded, else printed as it is (a helper used on its own)
_say() { if declare -F log >/dev/null; then log "$@"; else printf '%s\n' "$*"; fi; }
_detail() {
  if declare -F detail >/dev/null; then detail "$@"
  elif [[ -n "${SETUP_LOG:-}" ]]; then printf '%s\n' "$*" >> "$SETUP_LOG"
  else printf '%s\n' "$*"; fi
}

# sparks_phrase <all> <name>...: the workers a step goes to, as words: "the other three Sparks" when it is every one of
# <all>, else the names ("a", "a and b", "a, b and c")
sparks_phrase() {
  local all=$1; shift
  if (( $# == all )); then echo "the other three Sparks"
  elif (( $# == 1 )); then echo "$1"
  else local rest; rest=$(printf '%s, ' "${@:1:$#-1}"); echo "${rest%, } and ${!#}"
  fi
}

# fx_routes <list1> ... <listN>: the copy's lines for one run to N workers, from each worker's list of paths (one a line,
# relative to the run's source folder): a path every list holds is a plain line (it goes to all N), any other path is
# "i,j<TAB>path" (it goes to the workers whose lists hold it, 1 = the first list). Paths keep their first-seen order.
fx_routes() {
  local n=$# k=0 f
  for f in "$@"; do
    k=$((k + 1))
    awk -v k="$k" 'NF && !seen[$0]++ { print k "\t" $0 }' "$f"
  done | awk -F'\t' -v n="$n" '
    { p = $2; if (!(p in c)) order[++m] = p; c[p]++; t[p] = (t[p] == "" ? $1 : t[p] "," $1) }
    END { for (i = 1; i <= m; i++) { p = order[i]; print (c[p] == n ? p : t[p] "\t" p) } }'
}

# make_key_file <path> <label>: a key file of one "label: key" line (32 random bytes as hex), readable by its owner only
# (0600, in a 0700 folder); a file that already holds something is kept as it is.
make_key_file() {
  local f=$1 label=$2 d
  [[ -s $f ]] && return 0
  d=$(dirname "$f")
  mkdir -p "$d" && chmod 700 "$d"
  ( umask 077; printf '%s: %s\n' "$label" "$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')" > "$f" )
  chmod 600 "$f"
}

# key_of <file>: the first key of a key file ("label: key" lines or bare keys; # comments skipped)
key_of() { awk 'NF && $1 !~ /^#/ { n = index($0, ":"); s = n ? substr($0, n + 1) : $0; gsub(/^[ \t]+|[ \t]+$/, "", s); print s; exit }' "$1"; }

# netplan_conflicts <dir> <ours> "<rails>" "<ports>": the files in DIR other than OURS that configure one of RAILS (as a
# device, a bond's or bridge's member, a VLAN's link or a match), one a line: "move FILE" when the file configures nothing
# but plain ethernets that are all among PORTS (this Spark's ConnectX-7 interfaces: an earlier fabric file, such as the one
# NVIDIA's two-Spark playbook writes), "stop FILE" for any other (it is the user's to change) or one that does not parse.
# netplan merges every file, so a rail configured by two files is in a state neither describes. python3-yaml comes with
# netplan.io.
netplan_conflicts() {
  python3 -I - "$@" <<'PY'
import fnmatch, glob, os, sys
import yaml
folder, ours, rails, ports = sys.argv[1], sys.argv[2], set(sys.argv[3].split()), set(sys.argv[4].split())
def touches(net):
    for kind, devs in net.items():
        if not isinstance(devs, dict):
            continue
        for dev, cfg in devs.items():
            cfg = cfg if isinstance(cfg, dict) else {}
            names = {str(dev), str(cfg.get("link", ""))} | {str(m) for m in cfg.get("interfaces") or []}
            pattern = (cfg.get("match") or {}).get("name") if isinstance(cfg.get("match"), dict) else None
            if names & rails or (pattern and any(fnmatch.fnmatch(r, str(pattern)) for r in rails)):
                return True
    return False
def plain_ports(net):
    devs = net.get("ethernets")
    return (set(net) <= {"version", "renderer", "ethernets"} and isinstance(devs, dict)
            and all(str(d) in ports and not (isinstance(c, dict) and ("match" in c or "set-name" in c)) for d, c in devs.items()))
for f in sorted(glob.glob(os.path.join(folder, "*.yaml"))):
    if os.path.basename(f) == ours or not os.path.isfile(f):
        continue
    try:
        with open(f) as fh:
            doc = yaml.safe_load(fh)
    except Exception:
        print("stop", f)
        continue
    net = doc.get("network") if isinstance(doc, dict) else None
    if isinstance(net, dict) and touches(net):
        print("move" if plain_ports(net) else "stop", f)
PY
}

# hf_on_path: the Hugging Face CLI on PATH. `pip install --user` and pipx put it in ~/.local/bin, which only a login
# shell puts on PATH (not ssh with a command, cron or a detached run), so that folder is added when hf is there.
hf_on_path() {
  command -v hf >/dev/null 2>&1 && return 0
  [[ -x $HOME/.local/bin/hf ]] || return 1
  PATH=$HOME/.local/bin:$PATH
  export PATH
}

# build_step_name <instruction>: a Dockerfile step as people read it (the instruction itself when unknown)
build_step_name() {
  case "$1" in
    FROM*) echo "base image" ;;
    *"zig build"*) echo "compiling the engine (Zig and CUDA)" ;;
    *"zig test"*) echo "checks before the build" ;;
    *"pip install"*) echo "Python packages" ;;
    *ziglang.org*|*zig.tar*) echo "the Zig compiler" ;;
    COPY*|ADD*) echo "the engine's source" ;;
    *"git clone"*) echo "a library the engine builds with" ;;
    *libnccl*|*"grep "*) echo "checking the build" ;;
    WORKDIR*|ENV*|CMD*|ENTRYPOINT*|LABEL*|USER*) echo "finishing" ;;
    *) printf '%.60s\n' "$1" ;;
  esac
}

# docker_build_steps <log> <docker build arguments>...: `docker build` with its whole output in LOG and, here, one line a
# build step ("step 8/10: RUN ...", BuildKit's and the classic builder's), a line every BUILD_HEARTBEAT_S seconds (60)
# while one step runs on, and the log's last 30 lines when the build fails. Returns the build's status.
docker_build_steps() {
  local log=$1 pid rc=0 t0 now since next shown=0 last="" poll=${BUILD_POLL_S:-2} beat=${BUILD_HEARTBEAT_S:-60}
  shift
  : > "$log"
  docker build "$@" > "$log" 2>&1 &
  pid=$!
  t0=$(date +%s); since=$t0; next=$t0
  _build_steps_seen() {         # every step line so far, "N/M<TAB>what" (BuildKit's "#12 [ 8/10] RUN ...", classic "Step 8/10 : RUN ...")
    sed -n -E 's/^#[0-9]+ \[[^]0-9]*([0-9]+) *\/ *([0-9]+)\] (.*)/\1\/\2\t\3/p; s/^Step ([0-9]+\/[0-9]+) : (.*)/\1\t\2/p' "$log"
  }
  _build_steps_show() {
    local all n line
    all=$(_build_steps_seen); n=$(grep -c . <<< "$all" || true); now=$(date +%s)
    if (( n > shown )); then
      while IFS= read -r line; do
        _say "$(printf '    step %s  %-40s %d s in' "${line%%$'\t'*}" "$(build_step_name "${line#*$'\t'}")" $((now - t0)))"
        last=${line%%$'\t'*}
      done < <(tail -n "$((n - shown))" <<< "$all")
      shown=$n; since=$now; next=$(awk -v t="$now" -v b="$beat" 'BEGIN { printf "%.3f\n", t + b }')
    elif [[ -n $last ]] && awk -v n="$now" -v x="$next" 'BEGIN { exit !(n >= x) }'; then
      next=$(awk -v t="$now" -v b="$beat" 'BEGIN { printf "%.3f\n", t + b }')
      _say "$(printf '    step %s still running, %d s in (%d s into the build)' "$last" $((now - since)) $((now - t0)))"
    fi
  }
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$poll"
    _build_steps_show
  done
  wait "$pid" || rc=$?
  _build_steps_show
  if (( rc )); then
    _say "  the build failed; its last lines (the whole output: $log):"
    tail -n 30 "$log" | sed 's/^/    /'
  fi
  return "$rc"
}

# index_data_bytes <index file>: the bytes of a weight file's tensors, from the index `tf-dsv41-rank-cache --index-only`
# prints (rows [key, dtype, shape, offset, bytes]): the largest offset + bytes.
index_data_bytes() {
  python3 -I -c '
import re, sys
rows = re.findall(r"\[\s*\"(?:[^\"\\]|\\.)*\"\s*,\s*\"[^\"]*\"\s*,\s*\[[^\]]*\]\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", open(sys.argv[1]).read())
print(max(int(o) + int(b) for o, b in rows) if rows else 0)' "$1"
}

# file_progress <pid[,pid...]> <label> <total bytes> <file>...: every PROGRESS_S seconds (15) a line "<label>: X of Y GB
# (Z %), R GB/s, about T left" from the summed sizes of the files (a tool's dot file and the name it renames it to:
# whichever exists), until the files reach the total (a last line at 100 %) or every process has ended (a last line as
# they are). Nothing when the files are whole already.
file_progress() {
  local pids=${1//,/ } label=$2 total=$3 every=${PROGRESS_S:-15} t0 now done0 b next tick
  shift 3
  _fp_alive() { local q; for q in $pids; do kill -0 "$q" 2>/dev/null && return 0; done; return 1; }
  _fp_bytes() { local f s=0; for f in "$@"; do [[ -f $f ]] && s=$(( s + $(stat -c %s "$f") )); done; echo "$s"; }
  _fp_line() {
    awk -v l="$label" -v b="$1" -v t="$total" -v d="$2" -v s="$3" 'BEGIN {
      u = (t >= 1e9) ? 1e9 : 1e6; n = (t >= 1e9) ? "GB" : "MB"; r = (s > 0) ? (b - d) / s : 0
      pc = (t > 0) ? int(100 * b / t) : 0; if (pc > 100) pc = 100
      line = sprintf("    %s: %.1f of %.1f %s (%d %%)", l, b / u, t / u, n, pc)
      if (r > 0) { line = line sprintf(", %.2f %s/s", r / u, n); left = (t - b) / r
                   if (b < t) line = line (left >= 90 ? sprintf(", about %d min left", int(left / 60 + 0.5)) : sprintf(", about %d s left", (left < 1 ? 1 : int(left + 0.5)))) }
      print line }'
  }
  _fp_since() { awk -v a="$(date +%s.%N)" -v z="$t0" 'BEGIN { printf "%.3f\n", a - z }'; }
  t0=$(date +%s.%N); done0=$(_fp_bytes "$@")
  (( done0 < total )) || return 0
  next=$(awk -v z="$t0" -v e="$every" 'BEGIN { printf "%.3f\n", z + e }'); tick=$(awk -v e="$every" 'BEGIN { printf "%.3f\n", (e < 1 ? e : 1) }')
  while _fp_alive; do                            # a check every second at most, a line every PROGRESS_S
    sleep "$tick"
    b=$(_fp_bytes "$@")
    if (( b >= total )); then _say "$(_fp_line "$b" "$done0" "$(_fp_since)")"; return 0; fi
    _fp_alive || break
    now=$(date +%s.%N)
    awk -v n="$now" -v x="$next" 'BEGIN { exit !(n >= x) }' || continue
    next=$(awk -v n="$now" -v e="$every" 'BEGIN { printf "%.3f\n", n + e }')
    _say "$(_fp_line "$b" "$done0" "$(_fp_since)")"
  done
  _say "$(_fp_line "$(_fp_bytes "$@")" "$done0" "$(_fp_since)")"
}

# engram_pack_progress [engram-pack.sh's arguments]: scripts/engram-pack.sh with the packing's progress beside it (every
# node's shards in this Spark's packed folder; the packer writes each as a dot file and renames it whole). Needs
# scripts/config.sh's helpers.
engram_pack_progress() {
  local nodes pdir total=0 g name bytes pid
  local -a watched=()
  nodes=$(packed_nodes); pdir=$(packed_dir_here)
  for (( g = 0; g < nodes; g++ )); do
    while read -r name bytes _; do
      [[ -n $name ]] || continue
      total=$(( total + bytes )); watched+=("$pdir/$name" "$pdir/.$name.partial")
    done < <(packed_files "$g" "$nodes")
  done
  ./scripts/engram-pack.sh "$@" & pid=$!
  file_progress "$pid" "Engram's packed tables" "$total" "${watched[@]}"
  wait "$pid"
}

# no_settings_help <example>: what ./setup.sh says before scripts/local.sh exists: what the file is, how to make it from
# the example, the example's own worker lines (its placeholders called so), what else must be ready, where to read more.
no_settings_help() {
  local ex=$1 workers placeholder=0
  workers=$(sed -n -E 's/^(WORKER[123]=[^[:space:]#]*).*/\1/p' "$ex")
  grep -q '=user@192\.0\.2\.' <<<"$workers" && placeholder=1
  cat <<EOT

./setup.sh reads this install's settings from scripts/local.sh, and that file does not exist yet.
It is your own copy of scripts/local.sh.example (the template: every setting is explained there); git ignores it,
so your settings never mix with the repository's.

  1. Make it:

       cp scripts/local.sh.example scripts/local.sh

  2. In it, the three other Sparks, as this Spark reaches them with ssh (user@address):

$(sed 's/^/       /' <<<"$workers")

EOT
  if (( placeholder )); then
    echo "     These are the example's placeholders: replace them with your three Sparks."
  else
    echo "     The example already names these three: if they are your Sparks, the file works copied as it is."
  fi
  cat <<'EOT'
     Every other setting has a working default: where the model is or will be downloaded (HF_CACHE, ~510 GB),
     the model's weights plain or abliterated (ABLIT), the fabric rails' addresses (RAIL_A_ADDRS, RAIL_B_ADDRS).

  3. Run ./setup.sh again.

The setup also needs key-based ssh from this Spark to the other three (ssh-copy-id user@address, once for each),
and the switch set up as NETWORK.md describes. README.md, "Set up", walks through all of it.

EOT
}

# fx_lines <name>: tools/fabric-xfer.sh's output as the setup shows it: its progress lines as "<name>: X of Y GB (Z %),
# rate, about T left", its per-Spark verification lines folded into one "done" line, any failure kept on the screen, the
# rest (memory, rails, rings, timing) to the setup's log file only
fx_lines() {
  local name=$1 line files="" bad=0 eta left m s
  while IFS= read -r line; do
    if [[ $line =~ \]\ written\ ([0-9.]+)\ of\ ([0-9.]+)\ GB\ \(([0-9]+)\ %\),\ ([0-9.]+)\ GB/s(,\ ETA\ ([0-9:]+))? ]]; then
      left=""; eta=${BASH_REMATCH[6]}
      if [[ -n $eta ]]; then
        IFS=: read -r -a parts <<<"$eta"; s=0; for m in "${parts[@]}"; do s=$(( s * 60 + 10#$m )); done
        if (( s >= 60 )); then left=", about $(( s / 60 )) min $(( s % 60 )) s left"; else left=", about $s s left"; fi
      fi
      _say "  $name: ${BASH_REMATCH[1]} of ${BASH_REMATCH[2]} GB (${BASH_REMATCH[3]} %), ${BASH_REMATCH[4]} GB/s$left"
      _detail "$line"
    elif [[ $line =~ :\ ([0-9]+)\ files\ verified\ by\ per-chunk\ CRC-32C,\ ([0-9]+)\ failed ]]; then
      _detail "$line"
      if (( BASH_REMATCH[2] > 0 )); then bad=1; _say "$line"; else files=${BASH_REMATCH[1]}; fi
    elif [[ $line =~ ERROR|FAIL|[Ff]ailed|refused ]]; then
      _say "$line"; bad=1
    else
      _detail "$line"
    fi
  done
  (( bad )) || [[ -z $files ]] || _say "  $name: done, $files files on each Spark, every chunk checked"
  return 0
}

# sudo_rule_script <helper>: the script the setup runs as root (sudo -S, with the password typed once) on a Spark without
# passwordless sudo. The rule that lets the sudo user run the helper and nothing else is written to a temporary file and
# checked by visudo first; only then are the helper and the rule installed, root-owned, under their own names
# (/usr/local/sbin/dsv41-root, /etc/sudoers.d/90-dsv41-start). A user name a rule cannot hold safely installs nothing;
# no other sudoers file is read or written.
sudo_rule_script() {
  local b64; b64=$(base64 -w0 < "$1")
  cat <<SCRIPT
set -eu
u=\${SUDO_USER:-}
case "\$u" in ''|-*|ALL|*[!A-Za-z0-9._-]*) echo "the user name '\$u' is not one a sudo rule can hold safely: nothing installed" >&2; exit 3 ;; esac
t=\$(mktemp); h=\$(mktemp); trap 'rm -f "\$t" "\$h"' EXIT
printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/dsv41-root\n' "\$u" > "\$t"
visudo -cf "\$t" >/dev/null || { echo "visudo refused the sudo rule: nothing installed" >&2; exit 3; }
echo $b64 | base64 -d > "\$h"
install -o root -g root -m 0755 "\$h" /usr/local/sbin/dsv41-root
install -o root -g root -m 0440 "\$t" /etc/sudoers.d/90-dsv41-start
SCRIPT
}
