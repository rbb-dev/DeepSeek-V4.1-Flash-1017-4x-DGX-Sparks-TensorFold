#!/usr/bin/env bash
# Set up four DGX Sparks for this engine from your scripts/local.sh (copy it from scripts/local.sh.example first):
#   1. config   your local.sh: what you must still change from the example, and what the setup will use
#   2. access   key-based ssh from this Spark to the three others; sudo on all four: passwordless sudo as it is, or
#               on a Spark without it, a sudo rule for only the start's two root commands (the GPU clock limit and
#               the page-cache drop, through /usr/local/sbin/dsv41-root), installed with that Spark's password, which
#               the setup asks once (a network run uses the same password for the rails)
#   3. network  each Spark's two fabric rails (static addresses, MTU 9000: /etc/netplan/51-cx7-roce-static.yaml) and
#               its RoCE flow control (network/roce-pfc.service); then every rail checked: up, its MTU, its PCIe
#               link at x4, PFC on priority 3, and a full-size ping that may not be split, between every pair
#   4. data     scripts/prepare.sh (the image, the model, the kit, each rank's weight file, Engram's packed shards),
#               each worker's own copy sent over the fabric (no ssh in the data path), checked file by file
#   5. start    an API key and an admin key (made in KEY_DIR, default ~/.config/dsv41, unless API_KEY_FILE and
#               ADMIN_KEY_FILE name yours), the engine started (./start.sh), the addresses and keys printed
# Run it again any time: each step checks first and changes only what differs ("already set" otherwise).
#   ./setup.sh               every step          ./setup.sh check     steps 1-2, changing nothing
#   ./setup.sh access        steps 1-2           ./setup.sh network   steps 1-3
#   ./setup.sh data          steps 1, 2 and 4    ./setup.sh remove    the setup's sudo rule removed; says what stays
# The network step keeps a copy of every /etc/netplan file it finds (/var/backups/dsv41-setup/) before it writes its own.
# An earlier fabric file that configures only the ConnectX-7 ports (NVIDIA's two-Spark playbook writes one) moves into
# that copy; any other netplan file that also configures a rail stops the step: that file is yours to change.
# The switch is yours to set up (NETWORK.md: what any RoCE switch needs, and a MikroTik CRS812's commands).
# ./start.sh starts the engine again after a stop (./stop.sh).
set -euo pipefail
cd "$(dirname "$0")"
WHAT=${1:-all}
[[ $WHAT =~ ^(all|check|access|network|data|remove)$ ]] || { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
source ./scripts/setup-lib.sh
[[ -f scripts/local.sh ]] || { no_settings_help scripts/local.sh.example >&2; exit 1; }
source ./scripts/config.sh
source ./scripts/nodes.sh
# every line shown is also kept, with the details the helpers write (detail), in one log file a run
mkdir -p "$LOG_DIR"
export SETUP_LOG="$LOG_DIR/setup-$(date +%Y%m%d-%H%M%S).log"
exec 3>&1 4>&2                       # where the caller sees the output; the keys go only there
exec > >(tee -a "$SETUP_LOG") 2>&1
tee_pid=$!
trap 'exec 1>&3 2>&4; wait "$tee_pid" 2>/dev/null || true' EXIT
export SETUP_TAG

# ---------------------------------------------------------------- 1. config
SETUP_TAG="1/5 config"
log "Settings from scripts/local.sh"
todo=()
for v in WORKER1 WORKER2 WORKER3; do
  [[ -n ${!v:-} ]] || { todo+=("$v: the ssh target (user@address) of one of the three other Sparks"); continue; }
  [[ ${!v} != user@192.0.2.* ]] || todo+=("$v=${!v} is the example's: the ssh target (user@address) of one of the three other Sparks")
done
read -ra ra <<<"$RAIL_A_ADDRS"; read -ra rb <<<"$RAIL_B_ADDRS"
(( ${#ra[@]} == 4 && ${#rb[@]} == 4 )) || todo+=("RAIL_A_ADDRS and RAIL_B_ADDRS: four addresses each, this Spark's first")
[[ $(printf '%s\n' "${ra[@]}" "${rb[@]}" | sort -u | wc -l) == 8 ]] || todo+=("RAIL_A_ADDRS and RAIL_B_ADDRS: eight different addresses")
if (( ${#todo[@]} )); then
  printf '  %s\n' "Change these in scripts/local.sh, then run ./setup.sh again:" "${todo[@]/#/- }" >&2
  exit 1
fi
hosts=(this "$WORKER1" "$WORKER2" "$WORKER3")
log "  This Spark and three others: $WORKER1, $WORKER2, $WORKER3"
log "  Fabric: rail A ($RAIL_A_IF) ${ra[*]}; rail B ($RAIL_B_IF) ${rb[*]}; /$RAIL_PREFIX, MTU $RAIL_MTU"
snap="$HF_CACHE/hub/models--${MODEL_ID//\//--}"
if [[ -d $snap ]]; then log "  Model: $MODEL_ID in $HF_CACHE"
else log "  Model: not in $HF_CACHE yet; the data step downloads it there (~510 GB; set HF_CACHE if you have it elsewhere)"; fi
for i in 1 2 3; do
  c=$(worker_hf_cache "$i")
  if [[ -n $c ]]; then log "  ${hosts[i]} gets its own copy in $c, over the fabric"
  else log "  ${hosts[i]} reads this Spark's copy over NFS (WORKER_HF_CACHE$i is empty): export $HF_CACHE to it yourself"; fi
done

# on <rank> <command>: a command on that rank's Spark (0: this one), passed as an argument, never on stdin
on() { if (( $1 == 0 )); then bash -c "$2" < /dev/null; else worker "$1" "$2" < /dev/null; fi; }
name() { if (( $1 == 0 )); then hostname; else worker_host "$1"; fi; }

# ---------------------------------------------------------------- 2. access
SETUP_TAG="2/5 access"
log "Key-based ssh and sudo on every Spark"
bad=0
for i in 1 2 3; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=8 "${hosts[i]}" true 2>/dev/null \
    || { warn "no key-based ssh to ${hosts[i]}: run   ssh-copy-id ${hosts[i]}   (it asks that Spark's password once)"; bad=1; }
done
(( ! bad )) || die "set up ssh first (above), then run ./setup.sh again"
spark() { if (( $1 == 0 )); then echo "this Spark"; else echo "${hosts[$1]}"; fi; }
# root_on <rank> <script>: the script run as root on that Spark, with the password on this call's input (sudo -S): the
# password reaches sudo over ssh's input, never a command line or a file
root_on() {
  local cmd="sudo -S -p '' bash -c \"\$(echo $(printf '%s' "$2" | base64 -w0) | base64 -d)\""
  if (( $1 == 0 )); then bash -c "$cmd"; else worker "$1" "$cmd"; fi
}
# sudo (D111): a Spark with passwordless sudo keeps it and gets nothing. Any other gets the setup's narrow rule: a root
# helper (scripts/dsv41-root, installed as /usr/local/sbin/dsv41-root) the user may run with sudo, and nothing else, for
# the start's two root commands (the GPU clock limit and the page-cache drop). Its password is asked once, on a terminal,
# for what this run needs as root there: the rule when it is missing or older than this copy's, and the rails in a
# network run.
HELPER=/usr/local/sbin/dsv41-root
helper_sum=$(sha256sum < scripts/dsv41-root | cut -d' ' -f1)
# old_rule <rank>: what /etc/sudoers.d/90-dsv41 holds there: "full" when it is exactly the line an earlier setup wrote
# ("<user> ALL=(ALL) NOPASSWD: ALL"), "other" when it holds anything else, nothing when it is absent or unreadable
old_rule() {
  local t; t=$(on "$1" "sudo -n cat /etc/sudoers.d/90-dsv41" 2>/dev/null) || return 0
  if [[ $t =~ ^[A-Za-z0-9._-]+\ ALL=\(ALL\)\ NOPASSWD:\ ALL$ ]]; then echo full; elif [[ -n $t ]]; then echo other; fi
}
sudo_mode=(); sudo_pw=()
if [[ $WHAT == remove ]]; then
  SETUP_TAG="remove"
  for i in 0 1 2 3; do
    if on "$i" "sudo -n $HELPER check" >/dev/null 2>&1; then
      on "$i" "sudo -n $HELPER uninstall" && log "  $(spark "$i"): the setup's sudo rule removed"
    elif on "$i" "sudo -n test -e $HELPER -o -e /etc/sudoers.d/90-dsv41-start" >/dev/null 2>&1; then
      on "$i" "sudo -n rm -f $HELPER /etc/sudoers.d/90-dsv41-start" && log "  $(spark "$i"): the setup's sudo rule removed"
    else log "  $(spark "$i"): no sudo rule of the setup's here"; fi
    # last on this Spark: the full rule an earlier setup wrote, only if it is exactly that rule
    case $(old_rule "$i") in
      full) on "$i" "sudo -n rm -f /etc/sudoers.d/90-dsv41" && log "  $(spark "$i"): the earlier setup's full sudo rule removed (/etc/sudoers.d/90-dsv41)" ;;
      other) log "  $(spark "$i"): /etc/sudoers.d/90-dsv41 holds something the setup did not write: left as it is" ;;
    esac
  done
  log "What stays, yours to remove if you want (NETWORK.md says how): each Spark's rails (/etc/netplan/51-cx7-roce-static.yaml;"
  log "  the network files it replaced are kept in /var/backups/dsv41-setup/) and its flow control (roce-pfc.service); the"
  log "  model, the weight files and Engram's tables (in $HF_CACHE here, and in each worker's own copy); the engine image"
  exit 0
fi
for i in 0 1 2 3; do
  if on "$i" "sudo -n true" >/dev/null 2>&1; then
    sudo_mode[i]=full
    if [[ $(old_rule "$i") == full ]]; then log "  $(spark "$i"): OK (the earlier setup's full sudo rule, /etc/sudoers.d/90-dsv41: ./setup.sh remove takes it away)"
    else log "  $(spark "$i"): OK"; fi
  elif [[ "$(on "$i" "sudo -n $HELPER check" 2>/dev/null)" == "$helper_sum" ]]; then
    sudo_mode[i]=rule; log "  $(spark "$i"): OK (the setup's sudo rule)"
  else
    sudo_mode[i]=none
    log "  $(spark "$i"): no passwordless sudo: ./setup.sh asks its password once and installs a sudo rule for only the"
    log "    start's two root commands (the GPU clock limit and the page-cache drop)"
  fi
done
[[ $WHAT != check ]] || { log "Settings and access: ready"; exit 0; }
need=()
for i in 0 1 2 3; do
  if [[ ${sudo_mode[i]} == none ]] || [[ ${sudo_mode[i]} == rule && ( $WHAT == all || $WHAT == network ) ]]; then need+=("$i"); fi
done
if (( ${#need[@]} )); then
  names=(); for i in "${need[@]}"; do names+=("$(spark "$i")"); done
  [[ -t 0 ]] || die "this run needs the sudo password of $(sparks_phrase 0 "${names[@]}"): run ./setup.sh in a terminal (it asks each one once)"
  for i in "${need[@]}"; do
    for try in 1 2 3; do
      IFS= read -rs -p "  Password for sudo on $(spark "$i") (asked once; kept only while the setup runs): " pw || pw=
      echo
      if printf '%s\n' "$pw" | root_on "$i" true >/dev/null 2>&1; then sudo_pw[i]=$pw; break; fi
      warn "that password did not work on $(spark "$i")"
    done
    unset pw
    [[ -n ${sudo_pw[i]+x} ]] || die "no working sudo password for $(spark "$i"): nothing was changed there"
    [[ ${sudo_mode[i]} == none ]] || continue
    printf '%s\n' "${sudo_pw[i]}" | root_on "$i" "$(sudo_rule_script scripts/dsv41-root)" 2>&1 | while IFS= read -r l; do log "  $(spark "$i"): $l"; done
    [[ "$(on "$i" "sudo -n $HELPER check" 2>/dev/null)" == "$helper_sum" ]] || die "$(spark "$i"): the sudo rule is not in place (above)"
    sudo_mode[i]=rule
    log "  $(spark "$i"): the setup's sudo rule installed (/etc/sudoers.d/90-dsv41-start: $HELPER, nothing else)"
  done
fi
[[ $WHAT == all || $WHAT == network ]] || sudo_pw=()
[[ $WHAT != access ]] || { log "Settings and access: ready"; exit 0; }

# ---------------------------------------------------------------- 3. network
netplan_for() {  # <rank>: the rails' netplan file for that Spark
  cat <<EOF
# Written by setup.sh: this Spark's two fabric rails (NETWORK.md), one subnet each, MTU $RAIL_MTU.
network:
  version: 2
  ethernets:
    $RAIL_A_IF:
      addresses: [${ra[$1]}/$RAIL_PREFIX]
      mtu: $RAIL_MTU
      link-local: [ ipv4 ]
    $RAIL_B_IF:
      addresses: [${rb[$1]}/$RAIL_PREFIX]
      mtu: $RAIL_MTU
      link-local: [ ipv4 ]
EOF
}
unit_text() { sed -e "s/enp1s0f0np0/$RAIL_A_IF/g" -e "s/enP2p1s0f0np0/$RAIL_B_IF/g" network/roce-pfc.service; }
# setup_rails <rank>: writes and applies what differs (comments and blank lines aside), says what it did
setup_rails() {
  local i=$1 np unit
  np=$(netplan_for "$i" | base64 -w0); unit=$(unit_text | base64 -w0)
  # the check runs on the Spark with its ConnectX-7 interfaces (driver mlx5_core) as the ports an earlier fabric file may hold
  chk=$( { declare -f netplan_conflicts
           echo 'ports=$(for d in /sys/class/net/*; do [ "$(basename "$(readlink -f "$d/device/driver" 2>/dev/null)" 2>/dev/null)" = mlx5_core ] && echo "${d##*/}"; done)'
           printf 'netplan_conflicts /etc/netplan 51-cx7-roce-static.yaml %q "$ports"\n' "$RAIL_A_IF $RAIL_B_IF"; } | base64 -w0)
  local s="set -e
    norm() { grep -v -E '^[[:space:]]*(#|\$)' || true; }
    f=/etc/netplan/51-cx7-roce-static.yaml; u=/etc/systemd/system/roce-pfc.service
    c=\$(echo $chk | base64 -d | sudo -n bash) || { echo 'the network-file check failed (it needs python3 with yaml, which netplan brings)' >&2; exit 3; }
    stops=\$(printf '%s\n' \"\$c\" | sed -n 's/^stop //p'); moves=\$(printf '%s\n' \"\$c\" | sed -n 's/^move //p')
    if [ -n \"\$stops\" ]; then echo \"another network file also configures the rails, with more than this Spark's ConnectX-7 ports: \$(echo \$stops)\" >&2
      echo 'remove the two rails from it (the setup does not change your files), then run ./setup.sh network again' >&2; exit 3; fi
    want=\$(echo $np | base64 -d)
    if [ -z \"\$moves\" ] && [ \"\$(printf '%s\n' \"\$want\" | norm)\" = \"\$(sudo -n cat \$f 2>/dev/null | norm)\" ]; then echo 'rails: already set'
    else bk=/var/backups/dsv41-setup/netplan-\$(date +%Y%m%d-%H%M%S); sudo -n mkdir -p \$bk
         sudo -n sh -c \"cp -p /etc/netplan/*.yaml \$bk/ 2>/dev/null\" || true
         echo \"network files backed up (\$bk)\"
         printf '%s\n' \"\$moves\" | while IFS= read -r m; do [ -n \"\$m\" ] || continue; sudo -n mv \"\$m\" \$bk/
           echo \"an earlier fabric file (\${m##*/}, only this Spark's ConnectX-7 ports) moved into that backup\"; done
         printf '%s\n' \"\$want\" | sudo -n tee \$f >/dev/null; sudo -n chmod 600 \$f
         sudo -n netplan generate; sudo -n netplan apply; echo 'rails set'; fi
    wantu=\$(echo $unit | base64 -d)
    if [ \"\$(printf '%s\n' \"\$wantu\" | norm)\" = \"\$(cat \$u 2>/dev/null | norm)\" ] && systemctl is-enabled -q roce-pfc.service \
       && systemctl is-active -q roce-pfc.service; then echo 'flow control: already set'
    else printf '%s\n' \"\$wantu\" | sudo -n tee \$u >/dev/null
         sudo -n systemctl daemon-reload; sudo -n systemctl enable roce-pfc.service >/dev/null 2>&1
         sudo -n systemctl restart roce-pfc.service; echo 'flow control on'; fi"
  # a Spark without passwordless sudo: the same script as root, with the password typed in the access step
  if [[ ${sudo_mode[i]} == full ]]; then on "$i" "$s"; else printf '%s\n' "${sudo_pw[i]}" | root_on "$i" "$s"; fi
}
# rail_state <rank>: one line a rail: name, state, MTU, address, PCIe width, PFC on priority 3
rail_state() {
  on "$1" "for d in $RAIL_A_IF $RAIL_B_IF; do
    echo \"\$d \$(cat /sys/class/net/\$d/operstate 2>/dev/null) \$(cat /sys/class/net/\$d/mtu 2>/dev/null) \
\$(ip -4 -br addr show dev \$d 2>/dev/null | grep -oE '[0-9.]+/[0-9]+' | head -1) \
x\$(cat /sys/class/net/\$d/device/current_link_width 2>/dev/null) \
\$(dcb pfc show dev \$d 2>/dev/null | grep -oE 'prio-pfc.*' | grep -q '3:on' && echo pfc3 || echo no-pfc3)\"; done"
}
if [[ $WHAT == all || $WHAT == network ]]; then
  SETUP_TAG="3/5 network"
  log "Fabric rails and RoCE flow control, the other Sparks first"
  for i in 1 2 3 0; do
    who=$( (( i == 0 )) && echo "this Spark" || echo "${hosts[i]}")
    setup_rails "$i" 2>&1 | while IFS= read -r l; do log "  $who: $l"; done || die "$who: its rails are not set (above)"
  done
  sudo_pw=()
  log "Checking every rail"
  sleep 3
  bad=0
  for i in 0 1 2 3; do
    while read -r d st mtu addr w pfc; do
      want_addr=$([[ $d == "$RAIL_A_IF" ]] && echo "${ra[i]}" || echo "${rb[i]}")/$RAIL_PREFIX
      ok=1; [[ $st == up && $mtu == "$RAIL_MTU" && $addr == "$want_addr" && $w == x4 && $pfc == pfc3 ]] || { ok=0; bad=1; }
      log "$(printf '  %-21s rail %s  %-4s  MTU %-5s  %-16s  PCIe %-3s  flow control %-3s  %s' "$( (( i == 0 )) && echo "this Spark" || echo "${hosts[i]}")" "$([[ $d == "$RAIL_A_IF" ]] && echo A || echo B)" "$st" "$mtu" "${addr:--}" "$w" "$([[ $pfc == pfc3 ]] && echo on || echo off)" "$( (( ok )) && echo OK || echo '<- WRONG')")"
    done < <(rail_state "$i")
  done
  for i in 0 1 2 3; do
    for j in 0 1 2 3; do
      (( i != j )) || continue
      for r in A B; do
        if [[ $r == A ]]; then dev=$RAIL_A_IF; dst=${ra[j]}; else dev=$RAIL_B_IF; dst=${rb[j]}; fi
        on "$i" "ping -c2 -W2 -I $dev -M do -s $(( RAIL_MTU - 28 )) $dst >/dev/null 2>&1" \
          || { warn "  $(name "$i") -> $(name "$j") rail $r ($dst): no answer to a full-size ping"; bad=1; }
      done
    done
  done
  (( ! bad )) || die "the fabric is not right (above): check the cables and the switch (NETWORK.md), then run ./setup.sh network again"
  log "Full-size pings (MTU $RAIL_MTU) between every pair of Sparks on both rails: 24 of 24 answered"
fi
[[ $WHAT != network ]] || { log "Network: ready"; exit 0; }

# ---------------------------------------------------------------- 4. data
SETUP_TAG="4/5 data"
SETUP_STARTS=$([[ $WHAT == all ]] && echo 1 || echo 0) ./scripts/setup-data.sh   # step 5 starts the engine after it
[[ $WHAT == all ]] || exit 0

# ---------------------------------------------------------------- 5. keys, then the engine
SETUP_TAG="5/5 start"
log "API key and admin key"
KEY_DIR=${KEY_DIR:-$HOME/.config/dsv41}
add_local() { printf '\n# added by ./setup.sh: %s\n%s=%s\n' "$3" "$1" "$2" >> scripts/local.sh; }
if [[ -z $API_KEY_FILE ]]; then
  API_KEY_FILE=$KEY_DIR/api-keys
  make_key_file "$API_KEY_FILE" default
  add_local API_KEY_FILE "$API_KEY_FILE" "the API's keys, one \"label: key\" a line (README, \"API keys\")"
  log "  API key made: $API_KEY_FILE (readable by you only; scripts/local.sh names it: API_KEY_FILE)"
fi
if [[ -z $ADMIN_KEY_FILE ]]; then
  ADMIN_KEY_FILE=$KEY_DIR/admin-keys
  make_key_file "$ADMIN_KEY_FILE" admin
  add_local ADMIN_KEY_FILE "$ADMIN_KEY_FILE" "the admin page's keys (README, \"Opening the dashboard\")"
  log "  admin key made: $ADMIN_KEY_FILE (readable by you only; scripts/local.sh names it: ADMIN_KEY_FILE)"
fi
export API_KEY_FILE ADMIN_KEY_FILE
log "Starting the engine"
./start.sh
# the double reset every measured start has had: the one engine a setup started once (right after its data copy) answered
# every prompt ~0.9 s late; every restart after it, at full speed. A restart stops it, drops the page caches, compacts the
# memory and starts it again.
log "Starting it once more (stop, page caches dropped, memory compacted, start): a first start right after this much copying can come up slower"
./start.sh restart
lan=$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -1)
log "The engine is up"
# the keys are shown to the caller only, never written to the setup's log file: the log's copy ends here
exec 1>&3 2>&4; wait "$tee_pid" 2>/dev/null || true
cat <<EOF

  API         http://${lan:-<this Spark>}:$PORT/v1    (OpenAI and Anthropic APIs; model name: $SERVED_NAME)
  API key     $(key_of "$API_KEY_FILE")
  Admin page  http://${lan:-<this Spark>}:$PORT/admin/ui/
  Admin key   $(key_of "$ADMIN_KEY_FILE")

  The API answers on every address of this Spark, port $PORT; every request needs a key (send it as
  "Authorization: Bearer <key>"). Keep the keys private: they are in $KEY_DIR, readable by you only.
  README, "API keys", says how to add more keys and change them.
  Logs: $LOG_DIR    Stop: ./stop.sh    Start again: ./start.sh

EOF
