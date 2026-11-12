#!/usr/bin/env bash
# Install network/roce-pfc.service on each Spark over ssh and enable it for every boot; with --start, apply it now
# too (setting PFC can restart a fabric link for a few seconds: run it with nothing on the fabric). Prints each Spark's
# read-back: the unit's state, PFC and the DSCP map of both rails. Needs passwordless sudo on the Sparks.
#   network/install-roce-pfc.sh [--start] user@spark1 user@spark2 user@spark3 user@spark4
set -u
cd "$(dirname "$0")" || exit 2
start=0; [[ ${1:-} == --start ]] && { start=1; shift; }
(( $# )) || { echo "usage: $0 [--start] user@spark ..." >&2; exit 2; }
rc=0
for h in "$@"; do
  if ! scp -q -o BatchMode=yes roce-pfc.service "$h:/tmp/roce-pfc.service" < /dev/null; then echo "$h: copy failed"; rc=1; continue; fi
  ssh -o BatchMode=yes "$h" "sudo -n install -m 644 /tmp/roce-pfc.service /etc/systemd/system/roce-pfc.service &&
    sudo -n systemctl daemon-reload && sudo -n systemctl enable roce-pfc.service >/dev/null 2>&1 &&
    { [ $start = 0 ] || sudo -n systemctl start roce-pfc.service; } &&
    echo \"\$(hostname): \$(systemctl is-enabled roce-pfc.service), \$(systemctl is-active roce-pfc.service)\" &&
    for d in enp1s0f0np0 enP2p1s0f0np0; do echo \"  \$d: \$(dcb pfc show dev \$d | grep -o 'prio-pfc.*'), \$(dcb app show dev \$d | tr '\n' ' ')\"; done" < /dev/null || { echo "$h: install failed"; rc=1; }
done
exit $rc
