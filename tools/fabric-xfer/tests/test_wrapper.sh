#!/usr/bin/env bash
# The wrapper's argument handling, with ssh, ip and the binary replaced by recorders. No network, no RDMA.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d /tmp/fxwrap.XXXXXX); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/src"
cat > "$T/bin/ssh" <<'X'
#!/usr/bin/env bash
# last two args: target and remote command; stdin is "TOKEN\n" + the binary
args=("$@"); n=${#args[@]}
printf '%s\t%s\n' "${args[n-2]}" "${args[n-1]}" >> "$FXLOG.ssh"
head -n 1 >> "$FXLOG.sshtok"
cat > /dev/null
echo FX-READY
X
cat > "$T/bin/ip" <<'X'
#!/usr/bin/env bash
echo "$4 dev enp1s0f0np0 src ${4%.*}.99 uid 0"
X
cat > "$T/fakebin" <<'X'
#!/usr/bin/env bash
echo "$*" >> "$FXLOG.bin"; cat <&3 >> "$FXLOG.bintok"; cat > "$FXLOG.paths"
X
chmod +x "$T/bin/ssh" "$T/bin/ip" "$T/fakebin"
export FXLOG=$T/log FX_BIN=$T/fakebin PATH=$T/bin:$PATH
fails=0
check() { if ! eval "$2"; then echo "FAIL: $1"; fails=$((fails+1)); fi; }

printf 'a\nb/c\n' | FX_OPTS="--readback --ring-gib 12" FX_SEND_OPTS="--manifest /tmp/m.json" \
  "$HERE/fabric-xfer.sh" -n 4 -p 5500 -l lab node2,node3 "10.255.0.2,10.255.1.2;10.255.0.3,10.255.1.3" "$T/src" '$HOME/dst'; rc=$?
check "exit 0" '[[ $rc == 0 ]]'
check "sender args" 'grep -q -- "send -n 4 -p 5500 -l lab --local 10.255.0.99,10.255.1.99 --token-fd 3 --target node2=10.255.0.2,10.255.1.2 --target node3=10.255.0.3,10.255.1.3 --readback --ring-gib 12 --manifest /tmp/m.json $T/src" "$FXLOG.bin"'
check "paths pass through stdin" '[[ $(cat "$FXLOG.paths") == $'"'"'a\nb/c'"'"' ]]'
check "receivers get an allowlist of the sender's rail addresses" 'grep -q -- "--allow 10.255.0.99,10.255.1.99 " "$FXLOG.ssh"'
check "a receiver per worker" '[[ $(wc -l < "$FXLOG.ssh") == 2 ]]'
check "receiver binds rail A and gets both addresses" 'grep -q -- "node3.*recv --bind 10.255.0.3 --port 5500 --addrs 10.255.0.3,10.255.1.3 --dest .\$HOME/dst. -n 4 --token-fd 0 --allow 10.255.0.99,10.255.1.99 --readback --ring-gib 12" "$FXLOG.ssh"'

tok=$(head -n 1 "$FXLOG.bintok")
check "token is 64 hex chars" '[[ $tok =~ ^[0-9a-f]{64}$ ]]'
check "every ssh got the same token on stdin" '[[ $(sort -u "$FXLOG.sshtok") == "$tok" && $(wc -l < "$FXLOG.sshtok") == 2 ]]'
check "the token is on no command line (ssh args, sender args)" '! grep -q -- "$tok" "$FXLOG.ssh" "$FXLOG.bin"'
check "the sender is told to read it from fd 3, receivers from stdin" 'grep -q -- "--token-fd 3" "$FXLOG.bin" && grep -q -- "--token-fd 0" "$FXLOG.ssh"'
rm -f "$FXLOG".*
printf 'a\n' | "$HERE/fabric-xfer.sh" -l lan 192.0.2.2 10.255.0.2,10.255.1.2 "$T/src" /data/x
check "an IPv4 ssh target adds the LAN address ssh uses" 'grep -q -- "--allow 10.255.0.99,10.255.1.99,192.0.2.99 " "$FXLOG.ssh"'
rm -f "$FXLOG".*
printf 'a\n' | "$HERE/fabric-xfer.sh" -s node2 10.255.0.2,10.255.1.2 "$T/src" /data/x; rc=$?
check "-s passes -s" 'grep -q -- "send -n 8 -p 5400 -l node2 -s " "$FXLOG.bin"'
for bad in "node2 10.255.0.x $T/src /d" "node2,node3 10.255.0.2,10.255.1.2 $T/src /d" "node2 10.255.0.2 /nonexistent /d" "node2 10.255.0.2 $T/src rel/d" "node2 10.255.0.2 $T/src /d;rm"; do
  out=$(echo a | "$HERE/fabric-xfer.sh" $bad 2>&1); rc=$?
  check "rejects: $bad" '[[ $rc != 0 && $out == *ERROR* ]]'
done
check "usage on wrong arity" '"$HERE/fabric-xfer.sh" onlyone 2>&1 | grep -q fabric-xfer.sh'
echo "test_wrapper.sh: $fails failed"; exit $((fails ? 1 : 0))
