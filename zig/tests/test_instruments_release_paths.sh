#!/usr/bin/env bash
# No release path may build with test instruments (-Dtest_instruments, build.zig): a binary that carries them times its
# hot paths and refuses to serve without TF_DS_TEST_INSTRUMENTS=1, and an image or package built with the flag would
# slow or stop the engine it ships in. Scans the files and directories given (default: this tree's packaging, its
# release build file, its CI, its scripts and any Dockerfile) for the flag and fails, naming each line, when one passes
# it. The fork's image recipe lives outside this tree: pass its Dockerfile and scripts as arguments
# (zig/tests/test_instruments_release_paths.sh ~/src/ds41/fork/zig/Dockerfile ~/src/ds41/fork/scripts).
#   --binaries F...  fail when any binary contains the words "TEST INSTRUMENTS BUILD" (only an instrument build emits them;
#                    run it over a release image's engine binaries: tensorfold-native, tf-dsv41-lanes, ...)
#   --self-test   a file that does pass the flag must fail the scan (the check can fail), one that does not must pass
set -u
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

scan() {
  local bad=0 hit
  while IFS= read -r hit; do
    echo "test-instruments: a release path passes -Dtest_instruments: $hit" >&2
    bad=1
  done < <(grep -rnE -e 'test_instruments' -- "$@" 2>/dev/null | grep -vE '^[^:]*:[0-9]+:[[:space:]]*(#|//)')
  return $bad
}

marker_in() { grep -a -q -F 'TEST INSTRUMENTS BUILD' -- "$1"; }

if [[ ${1:-} == --binaries ]]; then
  shift; (( $# > 0 )) || { echo "test-instruments: no binaries given" >&2; exit 1; }
  bad=0
  for f in "$@"; do
    [[ -f $f ]] || { echo "test-instruments: not a file: $f" >&2; bad=1; continue; }
    if marker_in "$f"; then echo "test-instruments: $f carries the words TEST INSTRUMENTS BUILD: it was built with -Dtest_instruments" >&2; bad=1; fi
  done
  (( bad == 0 )) && echo "test-instruments: none of $# binaries carries the instrument build's words"
  exit $bad
fi

if [[ ${1:-} == --self-test ]]; then
  d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
  printf 'RUN zig build -Dcpu=gb10 -Dtest_instruments=true install\n' > "$d/Dockerfile.bad"
  printf 'RUN zig build -Dcpu=gb10 install\n' > "$d/Dockerfile.good"
  scan "$d/Dockerfile.bad" 2> /dev/null && { echo "test-instruments: self-test: a Dockerfile passing the flag was not caught" >&2; exit 1; }
  scan "$d/Dockerfile.good" || { echo "test-instruments: self-test: a clean Dockerfile was refused" >&2; exit 1; }
  printf 'xx TEST INSTRUMENTS BUILD: yy' > "$d/bin.bad"; printf 'clean' > "$d/bin.good"
  bash "${BASH_SOURCE[0]}" --binaries "$d/bin.bad" 2> /dev/null && { echo "test-instruments: self-test: a binary with the words was not caught" >&2; exit 1; }
  bash "${BASH_SOURCE[0]}" --binaries "$d/bin.good" > /dev/null || { echo "test-instruments: self-test: a clean binary was refused" >&2; exit 1; }
  echo "test-instruments: self-test ok (the scan refuses a release path that passes the flag and passes one that does not)"
  exit 0
fi

if (( $# > 0 )); then targets=("$@"); else
  targets=()
  for p in packaging zig/build/dist.zig .github scripts setup.py zig/Dockerfile; do [[ -e $root/$p ]] && targets+=("$root/$p"); done
  while IFS= read -r f; do targets+=("$f"); done < <(find "$root" -maxdepth 3 -name 'Dockerfile*' -not -path '*/.zig-cache/*' -not -path '*/node_modules/*')
fi
(( ${#targets[@]} > 0 )) || { echo "test-instruments: nothing to scan" >&2; exit 1; }
scan "${targets[@]}" || exit 1
echo "test-instruments: none of ${#targets[@]} release path(s) passes -Dtest_instruments"
