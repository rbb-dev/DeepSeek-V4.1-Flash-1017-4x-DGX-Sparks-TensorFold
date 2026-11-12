"""Regression tests for stop.sh, ./start.sh restart, the recipe giving the engine no NCCL, the prepared marker,
prepare.sh's weight-file loop, and what start.sh and prepare.sh give the engine: the replay floor on every rank,
the SSD tier's settings, folder and identity on every Spark, each Spark's weight file's digest (each test reads
each host's docker run line); Engram's packed shards: their sizes from the tables' headers, the verifier, prepare.sh's
step 7, scripts/engram-pack.sh and what each rank mounts.

Offline: no Spark, GPU, Docker or network. Each test copies the scripts into a temporary folder and puts fake `ssh`,
`docker`, `curl` and `getent` first on PATH. The fake ssh runs a worker's command locally with that worker's own HOME
and container list (under the test's folder); the fake docker keeps each Spark's containers as lines in a file.
Run: python3 -m unittest discover -s tests       SCRIPTS_DIR=<a repository copy> tests the scripts there instead.
The scripts need bash 4 or later: BASH4=<path> picks one, else the first bash on PATH (or Homebrew's) that is.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(os.environ.get("SCRIPTS_DIR") or Path(__file__).resolve().parent.parent)
COPY = ["stop.sh", "start.sh", "scripts", "model", "zig"]


def find_bash():
    for cand in [os.environ.get("BASH4"), shutil.which("bash"), "/opt/homebrew/bin/bash", "/usr/local/bin/bash"]:
        if not cand or not os.path.exists(cand):
            continue
        out = subprocess.run([cand, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True).stdout.strip()
        if out.isdigit() and int(out) >= 4:
            return cand
    return None


BASH = find_bash()

FAKE_SSH = r"""#!BASH
# fake ssh: ssh [-o opt]... host command...; a host listed in $SIM/unreachable cannot be reached, one in $SIM/flaky
# answers `true` (the reachability probe) and drops every other command
opts=""; while [[ "$1" == -o ]]; do opts+="$2 "; shift 2; done
host=$1; shift
echo "$host $*" >> "$SIM/ssh.log"
echo "$host $opts" >> "$SIM/ssh-opts.log"
grep -qxF "$host" "$SIM/unreachable" 2>/dev/null && { echo "ssh: connect to host $host: No route to host" >&2; exit 255; }
grep -qxF "$host" "$SIM/flaky" 2>/dev/null && [[ "$*" != true ]] && { echo "Connection to $host closed by remote host." >&2; exit 255; }
mkdir -p "$SIM/home/$host"
SIM_HOST=$host HOME="$SIM/home/$host" exec BASH -c "$*"
"""

FAKE_DOCKER = r"""#!BASH
# fake docker: each Spark's containers are lines in $SIM/hosts/<host>/containers; a container named in
# $SIM/hosts/<host>/stuck cannot be stopped or removed; with $SIM/hosts/<host>/ps_fails `docker ps` fails; with
# $SIM/hosts/<host>/exited its containers are listed but not running; with
# $SIM/image the image exists (its identity: "image-ident"); `run -d --name N` adds N
h=${SIM_HOST:-head}; d="$SIM/hosts/$h"; mkdir -p "$d"; touch "$d/containers"
echo "$h docker $*" >> "$SIM/docker.log"
has() { grep -qxF "$1" "$d/containers"; }
stuck() { grep -qxF "$1" "$d/stuck" 2>/dev/null; }
last=${!#}
case "$1" in
  ps) [[ ! -e "$d/ps_fails" ]] || { echo "Cannot connect to the Docker daemon" >&2; exit 1; }; cat "$d/containers" ;;
  inspect) has "$last" || { echo "Error: No such object: $last" >&2; exit 1; }
           [[ "$*" != *State.Running* ]] || { [[ -e "$d/exited" ]] && echo false || echo true; } ;;
  stop) stuck "$last" && { echo "Error response from daemon: cannot stop container: $last" >&2; exit 1; }; true ;;
  rm) stuck "$last" && { echo "Error response from daemon: cannot remove container: $last" >&2; exit 1; }
      grep -vxF "$last" "$d/containers" > "$d/c.new" || true; mv "$d/c.new" "$d/containers" ;;
  logs) if [[ -e "$d/engine.log" ]]; then cat "$d/engine.log"; else echo "a log line"; fi ;;
  image) [[ -e "$SIM/image" ]] || exit 1; echo image-ident ;;
  run) [[ ! -e "$d/run_fails" ]] || { echo "docker: Error response from daemon: run refused" >&2; exit 125; }
       n=""; prev=""; for a in "$@"; do [[ "$prev" != --name ]] || n=$a; prev=$a; done; echo "$n" >> "$d/containers"; echo container-id ;;
  *) exit 0 ;;
esac
"""

FAKE_CURL = """#!/bin/sh
echo '{"requests_running": 0}'
"""

METRICS_CURL = r"""#!BASH
# fake curl: /metrics from $SIM/metrics.seq, one "running waiting" line a call (the last one repeats); else {}
echo "$*" >> "$SIM/curl.log"
case "$*" in
  *metrics*) n=$(cat "$SIM/metrics.n" 2>/dev/null || echo 0); echo $((n + 1)) > "$SIM/metrics.n"
             line=$(sed -n "$((n + 1))p" "$SIM/metrics.seq"); [[ -n "$line" ]] || line=$(tail -1 "$SIM/metrics.seq")
             set -- $line; echo "tensorfold:requests_running $1"; echo "tensorfold:requests_waiting $2" ;;
  *) echo '{}' ;;
esac
"""

FAKE_GETENT = """#!/bin/sh
exit 2
"""

# macOS's paste reads stdin only when told "-" (GNU paste, on a Spark, by itself): stop.sh's `paste -sd, <<<...`
FAKE_PASTE = """#!/bin/sh
for a in "$@"; do case "$a" in -*) ;; *) exec /usr/bin/paste "$@" ;; esac; done
exec /usr/bin/paste "$@" -
"""

NAME = "dsv41-1017-tf"
WORKERS = {"WORKER1": "w1", "WORKER2": "w2", "WORKER3": "w3"}


@unittest.skipIf(BASH is None, "needs bash 4 or later")
class ScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tf-scripts-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        for name in COPY:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, self.repo / name, ignore=shutil.ignore_patterns("local.sh", "__pycache__"))
            else:
                shutil.copy2(src, self.repo / name)
        self.sim = self.tmp / "sim"
        (self.sim / "hosts").mkdir(parents=True)
        bin_ = self.tmp / "bin"
        bin_.mkdir()
        for name, body in [("ssh", FAKE_SSH), ("docker", FAKE_DOCKER), ("curl", FAKE_CURL), ("getent", FAKE_GETENT)]:
            p = bin_ / name
            p.write_text(body.replace("BASH", BASH))
            p.chmod(0o755)
        if sys.platform == "darwin":
            (bin_ / "paste").write_text(FAKE_PASTE)
            (bin_ / "paste").chmod(0o755)
        # the scripts' #!/usr/bin/env bash: a bash 4 first on PATH
        (bin_ / "bash").symlink_to(BASH)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.env = {
            "PATH": f"{bin_}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(self.home),
            "SIM": str(self.sim),
            "LANG": "C",
            **WORKERS,
        }

    def containers(self, host, *names):
        d = self.sim / "hosts" / host
        d.mkdir(parents=True, exist_ok=True)
        (d / "containers").write_text("".join(n + "\n" for n in names))

    def left(self, host):
        f = self.sim / "hosts" / host / "containers"
        return f.read_text().split() if f.exists() else []

    def stuck(self, host):
        d = self.sim / "hosts" / host
        d.mkdir(parents=True, exist_ok=True)
        (d / "stuck").write_text(NAME + "\n")

    def unreachable(self, *hosts):
        (self.sim / "unreachable").write_text("".join(h + "\n" for h in hosts))

    def flaky(self, *hosts):
        (self.sim / "flaky").write_text("".join(h + "\n" for h in hosts))

    def ps_fails(self, host):
        d = self.sim / "hosts" / host
        d.mkdir(parents=True, exist_ok=True)
        (d / "ps_fails").write_text("")

    def run_script(self, args, extra_env=None, timeout=60):
        env = {**self.env, **(extra_env or {})}
        r = subprocess.run([BASH, *args], cwd=self.repo, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr

    def bash(self, body, extra_env=None):
        """Source config.sh and nodes.sh in the copy, then run body; returns (exit code, output)."""
        script = f"set -euo pipefail\ncd {self.repo}\nsource ./scripts/config.sh\nsource ./scripts/nodes.sh\n{body}\n"
        return self.run_script(["-c", script], extra_env)

    def assertNoTraceback(self, out):
        self.assertNotIn("unbound variable", out)
        self.assertNotIn("syntax error", out)
        self.assertNotIn("command not found", out)


class StopSh(ScriptTest):
    def all_up(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)

    def test_every_rank_stopped_exits_zero(self):
        self.all_up()
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 0, out)
        self.assertIn("its GPU memory is free again", out)
        for h in ["head", "w1", "w2", "w3"]:
            self.assertEqual(self.left(h), [], h)

    def busy(self, *seq):
        (self.sim / "metrics.seq").write_text("".join(f"{r} {w}\n" for r, w in seq))
        p = self.tmp / "bin" / "curl"
        p.write_text(METRICS_CURL.replace("BASH", BASH))
        p.chmod(0o755)

    def test_running_requests_are_named_before_they_are_cut_off(self):
        self.all_up()
        self.busy((1, 0))
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 0, out)
        self.assertIn("1 request(s) running or waiting will be cut off", out)
        self.assertEqual(self.left("head"), [], out)

    def test_waiting_requests_count_too(self):
        self.all_up()
        self.busy((0, 3))
        code, out = self.run_script(["stop.sh"])
        self.assertIn("3 request(s) running or waiting will be cut off", out)

    def test_a_busy_engine_is_drained_first(self):
        self.all_up()
        self.busy((2, 1), (1, 0), (0, 0))
        code, out = self.run_script(["stop.sh"], {"DRAIN_TIMEOUT": "30"})
        self.assertEqual(code, 0, out)
        self.assertIn("Drained", out)
        self.assertNotIn("cut off", out)
        for h in ["head", "w1", "w2", "w3"]:
            self.assertEqual(self.left(h), [], h)

    def test_the_drain_gives_up_at_its_timeout(self):
        self.all_up()
        self.busy((1, 0))
        import time
        t0 = time.monotonic()
        code, out = self.run_script(["stop.sh"], {"DRAIN_TIMEOUT": "2"})
        self.assertEqual(code, 0, out)
        self.assertLess(time.monotonic() - t0, 20, out)
        self.assertIn("1 request(s) running or waiting will be cut off", out)
        self.assertEqual(self.left("head"), [], out)

    def test_dry_run_neither_drains_nor_stops(self):
        self.all_up()
        self.busy((1, 0))
        import time
        t0 = time.monotonic()
        code, out = self.run_script(["stop.sh"], {"DRY_RUN": "1", "DRAIN_TIMEOUT": "30"})
        self.assertEqual(code, 0, out)
        self.assertLess(time.monotonic() - t0, 10, out)
        self.assertEqual(self.left("head"), [NAME], out)

    def api_down(self):
        p = self.tmp / "bin" / "curl"
        p.write_text("#!/bin/sh\nexit 7\n")
        p.chmod(0o755)

    def test_a_running_rank_0_whose_api_does_not_answer_is_still_stopped(self):
        # a start that failed after rank 0 launched leaves it running with no API: curl exits non-zero, and under
        # set -e + pipefail would end stop.sh in silence, rank 0 still running
        self.containers("head", NAME)
        self.api_down()
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.left("head"), [], out)
        self.assertIn("does not answer", out)
        self.assertIn("Stopped and removed rank 0", out)

    def test_every_rank_with_no_api_is_stopped_too(self):
        self.all_up()
        self.api_down()
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 0, out)
        for h in ["head", "w1", "w2", "w3"]:
            self.assertEqual(self.left(h), [], (h, out))

    def test_nothing_to_stop_exits_zero(self):
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 0, out)
        self.assertIn("nothing to stop", out)

    def test_worker_that_cannot_be_stopped_fails(self):
        self.all_up()
        self.stuck("w2")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("could not stop rank 2", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("w2"), [NAME])
        for h in ["head", "w1", "w3"]:          # the others are still stopped
            self.assertEqual(self.left(h), [], h)
        self.assertNoTraceback(out)

    def test_unreachable_worker_fails(self):
        self.all_up()
        self.unreachable("w3")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("rank 3 was left alone", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("w3"), [NAME])
        for h in ["head", "w1", "w2"]:
            self.assertEqual(self.left(h), [], h)
        self.assertNoTraceback(out)

    def test_unreachable_worker_is_not_nothing_to_stop(self):
        self.unreachable("w2")
        self.containers("w2", NAME)
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertNotIn("nothing to stop", out)
        self.assertIn("rank 2 was left alone", out)
        self.assertNoTraceback(out)

    def test_container_list_failing_on_a_worker_fails(self):
        self.all_up()
        self.ps_fails("w2")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("rank 2 was left alone", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("w2"), [NAME])
        for h in ["head", "w1", "w3"]:
            self.assertEqual(self.left(h), [], h)
        self.assertNoTraceback(out)

    def test_ssh_dropping_after_the_probe_fails(self):
        self.all_up()
        self.flaky("w1")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("rank 1 was left alone", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("w1"), [NAME])
        self.assertNoTraceback(out)

    def test_container_list_failing_here_fails(self):
        self.all_up()
        self.ps_fails("head")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertIn("rank 0 was left alone", out)
        self.assertNotIn("nothing to stop", out)
        self.assertNotIn("its GPU memory is free again", out)
        self.assertEqual(self.left("head"), [NAME])
        for h in ["w1", "w2", "w3"]:
            self.assertEqual(self.left(h), [], h)
        self.assertNoTraceback(out)

    def test_container_list_failing_here_alone_is_not_nothing_to_stop(self):
        self.containers("head", NAME)
        self.ps_fails("head")
        code, out = self.run_script(["stop.sh"])
        self.assertEqual(code, 1, out)
        self.assertNotIn("nothing to stop", out)
        self.assertEqual(self.left("head"), [NAME])

    def test_dry_run_stops_nothing(self):
        self.all_up()
        code, out = self.run_script(["stop.sh"], {"DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertIn("nothing is stopped", out)
        for h in ["head", "w1", "w2", "w3"]:
            self.assertEqual(self.left(h), [NAME], h)

    def test_dry_run_with_a_spark_it_cannot_list_fails(self):
        for case, setup in [("docker ps fails", lambda: self.ps_fails("w2")), ("unreachable", lambda: self.unreachable("w2"))]:
            with self.subTest(case=case):
                self.setUp()
                self.all_up()
                setup()
                code, out = self.run_script(["stop.sh"], {"DRY_RUN": "1"})
                self.assertEqual(code, 1, out)
                self.assertIn("nothing is stopped", out)
                for h in ["head", "w1", "w2", "w3"]:
                    self.assertEqual(self.left(h), [NAME], h)


FAKE_FINDMNT = r"""#!BASH
# fake findmnt -n -o FSTYPE|SOURCE -T <path>: the file system $SIM/fstype names (ext4 by default) on /dev/nvme0n1p2;
# with $SIM/findmnt_hangs it hangs (a disk that does not answer)
[[ ! -e "$SIM/findmnt_hangs" ]] || exec sleep 600
case "$*" in
  *FSTYPE*) cat "$SIM/fstype" 2>/dev/null || echo ext4 ;;
  *SOURCE*) echo /dev/nvme0n1p2 ;;
esac
"""

# fake df -B1 --output=avail <path>: $SIM/df_avail bytes free (1 TiB by default)
FAKE_DF = r"""#!BASH
echo Avail; cat "$SIM/df_avail" 2>/dev/null || echo 1099511627776
"""

# fake stat: with $SIM/root_owned, "%u %g" of anything is "0 0" (a folder that belongs to root); with $SIM/stat_fails
# that call fails; else the real stat
FAKE_STAT = r"""#!BASH
if [[ -e "$SIM/root_owned" && "$*" == *"%u %g"* ]]; then echo "0 0"; exit 0; fi
if [[ -e "$SIM/stat_fails" && "$*" == *"%u %g"* ]]; then echo "stat: cannot statx" >&2; exit 1; fi
exec REAL "$@"
"""


class StartRun(ScriptTest):
    """./start.sh with only what needs a Spark replaced: the links (detect_links), flock, ss, free, hostname, curl's
    answers (the server ready, the smoke test's reply) and scripts/prepare.sh (it records that it ran), and what the SSD
    tier's folder check reads from the disk (findmnt: ext4, df: 1 TiB free; stat as it is). Helpers only: the tests are
    in the classes below."""
    FAKES = {
        "flock": "#!/bin/sh\nexit 0\n",
        "ss": "#!/bin/sh\nexit 0\n",
        "free": "#!/bin/sh\necho 'Mem: 128 1 1 0 0 120'\n",
        "hostname": "#!/bin/sh\necho 10.0.0.1\n",
        # the smoke test's reply: a completion; with -w its HTTP status (200) on the last line, as curl prints it
        "curl": "#!/bin/sh\necho \"$*\" >> \"$SIM/curl.log\"\ncase \"$*\" in *completions*) printf '%s' '{\"choices\": [{\"message\": {\"role\": \"assistant\", \"content\": \"OK\"}}], \"usage\": {\"completion_tokens\": 1}}'\n"
                "  case \"$*\" in *http_code*) printf '\\n200' ;; esac ;; "
                "*) echo '{\"requests_running\": 0}' ;; esac\n",
    }

    def setUp(self):
        super().setUp()
        bin_ = self.tmp / "bin"
        for name, body in self.FAKES.items():
            (bin_ / name).write_text(body)
            (bin_ / name).chmod(0o755)
        for name, body in (("findmnt", FAKE_FINDMNT), ("df", FAKE_DF), ("stat", FAKE_STAT)):
            (bin_ / name).write_text(body.replace("BASH", BASH).replace("REAL", shutil.which("stat")))
            (bin_ / name).chmod(0o755)
        nodes = self.repo / "scripts" / "nodes.sh"
        nodes.write_text(nodes.read_text() + """
detect_links() {
  MASTER_ADDR=127.0.0.1
  for r in 0 1 2 3; do NODE_HCAS[r]=fake; NODE_GID[r]=0; LINK_HEAD_ADDR[r]=127.0.0.1; done
}
node_order() { echo 0,1,2,3; }
""")
        (self.repo / "scripts" / "prepare.sh").write_text("#!/usr/bin/env bash\necho prepared >> \"$SIM/events\"\n")
        (self.sim / "image").write_text("")

    def launched(self):
        log = (self.sim / "docker.log").read_text() if (self.sim / "docker.log").exists() else ""
        return [l.split()[0] for l in log.splitlines() if " docker run -d " in l]

    def prepared(self):
        return (self.sim / "events").exists()

    def run_lines(self, host):
        log = (self.sim / "docker.log").read_text()
        return [l for l in log.splitlines() if l.startswith(f"{host} docker run -d ")]


class ServedConfig(ScriptTest):
    """scripts/local.sh made from scripts/local.sh.example loads scripts/served.sh (every setting this fork serves with)
    first: the site's own lines after it win, and the environment wins over both."""
    def write_local(self, extra=""):
        example = (self.repo / "scripts" / "local.sh.example").read_text()
        (self.repo / "scripts" / "local.sh").write_text(example + extra)

    def last(self, out):
        return out.strip().splitlines()[-1]

    def test_the_example_loads_every_served_setting(self):
        self.write_local()
        code, out = self.bash('echo "$CONTEXT $POOL $PARALLEL $PORT $SERVED_NAME $ABLIT $GPU_CLOCK ${TF_DS_OWNER_GATHER-unset} ${TF_DS_RDMA_WFE:-} ${TF_DS_CUDA_SCHED:-}"')
        self.assertEqual(code, 0, out)
        # TF_DS_OWNER_GATHER is a retired switch: the engine refuses a start that sets it, so served.sh leaves it unset
        self.assertEqual(self.last(out), "1048576 16777216 16 8000 deepseek-v4.1-flash-1017 0 2200 unset 1 blocking")

    def test_the_base_image_is_pinned_by_its_digest_in_both_places(self):
        # a tag can be moved to another build; the digest names exactly the image every published number was built on
        (self.repo / "scripts" / "local.sh").write_text("")
        code, out = self.bash('echo "$BASE_IMAGE"')
        self.assertEqual(code, 0, out)
        base = self.last(out)
        self.assertRegex(base, r"^nvcr\.io/nvidia/pytorch:26\.07-py3@sha256:[0-9a-f]{64}$")
        self.assertIn(f"ARG BASE_IMAGE={base}\n", (ROOT / "zig" / "Dockerfile").read_text())

    def test_the_model_is_served_under_its_own_name_with_or_without_served_sh(self):
        # clients see which model answers ("model" was this site's gateway name, not a name for anyone else)
        (self.repo / "scripts" / "local.sh").write_text("")
        code, out = self.bash('echo "$SERVED_NAME"')
        self.assertEqual((code, self.last(out)), (0, "deepseek-v4.1-flash-1017"), out)

    def test_the_readme_and_the_bench_tool_use_the_served_name(self):
        name = re.search(r"(?m)^SERVED_NAME=(\S+)", (self.repo / "scripts" / "served.sh").read_text()).group(1)
        readme = (ROOT / "README.md").read_text()
        self.assertNotIn("--model model", readme)
        self.assertNotIn('"model": "model"', readme)
        self.assertIn(f"--model {name}", readme)
        self.assertIn(f'"model": "{name}"', readme)
        self.assertRegex((ROOT / "tools" / "bench.py").read_text(), r'add_argument\("--model", default="' + re.escape(name) + '"')
        for tool in ("multi_needle.py", "needle_bench.py"):
            self.assertIn(f'MODEL = arg("--model", "{name}")', (ROOT / "tools" / tool).read_text(), tool)

    def test_served_sh_sets_no_retired_switch(self):
        table = ROOT / "zig" / "src" / "core" / "settings_table.zig"
        self.assertTrue(table.exists(), table)
        retired = re.findall(r'\.env = "(TF_[A-Z0-9_]+)", \.when = \.set', table.read_text())
        self.assertIn("TF_DS_OWNER_GATHER", retired)
        served = (self.repo / "scripts" / "served.sh").read_text()
        for name in retired:
            self.assertNotRegex(served, r"(?m)^\s*(export\s+)?" + name + r"=", name)

    def test_the_abliteration_is_the_sites_choice_and_off_by_default(self):
        served = (self.repo / "scripts" / "served.sh").read_text()
        self.assertNotRegex(served, r"(?m)^\s*ABLIT=", "served.sh must not choose the abliteration for the site")
        example = (self.repo / "scripts" / "local.sh.example").read_text()
        self.assertRegex(example, r"(?m)^ABLIT=0\b", "the example offers the choice as a line of its own, set to 0")
        self.write_local()
        code, out = self.bash('echo "$ABLIT"')
        self.assertEqual((code, self.last(out)), (0, "0"), out)
        self.write_local("\nABLIT=1\n")
        code, out = self.bash('echo "$ABLIT"')
        self.assertEqual((code, self.last(out)), (0, "1"), out)

    def test_the_sites_own_lines_win(self):
        self.write_local("\nPORT=9100\nexport TF_DS_RDMA_WFE=0\n")
        code, out = self.bash('echo "$PORT ${TF_DS_RDMA_WFE:-}"')
        self.assertEqual(code, 0, out)
        self.assertEqual(self.last(out), "9100 0")

    def test_the_environment_wins_over_both(self):
        self.write_local("\nPORT=9100\n")
        code, out = self.bash('echo "$PORT"', {"PORT": "9200"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.last(out), "9200")


class MasterOnTheFabric(ScriptTest):
    """The ranks' rendezvous (MASTER_ADDR) defaults to the fabric's rail A when this Spark holds rail A's first address
    (./setup.sh's network step puts it there), so the ranks' start-up exchange stays off the LAN;
    a value the site sets wins; without that address the rendezvous is what this Spark's name resolves to, as before."""
    def setUp(self):
        super().setUp()
        real = shutil.which("ip") or "/usr/sbin/ip"
        fake = self.tmp / "bin" / "ip"
        fake.write_text(f"""#!/bin/sh
if [ "$*" = "-o -4 addr show dev ${{FAKE_RAIL_DEV:-none}}" ] && [ -n "${{FAKE_RAIL_ADDR:-}}" ]; then
  echo "4: $FAKE_RAIL_DEV    inet $FAKE_RAIL_ADDR/24 scope global $FAKE_RAIL_DEV"; exit 0
fi
case "$*" in *"addr show dev"*) exit 1 ;; esac
exec {real} "$@"
""")
        fake.chmod(0o755)

    def last(self, out):
        return out.strip().splitlines()[-1]

    def test_rail_a_is_the_rendezvous_when_this_spark_holds_it(self):
        code, out = self.bash('echo "$MASTER_ADDR"',
                              {"FAKE_RAIL_DEV": "enp1s0f0np0", "FAKE_RAIL_ADDR": "10.255.0.1"})
        self.assertEqual((code, self.last(out)), (0, "10.255.0.1"), out)

    def test_the_sites_own_values_win(self):
        code, out = self.bash('echo "$MASTER_ADDR"',
                              {"FAKE_RAIL_DEV": "enp1s0f0np0", "FAKE_RAIL_ADDR": "10.255.0.1", "MASTER_ADDR": "192.0.2.9"})
        self.assertEqual((code, self.last(out)), (0, "192.0.2.9"), out)

    def test_without_rail_a_the_rendezvous_is_as_before(self):
        code, out = self.bash('echo "${MASTER_ADDR}|"')
        self.assertEqual(code, 0, out)
        addr = self.last(out).split("|")[0]
        self.assertNotEqual(addr, "10.255.0.1")
        self.assertTrue(addr, "the hostname's address, as before")


# a Spark with only the setup's narrow sudo rule (no_sudo + narrow): sudo runs /usr/local/sbin/dsv41-root and nothing else
NARROW_HELPER_FAKE = ('if [ "$1" = /usr/local/sbin/dsv41-root ]; then [ -e "$SIM/hosts/${{SIM_HOST:-head}}/narrow" ] || exit 1\n'
                      '  echo "${{SIM_HOST:-head}} helper $2 $3" >> "$SIM/{log}"; exit 0; fi\n')


class GpuClock(StartRun):
    """GPU_CLOCK (MHz; the served numbers ran at 2200): before any rank starts, start.sh limits every Spark's GPU clock
    the way the served setup does (sudo -n nvidia-smi -lgc 0,<MHz>); unset leaves the clocks alone; DRY_RUN prints it;
    a Spark that refuses (no passwordless sudo) is named and the start goes on."""
    def setUp(self):
        super().setUp()
        bin_ = self.tmp / "bin"
        for name, body in (("sudo", '#!/bin/sh\n[ "$1" = -n ] && shift\n' + NARROW_HELPER_FAKE.format(log="clock.log") +
                                    '[ -e "$SIM/hosts/${SIM_HOST:-head}/no_sudo" ] && exit 1\nexec "$@"\n'),
                           ("nvidia-smi", '#!/bin/sh\necho "${SIM_HOST:-head} nvidia-smi $*" >> "$SIM/clock.log"\n')):
            (bin_ / name).write_text(body)
            (bin_ / name).chmod(0o755)

    def clock(self):
        f = self.sim / "clock.log"
        return sorted(f.read_text().splitlines()) if f.exists() else []

    def test_the_limit_reaches_every_spark_before_the_ranks(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "GPU_CLOCK": "2200"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.clock(), sorted(f"{h} nvidia-smi -lgc 0,2200" for h in ["head", "w1", "w2", "w3"]))
        self.assertEqual(sorted(self.launched()), ["head", "w1", "w2", "w3"], out)

    def test_the_context_is_65536_times_the_yarn_factor(self):
        # 1,048,576 at the checkpoint's factor 16 (unset), 2,097,152 only with factor 32 named
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "DRY_RUN": "1", "CONTEXT": "2097152"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("CONTEXT is at most 1048576", out)
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "DRY_RUN": "1", "CONTEXT": "2097152", "POOL": "16777216", "TF_DS_YARN_FACTOR": "32"})
        self.assertEqual(code, 0, out)
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "DRY_RUN": "1", "TF_DS_YARN_FACTOR": "x"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("TF_DS_YARN_FACTOR is a whole number", out)

    def test_unset_leaves_the_clocks_alone(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.clock(), [])

    def test_dry_run_prints_the_limit_and_sets_nothing(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "GPU_CLOCK": "2200", "DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.clock(), [])
        self.assertIn("nvidia-smi -lgc 0,2200", out)

    def test_a_spark_without_passwordless_sudo_is_named_and_the_start_goes_on(self):
        (self.sim / "hosts" / "w2").mkdir(parents=True, exist_ok=True)
        (self.sim / "hosts" / "w2" / "no_sudo").write_text("")
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "GPU_CLOCK": "2200"})
        self.assertEqual(code, 0, out)
        self.assertIn("could not limit w2's GPU clock", out)
        self.assertEqual(sorted(self.launched()), ["head", "w1", "w2", "w3"], out)
        self.assertIn("GPU clock limited to 2200 MHz on 3 of 4 Sparks (this one, w1, w3)", out)

    def test_a_spark_with_only_the_setups_rule_limits_through_its_helper(self):
        d = self.sim / "hosts" / "w1"; d.mkdir(parents=True, exist_ok=True)
        (d / "no_sudo").write_text(""); (d / "narrow").write_text("")
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "GPU_CLOCK": "2200"})
        self.assertEqual(code, 0, out)
        self.assertIn("w1 helper clock 2200", self.clock())
        self.assertIn("GPU clock limited to 2200 MHz on all 4 Sparks", out)

    def test_the_log_says_every_spark_got_the_limit(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "GPU_CLOCK": "2200"})
        self.assertEqual(code, 0, out)
        self.assertIn("GPU clock limited to 2200 MHz on all 4 Sparks", out)
        self.assertNotIn("limited to 2200 MHz here", out)

    def test_a_clock_that_is_not_a_number_is_refused_before_anything_starts(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "GPU_CLOCK": "fast"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("GPU_CLOCK", out)
        self.assertEqual(self.launched(), [])



class PageCacheDrop(StartRun):
    """Before the ranks start, start.sh drops every Spark's page cache once the old ranks are gone (on a GB10 the GPU's
    memory is the same RAM: a cache a copy or a weight-file bake left full leaves the engine that much less), as every
    published number's start did; DROP_CACHES=0 skips it; DRY_RUN prints it; a Spark that refuses (no passwordless
    sudo) is named and the start goes on."""
    def setUp(self):
        super().setUp()
        sudo = ('#!/bin/sh\n[ "$1" = -n ] && shift\n' + NARROW_HELPER_FAKE.format(log="docker.log") +
                '[ -e "$SIM/hosts/${SIM_HOST:-head}/no_sudo" ] && exit 1\n'
                'case "$*" in *drop_caches*) echo "${SIM_HOST:-head} drop_caches" >> "$SIM/docker.log"\n'
                '  case "$*" in *compact_memory*) echo "${SIM_HOST:-head} compact_memory" >> "$SIM/docker.log" ;; esac; exit 0 ;; esac\n'
                'exec "$@"\n')
        (self.tmp / "bin" / "sudo").write_text(sudo)
        (self.tmp / "bin" / "sudo").chmod(0o755)

    def events(self):
        f = self.sim / "docker.log"
        return f.read_text().splitlines() if f.exists() else []

    def test_every_spark_drops_its_cache_before_its_rank_starts(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        ev = self.events()
        for h in ["head", "w1", "w2", "w3"]:
            drops = [k for k, l in enumerate(ev) if l == f"{h} drop_caches"]
            runs = [k for k, l in enumerate(ev) if l.startswith(f"{h} docker run -d ")]
            self.assertEqual(len(drops), 1, (h, ev))
            self.assertTrue(runs and drops[0] < runs[0], (h, ev))
        self.assertIn("Page caches dropped", out)

    def test_every_spark_compacts_its_memory_after_the_drop(self):
        # an engine started on memory left in small pieces (a setup's data copy on the Spark just before) served every
        # prompt ~0.9 s late until a restart; the kernel's compaction gives the ranks the large blocks a restart frees
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        ev = self.events()
        for h in ["head", "w1", "w2", "w3"]:
            drops = [k for k, l in enumerate(ev) if l == f"{h} drop_caches"]
            packs = [k for k, l in enumerate(ev) if l == f"{h} compact_memory"]
            runs = [k for k, l in enumerate(ev) if l.startswith(f"{h} docker run -d ")]
            self.assertEqual(len(packs), 1, (h, ev))
            self.assertTrue(drops and runs and drops[0] < packs[0] < runs[0], (h, ev))
        self.assertIn("Page caches dropped and memory compacted", out)

    def test_dry_run_prints_the_compaction_after_the_drop(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertEqual([l for l in self.events() if "compact_memory" in l], [])
        self.assertIn("compact_memory", out)
        self.assertLess(out.index("drop_caches"), out.index("compact_memory"), out)

    def test_drop_caches_0_leaves_them(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "DROP_CACHES": "0"})
        self.assertEqual(code, 0, out)
        self.assertEqual([l for l in self.events() if "drop_caches" in l or "compact_memory" in l], [])
        self.assertIn("DROP_CACHES=0", out)

    def test_dry_run_prints_it_and_drops_nothing(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertEqual([l for l in self.events() if "drop_caches" in l], [])
        self.assertIn("drop_caches", out)

    def test_a_spark_without_passwordless_sudo_is_named_and_the_start_goes_on(self):
        (self.sim / "hosts" / "w3").mkdir(parents=True, exist_ok=True)
        (self.sim / "hosts" / "w3" / "no_sudo").write_text("")
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertIn("could not drop w3's page cache", out)
        self.assertEqual(sorted(self.launched()), ["head", "w1", "w2", "w3"], out)

    def test_a_spark_with_only_the_setups_rule_drops_through_its_helper(self):
        d = self.sim / "hosts" / "w2"; d.mkdir(parents=True, exist_ok=True)
        (d / "no_sudo").write_text(""); (d / "narrow").write_text("")
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        ev = self.events()
        drops = [k for k, l in enumerate(ev) if l.startswith("w2 helper drop")]
        runs = [k for k, l in enumerate(ev) if l.startswith("w2 docker run -d ")]
        self.assertTrue(drops and runs and drops[0] < runs[0], ev)
        self.assertNotIn("could not drop w2", out)

    def test_a_value_other_than_0_or_1_is_refused_before_anything_starts(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "DROP_CACHES": "yes"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("DROP_CACHES", out)
        self.assertEqual(self.launched(), [])



ROW_CACHE_LINE = ("info: TF_DS_ENGRAM_PREFAULT=1: the Engram row cache's {want} MiB faulted in at the start in {ms} ms "
                  "(the kernel's populate; {got} MiB of it in 2 MiB pages)\n")


class RowCachePages(StartRun):
    """Once the engine answers, start.sh reads each rank's Engram row cache line: a Spark whose memory was in small pieces
    when its rank started gets fewer 2 MiB pages than the cache's size (measured: 316 of 2048 MiB, after 23 s, on the
    engine a setup started; it served every prompt ~0.9 s late until a restart). Such a Spark is named with what it got
    and the cure; a whole cache, or a log without the line, says nothing."""
    def log(self, host, got, want=2048, ms=45):
        d = self.sim / "hosts" / host
        d.mkdir(parents=True, exist_ok=True)
        (d / "engine.log").write_text("a log line\n" + ROW_CACHE_LINE.format(want=want, got=got, ms=ms) + "another line\n")

    def test_a_spark_short_of_large_pages_is_named_with_the_cure(self):
        self.log("head", 316, ms=23135)
        for h in ("w1", "w2", "w3"):
            self.log(h, 2048)
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertIn("this Spark (316 of 2048 MiB)", out)
        self.assertIn("./start.sh restart", out)
        for h in ("w1", "w2", "w3"):
            self.assertNotIn(f"{h} (", out)

    def test_a_worker_short_of_large_pages_is_named(self):
        for h in ("head", "w1", "w3"):
            self.log(h, 2048)
        self.log("w2", 1024)
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertIn("w2 (1024 of 2048 MiB)", out)
        self.assertNotIn("this Spark (", out)

    def test_whole_caches_say_nothing(self):
        for h in ("head", "w1", "w2", "w3"):
            self.log(h, 2048)
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertNotIn("2 MiB pages", out)
        self.assertNotIn("small pieces", out)

    def test_a_log_without_the_line_says_nothing(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertNotIn("small pieces", out)


class SetupStartsTwice(unittest.TestCase):
    """The setup's step 5 starts the engine, then restarts it (stop, page caches dropped, memory compacted, start): the
    double reset every measured start has had. The one engine a setup started once served every prompt ~0.9 s late; every
    restart after it served at full speed."""
    def test_step_five_starts_then_restarts(self):
        text = (ROOT / "setup.sh").read_text()
        step5 = text[text.index('SETUP_TAG="5/5 start"'):]
        calls = re.findall(r"(?m)^\s*\./start\.sh(?:\s+(\w+))?\s*$", step5)
        self.assertEqual(calls, ["", "restart"], calls)


class KernelKit(ScriptTest):
    """The default kernel kit is this repository's fork of BertholomusAI's kit, which carries the extra set (aot-extra/)
    TF_DS_ATTN_U, TF_DS_IDX_U and TF_DS_DRAFT_4WAY load; a kit without it (BertholomusAI's own, say) is named by
    prepare.sh with what it costs, and the start goes on."""
    def test_the_default_kit_is_this_repositorys_fork(self):
        code, out = self.bash('echo "$KIT_REPO $KIT_REF $KIT_SUB"')
        self.assertEqual(code, 0, out)
        self.assertEqual(out.strip().splitlines()[-1],
                         "https://github.com/rbb-dev/deepseek-v4.1-tensorfold-tp4-4xgb10.git "
                         "b88aca850cb2e473069a4cd5d225e7a4b188fe8b tensorfold-dsv41/kit-b88aca85")

    def test_kit_extra_tells_a_kit_with_the_extra_set_from_one_without(self):
        with_set = self.tmp / "with" / "kit" / "aot-extra"
        with_set.mkdir(parents=True)
        (with_set / "aot.json").write_text('{"kernels": []}\n')
        (self.tmp / "without" / "kit" / "aot").mkdir(parents=True)
        code, out = self.bash(f'kit_extra {self.tmp / "with"} && echo with-yes; kit_extra {self.tmp / "without"} || echo without-no')
        self.assertEqual(code, 0, out)
        self.assertIn("with-yes", out)
        self.assertIn("without-no", out)

    def test_prepare_names_a_kit_without_the_extra_set(self):
        text = (self.repo / "scripts" / "prepare.sh").read_text()
        self.assertIn('kit_extra "$KIT_DIR" ||', text)
        self.assertIn("TF_DS_ATTN_U, TF_DS_IDX_U and TF_DS_DRAFT_4WAY", text)



class WorkerNames(ScriptTest):
    """Every per-worker setting is numbered 1 to 3 (WORKER1, WORKER2, WORKER3; WORKER_HF_CACHE1..3, FABRIC_PEER1..3,
    NFS_SERVER1..3). The recipe's unnumbered names for the first worker (WORKER, WORKER_HF_CACHE, FABRIC_PEER,
    NFS_SERVER) still reach it, with a warning naming the numbered one; a numbered name wins over its old one."""
    def env_without_workers(self, **extra):
        env = {k: v for k, v in self.env.items() if not k.startswith("WORKER")}
        env.update({"WORKER2": "w2", "WORKER3": "w3"}, **extra)
        return env

    def run_bash(self, body, env):
        script = f"set -euo pipefail\ncd {self.repo}\nsource ./scripts/config.sh\nsource ./scripts/nodes.sh\n{body}\n"
        r = subprocess.run([BASH, "-c", script], cwd=self.repo, env=env, capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout + r.stderr

    def test_worker_one_is_worker1(self):
        code, out = self.run_bash('echo "[$(worker_host 1) $(worker_host 2) $(worker_host 3) $(wvar WORKER 1) $(wvar WORKER_HF_CACHE 1)]"',
                                  self.env_without_workers(WORKER1="w1", WORKER_HF_CACHE1="/w1/hf"))
        self.assertEqual(code, 0, out)
        self.assertIn("[w1 w2 w3 WORKER1 WORKER_HF_CACHE1]", out)
        self.assertNotIn("WARN", out)

    def test_the_recipes_unnumbered_names_still_reach_worker_one_with_a_warning(self):
        code, out = self.run_bash('echo "[$(worker_host 1) $(worker_hf_cache 1) $(wval FABRIC_PEER 1) $(wval NFS_SERVER 1)]"',
                                  self.env_without_workers(WORKER="w1", WORKER_HF_CACHE="/w1/hf", FABRIC_PEER="192.0.2.12", NFS_SERVER="192.0.2.11"))
        self.assertEqual(code, 0, out)
        self.assertIn("[w1 /w1/hf 192.0.2.12 192.0.2.11]", out)
        for old, new in (("WORKER", "WORKER1"), ("WORKER_HF_CACHE", "WORKER_HF_CACHE1"), ("FABRIC_PEER", "FABRIC_PEER1"), ("NFS_SERVER", "NFS_SERVER1")):
            self.assertIn(f"{old} is the recipe's name: this repository numbers the workers, so it is {new}", out)

    def test_a_numbered_name_wins_over_its_old_one(self):
        code, out = self.run_bash('echo "[$(worker_host 1)]"', self.env_without_workers(WORKER="old", WORKER1="new"))
        self.assertEqual(code, 0, out)
        self.assertIn("[new]", out)


class StartRestart(StartRun):
    def test_restart_starts_every_rank(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(self.launched()), ["head", "w1", "w2", "w3"], out)
        self.assertIn("is now LIVE", out)

    def test_under_the_setup_it_shows_a_few_plain_lines_and_logs_the_rest(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        logf = self.tmp / "setup.log"
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "SETUP_TAG": "5/5 start", "SETUP_LOG": str(logf)})
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(self.launched()), ["head", "w1", "w2", "w3"], out)
        shown = [l for l in out.splitlines() if l.strip()]
        self.assertTrue(shown and all(l.startswith("[5/5 start]") for l in shown), out)
        for want in ("engine started on all four Sparks", "answering after", "a first request through all four Sparks: OK"):
            self.assertIn(want, out)
        for gone in ("is now LIVE", "[1/5]", "TensorFold by Ash Hart", "Rank 0 here", "rank 1"):
            self.assertNotIn(gone, out)
        kept = logf.read_text()
        for want in ("[1/5] Setup", "Rank 0 here", "OK: "):
            self.assertIn(want, kept)

    def test_restart_stops_when_a_rank_cannot_be_stopped(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        self.stuck("w2")
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("could not stop rank 2", out)
        self.assertEqual(self.launched(), [], out)
        self.assertNotIn("is now LIVE", out)

    def test_restart_stops_when_a_worker_cannot_be_checked(self):
        for h in ["head", "w1", "w2", "w3"]:
            self.containers(h, NAME)
        self.ps_fails("w3")
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertNotEqual(code, 0, out)
        self.assertEqual(self.launched(), [], out)

    def test_template_that_cannot_be_hashed_starts_nothing(self):
        # no marker yet, or an empty one: a failed hash must not read as "Ready" and start the ranks
        (self.repo / "model" / "chat_template.jinja").unlink()
        state = self.home / ".local/state/dsv41-tensorfold"
        for marker in [None, ""]:
            with self.subTest(marker=marker):
                if marker is not None:
                    state.mkdir(parents=True, exist_ok=True)
                    (state / "prepared").write_text(marker)
                code, out = self.run_script(["start.sh"])
                self.assertNotEqual(code, 0, out)
                self.assertNotIn("Ready:", out)
                self.assertIn("chat_template", out)
                self.assertEqual(self.launched(), [], out)
                self.assertFalse(self.prepared(), out)

    def test_a_worker_with_its_own_copy_binds_it_instead_of_nfs(self):
        local = {"WORKER_HF_CACHE1": "/w1/hf", "WORKER_HF_CACHE2": "/w2/hf", "WORKER_HF_CACHE3": "/w3/hf"}
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", **local})
        self.assertEqual(code, 0, out)
        for i in (1, 2, 3):
            (line,) = self.run_lines(f"w{i}")
            self.assertIn(f"-v /w{i}/hf:/hf:ro ", line)
            self.assertNotIn("dsv41-hf:/hf", line)

    def test_without_one_the_worker_reads_over_nfs(self):
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "WORKER_HF_CACHE2": "/w2/hf"})
        self.assertEqual(code, 0, out)
        self.assertIn("-v dsv41-hf:/hf:ro ", self.run_lines("w1")[0])
        self.assertIn("-v /w2/hf:/hf:ro ", self.run_lines("w2")[0])

    def key_file(self, mode=0o600):
        kf = self.tmp / "api-keys"
        kf.write_text("# keys\ntools: k-one\ngateway: k-two\n")
        kf.chmod(mode)
        return kf

    def test_a_key_file_goes_to_rank_0_and_the_start_sends_its_first_key(self):
        kf = self.key_file()
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "API_KEY_FILE": str(kf)})
        self.assertEqual(code, 0, out)
        (head,) = self.run_lines("head")
        self.assertIn(f"-v {kf}:/run/tensorfold/api-keys:ro", head)
        self.assertIn("--api-key-file /run/tensorfold/api-keys", head)
        self.assertIn("--metrics-open", head)
        for i in (1, 2, 3):
            self.assertNotIn("api-keys", self.run_lines(f"w{i}")[0])
        curls = (self.sim / "curl.log").read_text()
        for path in ("/v1/models", "/v1/chat/completions"):
            calls = [l for l in curls.splitlines() if path in l]
            self.assertTrue(calls and all("-H @" in l for l in calls), (path, calls))
        self.assertNotIn("k-one", curls)                     # the key never on a command line
        hdr = self.home / ".local/state/dsv41-tensorfold/api-header"
        self.assertEqual(hdr.read_text(), "Authorization: Bearer k-one\n")
        self.assertEqual(hdr.stat().st_mode & 0o077, 0)

    def test_an_admin_key_file_goes_to_rank_0_alone(self):
        ak = self.tmp / "admin-keys"
        ak.write_text("admin: " + "a" * 40 + "\n")
        ak.chmod(0o600)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ADMIN_KEY_FILE": str(ak)})
        self.assertEqual(code, 0, out)
        (head,) = self.run_lines("head")
        self.assertIn(f"-v {ak}:/run/tensorfold/admin-keys:ro", head)
        self.assertIn("-e TENSORFOLD_ADMIN_KEY_FILE=/run/tensorfold/admin-keys", head)
        for i in (1, 2, 3):
            self.assertNotIn("admin-keys", self.run_lines(f"w{i}")[0])
        self.assertNotIn("a" * 40, out)

    def test_an_admin_key_file_others_can_read_is_refused(self):
        ak = self.tmp / "admin-keys"
        ak.write_text("admin: " + "a" * 40 + "\n")
        ak.chmod(0o644)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ADMIN_KEY_FILE": str(ak)})
        self.assertNotEqual(code, 0, out)
        self.assertIn("chmod 600", out)
        self.assertEqual(self.launched(), [], out)

    def test_an_engine_cfg_reaches_every_rank_and_rank_0_writes_its_folder(self):
        d = self.tmp / "cfgdir"; d.mkdir()
        cf = d / "engine.cfg"
        cf.write_text("# test\ntf_ds_issue_lead = 4\n")
        cf.chmod(0o600)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ENGINE_CFG": str(cf)})
        self.assertEqual(code, 0, out)
        self.assertNotIn("tf_ds_issue_lead", out)            # the contents never on a command line
        (head,) = self.run_lines("head")
        self.assertIn(f"-v {d}:/run/tensorfold/cfg ", head)
        self.assertNotIn(f"-v {d}:/run/tensorfold/cfg:ro", head)
        self.assertIn("-e TF_ENGINE_CFG=/run/tensorfold/cfg/engine.cfg", head)
        for i in (1, 2, 3):
            (w,) = self.run_lines(f"w{i}")
            self.assertIn(":/run/tensorfold/cfg:ro", w)
            self.assertIn("-e TF_ENGINE_CFG=/run/tensorfold/cfg/engine.cfg", w)

    def test_an_engine_cfg_others_can_read_or_beside_other_files_is_refused(self):
        d = self.tmp / "cfgdir2"; d.mkdir()
        cf = d / "engine.cfg"; cf.write_text("x = 1\n"); cf.chmod(0o644)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ENGINE_CFG": str(cf)})
        self.assertNotEqual(code, 0, out); self.assertIn("chmod 600", out)
        cf.chmod(0o600); (d / "api-keys").write_text("k\n")
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ENGINE_CFG": str(cf)})
        self.assertNotEqual(code, 0, out); self.assertIn("folder of its own", out)
        self.assertEqual(self.launched(), [], out)

    def test_an_engine_cfg_link_or_an_open_folder_is_refused(self):
        d = self.tmp / "cfgdir3"; d.mkdir(mode=0o700)
        real = d / "real.cfg"; real.write_text("x = 1\n"); real.chmod(0o600)
        link = d / "engine.cfg"; link.symlink_to(real)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ENGINE_CFG": str(link)})
        self.assertNotEqual(code, 0, out); self.assertIn("symbolic link", out)
        e = self.tmp / "cfgdir4"; e.mkdir(); e.chmod(0o777)
        cf = e / "engine.cfg"; cf.write_text("x = 1\n"); cf.chmod(0o600)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ENGINE_CFG": str(cf)})
        self.assertNotEqual(code, 0, out); self.assertIn("writable by you alone", out)
        self.assertEqual(self.launched(), [], out)

    def test_a_free_memory_target_replaces_the_fixed_pool_on_every_rank(self):
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "TF_DS_KV_FREE_TARGET_GIB": "14"})
        self.assertEqual(code, 0, out)
        (head,) = self.run_lines("head")
        self.assertNotIn("TF_DS_WINDOW=", head)
        self.assertIn("TF_DS_KV_FREE_TARGET_GIB=14", head)
        for i in (1, 2, 3):
            (w,) = self.run_lines(f"w{i}")
            self.assertNotIn("--pool ", w)
            self.assertNotIn("TF_DS_WINDOW=", w)
            self.assertIn("TF_DS_KV_FREE_TARGET_GIB=14", w)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "TF_DS_KV_FREE_TARGET_GIB": "lots"})
        self.assertNotEqual(code, 0, out); self.assertIn("TF_DS_KV_FREE_TARGET_GIB", out)

    def test_an_engine_cfg_that_is_missing_is_refused(self):
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ENGINE_CFG": str(self.tmp / "nope.cfg")})
        self.assertNotEqual(code, 0, out)
        self.assertIn("ENGINE_CFG", out)
        self.assertEqual(self.launched(), [], out)

    def test_an_admin_policy_file_goes_to_rank_0_alone(self):
        pf = self.tmp / "admin-policy.json"
        pf.write_text('{"request_caps": {}}\n'); pf.chmod(0o600)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ADMIN_POLICY_FILE": str(pf)})
        self.assertEqual(code, 0, out)
        (head,) = self.run_lines("head")
        self.assertIn(f"-v {pf}:/run/tensorfold/admin-policy.json:ro", head)
        self.assertIn("-e TENSORFOLD_ADMIN_POLICY=/run/tensorfold/admin-policy.json", head)
        for i in (1, 2, 3):
            self.assertNotIn("admin-policy", self.run_lines(f"w{i}")[0])
        pf.chmod(0o646)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ADMIN_POLICY_FILE": str(pf)})
        self.assertNotEqual(code, 0, out); self.assertIn("ADMIN_POLICY_FILE", out)

    def test_a_key_file_others_can_read_is_refused(self):
        kf = self.key_file(0o644)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "API_KEY_FILE": str(kf)})
        self.assertNotEqual(code, 0, out)
        self.assertIn("chmod 600", out)
        self.assertEqual(self.launched(), [], out)

    def test_metrics_open_0_keeps_metrics_behind_the_key(self):
        kf = self.key_file()
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "API_KEY_FILE": str(kf), "METRICS_OPEN": "0"})
        self.assertEqual(code, 0, out)
        (head,) = self.run_lines("head")
        self.assertIn("--api-key-file", head)
        self.assertNotIn("--metrics-open", head)

    def test_without_a_key_file_the_api_stays_open(self):
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        self.assertNotIn("--api-key-file", self.run_lines("head")[0])
        self.assertNotIn("-H @", (self.sim / "curl.log").read_text())

    def test_every_worker_command_reuses_one_ssh_connection(self):
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        opts = [l for l in (self.sim / "ssh-opts.log").read_text().splitlines() if l.strip()]
        self.assertTrue(opts)
        for l in opts:
            self.assertIn("ControlMaster=auto", l)
            self.assertIn("ControlPersist=", l)

    def test_a_rank_that_cannot_start_stops_the_start(self):
        d = self.sim / "hosts" / "w2"
        d.mkdir(parents=True, exist_ok=True)
        (d / "run_fails").write_text("")
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("could not start rank 2", out)
        self.assertNotIn("head", self.launched(), out)
        self.assertNotIn("is now LIVE", out)

    def first_name_docker(self):
        # the fake `docker run` records the last --name (the served name inside rank 0's command); these tests need the container's own
        dk = self.tmp / "bin" / "docker"
        dk.write_text(dk.read_text().replace('[[ "$prev" != --name ]] || n=$a', '[[ "$prev" != --name || -n "$n" ]] || n=$a'))

    def api_down(self):
        p = self.tmp / "bin" / "curl"
        p.write_text("#!/bin/sh\nexit 7\n")
        p.chmod(0o755)

    def test_a_follower_that_exits_after_rank_0_started_takes_every_container_down(self):
        self.first_name_docker()
        # PoolBelowContext: the followers refused at load, start.sh exited, rank 0 kept running (75 GB)
        (self.sim / "hosts" / "w1").mkdir(parents=True, exist_ok=True)
        (self.sim / "hosts" / "w1" / "exited").write_text("")
        self.api_down()
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("rank 1", out)
        self.assertIn("head", self.launched(), out)               # rank 0 had started
        for h in ("head", "w1", "w2", "w3"):
            self.assertEqual(self.left(h), [], (h, out))
        self.assertIn("stopping every container", out)

    def test_a_worker_that_cannot_start_takes_the_ones_already_started_down(self):
        self.first_name_docker()
        (self.sim / "hosts" / "w2").mkdir(parents=True, exist_ok=True)
        (self.sim / "hosts" / "w2" / "run_fails").write_text("")
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("could not start rank 2", out)
        for h in ("head", "w1", "w2", "w3"):
            self.assertEqual(self.left(h), [], (h, out))

    def test_a_start_that_succeeds_leaves_every_container_running(self):
        self.first_name_docker()
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0"})
        self.assertEqual(code, 0, out)
        for h in ("head", "w1", "w2", "w3"):
            self.assertEqual(self.left(h), [NAME], (h, out))

    def test_old_marker_runs_prepare_once(self):
        state = self.home / ".local/state/dsv41-tensorfold"
        state.mkdir(parents=True)
        (state / "prepared").write_text("model=old image=old kit=old\n")
        code, out = self.run_script(["start.sh"])
        self.assertEqual(code, 0, out)
        self.assertTrue(self.prepared(), out)
        self.assertIn("Not ready yet", out)


class ReplayFloor(StartRun):
    """REPLAY_FLOOR (bertholomus's replay floor, TF_DS_REPLAY_FLOOR): one recipe setting, the same value to all four
    ranks (the ranks' start-up settings exchange refuses a difference), 0 by default; refused unless 0 or 1."""
    ALL = ("head", "w1", "w2", "w3")

    def floors(self, extra):
        for h in self.ALL:
            self.containers(h, NAME)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", **extra})
        self.assertEqual(code, 0, out)
        lines = {h: self.run_lines(h) for h in self.ALL}
        for h, ls in lines.items():
            self.assertEqual(len(ls), 1, (h, ls))
            self.assertEqual(ls[0].count("TF_DS_REPLAY_FLOOR="), 1, ls[0])
        return {h: ls[0].split("TF_DS_REPLAY_FLOOR=")[1].split()[0] for h, ls in lines.items()}

    def test_off_by_default_on_every_rank(self):
        self.assertEqual(self.floors({}), dict.fromkeys(self.ALL, "0"))

    def test_on_reaches_every_rank_alike(self):
        self.assertEqual(self.floors({"REPLAY_FLOOR": "1"}), dict.fromkeys(self.ALL, "1"))

    def test_the_recipe_setting_wins_over_an_exported_engine_name(self):
        # one value a rank, the recipe's: an exported TF_DS_REPLAY_FLOOR does not pass through beside it
        self.assertEqual(self.floors({"REPLAY_FLOOR": "1", "TF_DS_REPLAY_FLOOR": "0"}), dict.fromkeys(self.ALL, "1"))

    def test_an_exported_engine_name_alone_is_the_setting(self):
        self.assertEqual(self.floors({"TF_DS_REPLAY_FLOOR": "1"}), dict.fromkeys(self.ALL, "1"))

    def test_anything_but_0_or_1_is_refused_before_anything_is_stopped(self):
        for bad in ("2", "on", "-1"):
            with self.subTest(REPLAY_FLOOR=bad):
                self.setUp()
                for h in self.ALL:
                    self.containers(h, NAME)
                code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "REPLAY_FLOOR": bad})
                self.assertNotEqual(code, 0, out)
                self.assertIn("REPLAY_FLOOR is 0 or 1", out)
                self.assertEqual(self.launched(), [], out)
                for h in self.ALL:
                    self.assertEqual(self.left(h), [NAME], (h, out))


def run_env(line):
    """The -e NAME=value words of a docker run line, as {NAME: [values]}."""
    words = line.split()
    env = {}
    for a, b in zip(words, words[1:]):
        if a == "-e":
            k, _, v = b.partition("=")
            env.setdefault(k, []).append(v)
    return env


def tier_names(env):
    return sorted(k for k in env if k.startswith(("TF_DS_SPILL_", "TF_SPILL_")))


class SpillSettings(StartRun):
    """The SSD tier's settings (scripts/config.sh SPILL_*) reach every rank, by the engine's names (TF_DS_SPILL_*,
    TF_SPILL_*): each Spark keeps a store of its own, so a worker gets rank 0's whole set as it is, the store's identity,
    its own weight digest, the kit and the token map, and its own folder mounted at /spill. With the tier off a worker
    gets TF_DS_SPILL_GIB=0 alone (the ranks' NCCL decision) and mounts nothing. Every other TF_DS_SPILL_* / TF_SPILL_*
    exported passes through to every rank as it is; a --spill-* serve argument is refused before anything is stopped; a bad value
    is passed on for the engine to name (the tier never refuses a start). The stop's bounds follow STOP_TIMEOUT."""
    ALL = ("head", "w1", "w2", "w3")
    WORKERS = ("w1", "w2", "w3")
    ON = {"SPILL_GIB": "256", "SPILL_FLOOR_GIB": "50", "SPILL_TURNS": "4"}

    def start(self, extra=None, args=()):
        for h in self.ALL:
            self.containers(h, NAME)
        return self.run_script(["start.sh", "restart", *args], {"PREPARE": "0", **(extra or {})})

    def envs(self, extra=None, args=()):
        code, out = self.start(extra, args)
        self.assertEqual(code, 0, out)
        envs = {}
        for h in self.ALL:
            (line,) = self.run_lines(h)
            envs[h] = run_env(line)
        return envs, out

    def assertNoTier(self, envs, hosts):
        for h in hosts:
            self.assertEqual(tier_names(envs[h]), [], h)

    def assertWorkersHoldTheTier(self, envs):
        """With the tier on, a worker holds every tier name rank 0 holds, each once and with rank 0's value (the ranks' cap, quota,
        shortest point, turns and image cache must be equal or they refuse each other), the store's identity as it is, the
        kit and the token map the store's name is made from, and its own weight digest."""
        head = envs["head"]
        for h in self.WORKERS:
            w = envs[h]
            for k in tier_names(head):
                self.assertEqual(w.get(k), head[k], (h, k))
            self.assertEqual(w["TF_DS_SPILL_DIR"], ["/spill"], h)
            for k in ("TF_DS_ENGINE_ID", "TF_DS_MODEL_REVISION", "TF_DS_ABLATION_ID", "TF_DS_ENGRAM_REVISION", "TF_DS_KIT_ID"):
                self.assertEqual(w.get(k), head[k], (h, k))
            self.assertEqual(len(w["TF_DS_KIT"]), 1, h)
            self.assertEqual(len(w["TF_DS_TOKEN_MAP"]), 1, h)
            self.assertEqual(len(w["TF_DS_RANK_CACHE_SHA"]), 1, h)          # once: its own digest, not rank 0's
            self.assertNotIn("TF_DS_SSD_EVERY_RANK", w, h)

    def assertWorkersGibOnly(self, envs):
        """With the tier off a follower holds TF_DS_SPILL_GIB alone, the value rank 0 has: the ranks decide together from it whether NCCL still has
        a landing to carry (a tier that is off has none), so one rank without it cannot start beside the others."""
        for h in self.WORKERS:
            self.assertEqual(tier_names(envs[h]), ["TF_DS_SPILL_GIB"], h)
            self.assertEqual(envs[h]["TF_DS_SPILL_GIB"], envs["head"]["TF_DS_SPILL_GIB"], h)

    def test_off_by_default_rank_0_gets_the_switch_alone(self):
        envs, out = self.envs()
        self.assertEqual(tier_names(envs["head"]), ["TF_DS_SPILL_GIB"])
        self.assertEqual(envs["head"]["TF_DS_SPILL_GIB"], ["0"])
        self.assertWorkersGibOnly(envs)

    def test_on_every_rank_gets_every_setting_by_the_engines_names(self):
        envs, out = self.envs(self.ON)
        head = envs["head"]
        want = {"TF_DS_SPILL_GIB": "256", "TF_DS_SPILL_MIN_FREE_GIB": "50", "TF_DS_SPILL_DIR": "/spill",
                "TF_SPILL_TURNS": "4", "TF_SPILL_FLUSH_MS": "15000", "TF_DS_SPILL_QUOTA": "0",
                "TF_DS_SPILL_MIN_TOKENS": "8192", "TF_DS_SPILL_STAGING_MIB": "128"}
        for k, v in want.items():
            self.assertEqual(head.get(k), [v], k)
        self.assertWorkersHoldTheTier(envs)

    def test_an_exported_tier_name_passes_to_every_rank_once(self):
        envs, out = self.envs({**self.ON, "TF_SPILL_LOAD_MS": "40", "TF_DS_SPILL_RESTORES": "2"})
        self.assertEqual(envs["head"].get("TF_SPILL_LOAD_MS"), ["40"])
        self.assertEqual(envs["head"].get("TF_DS_SPILL_RESTORES"), ["2"])
        for h in self.WORKERS:
            self.assertEqual(envs[h].get("TF_SPILL_LOAD_MS"), ["40"], h)
            self.assertEqual(envs[h].get("TF_DS_SPILL_RESTORES"), ["2"], h)
        self.assertWorkersHoldTheTier(envs)

    def test_an_exported_engine_name_is_the_settings_value_when_the_setting_is_not_set(self):
        envs, out = self.envs({"TF_DS_SPILL_GIB": "64", "TF_SPILL_TURNS": "2"})
        self.assertEqual(envs["head"]["TF_DS_SPILL_GIB"], ["64"])
        self.assertEqual(envs["head"]["TF_SPILL_TURNS"], ["2"])
        self.assertWorkersHoldTheTier(envs)

    def test_the_setting_wins_over_its_exported_engine_name_one_value(self):
        envs, out = self.envs({**self.ON, "TF_DS_SPILL_GIB": "64", "TF_SPILL_TURNS": "2"})
        self.assertEqual(envs["head"]["TF_DS_SPILL_GIB"], ["256"])
        self.assertEqual(envs["head"]["TF_SPILL_TURNS"], ["4"])
        self.assertIn("TF_DS_SPILL_GIB=64", out)            # said: the exported value is not used
        self.assertWorkersHoldTheTier(envs)
        for h in self.WORKERS:
            self.assertEqual(envs[h]["TF_DS_SPILL_GIB"], ["256"], h)

    def test_spill_gib_0_sends_no_other_tier_name(self):
        envs, out = self.envs({"SPILL_GIB": "0", "SPILL_TURNS": "4", "TF_SPILL_LOAD_MS": "40"})
        self.assertEqual(tier_names(envs["head"]), ["TF_DS_SPILL_GIB"])
        self.assertEqual(envs["head"]["TF_DS_SPILL_GIB"], ["0"])
        self.assertWorkersGibOnly(envs)

    def test_an_off_reason_exported_turns_the_tier_off_and_reaches_rank_0(self):
        envs, out = self.envs({**self.ON, "TF_DS_SPILL_OFF_REASON": "watchdog"})
        self.assertEqual(tier_names(envs["head"]), ["TF_DS_SPILL_GIB", "TF_DS_SPILL_OFF_REASON"])
        self.assertEqual(envs["head"]["TF_DS_SPILL_GIB"], ["0"])
        self.assertEqual(envs["head"]["TF_DS_SPILL_OFF_REASON"], ["watchdog"])
        self.assertWorkersGibOnly(envs)

    def test_a_bad_value_is_passed_on_for_the_engine_to_name(self):
        envs, out = self.envs({**self.ON, "SPILL_TURNS": "abc", "SPILL_QUOTA": "2"})
        self.assertEqual(envs["head"]["TF_SPILL_TURNS"], ["abc"])
        self.assertEqual(envs["head"]["TF_DS_SPILL_QUOTA"], ["2"])
        self.assertIn("is now LIVE", out)

    def test_spill_serve_arguments_are_refused_before_anything_is_stopped(self):
        for args in (["--spill-gib", "8"], ["--spill-dir=/x"], ["--metrics-open", "--spill-min-tokens", "1"]):
            with self.subTest(args=args):
                self.setUp()
                code, out = self.start(self.ON, args)
                self.assertNotEqual(code, 0, out)
                self.assertIn("--spill-", out)
                self.assertIn("SPILL_GIB", out)
                self.assertEqual(self.launched(), [], out)
                for h in self.ALL:
                    self.assertEqual(self.left(h), [NAME], (h, out))

    def test_the_stop_bounds_follow_stop_timeout_on_rank_0_alone(self):
        envs, out = self.envs(self.ON)
        self.assertEqual(envs["head"]["TENSORFOLD_CLOSE_MS"], ["20000"])      # STOP_TIMEOUT 30 x 2000 / 3
        self.assertEqual(envs["head"]["TF_SPILL_FLUSH_MS"], ["15000"])        # STOP_TIMEOUT 30 x 500
        for h in self.WORKERS:
            self.assertNotIn("TENSORFOLD_CLOSE_MS", envs[h], h)
        self.setUp()
        envs, out = self.envs({**self.ON, "STOP_TIMEOUT": "60"})
        self.assertEqual(envs["head"]["TENSORFOLD_CLOSE_MS"], ["40000"])
        self.assertEqual(envs["head"]["TF_SPILL_FLUSH_MS"], ["30000"])
        self.setUp()
        envs, out = self.envs({**self.ON, "TENSORFOLD_CLOSE_MS": "9000", "SPILL_FLUSH_MS": "8000"})
        self.assertEqual(envs["head"]["TENSORFOLD_CLOSE_MS"], ["9000"])
        self.assertEqual(envs["head"]["TF_SPILL_FLUSH_MS"], ["8000"])
        for h in self.WORKERS:
            self.assertNotIn("TENSORFOLD_CLOSE_MS", envs[h], h)

    def test_a_flush_that_does_not_fit_the_close_is_said_not_refused(self):
        envs, out = self.envs({**self.ON, "SPILL_FLUSH_MS": "25000"})
        self.assertIn("SPILL_FLUSH_MS", out)
        self.assertIn("TENSORFOLD_CLOSE_MS", out)
        self.assertEqual(envs["head"]["TF_SPILL_FLUSH_MS"], ["25000"])

    def test_stop_settings_that_are_not_numbers_are_refused_before_anything_is_stopped(self):
        for k, bad in (("STOP_TIMEOUT", "abc"), ("STOP_TIMEOUT", "0"), ("TENSORFOLD_CLOSE_MS", "20s")):
            with self.subTest(k=k, bad=bad):
                self.setUp()
                code, out = self.start({k: bad})
                self.assertNotEqual(code, 0, out)
                self.assertIn(k, out)
                self.assertEqual(self.launched(), [], out)
                for h in self.ALL:
                    self.assertEqual(self.left(h), [NAME], (h, out))


    # Each worker keeps a store of its own when the tier is asked for. It gets rank 0's tier settings and the store's identity as
    # they are (the ranks' cap, quota, shortest point, turns and image cache must be equal or they refuse each other), its own weight
    # digest, the kit and the token map the store's name is made from, and its own folder (the same path as this Spark's, made as
    # you) mounted at /spill. With the tier off a worker holds TF_DS_SPILL_GIB alone.

    def test_a_worker_gets_the_tier_settings_the_identity_the_kit_and_its_folder(self):
        envs, out = self.envs(self.ON)
        self.assertWorkersHoldTheTier(envs)
        # (each rank's own digest: WeightDigests)

    def test_a_worker_mounts_the_same_folder_as_you_made_before_docker_does(self):
        envs, out = self.envs(self.ON)
        (head,) = self.run_lines("head")
        folder = re.search(r"-v (\S+):/spill", head).group(1)
        for h in self.WORKERS:
            (line,) = self.run_lines(h)
            self.assertIn(f"-v {folder}:/spill", line, h)
        ssh = (self.sim / "ssh.log").read_text().splitlines()
        for h in self.WORKERS:
            made = [l for l in ssh if l.startswith(h + " ") and f"mkdir -p -m 0700 {folder} && docker run" in l]
            self.assertEqual(len(made), 1, (h, made))

    def test_with_the_tier_off_a_worker_mounts_nothing_and_holds_the_one_name(self):
        envs, out = self.envs({"SPILL_GIB": "0"})
        self.assertWorkersGibOnly(envs)
        for h in self.WORKERS:
            (line,) = self.run_lines(h)
            self.assertNotIn(":/spill", line, h)
        self.assertNotIn("-m 0700", (self.sim / "ssh.log").read_text())

    def test_a_tier_turned_off_by_a_failed_check_leaves_a_worker_the_one_name_with_zero(self):
        (self.sim / "fstype").write_text("tmpfs\n")
        envs, out = self.envs(self.ON)
        self.assertEqual(envs["head"]["TF_DS_SPILL_GIB"], ["0"])
        self.assertWorkersGibOnly(envs)
        for h in self.WORKERS:
            (line,) = self.run_lines(h)
            self.assertNotIn(":/spill", line, h)


class SpillFolder(StartRun):
    """The SSD tier's folder on this Spark (SPILL_DIR): created as you before anything is stopped, checked (a local
    disk, no overlap with another tier's folder, HF_CACHE or the recipe's own folders, not / or home, not root's, at
    least SPILL_FLOOR_GIB free) and mounted at /spill in every rank's container (each Spark keeps a store of its own). A check that fails, or does not end
    within SPILL_CHECK_TIMEOUT, starts the engine without the tier: TF_DS_SPILL_GIB=0 and TF_DS_SPILL_OFF_REASON naming
    it; never a refused start."""
    ALL = ("head", "w1", "w2", "w3")
    ON = {"SPILL_GIB": "256", "SPILL_FLOOR_GIB": "0"}

    def setUp(self):
        super().setUp()
        self.glm = self.home / ".cache" / "tensorfold-spill-blocks"
        self.default = self.home / ".cache" / "dsv41-tensorfold-spill"

    def start(self, extra=None, timeout=60):
        for h in self.ALL:
            self.containers(h, NAME)
        return self.run_script(["start.sh", "restart"], {"PREPARE": "0", **self.ON, **(extra or {})}, timeout=timeout)

    def head(self, extra=None):
        code, out = self.start(extra)
        self.assertEqual(code, 0, out)
        self.assertIn("is now LIVE", out)
        self.assertEqual(sorted(self.launched()), list(self.ALL), out)       # all four ranks launched
        (line,) = self.run_lines("head")
        folder = re.search(r"-v (\S+):/spill ", line)
        for h in self.ALL[1:]:
            (w,) = self.run_lines(h)
            if folder:        # the tier on: every Spark mounts its own folder, the same path as this Spark's, and holds the tier's settings
                self.assertIn(f"-v {folder.group(1)}:/spill ", w, h)
                for k in tier_names(run_env(line)):
                    self.assertEqual(run_env(w)[k], run_env(line)[k], (h, k))
            else:             # the tier off: nothing mounted, the NCCL decision's one name, rank 0's value
                self.assertNotIn(":/spill", w, h)
                self.assertEqual(tier_names(run_env(w)), ["TF_DS_SPILL_GIB"], h)
                self.assertEqual(run_env(w)["TF_DS_SPILL_GIB"], run_env(line)["TF_DS_SPILL_GIB"], h)
        return line, run_env(line), out

    def assertOff(self, extra, *words):
        line, env, out = self.head(extra)
        self.assertNotIn(":/spill", line)
        self.assertEqual(env["TF_DS_SPILL_GIB"], ["0"], line)
        (why,) = env.get("TF_DS_SPILL_OFF_REASON", [""])
        reason = line.split("TF_DS_SPILL_OFF_REASON=")[1].split(" -e ")[0] if "TF_DS_SPILL_OFF_REASON=" in line else ""
        for w in words:
            self.assertIn(w, reason, (w, reason))
        self.assertLessEqual(len(reason.encode()), 120, reason)
        self.assertIn("SSD TIER OFF", out)
        return reason, out

    def test_a_missing_folder_is_created_as_you_and_mounted_on_every_rank(self):
        self.assertFalse(self.default.exists())
        line, env, out = self.head()
        self.assertIn(f"-v {self.default}:/spill ", line)
        self.assertTrue(self.default.is_dir())
        self.assertEqual(self.default.stat().st_uid, os.getuid())
        self.assertEqual(env["TF_DS_SPILL_GIB"], ["256"])
        self.assertEqual(env["TF_DS_SPILL_DIR"], ["/spill"])
        self.assertNotIn("TF_DS_SPILL_OFF_REASON", env)

    def test_a_folder_of_its_own_elsewhere_is_mounted(self):
        d = self.tmp / "disk" / "ds-spill"
        line, env, out = self.head({"SPILL_DIR": str(d)})
        self.assertIn(f"-v {d}:/spill ", line)
        self.assertTrue(d.is_dir())

    def test_glms_folder_equal_inside_or_holding_it_is_off_and_named(self):
        for d in (lambda: self.glm, lambda: self.glm / "ds", lambda: self.home / ".cache"):
            self.setUp()
            with self.subTest(SPILL_DIR=str(d())):
                reason, out = self.assertOff({"SPILL_DIR": str(d())}, "overlaps", "tensorfold-spill")
                self.assertIn(str(self.glm), out)

    def test_another_forbidden_folder_given_by_hand_is_off(self):
        other = self.tmp / "other-tier"
        self.assertOff({"SPILL_DIR": str(other / "x"), "SPILL_FORBID": str(other)}, "overlaps")

    def test_the_model_cache_and_the_recipes_own_folders_are_off(self):
        for d, word in ((lambda: self.home / ".cache" / "huggingface" / "spill", "overlaps"),
                        (lambda: self.home / ".cache" / "dsv41-tensorfold", "overlaps"),
                        (lambda: self.home / ".local" / "state" / "dsv41-tensorfold" / "spill", "overlaps"),
                        (lambda: self.home, "home"), (lambda: Path("/"), "home")):
            self.setUp()
            with self.subTest(SPILL_DIR=str(d())):
                self.assertOff({"SPILL_DIR": str(d())}, word)

    def test_memory_or_a_network_file_system_is_off_and_named(self):
        for fs in ("tmpfs", "nfs4", "overlay"):
            self.setUp()
            with self.subTest(fs=fs):
                (self.sim / "fstype").write_text(fs + "\n")
                self.assertOff({}, fs)

    def test_a_folder_that_belongs_to_root_is_off(self):
        self.default.mkdir(parents=True)
        (self.sim / "root_owned").write_text("")
        self.assertOff({}, "root")

    def test_an_owner_that_cannot_be_read_is_off(self):
        (self.sim / "stat_fails").write_text("")
        self.assertOff({}, "owner")

    def test_free_space_under_the_floor_is_off_and_named(self):
        reason, out = self.assertOff({"SPILL_FLOOR_GIB": "1025"}, "1024 GiB free", "SPILL_FLOOR_GIB")
        self.setUp()
        (self.sim / "df_avail").write_text(str(49 * 2 ** 30) + "\n")
        self.assertOff({"SPILL_FLOOR_GIB": "50"}, "49 GiB free", "SPILL_FLOOR_GIB 50")

    def test_free_space_under_twice_the_cap_and_the_floor_warns_and_stays_on(self):
        line, env, out = self.head({"SPILL_GIB": "500", "SPILL_FLOOR_GIB": "50"})    # 2 x 500 + 50 > 1024 free
        self.assertIn(":/spill ", line)
        self.assertEqual(env["TF_DS_SPILL_GIB"], ["500"])
        self.assertIn("1050 GiB", out)
        self.assertIn("kept builds", out)
        self.assertNotIn("SSD TIER OFF", out)
        self.setUp()
        line, env, out = self.head({"SPILL_GIB": "480", "SPILL_FLOOR_GIB": "50"})    # 1010 GiB: no warning
        self.assertNotIn("kept builds", out)

    def test_a_check_that_hangs_is_cut_and_the_start_goes_on(self):
        import time
        (self.sim / "findmnt_hangs").write_text("")
        t0 = time.monotonic()
        reason, out = self.assertOff({"SPILL_CHECK_TIMEOUT": "2"}, "2 s")
        self.assertLess(time.monotonic() - t0, 30, out)

    def test_a_path_with_other_characters_is_off(self):
        self.assertOff({"SPILL_DIR": str(self.tmp / "my spill")}, "characters")

    def test_a_check_timeout_that_is_no_number_is_off_not_refused(self):
        self.assertOff({"SPILL_CHECK_TIMEOUT": "soon"}, "SPILL_CHECK_TIMEOUT")

    def test_the_tier_off_checks_and_creates_nothing(self):
        line, env, out = self.head({"SPILL_GIB": "0"})
        self.assertNotIn(":/spill", line)
        self.assertFalse(self.default.exists())

    def test_dry_run_creates_nothing_and_prints_the_mount(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "DRY_RUN": "1", **self.ON})
        self.assertEqual(code, 0, out)
        self.assertFalse(self.default.exists(), out)
        self.assertIn(f"{self.default}:/spill", out)
        self.assertIn("would create it", out)


SNAP = "hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
DATA = "tensorfold-dsv41/1bae0e10"

# fake python3: with $SIM/hash_hangs, the weight files' hashing (scripts/nodes.sh weights_sha) hangs; else python3
FAKE_PYTHON = r"""#!BASH
if [[ -e "$SIM/hash_hangs" && "$*" == *weights-sha* && "$*" == *" hash "* ]]; then exec sleep 600; fi
exec REAL "$@"
"""


class WeightDigests(StartRun):
    """Each Spark's weight file's SHA-256 (the SSD tier's store is named from all four): prepare.sh hashes each Spark's
    own file, all four at once, each under WEIGHTS_SHA_TIMEOUT, into a cache in that Spark's state folder keyed by the
    file's size, mtime_ns, ctime_ns, inode and device; start.sh only reads the caches and gives each rank its own
    TF_DS_RANK_CACHE_SHA and every rank the identity's names. A missing or stale digest is that rank's empty
    TF_DS_RANK_CACHE_SHA (the engine names it, the tier off) and the start goes on."""
    ALL = ("head", "w1", "w2", "w3")
    ON = {"SPILL_GIB": "256"}

    def setUp(self):
        super().setUp()
        bin_ = self.tmp / "bin"
        (bin_ / "python3").write_text(FAKE_PYTHON.replace("BASH", BASH).replace("REAL", shutil.which("python3")))
        (bin_ / "python3").chmod(0o755)
        self.hf = self.home / ".cache" / "huggingface"
        self.copies = {f"WORKER_HF_CACHE{i}": str(self.sim / f"w{i}-hf") for i in (1, 2, 3)}
        for g in range(4):     # this Spark's: every node's file (a worker without its own copy reads it over NFS)
            self.weight(self.hf, g)
        for i in (1, 2, 3):    # each worker's own copy: its node's file
            self.weight(self.sim / f"w{i}-hf", i)

    def weight(self, root, g, name=None, body=None):
        d = root / DATA / "rank-cache"
        d.mkdir(parents=True, exist_ok=True)
        f = d / (name or f"rank{g}of4-abc{g}.bin")
        f.write_bytes(body if body is not None else (b"%d" % g) * (5000 + 997 * g))
        return f

    def sha(self, f):
        import hashlib
        return hashlib.sha256(Path(f).read_bytes()).hexdigest()

    def hash_all(self, extra=None):
        body = "node_order() { echo 0,1,2,3; }\nmkdir -p \"$STATE_DIR\"\nweights_hash_all"
        code, out = self.bash(body, {**self.copies, **(extra or {})})
        self.assertEqual(code, 0, out)
        self.assertNoTraceback(out)
        return out

    def entries(self, home):
        d = home / ".local" / "state" / "dsv41-tensorfold" / "weights-sha"
        return sorted(d.glob("*.json")) if d.is_dir() else []

    def start_envs(self, extra=None):
        for h in self.ALL:
            self.containers(h, NAME)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", **self.ON, **self.copies, **(extra or {})})
        self.assertEqual(code, 0, out)
        self.assertIn("is now LIVE", out)
        envs = {}
        for h in self.ALL:
            (line,) = self.run_lines(h)
            envs[h] = run_env(line)
        return envs, out

    def test_prepare_writes_one_keyed_cache_entry_per_spark(self):
        import json
        out = self.hash_all()
        homes = {"head": self.home, **{f"w{i}": self.sim / "home" / f"w{i}" for i in (1, 2, 3)}}
        files = {"head": self.hf / DATA / "rank-cache" / "rank0of4-abc0.bin",
                 **{f"w{i}": self.sim / f"w{i}-hf" / DATA / "rank-cache" / f"rank{i}of4-abc{i}.bin" for i in (1, 2, 3)}}
        for h, home in homes.items():
            (e,) = self.entries(home)
            entry = json.loads(e.read_text())
            st = files[h].stat()
            self.assertEqual(entry["file"], str(files[h]), h)
            self.assertEqual(entry["key"], [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev], h)
            self.assertEqual(entry["sha256"], self.sha(files[h]), h)
        self.assertIn("rank 3", out)

    def test_workers_reading_over_nfs_are_hashed_here_from_this_sparks_files(self):
        self.copies = {}
        self.hash_all()
        self.assertEqual(len(self.entries(self.home)), 4)
        for i in (1, 2, 3):
            self.assertEqual(self.entries(self.sim / "home" / f"w{i}"), [])

    def test_start_gives_each_rank_its_own_digest_and_every_rank_the_identity(self):
        self.hash_all()
        envs, out = self.start_envs()
        want = {"head": self.sha(self.hf / DATA / "rank-cache" / "rank0of4-abc0.bin"),
                **{f"w{i}": self.sha(self.sim / f"w{i}-hf" / DATA / "rank-cache" / f"rank{i}of4-abc{i}.bin") for i in (1, 2, 3)}}
        for h in self.ALL:
            self.assertEqual(envs[h].get("TF_DS_RANK_CACHE_SHA"), [want[h]], h)
        self.assertEqual(len(set(want.values())), 4)
        head = envs["head"]
        self.assertRegex(head["TF_DS_ENGINE_ID"][0], r"^[0-9a-f]{64}$")
        self.assertEqual(head["TF_DS_MODEL_REVISION"], ["deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170"])
        self.assertEqual(head["TF_DS_ENGRAM_REVISION"], ["deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170"])
        self.assertEqual(head["TF_DS_ABLATION_ID"], ["none"])
        self.assertEqual(head["TF_DS_KIT_ID"], ["b88aca850cb2e473069a4cd5d225e7a4b188fe8b"])
        for h in self.ALL[1:]:      # every Spark keeps a store, named from the same identity
            for k in ("TF_DS_ENGINE_ID", "TF_DS_MODEL_REVISION", "TF_DS_ABLATION_ID", "TF_DS_ENGRAM_REVISION", "TF_DS_KIT_ID"):
                self.assertEqual(envs[h].get(k), head[k], (h, k))

    def test_the_overlay_names_its_revision(self):
        envs, out = self.start_envs({"ABLIT": "1"})
        self.assertEqual(envs["head"]["TF_DS_ABLATION_ID"], ["keys@de412d2249632903dace23399526e00a145aada5"])

    def test_a_weight_file_touched_since_it_was_hashed_leaves_its_rank_without_a_digest(self):
        self.hash_all()
        f = self.sim / "w2-hf" / DATA / "rank-cache" / "rank2of4-abc2.bin"
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
        envs, out = self.start_envs()
        self.assertEqual(envs["w2"]["TF_DS_RANK_CACHE_SHA"], [""])
        for h in ("head", "w1", "w3"):
            self.assertRegex(envs[h]["TF_DS_RANK_CACHE_SHA"][0], r"^[0-9a-f]{64}$", h)
        self.assertIn("rank 2", out)
        self.assertIn("changed", out)
        self.assertIn("prepare.sh", out)
        self.assertEqual(envs["head"]["TF_DS_SPILL_GIB"], ["256"])      # the engine names it and turns the tier off

    def test_no_digest_yet_is_an_empty_digest_and_the_start_goes_on(self):
        envs, out = self.start_envs()
        for h in self.ALL:
            self.assertEqual(envs[h]["TF_DS_RANK_CACHE_SHA"], [""], h)
        self.assertIn("no weight-file digest", out)

    def test_dry_run_computes_nothing(self):
        code, out = self.run_script(["start.sh"], {"PREPARE": "0", "DRY_RUN": "1", **self.ON, **self.copies})
        self.assertEqual(code, 0, out)
        for home in [self.home] + [self.sim / "home" / f"w{i}" for i in (1, 2, 3)]:
            self.assertEqual(self.entries(home), [], home)
        self.assertIn("TF_DS_RANK_CACHE_SHA=", out)

    def test_a_hashing_run_that_hangs_is_cut_by_its_timeout(self):
        import time
        (self.sim / "hash_hangs").write_text("")
        t0 = time.monotonic()
        out = self.hash_all({"WEIGHTS_SHA_TIMEOUT": "2"})
        self.assertLess(time.monotonic() - t0, 30, out)
        self.assertIn("2 s", out)
        for home in [self.home] + [self.sim / "home" / f"w{i}" for i in (1, 2, 3)]:
            self.assertEqual(self.entries(home), [], home)

    def test_a_second_weight_file_of_a_node_is_no_digest(self):
        self.weight(self.sim / "w1-hf", 1, name="rank1of4-old.bin")
        out = self.hash_all()
        self.assertEqual(self.entries(self.sim / "home" / "w1"), [])
        self.assertIn("2 weight files", out)

    def test_a_cached_digest_is_read_not_recomputed(self):
        import json
        self.hash_all()
        (e,) = self.entries(self.home)
        entry = json.loads(e.read_text())
        entry["sha256"] = "f" * 64                 # a digest the file does not have: only a read would keep it
        e.write_text(json.dumps(entry))
        self.hash_all()
        self.assertEqual(json.loads(e.read_text())["sha256"], "f" * 64)
        envs, out = self.start_envs()
        self.assertEqual(envs["head"]["TF_DS_RANK_CACHE_SHA"], ["f" * 64])

    def test_the_tier_off_reads_no_digest_and_names_no_identity(self):
        self.hash_all()
        envs, out = self.start_envs({"SPILL_GIB": "0"})
        for h in self.ALL:
            for k in ("TF_DS_RANK_CACHE_SHA", "TF_DS_ENGINE_ID", "TF_DS_MODEL_REVISION", "TF_DS_KIT_ID"):
                self.assertNotIn(k, envs[h], (h, k))

    def test_identity_names_exported_by_hand_do_not_pass(self):
        self.hash_all()
        envs, out = self.start_envs({"TF_DS_ENGINE_ID": "mine", "TF_DS_RANK_CACHE_SHA": "0" * 64, "TF_DS_KIT_ID": "x"})
        self.assertNotEqual(envs["head"]["TF_DS_ENGINE_ID"], ["mine"])
        self.assertEqual(len(envs["head"]["TF_DS_ENGINE_ID"]), 1)
        self.assertEqual(envs["head"]["TF_DS_KIT_ID"], ["b88aca850cb2e473069a4cd5d225e7a4b188fe8b"])
        for h in self.ALL:
            self.assertEqual(len(envs[h]["TF_DS_RANK_CACHE_SHA"]), 1, h)
            self.assertNotEqual(envs[h]["TF_DS_RANK_CACHE_SHA"], ["0" * 64], h)
        self.assertIn("TF_DS_ENGINE_ID", out)

    def test_direct_and_buffered_reads_give_the_files_digest(self):
        import json
        f = self.weight(self.tmp / "big", 2, body=os.urandom(40 * 2 ** 20 + 123))     # chunks of 16 MiB and a tail
        for how, extra in (("O_DIRECT", ""), ("buffered", " buffered")):
            with self.subTest(how=how):
                cache = self.tmp / f"cache-{how}"
                code, out = self.bash(f"weights_sha 60 hash '{cache}' '{f.parent}' 2{extra}")
                self.assertEqual(code, 0, out)
                kind, sha = out.split()[:2]
                self.assertEqual((kind, sha), ("ok", self.sha(f)), out)
                (e,) = (cache / "weights-sha").glob("*.json")
                self.assertEqual(json.loads(e.read_text())["sha256"], self.sha(f))
                if how == "buffered":
                    self.assertIn("buffered", out)

    def test_prepare_hashes_the_weight_files_before_it_records_its_marker(self):
        text = (ROOT / "scripts" / "prepare.sh").read_text()
        self.assertIn("weights_hash_all", text)
        self.assertLess(text.index("weights_hash_all"), text.index("write_prepared_marker ||"))


class PreparedState(ScriptTest):
    BODY = 'image_ident() { echo image-ident; }\nprepared_state'

    def state(self, extra=None):
        code, out = self.bash(self.BODY, extra)
        self.assertEqual(code, 0, out)
        self.assertNoTraceback(out)
        return out.strip()

    def assertChanges(self, extra=None, edit=None):
        before = self.state()
        if edit:
            edit()
        after = self.state(extra)
        self.assertNotEqual(before, after, f"the marker did not change: {after}")

    def test_same_settings_same_marker(self):
        self.assertEqual(self.state(), self.state())

    def test_hf_cache_changes_marker(self):
        self.assertChanges({"HF_CACHE": str(self.tmp / "other-hf")})

    def test_nfs_path_changes_marker(self):
        self.assertChanges({"NFS_PATH": "/other"})

    def test_nfs_volume_changes_marker(self):
        self.assertChanges({"NFS_VOLUME": "other-volume"})

    def test_each_nfs_server_changes_marker(self):
        for var in ["NFS_SERVER1", "NFS_SERVER2", "NFS_SERVER3"]:
            with self.subTest(var=var):
                self.assertChanges({var: "10.9.9.9"})

    def test_chat_template_changes_marker(self):
        tpl = self.repo / "model" / "chat_template.jinja"
        self.assertChanges(edit=lambda: tpl.write_text(tpl.read_text() + "\n{# changed #}\n"))

    def test_same_length_template_change_changes_marker(self):
        tpl = self.repo / "model" / "chat_template.jinja"
        def edit():
            t = tpl.read_text()
            i = next(k for k, ch in enumerate(t) if ch.isalpha())
            tpl.write_text(t[:i] + ("X" if t[i] != "X" else "Y") + t[i + 1:])
        self.assertChanges(edit=edit)

    def test_spaces_in_paths_count(self):
        a = self.state({"HF_CACHE": "/data/hf cache"})
        b = self.state({"HF_CACHE": "/data/hfcache"})
        self.assertNotEqual(a, b)

    def test_no_two_settings_give_the_same_marker(self):
        # a field's value holding the next field's text must not shift it into that field
        pairs = [
            ({"HF_CACHE": "/a nfs=vol:/b", "NFS_PATH": "/c", "NFS_VOLUME": "vol"},
             {"HF_CACHE": "/a", "NFS_PATH": "/b nfs=vol:/c", "NFS_VOLUME": "vol"}),
            ({"HF_CACHE": "/a nfs_path=/b", "NFS_PATH": "/c"}, {"HF_CACHE": "/a", "NFS_PATH": "/b nfs_path=/c"}),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertNotEqual(self.state(a), self.state(b))

    def test_template_that_cannot_be_hashed_fails(self):
        (self.repo / "model" / "chat_template.jinja").unlink()
        code, out = self.bash(self.BODY)
        self.assertNotEqual(code, 0, out)
        self.assertIn("chat_template", out)

    def test_marker_does_not_depend_on_the_locale(self):
        a = self.state({"HF_CACHE": "/data/\u00e5", "LC_ALL": "C"})
        b = self.state({"HF_CACHE": "/data/\u00e5", "LC_ALL": "C.UTF-8"})
        self.assertEqual(a, b)

    def test_writing_the_marker(self):
        marker = self.home / ".local/state/dsv41-tensorfold/prepared"
        body = 'image_ident() { echo image-ident; }\nmkdir -p "$STATE_DIR"\nwrite_prepared_marker'
        code, out = self.bash(body)
        self.assertEqual(code, 0, out)
        self.assertEqual(marker.read_text().strip(), self.state())

    def test_failed_hash_keeps_the_previous_marker(self):
        marker = self.home / ".local/state/dsv41-tensorfold/prepared"
        marker.parent.mkdir(parents=True)
        marker.write_text("model=valid-before\n")
        (self.repo / "model" / "chat_template.jinja").unlink()
        code, out = self.bash('image_ident() { echo image-ident; }\nwrite_prepared_marker')
        self.assertNotEqual(code, 0, out)
        self.assertEqual(marker.read_text(), "model=valid-before\n")
        self.assertFalse(marker.with_name("prepared.new").exists(), out)

    def test_failed_rename_keeps_the_previous_marker(self):
        marker = self.home / ".local/state/dsv41-tensorfold/prepared"
        marker.parent.mkdir(parents=True)
        marker.write_text("model=valid-before\n")
        fake = self.tmp / "bin" / "mv"
        fake.write_text("#!/bin/sh\necho 'mv: cannot move' >&2\nexit 1\n")
        fake.chmod(0o755)
        code, out = self.bash('image_ident() { echo image-ident; }\nwrite_prepared_marker')
        self.assertNotEqual(code, 0, out)
        self.assertEqual(marker.read_text(), "model=valid-before\n")

    def test_existing_fields_still_change_marker(self):
        for var, value in [("ORDER", "0,1,2,3"), ("ABLIT", "1"), ("WORKER2", "w9")]:
            with self.subTest(var=var):
                self.assertChanges({var: value})

    def test_each_worker_hf_cache_changes_marker(self):
        for var in ["WORKER_HF_CACHE1", "WORKER_HF_CACHE2", "WORKER_HF_CACHE3"]:
            with self.subTest(var=var):
                self.assertChanges({var: "/data/hf"})


class WorkerHfCache(ScriptTest):
    """WORKER_HF_CACHE1, WORKER_HF_CACHE2, WORKER_HF_CACHE3: worker i reads its own copy of what rank i reads, bound at
    /hf, instead of this Spark's HF_CACHE over NFS. prepare.sh checks the copy: every file the rank reads, at this
    Spark's size, and no other weight file of its node."""
    SNAP = "hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
    DATA = "tensorfold-dsv41/1bae0e10"
    KIT = "tensorfold-dsv41/kit-b88aca85/kit"   # the default kit (the 1M window's)

    def tree(self, root, ranks=(0, 1, 2, 3), shards=True):
        """An HF cache: the snapshot (two small files, a weight shard), the checkpoint view's links, the token map,
        the weight files of `ranks`, the kit (its own .safetensors too)."""
        s = root / self.SNAP
        s.mkdir(parents=True)
        (s / "config.json").write_text('{"model_type": "deepseek_v41"}')
        (s / "tokenizer.json").write_text("{}")
        if shards:
            (s / "model-00001-of-00048.safetensors").write_bytes(b"w" * 64)
        d = root / self.DATA
        (d / "rank-cache").mkdir(parents=True)
        (d / "checkpoint").mkdir()
        for f in ["config.json", "tokenizer.json", "model-00001-of-00048.safetensors"]:
            (d / "checkpoint" / f).symlink_to(f"../../../{self.SNAP}/{f}")
        (d / "dsv41_token_map.json").write_text("{}")
        for g in ranks:
            (d / "rank-cache" / f"rank{g}of4-abc{g}.bin").write_bytes(b"r" * (100 + g))
        k = root / self.KIT
        (k / "vision").mkdir(parents=True)
        (k / "engram.json").write_text("{}")
        (k / "vision" / "gate_bias_vl.safetensors").write_bytes(b"b" * 8)
        return root

    def setUp(self):
        super().setUp()
        self.tree(self.home / ".cache" / "huggingface")
        self.local = {f"WORKER_HF_CACHE{i}": str(self.sim / f"w{i}-hf") for i in (1, 2, 3)}

    def check(self, i):
        return self.bash(f"node_order() {{ echo 0,1,2,3; }}\ncheck_worker_files {i}", self.local)

    def test_a_complete_copy_passes(self):
        self.tree(self.sim / "w2-hf", ranks=(2,), shards=False)
        code, out = self.check(2)
        self.assertEqual(code, 0, out)
        self.assertNoTraceback(out)

    def test_a_missing_weight_file_fails_and_is_named(self):
        self.tree(self.sim / "w2-hf", ranks=(), shards=False)
        code, out = self.check(2)
        self.assertNotEqual(code, 0, out)
        self.assertIn("rank2of4-abc2.bin", out)

    def test_a_weight_file_of_another_size_fails(self):
        self.tree(self.sim / "w1-hf", ranks=(), shards=False)
        (self.sim / "w1-hf" / self.DATA / "rank-cache" / "rank1of4-abc1.bin").write_bytes(b"r" * 5)
        code, out = self.check(1)
        self.assertNotEqual(code, 0, out)
        self.assertIn("rank1of4-abc1.bin", out)

    def test_a_second_weight_file_of_its_node_fails(self):
        self.tree(self.sim / "w3-hf", ranks=(3,), shards=False)
        (self.sim / "w3-hf" / self.DATA / "rank-cache" / "rank3of4-old.bin").write_bytes(b"o")
        code, out = self.check(3)
        self.assertNotEqual(code, 0, out)
        self.assertIn("rank3of4-old.bin", out)

    def test_other_nodes_weight_files_and_the_shards_are_not_needed(self):
        # node 1's copy has neither the checkpoint's weight shards nor the other nodes' weight files: still complete
        self.tree(self.sim / "w1-hf", ranks=(1,), shards=False)
        code, out = self.check(1)
        self.assertEqual(code, 0, out)

    def test_a_missing_kit_file_fails(self):
        self.tree(self.sim / "w1-hf", ranks=(1,), shards=False)
        (self.sim / "w1-hf" / self.KIT / "vision" / "gate_bias_vl.safetensors").unlink()
        code, out = self.check(1)
        self.assertNotEqual(code, 0, out)
        self.assertIn("gate_bias_vl.safetensors", out)



class RailCheck(ScriptTest):
    """At three or four Sparks, every rail between this node and a worker (pair_links pairs them by subnet) carries a
    full-size, unfragmentable ping both ways before the engine's rings use it, as the second rail is at two."""
    PING = r"""#!BASH
# fake ping: logged as "<host> ping <args>"; a "<host> <address>" line in $SIM/noping gets no answer
echo "${SIM_HOST:-head} ping $*" >> "$SIM/ping.log"
for a; do dest=$a; done
! grep -qxF "${SIM_HOST:-head} $dest" "$SIM/noping" 2>/dev/null
"""
    # each node's RoCE ports as node_inventory prints them: a port and its PCIe twin, a subnet each
    INV = {h: f"default enP7s7\nmaster ok\nenpA mlx5_0 10.255.0.{n}/24 3\nenpB mlx5_2 10.255.1.{n}/24 3\n"
           for h, n in (("head", 1), ("w1", 2), ("w2", 3), ("w3", 4))}
    RAILS = ('"1 enpA 10.255.0.1 enpA 10.255.0.2" "1 enpB 10.255.1.1 enpB 10.255.1.2" '
             '"2 enpA 10.255.0.1 enpA 10.255.0.3" "2 enpB 10.255.1.1 enpB 10.255.1.3" '
             '"3 enpA 10.255.0.1 enpA 10.255.0.4" "3 enpB 10.255.1.1 enpB 10.255.1.4"')
    MTU = 'net_mtu() { cat "$SIM/mtu.${SIM_HOST:-head}.$1" 2>/dev/null || echo 9000; }'

    def setUp(self):
        super().setUp()
        p = self.tmp / "bin" / "ping"
        p.write_text(self.PING.replace("BASH", BASH))
        p.chmod(0o755)

    def pings(self):
        f = self.sim / "ping.log"
        return sorted(f.read_text().splitlines()) if f.exists() else []

    def probe(self, extra="", env=None):
        return self.bash(f"{self.MTU}\n{extra}\nRAILS=({self.RAILS})\nprobe_links\necho probed", env)

    @staticmethod
    def both_ways(size=8972):
        rails = (("A", 0), ("B", 1))
        return sorted([f"head ping -c2 -W2 -I enp{r} -M do -s {size} 10.255.{k}.{1 + i}" for i in (1, 2, 3) for r, k in rails]
                      + [f"w{i} ping -c2 -W2 -I enp{r} -M do -s {size} 10.255.{k}.1" for i in (1, 2, 3) for r, k in rails])

    def test_pair_links_names_each_rail_between_this_node_and_a_worker(self):
        inv = "".join(f"{r} {line}\n" for r, h in enumerate(("head", "w1", "w2", "w3")) for line in self.INV[h].splitlines())
        code, out = self.bash(f"pair_links 0 1 2 3 <<'EOF'\n{inv}EOF")
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(l for l in out.splitlines() if l.startswith("rail ")),
                         sorted(f"rail {i} enp{r} 10.255.{k}.1 enp{r} 10.255.{k}.{1 + i}"
                                for i in (1, 2, 3) for r, k in (("A", 0), ("B", 1))))

    def test_pair_links_pairs_a_port_and_its_twin_in_one_subnet_by_the_split_prefix(self):
        # a port and its PCIe twin addressed .N and .10N in one /24: the /26 split tells them apart
        inv = "".join(f"{r} default eth0\n{r} enpA mlx5_0 10.0.0.{n}/24 3\n{r} enpB mlx5_1 10.0.0.{100 + n}/24 3\n"
                      for r, n in ((0, 1), (1, 2)))
        code, out = self.bash(f"pair_links 0 1 <<'EOF'\n{inv}EOF")
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(l for l in out.splitlines() if l.startswith("rail ")),
                         ["rail 1 enpA 10.0.0.1 enpA 10.0.0.2", "rail 1 enpB 10.0.0.101 enpB 10.0.0.102"])

    def test_every_rail_answers_a_full_size_ping_both_ways(self):
        code, out = self.probe()
        self.assertEqual(code, 0, out)
        self.assertIn("probed", out)
        self.assertEqual(self.pings(), self.both_ways())
        self.assertIn("6 of 6 answer full-size pings both ways", out)

    def test_a_rail_that_does_not_answer_stops_the_start_and_is_named(self):
        (self.sim / "noping").write_text("head 10.255.1.3\n")
        code, out = self.probe()
        self.assertNotEqual(code, 0, out)
        self.assertNotIn("probed", out)
        self.assertIn("rank 2 (w2)", out)
        self.assertIn("enpB 10.255.1.1", out)
        self.assertIn("enpB 10.255.1.3", out)
        self.assertIn("RAIL_CHECK=0", out)
        self.assertNotIn("rank 1 (w1)", out)

    def test_the_way_back_is_checked_too(self):
        (self.sim / "noping").write_text("w3 10.255.0.1\n")
        code, out = self.probe()
        self.assertNotEqual(code, 0, out)
        self.assertIn("rank 3 (w3)", out)
        self.assertIn("enpA 10.255.0.4", out)

    def test_the_smaller_mtu_of_the_two_ends_sets_the_size(self):
        (self.sim / "mtu.w3.enpB").write_text("1500\n")
        code, out = self.probe()
        self.assertEqual(code, 0, out)
        w3b = [l for l in self.pings() if " -I enpB " in l and (l.startswith("w3 ") or l.endswith(" 10.255.1.4"))]
        self.assertEqual(len(w3b), 2, self.pings())
        self.assertTrue(all(" -s 1472 " in l for l in w3b), w3b)
        self.assertEqual(len([l for l in self.pings() if " -s 8972 " in l]), 10)

    def test_rail_check_0_sends_no_ping(self):
        code, out = self.probe(env={"RAIL_CHECK": "0"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.pings(), [])
        self.assertIn("RAIL_CHECK=0", out)

    def test_dry_run_skips_a_worker_it_cannot_reach(self):
        code, out = self.probe("WORKER_DOWN[2]=1", {"DRY_RUN": "1"})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.pings(), [l for l in self.both_ways() if "10.255.0.3" not in l and "10.255.1.3" not in l
                                        and not l.startswith("w2 ")])
        self.assertIn("w2", out)

    def test_detect_links_probes_the_rails_it_paired(self):
        for h, text in self.INV.items():
            (self.sim / f"inv.{h}").write_text(text)
        body = (f'{self.MTU}\nnode_inventory() {{ cat "$SIM/inv.${{SIM_HOST:-head}}"; }}\n'
                'link_info() { echo "10.255.0.1 enpA"; }\nMASTER_ADDR=192.0.2.1\ndetect_links\n'
                'echo "hcas ${NODE_HCAS[*]}"')
        code, out = self.bash(body)
        self.assertEqual(code, 0, out)
        self.assertIn("hcas mlx5_0,mlx5_2 mlx5_0,mlx5_2 mlx5_0,mlx5_2 mlx5_0,mlx5_2", out)
        self.assertEqual(self.pings(), self.both_ways())

    def test_a_rank_without_a_default_route_still_links(self):
        # the rings need each rank's RoCE devices only; a worker whose LAN gives it no default route starts as any other
        for h, text in self.INV.items():
            (self.sim / f"inv.{h}").write_text(text if h != "w2" else text.replace("default enP7s7\n", ""))
        body = (f'{self.MTU}\nnode_inventory() {{ cat "$SIM/inv.${{SIM_HOST:-head}}"; }}\n'
                'link_info() { echo "10.255.0.1 enpA"; }\nMASTER_ADDR=192.0.2.1\ndetect_links\n'
                'echo "hcas ${NODE_HCAS[*]}"')
        code, out = self.bash(body)
        self.assertEqual(code, 0, out)
        self.assertIn("hcas mlx5_0,mlx5_2 mlx5_0,mlx5_2 mlx5_0,mlx5_2 mlx5_0,mlx5_2", out)



class NoNcclEnvironment(StartRun):
    """The engine uses no NCCL, so the recipe gives it none: no NCCL_* variable reaches any rank's container (exported or not), no NCCL
    graph file is written on any Spark, and the builders of both are gone. The nodes and devices the engine's RDMA rings use stay."""
    ALL = ("head", "w1", "w2", "w3")
    EXPORTED = {"NCCL_DEBUG": "INFO", "NCCL_NET_PLUGIN": "none", "NCCL_CHANNELS": "4", "NCCL_ALGO": "Ring", "NCCL_IB_HCA": "mlx5_0"}

    def started(self, extra=None):
        for h in self.ALL:
            self.containers(h, NAME)
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", **(extra or {})})
        self.assertEqual(code, 0, out)
        return {h: self.run_lines(h) for h in self.ALL}, out

    def test_no_nccl_variable_reaches_any_rank(self):
        lines, out = self.started(self.EXPORTED)
        for h in self.ALL:
            (line,) = lines[h]
            self.assertEqual(sorted(k for k in run_env(line) if k.startswith("NCCL_")), [], h)
            self.assertNotIn("nccl-graph", line, h)
            self.assertNotIn("nccl-link", line, h)

    def test_no_graph_file_is_written(self):
        self.started()
        self.assertEqual(sorted(str(p) for p in self.tmp.rglob("nccl-graph-*")), [])
        self.assertEqual(sorted(str(p) for p in self.tmp.rglob("nccl-link-*")), [])

    def test_the_builders_are_gone_and_the_rings_nodes_stay(self):
        code, out = self.bash("for f in nccl_env nccl_env_n rank_nccl_env ring_graphs graph_xml; do type -t $f >/dev/null 2>&1 && echo \"still $f\"; done; declare -p NODE_DEV >/dev/null 2>&1 && echo still-NODE_DEV; [[ -v SOCKET_IFNAME ]] && echo still-SOCKET_IFNAME; declare -p NODE_HCAS NODE_GID >/dev/null && echo nodes-stay")
        self.assertEqual(code, 0, out)
        self.assertEqual(out.split(), ["nodes-stay"], out)
        for f in ("start.sh", "scripts/nodes.sh", "scripts/config.sh", "scripts/served.sh"):
            text = (ROOT / f).read_text()
            for gone in ("nccl_env", "rank_nccl_env", "ring_graphs", "graph_xml", "NCCL_NET_PLUGIN", "NODE_GRAPH"):
                self.assertNotIn(gone, text, (f, gone))

    def test_the_scripts_speak_of_nccl_only_to_say_it_is_not_used(self):
        files = [ROOT / "start.sh", ROOT / "stop.sh", ROOT / "setup.sh", ROOT / "scripts" / "local.sh.example",
                 *sorted((ROOT / "scripts").glob("*.sh"))]
        said = [(f.name, n, line.strip()) for f in files if f.exists()
                for n, line in enumerate(f.read_text().splitlines(), 1) if "NCCL" in line and "NCCL is not used" not in line]
        self.assertEqual(said, [])


# a stand-in for the image's tf-dsv41-rank-cache: --index-only prints "[rank G]" (or fails for a rank in $FAIL_INDEX,
# with nothing on stdout, as the real one does on a tensor it cannot describe); otherwise it writes the weight file
# under the name the engine looks for, and refuses an OUT_DIR that already holds one of that rank (as the real one)
FAKE_RANK_CACHE = r"""#!BASH
g=$3; out=${2/#\/hf/$HF_CACHE}
if [[ " $* " == *" --index-only "* ]]; then
  [[ ",$FAIL_INDEX," != *",$g,"* ]] || { echo "layers.0.attn.wq_a.weight: FileNotFound" >&2; exit 1; }
  printf '[rank %s]' "$g"; exit 0
fi
echo "$g" >> "$SIM/built"
echo "rank $g of 4: 1 MiB written" >&2
[[ -z "${FAIL_BUILD:-}" ]] || { echo "embed.weight: short read" >&2; exit 1; }
compgen -G "$out/rank${g}of4-*.bin" >/dev/null && { echo "$out already holds a weight file of rank $g" >&2; exit 1; }
printf 'weights %s' "$g" > "$out/rank${g}of4-$(printf '[rank %s]' "$g" | sha256sum | cut -c1-20).bin"
"""


class PrepareWeightFiles(ScriptTest):
    """Step 5 of prepare.sh, the loop that keeps or rebuilds each node's weight file, run with in_hf as a local stand-in."""

    def setUp(self):
        super().setUp()
        tools = self.tmp / "tools"
        tools.mkdir()
        (tools / "tf-dsv41-rank-cache").write_text(FAKE_RANK_CACHE.replace("BASH", BASH))
        (tools / "tf-dsv41-rank-cache").chmod(0o755)
        if not shutil.which("sha256sum"):
            (tools / "sha256sum").write_text("#!/bin/sh\nexec shasum -a 256 \"$@\"\n")
            (tools / "sha256sum").chmod(0o755)
        self.env["TOOLS"] = str(tools)
        self.env["HF_CACHE"] = str(self.tmp / "hf")
        lines = (self.repo / "scripts" / "prepare.sh").read_text().splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("for g in 0 1 2 3; do")
                     and "--index-only" in "\n".join(lines[i:i + 4]))
        end = next(i for i in range(start, len(lines)) if lines[i] == "done")
        self.loop = "\n".join(lines[start:end + 1])

    def run_loop(self, fail_index=""):
        body = (
            'in_hf() { local ep; while [[ "$1" == --* ]]; do [[ "$1" != --entrypoint ]] || ep=$2; shift 2; done; shift\n'
            '  PATH="$TOOLS:$PATH" "$ep" "$@"; }\n'
            'source ./scripts/setup-lib.sh\nIMAGE=image SNAP=/snap SNAP_C=/hf/snap\n'
            'mkdir -p "$DATA_DIR/rank-cache"\n' + self.loop)
        return self.bash(body, {"FAIL_INDEX": fail_index})

    def rank_files(self):
        d = next((self.tmp / "hf").glob("tensorfold-dsv41/*/rank-cache"), None)
        return sorted(f.name for f in d.glob("rank*of4-*.bin")) if d else []

    def built(self):
        f = self.sim / "built"
        return f.read_text().split() if f.exists() else []

    def test_builds_then_keeps_the_weight_files(self):
        code, out = self.run_loop()
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.rank_files()), 4, out)
        self.assertEqual(self.built(), ["0", "1", "2", "3"])
        code, out = self.run_loop()
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("ready"), 4, out)
        self.assertEqual(self.built(), ["0", "1", "2", "3"], "a ready weight file was rebuilt")

    def test_each_build_says_how_far_it_is(self):
        code, out = self.run_loop()
        self.assertEqual(code, 0, out)
        for g in "0123":
            self.assertRegex(out, rf"weight file {int(g) + 1} of 4: [0-9.]+ of [0-9.]+ MB \([0-9]+ %\)")
        self.assertNotIn("MiB written", out)                   # the tool's own counts stay in its log: one progress line

    def test_a_failed_build_shows_the_tools_last_lines(self):
        self.env["FAIL_BUILD"] = "1"
        code, out = self.run_loop()
        self.assertNotEqual(code, 0, out)
        self.assertIn("embed.weight: short read", out)
        self.assertIn("node 0's weight file was not built", out)

    def test_failed_index_keeps_that_ranks_weight_file(self):
        code, out = self.run_loop()
        self.assertEqual(code, 0, out)
        before = self.rank_files()
        for rank in "0123":
            with self.subTest(rank=rank):
                code, out = self.run_loop(fail_index=rank)
                self.assertNotEqual(code, 0, out)
                self.assertIn(f"node {rank}'s weight-file index", out)
                self.assertEqual(self.rank_files(), before, out)
        self.assertEqual(self.built(), ["0", "1", "2", "3"], "a weight file was rebuilt after a failed index")


# ---------------------------------------------------------------- Engram's packed shards (ENGRAM_PACKED=1)

REV = "1bae0e1088993b9b82de08f59de5be5a34e42170"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017@" + REV
MODEL_DIR = "hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
PACKED_SUB = ".cache/dsv41-tensorfold/engram-packed-1bae0e10"
TABLES_SUB = ".cache/dsv41-tensorfold/engram-1bae0e10"
STATE_SUB = ".local/state/dsv41-tensorfold"
ENGRAM_FILES = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")
SPARKS = ("head", "w1", "w2", "w3")
# the toy checkpoint's Engram layers: (layer, rows, value bytes a row, scale bytes a row, which of the two files); each
# file also holds another tensor, as the checkpoint's do. The widths take the verifier's widest word (8 B: 16 + 8) and
# its narrowest (1 B: 6 + 3).
TOY_LAYERS = ((1, 37, 16, 8, 0), (14, 23, 6, 3, 1))


def toy_value(layer, row, col):
    return (row * 31 + col * 7 + layer * 5) % 251


def toy_scale(layer, row, col):
    return (row * 13 + col * 3 + layer) % 253 + 1


def safetensors_bytes(entries):
    header, data = {}, b""
    for name, dtype, shape, body in entries:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(data), len(data) + len(body)]}
        data += body
    text = json.dumps(header).encode()
    return struct.pack("<Q", len(text)) + text + data


def toy_checkpoint(hf):
    """The checkpoint's two Engram files in an HF cache (the snapshot's files link into blobs) and its index."""
    base = hf / MODEL_DIR
    snap = base / "snapshots" / REV
    (base / "blobs").mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    weight_map = {"layers.0.attn.wq_a.weight": "model-00001-of-00048.safetensors"}
    for i, fname in enumerate(ENGRAM_FILES):
        entries = [(f"layers.{30 + i}.ffn.gate.weight", "U8", (4, 25), bytes(range(100)))]
        for layer, rows, w, s, f in TOY_LAYERS:
            if f != i:
                continue
            vals = bytes(toy_value(layer, r, c) for r in range(rows) for c in range(w))
            scales = bytes(toy_scale(layer, r, c) for r in range(rows) for c in range(s))
            entries += [(f"layers.{layer}.engram.embed.weight", "F8_E4M3", (rows, w), vals),
                        (f"layers.{layer}.engram.embed.scale", "F8_E8M0", (rows, s), scales)]
        body = safetensors_bytes(entries)
        blob = hashlib.sha256(body).hexdigest()
        (base / "blobs" / blob).write_bytes(body)
        (snap / fname).symlink_to(f"../../blobs/{blob}")
        weight_map.update({e[0]: fname for e in entries})
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (snap / "config.json").write_text("{}")
    return snap


def toy_range(rows, node, nodes):
    return rows * node // nodes, rows * (node + 1) // nodes


def toy_size(layer_spec, node, nodes):
    _, rows, w, s, _ = layer_spec
    lo, hi = toy_range(rows, node, nodes)
    return 4096 + (hi - lo) * (w + s)


# An Engram packer of its own (engram_pack.zig's format: a 4 KiB header, then a record a row, values then scales,
# written in runs of TOY_RUN rows), independent of the verifier under test. Keeps a shard whose header and size check out.
TOY_PACK = r"""
import json, os, struct, sys
tables, out, node, nodes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
run = int(os.environ.get("TOY_RUN", "4"))
lay = {}
for name in sorted(os.listdir(tables)):
    if not name.endswith(".safetensors"):
        continue
    p = os.path.join(tables, name)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    for k, e in h.items():
        for suffix, which in (("engram.embed.weight", 0), ("engram.embed.scale", 1)):
            if k.endswith(suffix):
                layer = int(k.split("layers.")[1].split(".")[0])
                lay.setdefault(layer, [None, None])[which] = (p, 8 + n + e["data_offsets"][0], e["shape"][1], e["shape"][0])
os.makedirs(out, exist_ok=True)
for layer, ((wp, wb, ww, rows), (sp, sb, sw, _)) in sorted(lay.items()):
    lo, hi = rows * node // nodes, rows * (node + 1) // nodes
    final = os.path.join(out, "engram-l%d-n%dof%d.tfe" % (layer, node, nodes))
    head = b"TFENGRM1" + struct.pack("<IIQQQII", 1, layer, rows, lo, hi, ww, sw)
    head += bytes(4096 - len(head))
    size = 4096 + (hi - lo) * (ww + sw)
    if os.path.exists(final) and os.path.getsize(final) == size and open(final, "rb").read(4096) == head:
        continue
    with open(wp, "rb") as fw, open(sp, "rb") as fs, open(final + ".partial", "wb") as o:
        o.write(head)
        at = lo
        while at < hi:
            k = min(run, hi - at)
            fw.seek(wb + at * ww)
            w = fw.read(k * ww)
            fs.seek(sb + at * sw)
            s = fs.read(k * sw)
            o.write(b"".join(w[i * ww:(i + 1) * ww] + s[i * sw:(i + 1) * sw] for i in range(k)))
            at += k
    os.replace(final + ".partial", final)
    print(json.dumps({"file": os.path.basename(final), "layer": layer, "lo": lo, "hi": hi, "bytes": size}))
"""

# fake docker for the packed shards: `run` maps each -v DST in the command's arguments back to its SRC (a named volume:
# $SIM_NFS, the head's HF cache as the NFS export shows it) and runs the image's packer (the toy packer above) or its
# python3 (stdin passed on) on those paths; `image inspect` finds the image when $SIM/image exists. Every call logged.
FAKE_DOCKER_ENGRAM = r"""#!PYTHON
import os, subprocess, sys
sim = os.environ["SIM"]
args = sys.argv[1:]
with open(os.path.join(sim, "docker.log"), "a") as f:
    f.write(os.environ.get("SIM_HOST", "head") + " docker " + " ".join(args) + "\n")
if args[:1] == ["image"]:
    sys.exit(0 if os.path.exists(os.path.join(sim, "image")) else 1)
if args[:1] != ["run"]:
    sys.exit(0)
VALUED = {"--name", "--user", "-e", "--network", "--entrypoint", "-v", "--device-read-bps", "--device-write-bps",
          "--ulimit", "--gpus", "--ipc", "--device", "--cap-add", "-w"}
mounts, entry, i = {}, None, 1
while i < len(args):
    a = args[i]
    if a == "-v":
        src, dst = args[i + 1].split(":")[:2]
        mounts[dst] = src if src.startswith("/") else os.environ["SIM_NFS"]
        i += 2
    elif a == "--entrypoint":
        entry = args[i + 1]
        i += 2
    elif a in VALUED:
        i += 2
    elif a.startswith("-"):
        i += 1
    else:
        break
def mapped(p):
    for dst in sorted(mounts, key=len, reverse=True):
        if p == dst or p.startswith(dst + "/"):
            return mounts[dst] + p[len(dst):]
    return p
rest = [mapped(x) for x in args[i + 1:]]
if entry == "tf-dsv41-engram-pack":
    sys.exit(subprocess.call([sys.executable, os.path.join(sim, "toy-pack.py")] + rest))
if entry == "python3":
    sys.exit(subprocess.call([sys.executable] + rest))
"""


class EngramToy(ScriptTest):
    """A toy checkpoint (two Engram layers in two files that hold another tensor each) on four Sparks: the head here,
    the workers w1-w3 each with its own HOME under the test's folder (reached by the fake ssh)."""

    def setUp(self):
        super().setUp()
        bin_ = self.tmp / "bin"
        (bin_ / "docker").write_text(FAKE_DOCKER_ENGRAM.replace("PYTHON", sys.executable))
        (bin_ / "docker").chmod(0o755)
        (bin_ / "df").write_text(FAKE_DF.replace("BASH", BASH))
        (bin_ / "df").chmod(0o755)
        (self.sim / "toy-pack.py").write_text(TOY_PACK)
        (self.sim / "image").write_text("")
        self.hf = self.home / ".cache" / "huggingface"
        self.snap = toy_checkpoint(self.hf)
        self.env["SIM_NFS"] = str(self.hf)
        self.own = {"WORKER_HF_CACHE1": "/w/hf", "WORKER_HF_CACHE2": "/w/hf", "WORKER_HF_CACHE3": "/w/hf"}

    def spark(self, host):
        return self.home if host == "head" else self.sim / "home" / host

    def packed_dir(self, host):
        return self.spark(host) / PACKED_SUB

    def ok_file(self, host):
        return self.spark(host) / STATE_SUB / "engram-packed.ok"

    def tables(self):
        return [str(self.snap / f) for f in ENGRAM_FILES]

    def toy_pack(self, out, node, nodes, run=4):
        r = subprocess.run([sys.executable, str(self.sim / "toy-pack.py"), str(self.snap), str(out), str(node), str(nodes)],
                           env={**os.environ, "TOY_RUN": str(run)}, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def verifier(self, *args, env=None, stdin=None):
        script = self.repo / "scripts" / "engram-verify.py"
        cmd = [sys.executable, "-I", "-" if stdin else str(script), *map(str, args)]
        with open(script) as f:
            r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})},
                               stdin=f if stdin else None)
        return r.returncode, r.stdout + r.stderr

    @staticmethod
    def key(p):
        s = os.stat(p)
        return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev]

    def write_ok(self, host, names):
        lines = [json.dumps({"path": str(self.packed_dir(host) / n), "key": self.key(self.packed_dir(host) / n),
                             "sha256": hashlib.sha256((self.packed_dir(host) / n).read_bytes()).hexdigest()})
                 for n in names]
        self.ok_file(host).parent.mkdir(parents=True, exist_ok=True)
        self.ok_file(host).write_text("".join(l + "\n" for l in lines))

    def deploy(self, nodes=4):
        """What engram-deploy.sh leaves: every node's shards packed on the head (the copies' source), each worker's
        node's shards on that worker, MANIFEST.sha256 of every shard and PACKED.json beside them on every Spark, each
        Spark's engram-packed.ok for the files of its own node, and each worker's copy of the tables."""
        head = self.packed_dir("head")
        for g in range(nodes):
            self.toy_pack(head, g, nodes)
        names = sorted(p.name for p in head.glob("engram-l*-n*of%d.tfe" % nodes))
        shas = {n: hashlib.sha256((head / n).read_bytes()).hexdigest() for n in names}
        sources = {f: os.path.basename(os.readlink(self.snap / f)) for f in ENGRAM_FILES}
        packed = {"format": 1, "model": MODEL, "sources": sources,
                  "sets": {str(nodes): {"image": "image-ident", "at": "2026-11-01T08:00:00+11:00",
                                        "files": {n: (head / n).stat().st_size for n in names}}}}
        for r, host in enumerate(SPARKS):
            node = r if nodes == 4 else 0
            d = self.packed_dir(host)
            d.mkdir(parents=True, exist_ok=True)
            mine = [n for n in names if n.endswith("-n%dof%d.tfe" % (node, nodes))]
            if host != "head":
                for n in mine:
                    shutil.copy2(head / n, d / n)
                t = self.spark(host) / TABLES_SUB
                t.mkdir(parents=True, exist_ok=True)
                for f in ENGRAM_FILES:
                    shutil.copyfile(self.snap / f, t / f)
            (d / "MANIFEST.sha256").write_text("".join(f"{shas[n]}  {n}\n" for n in names))
            (d / "PACKED.json").write_text(json.dumps(packed))
            self.write_ok(host, mine)
        return names

    def step7(self, extra=None):
        """prepare.sh's step 7 as it stands, with the checkpoint's paths prepare.sh's earlier steps set."""
        lines = (self.repo / "scripts" / "prepare.sh").read_text().splitlines()
        start = next(i for i, l in enumerate(lines) if re.match(r"^# -+ 7\.", l))
        end = next(i for i, l in enumerate(lines) if re.match(r"^# -+ 8\.", l))
        body = ('node_order() { echo 0,1,2,3; }\nSNAP="$(model_cache_dir)/snapshots/$(snapshot_rev)"\n'
                'SNAP_C=$(model_in_container)\n' + "\n".join(lines[start:end]))
        return self.bash(body, {"ENGRAM_PACKED": "1", **(extra or {})})

    def packer_runs(self):
        log = (self.sim / "docker.log").read_text() if (self.sim / "docker.log").exists() else ""
        return [l for l in log.splitlines() if "--entrypoint tf-dsv41-engram-pack" in l]


class PackedFiles(EngramToy):
    def test_each_shards_size_comes_from_the_tables_own_headers(self):
        for nodes in (4, 1):
            ranges = {L[0]: [] for L in TOY_LAYERS}
            for g in range(nodes):
                code, out = self.bash(f"packed_files {g} {nodes}")
                self.assertEqual(code, 0, out)
                rows = [l.split() for l in out.strip().splitlines()]
                self.assertEqual([r[:2] for r in rows],
                                 [[f"engram-l{L[0]}-n{g}of{nodes}.tfe", str(toy_size(L, g, nodes))] for L in TOY_LAYERS], out)
                for r in rows:
                    ranges[int(r[2])].append((int(r[4]), int(r[5])))
            # the nodes' ranges tile each table: every row in exactly one shard
            for layer, rows, *_ in TOY_LAYERS:
                self.assertEqual(ranges[layer], [toy_range(rows, g, nodes) for g in range(nodes)])
                self.assertEqual((ranges[layer][0][0], ranges[layer][-1][1]), (0, rows))

    def test_a_table_without_its_scales_is_named(self):
        code, out = self.verifier("plan", 0, 1, self.snap / ENGRAM_FILES[0])
        self.assertEqual(code, 0, out)
        bad = self.tmp / "bad.safetensors"
        bad.write_bytes(safetensors_bytes([("layers.1.engram.embed.weight", "F8_E4M3", (2, 4), bytes(8))]))
        code, out = self.verifier("plan", 0, 1, bad)
        self.assertEqual(code, 1, out)
        self.assertIn("layer 1", out)
        self.assertIn("scales", out)


class EngramVerify(EngramToy):
    def shard(self, node=0, nodes=1, layer=1, run=4):
        d = self.tmp / "shards"
        self.toy_pack(d, node, nodes, run=run)
        return d, d / f"engram-l{layer}-n{node}of{nodes}.tfe"

    def poke(self, path, offset, value=None):
        b = bytearray(path.read_bytes())
        b[offset] = (b[offset] ^ 0xFF) if value is None else value
        path.write_bytes(bytes(b))

    def test_a_shard_equal_to_its_tables_passes_with_its_sha256(self):
        d, _ = self.shard(1, 4)
        code, out = self.verifier("verify", 1, 4, d, *self.tables())
        self.assertEqual(code, 0, out)
        got = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
        self.assertEqual([x["file"] for x in got], ["engram-l1-n1of4.tfe", "engram-l14-n1of4.tfe"])
        for x in got:
            self.assertEqual(x["sha256"], hashlib.sha256((d / x["file"]).read_bytes()).hexdigest())
            self.assertEqual(x["key"], self.key(d / x["file"]))

    def test_a_defect_at_a_run_boundary_is_found(self):
        # the packer writes runs of 4 rows: one byte changed in the first record of the third run (row 8), and one in
        # the last record; the verifier's own runs of 4 rows too (a defect at its boundary)
        rec = 16 + 8
        for run_rows in ("65536", "4"):
            for row in (8, 36):
                with self.subTest(row=row, run_rows=run_rows):
                    d, p = self.shard()
                    self.poke(p, 4096 + row * rec)
                    code, out = self.verifier("verify", 0, 1, d, *self.tables(), "--run-rows", run_rows)
                    self.assertEqual(code, 1, out)
                    self.assertIn(f"row {row} ", out)
                    self.assertIn("value byte 0", out)
                    shutil.rmtree(d)

    def test_a_record_that_differs_from_its_table_row_is_named(self):
        d, p = self.shard(layer=14)
        rec = 6 + 3
        self.poke(p, 4096 + 20 * rec + 6 + 2)
        code, out = self.verifier("verify", 0, 1, d, *self.tables())
        self.assertEqual(code, 1, out)
        self.assertIn("row 20 ", out)
        self.assertIn("scale byte 2", out)
        self.assertIn(str(p), out)

    def test_a_shard_from_another_range_is_refused(self):
        d, p = self.shard(1, 4)
        os.rename(p, d / "engram-l1-n2of4.tfe")
        (d / "engram-l14-n1of4.tfe").rename(d / "engram-l14-n2of4.tfe")
        code, out = self.verifier("verify", 2, 4, d, *self.tables())
        self.assertEqual(code, 1, out)
        # node 1 holds 9 rows ([9, 18)), node 2 nine too ([18, 27)): the same size, another range
        self.assertIn("rows [9, 18)", out)
        self.assertIn("reads layer 1 rows [18, 27)", out)

    def test_a_shard_of_another_size_is_refused(self):
        d, p = self.shard()
        with open(p, "ab") as f:
            f.write(b"\0" * 24)
        code, out = self.verifier("verify", 0, 1, d, *self.tables())
        self.assertEqual(code, 1, out)
        self.assertIn(f"{toy_size(TOY_LAYERS[0], 0, 1) + 24} bytes", out)

    def test_another_header_version_is_refused(self):
        d, p = self.shard()
        self.poke(p, 8, 2)
        code, out = self.verifier("verify", 0, 1, d, *self.tables())
        self.assertEqual(code, 1, out)
        self.assertIn("header version 2", out)

    def test_it_paces_its_reads(self):
        spec = importlib.util.spec_from_file_location("engram_verify", self.repo / "scripts" / "engram-verify.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        d, _ = self.shard(run=4)
        now = [100.0]
        sleeps = []

        def sleep(s):
            sleeps.append(s)
            now[0] += s
        args = ["0", "1", str(d), *self.tables(), "--run-rows", "4"]
        read = sum(p.stat().st_size for p in d.glob("*.tfe")) * 2 - 2 * 4096   # each record and its table row
        for mbps in ("0.001", "0"):
            sleeps.clear()
            now[0] = 100.0
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ev.cmd_verify(args + ["--mbps", mbps], clock=lambda: now[0], sleep=sleep), 0)
            if mbps == "0":
                self.assertEqual(sleeps, [])
            else:
                # 1,000 bytes a second: the run lasts as long as its bytes take at that rate
                self.assertGreater(len(sleeps), 4)
                self.assertGreaterEqual(now[0] - 100.0, (read - 4096 - 4 * 24) / 1000.0)
                self.assertLessEqual(now[0] - 100.0, (read + 4096) / 1000.0)

    def test_it_runs_from_stdin_with_paths_in_its_home(self):
        d = self.home / "packed"
        self.toy_pack(d, 3, 4)
        code, out = self.verifier("verify", 3, 4, "$HOME/packed", *self.tables(), "--ok", "$HOME/state/ok",
                                  env={"HOME": str(self.home)}, stdin=True)
        self.assertEqual(code, 0, out)
        oks = [json.loads(l) for l in (self.home / "state" / "ok").read_text().splitlines()]
        self.assertEqual(sorted(x["path"] for x in oks), sorted(str(p) for p in d.glob("*.tfe")))


class PackedPrepare(EngramToy):
    """prepare.sh's step 7 with ENGRAM_PACKED=1."""

    def test_packed_shards_are_checked_at_their_exact_size(self):
        for nodes, owned in ((4, "1"), (1, "0")):
            with self.subTest(nodes=nodes):
                self.deploy(nodes)
                code, out = self.step7({**self.own, "ENGRAM_OWNED": owned})
                self.assertEqual(code, 0, out)
                self.assertNoTraceback(out)
                self.assertEqual(self.packer_runs(), [], out)
                # one byte short on rank 2: refused, named, nothing packed (a worker with its own copy cannot pack)
                g = 2 if nodes == 4 else 0
                p = self.packed_dir("w2") / f"engram-l14-n{g}of{nodes}.tfe"
                p.write_bytes(p.read_bytes()[:-1])
                code, out = self.step7({**self.own, "ENGRAM_OWNED": owned})
                self.assertNotEqual(code, 0, out)
                self.assertIn(str(p), out)
                self.assertIn("rank 2", out)
                self.assertIn("engram-pack.sh", out)
                self.assertEqual(self.packer_runs(), [], out)
                for h in SPARKS:
                    shutil.rmtree(self.spark(h) / ".cache" / "dsv41-tensorfold", ignore_errors=True)
                    shutil.rmtree(self.spark(h) / STATE_SUB, ignore_errors=True)

    def test_every_ranks_check_runs_at_once(self):
        # each rank hashes its own disk (a copied shard is read once against its manifest, ~100 s a 100 GB shard), so the four
        # checks run together and their results are read in rank order
        self.deploy()
        wrap = ('eval "$(declare -f engram_check | sed \'1s/engram_check/engram_check_real/\')"\n'
                'engram_check() { echo "start $1" >> "$SIM/checks"; sleep 1; local rc=0; engram_check_real "$@" || rc=$?; '
                'echo "end $1" >> "$SIM/checks"; return $rc; }\n')
        lines = (self.repo / "scripts" / "prepare.sh").read_text().splitlines()
        start = next(i for i, l in enumerate(lines) if re.match(r"^# -+ 7\.", l))
        end = next(i for i, l in enumerate(lines) if re.match(r"^# -+ 8\.", l))
        body = ('node_order() { echo 0,1,2,3; }\nSNAP="$(model_cache_dir)/snapshots/$(snapshot_rev)"\n'
                'SNAP_C=$(model_in_container)\n' + wrap + "\n".join(lines[start:end]))
        code, out = self.bash(body, {"ENGRAM_PACKED": "1", **self.own})
        self.assertEqual(code, 0, out)
        events = (self.sim / "checks").read_text().split("\n")
        first_end = next(i for i, e in enumerate(events) if e.startswith("end"))
        self.assertEqual(sorted(e for e in events[:first_end]), ["start 0", "start 1", "start 2", "start 3"], events)
        ranks = [int(m) for m in re.findall(r"Rank ([0-3]) \(", out)]
        self.assertEqual(ranks, [0, 1, 2, 3], out)                  # read and logged in rank order

    def test_packed_shards_without_a_manifest_are_refused(self):
        self.deploy()
        (self.packed_dir("w1") / "MANIFEST.sha256").unlink()
        code, out = self.step7(self.own)
        self.assertNotEqual(code, 0, out)
        self.assertIn("without a manifest", out)
        self.assertIn("engram-pack.sh", out)

    def test_packed_files_of_another_checkpoint_are_refused(self):
        self.deploy()
        p = self.packed_dir("head") / "PACKED.json"
        x = json.loads(p.read_text())
        x["model"] = "deepseek-ai/DeepSeek-V4.1-Flash-1017@0000000000000000000000000000000000000000"
        p.write_text(json.dumps(x))
        code, out = self.step7(self.own)
        self.assertNotEqual(code, 0, out)
        self.assertIn("cut from deepseek-ai/DeepSeek-V4.1-Flash-1017@0000000000", out)
        # and on rank 0, shards cut from other blobs of the checkpoint's Engram files
        x["model"] = MODEL
        x["sources"][ENGRAM_FILES[1]] = "0" * 64
        p.write_text(json.dumps(x))
        code, out = self.step7(self.own)
        self.assertNotEqual(code, 0, out)
        self.assertIn(ENGRAM_FILES[1] + " blob " + "0" * 64, out)

    def test_a_worker_without_its_shards_is_refused_naming_engram_pack(self):
        self.deploy()
        for p in self.packed_dir("w3").glob("*.tfe"):
            p.unlink()
        code, out = self.step7(self.own)
        self.assertNotEqual(code, 0, out)
        self.assertIn("rank 3", out)
        self.assertIn("lacks its packed shards in ENGRAM_PACKED_DIR: run scripts/engram-pack.sh on rank 0", out)
        self.assertEqual(self.packer_runs(), [], out)

    def test_a_copied_shard_is_read_once_against_its_manifest(self):
        self.deploy()
        self.ok_file("w1").unlink()
        code, out = self.step7(self.own)
        self.assertEqual(code, 0, out)
        self.assertIn("hashed engram-l1-n1of4.tfe", out)
        self.assertTrue(self.ok_file("w1").exists())
        code, out = self.step7(self.own)
        self.assertEqual(code, 0, out)
        self.assertNotIn("hashed", out)

    def test_a_shard_that_differs_from_its_manifest_is_refused(self):
        self.deploy()
        p = self.packed_dir("w2") / "engram-l1-n2of4.tfe"
        b = bytearray(p.read_bytes())
        b[4096 + 10] ^= 1                      # a byte of its first record: the same size, other content
        p.write_bytes(bytes(b))
        code, out = self.step7(self.own)
        self.assertNotEqual(code, 0, out)
        self.assertIn(str(p), out)
        self.assertIn("differs", out)

    def test_packed_still_checks_the_workers_table_copies(self):
        self.deploy()
        (self.spark("w3") / TABLES_SUB / ENGRAM_FILES[0]).unlink()
        code, out = self.step7(self.own)
        self.assertNotEqual(code, 0, out)
        self.assertIn("w3", out)
        self.assertIn("Engram's tables", out)

    def test_rank_0_packs_and_verifies_its_own_missing_shards(self):
        self.deploy()
        d = self.packed_dir("head")
        for f in ("engram-l1-n0of4.tfe", "engram-l14-n0of4.tfe", "MANIFEST.sha256", "PACKED.json"):
            (d / f).unlink()
        self.ok_file("head").unlink()
        code, out = self.step7(self.own)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.packer_runs()), 1, out)
        self.assertIn(" /out 0 4", self.packer_runs()[0])
        manifest = (d / "MANIFEST.sha256").read_text()
        for n in ("engram-l1-n0of4.tfe", "engram-l14-n0of4.tfe"):
            self.assertIn(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}", manifest)
        self.assertEqual(json.loads((d / "PACKED.json").read_text())["model"], MODEL)
        self.assertTrue(self.ok_file("head").exists())

    def test_a_worker_over_nfs_packs_and_verifies_its_own(self):
        # rank 2 reads over NFS (no WORKER_HF_CACHE2): the recipe's own layout, its shards packed and verified there
        self.deploy()
        d = self.packed_dir("w2")
        shutil.rmtree(d)
        self.ok_file("w2").unlink()
        own = {k: v for k, v in self.own.items() if k != "WORKER_HF_CACHE2"}
        code, out = self.step7(own)
        self.assertEqual(code, 0, out)
        runs = self.packer_runs()
        self.assertEqual(len(runs), 1, out)
        self.assertTrue(runs[0].startswith("w2 docker run") and "dsv41-hf:/hf:ro" in runs[0], runs)
        verify = [l for l in (self.sim / "docker.log").read_text().splitlines() if "--entrypoint python3" in l]
        self.assertEqual(len(verify), 1, out)
        self.assertIn(" verify 2 4 /out ", verify[0])
        for n in ("engram-l1-n2of4.tfe", "engram-l14-n2of4.tfe"):
            self.assertIn(n, (d / "MANIFEST.sha256").read_text())


class PackedStart(EngramToy, StartRun):
    def runs(self, extra):
        code, out = self.run_script(["start.sh", "restart"], {"PREPARE": "0", "ENGRAM_PACKED": "1", **extra})
        self.assertEqual(code, 0, out)
        return {h: self.run_lines(h)[0] for h in ("w1", "w2", "w3")}

    def test_packed_worker_with_its_own_copy_reads_the_table_headers_from_it(self):
        for h, line in self.runs(self.own).items():
            home = self.spark(h)
            self.assertIn(" --engram /engram ", line)
            self.assertIn(f"-v {home}/{TABLES_SUB}:/engram:ro ", line)
            self.assertIn(f"-v {home}/{PACKED_SUB}:/engram-packed:ro ", line)
            self.assertIn("-e TF_DS_ENGRAM_PACKED=/engram-packed ", line)

    def test_packed_worker_over_nfs_reads_the_headers_over_nfs(self):
        line = self.runs({})["w1"]
        self.assertIn(f" --engram /hf/{MODEL_DIR}/snapshots/{REV} ", line)
        self.assertNotIn(":/engram:ro", line)
        self.assertIn(":/engram-packed:ro ", line)


FAKE_FINDMNT_DISK = """#!/bin/sh
case "$*" in *SOURCE*) echo /dev/nvme0n1p2 ;; *FSTYPE*) echo ext4 ;; esac
"""
FAKE_LSBLK = """#!/bin/sh
[ "$*" = "-no PKNAME /dev/nvme0n1p2" ] && echo nvme0n1
"""


class EngramPack(EngramToy):
    def pack(self, *args, extra=None):
        return self.run_script(["scripts/engram-pack.sh", *args], {"ENGRAM_PACKED": "1", **self.own, **(extra or {})},
                               timeout=120)

    def test_it_packs_every_node_verifies_and_records_them(self):
        code, out = self.pack()
        self.assertEqual(code, 0, out)
        d = self.packed_dir("head")
        names = sorted(p.name for p in d.glob("*.tfe"))
        self.assertEqual(names, sorted(f"engram-l{L[0]}-n{g}of4.tfe" for L in TOY_LAYERS for g in range(4)))
        self.assertEqual(sorted(l.split()[-2:][0] for l in self.packer_runs()), ["0", "1", "2", "3"], out)
        self.assertEqual({l.split()[-1] for l in self.packer_runs()}, {"4"}, out)
        manifest = dict(reversed(l.split()) for l in (d / "MANIFEST.sha256").read_text().splitlines())
        self.assertEqual(manifest, {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in names})
        packed = json.loads((d / "PACKED.json").read_text())
        self.assertEqual(packed["model"], MODEL)
        self.assertEqual(packed["sources"], {f: os.path.basename(os.readlink(self.snap / f)) for f in ENGRAM_FILES})
        self.assertEqual(packed["sets"]["4"]["files"], {n: (d / n).stat().st_size for n in names})
        oks = [json.loads(l) for l in self.ok_file("head").read_text().splitlines()]
        self.assertEqual(sorted(x["path"] for x in oks), sorted(str(d / n) for n in names))
        for i in (1, 2, 3):
            self.assertIn(f"engram-l1-n{i}of4.tfe", out)
        # again: nothing packed, everything verified again
        code, out = self.pack()
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.packer_runs()), 4, out)

    def test_the_setup_shows_the_packing_progress_to_its_end(self):
        code, out = self.bash("source ./scripts/setup-lib.sh\nPROGRESS_S=0.2 engram_pack_progress",
                              {"ENGRAM_PACKED": "1", **self.own})
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"Engram's packed tables: [0-9.]+ of [0-9.]+ MB \(100 %\)")
        self.assertEqual(len(self.packer_runs()), 4, out)
        code, out = self.bash("source ./scripts/setup-lib.sh\nPROGRESS_S=0.2 engram_pack_progress",
                              {"ENGRAM_PACKED": "1", **self.own})
        self.assertEqual(code, 0, out)
        self.assertNotIn("Engram's packed tables:", out)           # all packed already: no progress to show

    def test_whole_tables_are_one_set_every_node_reads(self):
        code, out = self.pack(extra={"ENGRAM_OWNED": "0"})
        self.assertEqual(code, 0, out)
        d = self.packed_dir("head")
        self.assertEqual(sorted(p.name for p in d.glob("*.tfe")), sorted(["engram-l1-n0of1.tfe", "engram-l14-n0of1.tfe"]))
        self.assertEqual(sorted(json.loads((d / "PACKED.json").read_text())["sets"]), ["1"])

    def test_it_paces_the_packers_on_the_disk(self):
        bin_ = self.tmp / "bin"
        for name, body in (("findmnt", FAKE_FINDMNT_DISK), ("lsblk", FAKE_LSBLK)):
            (bin_ / name).write_text(body)
            (bin_ / name).chmod(0o755)
        code, out = self.pack("--mbps", "400", "--verify", "none")
        self.assertEqual(code, 0, out)
        runs = self.packer_runs()
        self.assertEqual(len(runs), 4, out)
        for l in runs:
            self.assertIn("--device-read-bps /dev/nvme0n1:100000000 ", l)
            self.assertIn("--device-write-bps /dev/nvme0n1:100000000 ", l)
        self.assertFalse((self.packed_dir("head") / "MANIFEST.sha256").exists())

    def test_it_refuses_a_folder_without_packed_rows(self):
        code, out = self.pack(extra={"ENGRAM_PACKED": "0"})
        self.assertNotEqual(code, 0, out)
        self.assertIn("ENGRAM_PACKED", out)
        self.assertEqual(self.packer_runs(), [])



def _gnu_tar_and_nc():
    try:
        tar = subprocess.run(["tar", "--version"], capture_output=True, text=True).stdout
        nc = subprocess.run(["nc", "-h"], capture_output=True, text=True)
        return "GNU tar" in tar and "-N" in (nc.stdout + nc.stderr)
    except OSError:
        return False


# the worker's tar (SIM_HOST set by the fake ssh): the real one, then, when $SIM/corrupt names a path, one byte of that
# file under the extract's -C folder flipped (a copy that arrived damaged)
CORRUPTING_TAR = r"""#!BASH
REAL "$@"; rc=$?
if [[ -n "${SIM_HOST:-}" && -s "$SIM/corrupt" && " $* " == *" -xpf "* ]]; then
  dir=""; prev=""; for a in "$@"; do [[ "$prev" != -C ]] || dir=$a; prev=$a; done
  f="$dir/$(cat "$SIM/corrupt")"
  if [[ -f "$f" ]]; then printf 'X' | dd of="$f" bs=1 seek=0 conv=notrunc 2>/dev/null; fi
fi
exit $rc
"""


@unittest.skipUnless(sys.platform.startswith("linux") and _gnu_tar_and_nc(), "needs Linux, GNU tar and OpenBSD nc")
class FabricCopy(ScriptTest):
    """scripts/fabric-copy.sh: files from this Spark to a worker as tar | nc streams over the worker's fabric addresses
    (here two on loopback), ssh only for control. Every listed path lands at the same path under the destination,
    links followed (a regular file there); what the worker holds at this Spark's size is not sent again; a copy cut
    short is sent again whole; every sent file is checked by SHA-256 on both ends, and a difference fails the copy."""
    def setUp(self):
        super().setUp()
        self.src = self.tmp / "src"
        (self.src / "sub").mkdir(parents=True)
        (self.src / "chain").mkdir()
        (self.src / "a.bin").write_bytes(os.urandom(3 << 20))
        (self.src / "sub" / "b.bin").write_bytes(os.urandom(300 << 10))
        (self.src / "small.txt").write_text("a small file\n")
        (self.src / "link.bin").symlink_to("sub/b.bin")
        (self.src / "chain" / "x").symlink_to("../link.bin")
        self.paths = ["a.bin", "sub/b.bin", "small.txt", "link.bin", "chain/x"]
        self.dest = self.sim / "home" / "w1" / "dest"
        tar = shutil.which("tar")
        (self.tmp / "bin" / "tar").write_text(CORRUPTING_TAR.replace("BASH", BASH).replace("REAL", tar))
        (self.tmp / "bin" / "tar").chmod(0o755)
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        if self.port > 65000:
            self.port = 41000

    def copy(self, paths=None):
        r = subprocess.run([BASH, "scripts/fabric-copy.sh", "-n", "3", "-p", str(self.port), "w1", "127.0.0.1,127.0.0.1",
                            str(self.src), "$HOME/dest"], cwd=self.repo, env=self.env, capture_output=True, text=True,
                           timeout=120, input="".join(p + "\n" for p in (paths or self.paths)))
        return r.returncode, r.stdout + r.stderr

    def assertSameTree(self):
        for p in self.paths:
            there = self.dest / p
            self.assertTrue(there.is_file() and not there.is_symlink(), p)
            self.assertEqual(there.read_bytes(), (self.src / p).read_bytes(), p)

    def test_every_path_arrives_whole_links_followed(self):
        code, out = self.copy()
        self.assertEqual(code, 0, out)
        self.assertSameTree()
        self.assertIn("5 to send", out)
        self.assertIn("verified: 5 files by SHA-256 on both ends", out)

    def test_what_is_there_at_its_size_is_not_sent_again(self):
        self.assertEqual(self.copy()[0], 0)
        code, out = self.copy()
        self.assertEqual(code, 0, out)
        self.assertIn("nothing to send", out)

    def test_a_copy_cut_short_is_sent_again_whole(self):
        self.assertEqual(self.copy()[0], 0)
        with open(self.dest / "a.bin", "r+b") as f:
            f.truncate(1 << 20)
        code, out = self.copy()
        self.assertEqual(code, 0, out)
        self.assertIn("1 to send", out)
        self.assertSameTree()

    def test_a_file_that_arrives_damaged_fails_the_copy(self):
        (self.sim / "corrupt").write_text("a.bin")
        code, out = self.copy()
        self.assertNotEqual(code, 0, out)
        self.assertIn("SHA-256 differs", out)
        self.assertIn("a.bin", out)

    def test_a_path_missing_here_is_named(self):
        code, out = self.copy(self.paths + ["no/such.file"])
        self.assertNotEqual(code, 0, out)
        self.assertIn("no/such.file", out)
        self.assertFalse(self.dest.exists())


class ImageTag(ScriptTest):
    """The image's tag is the same in every locale. A folder without git (a release archive) hashes its files in a sorted
    list; the launcher runs in its user's locale and other tools read the tag in an empty environment (the C locale), so
    the order must be bytewise in both. The stand-in sort orders names unlike C unless the C locale is asked for."""

    def test_the_tag_does_not_depend_on_the_locale(self):
        real = shutil.which("sort", path=os.environ.get("PATH", "/usr/bin:/bin"))
        fake = self.tmp / "bin" / "sort"
        fake.write_text("#!/bin/sh\n"
                        f'case "${{LC_ALL:-${{LC_COLLATE:-}}}}" in C|POSIX) exec {real} "$@" ;; esac\n'
                        f'exec {real} -r "$@"\n')
        fake.chmod(0o755)
        tags = []
        for loc in ({"LANG": "en_US.UTF-8"}, {"LC_ALL": "C"}):
            code, out = self.bash('echo "tag=$IMAGE"', loc)
            self.assertEqual(code, 0, out)
            (tag,) = [l[4:] for l in out.splitlines() if l.startswith("tag=")]
            self.assertRegex(tag, r"^tensorfold-dsv41:native-[0-9a-f]{12}$")
            tags.append(tag)
        self.assertEqual(tags[0], tags[1])


class OneFabric(ScriptTest):
    """Every pair of Sparks shares a RoCE subnet (a switch, or at three Sparks a triangle of cables): there is no cabled-ring
    mode. pair_links names each pair without a common subnet and describes no ring; no script or guide names a ring launcher."""
    # four Sparks cabled 0-1-2-3-0: each pair of neighbours on a subnet of its own; ranks 0 and 2, 1 and 3 unlinked
    RING = "".join(f"{r} default eth0\n{r} enpA mlx5_0 10.1.{r}.{r + 1}/24 3\n{r} enpB mlx5_1 10.1.{(r - 1) % 4}.{r + 1}/24 3\n"
                   for r in range(4))

    def test_a_ring_of_cables_is_two_pairs_without_a_link(self):
        code, out = self.bash(f"pair_links 0 1 2 3 <<'EOF'\n{self.RING}EOF")
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(l for l in out.splitlines() if l.startswith("missing ")), ["missing 0 2", "missing 1 3"])
        self.assertEqual([l for l in out.splitlines() if l.split()[:1] in (["ring"], ["toward"])], [])

    def test_no_script_or_guide_names_a_ring_launcher(self):
        self.assertFalse((ROOT / "start-switchless.sh").exists())
        for f in ("start.sh", "stop.sh", "setup.sh", "scripts/nodes.sh", "scripts/config.sh", "scripts/prepare.sh",
                  "scripts/local.sh.example", "README.md", "NETWORK.md"):
            text = (ROOT / f).read_text()
            for gone in ("switchless", "FABRIC_EXPECT", "TOWARD", "FABRIC=ring", "$FABRIC"):
                self.assertNotIn(gone, text, (f, gone))


class KitFollowsTheWindow(ScriptTest):
    """A kit's RoPE tables are cut for one YaRN factor (its rope.json) and the engine refuses any other, so the default kit
    follows TF_DS_YARN_FACTOR: the 1M window's kit unset or at 16, the 2M window's at 32, the 4M window's at 64. Three
    commits of the kit that differ only in those tables. A KIT_REF that is set wins."""
    KITS = {"16": "b88aca850cb2e473069a4cd5d225e7a4b188fe8b", "32": "30cb665fe6cd15ed113c198f4ba34b5da944923d",
            "64": "3abe1d5ecb7a718b1ef16ee425de5d9ab5542c17"}

    def kit(self, **env):
        code, out = self.bash('echo "kit=$KIT_REF"', env)
        self.assertEqual(code, 0, out)
        (k,) = [l[4:] for l in out.splitlines() if l.startswith("kit=")]
        return k

    def test_the_default_kit_is_cut_for_the_window(self):
        self.assertEqual(self.kit(), self.KITS["16"])
        for factor, ref in self.KITS.items():
            self.assertEqual(self.kit(TF_DS_YARN_FACTOR=factor), ref, factor)

    def test_a_kit_that_is_set_wins(self):
        self.assertEqual(self.kit(TF_DS_YARN_FACTOR="64", KIT_REF="ab" * 20), "ab" * 20)


class SetupTags(ScriptTest):
    """The scripts' log lines under ./setup.sh: tagged with the setup's step (SETUP_TAG) instead of the script's name, and
    detail lines (internals) written to the setup's log file (SETUP_LOG) instead of the terminal; a script run on its own
    keeps its own name and prints its details."""

    def test_a_line_carries_the_setups_step(self):
        code, out = self.bash('log "Model files: complete"; warn "slow disk"', {"SETUP_TAG": "4/5 data"})
        self.assertEqual(code, 0, out)
        self.assertIn("[4/5 data] Model files: complete", out)
        self.assertIn("[4/5 data] WARN: slow disk", out)
        self.assertNotIn("[bash]", out)

    def test_details_go_to_the_setups_log(self):
        logf = self.tmp / "setup.log"
        code, out = self.bash('detail "ring 8 GiB registered"; log "copying"', {"SETUP_TAG": "4/5 data", "SETUP_LOG": str(logf)})
        self.assertEqual(code, 0, out)
        self.assertNotIn("ring 8 GiB", out)
        self.assertIn("[4/5 data] copying", out)
        self.assertIn("[4/5 data] ring 8 GiB registered", logf.read_text())

    def test_a_script_on_its_own_prints_its_details(self):
        code, out = self.bash('detail "ring 8 GiB registered"')
        self.assertEqual(code, 0, out)
        self.assertIn("ring 8 GiB registered", out)


class SetupCheckRun(ScriptTest):
    """./setup.sh check (steps 1 and 2) with the stand-in ssh: every line carries its step, the Sparks by their ssh names,
    and the run's log file (LOG_DIR/setup-*.log) holds what was shown."""

    def setUp(self):
        super().setUp()
        shutil.copy2(ROOT / "setup.sh", self.repo / "setup.sh")
        shutil.copy2(self.repo / "scripts" / "local.sh.example", self.repo / "scripts" / "local.sh")   # workers: the env's w1-w3
        sudo = self.tmp / "bin" / "sudo"
        sudo.write_text("#!/bin/sh\n[ \"$1\" = -n ] && shift\nexec \"$@\"\n"); sudo.chmod(0o755)
        ssh = self.tmp / "bin" / "ssh"                          # the setup's own ssh calls also pass -n and -t
        ssh.write_text(ssh.read_text().replace('opts=""; while [[ "$1" == -o ]]; do opts+="$2 "; shift 2; done',
            'opts=""; while [[ "$1" == -[ont] ]]; do if [[ "$1" == -o ]]; then opts+="$2 "; shift 2; else shift; fi; done'))

    def test_steps_one_and_two_speak_plainly_and_are_logged(self):
        code, out = self.run_script(["setup.sh", "check"], {"LOG_DIR": str(self.tmp / "logs")})
        self.assertEqual(code, 0, out)
        lines = [l for l in out.splitlines() if l.strip()]
        self.assertTrue(all(l.startswith(("[1/5 config]", "[2/5 access]")) for l in lines), out)
        for want in ("[1/5 config] Settings from scripts/local.sh", "This Spark and three others: w1, w2, w3",
                     "[2/5 access]   this Spark: OK", "[2/5 access]   w2: OK", "[2/5 access] Settings and access: ready"):
            self.assertIn(want, out)
        for gone in ("rank", "[setup.sh]", "node"):
            self.assertNotIn(gone, out)
        (logf,) = (self.tmp / "logs").glob("setup-*.log")
        self.assertIn("[2/5 access] Settings and access: ready", logf.read_text())


HELPER = "/usr/local/sbin/dsv41-root"
RULE = "/etc/sudoers.d/90-dsv41-start"
# root's install, writing under $SIMROOT: what the setup's root script installs, and with which mode (-o/-g need root)
FAKE_ROOT_INSTALL = """#!/bin/sh
mode=; while [ $# -gt 2 ]; do case "$1" in -o|-g) shift 2 ;; -m) mode=$2; shift 2 ;; *) break ;; esac; done
dst="$SIMROOT$2"; mkdir -p "$(dirname "$dst")"; cp "$1" "$dst"; [ -z "$mode" ] || chmod "$mode" "$dst"
echo "install $mode $2" >> "$SIMROOT.log"
"""


class RootHelper(unittest.TestCase):
    """scripts/dsv41-root, the setup's one root helper for a Spark without passwordless sudo: the sudo rule lets the user
    run this file only, so it takes four commands (check, drop, clock <whole MHz>, uninstall) and refuses anything else
    before it does anything; run without root it stops there."""
    def run_helper(self, *args):
        r = subprocess.run(["/bin/sh", str(ROOT / "scripts" / "dsv41-root"), *args], capture_output=True, text=True, timeout=10)
        return r.returncode, r.stdout + r.stderr

    def test_anything_but_its_four_commands_is_refused_first(self):
        for args in ([], ["true"], ["sh"], ["clock"], ["clock", "2200;id"], ["clock", "22a0"], ["clock", "-1"],
                     ["clock", "123456"], ["clock", "2200", "x"], ["drop", "x"], ["check", "x"], ["uninstall", "/"]):
            code, out = self.run_helper(*args)
            self.assertEqual(code, 2, (args, out))
            self.assertIn("usage", out)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "run as a user")
    def test_its_four_commands_need_root(self):
        for args in (["check"], ["drop"], ["clock", "2200"], ["uninstall"]):
            code, out = self.run_helper(*args)
            self.assertEqual(code, 3, (args, out))
            self.assertIn("run it with sudo", out)

    def test_it_runs_on_a_fixed_path_and_removes_only_its_own_files(self):
        text = (ROOT / "scripts" / "dsv41-root").read_text()
        self.assertIn("PATH=/usr/sbin:/usr/bin:/sbin:/bin", text)
        self.assertIn(f"RULE={RULE}", text)
        self.assertIn(f"SELF={HELPER}", text)
        self.assertIn('rm -f "$RULE" "$SELF"', text)
        self.assertTrue(os.access(ROOT / "scripts" / "dsv41-root", os.X_OK))


class SudoRuleScript(unittest.TestCase):
    """setup-lib.sh's sudo_rule_script: the script the setup runs as root on a Spark without passwordless sudo. It writes
    the rule (the sudo user may run the helper, nothing else) to a temporary file, has visudo check that file, and only
    then installs the helper and the rule, root-owned, under their own names; a rule visudo refuses, or a user name a rule
    cannot hold safely, installs nothing; no other sudoers file is touched."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tf-sudo-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bin = self.tmp / "bin"; self.bin.mkdir()
        (self.bin / "install").write_text(FAKE_ROOT_INSTALL); (self.bin / "install").chmod(0o755)
        self.root = self.tmp / "root"

    def run_rule_script(self, user, visudo_ok=True):
        if not visudo_ok:
            (self.bin / "visudo").write_text("#!/bin/sh\necho 'syntax error' >&2\nexit 1\n"); (self.bin / "visudo").chmod(0o755)
        r = subprocess.run([BASH, "-c", f"set -euo pipefail\nsource '{ROOT}/scripts/setup-lib.sh'\n"
                                         f"sudo_rule_script '{ROOT}/scripts/dsv41-root' > '{self.tmp}/root.sh'"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        env = {"PATH": f"{self.bin}:/usr/sbin:/usr/bin:/sbin:/bin", "SIMROOT": str(self.root), "SUDO_USER": user}
        r = subprocess.run([BASH, str(self.tmp / "root.sh")], env=env, capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout + r.stderr

    @unittest.skipIf(shutil.which("visudo", path="/usr/sbin:/usr/bin:/sbin:/bin") is None, "needs visudo")
    def test_it_installs_the_helper_and_a_rule_for_only_it(self):
        code, out = self.run_rule_script("alice")
        self.assertEqual(code, 0, out)
        helper = self.root / HELPER.lstrip("/"); rule = self.root / RULE.lstrip("/")
        self.assertEqual(helper.read_bytes(), (ROOT / "scripts" / "dsv41-root").read_bytes())
        self.assertEqual(helper.stat().st_mode & 0o7777, 0o755)
        self.assertEqual(rule.read_text(), f"alice ALL=(root) NOPASSWD: {HELPER}\n")
        self.assertEqual(rule.stat().st_mode & 0o7777, 0o440)
        self.assertEqual((self.tmp / "root.log").read_text().splitlines(), [f"install 0755 {HELPER}", f"install 0440 {RULE}"])

    def test_a_rule_visudo_refuses_installs_nothing(self):
        code, out = self.run_rule_script("alice", visudo_ok=False)
        self.assertNotEqual(code, 0, out)
        self.assertIn("visudo", out)
        self.assertFalse(self.root.exists(), list(self.root.rglob("*")) if self.root.exists() else None)

    def test_a_user_name_a_rule_cannot_hold_installs_nothing(self):
        for user in ("", "bad user", "a,b", "a:b", "ALL", "-x", "root=x", "%sudo"):
            code, out = self.run_rule_script(user)
            self.assertNotEqual(code, 0, (user, out))
            self.assertFalse(self.root.exists(), user)

    @unittest.skipIf(shutil.which("visudo", path="/usr/sbin:/usr/bin:/sbin:/bin") is None, "needs visudo")
    def test_no_other_sudoers_file_is_touched(self):
        d = self.root / "etc" / "sudoers.d"; d.mkdir(parents=True)
        (d / "90-dsv41").write_text("alice ALL=(ALL) NOPASSWD: ALL\n"); (d / "README").write_text("x\n")
        code, out = self.run_rule_script("alice")
        self.assertEqual(code, 0, out)
        self.assertEqual((d / "90-dsv41").read_text(), "alice ALL=(ALL) NOPASSWD: ALL\n")
        self.assertEqual((d / "README").read_text(), "x\n")
        self.assertEqual(sorted(p.name for p in d.iterdir()), ["90-dsv41", "90-dsv41-start", "README"])


FAKE_SETUP_SUDO = r"""#!BASH
# fake sudo for the setup's access step. Each Spark under $SIM/hosts/<host>/: no_sudo (no passwordless sudo), password
# (what sudo -S takes), root/ (what root wrote there through the fake install: the helper and the rule)
h=${SIM_HOST:-head}; d="$SIM/hosts/$h"; mkdir -p "$d"
echo "$h sudo $*" >> "$SIM/sudo.log"
if [[ $1 == -S ]]; then
  shift; [[ $1 == -p ]] && shift 2
  IFS= read -r pw || pw=
  [[ -f $d/password && $pw == "$(cat "$d/password")" ]] || { echo "Sorry, try again." >&2; exit 1; }
  echo "$h root-session" >> "$SIM/sudo.log"
  SUDO_USER=me SIMROOT="$d/root" PATH="$SIM/rootbin:$PATH" exec "$@"
fi
[[ $1 == -n ]] && shift
if [[ $* == "cat /etc/sudoers.d/90-dsv41" || $* == "rm -f /etc/sudoers.d/90-dsv41" ]]; then
  [[ -e $d/no_sudo ]] && exit 1
  [[ $1 == cat ]] && exec cat "$d/root/etc/sudoers.d/90-dsv41"
  rm -f "$d/root/etc/sudoers.d/90-dsv41"; echo "$h removed 90-dsv41" >> "$SIM/sudo.log"; exit 0
fi
if [[ $1 == /usr/local/sbin/dsv41-root ]]; then
  [[ -f $d/root/usr/local/sbin/dsv41-root && -f $d/root/etc/sudoers.d/90-dsv41-start ]] || exit 1
  case $2 in check) sha256sum < "$d/root/usr/local/sbin/dsv41-root" | cut -d' ' -f1 ;;
             uninstall) rm -f "$d/root/usr/local/sbin/dsv41-root" "$d/root/etc/sudoers.d/90-dsv41-start" ;; esac
  exit 0
fi
[[ -e $d/no_sudo ]] && exit 1
exec "$@"
"""


class SetupSudo(SetupCheckRun):
    """The setup's access step on a Spark without passwordless sudo (D111): ./setup.sh check names it and what the setup
    will do; any other run asks that Spark's password once, on a terminal (never shown, never on a command line or in the
    log), and installs a sudo rule for only the start's two root commands, through the helper; a current rule needs no
    password; a changed helper is installed again; ./setup.sh remove takes the rule away and says what stays."""

    def setUp(self):
        super().setUp()
        sudo = self.tmp / "bin" / "sudo"
        sudo.write_text(FAKE_SETUP_SUDO.replace("BASH", BASH)); sudo.chmod(0o755)
        rb = self.sim / "rootbin"; rb.mkdir()
        (rb / "install").write_text(FAKE_ROOT_INSTALL); (rb / "install").chmod(0o755)
        self.env["LOG_DIR"] = str(self.tmp / "logs")

    def no_sudo(self, host, password=None):
        d = self.sim / "hosts" / host; d.mkdir(parents=True, exist_ok=True)
        (d / "no_sudo").write_text("")
        if password is not None:
            (d / "password").write_text(password)

    def installed(self, host, helper_text=None):
        r = self.sim / "hosts" / host / "root"
        h = r / HELPER.lstrip("/"); h.parent.mkdir(parents=True, exist_ok=True)
        h.write_bytes(helper_text.encode() if helper_text is not None else (ROOT / "scripts" / "dsv41-root").read_bytes())
        u = r / RULE.lstrip("/"); u.parent.mkdir(parents=True, exist_ok=True)
        u.write_text(f"me ALL=(root) NOPASSWD: {HELPER}\n")

    def run_tty(self, args, lines, timeout=60):
        import pty
        m, s = pty.openpty()
        p = subprocess.Popen([BASH, *args], cwd=self.repo, env=self.env, stdin=s, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        os.close(s)
        os.write(m, "".join(l + "\n" for l in lines).encode())
        try:
            out, _ = p.communicate(timeout=timeout)
        finally:
            os.close(m)
        return p.returncode, out

    def logged(self):
        return "".join(f.read_text() for f in (self.tmp / "logs").glob("setup-*.log"))

    def sim_text(self, name):
        f = self.sim / name
        return f.read_text() if f.exists() else ""

    def test_check_names_the_spark_and_what_the_setup_will_do(self):
        self.no_sudo("w2", "pw")
        code, out = self.run_script(["setup.sh", "check"])
        self.assertEqual(code, 0, out)
        self.assertIn("[2/5 access]   w2: no passwordless sudo", out)
        self.assertIn("asks its password once", out)
        self.assertNotIn("Password for sudo", out)
        self.assertFalse((self.sim / "hosts" / "w2" / "root").exists())

    def test_access_installs_the_narrow_rule_with_the_password_typed_once(self):
        self.no_sudo("w2", "s3cret-pw")
        code, out = self.run_tty(["setup.sh", "access"], ["s3cret-pw"])
        self.assertEqual(code, 0, out)
        r = self.sim / "hosts" / "w2" / "root"
        self.assertEqual((r / HELPER.lstrip("/")).read_bytes(), (ROOT / "scripts" / "dsv41-root").read_bytes())
        self.assertEqual((r / RULE.lstrip("/")).read_text(), f"me ALL=(root) NOPASSWD: {HELPER}\n")
        self.assertIn("w2: the setup's sudo rule installed", out)
        self.assertEqual(out.count("Password for sudo on w2"), 1, out)
        for where in (out, self.logged(), self.sim_text("ssh.log"), self.sim_text("sudo.log")):
            self.assertNotIn("s3cret-pw", where)
        for h in ("head", "w1", "w3"):
            self.assertFalse((self.sim / "hosts" / h / "root").exists(), h)

    def test_without_a_terminal_it_stops_and_says_how(self):
        self.no_sudo("w2", "pw")
        code, out = self.run_script(["setup.sh", "access"])
        self.assertNotEqual(code, 0, out)
        self.assertIn("w2", out)
        self.assertIn("in a terminal", out)
        self.assertFalse((self.sim / "hosts" / "w2" / "root").exists())

    def test_a_wrong_password_is_asked_again_then_the_step_stops(self):
        self.no_sudo("w2", "right")
        code, out = self.run_tty(["setup.sh", "access"], ["no1", "no2", "no3"])
        self.assertNotEqual(code, 0, out)
        self.assertEqual(out.count("Password for sudo on w2"), 3, out)
        self.assertFalse((self.sim / "hosts" / "w2" / "root").exists())

    def test_a_current_rule_needs_no_password(self):
        self.no_sudo("w2", "pw"); self.installed("w2")
        code, out = self.run_script(["setup.sh", "access"])
        self.assertEqual(code, 0, out)
        self.assertIn("w2: OK (the setup's sudo rule)", out)
        self.assertNotIn("root-session", self.sim_text("sudo.log"))

    def test_a_changed_helper_is_installed_again(self):
        self.no_sudo("w2", "pw"); self.installed("w2", helper_text="#!/bin/sh\n# an older helper\n")
        code, out = self.run_tty(["setup.sh", "access"], ["pw"])
        self.assertEqual(code, 0, out)
        self.assertEqual((self.sim / "hosts" / "w2" / "root" / HELPER.lstrip("/")).read_bytes(),
                         (ROOT / "scripts" / "dsv41-root").read_bytes())

    def test_sparks_with_passwordless_sudo_get_nothing_installed(self):
        code, out = self.run_script(["setup.sh", "access"])
        self.assertEqual(code, 0, out)
        for h in ("head", "w1", "w2", "w3"):
            self.assertFalse((self.sim / "hosts" / h / "root").exists(), h)
        self.assertNotIn("root-session", self.sim_text("sudo.log"))

    def old_full_rule(self, host, text="me ALL=(ALL) NOPASSWD: ALL\n"):
        f = self.sim / "hosts" / host / "root" / "etc" / "sudoers.d" / "90-dsv41"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        return f

    def test_remove_takes_away_the_earlier_setups_full_rule(self):
        # the setup before the narrow rule wrote "<user> ALL=(ALL) NOPASSWD: ALL" to /etc/sudoers.d/90-dsv41
        f = self.old_full_rule("w1")
        code, out = self.run_script(["setup.sh", "remove"])
        self.assertEqual(code, 0, out)
        self.assertFalse(f.exists())
        self.assertIn("w1: the earlier setup's full sudo rule removed (/etc/sudoers.d/90-dsv41)", out)

    def test_remove_leaves_a_90_dsv41_it_did_not_write(self):
        f = self.old_full_rule("w1", "me ALL=(ALL) NOPASSWD: /usr/bin/true\n")
        code, out = self.run_script(["setup.sh", "remove"])
        self.assertEqual(code, 0, out)
        self.assertTrue(f.exists())
        self.assertIn("w1: /etc/sudoers.d/90-dsv41 holds something the setup did not write: left as it is", out)

    def test_access_names_the_earlier_setups_full_rule(self):
        self.old_full_rule("w2")
        code, out = self.run_script(["setup.sh", "check"])
        self.assertEqual(code, 0, out)
        self.assertIn("w2: OK (the earlier setup's full sudo rule, /etc/sudoers.d/90-dsv41: ./setup.sh remove takes it away)", out)

    def test_remove_takes_the_rule_away_and_says_what_stays(self):
        self.no_sudo("w2", "pw"); self.installed("w2")
        code, out = self.run_script(["setup.sh", "remove"])
        self.assertEqual(code, 0, out)
        r = self.sim / "hosts" / "w2" / "root"
        self.assertFalse((r / HELPER.lstrip("/")).exists())
        self.assertFalse((r / RULE.lstrip("/")).exists())
        self.assertIn("w2: the setup's sudo rule removed", out)
        for want in ("NETWORK.md", "roce-pfc", "51-cx7-roce-static.yaml"):
            self.assertIn(want, out)


class RocePfcUnit(unittest.TestCase):
    """network/roce-pfc.service, the flow control the setup installs on each Spark: each ExecStart line run as systemd runs
    it (a stand-in dcb, sleep made instant). Setting PFC can restart the port's link for a few seconds, and both rails are
    functions of one port: a dcb command that lands while it settles fails once ("Attribute write: Protocol error", seen
    on a Spark right after netplan apply), so each is retried for a while; one that keeps failing fails the unit."""
    DCB = """#!/bin/sh
key=$(echo "$*" | tr ' :' '__'); n=$(cat "$STATE/$key" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$STATE/$key"
echo "dcb $*" >> "$STATE/calls"
[ "$n" -gt "$FAILS" ] || { echo "Attribute write: Protocol error" >&2; exit 1; }
"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tf-pfc-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        b = self.tmp / "bin"; b.mkdir()
        (b / "dcb").write_text(self.DCB); (b / "dcb").chmod(0o755)
        (b / "sleep").write_text("#!/bin/sh\nexit 0\n"); (b / "sleep").chmod(0o755)
        self.bin = b
        text = (ROOT / "network" / "roce-pfc.service").read_text()
        # systemd's own unescaping of a command line: $$ is a literal $ (no other specifier is used in this unit)
        self.lines = [l.split("=", 1)[1].replace("$$", "$").replace("/usr/sbin/dcb", str(b / "dcb"))
                      for l in text.splitlines() if l.startswith("ExecStart=")]

    def run_unit(self, fails):
        (self.tmp / "state").mkdir(exist_ok=True)
        env = {"PATH": f"{self.bin}:/usr/bin:/bin", "STATE": str(self.tmp / "state"), "FAILS": str(fails)}
        codes = [subprocess.run(["/bin/sh", "-c", l], env=env, capture_output=True, text=True, timeout=30).returncode
                 for l in self.lines]
        calls = (self.tmp / "state" / "calls").read_text().splitlines()
        return codes, calls

    def test_it_sets_both_rails(self):
        joined = "\n".join(self.lines)
        for dev in ("enp1s0f0np0", "enP2p1s0f0np0"):
            self.assertIn(f"app replace dev {dev} dscp-prio 0:3 26:3", joined)
            self.assertIn(f"pfc set dev {dev} prio-pfc 3:on", joined)

    def test_a_command_that_fails_while_the_port_settles_is_tried_again(self):
        codes, calls = self.run_unit(fails=2)
        self.assertEqual(codes, [0, 0, 0, 0], calls)
        self.assertEqual(len(calls), 4 * 3, calls)                     # each: two failures, then done

    def test_a_command_that_keeps_failing_fails_the_unit(self):
        codes, calls = self.run_unit(fails=1000)
        self.assertTrue(all(c != 0 for c in codes), codes)
        self.assertLessEqual(len(calls), 4 * 30, len(calls))           # bounded: it gives up


class SetupWithoutConfig(ScriptTest):
    """./setup.sh before scripts/local.sh exists: it says what the file is, how to make it from the example, what to set in
    it (the example's own worker lines shown, placeholders called so), what else must be ready, and where to read more."""

    def setUp(self):
        super().setUp()
        shutil.copy2(ROOT / "setup.sh", self.repo / "setup.sh")
        (self.repo / "scripts" / "local.sh").unlink(missing_ok=True)

    def test_it_explains_the_settings_file_and_stops(self):
        code, out = self.run_script(["setup.sh"])
        self.assertEqual(code, 1, out)
        for want in ("scripts/local.sh", "cp scripts/local.sh.example scripts/local.sh", "WORKER1=user@192.0.2.2",
                     "WORKER2=user@192.0.2.3", "WORKER3=user@192.0.2.4", "placeholders", "Run ./setup.sh again",
                     "ssh-copy-id", "NETWORK.md", "README.md"):
            self.assertIn(want, out)
        self.assertGreaterEqual(len([l for l in out.splitlines() if l.strip()]), 10, out)
        self.assertFalse((self.repo / "scripts" / "local.sh").exists())          # it explains; the copy is the user's

    def test_an_example_that_names_real_sparks_is_ready_to_copy(self):
        ex = self.repo / "scripts" / "local.sh.example"
        ex.write_text(ex.read_text().replace("WORKER1=user@192.0.2.2 ", "WORKER1=me@10.9.0.2 ")
                      .replace("WORKER2=user@192.0.2.3 ", "WORKER2=me@10.9.0.3 ").replace("WORKER3=user@192.0.2.4 ", "WORKER3=me@10.9.0.4 "))
        code, out = self.run_script(["setup.sh"])
        self.assertEqual(code, 1, out)
        for want in ("WORKER1=me@10.9.0.2", "WORKER2=me@10.9.0.3", "WORKER3=me@10.9.0.4", "copied as it is"):
            self.assertIn(want, out)
        self.assertNotIn("placeholders", out)


class SetupHelpers(unittest.TestCase):
    """scripts/setup-lib.sh, the setup's own helpers: the copy's routes (one run for every worker's share: a path every
    worker needs goes to all, another only to the workers that list it), the keys the setup makes (mode 600, never
    overwritten), and the netplan files besides the setup's own that configure a rail: an earlier fabric file of only the
    ConnectX-7 ports (NVIDIA's two-Spark playbook writes one) is moved aside, any other stops the setup (it never edits a
    user's file)."""

    # the clock the helpers read, moved to just past 1,794,375,000 s: a time printed with awk's default %.6g there reads
    # 1.79438e+09, 5,000 s late, so a helper that does that never prints its periodic lines (seen 11-12)
    DATE = """#!/bin/sh
real=$(/bin/date +%s.%N)
case "$1" in
  +%s.%N) awk -v r="$real" -v s="$DATE_SHIFT" 'BEGIN { printf "%.9f\\n", r + s }' ;;
  +%s) awk -v r="$real" -v s="$DATE_SHIFT" 'BEGIN { printf "%d\\n", r + s }' ;;
  *) exec /bin/date "$@" ;;
esac
"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tf-setuplib-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bin = self.tmp / "clock"; self.bin.mkdir()
        (self.bin / "date").write_text(self.DATE); (self.bin / "date").chmod(0o755)
        self.shift = str(1794375000 - int(time.time()))

    def path(self):
        return f"{self.bin}:" + os.environ.get("PATH", "/usr/bin:/bin")

    def sh(self, body):
        r = subprocess.run([BASH, "-c", f"set -euo pipefail\nsource '{ROOT}/scripts/setup-lib.sh'\n{body}\n"], capture_output=True,
                           text=True, cwd=self.tmp, env={"PATH": self.path(), "HOME": str(self.tmp), "DATE_SHIFT": self.shift})
        return r.returncode, r.stdout, r.stderr

    def test_the_workers_a_step_goes_to_read_as_words(self):
        # seen in a run: "Engine image to the other three Sparksover the fabric"
        code, out, err = self.sh('for a in "w1 w2 w3" "w2" "w1 w3"; do printf "[%s]\\n" "$(sparks_phrase 3 $a)"; done')
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), ["[the other three Sparks]", "[w2]", "[w1 and w3]"])

    def test_the_image_line_names_its_workers_with_the_helper(self):
        line = next(l for l in (ROOT / "scripts" / "setup-data.sh").read_text().splitlines() if 'log "Engine image to ' in l)
        self.assertIn('$(sparks_phrase 3 ', line)
        self.assertIn(') over the fabric"', line)

    def test_routes_send_a_shared_path_to_all_and_an_own_path_to_its_worker(self):
        for i, paths in enumerate((["kit/a", "data/r1.bin"], ["kit/a", "data/r2.bin"], ["kit/a", "data/r3.bin", "data/x"]), 1):
            (self.tmp / f"l{i}").write_text("".join(p + "\n" for p in paths))
        code, out, err = self.sh("fx_routes l1 l2 l3")
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(out.splitlines()), sorted(["kit/a", "1\tdata/r1.bin", "2\tdata/r2.bin", "3\tdata/r3.bin", "3\tdata/x"]))

    def test_routes_name_every_worker_that_lists_a_path(self):
        (self.tmp / "l1").write_text("p\nq\n"); (self.tmp / "l2").write_text("q\n"); (self.tmp / "l3").write_text("p\n")
        code, out, err = self.sh("fx_routes l1 l2 l3")
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(out.splitlines()), ["1,2\tq", "1,3\tp"])

    def test_a_key_file_is_made_once_owner_only_and_never_overwritten(self):
        code, out, err = self.sh("make_key_file keys/api admin; stat -c %a keys/api; cat keys/api")
        self.assertEqual(code, 0, err)
        mode, line = out.split("\n")[0], out.split("\n")[1]
        self.assertEqual(mode, "600")
        self.assertRegex(line, r"^admin: [0-9a-f]{64}$")
        code, out2, err = self.sh("make_key_file keys/api admin; cat keys/api")
        self.assertEqual(code, 0, err)
        self.assertEqual(out2.strip(), line)                      # the existing key is kept

    # a stand-in for `docker build`: prints $FAKE_BUILD's lines (a line "sleep N" waits N seconds), then exits $FAKE_RC
    BUILD = """#!/bin/sh
echo "docker $*" >> "$FAKE_ARGS"
while IFS= read -r line; do
  case "$line" in sleep\\ *) sleep "${line#sleep }" ;; *) printf '%s\\n' "$line" ;; esac
done < "$FAKE_BUILD"
exit "${FAKE_RC:-0}"
"""

    def build(self, lines, rc=0, beat="60"):
        b = self.tmp / "bin"; b.mkdir(exist_ok=True)
        (b / "docker").write_text(self.BUILD); (b / "docker").chmod(0o755)
        (self.tmp / "build.txt").write_text("".join(l + "\n" for l in lines))
        r = subprocess.run([BASH, "-c", f"set -euo pipefail\nsource '{ROOT}/scripts/setup-lib.sh'\n"
                            "docker_build_steps build.log -t img:tag ctx"], capture_output=True, text=True, cwd=self.tmp,
                           env={"PATH": f"{b}:" + self.path(), "HOME": str(self.tmp), "DATE_SHIFT": self.shift,
                                "FAKE_BUILD": str(self.tmp / "build.txt"), "FAKE_ARGS": str(self.tmp / "args"),
                                "FAKE_RC": str(rc), "BUILD_POLL_S": "0.1", "BUILD_HEARTBEAT_S": beat})
        return r.returncode, r.stdout + r.stderr

    def test_the_build_shows_one_line_a_step_and_keeps_the_rest_in_its_log(self):
        lines = ["#1 [internal] load build definition", "#5 [ 1/10] FROM nvcr.io/base:1", "#5 DONE 0.1s",
                 "#12 [ 8/10] RUN cd /tensorfold && /opt/zig/zig build -Dcpu=gb10 install native", "sleep 0.5",
                 "#12 55.62 ./zig/kernels/cuda/x.cu(153): warning #177-D: variable RW declared but never referenced",
                 "#13 [10/10] RUN ldconfig", "#14 naming to img:tag done"]
        code, out = self.build(lines)
        self.assertEqual(code, 0, out)
        self.assertIn("step 1/10", out)
        self.assertIn("step 8/10  compiling the engine (Zig and CUDA)", out)
        self.assertIn("step 10/10", out)
        self.assertNotIn("warning #177-D", out)                          # the compiler's own lines stay in the log
        self.assertIn("warning #177-D", (self.tmp / "build.log").read_text())
        self.assertIn("docker build -t img:tag ctx", (self.tmp / "args").read_text())

    def test_the_progress_ends_when_the_files_are_whole(self):
        # the tool goes on (a check of what it wrote) after its file is whole: no more lines at 100 %
        body = ("( sleep 0.3; head -c 150000 /dev/zero > .w.partial; sleep 0.3; head -c 150000 /dev/zero >> .w.partial; mv .w.partial w.bin; sleep 2 ) & pid=$!\n"
                "t0=$SECONDS; PROGRESS_S=0.2 file_progress $pid 'the file' 300000 w.bin .w.partial; echo \"took $((SECONDS - t0))\"; wait $pid")
        code, out, err = self.sh(body)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.count("(100 %)"), 1, out)
        self.assertLess(int(out.split("took ")[1].split()[0]), 2, out)   # the job runs 2.6 s; the lines end at the file's last byte

    def test_nothing_to_write_shows_nothing(self):
        (self.tmp / "w.bin").write_bytes(b"x" * 1000)
        code, out, err = self.sh("sleep 0.5 & pid=$!\nPROGRESS_S=0.2 file_progress $pid 'the file' 1000 w.bin; wait $pid")
        self.assertEqual((code, out), (0, ""), err)

    def test_a_weight_files_size_is_read_from_its_index(self):
        (self.tmp / "idx").write_text('[["embed.weight", "BF16", [129280, 4096], 0, 1059061760], '
                                      '["layers.0.attn.wq_a.weight", "F8_E4M3", [1024, 4096], 1059061760, 4194304], '
                                      '["norm.weight", "BF16", [4096], 4096, 8192]]')
        code, out, err = self.sh("index_data_bytes idx")
        self.assertEqual((code, out.strip()), (0, str(1059061760 + 4194304)), err)

    def test_a_long_write_shows_its_bytes_per_cent_and_time_left(self):
        # a writer that grows a dot file, then renames it, as the weight and Engram tools do
        body = ("( for i in 1 2 3 4 5 6; do head -c 100000 /dev/zero >> .w.partial; sleep 0.25; done; mv .w.partial w.bin ) & pid=$!\n"
                "PROGRESS_S=0.3 file_progress $pid 'the weight file' 600000 w.bin .w.partial; wait $pid")
        code, out, err = self.sh(body)
        self.assertEqual(code, 0, err)
        lines = [l for l in out.splitlines() if "the weight file" in l]
        self.assertGreaterEqual(len(lines), 3, out)
        self.assertRegex(lines[0], r"the weight file: [0-9.]+ of 0\.6 MB \([0-9]+ %\)")
        self.assertIn("left", lines[1])
        self.assertRegex(lines[-1], r"0\.6 of 0\.6 MB \(100 %\)")

    FX_OUT = """[fabric-xfer model] memory: MemAvailable 104.2 GiB; ring 8.0 GiB (7 % of MemAvailable 104.2 GiB, at most 8 GiB)
[fabric-xfer model] rails: mlx5_0:1 gid 3 (10.255.0.1) mlx5_2:1 gid 3 (10.255.1.1)
[me@w1] [recv spark2] connection from 10.255.0.1
[me@w1] FX-READY
[fabric-xfer model] written 58.3 of 243.5 GB (23 %), 5.83 GB/s, ETA 0:31 | each receiver (% written, GB/s): me@w1 23 % 1.95 | sent 23 %, GB/s a rail: A 2.93 B 2.91
[fabric-xfer model] written 112.9 of 243.5 GB (46 %), 5.40 GB/s, ETA 1:43 | each receiver (% written, GB/s): me@w1 46 % 1.80
[fabric-xfer model] 243.5 GB to send in total, 81.2 GB sent in 44 s: 5.53 GB/s over 3 receiver(s)
[fabric-xfer model] timing (s): connect to receivers 0.40 | inventory + plan 0.00
[fabric-xfer model] me@w1: 1029 files verified by per-chunk CRC-32C, 0 failed, 81.2 GB sent, 81.2 GB written
[fabric-xfer model] me@w2: 1029 files verified by per-chunk CRC-32C, 0 failed, 81.2 GB sent, 81.2 GB written
"""

    def test_the_copys_lines_become_plain_progress_and_one_done_line(self):
        (self.tmp / "fx.txt").write_text(self.FX_OUT)
        logf = self.tmp / "setup.log"
        code, out, err = self.sh(f"SETUP_LOG={logf}; fx_lines 'model files' < fx.txt")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), [
            "  model files: 58.3 of 243.5 GB (23 %), 5.83 GB/s, about 31 s left",
            "  model files: 112.9 of 243.5 GB (46 %), 5.40 GB/s, about 1 min 43 s left",
            "  model files: done, 1029 files on each Spark, every chunk checked"])
        kept = logf.read_text()
        for line in ("ring 8.0 GiB", "FX-READY", "timing (s)", "me@w2: 1029 files verified"):
            self.assertIn(line, kept)

    def test_a_failed_chunk_stays_on_the_screen(self):
        (self.tmp / "fx.txt").write_text("[fabric-xfer model] me@w1: 1029 files verified by per-chunk CRC-32C, 2 failed, 81.2 GB sent\n"
                                         "[fabric-xfer model] ERROR: me@w1: chunk 17 of a.bin: CRC differs\n")
        code, out, err = self.sh(f"SETUP_LOG={self.tmp}/s.log; fx_lines 'model files' < fx.txt")
        self.assertIn("2 failed", out + err)
        self.assertIn("CRC differs", out + err)
        self.assertNotIn("every chunk checked", out)

    def test_build_steps_are_named_for_people(self):
        code, out = self.build(["#5 [ 1/10] FROM nvcr.io/nvidia/pytorch:26.07-py3",
                                "#12 [ 8/10] RUN cd /tensorfold && /opt/zig/zig build -Dcpu=gb10 install native",
                                "#13 [ 9/10] RUN ! grep -a -l libnccl /opt/tensorfold/native/bin/*"])
        self.assertEqual(code, 0, out)
        self.assertIn("step 1/10  base image", out)
        self.assertIn("step 8/10  compiling the engine (Zig and CUDA)", out)
        self.assertIn("step 9/10  checking the build", out)
        self.assertNotIn("/opt/zig", out)

    def test_hf_in_local_bin_is_found_without_a_login_shell(self):
        # pip install --user and pipx put hf in ~/.local/bin, which only a login shell puts on PATH
        b = self.tmp / ".local" / "bin"; b.mkdir(parents=True)
        (b / "hf").write_text("#!/bin/sh\necho hf-here\n"); (b / "hf").chmod(0o755)
        code, out, err = self.sh("hf_on_path && hf")
        self.assertEqual((code, out.strip()), (0, "hf-here"), err)

    def test_no_hf_anywhere_is_said(self):
        code, out, err = self.sh("if hf_on_path; then echo found; else echo none; fi")
        self.assertEqual((code, out.strip()), (0, "none"), err)

    def test_a_long_step_says_it_is_still_running(self):
        code, out = self.build(["#12 [ 8/10] RUN zig build", "sleep 1.5", "#13 [ 9/10] RUN true"], beat="0.5")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"step 8/10 still running, \d+ s in")

    def test_a_failed_build_shows_its_last_lines_and_returns_its_status(self):
        code, out = self.build(["#12 [ 8/10] RUN zig build", "#12 12.0 error: expected type 'u32', found 'u64'",
                                "#12 ERROR: process did not complete successfully"], rc=3)
        self.assertEqual(code, 3, out)
        self.assertIn("error: expected type 'u32', found 'u64'", out)
        self.assertIn("build.log", out)

    def test_the_classic_builders_steps_are_shown_too(self):
        code, out = self.build(["Step 3/5 : RUN make", " ---> Running in 1a2b", "cc -O2 x.c", "Step 5/5 : CMD [\"x\"]"])
        self.assertEqual(code, 0, out)
        self.assertIn("step 3/5  RUN make", out)
        self.assertNotIn("cc -O2", out)

    RAILS = "enp1s0f0np0 enP2p1s0f0np0"
    PORTS = "enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"
    NVIDIA = ("network:\n  version: 2\n  ethernets:\n" + "".join(
        f"    {p}:\n      link-local: [ ipv4 ]\n      mtu: 9000\n" for p in PORTS.split()))   # NVIDIA's two-Spark playbook file

    def conflicts(self, files, body="netplan_conflicts {d} 51-cx7-roce-static.yaml \"{rails}\" \"{ports}\""):
        d = self.tmp / "netplan"; d.mkdir()
        for name, text in files.items():
            (d / name).write_text(text)
        code, out, err = self.sh(body.format(d=d, rails=self.RAILS, ports=self.PORTS))
        self.assertEqual(code, 0, err)
        return sorted(tuple(line.split(" ", 1)) for line in out.splitlines()), d

    def test_netplan_files_that_never_name_a_rail_and_the_setups_own_are_left_out(self):
        got, d = self.conflicts({"00-installer-config.yaml": "network:\n  ethernets:\n    enP7s7:\n      dhcp4: true\n",
                                 "51-cx7-roce-static.yaml": "network:\n  ethernets:\n    enp1s0f0np0: {}\n",
                                 "70-near.yaml": "network:\n  ethernets:\n    enp1s0f0np0x: {}\n",   # a name that starts the same
                                 "80-empty.yaml": ""})
        self.assertEqual(got, [])

    def test_an_earlier_fabric_file_of_only_connectx_ports_is_moved(self):
        got, d = self.conflicts({"50-cx7-fabric.yaml": self.NVIDIA})
        self.assertEqual(got, [("move", str(d / "50-cx7-fabric.yaml"))])

    def test_a_file_that_also_configures_another_interface_stops_the_setup(self):
        got, d = self.conflicts({"60-mine.yaml": "network:\n  ethernets:\n    enP7s7:\n      dhcp4: true\n"
                                                 "    enp1s0f0np0:\n      mtu: 1500\n"})
        self.assertEqual(got, [("stop", str(d / "60-mine.yaml"))])

    def test_a_rail_in_a_bond_a_vlan_or_a_match_stops_the_setup(self):
        got, d = self.conflicts({
            "61-bond.yaml": "network:\n  ethernets:\n    enp1s0f0np0: {}\n    enP2p1s0f0np0: {}\n"
                            "  bonds:\n    bond0:\n      interfaces: [enp1s0f0np0, enP2p1s0f0np0]\n",
            "62-vlan.yaml": "network:\n  vlans:\n    v200:\n      id: 200\n      link: enP2p1s0f0np0\n",
            "63-match.yaml": "network:\n  ethernets:\n    rail:\n      match:\n        name: enp1s0f*\n      mtu: 1500\n"})
        self.assertEqual(got, [("stop", str(d / n)) for n in ("61-bond.yaml", "62-vlan.yaml", "63-match.yaml")])

    def test_a_file_netplan_cannot_read_stops_the_setup(self):
        got, d = self.conflicts({"64-broken.yaml": "network:\n  ethernets: [enp1s0f0np0\n"})
        self.assertEqual(got, [("stop", str(d / "64-broken.yaml"))])

    def test_the_check_runs_as_the_setup_sends_it_to_a_spark(self):
        # setup.sh sends the function's text (declare -f) to each Spark and runs it under sudo there
        got, d = self.conflicts({"50-cx7-fabric.yaml": self.NVIDIA},
                                body="{{ declare -f netplan_conflicts; echo 'netplan_conflicts {d} 51-cx7-roce-static.yaml "
                                     "\"{rails}\" \"{ports}\"'; }} | env -i PATH=\"$PATH\" bash")
        self.assertEqual(got, [("move", str(d / "50-cx7-fabric.yaml"))])


@unittest.skipUnless(shutil.which("gcc") and shutil.which("make"), "needs gcc and make")
class FabricXferSuite(unittest.TestCase):
    """tools/fabric-xfer's own tests (the core, the engines over an in-process wire, the wrapper), built in a copy so the
    checkout keeps no binaries. They need no RDMA device."""

    def test_its_suite_passes(self):
        tmp = Path(tempfile.mkdtemp(prefix="tf-fx-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(ROOT / "tools" / "fabric-xfer", tmp / "fx")
        r = subprocess.run(["make", "-s", "-C", str(tmp / "fx"), "test"], capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertRegex(r.stdout, r"test_core\.c: \d+ checks, 0 failed")
        self.assertRegex(r.stdout, r"test_engine\.c: \d+ checks, 0 failed")


if __name__ == "__main__":
    unittest.main()
