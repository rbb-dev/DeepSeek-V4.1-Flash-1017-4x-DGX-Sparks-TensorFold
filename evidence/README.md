# Evidence

The files this repository's published numbers are computed from, one folder per engine build, named by the image tag of
the build the runs used (`tensorfold-dsv41:native-<tag>`). The tag hashes the engine's build inputs (`ENGINE_PATHS` in
[`scripts/config.sh`](../scripts/config.sh)): a commit with the same files there prints the same tag, and a later commit
that changes any of them, its tests included, prints a new one while the folder keeps the tag its runs used.

Each folder holds the runs of four Sparks set up from scratch with `./setup.sh` (the kernel kit fetched from its
published default, stock weights, every setting in [`scripts/served.sh`](../scripts/served.sh)), driven from a separate
machine on the same network. The client's host name, the engine's address, wall-clock times and local paths are removed;
nothing else is changed.

| File | What it is | Made by |
|---|---|---|
| `MANIFEST.json` | The image, base image, kernel kit, model revision, driver, kernel, OS, hardware and the settings of each run | the run |
| `recipe-main.json` | Reading a prompt (4K to 1M tokens), one request writing, and 1 to 16 requests at once; every request | [`tools/recipe_bench.py`](../tools/recipe_bench.py) (MiaAI-Lab's benchmark) |
| `recipe-turbo.json` | Reading a prompt with Turbo on; every request | [`tools/recipe_bench.py`](../tools/recipe_bench.py) (MiaAI-Lab's benchmark) |
| `code.jsonl` | Writing code, 1 to 16 requests at once, every repetition | [`tools/bench.py`](../tools/bench.py) |
| `tables.md` | The README's three performance tables, rebuilt from the three files above | [`tools/readme_tables.py`](../tools/readme_tables.py) |
| `longbench2-long-thinking-off.jsonl` | LongBench v2's long items (THUDM's set), one line an item: the question's id, the expected and given letters, the reply | [`tools/longbench2.py`](../tools/longbench2.py) |

To rebuild the tables from a folder:

```bash
cd evidence/native-8c6e8e88d875
python3 ../../tools/readme_tables.py --main recipe-main.json --turbo recipe-turbo.json --code code.jsonl
```

To score the LongBench v2 file again (with LongBench v2's `data.json` from Hugging Face, `zai-org/LongBench-v2`; every
item is already in the file, so nothing is sent, though the tool opens `--out` to append, so point it at a copy):

```bash
cp evidence/native-8c6e8e88d875/longbench2-long-thinking-off.jsonl /tmp/lb2.jsonl
python3 tools/longbench2.py --data data.json --length long --out /tmp/lb2.jsonl
```

The speed runs had the prompt cache and the SSD tier off (every prompt read cold); LongBench ran on the shipped settings.
The commands are the README's own ([Performance](../README.md#performance)) and
[`docs/long-context.md`](../docs/long-context.md#does-it-reason-over-it).
