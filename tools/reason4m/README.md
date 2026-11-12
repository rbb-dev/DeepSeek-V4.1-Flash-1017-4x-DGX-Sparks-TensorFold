# reason4m: does the model reason over 1M, 2M and 4M tokens, or only find?

Needle tests show that the model finds a fact anywhere in a long prompt. These suites ask for more: telling
near-identical passages apart and putting them in order (MRCR), or chaining facts scattered through millions of tokens
(BABILong). A model that only retrieves scores near chance on both. One request at a time against the engine's API,
streamed; Python's standard library only.

```bash
python3 tools/reason4m/prepare_data.py mrcr                 # Google's MRCR v2 files (first 96 rows of each length)
python3 tools/reason4m/reason4m.py verify-data              # the downloaded data has the format this code assumes
python3 tools/reason4m/reason4m.py run --suite mrcr --lengths 1m-2n --modes off,high --samples 20 \
    --url http://localhost:8000 --key-file ~/.config/dsv41/api-keys --out mrcr-1m.jsonl
python3 tools/reason4m/reason4m.py summary mrcr-1m.jsonl
python3 tools/reason4m/reason4m.py run ... --dry-run        # the plan, the sizes and a time estimate; sends nothing
python3 tools/reason4m/reason4m.py run ... --resume         # skips rows already in --out, sends the same prompts again
```

The data lands in `tools/reason4m/data/` (git ignores it). `--key-file` takes the setup's key file: its first `label:
key` line is sent. The 2M and 4M lengths need the engine started with the longer window
([`docs/long-context.md`](../../docs/long-context.md)).

## The suites

**MRCR v2** (Google DeepMind, [eval_hub](https://github.com/google-deepmind/eval_hub/tree/master/eval_hub/mrcr_v2)). A
long fake chat of "Write a {format} about {topic} in {style} style" turns, where 2, 4 or 8 turns (`-2n`, `-4n`, `-8n`)
answer the same request with different texts (a length label without a suffix means 8). The last question asks for the
n-th of them, behind a random string the reply must carry. Score (their `mrcr_v2_metric`): 0 without the string, else
the similarity of the text after it to the expected one; `score_strict` also needs the reply to begin with the string.
The 1M, 2M and 4M samples are the published files' own transcripts.

**OpenAI MRCR** (`omrcr`, [openai/mrcr](https://huggingface.co/datasets/openai/mrcr)): the same task as real chat turns,
from the set's largest bin (524K-1M tokens), one conversation a sample.

**BABILong qa2 and qa3** ([booydar/babilong](https://github.com/booydar/babilong)): bAbI stories (two or three
supporting facts: who carried what where) hidden between sentences of PG19 books, asked with the official prompts and
scored with the official rule. The published set stops at 1M and has one story a haystack, so this tool builds its own:
one haystack a length carrying 20 stories (characters renamed per story, so stories cannot contradict each other), each
story's facts at random places in order. The other stories' facts are then look-alike distractors: harder than the
published set, and not comparable with its curves.

## How a sample is built

Per suite and length, every question shares ONE context, byte for byte, and only the final question differs: the engine
reads the context once and answers each further question in seconds, and every mode is asked the same questions (a
paired comparison). The cost: 20 questions about one context are not 20 independent contexts.

Lengths are counted in this model's tokens by asking the server: before a length, a few slices of the text are sent as
one-token requests, and `usage.prompt_tokens` gives characters per token for this text. An MRCR transcript longer than
the target loses whole filler turns, spread evenly, never a relevant turn, so every question keeps its answer (the tests
re-solve the questions on the cut transcript). The plan, the calibration and the cut are written to `--out`, so a
resumed run sends the same prompts.

## Requests and modes

One user message, streamed, temperature 1.0 and top_p 1.0 (`--temperature` changes it), no tools.

| mode | thinking | `reasoning_effort` | completion budget, MRCR (thinking included) | completion budget, BABILong |
|---|---|---|---|---|
| off | off | (not sent) | 2,048 | 256 |
| high | on | high | 16,384 | 16,384 |
| max | on | max | 32,768 | 32,768 |

A thinking reply that reaches its budget has no answer and scores 0; `summary` counts these as cut. Prompt plus budget
must fit the engine's window (`--target 1m=900000` lowers a length).

## Output

`--out` is JSON lines: a plan line per suite and length, then one result line a request (the prompt's tokens as the
server counted them, the mode, the expected and given answers, the scores, reasoning length, first-token and total time,
the finish reason). `summary` prints, per suite, length and mode: the number of questions, the mean score with its
standard error, the strict score, errors, cut replies and chance.

## Tests

`python3 -m unittest tests.test_reason4m` (34 tests, a fake server, no engine; the tests on the published data run once
`prepare_data.py mrcr` and `prepare_data.py babilong` have fetched it, and are skipped before): both scoring rules, an
independent bAbI solver that matches the published targets for all 100 qa2 and 100 qa3 stories, the request shape of
each mode, resume, the shared prefix byte for byte, the filler-only cut on synthetic and on the real 1M, 2M and 4M data,
the BABILong haystacks (each story once and in order, solvable from the facts found in the haystack), the dry run, and
the key sent as a bearer token and never printed.

## Limits

- 20 questions of one context a cell, at temperature 1.0: a rerun differs; claim no more than the printed standard
  error.
- The cut MRCR transcript has less filler between the needles than the published one.
- BABILong's 20-stories-a-haystack construction differs from the published set at every length.
- Only the first 96 rows of each MRCR file are fetched (12 needle groups); `prepare_data.py mrcr --rows N` fetches more.
