# About this fork

This is a fork of [MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold),
Mia's recipe for DeepSeek-V4.1-Flash-1017 on four DGX Sparks with [TensorFold](https://github.com/ashhart/TensorFold)'s
native engine and BertholomusAI's `dsv41` family. Its `main` is the recipe's `main` plus one commit: the two patches
below, their notes and their credits. Every other file is the recipe's. As the recipe's NOTICE asks, this file states
what the fork adds.

Status (2026-10-27): both patches serve production on four DGX Sparks (`./start.sh`, a switch; `ABLIT=1`,
`PARALLEL=16`, `CONTEXT=1048576`, `POOL=16777216`).

## This fork's patches

They apply after the recipe's `0001`-`0026`, in filename order, as the recipe's do. Each file begins with a header that
explains it in full, and each carries its tests in the engine's own suites.

| Patch | What it does | Tests |
|---|---|---|
| `9001-mgllm-usable-window` | The server is told the window a stream really holds, so a request without `max_tokens` is served (every token left) and one asking for more than fits gets a 400 naming the most it may ask, instead of a 500 | `rows.zig` (dsv41 suite) |
| `9002-mgllm-live-prefill` | A prompt pass in flight shows on `/health` and `/metrics`: its tokens, its speed so far, a counter of the prompt tokens computed | `stream.zig`, `lane_host_test.zig`, `live.zig`, `metrics.zig` |

### 9001: a request without `max_tokens`

Without `max_tokens` the server clamps a request's reply to `context window - prompt`, and it was told the whole
window (1,048,576). The lanes place a stream only when `prompt + reply + ROUND_ROWS + 2`, in whole 2,048-row chunks,
fits that window (`lanes.zig` prefillFn), so every such request was refused there: `PromptTooLong`, HTTP 500, at any
prompt size. `families/dsv41/rows.zig` now holds both rules, and the server is told `usableRows`: the window rounded
down to whole chunks, less `ROUND_ROWS + 2`, which is 1,048,526 at the recipe's defaults. A request without
`max_tokens` gets every token left; one that asks for more than fits is refused by the server with HTTP 400.

### 9002: the prompt pass in flight

A prompt pass runs inside its request's admission and reported nothing until it ended:
`live.prefill_tokens_per_second` read 0 for the whole pass (38 s for a 131K-token prompt) and showed the pass's speed
only for the 2 s after it. Now the dsv41 prefill loop stores the rows it has passed after each 2,048-row chunk (one
atomic store; no lock and no GPU wait in the loop), and the lane host reports the pass while it runs:

- `/health` `live`: `prefill_tokens_per_second` is the running pass's speed so far (the last pass's for 2 s after it,
  as before); `prompt_pass` = `{computed, total, resumed, seconds}`, or `null` between passes;
  `prompt_tokens_computed_total`.
- `/metrics`: `tensorfold:prompt_tokens_computed_total`, a counter with the running pass's tokens included (its rate
  is the prefill speed; a pass that fails or is cancelled keeps what it computed, so the counter never goes back), and
  the gauges `tensorfold:prompt_pass_tokens_computed`, `_total` and `_resumed` (0 between passes).

An engine whose host does not count passes keeps `/health` and `/metrics` as they were.

## Checks on four DGX Sparks (2026-10-27)

Measured on the Sparks above, requests sent from another machine on the same network.

- `tools/validate.py`: all pass (text, thinking, tool call and result, image, video, a 21,875-token prompt, 8 at once
  equal to solo), and the 8 greedy replies are byte-identical to those of the same build without these patches.
- Prefill, fresh prompts one at a time: 131,331 tokens 3,343 / 3,417 / 3,556 tok/s with the patches, 3,419 without;
  32,855 tokens 3,247 / 3,575 with, 3,118-3,556 without.
- `9001`: a short request (`Say OK.`, 33 prompt tokens) without `max_tokens`: HTTP 200, where the same request got 500
  without the patch; `max_tokens` 1,048,493 (all the room left): 200; 1,048,494: 400.
- `9002`: a 131,331-token prompt, `/health` read once a second: `prompt_pass.computed` 4,096 to 129,024 of 131,331,
  the prefill figure 3,020-3,460 tok/s while the pass ran, the counter ending at the prompt's size.
- Engine tests: the dsv41 suite (LLVM build, x86_64) 120 pass and 1 fails, the same one with and without these
  patches (`lanes.test.draft chances are the prefix survivals chooseK sums`); the server suite (`zig/tests/server/zig_test.sh`)
  67 + 31 pass; `tensorfold-native` cross-compiles for aarch64. Each new test was seen to fail with its code broken.
