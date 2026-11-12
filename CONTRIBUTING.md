# Contributing

This repository serves DeepSeek-V4.1-Flash-1017 on four DGX Sparks. It is built on BertholomusAI's DeepSeek branch of
TensorFold, and TensorFold is Ash Hart's engine; see [`CREDITS.md`](CREDITS.md). Send each change where it belongs:

- the engine's core and server, for every model: [ashhart/TensorFold](https://github.com/ashhart/TensorFold);
- the DeepSeek family's four-Spark decomposition, drafting, vision and decode rings:
  [BertholomusAI's TensorFold](https://github.com/bertholomus/TensorFold) and its kernel kit;
- what this repository adds (Rocinante, the prompt path's exchanges, the SSD tier's integration, the setup, the scripts
  and the tools in [`FORK.md`](FORK.md); the setup began as MiaAI-Lab's recipe): an issue or a pull request here, against
  the `deepseek-v41-zig-tp4` branch.

Open an issue before a change of arithmetic, of a kernel, or of what an exchange carries, so the test it needs is agreed
first.

## The checks

Every pull request passes these; the no-GPU and format workflows run them on every pull request and on every push to
`deepseek-v41-zig-tp4`:

```sh
python3 -m unittest discover -s tests                   # the scripts, the setup, the tools (no GPU, no Sparks)
zig build test                                          # the engine's host tests, Zig 0.17.0 (.zig-version)
zig build test-kernel-contract test-golden              # the kernel contract and the server's golden corpus
zig build fuzz-cortheon                                 # the web-text readers under the fuzz harness
python3 tools/zig/lean_check.py                         # file and comment length: a pull request adds no problem
zig fmt --check build.zig zig                           # formatting, as Zig 0.17.0 writes it
python3 tools/schema/check_env_schema.py                # every setting the engine reads has a schema row
make -C tools/fabric-xfer test                          # the RDMA copy's own tests (a loopback backend)
```

None of these runs the model. A change that reaches the GPU, an exchange or the replies is also run on four Sparks,
and its pull request says how: the replies compared byte for byte with the build before it (`tools/validate.py`
`--compare`), and the speed measured as below.

## Exactness

A change keeps these equal, at the same checkpoint, settings and kernel kit (TensorFold's own rules, kept here):

- drafted output equals `"draft": false` output, token for token;
- a reply from a cached or SSD-restored prompt equals the reply from the whole prompt read fresh;
- each of several requests at once equals its run alone;
- an exchange that carries less (owned rows, FP8 codes) gives the receiver the same bytes it would have computed.

A changed reduction order or accumulation needs a reference that shows its error is no worse than before, column by
column, not only on average.

## Measuring

Measure before and after on the same four Sparks, the same build of everything else, the same prompts and settings,
with the GPU clock limit `served.sh` sets. Run-to-run noise on these Sparks is about ±4-5 %: a difference smaller than
that needs repeated, alternating runs before it is called a gain. Label every number with what was measured, on which
build, and against what. Use one client per comparison: `tools/recipe_bench.py` and `tools/bench.py` count totals
differently, so their tables are not compared with each other.

## Commits

One commit a feature: its code, its tests and its documentation together, with a message that says what it does and
what it measured. Use your public Git identity with GitHub's noreply address. Keep private names, addresses, paths and
keys out of the code, the tests and the history; examples use reserved addresses (192.0.2.x).

## License

TensorFold is Apache-2.0, and a contribution follows that license. The model's weights keep their own license
([`ATTRIBUTION.md`](ATTRIBUTION.md)).
