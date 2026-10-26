# Benchmark adapters

The testbed defines 13 canonical slots: Vectorization, House Price, Spaceship
Titanic, Parkinson's progression, Feedback, CIFAR-10, IMDb, OGBN-Arxiv,
Contrails, FathomNet, CLRS, BabyLM, and Llama inference. Their presence in the
[catalog](../src/scientisttwo/testbed/catalog.py) means a preparation and
evaluation contract exists. It does not mean all assets are available or that a
complete live campaign has passed.

The MLAgentBench reference is pinned to
[`5d71205cc20a8e95d43aa7cb7120e89ca3323e31`](https://github.com/snap-stanford/MLAgentBench/tree/5d71205cc20a8e95d43aa7cb7120e89ca3323e31).
The repository also includes a separate MLE-bench integration with its own pin
and graders in the [benchmark registry](../src/scientisttwo/benchmarks/registry.py).
Neither path downloads or accepts restricted dataset terms automatically.

Inspect the available commands and local preparation state:

```sh
uv run scientisttwo benchmark --help
uv run scientisttwo benchmark readiness --root .testbed
uv run scientisttwo benchmark prepare-plan mlagentbench vectorization .upstream/MLAgentBench
```

The last command requires the separately obtained pinned upstream checkout.
Preparation, original starters, adapted references, development validation,
protected final grading, and live workflow completion are tracked separately.
Missing or failed tasks remain in the readiness denominator.

Several adaptations materially affect comparison with upstream results:

- Vectorization and Llama inference use independent warmed, paired timing and
  output-equivalence checks. A latency ratio is the candidate median divided by
  its interleaved baseline median; raw timing evidence is retained.
- Prepared train/development/final partitions and protected labels replace
  unsafe or underspecified data access where the recipe declares this change.
- Some original starters are incomplete. Named adapted references are reported
  separately from the upstream starter.
- FathomNet reports an adapted MAP@20 result; the original composite score is
  unavailable in this path. CLRS also declares an adapted task-specific metric.

These changes improve the inspectability of this implementation's protocol,
but they change the experiment. Report the exact recipe and deviations with any
result. Full 13-slot live end-to-end validation remains unverified; see
[validation status](validation.md).
