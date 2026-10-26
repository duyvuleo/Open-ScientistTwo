# Open-ScientistTwo

An independent, experimental implementation of the research workflow described in
[ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI](https://arxiv.org/abs/2609.19644).
This project is unofficial and is not affiliated with the paper's authors.
The Python package and command remain `scientisttwo`.

Research agents need more than a plausible final answer: experiments must be
reproducible, comparisons must use the declared protocol, and manuscript claims
must remain connected to the evidence that produced them. Open-ScientistTwo
implements a persistent research controller with task adapters, isolated
experiments, trusted scoring, bounded refinement, manuscript generation, and
auditable checkpoints.

The default run uses **scripted agents and real CPU experiments, with zero live
model calls**. The distribution includes a pinned PaperOrchestra writing engine;
selecting that writer and enabling live agents are separate opt-in choices.
This release does not reproduce the paper's reported benchmark results, model
mix, costs, or scientific novelty. See [validation status](docs/validation.md)
before interpreting a successful run.

Local release checks passed on 26 October 2026, including an installed-wheel
CPU/PaperOrchestra campaign, real PDF operations, audit, and unchanged completed
resume. See the [verification record](docs/validation.md#current-public-release)
for source identity, test counts, and limits. Hosted results are tracked in the
[release workflow](https://github.com/duyvuleo/Open-ScientistTwo/actions/workflows/release.yml).

## Start with the CPU example

Use Python 3.12, `uv`, and an existing Docker installation on Linux or macOS.
Installation and image preparation need network access. The experiment itself
runs without network access; the CLI does not start a container service or
silently execute candidate code on the host.

On macOS with Docker Desktop or Colima, the container VM must share the checkout,
installed virtual environment, and run directories. Worker and source files are
bind-mounted from their resolved host paths; an unshared directory causes
container path-access failures. Include an external wheel environment in the
shared locations when testing an installed package.

```sh
git clone https://github.com/duyvuleo/Open-ScientistTwo.git
cd Open-ScientistTwo
uv sync --python 3.12 --frozen
docker build -t scientisttwo-worker:0.1 .
uv run scientisttwo doctor
uv run scientisttwo run examples/quickstart.json --runs-dir runs
```

The [example](examples/quickstart.json) generates a quadratic regression dataset
and compares established methods under fixed splits. It is a workflow example,
not a discovery claim. No API key, model account, benchmark download, GPU,
or LaTeX installation is needed for this default path.

The run prints its directory. Use that directory's basename as `RUN_ID`:

```sh
uv run scientisttwo inspect RUN_ID --runs-dir runs
uv run scientisttwo audit RUN_ID --runs-dir runs
uv run scientisttwo resume RUN_ID --runs-dir runs
```

`audit` checks stored artifacts and re-executes the selected development
experiment. `audit --no-reexecute` checks stored evidence without that replay.
`--include-final` also checks frozen final evidence. A completed `resume` should
reuse committed work; an interrupted run continues only when its saved source,
runtime, and protocol still match. Preserve an unsuccessful campaign and inspect
its reported outcome before retrying.

For an existing named Docker context, pass `--context CONTEXT` to `doctor` and
`run`. The saved context is reused by `audit` and `resume`. Podman is also exposed
through `--runtime podman`; the release verification scope is recorded separately.

## What a campaign records

- `campaign.sqlite`: stage state, operation identities, decisions, and accounting.
- `artifacts/`: content-addressed protocols, code, observations, source evidence,
  reviews, and manuscript bundles.
- `work/`: experiment staging and recovery state.
- `report.md` and `summary.json`: convenient summaries; the sealed artifact
  references remain the evidence of record.

The controller separates adaptive development from protected final evaluation.
An internal review decision is a workflow result, not external peer review or
publication acceptance. Scripted review scores measure neither writing quality
nor scientific merit.

## Optional paths

- [PaperOrchestra](docs/paperorchestra.md): select the included pinned engine and
  prepare its isolated PDF image, first with scripted agent responses. No second
  source checkout is required.
- [Live agents](docs/live.md): select Codex, OpenAI, Anthropic/Claude, or a
  [custom compatible endpoint](docs/model-providers.md); freeze an explicit model, call limit, token stop
  threshold, experiment limit, and deadlines; execution requires `--allow-live`.
- [Benchmark adapters](docs/benchmarks.md): inspect the 13-slot testbed and its
  preparation and evaluation boundaries. Adapter availability does not establish
  complete benchmark validation.
- [Architecture](docs/architecture.md) and [paper fidelity](docs/fidelity.md):
  inspect the mechanisms, adaptations, and unsupported claims.

## Development and citation

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and evidence requirements.
When using this implementation, cite the exact repository revision and identify
it as Open-ScientistTwo. Cite the [ScientistTwo paper](https://arxiv.org/abs/2609.19644)
separately for the original method; [CITATION.cff](CITATION.cff) records both.
Original implementation code is covered by the [MIT license](LICENSE).
The bundled PaperOrchestra engine and templates retain their upstream licenses;
see [third-party notices](THIRD_PARTY_NOTICES.md) for the file-specific terms and
attribution. Benchmark inputs retain their own upstream terms.
