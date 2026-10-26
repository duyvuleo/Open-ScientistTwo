# Explicit live execution

The default `run` command uses scripted agents. Live execution is a separate
opt-in path that can consume model quota and compute. Authenticate the selected
model transport outside this repository; do not put credentials in task JSON,
manifests, reports, or committed files.

Providers include Codex CLI, OpenAI, Anthropic/Claude, and custom
OpenAI-compatible endpoints. See [model providers](model-providers.md) for API
configuration, supported response modes, image capability and secret handling.
One provider/model is frozen per campaign; the same backend supplies its research
and integrated writing roles.

Every live attempt needs an explicit model identifier, maximum calls, observed
token stop threshold, campaign and per-call deadlines, and experiment limit.
The threshold stops new calls after recorded usage reaches it; an in-flight call
may cross it. It is not a guaranteed billing cap. If completion or usage is
unknown, preserve the evidence and resolve that state before authorizing a retry.

## A bounded pilot configuration

Copy [live-options.example.json](../examples/live-options.example.json) to a
local options file and replace `MODEL_ID` with an available, explicitly selected
model. This template defaults to Codex CLI. The
[provider templates](model-providers.md#configure-a-campaign) select direct API
backends without requiring Codex. The example declares 40 calls, a 400,000 observed-token threshold,
30 experiments, 1,800 campaign seconds, and 120 seconds per call. These are
illustrative ceilings, not an estimate that the full workflow will finish.
The writer is explicitly `canonical` and literature is local-only.

After preparing the [CPU worker](../README.md#start-with-the-cpu-example), freeze
the launch without making model calls:

```sh
export SCIENTISTTWO_WORKER_IMAGE="$(docker image inspect scientisttwo-worker:0.1 --format '{{.Id}}')"
uv run scientisttwo live-pilot create \
  examples/quickstart.json live-options.local.json live-manifest.local.json \
  --image "$SCIENTISTTWO_WORKER_IMAGE"
uv run scientisttwo live-pilot check live-manifest.local.json
```

Review the generated manifest. The explicit execution commands are:

```sh
uv run scientisttwo live-pilot run \
  live-manifest.local.json runs/live-quadratic --allow-live
uv run scientisttwo live-pilot resume runs/live-quadratic --allow-live
```

These commands can make model calls. Resume preserves the original deadlines,
limits, source binding, and completed operations; it does not grant a new budget.
The ordinary `resume` command refuses live-pilot campaigns. A source change
requires a new manifest and campaign rather than editing a frozen launch.

## Optional writing and benchmark research

To use PaperOrchestra in a live pilot, set the options' `workflow.writer_backend`
to `paperorchestra` and supply the immutable `workflow.compiler_image` after
preparing the [PDF container](paperorchestra.md#prepare-the-pdf-container).
An API provider must also declare `provider.vision: true` and use a model that
supports the page images; unsupported image capability is rejected before work.
The pinned writing engine is included and is used by default;
`workflow.paperorchestra_root` is an optional external-source override. Its inner
calls, binding steps, and PDF work consume the same overall campaign budget.
Do not infer a sufficient budget from the scripted example.

The separate native `live create --mode research` command defaults to
PaperOrchestra and requires a configured PDF image; canonical
writing is an explicit alternative. `live-pilot` uses the writer in its options
file. Both paths freeze their selected configuration rather than silently
switching writers. Inspect `scientisttwo live --help` and the
[benchmark adapter guide](benchmarks.md) before preparing native tasks.

A useful live result must report the exact status, accounting completeness,
development/final boundary, and any operator intervention. A budget-exhausted
attempt can contain valid development measurements while failing the campaign's
completion contract. Current live end-to-end and full 13-slot acceptance remain
unverified; see [validation status](validation.md).
