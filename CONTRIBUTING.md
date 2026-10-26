# Contributing

Changes should make the workflow easier to inspect, reproduce, or extend.
Describe the concrete failure or missing capability, the resulting behavior, and
the evidence supporting it. Keep claims about implemented behavior separate from
scripted tests, real experiments, and live validation.

Use Python 3.12 and the frozen dependency lock:

```sh
uv sync --python 3.12 --frozen --extra preparation
uv run ruff check src tests scripts/verify_release.py
uv run pytest -m 'not container and not native_container'
```

The `preparation` extra enables the tokenizer and image-preparation tests.
Platform-specific checks can still skip when their required operating-system
interfaces are unavailable; report those skips explicitly.

Run relevant focused checks while developing. Changes to execution, scoring,
recovery, or manuscript binding should include a meaningful regression check.
Use the [CPU example](README.md#start-with-the-cpu-example) for real container
execution, and report the image identity, audit result, and resume behavior.
Container and native benchmark tests require their declared external runtimes
and inputs; a skip is not a pass.

Preserve existing run artifacts. Do not edit a frozen campaign to make a new
implementation pass its old identity checks. Use a fresh output directory for a
new source or protocol. Live experiments require explicit authorization and
declared budgets; ordinary tests must not make model calls.

Task adapters are trusted evaluator code. Keep candidate execution isolated,
protect final labels, validate observation identity and shape, and declare
adapted baselines or metrics. The
[architecture guide](docs/architecture.md#extension-points) describes registration.

Do not commit credentials, private datasets, model transcripts, run directories,
or environment-specific configuration. Original implementation code uses the
[MIT license](LICENSE); retain the distinct upstream licenses and attribution for
the bundled PaperOrchestra files and templates. See
[third-party notices](THIRD_PARTY_NOTICES.md). Before proposing a public result,
check the [validation reporting requirements](docs/validation.md).
