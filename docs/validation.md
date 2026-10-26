# Validation status

Validation is scoped to a source version, task, backend, and runtime. Passing a
unit test, executing a real experiment, and completing a live research campaign
are different claims.

## API provider extension

The [provider interface](model-providers.md) supports Codex, OpenAI Responses,
Anthropic Messages, and explicitly configured OpenAI-compatible endpoints. Its
verification uses scripted responses and local HTTP servers. It does not claim
access to a commercial model or compatibility with every custom server.

The full scripted API research-cycle test exercises 25 roles and 61 completions,
including evolution, ablation-driven refinement, rebuttal, manuscript revision,
freeze, final evaluation and audit. Its 732 synthetic usage tokens reconcile
exactly; completed resume adds no calls or experiments. Both provider responses
and experiment outcomes in this test are synthetic.

Configuration, credentials, schemas, images, timeout, refusal, missing usage,
failed-call accounting, audit tampering and suite recovery have separate tests.
The [verification record](provider-verification.json) records the source identity,
regression results and independent real offline container check. Real API access
and live research quality remain unverified; no paid calls were made.

On 26 October 2026, the fixed-source regression passed **2,605 tests**, with one
Linux-only sampler test skipped on macOS and 15 container tests deselected.
Ruff, the installed-wheel resource check, and a staged credential scan passed.
The separate real CPU/PaperOrchestra run completed with 15 recorded observations,
four bundled writing attempts, real PDF operations, audit re-execution, and
unchanged completed resume. Its agents were scripted and its live-call count was
zero. The first container attempt used an unshared installation path and failed;
the same wheel passed after moving its installation into the existing shared
validation directory. That failed attempt was preserved.

## Bundled-writer release baseline, before API providers

Local release verification passed on **26 October 2026** for version `0.4.0`,
using a macOS ARM64 controller and Linux ARM64 experiment/PDF containers. The
[machine-readable record](release-verification.json) binds the checks to
implementation identity
`b848712ada25c49873252de66bef39b440a36c1b434e254c94136fe8b794e392`.

| Check | Verified local result |
| --- | --- |
| Main test suite | 2,379 passed; one Linux `/proc` sampler test skipped on macOS; 15 container tests deselected |
| Release verifier tests | 20 passed in a separate run |
| Ruff | Passed |
| Installed wheel outside the checkout | 189 source/resource files verified, including 17 pinned PaperOrchestra files and the original `natbib.dtx` license source |
| Generated quadratic CPU campaign with bundled PaperOrchestra | Completed with `development_improvement`; 16 recorded container observations and four successful PaperOrchestra runtime attempts |
| PDF operations | Real isolated compilation, text extraction, and page rendering passed |
| Audit | Final evidence checks and selected development container re-execution passed |
| Completed resume | Database semantics, operation order, and all artifact bytes remained unchanged |
| Live model calls | Zero; all research and writing agents were scripted |

The real container campaign was verified separately from the test suite's
deselected container tests. This result validates the declared generated task and
bundled writing path; it does not establish live agent quality, scientific
novelty, or coverage of every possible controller branch.

This record predates the provider extension. Consult the
[release workflow](https://github.com/duyvuleo/Open-ScientistTwo/actions/workflows/release.yml)
for checks on each commit. The current source has **not** completed a fresh live
end-to-end campaign, and full live validation across all 13 testbed slots remains
unverified.

## Repeat the release checks

After preparing the environment and worker image from the
[quickstart](../README.md#start-with-the-cpu-example), run:

```sh
uv run --frozen python scripts/verify_release.py container \
  --task examples/quickstart.json --image scientisttwo-worker:0.1
```

The verifier runs the real CPU campaign, checks completed resume, and then runs
the re-executing audit with scripted agents. It checks that resume adds no
operations or changes to sealed evidence before the audit records its own
re-execution. A later resume can refresh the report to include that audit
evidence. The verifier preserves the attempt's local receipts. Pass `--context`
for an existing named Docker context. After preparing the
[PDF container](paperorchestra.md#prepare-the-pdf-container), verify the bundled
writer separately:

```sh
uv run --frozen python scripts/verify_release.py container \
  --task examples/quickstart.json --image scientisttwo-worker:0.1 \
  --writer-backend paperorchestra --max-prompt-chars 1000000 \
  --compiler-image "$SCIENTISTTWO_PDF_IMAGE"
```

The 1,000,000-character ceiling accommodates the larger serialized writing
context; it is an input-size limit, not a model-token budget. Both examples use
scripted agents and zero live calls. `--paperorchestra-root` selects an optional
external pinned copy.

The [release workflow](../.github/workflows/release.yml) defines separate jobs for
offline tests and installed-wheel resources, pinned upstream integration
fixtures, real CPU execution, and the installed bundled PaperOrchestra engine
with real PDF compilation. Defining these jobs is not evidence that a hosted
run has passed.

## Historical observations

These are maintainer-reported observations from earlier source versions, not a
publicly replayable benchmark result. Raw run directories, model transcripts,
PDFs, and environment-specific artifacts are intentionally not distributed in
this source release.

| Path | Observed result | What it does not establish |
| --- | --- | --- |
| Generated quadratic task, scripted agents, real CPU execution, integrated PaperOrchestra | An earlier controller campaign completed research, writing, review, freeze, final evaluation, audit, and completed-resume checks | Live model quality, autonomous novelty, or current-release validation |
| Latest live Vectorization development attempt | 39 model calls and 1,914,473 observed tokens; development latency ratio `0.0095744` (about 104 times faster than its measured reference); ended `budget_exhausted` before final evaluation | Protected final performance, completed research acceptance, or fully autonomous recovery |

The Vectorization attempt included operator-assisted binding recovery. Its
development measurement remains an adaptive observation from that protocol and
hardware. It is not a final benchmark score. Observed tokens are the recorded
usage quantity, not a dollar cost or a universal upper bound on spending.

## How to interpret checks

- **Scripted tests:** exercise expected controller branches and invalid-input
  rejection. They cannot measure agent research ability.
- **Real CPU campaign:** executes candidate code and scores its predictions under
  the declared task protocol. Scripted decisions remain scripted.
- **Audit replay:** tests stored-artifact integrity, trusted scoring, and the
  selected development re-execution. It does not certify all scientific claims.
- **Completed resume:** verifies reuse of committed work. It does not authorize
  another live call or repair an ambiguous completion.
- **Live completion:** requires a complete campaign under its frozen model,
  source, budget, writer, and final-evaluation contract. A development gain or a
  compiled manuscript alone is insufficient.

For a new result, record the commit, task JSON hash, runtime image identities,
backend, model limits, final status/outcome, accounting completeness, audit scope,
and any operator intervention. Report skipped, failed, and unavailable checks
alongside passes. Keep protected data and credentials out of public reports.
