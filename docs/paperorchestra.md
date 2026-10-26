# Bundled PaperOrchestra writer

Open-ScientistTwo includes the files needed for an adapted, bounded
[PaperOrchestra writing engine](https://github.com/google-research/paper-orchestra/tree/ca1b3fa01c2970fc7cda32d16245db38d57b3f56)
at revision `ca1b3fa01c2970fc7cda32d16245db38d57b3f56`. Selecting the
`paperorchestra` writer uses this bundle by default. A second checkout is not
required.

The bundle contains 17 hash-locked upstream files covering the selected writing
agents, internal reviewer, prompts, utilities, templates, and upstream license.
It is shipped in the source tree and Python distribution. The integration
verifies the file hashes before use and adapts model calls and PDF operations
to bounded services. Frozen supplied literature replaces upstream retrieval.
The full upstream frontend, retrieval system, and optional plotting pipeline are
outside this bundle. See [third-party notices](../THIRD_PARTY_NOTICES.md) for
Apache-2.0 attribution and file-specific template terms.

Scripted responses or explicitly enabled live calls supply the agent outputs.
PaperOrchestra's internal reviewer is not ScholarPeer. The default offline
quickstart retains the canonical writer; selecting PaperOrchestra additionally
requires the PDF container below.

## Prepare the PDF container

The [PDF Dockerfile](../docker/paperorchestra/Dockerfile) requires two explicit
build inputs: an official Python 3.12 patch-version `slim-bookworm` image pinned
by SHA-256 digest for your architecture, and a Debian snapshot timestamp. Set
`SCIENTISTTWO_PYTHON_BASE` to that complete image reference and
`SCIENTISTTWO_DEBIAN_SNAPSHOT` to the chosen `YYYYMMDDTHHMMSSZ` snapshot. These
values must be real pins; mutable tags and placeholders are rejected.

```sh
docker build \
  --build-arg PYTHON_BASE="$SCIENTISTTWO_PYTHON_BASE" \
  --build-arg DEBIAN_SNAPSHOT="$SCIENTISTTWO_DEBIAN_SNAPSHOT" \
  -f docker/paperorchestra/Dockerfile \
  -t scientisttwo-paperorchestra:local .
export SCIENTISTTWO_PDF_IMAGE="$(docker image inspect scientisttwo-paperorchestra:local --format '{{.Id}}')"
```

The build needs network access. Execution uses the resulting immutable image
identity, disables network and TeX shell escape, and applies resource and artifact
limits. Missing PDF tools cause an explicit failure; there is no host compiler
fallback. For a named Docker context, use the same `--context CONTEXT` for build,
image inspection, and the campaign.

## Run the controller with scripted writing agents

Prepare the CPU worker using the [quickstart](../README.md#start-with-the-cpu-example),
then run:

```sh
uv run scientisttwo run examples/quickstart.json \
  --writer-backend paperorchestra \
  --max-prompt-chars 1000000 \
  --compiler-image "$SCIENTISTTWO_PDF_IMAGE" --require-pdf
uv run scientisttwo inspect RUN_ID
uv run scientisttwo audit RUN_ID
uv run scientisttwo resume RUN_ID
```

Use the emitted directory basename for `RUN_ID`. Experiments and PDF operations
are real container executions; all agent responses remain scripted and make zero
live model calls. Completion depends on the saved controller status and audit,
not on the mere existence of a PDF.

The example explicitly raises the frozen input-character ceiling from the
default 200,000 to 1,000,000 because writing receives a larger evidence and
manuscript context. This bounds serialized agent input; it is not a model-token
budget or an output-size limit. The example still makes zero live model calls.
Resume retains the saved ceiling.

The controller owns scientific decisions and experiment scheduling. The writer
receives frozen development evidence and preserves its authoritative LaTeX/PDF
through revisions. A source index connects manuscript claims to recorded
evidence. Final results enter only after method freeze. Writing, binding, and
PDF operations share campaign accounting. See [validation status](validation.md)
for the distinction between historical completion and current-source evidence.

## Write from already frozen evidence

The separate command below does not run experiments or assess acceptance by the
research controller. Its default backend is `fixture`:

```sh
uv run scientisttwo paperorchestra run \
  --materials examples/paperorchestra/materials \
  --output runs/frozen-writing-example \
  --compiler-image "$SCIENTISTTWO_PDF_IMAGE"
uv run scientisttwo paperorchestra inspect runs/frozen-writing-example
uv run scientisttwo paperorchestra audit runs/frozen-writing-example
```

The [included materials](../examples/paperorchestra/README.md) are a historical,
nonfinal aggregate-evidence fixture. This command does not recompute its metrics.
A completed output is verified and reused. Preserve a partial output and choose
a new destination for a separately authorized attempt.

For live writing, explicitly add `--backend codex --model MODEL --allow-live`,
along with `--max-calls`, `--max-tokens`, and `--timeout-seconds`. Token limits
stop before the next call using observed usage; the final completed call can
cross the threshold. A small budget can end without a complete manuscript.
See [live execution](live.md) before making model calls.

## Advanced: use an external pinned checkout

For source inspection or an explicitly selected external copy, obtain the same
revision in a new directory:

```sh
mkdir -p .upstream
git clone https://github.com/google-research/paper-orchestra.git .upstream/paper-orchestra
git -C .upstream/paper-orchestra checkout --detach ca1b3fa01c2970fc7cda32d16245db38d57b3f56
export SCIENTISTTWO_PAPERORCHESTRA_ROOT="$PWD/.upstream/paper-orchestra"
```

Add `--paperorchestra-root "$SCIENTISTTWO_PAPERORCHESTRA_ROOT"` to controller
`run`, or `--upstream "$SCIENTISTTWO_PAPERORCHESTRA_ROOT"` to standalone
`paperorchestra run`. These explicit overrides remain subject to the same
locked file hashes. Keep the external copy unchanged; changing a source file
does not bypass the pin. The integration does not require the entire upstream
dependency stack.
