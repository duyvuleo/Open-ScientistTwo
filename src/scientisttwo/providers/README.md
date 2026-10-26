# Live model providers

Open-ScientistTwo can use commercial APIs or a custom model server through the
same research controller. This package implements response-only API calls:
models return structured proposals, critiques and writing artifacts, while the
controller owns workflow transitions, experiment execution and trusted scoring.

Choose **one provider and model per campaign**. Suite assignments can use
different configurations; routing individual roles to different models is not
implemented. Codex CLI remains the default when `provider` is omitted.

## Supported transports

| `provider.kind` | Interface | Default endpoint | Credential configuration |
| --- | --- | --- | --- |
| `codex` | Codex CLI | Local `codex exec` | CLI authentication configured separately |
| `openai` | OpenAI Responses | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages, including Claude | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `openai_compatible` | OpenAI-compatible Chat Completions | Explicit `base_url` required | Optional, explicitly named environment variable |

The names above identify transports, not a model catalog. Supply the model ID
available to your provider. A custom server must implement the selected API
format and report input/output token usage. Deploying that server, obtaining
weights and selecting a model revision are separate steps. There is no automatic
model substitution or provider fallback.

The Codex backend lives in [`../backends.py`](../backends.py); this package
implements the three HTTP transports.

## Configure and run

Run these commands from the repository root, after completing the
[CPU setup](../../../README.md#start-with-the-cpu-example). Choose a template:

- [OpenAI](../../../examples/live-options.openai.example.json)
- [Anthropic / Claude](../../../examples/live-options.anthropic.example.json)
- [Custom compatible server](../../../examples/live-options.custom.example.json)
- [Codex CLI](../../../examples/live-options.example.json)

For example, copy the OpenAI template:

```sh
cp examples/live-options.openai.example.json live-options.local.json
```

Replace `MODEL_ID` with an available model ID and review the campaign limits.
Configure the named credential in your environment or secret manager.
**`api_key_env` contains the variable name, never its value.** Do not put keys in
configuration files, prompts or command arguments. The repository ignores
`*.local.json`; the supplied examples contain no credentials.

Resolve the prepared image to its immutable ID, then create and check the frozen
manifest without making an inference request:

```sh
SCIENTISTTWO_WORKER_IMAGE="$(docker image inspect scientisttwo-worker:0.1 --format '{{.Id}}')"
uv run scientisttwo live-pilot create \
  examples/quickstart.json live-options.local.json live-manifest.local.json \
  --image "$SCIENTISTTWO_WORKER_IMAGE"
uv run scientisttwo live-pilot check live-manifest.local.json
```

Preflight checks local configuration, credential presence, source identity and
container images. It does not verify provider access or model capabilities.
When you intend to permit live requests, run:

```sh
uv run scientisttwo live-pilot run \
  live-manifest.local.json runs/live-example --allow-live
uv run scientisttwo live-pilot resume runs/live-example --allow-live
```

API campaigns require the `research-v2` engine, explicit call/token/time limits,
positive `provider.max_output_tokens`, and `model_retries: 0`. The API templates
set the retry policy; `live-pilot` selects the research engine. The native
`live create` command also exposes provider options. See the
[complete provider guide](../../../docs/model-providers.md) and
[live execution guide](../../../docs/live.md) for the surrounding workflow.

### Custom models

The following is a fragment of a full live-options file. Use the
[custom template](../../../examples/live-options.custom.example.json) to include
the required campaign budgets:

```json
{
  "model": "SERVED_MODEL_ID",
  "provider": {
    "kind": "openai_compatible",
    "base_url": "http://127.0.0.1:8000/v1",
    "max_output_tokens": 16384,
    "structured_output": "json_schema",
    "vision": false
  }
}
```

Start the server separately and replace `SERVED_MODEL_ID` with its advertised
model ID. This example uses no authentication. If your server requires it, add
`"api_key_env": "CUSTOM_MODEL_API_KEY"` and configure that variable externally.
The loopback address refers to the machine running the research controller.

HTTPS is required except for loopback addresses or explicitly opted-in insecure
HTTP. URLs cannot contain embedded credentials, query parameters or fragments.
Requests do not follow redirects, and a custom endpoint never inherits an API
key implicitly.

## Structured output and PaperOrchestra

| Mode | Behavior |
| --- | --- |
| `json_schema` | Requests schema-constrained generation; the default. |
| `json_object` | Requests JSON mode on OpenAI and compatible transports; rejected for Anthropic. |
| `prompt` | Includes the required schema in the prompt without requesting provider-side constraints. |

All modes still use the mapping codec and local Pydantic validation. Invalid,
truncated, refused or tool-bearing responses fail the invocation. No execution
tools are provided to the model, and unsupported modes are not retried using a
different format.

API-backed [PaperOrchestra](../../../docs/paperorchestra.md) uses the campaign's
selected model for writing and review. It requires `vision: true` and a model
that accepts PNG/JPEG inputs, because rendered pages are sent as image content.
The configuration examples use the canonical writer and `vision: false`.
Selecting PaperOrchestra also requires its PDF runtime configuration.

The legacy engine, standalone PaperOrchestra CLI, and historical remote
paired-study/repair paths retain their Codex-specific live transport. Use
`live-pilot` or the native `live` research path for API-backed campaigns.

## Accounting, audit and recovery

Before sending a request, the backend records an immutable claim. After the
worker returns, it validates the response and stores a receipt containing known
usage or an explicit unknown outcome. The ledger is kept inside the campaign:

```text
api-model-ledger/
  policy.json
  calls/<call-id>/claim.json
  calls/<call-id>/receipt.json
```

Requests, accepted results and receipt copies are also bound to the campaign's
content-addressed artifact store. Configuration records contain environment
names; authentication headers and credential values are not ledger fields.

- Known usage is charged even when an output fails validation.
- Missing usage, interrupted calls and ambiguous completions block further
  dispatch. They are never counted as zero-cost completed calls.
- Transport calls are not automatically retried. A local timeout stops the
  worker process; it does not prove remote inference stopped.
- Token usage is observed after completion. The last call may cross the stop
  threshold, which is not a dollar spending cap.
- Resume reuses committed work. A completed campaign makes no new model call.
  Changes to the frozen provider, model, source or policy require a new campaign.

Keep unresolved evidence intact. Do not delete a claim or change a receipt to
force a retry. Accounting for a custom server depends on the usage it reports;
the controller cannot independently measure the server's billed tokens.

## Module map

| File | Responsibility |
| --- | --- |
| [`config.py`](config.py) | Provider settings, endpoint validation and named credential checks. |
| [`protocols.py`](protocols.py) | Request/response formats, schema projection, images and normalized usage. |
| [`worker.py`](worker.py) | One bounded HTTP request in an isolated subprocess; no automatic retries. |
| [`backend.py`](backend.py) | The `invoke(role, inputs, output_schema)` interface, durable ledger, budget checks and local output validation. |
| [`audit.py`](audit.py) | Independent replay of ledger checks and reconciliation with controller accounting. |

Use the campaign entry points for execution; calling a transport alone does not
provide the controller's complete experiment and recovery protocol.

## Tests and validation limits

Run the provider tests from the repository root:

```sh
uv run --frozen pytest -q \
  tests/test_provider_protocols.py tests/test_provider_wiring.py \
  tests/test_api_backend.py tests/test_api_process_transport.py \
  tests/test_provider_audit.py tests/test_api_research_cycle.py
```

These tests use synthetic responses, local HTTP servers and scripted experiment
fixtures. They cover protocol handling, credential exclusion, accounting,
timeouts, tampering, workflow completion and resume. They make no real model
calls. The separate real CPU/PaperOrchestra check uses scripted agents and
container-executed experiments.

See the [source-bound verification record](../../../docs/provider-verification.json)
and [validation status](../../../docs/validation.md). Commercial model access,
compatibility of every custom server, and live autonomous research quality
remain unverified by these checks.
