# Live model providers

The same research controller can use Codex CLI, the OpenAI Responses API, the
Anthropic Messages API, or an OpenAI-compatible Chat Completions endpoint.
Select one provider and model per campaign. The selected backend handles the
research roles and, when enabled, PaperOrchestra's writing and review calls.
Frozen suite assignments may select different providers; per-role model routing
is not part of this interface.

| `provider.kind` | Transport | Authentication configuration |
| --- | --- | --- |
| `codex` | Existing response-only Codex CLI | Configure CLI authentication outside the repository |
| `openai` | Responses API, default `https://api.openai.com/v1` | Environment name, default `OPENAI_API_KEY` |
| `anthropic` | Messages API, default `https://api.anthropic.com/v1` | Environment name, default `ANTHROPIC_API_KEY` |
| `openai_compatible` | Chat Completions at an explicitly supplied base URL | Optional explicitly selected environment name |

Model identifiers are supplied by the user. The implementation does not maintain
a model-name catalog or silently substitute models. A custom endpoint must
implement the selected request/response format and report input/output token
usage. Hosting the model, obtaining its weights, and declaring the served model
revision remain separate responsibilities.

## Configure a campaign

Use one of these templates with the [live execution guide](live.md):

- [OpenAI](../examples/live-options.openai.example.json)
- [Anthropic/Claude](../examples/live-options.anthropic.example.json)
- [Custom compatible endpoint](../examples/live-options.custom.example.json)
- [Codex CLI](../examples/live-options.example.json), which is also the default
  when a provider declaration is omitted.

Copy the chosen template to a `*.local.json` file and replace `MODEL_ID` or
`SERVED_MODEL_ID` with an identifier available to that endpoint. Configure the
credential through the named environment variable or your environment manager.
Do not put its value in the JSON file, command arguments, task data, or repository.
`api_key_env` accepts a variable name only; a literal `api_key` field is rejected.

A minimal custom declaration is:

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

This is the provider portion of a full live-options file; campaign, call, token,
and experiment limits are still required. The example's local endpoint does
not select a credential implicitly. For an authenticated custom service, add
`"api_key_env": "CUSTOM_MODEL_API_KEY"` and configure that variable externally.

API calls require an explicit positive `max_output_tokens` value. Set it to a
limit supported by the selected model and appropriate for the expected artifact.
A model may still return an incomplete artifact at that limit; incomplete
responses are rejected and their known usage remains charged.

Base URLs may not contain user information, query parameters, or fragments.
HTTPS is required except for loopback addresses. A trusted non-loopback HTTP
deployment requires the explicit `allow_insecure_http` option; credentials and
research inputs would then travel without TLS. Requests do not follow redirects.

## Structured outputs and capabilities

The default `json_schema` mode requests provider-side schema constraints. The
existing mapping codec and original Pydantic validation still apply locally.
Provider-side constrained generation is not a substitute for scientific review.

For a compatible model/server that lacks that facility, choose an explicit
alternative:

- `json_object` requests JSON mode. It is available on OpenAI/compatible paths,
  and is rejected for Anthropic.
- `prompt` supplies the schema in the request without a provider format field.
  Local parsing and schema validation remain mandatory.

No request automatically falls back between modes, providers, endpoints or
models. Unsupported parameters, refusals, tool calls, truncated output and
invalid responses fail the invocation. Models do not receive execution tools.

The API provider's `vision` declaration defaults to `false`. Set it to `true`
only for an endpoint/model that accepts the supplied PNG/JPEG image blocks.
PaperOrchestra requires this declaration because it reviews rendered pages;
text-only models can use the canonical writer. Page images are sent as actual
image content, not silently omitted or replaced with text.

Anthropic's dated API version is frozen in the provider configuration. The
current default is `2026-10-21`; its API does not accept `temperature`. An explicit
older compatible version can be selected when needed. Temperature is otherwise
omitted by default; support remains model-specific.

## Preflight and execution

Creating and checking a manifest makes no inference request:

```sh
uv run scientisttwo live-pilot create \
  examples/quickstart.json live-options.local.json live-manifest.local.json \
  --image "$SCIENTISTTWO_WORKER_IMAGE"
uv run scientisttwo live-pilot check live-manifest.local.json
```

The check validates local configuration, credential presence, frozen source and
image availability. Provider access and model capability remain unverified until
a separately authorized request runs. API providers do not require Codex CLI.

Execution still requires `--allow-live`:

```sh
uv run scientisttwo live-pilot run \
  live-manifest.local.json runs/live-example --allow-live
uv run scientisttwo live-pilot resume runs/live-example --allow-live
```

For a native task, `live create` exposes `--provider`, `--base-url`,
`--api-key-env`, `--max-output-tokens`, `--structured-output`, `--vision` and
optional provider controls. It requires the task's existing readiness gates.
`live check` performs provider-specific checks without contacting a model.

The existing `--executable` option is specific to Codex. Remote paired-study
relays, their historical repair operators, and the standalone PaperOrchestra CLI
retain their Codex-specific live transport. The `legacy-live-v1` engine also
remains Codex-only; API providers require `research-v2`. Use the shared `live-pilot` or native
`live` research paths for API-backed PaperOrchestra campaigns.

## Accounting and recovery

Each API attempt has a durable claim and a completion/usage receipt. Credentials
and authentication headers are not included in those records. Campaign provider,
endpoint, schema mode, model and limits are frozen; changing them requires a new
campaign rather than rewriting a saved manifest.

Known failure usage consumes the budget. Missing usage, interrupted calls and
ambiguous network outcomes remain unknown and block additional dispatch. The
client does not automatically retry a request that may have been billed. A local
timeout cancels the worker process; it does not prove remote inference stopped.

Recorded input totals include applicable reported cache-input tokens. Reasoning
and cached subtotals are not counted twice. The observed-token threshold is not
a dollar billing cap: usage is reported after completion, and the final call may
cross it. Custom servers must report trustworthy usage for this accounting to
be meaningful.

Completed resume performs no model call. Native suite accounting aggregates the
same charged operations and stops further model admission when API usage is
unresolved. Protected final evidence never returns to a research role.

## Verification boundary and sources

Protocol, authorization, accounting, failure and recovery behavior is tested with
simulated responses and local test servers. This does not establish access to a
commercial model, compatibility of every custom server, or successful live
research. No paid provider call is part of the default test suite.

- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
