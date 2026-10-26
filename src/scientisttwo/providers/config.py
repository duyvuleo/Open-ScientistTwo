"""Frozen provider declarations contain environment names, never credentials."""

from datetime import date
import ipaddress
import json
import os
import re
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import ConfigDict, Field, model_validator

from ..models import Record


ANTHROPIC_VERSION = "2026-10-21"
OFFICIAL_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}
DEFAULT_KEY_ENVS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _credential_from_environment(name, environ, *, allow_missing=False):
    if name is None:
        return None
    environment = os.environ if environ is None else environ
    if allow_missing and name not in environment:
        return None
    value = environment.get(name)
    if (not isinstance(value, str) or not value or len(value) > 8192
            or not re.fullmatch(r"[\x21-\x7e]+", value)):
        raise ValueError("Configured credential environment variable is missing or invalid")
    return value


def _contains_credential(value, credential, depth=0):
    if depth > 64:
        # Never treat an uninspectable nested value as evidence of no credential.
        return True
    if isinstance(value, str):
        if credential in value:
            return True
        unescaped = re.sub(r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match[1], 16)), value)
        if credential in unescaped:
            return True
        if value.lstrip().startswith(("{", "[", '"')):
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                return False
            if decoded != value:
                return _contains_credential(decoded, credential, depth + 1)
        return False
    if isinstance(value, dict):
        return any(_contains_credential(key, credential, depth + 1)
                   or _contains_credential(item, credential, depth + 1)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_credential(item, credential, depth + 1) for item in value)
    return False


def assert_credential_free(provider, value, environ=None):
    """Reject a selected credential before persistence, without returning its value.

    Configurations may be created on a machine without the selected environment
    variable. A present but invalid value is an error, even during preparation.
    The provider is already validated (or is in its final validator); deliberately
    do not call model_validate here, which would recurse through this guard.
    """
    credential = _credential_from_environment(provider.api_key_env, environ, allow_missing=True)
    if credential is not None and _contains_credential(value, credential):
        raise ValueError("Selected credential cannot be included in persisted configuration or inputs")


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ModelProvider(Record):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, hide_input_in_errors=True)

    kind: Literal["codex", "openai", "anthropic", "openai_compatible"] = "codex"
    base_url: str | None = Field(default=None, min_length=1, max_length=2048)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    max_output_tokens: int | None = Field(default=None, gt=0, le=1_000_000, strict=True)
    structured_output: Literal["json_schema", "json_object", "prompt"] = "json_schema"
    vision: bool = Field(default=False, strict=True)
    temperature: float | None = Field(default=None, ge=0, le=2, strict=True)
    api_version: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    allow_insecure_http: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def configured_transport(self):
        if self.kind == "codex":
            if (self.base_url is not None or self.api_key_env is not None
                    or self.max_output_tokens is not None or self.api_version is not None
                    or self.temperature is not None or self.vision or self.allow_insecure_http
                    or self.structured_output != "json_schema"):
                raise ValueError("Codex uses its CLI transport; API-only options are not accepted")
            return self
        if self.max_output_tokens is None:
            raise ValueError("API providers require an explicit max_output_tokens bound")
        self.base_url = self.base_url or OFFICIAL_BASE_URLS.get(self.kind)
        if self.base_url is None:
            raise ValueError("A custom compatible provider requires an explicit base_url")
        if any(ord(char) < 33 or ord(char) == 127 for char in self.base_url) or "\\" in self.base_url:
            raise ValueError("Provider base_url contains invalid URL characters")
        try:
            value = urlsplit(self.base_url)
            host, port = value.hostname, value.port
        except ValueError:
            raise ValueError("Provider base_url is not a valid HTTP endpoint") from None
        if (value.scheme not in {"https", "http"} or not host
                or value.username is not None or value.password is not None
                or value.query or value.fragment or port == 0):
            raise ValueError("Provider base_url requires HTTP(S), a host, and no credentials, query or fragment")
        if value.scheme == "http" and not _loopback(host) and not self.allow_insecure_http:
            raise ValueError("Non-loopback HTTP requires explicit allow_insecure_http")
        self.base_url = urlunsplit((value.scheme, value.netloc, value.path.rstrip("/"), "", ""))
        if self.kind in OFFICIAL_BASE_URLS:
            official = urlsplit(OFFICIAL_BASE_URLS[self.kind])
            origin = (value.scheme, host.lower(), port or (443 if value.scheme == "https" else 80))
            official_origin = (official.scheme, official.hostname, 443)
            if origin != official_origin and self.api_key_env is None:
                raise ValueError("A nonofficial provider origin requires an explicit api_key_env")
        if self.api_key_env is None:
            self.api_key_env = DEFAULT_KEY_ENVS.get(self.kind)
        if self.kind == "anthropic":
            self.api_version = self.api_version or ANTHROPIC_VERSION
            try:
                date.fromisoformat(self.api_version)
            except ValueError:
                raise ValueError("Anthropic API version must be a calendar date") from None
            if self.structured_output == "json_object":
                raise ValueError("Anthropic supports explicit json_schema or prompt mode, not json_object")
            if self.temperature is not None:
                if self.api_version >= "2026-10-21":
                    raise ValueError("This Anthropic API version does not accept temperature")
                if self.temperature > 1:
                    raise ValueError("Legacy Anthropic temperature must be at most one")
        elif self.api_version is not None:
            raise ValueError("api_version is an Anthropic-only option")
        assert_credential_free(self, self.model_dump(mode="json"))
        return self

    def credential_check(self, environ=None):
        return readiness(self, environ=environ)


def credential_value(provider, environ=None):
    """Read one declared credential in memory; callers must never serialize it."""
    provider = ModelProvider.model_validate(provider)
    return _credential_from_environment(provider.api_key_env, environ)


def readiness(provider, environ=None):
    """Configuration/environment inspection only; no client, request or model."""
    provider = ModelProvider.model_validate(provider)
    errors = []
    credential_configured = None
    if provider.kind != "codex" and provider.api_key_env is not None:
        try:
            credential_value(provider, environ=environ)
            credential_configured = True
        except ValueError:
            errors.append("credential_environment_missing_or_invalid")
            credential_configured = False
    return {"ready": not errors, "provider": provider.kind, "api_key_env": provider.api_key_env,
            "credential_configured": credential_configured, "errors": errors,
            "model_access": "unverified_without_inference", "live_model_calls": 0}
