"""One bounded HTTP request in a killable process, with no automatic retries.

Only the selected credential environment variable is read. API keys, response
headers, arbitrary server errors and Python exception text never leave this
worker. The parent records an unknown outcome if this process times out or dies.
"""

import json
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from ..backends import strict_json_loads
from .config import credential_value
from .protocols import (
    REQUEST_LIMIT, RESPONSE_LIMIT, build_request, normalize_response, outcome,
    response_hash, validate_request,
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _read_bounded(response, deadline):
    chunks, size = [], 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Request deadline reached")
        data = response.read(min(65536, RESPONSE_LIMIT + 1 - size))
        if not isinstance(data, bytes):
            raise ValueError("Invalid HTTP body")
        if not data:
            return b"".join(chunks)
        size += len(data)
        if size > RESPONSE_LIMIT:
            raise ValueError("HTTP response exceeds its bound")
        chunks.append(data)


def _contains_secret(value, secret, depth=0):
    if secret is None:
        return False
    if depth > 64:
        return False
    if isinstance(value, str):
        if secret in value:
            return True
        # Provider output text is often a JSON document inside outer JSON. Check
        # its decoded form too, without changing the scientific response.
        unescaped = re.sub(r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match[1], 16)), value)
        if secret in unescaped:
            return True
        if value.lstrip().startswith(("{", "[", '"')):
            try:
                decoded = strict_json_loads(value)
            except (ValueError, RecursionError):
                return False
            if decoded != value:
                return _contains_secret(decoded, secret, depth + 1)
        return False
    if isinstance(value, dict):
        return any(_contains_secret(key, secret, depth + 1) or _contains_secret(item, secret, depth + 1)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret(item, secret, depth + 1) for item in value)
    return False


def execute_request(value, *, opener=None, environ=None):
    """Execute at most one request; injected opener/environ are for local tests."""
    zero = {"input_tokens": 0, "output_tokens": 0}
    try:
        parsed = validate_request(value)
        url, body, headers = build_request(parsed)
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > REQUEST_LIMIT:
            raise ValueError("Request exceeds its byte bound")
    except (ValueError, TypeError, UnicodeError, RecursionError, KeyError):
        return outcome("not_sent", request_sent=False, usage=zero, error_code="invalid_request")
    try:
        credential = credential_value(parsed.provider, environ=environ)
    except ValueError:
        return outcome("not_sent", request_sent=False, usage=zero, error_code="credential_environment_missing_or_invalid")
    if credential is not None and _contains_secret(body, credential):
        return outcome("not_sent", request_sent=False, usage=zero, error_code="credential_in_request_body")
    if credential is not None:
        headers["x-api-key" if parsed.provider.kind == "anthropic" else "Authorization"] = (
            credential if parsed.provider.kind == "anthropic" else "Bearer " + credential)
    try:
        request = Request(url, data=encoded, headers=headers, method="POST")
        # Proxy environment and platform proxy discovery are deliberately disabled.
        # urllib does not consult .netrc, and HTTPS keeps its verified default TLS.
        client = opener or build_opener(ProxyHandler({}), _NoRedirect())
    except Exception:
        return outcome("not_sent", request_sent=False, usage=zero, error_code="invalid_request")
    deadline = time.monotonic() + parsed.timeout_seconds
    http_status, raw = None, None
    try:
        try:
            response = client.open(request, timeout=parsed.timeout_seconds)
        except HTTPError as error:
            response = error
        with response:
            http_status = getattr(response, "status", None) or getattr(response, "code", None)
            if type(http_status) is not int or not 100 <= http_status <= 599:
                raise ValueError("Invalid HTTP status")
            raw = _read_bounded(response, deadline)
    except (TimeoutError, URLError, OSError):
        return outcome("transport_error", request_sent=True, http_status=http_status,
                       error_code="transport_failed")
    except Exception:
        return outcome("invalid_response", request_sent=True, http_status=http_status,
                       error_code="response_read_failed")
    fingerprint = response_hash(raw)
    try:
        payload = strict_json_loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError):
        result = (normalize_response(parsed.provider.kind, {}, http_status=http_status)
                  if not 200 <= http_status < 300 else
                  outcome("invalid_response", request_sent=True, http_status=http_status,
                          error_code="invalid_response_json"))
        result["response_sha256"] = fingerprint
        return result
    try:
        result = normalize_response(parsed.provider.kind, payload, http_status=http_status)
        # JSON escaping or separate output blocks must not bypass the echo guard.
        if (_contains_secret(payload, credential) or _contains_secret(result, credential)):
            result = outcome("invalid_response", request_sent=True, http_status=http_status,
                             usage=result["usage"], error_code="credential_echo_detected")
        elif result["output_text"] is not None:
            try:
                result["output_text"].encode("utf-8")
            except UnicodeError:
                result = outcome("invalid_response", request_sent=True, http_status=http_status,
                                 usage=result["usage"], error_code="invalid_output_encoding")
        result["response_sha256"] = fingerprint
        return result
    except Exception:
        return outcome("invalid_response", request_sent=True, http_status=http_status,
                       response_sha256=fingerprint, error_code="response_normalization_failed")


def main():
    try:
        raw = sys.stdin.buffer.read(REQUEST_LIMIT + 1)
        if len(raw) > REQUEST_LIMIT:
            raise ValueError("Oversized request")
        value = strict_json_loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError):
        result = outcome("not_sent", request_sent=False, usage={"input_tokens": 0, "output_tokens": 0},
                         error_code="invalid_request")
    else:
        try:
            result = execute_request(value)
        except Exception:
            # A programming/process failure is not evidence of zero usage.
            result = outcome("transport_error", request_sent=True, error_code="worker_failed")
    sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
