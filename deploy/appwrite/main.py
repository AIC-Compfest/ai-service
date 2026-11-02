"""Appwrite Functions entrypoint for the ai-service pipeline.

Appwrite calls ``main(context)`` once per HTTP execution. The request is forwarded
unchanged to the FastAPI application that the Dockerfile serves (``app.main:app``),
and that application's response is returned. Nothing under ``app/``, ``config/``
or ``models/`` changes, so the Docker/uvicorn deployment keeps working as before.

Access control: the backend cannot send extra headers, but its ``AI_SERVICE_URL``
may carry credentials. Set it to ``https://<user>:<AI_SERVICE_SECRET>@<domain>``.
Go's HTTP client sends the password as HTTP Basic auth on every request, and it
masks the password in its error messages, so the secret is not echoed back in the
backend's API error responses. Requests without the right password get 401. If
``AI_SERVICE_SECRET`` is not set, every request is refused.
"""

from __future__ import annotations

import atexit
import base64
import hmac
import json
import os
import sys
import threading
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.appwrite.asgi_bridge import ASGIRunner, BridgeResponse  # noqa: E402

SECRET_ENV = "AI_SERVICE_SECRET"

# Caches and rendered pages go to /tmp, the writable scratch area in a function
# container. Values set in the Appwrite console take precedence (setdefault).
_ENV_DEFAULTS = {
    "PAGE_OUTPUT_DIR": "/tmp/ai-service/pages",
    "HF_HOME": "/tmp/huggingface",
    "XDG_CACHE_HOME": "/tmp/cache",
    "TORCH_HOME": "/tmp/torch",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",  # same value as the Dockerfile
}
for _name, _value in _ENV_DEFAULTS.items():
    os.environ.setdefault(_name, _value)
if not os.access(os.path.expanduser("~"), os.W_OK):
    os.environ["HOME"] = "/tmp"

# Headers that belong to the transport or to Appwrite, not to the caller's request.
_DROPPED_REQUEST_HEADERS = frozenset(
    {
        "accept-encoding",
        "connection",
        "content-length",
        "expect",
        "host",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_DROPPED_RESPONSE_HEADERS = frozenset(
    {"connection", "content-length", "keep-alive", "transfer-encoding"}
)

_runner: ASGIRunner | None = None
_runner_lock = threading.Lock()


def main(context: Any) -> Any:
    secret = os.environ.get(SECRET_ENV, "").strip()
    if not secret:
        return _error(context, 503, f"{SECRET_ENV} is not set; all requests are refused")

    req = context.req
    if not _has_valid_credentials(req.headers or {}, secret):
        challenge = {"www-authenticate": 'Basic realm="ai-service"'}
        return _error(context, 401, "Unauthorized", challenge)

    try:
        body = _request_body(req)
        headers = _forward_headers(req.headers or {})
        query_string = str(getattr(req, "query_string", "") or "")
        with _runner_lock:
            response = _get_runner().request(
                str(req.method),
                str(req.path or "/"),
                query_string=query_string,
                headers=headers,
                body=body,
            )
    except Exception as exc:
        # The caller gets a generic message; the details stay in the function log.
        context.error(f"ai-service request failed: {exc!r}\n{traceback.format_exc()}")
        return _error(context, 500, "Internal Server Error")

    return _reply(context, response)


def _has_valid_credentials(headers: Mapping[str, Any], secret: str) -> bool:
    """True when Authorization is HTTP Basic and its password equals the secret."""
    value = str(headers.get("authorization", "") or "")
    scheme, _, encoded = value.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return False
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    _, _, password = decoded.partition(":")
    return hmac.compare_digest(password.encode("utf-8"), secret.encode("utf-8"))


def _request_body(req: Any) -> bytes:
    if not hasattr(req, "body_binary"):
        raise RuntimeError("this Appwrite runtime does not expose req.body_binary")
    raw = req.body_binary
    if raw is None:
        return b""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    raise RuntimeError(f"req.body_binary has unexpected type {type(raw).__name__}")


def _forward_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for name, value in headers.items():
        key = str(name).lower()
        if key == "authorization" or key.startswith("x-appwrite-"):
            continue
        if key in _DROPPED_REQUEST_HEADERS:
            continue
        forwarded[key] = str(value)
    return forwarded


def _reply(context: Any, response: BridgeResponse) -> Any:
    headers = {
        name: value for name, value in response.headers if name not in _DROPPED_RESPONSE_HEADERS
    }
    try:
        text = response.body.decode("utf-8")
    except UnicodeDecodeError:
        return context.res.binary(response.body, response.status, headers)
    return context.res.text(text, response.status, headers)


def _error(
    context: Any, status: int, message: str, extra_headers: Mapping[str, str] | None = None
) -> Any:
    headers = {"content-type": "application/json", **(extra_headers or {})}
    return context.res.text(json.dumps({"detail": message}), status, headers)


def _get_runner() -> ASGIRunner:
    """Build the runner on first use, so refused requests never import the models."""
    global _runner
    if _runner is None:
        from app.main import app as asgi_app  # the same app the Dockerfile serves

        _runner = ASGIRunner(asgi_app)
    return _runner


def _close_runner() -> None:
    """Run the app's shutdown lifespan when the process exits, if it was started."""
    global _runner
    with _runner_lock:
        if _runner is not None:
            _runner.close()
            _runner = None


atexit.register(_close_runner)
