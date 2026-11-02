"""Tests for deploy/appwrite/main.py: credentials, forwarding and response mapping.

The real app.main is replaced with a stub, so these tests need neither FastAPI nor
the models.
"""

import base64
import contextlib
import importlib
import os
import sys
import types

SECRET = "unit-test-secret"
ENDPOINT = "/v1/extract/pipeline/full"


class EchoApp:
    """Stands in for app.main:app. Records what it receives and answers with JSON."""

    def __init__(self):
        self.calls = []
        self.reply_body = b'{"status":"ok"}'

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            raise RuntimeError("no lifespan in this fake")
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        self.calls.append(
            {
                "method": scope["method"],
                "path": scope["path"],
                "query": scope["query_string"],
                "headers": {name.decode(): value.decode() for name, value in scope["headers"]},
                "body": body,
            }
        )
        await send(
            {
                "type": "http.response.start",
                "status": 202,
                "headers": [(b"content-type", b"application/json"), (b"content-length", b"999")],
            }
        )
        await send({"type": "http.response.body", "body": self.reply_body})


class FakeRequest:
    def __init__(
        self, path, *, method="POST", headers=None, query_string="", body=b"", with_body=True
    ):
        self.path = path
        self.method = method
        self.headers = dict(headers or {})
        self.query_string = query_string
        if with_body:
            self.body_binary = body


class FakeResponse:
    def text(self, body, status=200, headers=None):
        return ("text", body, status, headers)

    def binary(self, body, status=200, headers=None):
        return ("binary", body, status, headers)


class FakeContext:
    def __init__(self, req):
        self.req = req
        self.res = FakeResponse()
        self.errors = []

    def error(self, message):
        self.errors.append(message)


def basic(secret=SECRET, user="ai"):
    token = base64.b64encode(f"{user}:{secret}".encode()).decode()
    return f"Basic {token}"


@contextlib.contextmanager
def loaded_entrypoint(secret=SECRET):
    """Load deploy.appwrite.main with a stub app.main, and restore the environment afterwards."""
    fake_app = EchoApp()
    package = types.ModuleType("app")
    app_main = types.ModuleType("app.main")
    app_main.app = fake_app
    saved_modules = {name: sys.modules.get(name) for name in ("app", "app.main")}
    saved_secret = os.environ.get("AI_SERVICE_SECRET")
    sys.modules["app"] = package
    sys.modules["app.main"] = app_main
    if secret is None:
        os.environ.pop("AI_SERVICE_SECRET", None)
    else:
        os.environ["AI_SERVICE_SECRET"] = secret
    entry = importlib.import_module("deploy.appwrite.main")
    entry._runner = None
    try:
        yield entry, fake_app
    finally:
        if entry._runner is not None:
            entry._runner.close()
            entry._runner = None
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if saved_secret is None:
            os.environ.pop("AI_SERVICE_SECRET", None)
        else:
            os.environ["AI_SERVICE_SECRET"] = saved_secret


def test_refuses_every_request_when_the_secret_is_not_configured():
    with loaded_entrypoint(secret=None) as (entry, fake):
        result = entry.main(FakeContext(FakeRequest(ENDPOINT, headers={"authorization": basic()})))
    assert result[0] == "text"
    assert result[2] == 503
    assert fake.calls == []


def test_missing_credentials_get_401_with_a_basic_challenge():
    with loaded_entrypoint() as (entry, fake):
        result = entry.main(FakeContext(FakeRequest(ENDPOINT)))
    assert result[2] == 401
    assert result[3]["www-authenticate"].startswith("Basic")
    assert fake.calls == []


def test_wrong_password_wrong_scheme_or_garbage_get_401():
    with loaded_entrypoint() as (entry, fake):
        wrong_password = entry.main(
            FakeContext(FakeRequest(ENDPOINT, headers={"authorization": basic("guess")}))
        )
        bearer = entry.main(
            FakeContext(FakeRequest(ENDPOINT, headers={"authorization": f"Bearer {SECRET}"}))
        )
        garbage = entry.main(
            FakeContext(FakeRequest(ENDPOINT, headers={"authorization": "Basic !!!"}))
        )
    assert [wrong_password[2], bearer[2], garbage[2]] == [401, 401, 401]
    assert fake.calls == []


def test_valid_credentials_forward_the_request_and_drop_transport_headers():
    headers = {
        "authorization": basic(),
        "content-type": "multipart/form-data; boundary=abc",
        "x-correlation-id": "sha256:1",
        "x-appwrite-key": "ephemeral-key",
        "host": "ai.example.test",
        "accept-encoding": "gzip",
    }
    request = FakeRequest(ENDPOINT, headers=headers, query_string="dry_run=1", body=b"hello")
    with loaded_entrypoint() as (entry, fake):
        result = entry.main(FakeContext(request))

    assert result[0] == "text"
    call = fake.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == ENDPOINT
    assert call["query"] == b"dry_run=1"
    assert call["body"] == b"hello"
    assert call["headers"]["content-type"] == "multipart/form-data; boundary=abc"
    assert call["headers"]["x-correlation-id"] == "sha256:1"
    assert call["headers"]["content-length"] == "5"
    for hidden in ("authorization", "x-appwrite-key", "host", "accept-encoding"):
        assert hidden not in call["headers"]


def test_status_and_body_pass_through_and_transport_headers_are_dropped():
    request = FakeRequest(ENDPOINT, headers={"authorization": basic()}, body=b"abc")
    with loaded_entrypoint() as (entry, fake):
        result = entry.main(FakeContext(request))
    kind, body, status, headers = result
    assert (kind, status) == ("text", 202)
    assert body == '{"status":"ok"}'
    assert headers == {"content-type": "application/json"}


def test_non_utf8_replies_are_returned_as_binary():
    request = FakeRequest(ENDPOINT, headers={"authorization": basic()})
    with loaded_entrypoint() as (entry, fake):
        fake.reply_body = b"\xff\xfe\x00"
        result = entry.main(FakeContext(request))
    assert result[0] == "binary"
    assert result[1] == b"\xff\xfe\x00"


def test_missing_body_binary_is_a_logged_500_not_a_silent_corruption():
    context = FakeContext(
        FakeRequest(ENDPOINT, headers={"authorization": basic()}, with_body=False)
    )
    with loaded_entrypoint() as (entry, fake):
        result = entry.main(context)
    assert result[2] == 500
    assert any("body_binary" in message for message in context.errors)
    assert fake.calls == []


def test_non_ascii_credentials_do_not_crash_the_check():
    with loaded_entrypoint() as (entry, fake):
        result = entry.main(
            FakeContext(FakeRequest(ENDPOINT, headers={"authorization": "Basic ÿÿÿ"}))
        )
    assert result[2] == 401
    assert fake.calls == []
