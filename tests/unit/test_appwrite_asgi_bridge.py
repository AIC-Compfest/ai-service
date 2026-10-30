"""Tests for deploy/appwrite/asgi_bridge.py. Standard library and asyncio only; no network."""

from deploy.appwrite.asgi_bridge import ASGIRunner, LifespanStartupError


class RecordingApp:
    """ASGI app that echoes the request body reversed and records each HTTP scope."""

    def __init__(self, *, lifespan_supported: bool = True, startup_error: str | None = None):
        self.lifespan_supported = lifespan_supported
        self.startup_error = startup_error
        self.startups = 0
        self.shutdowns = 0
        self.scopes: list[dict] = []

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            if not self.lifespan_supported:
                raise RuntimeError("this fake app has no lifespan support")
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    if self.startup_error is not None:
                        failure = {"type": "lifespan.startup.failed", "message": self.startup_error}
                        await send(failure)
                        return
                    self.startups += 1
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    self.shutdowns += 1
                    await send({"type": "lifespan.shutdown.complete"})
                    return

        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        self.scopes.append(scope)
        await send(
            {
                "type": "http.response.start",
                "status": 201,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body[::-1]})


class SilentApp:
    """Refuses the lifespan protocol and reads the request without ever responding."""

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            raise RuntimeError("no lifespan")
        await receive()


def test_lifespan_runs_once_and_requests_reach_the_app():
    app = RecordingApp()
    runner = ASGIRunner(app)
    try:
        first = runner.request(
            "post",
            "/v1/extract/pipeline/full",
            query_string="dry_run=1",
            headers={"Content-Type": "multipart/form-data; boundary=x", "X-Trace": "t-1"},
            body=b"abc",
        )
        second = runner.request("GET", "/healthz")
    finally:
        runner.close()

    assert app.startups == 1
    assert first.status == 201
    assert first.body == b"cba"
    assert ("content-type", "application/json") in first.headers
    assert second.body == b""
    scope = app.scopes[0]
    assert scope["method"] == "POST"
    assert scope["path"] == "/v1/extract/pipeline/full"
    assert scope["query_string"] == b"dry_run=1"
    seen = {name.decode(): value.decode() for name, value in scope["headers"]}
    assert seen["content-type"] == "multipart/form-data; boundary=x"
    assert seen["x-trace"] == "t-1"
    assert seen["content-length"] == "3"
    assert app.shutdowns == 1


def test_failed_startup_is_raised_and_retried_on_the_next_request():
    app = RecordingApp(startup_error="model files missing")
    runner = ASGIRunner(app)
    try:
        try:
            runner.request("GET", "/")
        except LifespanStartupError as error:
            assert "model files missing" in str(error)
        else:
            raise AssertionError("a failed startup must be raised to the caller")

        app.startup_error = None
        response = runner.request("POST", "/", body=b"xy")
    finally:
        runner.close()

    assert response.body == b"yx"
    assert app.startups == 1


def test_application_without_lifespan_support_still_serves_requests():
    app = RecordingApp(lifespan_supported=False)
    runner = ASGIRunner(app)
    try:
        response = runner.request("POST", "/", body=b"ab")
    finally:
        runner.close()

    assert response.body == b"ba"
    assert app.startups == 0


def test_application_that_never_responds_is_reported():
    runner = ASGIRunner(SilentApp())
    try:
        try:
            runner.request("GET", "/")
        except RuntimeError as error:
            assert "without starting a response" in str(error)
        else:
            raise AssertionError("expected a RuntimeError")
    finally:
        runner.close()


def test_requests_after_close_are_rejected():
    runner = ASGIRunner(RecordingApp())
    runner.close()
    try:
        runner.request("GET", "/")
    except RuntimeError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("expected a RuntimeError after close()")
