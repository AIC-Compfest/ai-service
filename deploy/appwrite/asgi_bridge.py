"""Run HTTP requests through an ASGI application inside the current process.

The Appwrite entrypoint uses this to drive the same FastAPI app that uvicorn
serves in Docker. The module only speaks the ASGI protocol. It does not import
Appwrite or FastAPI, so it works with any ASGI application.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]
Scope = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_SHUTDOWN_TIMEOUT_SECONDS = 30


class LifespanStartupError(RuntimeError):
    """The application reported that its lifespan startup failed."""


@dataclass
class BridgeResponse:
    """What the application sent back: status, headers (lowercase names) and body."""

    status: int
    headers: list[tuple[str, str]] = field(default_factory=list)
    body: bytes = b""


class ASGIRunner:
    """Feeds requests to one ASGI application on a private event loop.

    The loop runs on its own thread. The caller may already be inside a running
    event loop (Appwrite's runtime calls the function from one), and asyncio
    cannot run a second loop on the same thread, so the caller only submits
    coroutines and waits for the result.

    Lifespan startup runs once, on the first request, so expensive setup such as
    model loading is paid once per process. An application without lifespan
    support is served without it, as uvicorn does by default.
    """

    def __init__(self, app: ASGIApp, *, client: tuple[str, int] = ("127.0.0.1", 0)) -> None:
        self._app = app
        self._client = client
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="asgi-runner", daemon=True
        )
        self._thread.start()
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._lifespan_queue: asyncio.Queue[Message] | None = None
        self._lifespan_task: asyncio.Task[None] | None = None

    def request(
        self,
        method: str,
        path: str,
        *,
        query_string: str = "",
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
    ) -> BridgeResponse:
        """Run one request to completion. Calls are serialised with a lock."""
        with self._lock:
            if self._closed:
                raise RuntimeError("ASGIRunner has been closed")
            self._call(self._ensure_started())
            return self._call(
                self._dispatch(method, path, query_string, dict(headers or {}), body)
            )

    def close(self) -> None:
        """Run the application's shutdown lifespan and release the event loop."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._call(self._shutdown())
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
                if not self._thread.is_alive():
                    self._loop.close()

    def _call(self, coro: Awaitable[Any]) -> Any:
        """Run a coroutine on the runner's loop thread and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()  # type: ignore[arg-type]

    async def _ensure_started(self) -> None:
        if self._started:
            return
        queue: asyncio.Queue[Message] = asyncio.Queue()
        queue.put_nowait({"type": "lifespan.startup"})
        startup: asyncio.Future[None] = self._loop.create_future()

        async def receive() -> Message:
            return await queue.get()

        async def send(message: Message) -> None:
            if startup.done():
                return
            if message["type"] == "lifespan.startup.complete":
                startup.set_result(None)
            elif message["type"] == "lifespan.startup.failed":
                startup.set_exception(LifespanStartupError(str(message.get("message", ""))))

        scope: Scope = {
            "type": "lifespan",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "state": {},
        }
        task = self._loop.create_task(self._app(scope, receive, send))
        await asyncio.wait({startup, task}, return_when=asyncio.FIRST_COMPLETED)

        if startup.done():
            error = startup.exception()
            if error is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise error
            self._lifespan_queue, self._lifespan_task = queue, task
        elif task.done() and not task.cancelled():
            # The application ended without reporting startup, so it has no lifespan
            # support. Uvicorn's default ("auto") mode carries on in that case too.
            task.exception()
        self._started = True

    async def _dispatch(
        self,
        method: str,
        path: str,
        query_string: str,
        headers: dict[str, str],
        body: bytes,
    ) -> BridgeResponse:
        raw_headers = [
            (name.lower().encode("latin-1", "replace"), value.encode("latin-1", "replace"))
            for name, value in headers.items()
            if name.lower() != "content-length"
        ]
        raw_headers.append((b"content-length", str(len(body)).encode("ascii")))
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "https",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "root_path": "",
            "query_string": query_string.encode("latin-1", "replace"),
            "headers": raw_headers,
            "client": self._client,
            "server": ("localhost", 443),
            "state": {},
        }

        request_sent = False
        finished = asyncio.Event()
        status = 500
        response_headers: list[tuple[str, str]] = []
        chunks: list[bytes] = []
        started = False

        async def receive() -> Message:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await finished.wait()
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            nonlocal status, response_headers, started
            if message["type"] == "http.response.start":
                started = True
                status = int(message["status"])
                response_headers = [
                    (str(name, "latin-1").lower(), str(value, "latin-1"))
                    for name, value in message.get("headers", [])
                ]
            elif message["type"] == "http.response.body":
                chunks.append(bytes(message.get("body", b"")))
                if not message.get("more_body", False):
                    finished.set()

        try:
            await self._app(scope, receive, send)
        finally:
            finished.set()
        if not started:
            raise RuntimeError("the ASGI application finished without starting a response")
        return BridgeResponse(status=status, headers=response_headers, body=b"".join(chunks))

    async def _shutdown(self) -> None:
        if self._lifespan_queue is None or self._lifespan_task is None:
            return
        await self._lifespan_queue.put({"type": "lifespan.shutdown"})
        await asyncio.wait({self._lifespan_task}, timeout=_SHUTDOWN_TIMEOUT_SECONDS)
