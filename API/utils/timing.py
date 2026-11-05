"""Timing utilities used across the API surface."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from starlette.datastructures import MutableHeaders


class TimingMiddleware:
    """Attach X-Server-Time header to responses.

    A plain ASGI middleware: Starlette's BaseHTTPMiddleware runs each request in
    an extra task group and streams the body through it, which cost more than
    the timing it adds.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()

        async def send_with_timing(message):
            if message["type"] == "http.response.start":
                total_ms = (time.perf_counter() - start) * 1000
                MutableHeaders(scope=message)["X-Server-Time"] = f"{total_ms:.1f}"
            await send(message)

        await self.app(scope, receive, send_with_timing)


@dataclass
class TimingTracker:
    """Lightweight timing helper that logs durations when enabled."""

    logger: logging.Logger
    enabled: bool = False
    prefix: str = ""
    marks: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def track(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._log_duration(label, start)

    def _log_duration(self, label: str, start_time: float) -> None:
        if not self.enabled:
            return
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.marks[label] = duration_ms
        self.logger.debug("%s%s: %.1f ms", self.prefix, label, duration_ms)


class StepTimer:
    """Helper to log timing steps relative to a start time."""

    def __init__(self, start_time, enabled: bool = False):
        self.start_time = start_time
        self.enabled = enabled

    def log(self, message: str):
        if self.enabled:
            import datetime

            print(message)
            print(
                datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
                - self.start_time
            )
