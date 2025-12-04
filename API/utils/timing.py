"""Timing utilities used across the API surface."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict

from starlette.middleware.base import BaseHTTPMiddleware


class TimingMiddleware(BaseHTTPMiddleware):
    """Attach X-Server-Time header to responses."""

    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        total_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Server-Time"] = f"{total_ms:.1f}"
        return response


@dataclass
class TimingTracker:
    """Lightweight timing helper that logs durations when enabled."""

    logger: logging.Logger
    enabled: bool = False
    prefix: str = ""
    marks: Dict[str, float] = field(default_factory=dict)

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
