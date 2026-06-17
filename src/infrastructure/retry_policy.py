"""Retry policy with exponential backoff, jitter, and circuit breaker.

Handles transient failures for LLM API calls and external services.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403}


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    circuit_break_threshold: int = 5
    circuit_break_duration_seconds: int = 300


class CircuitBreaker:
    """Tracks consecutive failures per endpoint and opens/closes circuit."""

    def __init__(self, threshold: int = 5, duration_seconds: int = 300) -> None:
        self._threshold = threshold
        self._duration = duration_seconds
        self._failures: dict[str, int] = {}
        self._open_since: dict[str, float] = {}
        self._half_open: dict[str, bool] = {}

    def record_failure(self, endpoint: str) -> None:
        """Record a failure for an endpoint."""
        self._failures[endpoint] = self._failures.get(endpoint, 0) + 1
        if self._failures[endpoint] >= self._threshold:
            self._open_since[endpoint] = time.monotonic()
            self._half_open.pop(endpoint, None)
            logger.warning("Circuit opened for endpoint %s after %d failures",
                           endpoint, self._failures[endpoint])

    def record_success(self, endpoint: str) -> None:
        """Record a success, closing the circuit."""
        self._failures[endpoint] = 0
        self._open_since.pop(endpoint, None)
        self._half_open.pop(endpoint, None)

    def is_open(self, endpoint: str) -> bool:
        """Check if circuit is open (rejecting calls)."""
        if endpoint not in self._open_since:
            return False
        elapsed = time.monotonic() - self._open_since[endpoint]
        if elapsed >= self._duration:
            return False  # Duration expired, allow probe
        return True

    def allow_probe(self, endpoint: str) -> bool:
        """Check if a probe request is allowed (half-open state)."""
        if endpoint not in self._open_since:
            return True
        elapsed = time.monotonic() - self._open_since[endpoint]
        if elapsed >= self._duration and not self._half_open.get(endpoint, False):
            self._half_open[endpoint] = True
            return True
        return False


class RetryableError(Exception):
    """Raised for retryable HTTP errors."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class NonRetryableError(Exception):
    """Raised for non-retryable HTTP errors."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        super().__init__(f"Circuit breaker open for endpoint: {endpoint}")


class RetryPolicy:
    """Unified retry strategy with exponential backoff and circuit breaker."""

    RETRYABLE_STATUS_CODES = RETRYABLE_STATUS_CODES
    NON_RETRYABLE_STATUS_CODES = NON_RETRYABLE_STATUS_CODES

    def __init__(self, config: RetryConfig | None = None) -> None:
        self._config = config or RetryConfig()
        self._circuit_breaker = CircuitBreaker(
            threshold=self._config.circuit_break_threshold,
            duration_seconds=self._config.circuit_break_duration_seconds,
        )

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    @staticmethod
    def classify_status(status_code: int) -> str:
        """Classify an HTTP status code as retryable, non-retryable, or success."""
        if status_code < 400:
            return "success"
        if status_code in RETRYABLE_STATUS_CODES:
            return "retryable"
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return "non_retryable"
        return "unknown"

    def _compute_delay(self, attempt: int) -> float:
        """Compute delay with exponential backoff and optional jitter."""
        delay = self._config.base_delay_seconds * (self._config.backoff_multiplier ** attempt)
        delay = min(delay, self._config.max_delay_seconds)
        if self._config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay

    async def execute(
        self,
        func: Callable,
        *args: Any,
        endpoint: str = "default",
        **kwargs: Any,
    ) -> Any:
        """Execute func with retry logic. Raises after exhausting retries."""
        if self._circuit_breaker.is_open(endpoint):
            if not self._circuit_breaker.allow_probe(endpoint):
                raise CircuitOpenError(endpoint)

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                self._circuit_breaker.record_success(endpoint)
                return result
            except NonRetryableError:
                raise
            except RetryableError as e:
                last_error = e
                self._circuit_breaker.record_failure(endpoint)
                if attempt < self._config.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1, self._config.max_retries, endpoint, delay, e,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Exhausted %d retries for %s", self._config.max_retries, endpoint)
            except Exception as e:
                last_error = e
                self._circuit_breaker.record_failure(endpoint)
                if attempt < self._config.max_retries:
                    delay = self._compute_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    break

        raise last_error or RuntimeError(f"Retry exhausted for {endpoint}")
