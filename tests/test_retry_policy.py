"""Tests for RetryPolicy and CircuitBreaker."""

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.infrastructure.retry_policy import (
    CircuitBreaker,
    CircuitOpenError,
    NonRetryableError,
    RetryableError,
    RetryConfig,
    RetryPolicy,
)


# Property 20: Retry Policy Classification
@given(status_code=st.sampled_from([429, 500, 502, 503, 504]))
@settings(max_examples=20)
def test_retryable_status_codes(status_code):
    """Retryable status codes should be classified as retryable."""
    assert RetryPolicy.classify_status(status_code) == "retryable"


@given(status_code=st.sampled_from([400, 401, 403]))
@settings(max_examples=10)
def test_non_retryable_status_codes(status_code):
    """Non-retryable status codes should not be retried."""
    assert RetryPolicy.classify_status(status_code) == "non_retryable"


def test_success_status():
    assert RetryPolicy.classify_status(200) == "success"
    assert RetryPolicy.classify_status(201) == "success"


# Property 21: Circuit Breaker State Machine
def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(threshold=3, duration_seconds=10)
    for _ in range(3):
        cb.record_failure("ep1")
    assert cb.is_open("ep1")


def test_circuit_breaker_closed_initially():
    cb = CircuitBreaker(threshold=3, duration_seconds=10)
    assert not cb.is_open("ep1")


def test_circuit_breaker_success_resets():
    cb = CircuitBreaker(threshold=3, duration_seconds=10)
    cb.record_failure("ep1")
    cb.record_failure("ep1")
    cb.record_success("ep1")
    cb.record_failure("ep1")
    assert not cb.is_open("ep1")  # reset by success


def test_circuit_breaker_probe_after_duration():
    cb = CircuitBreaker(threshold=2, duration_seconds=0)  # 0s duration
    cb.record_failure("ep1")
    cb.record_failure("ep1")
    # Duration expired immediately, should allow probe
    assert cb.allow_probe("ep1")


# Property 22: Retry Termination (Liveness)
def test_retry_terminates_on_success():
    """Retry should terminate when function succeeds."""
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RetryableError(500, "server error")
        return "ok"

    config = RetryConfig(max_retries=3, base_delay_seconds=0.01)
    policy = RetryPolicy(config)
    result = asyncio.run(
        policy.execute(flaky, endpoint="test")
    )
    assert result == "ok"
    assert call_count == 3


def test_retry_terminates_on_exhaustion():
    """Retry should terminate after exhausting retries."""
    async def always_fail():
        raise RetryableError(500, "always fails")

    config = RetryConfig(max_retries=2, base_delay_seconds=0.01)
    policy = RetryPolicy(config)
    with pytest.raises(RetryableError):
        asyncio.run(
            policy.execute(always_fail, endpoint="test")
        )


def test_non_retryable_not_retried():
    """Non-retryable errors should propagate immediately."""
    call_count = 0

    async def bad_request():
        nonlocal call_count
        call_count += 1
        raise NonRetryableError(400, "bad request")

    config = RetryConfig(max_retries=3, base_delay_seconds=0.01)
    policy = RetryPolicy(config)
    with pytest.raises(NonRetryableError):
        asyncio.run(
            policy.execute(bad_request, endpoint="test")
        )
    assert call_count == 1  # No retries


def test_circuit_open_rejects():
    """Open circuit should reject calls."""
    config = RetryConfig(
        max_retries=0, base_delay_seconds=0.01,
        circuit_break_threshold=1, circuit_break_duration_seconds=300,
    )
    policy = RetryPolicy(config)
    # Force circuit open
    policy.circuit_breaker.record_failure("ep")
    with pytest.raises(CircuitOpenError):
        asyncio.run(
            policy.execute(lambda: None, endpoint="ep")
        )


def test_retry_with_sync_function():
    """RetryPolicy should also work with sync callables."""
    call_count = 0

    def sync_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RetryableError(500, "fail")
        return "done"

    config = RetryConfig(max_retries=3, base_delay_seconds=0.01)
    policy = RetryPolicy(config)
    result = asyncio.run(
        policy.execute(sync_func, endpoint="test")
    )
    assert result == "done"
    assert call_count == 2
