"""Unit tests for CircuitBreaker (spec §9)."""

import time
import pytest
from retriever_service.retrieval.circuit_breaker import CircuitBreaker


class TestInitialState:
    def test_initially_closed(self):
        cb = CircuitBreaker("test", consecutive_failures=3, cooldown_seconds=60)
        assert not cb.is_open()

    def test_zero_failures(self):
        cb = CircuitBreaker("test")
        assert cb.failure_count == 0


class TestFailureThreshold:
    def test_below_threshold_stays_closed(self):
        cb = CircuitBreaker("test", consecutive_failures=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open()

    def test_at_threshold_opens(self):
        cb = CircuitBreaker("test", consecutive_failures=3, cooldown_seconds=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open()

    def test_beyond_threshold_stays_open(self):
        cb = CircuitBreaker("test", consecutive_failures=3, cooldown_seconds=60)
        for _ in range(10):
            cb.record_failure()
        assert cb.is_open()


class TestSuccessReset:
    def test_success_resets_counter(self):
        cb = CircuitBreaker("test", consecutive_failures=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert not cb.is_open()

    def test_success_after_failures_closes(self):
        cb = CircuitBreaker("test", consecutive_failures=2, cooldown_seconds=60)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()  # only 1 consecutive failure now
        assert not cb.is_open()


class TestCooldown:
    def test_cooldown_elapsed_allows_retry(self):
        # Use a very short cooldown so we can observe the transition.
        cb = CircuitBreaker("test", consecutive_failures=1, cooldown_seconds=100)
        cb.record_failure()
        assert cb.is_open()  # tripped, cooldown not elapsed
        # Manually backdate the trip time so it looks like cooldown has elapsed.
        cb._tripped_at = time.monotonic() - 200
        assert not cb.is_open()  # cooldown elapsed → resets to closed

    def test_within_cooldown_stays_open(self):
        cb = CircuitBreaker("test", consecutive_failures=1, cooldown_seconds=3600)
        cb.record_failure()
        assert cb.is_open()


class TestIsTripped:
    def test_tripped_after_threshold(self):
        cb = CircuitBreaker("test", consecutive_failures=2, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_tripped

    def test_not_tripped_initially(self):
        cb = CircuitBreaker("test", consecutive_failures=2, cooldown_seconds=60)
        assert not cb.is_tripped
