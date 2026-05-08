"""Per-component circuit breaker for retrieval fault isolation (spec §9).

State machine per component:
  CLOSED  → normal operation; failures increment counter.
  OPEN    → tripped after N consecutive failures; requests skipped for cooldown.
  (no HALF-OPEN state; the component is retried after cooldown naturally when
   the next request arrives and the cooldown has elapsed.)

Thread-safe via threading.Lock (pipeline runs executor tasks in threads).
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Tracks consecutive failures for a single named component.

    Usage:
        cb = CircuitBreaker("bm25", failures=5, cooldown=60)

        if cb.is_open():
            # skip the call — return empty result
            pass
        else:
            try:
                result = do_work()
                cb.record_success()
            except Exception:
                cb.record_failure()
                raise
    """

    def __init__(
        self,
        name: str,
        consecutive_failures: int = 5,
        cooldown_seconds: int = 60,
    ) -> None:
        self._name = name
        self._threshold = consecutive_failures
        self._cooldown = cooldown_seconds

        self._failures = 0
        self._tripped_at: float | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_open(self) -> bool:
        """Return True if the circuit is open (component should be skipped)."""
        with self._lock:
            if self._tripped_at is None:
                return False
            elapsed = time.monotonic() - self._tripped_at
            if elapsed >= self._cooldown:
                # Cooldown elapsed — allow one attempt (reset counter).
                logger.info(
                    "Circuit breaker '%s' cooldown elapsed; allowing retry.",
                    self._name,
                )
                self._failures = 0
                self._tripped_at = None
                return False
            return True

    def record_success(self) -> None:
        """Record a successful call; reset failure counter."""
        with self._lock:
            self._failures = 0
            self._tripped_at = None

    def record_failure(self) -> None:
        """Record a failed call; trip the breaker if threshold is reached."""
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold and self._tripped_at is None:
                self._tripped_at = time.monotonic()
                logger.warning(
                    "Circuit breaker '%s' tripped after %d consecutive failures "
                    "(cooldown=%ds).",
                    self._name,
                    self._failures,
                    self._cooldown,
                )

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failures

    @property
    def is_tripped(self) -> bool:
        with self._lock:
            return self._tripped_at is not None
