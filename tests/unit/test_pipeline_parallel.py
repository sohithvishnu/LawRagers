"""Unit tests for retrieval pipeline concurrency behavior and reranker circuit breaker."""

from __future__ import annotations

import asyncio
import time

import pytest

from retriever_service.config import CircuitBreakerConfig, RerankerConfig, RRFConfig, TimeoutsConfig
from retriever_service.retrieval.pipeline import RetrievalPipeline


class _SleepingRetriever:
    def __init__(self, label: str, delay: float) -> None:
        self.label = label
        self.delay = delay

    def query_case_law(self, query_text, k, filters):
        time.sleep(self.delay)
        return [{
            "chunk_id": f"{self.label}-1",
            "text": "Body",
            "metadata": {"case_id": 1, "case_name": self.label, "corpus": "ny_case_law"},
            "rank": 1,
        }]

    def query_user_workspace(self, session_id, query_text, k, filters):
        return []


class _NoopReranker:
    def rerank(self, query, candidates, top_n):
        return candidates[:top_n]


class _NoopRelational:
    pass


class TestRetrievalPipelineParallelism:
    async def test_bm25_and_dense_run_concurrently(self):
        pipeline = RetrievalPipeline(
            bm25=_SleepingRetriever("bm25", 0.2),
            dense=_SleepingRetriever("dense", 0.2),
            reranker=_NoopReranker(),
            relational=_NoopRelational(),
            rrf_cfg=RRFConfig(candidate_pool=10, k=60),
            reranker_cfg=RerankerConfig(enabled_default=False),
            timeouts_cfg=TimeoutsConfig(bm25=1000, dense=1000, rerank=1000),
            circuit_breaker_cfg=CircuitBreakerConfig(consecutive_failures=5, cooldown_seconds=60),
        )

        started = time.perf_counter()
        result = await pipeline.retrieve(
            query="test query",
            session_id=None,
            k=5,
            corpora=["ny_case_law"],
            filters={},
            rerank=False,
        )
        elapsed = time.perf_counter() - started

        assert result["results"]
        assert elapsed < 0.35

    async def test_zero_hits_do_not_raise_unavailable_error(self):
        class _EmptyRetriever:
            def query_case_law(self, query_text, k, filters):
                return []

            def query_user_workspace(self, session_id, query_text, k, filters):
                return []

        pipeline = RetrievalPipeline(
            bm25=_EmptyRetriever(),
            dense=_EmptyRetriever(),
            reranker=_NoopReranker(),
            relational=_NoopRelational(),
            rrf_cfg=RRFConfig(candidate_pool=10, k=60),
            reranker_cfg=RerankerConfig(enabled_default=False),
            timeouts_cfg=TimeoutsConfig(bm25=1000, dense=1000, rerank=1000),
            circuit_breaker_cfg=CircuitBreakerConfig(consecutive_failures=5, cooldown_seconds=60),
        )

        result = await pipeline.retrieve(
            query="test query",
            session_id=None,
            k=5,
            corpora=["ny_case_law"],
            filters={},
            rerank=False,
        )

        assert result["results"] == []
        assert result["degraded"] is False


# ---------------------------------------------------------------------------
# Reranker circuit breaker
# ---------------------------------------------------------------------------

def _make_pipeline(reranker, cb_failures=3):
    """Build a pipeline with a single stub retriever and configurable CB threshold."""

    class _StubRetriever:
        def query_case_law(self, query_text, k, filters):
            return [
                {
                    "chunk_id": f"c{i}",
                    "text": f"text {i}",
                    "metadata": {"case_id": i, "corpus": "ny_case_law"},
                    "rank": i + 1,
                }
                for i in range(5)
            ]

        def query_user_workspace(self, session_id, query_text, k, filters):
            return []

    class _NoopRelational:
        pass

    return RetrievalPipeline(
        bm25=_StubRetriever(),
        dense=_StubRetriever(),
        reranker=reranker,
        relational=_NoopRelational(),
        rrf_cfg=RRFConfig(candidate_pool=10, k=60),
        reranker_cfg=RerankerConfig(enabled_default=True, top_n_input=50, top_n_output=5),
        timeouts_cfg=TimeoutsConfig(bm25=2000, dense=2000, rerank=2000),
        circuit_breaker_cfg=CircuitBreakerConfig(
            consecutive_failures=cb_failures, cooldown_seconds=60
        ),
    )


class TestRerankerCircuitBreaker:
    async def test_reranker_success_recorded_and_results_returned(self):
        class _GoodReranker:
            def rerank(self, query, candidates, top_n):
                for i, c in enumerate(candidates[:top_n], start=1):
                    c = dict(c)
                    c["reranker_score"] = 1.0 / i
                    c["rerank_rank"] = i
                    c["score_components"] = {"reranker_logit": 1.0 / i, "pagerank_boost": 0.0}
                return [dict(c, reranker_score=1.0 / (j + 1), rerank_rank=j + 1,
                             score_components={"reranker_logit": 1.0 / (j + 1), "pagerank_boost": 0.0})
                        for j, c in enumerate(candidates[:top_n])]

        pipeline = _make_pipeline(_GoodReranker())
        result = await pipeline.retrieve(
            query="test", session_id=None, k=5,
            corpora=["ny_case_law"], filters={}, rerank=True,
        )

        assert "rerank" in result["retrievers_used"]
        assert result["degraded"] is False
        assert pipeline._cb_reranker.failure_count == 0

    async def test_reranker_exception_trips_circuit_after_threshold(self):
        class _BrokenReranker:
            def rerank(self, query, candidates, top_n):
                raise RuntimeError("model exploded")

        pipeline = _make_pipeline(_BrokenReranker(), cb_failures=3)

        for _ in range(3):
            result = await pipeline.retrieve(
                query="test", session_id=None, k=5,
                corpora=["ny_case_law"], filters={}, rerank=True,
            )
            assert result["degraded"] is True
            assert "rerank" not in result["retrievers_used"]

        assert pipeline._cb_reranker.is_tripped

    async def test_open_circuit_skips_reranker_without_calling_it(self):
        call_count = 0

        class _CountingReranker:
            def rerank(self, query, candidates, top_n):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("always fails")

        pipeline = _make_pipeline(_CountingReranker(), cb_failures=1)

        # First call trips the breaker.
        await pipeline.retrieve(
            query="test", session_id=None, k=5,
            corpora=["ny_case_law"], filters={}, rerank=True,
        )
        assert pipeline._cb_reranker.is_tripped
        assert call_count == 1

        # Second call: circuit open — reranker is never invoked.
        result = await pipeline.retrieve(
            query="test", session_id=None, k=5,
            corpora=["ny_case_law"], filters={}, rerank=True,
        )
        assert call_count == 1  # no additional call
        assert result["degraded"] is True
        assert "rerank" not in result["retrievers_used"]

    async def test_reranker_timeout_trips_circuit(self):
        class _SlowReranker:
            def rerank(self, query, candidates, top_n):
                time.sleep(5)
                return candidates[:top_n]

        pipeline = _make_pipeline(_SlowReranker(), cb_failures=2)
        # Use a very short timeout to force the timeout path.
        pipeline._timeouts.rerank = 50  # 50ms

        result = await pipeline.retrieve(
            query="test", session_id=None, k=5,
            corpora=["ny_case_law"], filters={}, rerank=True,
        )

        assert result["degraded"] is True
        assert pipeline._cb_reranker.failure_count >= 1

    async def test_reranker_success_resets_failure_count(self):
        fail_count = 0

        class _FlakyReranker:
            def rerank(self, query, candidates, top_n):
                nonlocal fail_count
                if fail_count < 2:
                    fail_count += 1
                    raise RuntimeError("transient failure")
                return candidates[:top_n]

        pipeline = _make_pipeline(_FlakyReranker(), cb_failures=5)

        # Two failures — below the threshold, so circuit stays closed.
        for _ in range(2):
            result = await pipeline.retrieve(
                query="test", session_id=None, k=5,
                corpora=["ny_case_law"], filters={}, rerank=True,
            )
            assert result["degraded"] is True

        # Third call succeeds — failure counter should reset.
        result = await pipeline.retrieve(
            query="test", session_id=None, k=5,
            corpora=["ny_case_law"], filters={}, rerank=True,
        )
        assert result["degraded"] is False
        assert pipeline._cb_reranker.failure_count == 0
