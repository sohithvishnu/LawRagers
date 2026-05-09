"""Cross-encoder reranker (spec §7.2).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (MIT, 22M params).
Loaded once at service startup and shared across all requests.
MPS device used on Apple Silicon with CPU fallback.

Pagerank-aware final scoring (spec §7.2.1):
    final_score = reranker_logit + alpha * pagerank_percentile

alpha defaults to 0.0 (disabled until eval validates it).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from retriever_service.config import RerankerConfig, settings as default_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker loaded at service startup."""

    def __init__(self, cfg: Optional[RerankerConfig] = None) -> None:
        if cfg is None:
            cfg = default_settings.reranker
        self._cfg = cfg
        self._model = None
        if cfg.enabled_default:
            self._load()

    def _load(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading reranker %s on device=%s", self._cfg.model, device)
        self._model = CrossEncoder(
            self._cfg.model,
            max_length=self._cfg.max_length,
            device=device,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: Optional[int] = None,
        pagerank_boost_weight: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Rerank a list of candidate chunks for a query.

        Args:
            query:      The retrieval query.
            candidates: List of chunk dicts with at least {'chunk_id', 'text', 'metadata'}.
            top_n:      Return at most top_n results (defaults to cfg.top_n_output).
            pagerank_boost_weight: Override for alpha (spec §7.2.1); None uses cfg default.

        Returns:
            Candidates reordered by final score, each annotated with:
              - reranker_score (logit)
              - score_components: {reranker_logit, pagerank_boost}
              - rerank_rank (1-indexed)
        """
        if not candidates or self._model is None:
            return candidates

        if top_n is None:
            top_n = self._cfg.top_n_output
        alpha = pagerank_boost_weight if pagerank_boost_weight is not None else self._cfg.pagerank_boost_weight

        pairs = [(query, c.get("text", "")) for c in candidates]
        logits = self._model.predict(
            pairs,
            batch_size=self._cfg.batch_size,
            show_progress_bar=False,
        )

        scored: list[tuple[float, float, int]] = []
        for i, (logit, candidate) in enumerate(zip(logits, candidates)):
            pagerank = float(candidate.get("metadata", {}).get("pagerank_percentile") or 0.0)
            boost = alpha * pagerank
            final = float(logit) + boost
            scored.append((final, float(logit), i))

        scored.sort(key=lambda t: t[0], reverse=True)
        if top_n:
            scored = scored[:top_n]

        results = []
        for rerank_rank, (final_score, logit, original_idx) in enumerate(scored, start=1):
            candidate = dict(candidates[original_idx])
            pagerank = float(candidate.get("metadata", {}).get("pagerank_percentile") or 0.0)
            candidate["reranker_score"] = final_score
            candidate["score_components"] = {
                "reranker_logit": logit,
                "pagerank_boost": alpha * pagerank,
            }
            candidate["rerank_rank"] = rerank_rank
            results.append(candidate)

        return results

    def health(self) -> str:
        return "loaded" if self._model is not None else "not_loaded"
