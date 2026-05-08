"""Helpers for delegating legacy api.py retrieval and ingest to the retriever service."""

from __future__ import annotations

import os
from typing import Any

import httpx


DEFAULT_RETRIEVER_URL = "http://localhost:8001"


def get_retriever_url() -> str:
    return os.environ.get("RETRIEVER_URL", DEFAULT_RETRIEVER_URL).rstrip("/")


def retrieve_for_search(argument: str, session_id: str, k: int = 10) -> dict[str, Any]:
    response = httpx.post(
        f"{get_retriever_url()}/retrieve",
        json={
            "query": argument,
            "k": k,
            "session_id": session_id,
            "corpora": ["ny_case_law", "user_workspace"],
            "rerank": True,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


async def ingest_user_document(
    *,
    session_id: str,
    source: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{get_retriever_url()}/ingest",
            json={
                "corpus": "user_workspace",
                "session_id": session_id,
                "documents": [
                    {
                        "source": source,
                        "text": text,
                        "metadata": metadata or {},
                    }
                ],
            },
        )
    response.raise_for_status()
    return response.json()


def build_search_response(payload: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    case_context_lines = ["--- BINDING PRECEDENT (NY CASE LAW) ---"]
    workspace_context_lines = ["", "--- USER UPLOADED DOCUMENTS ---"]

    for result in payload.get("results", []):
        source = result.get("source") or {}
        corpus = source.get("corpus")
        text = result.get("text", "")

        if corpus == "ny_case_law":
            case_name = source.get("case_name")
            cases.append(
                {
                    "id": case_name or result.get("chunk_id"),
                    "case_id": result.get("case_id"),
                    "date": source.get("decision_date", ""),
                    "text": text,
                    "distance": result.get("score"),
                }
            )
            case_context_lines.append(f"\n[Case: {case_name}]\n{text}\n")
        elif corpus == "user_workspace":
            workspace_context_lines.append(f"\n[Source: user_workspace]\n{text}\n")

    return {
        "cases": cases,
        "context_text": "\n".join(case_context_lines + workspace_context_lines),
    }
