"""E2E tests: three frontend user journeys against the FastAPI application.

These tests drive the HTTP API exactly as the React Native frontend does.
The retriever service (port 8001) is called for real — no stubs — because it
is a critical dependency whose failure modes must be exercised.  Only Ollama
is stubbed (user does not have it running locally).

All tests are skipped automatically when the retriever service is unreachable.

Run with:
    .venv/bin/pytest tests/e2e/test_user_journeys.py -v

Journeys covered
----------------
1. Create New Matter     – Lobby → Configure Workspace → session created
2. Analyze Legal Question – POST /search (real retriever) → POST /generate
3. Resume Session with Graph Time-Travel – reload history + graph checkpoints
"""
from __future__ import annotations

import json
import sqlite3

import httpx
import ollama
import pytest
from httpx import ASGITransport, AsyncClient

import api as api_module

# ---------------------------------------------------------------------------
# Module-level retriever availability guard
# ---------------------------------------------------------------------------

def _retriever_available() -> bool:
    try:
        r = httpx.get("http://localhost:8001/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _retriever_available(),
    reason="Retriever service not reachable at localhost:8001 — start it with "
           "'uvicorn retriever_service.main:app --port 8001'",
)

# ---------------------------------------------------------------------------
# Shared fake data
# ---------------------------------------------------------------------------

_FAKE_AI_CHUNKS = [
    "**Issue**\n\n",
    "Does the Storm in Progress doctrine bar recovery?\n\n",
    "**Rule**\n\n",
    "A property owner is not liable for injuries sustained during an ongoing storm.\n\n",
    "**Application**\n\n",
    "Here, the plaintiff fell at 2:00 PM while snow was still actively falling.\n\n",
    "**Conclusion**\n\n",
    "The Storm in Progress doctrine likely bars recovery.",
]

_FAKE_MEMO = "".join(_FAKE_AI_CHUNKS)

# Graph state snapshots the frontend computes and persists per turn
_GRAPH_STATE_TURN_1 = json.dumps([
    {
        "id": "Solazzo v. New York City Transit Auth.",
        "date": "2006-03-28",
        "distance": 0.92,
        "hitCount": 1,
    }
])

_GRAPH_STATE_TURN_2 = json.dumps([
    {
        "id": "Solazzo v. New York City Transit Auth.",
        "date": "2006-03-28",
        "distance": 0.85,
        "hitCount": 2,
    },
    {
        "id": "Pippo v. City of New York",
        "date": "2005-11-15",
        "distance": 0.78,
        "hitCount": 1,
    },
])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    """Replace the module-level SQLite connection with an isolated in-memory DB.

    Session management lives entirely in api.py; using an in-memory DB keeps
    tests hermetic without touching the real legal_sessions.db on disk.
    The retriever service manages its own persistence independently.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            name        TEXT,
            description TEXT,
            databases   TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT PRIMARY KEY,
            session_id  TEXT,
            role        TEXT,
            content     TEXT,
            graph_state TEXT DEFAULT '[]',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    conn.commit()
    monkeypatch.setattr(api_module, "db_conn", conn)
    monkeypatch.setattr(api_module, "cursor", cur)
    yield
    conn.close()


@pytest.fixture(autouse=True)
def stub_ollama(monkeypatch):
    """Stub Ollama only — the retriever service is NOT stubbed."""
    def _fake_chat(model, messages, stream):
        for text in _FAKE_AI_CHUNKS:
            yield {"message": {"content": text}}

    monkeypatch.setattr(ollama, "chat", _fake_chat)


@pytest.fixture
async def http():
    """ASGI test client that triggers the FastAPI lifespan.

    retrieve_for_search() inside api.py makes a real synchronous httpx call to
    localhost:8001, so requests flow through:
        test → api.py (ASGI) → retriever service (real HTTP, port 8001)
    """
    async with AsyncClient(
        transport=ASGITransport(app=api_module.app), base_url="http://test"
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Journey 1: Create New Matter
# ---------------------------------------------------------------------------

class TestCreateMatterJourney:
    """Lobby → Configure Workspace → Workspace initialized.

    Frontend call sequence (index.tsx):
        fetchSessions()          → GET  /sessions
        createWorkspaceSession() → POST /sessions
        GET /sessions            → lobby re-renders with new matter
        GET /sessions/{id}/messages → workspace opens (empty history)
    """

    async def test_lobby_starts_with_empty_session_list(self, http):
        r = await http.get("/sessions")

        assert r.status_code == 200
        assert r.json() == {"sessions": []}

    async def test_create_matter_returns_uuid_and_name(self, http):
        r = await http.post("/sessions", json={
            "name": "Smith v. Jones",
            "description": "Premises liability slip-and-fall",
            "databases": "NY_Case_Law,User_Workspace",
        })

        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Smith v. Jones"
        assert "id" in body
        assert len(body["id"]) == 36  # UUID — 8-4-4-4-12 + dashes

    async def test_new_matter_appears_in_session_list_with_all_fields(self, http):
        create_r = await http.post("/sessions", json={
            "name": "Smith v. Jones",
            "description": "Premises liability slip-and-fall",
            "databases": "NY_Case_Law,User_Workspace",
        })
        session_id = create_r.json()["id"]

        list_r = await http.get("/sessions")

        assert list_r.status_code == 200
        sessions = list_r.json()["sessions"]
        assert len(sessions) == 1
        s = sessions[0]
        assert s["id"] == session_id
        assert s["name"] == "Smith v. Jones"
        assert s["description"] == "Premises liability slip-and-fall"
        assert s["databases"] == "NY_Case_Law,User_Workspace"
        assert "created_at" in s

    async def test_new_matter_has_empty_message_history(self, http):
        create_r = await http.post("/sessions", json={
            "name": "Smith v. Jones",
            "description": "",
            "databases": "NY_Case_Law,User_Workspace",
        })
        session_id = create_r.json()["id"]

        msg_r = await http.get(f"/sessions/{session_id}/messages")

        assert msg_r.status_code == 200
        assert msg_r.json() == {"messages": []}

    async def test_messages_endpoint_rejects_unknown_session(self, http):
        r = await http.get("/sessions/00000000-0000-0000-0000-000000000000/messages")

        assert r.status_code == 404
        detail = r.json()["detail"]
        assert detail["error"] == "session_not_found"
        assert "00000000-0000-0000-0000-000000000000" in detail["session_id"]


# ---------------------------------------------------------------------------
# Journey 2: Analyze a Legal Question (real retriever)
# ---------------------------------------------------------------------------

class TestAnalyzeLegalQuestionJourney:
    """User sends a query → real retriever search → generate memo → persist.

    The retriever service is called for real.  Assertions check that the
    pipeline delivers correctly shaped data from the retriever through api.py
    back to the client — accuracy of retrieved documents is not asserted.

    Frontend call sequence (handleAnalyze in index.tsx):
        POST /search            → build cumulative graph from real results
        POST /generate          → stream IRAC memo (Ollama stubbed)
        POST /messages (user)   → persist user turn
        POST /messages (asst)   → persist AI turn with graph_state checkpoint
        GET  /sessions/{id}/messages → reload / verify full history
    """

    QUERY = "Storm in Progress doctrine slip and fall premises liability New York"

    @pytest.fixture
    async def session(self, http):
        r = await http.post("/sessions", json={
            "name": "Storm Doctrine Matter",
            "description": "Slip-and-fall during snowstorm",
            "databases": "NY_Case_Law,User_Workspace",
        })
        return r.json()["id"]

    async def test_search_returns_200_with_cases_and_context(self, http, session):
        r = await http.post("/search", json={
            "session_id": session,
            "argument": self.QUERY,
        })

        assert r.status_code == 200
        body = r.json()
        assert "cases" in body
        assert "context_text" in body
        assert isinstance(body["cases"], list)
        assert isinstance(body["context_text"], str)

    async def test_search_returns_at_least_one_case_from_retriever(self, http, session):
        r = await http.post("/search", json={
            "session_id": session,
            "argument": self.QUERY,
        })

        cases = r.json()["cases"]
        assert len(cases) >= 1, (
            f"Expected the retriever to return at least one NY case law result "
            f"for '{self.QUERY}', got 0.  Check that the ny_case_law index is populated."
        )

    async def test_search_cases_have_required_fields(self, http, session):
        r = await http.post("/search", json={
            "session_id": session,
            "argument": self.QUERY,
        })

        for case in r.json()["cases"]:
            assert isinstance(case["id"], str) and case["id"], \
                f"case 'id' must be a non-empty string, got: {case.get('id')!r}"
            assert isinstance(case["date"], str), \
                f"case 'date' must be a string, got: {case.get('date')!r}"
            assert isinstance(case["text"], str) and case["text"], \
                f"case 'text' must be non-empty, got: {case.get('text')!r}"
            assert isinstance(case["distance"], (int, float)), \
                f"case 'distance' must be numeric, got: {case.get('distance')!r}"

    async def test_search_context_text_contains_precedent_block(self, http, session):
        r = await http.post("/search", json={
            "session_id": session,
            "argument": self.QUERY,
        })

        context = r.json()["context_text"]
        # build_search_response always opens with this header for ny_case_law hits
        assert "BINDING PRECEDENT" in context
        assert len(context) > len("--- BINDING PRECEDENT (NY CASE LAW) ---")

    async def test_generate_streams_irac_memo_using_retrieved_context(self, http, session):
        # Use real search output as context — same pipeline the frontend follows
        search_r = await http.post("/search", json={
            "session_id": session,
            "argument": self.QUERY,
        })
        context_text = search_r.json()["context_text"]

        gen_r = await http.post("/generate", json={
            "session_id": session,
            "argument": self.QUERY,
            "context_text": context_text,
        })

        assert gen_r.status_code == 200
        full_text = gen_r.text
        for section in ("Issue", "Rule", "Application", "Conclusion"):
            assert section in full_text, f"IRAC section '{section}' missing from generated memo"

    async def test_messages_persisted_and_retrieved_with_graph_state(self, http, session):
        # Step 1: search for real context
        search_r = await http.post("/search", json={
            "session_id": session,
            "argument": self.QUERY,
        })
        cases = search_r.json()["cases"]
        context_text = search_r.json()["context_text"]

        # Step 2: the frontend builds a graph_state from the returned cases
        graph_state = json.dumps([
            {"id": c["id"], "date": c["date"], "distance": c["distance"], "hitCount": 1}
            for c in cases
        ])

        # Step 3: save user message
        r_user = await http.post("/messages", json={
            "session_id": session,
            "role": "user",
            "content": self.QUERY,
            "graph_state": "[]",
        })
        assert r_user.status_code == 200
        assert r_user.json()["status"] == "success"
        user_msg_id = r_user.json()["id"]

        # Step 4: generate memo, then save assistant message with checkpoint
        gen_r = await http.post("/generate", json={
            "session_id": session,
            "argument": self.QUERY,
            "context_text": context_text,
        })
        memo_text = gen_r.text

        r_ai = await http.post("/messages", json={
            "session_id": session,
            "role": "assistant",
            "content": memo_text,
            "graph_state": graph_state,
        })
        assert r_ai.status_code == 200
        ai_msg_id = r_ai.json()["id"]

        # Step 5: reload — both messages must be present, ordered, with correct values
        msgs_r = await http.get(f"/sessions/{session}/messages")
        assert msgs_r.status_code == 200
        messages = msgs_r.json()["messages"]

        assert len(messages) == 2

        user_msg = messages[0]
        assert user_msg["id"] == user_msg_id
        assert user_msg["role"] == "user"
        assert user_msg["text"] == self.QUERY

        ai_msg = messages[1]
        assert ai_msg["id"] == ai_msg_id
        assert ai_msg["role"] == "assistant"
        # Memo content flows through correctly
        assert "Issue" in ai_msg["text"]
        assert "Conclusion" in ai_msg["text"]

        # Graph state was round-tripped correctly and deserializes
        stored_graph = json.loads(ai_msg["graph_state"])
        assert isinstance(stored_graph, list)
        assert len(stored_graph) == len(cases)
        stored_ids = {c["id"] for c in stored_graph}
        expected_ids = {c["id"] for c in cases}
        assert stored_ids == expected_ids


# ---------------------------------------------------------------------------
# Journey 3: Resume Session with Graph Time-Travel
# ---------------------------------------------------------------------------

class TestSessionResumeWithTimeTravelJourney:
    """User returns to an existing session and rewinds the graph to a prior turn.

    No retriever calls in this journey — it exercises session persistence and
    the graph checkpoint (time-travel) mechanism only.

    Frontend call sequence (loadExistingSession + "View Graph Checkpoint" tap):
        GET  /sessions                  → pick "Slip-and-Fall Matter"
        GET  /sessions/{id}/messages    → restore chat + last graph state
        parse messages[1].graph_state   → rewind graph to turn-1 (1 node)
        parse messages[3].graph_state   → verify cumulative turn-2 graph (2 nodes)
    """

    @pytest.fixture
    async def session_with_history(self, http):
        """Create a session pre-seeded with two full exchange turns."""
        r = await http.post("/sessions", json={
            "name": "Slip-and-Fall Matter",
            "description": "Multi-turn research with graph evolution",
            "databases": "NY_Case_Law,User_Workspace",
        })
        session_id = r.json()["id"]

        # Turn 1 — initial query
        await http.post("/messages", json={
            "session_id": session_id, "role": "user",
            "content": "Does the Storm in Progress doctrine apply?",
            "graph_state": "[]",
        })
        await http.post("/messages", json={
            "session_id": session_id, "role": "assistant",
            "content": "**Issue**\nStorm doctrine applicability.\n\n**Conclusion**\nLikely applies.",
            "graph_state": _GRAPH_STATE_TURN_1,
        })

        # Turn 2 — follow-up that grows the graph with a second case
        await http.post("/messages", json={
            "session_id": session_id, "role": "user",
            "content": "How does it change if the plaintiff testifies it had stopped snowing an hour prior?",
            "graph_state": "[]",
        })
        await http.post("/messages", json={
            "session_id": session_id, "role": "assistant",
            "content": "**Issue**\nPost-storm liability.\n\n**Rule**\nLiability attaches after a reasonable time.\n\n**Conclusion**\nLiability may now attach.",
            "graph_state": _GRAPH_STATE_TURN_2,
        })

        return session_id

    async def test_session_appears_in_lobby_list(self, http, session_with_history):
        r = await http.get("/sessions")

        assert r.status_code == 200
        sessions = r.json()["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["name"] == "Slip-and-Fall Matter"
        assert sessions[0]["id"] == session_with_history

    async def test_full_message_history_restored_in_chronological_order(self, http, session_with_history):
        r = await http.get(f"/sessions/{session_with_history}/messages")

        assert r.status_code == 200
        messages = r.json()["messages"]
        assert len(messages) == 4

        assert messages[0]["role"] == "user"
        assert "Storm in Progress" in messages[0]["text"]

        assert messages[1]["role"] == "assistant"
        assert "Conclusion" in messages[1]["text"]

        assert messages[2]["role"] == "user"
        assert "stopped snowing" in messages[2]["text"]

        assert messages[3]["role"] == "assistant"
        assert "Post-storm" in messages[3]["text"]

    async def test_turn_1_graph_checkpoint_contains_single_case(self, http, session_with_history):
        """Rewinding to the turn-1 checkpoint gives a one-node graph (Solazzo)."""
        r = await http.get(f"/sessions/{session_with_history}/messages")
        messages = r.json()["messages"]

        turn_1_assistant = messages[1]
        assert turn_1_assistant["role"] == "assistant"

        graph = json.loads(turn_1_assistant["graph_state"])
        assert len(graph) == 1
        assert graph[0]["id"] == "Solazzo v. New York City Transit Auth."
        assert graph[0]["date"] == "2006-03-28"
        assert graph[0]["hitCount"] == 1

    async def test_turn_2_graph_is_cumulative_with_updated_hit_count(self, http, session_with_history):
        """The latest checkpoint has both cases; Solazzo's hitCount is 2."""
        r = await http.get(f"/sessions/{session_with_history}/messages")
        messages = r.json()["messages"]

        graph = json.loads(messages[3]["graph_state"])
        assert len(graph) == 2

        solazzo = next(c for c in graph if "Solazzo" in c["id"])
        pippo = next(c for c in graph if "Pippo" in c["id"])

        assert solazzo["hitCount"] == 2
        assert pippo["hitCount"] == 1

    async def test_graph_state_grows_strictly_between_turns(self, http, session_with_history):
        """Turn-1 node set is a strict subset of turn-2 — cumulative network growth."""
        r = await http.get(f"/sessions/{session_with_history}/messages")
        messages = r.json()["messages"]

        turn_1_ids = {c["id"] for c in json.loads(messages[1]["graph_state"])}
        turn_2_ids = {c["id"] for c in json.loads(messages[3]["graph_state"])}

        assert turn_1_ids < turn_2_ids, (
            f"Expected turn-1 graph {turn_1_ids} to be a strict subset of "
            f"turn-2 graph {turn_2_ids}"
        )
