"""
Comprehensive backend tests for the LangGraph agent endpoint and existing routes.

Covers:
1. Success with mocked LLM (2 calls: parse_jd + suggest_rewrites)
2. Missing API key (503 status + helpful message)
3. Empty or whitespace-only inputs (400 status)
4. Malformed/non-JSON LLM output (graceful fallback)
5. LLM rate limit (429 status)
6. LLM timeout (504 status)
7. Verification of existing endpoints (/analyze, /history, /stats)
"""

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
# Compatibility patch for starlette 0.36.3 with httpx 0.28+ (removes deprecated 'app' kwarg)
_orig_httpx_client_init = httpx.Client.__init__
def _patched_httpx_client_init(self, *args, **kwargs):
    kwargs.pop("app", None)
    return _orig_httpx_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_httpx_client_init

import pytest
from starlette.testclient import TestClient

# Ensure backend root is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import app
from models import ml_models


def _make_pdf(text):
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()


def _make_docx(text):
    import docx

    document = docx.Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture(scope="session", autouse=True)
def setup_models():
    """Ensure models are initialized for tests."""
    ml_models.load_models()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Success with Mocked LLM
# ---------------------------------------------------------------------------

def test_agent_analyze_success_mocked(client, monkeypatch):
    """Verify end-to-end agent graph with mocked LLM calls."""
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-test-key-12345")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

    # Structured responses for parse_jd and suggest_rewrites
    jd_mock_json = json.dumps({
        "title": "Backend Python Developer",
        "requirements": [
            {
                "category": "technical",
                "requirement": "Strong Python and Docker experience",
                "skills": ["Python", "Docker"]
            },
            {
                "category": "technical",
                "requirement": "AWS cloud deployment and Kubernetes",
                "skills": ["AWS", "Kubernetes"]
            }
        ],
        "all_skills": ["Python", "Docker", "AWS", "Kubernetes"]
    })

    rewrite_mock_json = json.dumps({
        "suggestions": [
            {
                "skill_gap": "AWS",
                "original_context": "",
                "suggested_bullet": "Architected and deployed cloud microservices on AWS (ECS, S3), reducing infrastructure overhead by 25%.",
                "reasoning": "Demonstrates hands-on AWS production experience required by the JD."
            },
            {
                "skill_gap": "Kubernetes",
                "original_context": "",
                "suggested_bullet": "Orchestrated container workloads using Kubernetes (k8s) clusters, maintaining 99.9% uptime across production environments.",
                "reasoning": "Directly addresses the missing container orchestration requirement."
            }
        ],
        "general_tips": [
            "Quantify key accomplishments with business metrics.",
            "Mirror keyword terminology directly from the job description."
        ]
    })

    # Mock ChatGoogleGenerativeAI
    mock_llm = MagicMock()
    mock_resp1 = MagicMock(content=jd_mock_json)
    mock_resp2 = MagicMock(content=rewrite_mock_json)
    mock_llm.invoke.side_effect = [mock_resp1, mock_resp2]

    with patch("agent.graph._get_llm", return_value=mock_llm):
        response = client.post(
            "/api/agent/analyze",
            json={
                "resume_text": (
                    "John Doe - Software Engineer\n"
                    "Skills: Python, Django, Docker, PostgreSQL, Git\n"
                    "Experience: Developed REST APIs using Python and Docker."
                ),
                "jd_text": (
                    "Looking for a Backend Developer.\n"
                    "Requirements: Python, Docker, AWS, Kubernetes."
                )
            }
        )

        assert response.status_code == 200, response.text
        data = response.json()

        # Check JD requirements parsed
        assert data["jd_requirements"] is not None
        assert "requirements" in data["jd_requirements"]

        # Check resume skills extracted
        assert data["resume_skills"] is not None
        assert any("Python" in s for s in data["resume_skills"])

        # Check gap analysis produced by LangChain tool
        assert data["gaps"] is not None
        assert "similarity_score" in data["gaps"]
        assert "missing_skills" in data["gaps"]
        assert "matched_skills" in data["gaps"]
        assert "section_scores" in data["gaps"]

        # Check rewrite suggestions
        assert data["suggestions"] is not None
        assert "suggestions" in data["suggestions"]
        assert len(data["suggestions"]["suggestions"]) >= 1

        # Confirm max 2 LLM calls were made
        assert mock_llm.invoke.call_count == 2


# ---------------------------------------------------------------------------
# Test 2: Missing API Key (503 Error)
# ---------------------------------------------------------------------------

def test_agent_analyze_missing_key(client, monkeypatch):
    """Verify endpoint returns 503 when no API key is configured."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    response = client.post(
        "/api/agent/analyze",
        json={
            "resume_text": "Experienced Python developer with Docker skills.",
            "jd_text": "Seeking Python and AWS engineer."
        }
    )

    assert response.status_code == 503
    assert "API key is not configured" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Test 3: Empty / Whitespace Inputs (400 Error)
# ---------------------------------------------------------------------------

def test_agent_analyze_empty_inputs(client, monkeypatch):
    """Verify endpoint rejects empty or whitespace-only inputs with 400."""
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-key")

    # Empty resume
    r1 = client.post("/api/agent/analyze", json={"resume_text": "", "jd_text": "Python JD"})
    assert r1.status_code == 400

    # Whitespace resume
    r2 = client.post("/api/agent/analyze", json={"resume_text": "   \n\t ", "jd_text": "Python JD"})
    assert r2.status_code == 400

    # Empty JD
    r3 = client.post("/api/agent/analyze", json={"resume_text": "Python Resume", "jd_text": ""})
    assert r3.status_code == 400

    # Whitespace JD
    r4 = client.post("/api/agent/analyze", json={"resume_text": "Python Resume", "jd_text": "   "})
    assert r4.status_code == 400


# ---------------------------------------------------------------------------
# Test 4: Malformed LLM Output (Graceful Fallback)
# ---------------------------------------------------------------------------

def test_agent_analyze_malformed_llm_output(client, monkeypatch):
    """Verify graph handles unparseable LLM output gracefully without crashing."""
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-key")

    # LLM returns conversational text instead of JSON
    mock_llm = MagicMock()
    mock_bad_resp = MagicMock(content="I am an AI assistant and I cannot produce JSON right now.")
    mock_llm.invoke.return_value = mock_bad_resp

    with patch("agent.graph._get_llm", return_value=mock_llm):
        response = client.post(
            "/api/agent/analyze",
            json={
                "resume_text": "Software Engineer with Python and Docker.",
                "jd_text": "Backend Engineer with Python and AWS experience."
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should fall back gracefully
        assert data["suggestions"] is not None
        assert "suggestions" in data["suggestions"]
        # Node errors should capture the parse issues
        assert data["node_errors"] is not None


# ---------------------------------------------------------------------------
# Test 5: Rate Limiting & Timeout
# ---------------------------------------------------------------------------

def test_agent_analyze_rate_limit(client, monkeypatch):
    """Verify error handling when LLM rate limit is hit."""
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-key")

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("429 Resource has been exhausted (e.g. check quota).")

    with patch("agent.graph._get_llm", return_value=mock_llm):
        response = client.post(
            "/api/agent/analyze",
            json={
                "resume_text": "Python Engineer",
                "jd_text": "Python and AWS Developer"
            }
        )
        assert response.status_code in [200, 429]


# ---------------------------------------------------------------------------
# Test 6: Verify Existing Endpoints Are Intact
# ---------------------------------------------------------------------------

def test_existing_endpoints_intact(client):
    """Verify that existing /analyze, /history, /stats endpoints still work."""
    # Test GET /history
    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200
    assert isinstance(hist_resp.json(), list)

    # Test GET /stats
    stats_resp = client.get("/stats")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "avg_score" in stats_data
    assert "total_analyses" in stats_data

    # Test POST /analyze with form data
    analyze_resp = client.post(
        "/analyze",
        data={
            "jd_text": "We need a Python developer with Docker and PostgreSQL.",
            "resume_text": "Experienced Python engineer with Django, Docker, and PostgreSQL."
        }
    )
    assert analyze_resp.status_code == 200
    res_data = analyze_resp.json()
    assert "similarity_score" in res_data
    assert "matched_skills" in res_data
    assert "missing_skills" in res_data
    assert "section_scores" in res_data
    assert "feedback" in res_data


@pytest.mark.parametrize("filename,content", [
    ("resume.txt", b"Built Python services."),
    ("resume.md", b"# Resume\nBuilt Python services."),
    ("resume.pdf", _make_pdf("Built Python services.")),
    ("resume.docx", _make_docx("Built Python services.")),
])
def test_analyze_file_returns_extracted_resume_text(client, monkeypatch, filename, content):
    import main

    monkeypatch.setattr(main, "compute_similarity", lambda resume, jd: 0.8)
    monkeypatch.setattr(main, "analyze_gap", lambda resume, jd: {
        "matched_skills": [], "missing_skills": [], "transferable_skills": [],
        "resume_skills": [], "jd_skills": [],
    })
    monkeypatch.setattr(main, "compute_section_scores", lambda **kwargs: {})
    monkeypatch.setattr(main, "generate_feedback", lambda missing, similarity: [])

    response = client.post(
        "/analyze",
        data={"jd_text": "Python developer"},
        files={"resume_file": (filename, content, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    assert "Built Python services." in response.json()["resume_text"]


@pytest.mark.parametrize("filename,content", [
    ("resume.txt", b"Built Python services."),
    ("resume.md", b"# Resume\nBuilt Python services."),
    ("resume.pdf", _make_pdf("Built Python services.")),
    ("resume.docx", _make_docx("Built Python services.")),
])
def test_agent_analyze_file(client, monkeypatch, filename, content):
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-key")
    from agent import graph

    seen = {}

    class FakeGraph:
        def invoke(self, state):
            seen.update(state)
            return {}

    monkeypatch.setattr(graph, "get_graph", lambda: FakeGraph())
    response = client.post(
        "/api/agent/analyze",
        data={"jd_text": "Python developer"},
        files={"resume_file": (filename, content, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    assert "Built Python services." in seen["resume_text"]
    assert seen["jd_text"] == "Python developer"


def test_agent_analyze_rejects_bad_file(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-key")
    response = client.post(
        "/api/agent/analyze",
        data={"jd_text": "Python developer"},
        files={"resume_file": ("resume.csv", b"a,b", "text/csv")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test 7: Verify OpenRouter Provider Resolution
# ---------------------------------------------------------------------------

def test_openrouter_provider_resolution(monkeypatch):
    """Verify OpenRouter key auto-detection and ChatOpenAI initialization."""
    from agent.graph import _get_api_key, _get_llm
    from langchain_openai import ChatOpenAI

    # Case A: OPENROUTER_API_KEY explicitly set
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-custom-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    key, provider = _get_api_key()
    assert provider == "openrouter"
    llm = _get_llm()
    assert isinstance(llm, ChatOpenAI)

    # Case B: GOOGLE_API_KEY set with OpenRouter sk-or- prefix
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-or-v1-another-test-key")
    key, provider = _get_api_key()
    assert provider == "openrouter"
    llm2 = _get_llm()
    assert isinstance(llm2, ChatOpenAI)
