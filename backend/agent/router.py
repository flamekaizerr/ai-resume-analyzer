"""
FastAPI router for the LangGraph agent endpoint.

Exposes:
- POST /api/agent/analyze (when mounted with /api prefix)
- POST /agent/analyze (when mounted at root)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Ensure environment is loaded
load_dotenv(find_dotenv())

# Ensure backend root is in sys.path
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AgentAnalyzeRequest(BaseModel):
    resume_text: str = Field(default="", description="The resume text to analyze")
    jd_text: str = Field(default="", description="The job description text")


class AgentAnalyzeResponse(BaseModel):
    jd_requirements: Optional[Dict[str, Any]] = None
    resume_skills: Optional[List[str]] = None
    jd_skills: Optional[List[str]] = None
    gaps: Optional[Dict[str, Any]] = None
    suggestions: Optional[Dict[str, Any]] = None
    node_errors: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=AgentAnalyzeResponse)
async def agent_analyze(request: AgentAnalyzeRequest):
    """Run the LangGraph agent to analyze a resume against a job description."""

    # 1. Validate empty or whitespace-only inputs (returns 400)
    if not request.resume_text or not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")
    if not request.jd_text or not request.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty")

    # 2. Check for API key before running graph (returns 503)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI Agent is not available: GOOGLE_API_KEY is not configured. "
                "Set the GOOGLE_API_KEY environment variable with a Google AI Studio API key."
            ),
        )

    # 3. Lazy import so app starts even if agent dependencies are missing
    try:
        from agent.graph import get_graph
    except ImportError as e:
        logger.error("Agent dependencies not installed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"AI Agent dependencies are not installed: {e}",
        )

    # 4. Invoke the stateful graph
    try:
        graph = get_graph()
        result = graph.invoke({
            "resume_text": request.resume_text.strip(),
            "jd_text": request.jd_text.strip(),
        })

        return AgentAnalyzeResponse(
            jd_requirements=result.get("jd_requirements"),
            resume_skills=result.get("resume_skills"),
            jd_skills=result.get("jd_skills"),
            gaps=result.get("gaps"),
            suggestions=result.get("suggestions"),
            node_errors=result.get("node_errors"),
        )

    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        error_type = type(e).__name__
        error_str = str(e).lower()
        logger.error("Agent error (%s): %s", error_type, e)

        # Handle rate limiting / quota exhaustion
        if "429" in error_str or "resource_exhausted" in error_str or "rate limit" in error_str or "quota" in error_str:
            raise HTTPException(
                status_code=429,
                detail="AI Agent rate limit or quota exceeded. Please wait a moment and try again.",
            )

        # Handle timeouts
        if "timeout" in error_str or "timed out" in error_str:
            raise HTTPException(
                status_code=504,
                detail="AI Agent request timed out. Please try again with shorter text.",
            )

        raise HTTPException(
            status_code=500,
            detail=f"AI Agent processing error: {error_type} - {str(e)[:200]}",
        )
