"""
LangGraph agent for resume analysis.

A multi-step stateful graph that reads a resume and a JD, finds skill gaps,
and suggests resume improvements. Uses two LLM calls (parse_jd and
suggest_rewrites) and reuses existing analyzer functions for the non-LLM nodes.

Graph: parse_jd -> extract_resume_skills -> analyze_gaps -> suggest_rewrites
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from typing_extensions import TypedDict

# Load environment variables (.env) safely
load_dotenv(find_dotenv())

# Ensure backend root directory is in sys.path for direct imports
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output validation
# ---------------------------------------------------------------------------

class JDRequirement(BaseModel):
    """A single requirement extracted from a job description."""
    category: str = Field(default="general", description="Category like technical, soft_skill, experience, education")
    requirement: str = Field(description="The requirement text")
    skills: List[str] = Field(default_factory=list, description="Specific skills mentioned")


class JDParseResult(BaseModel):
    """Structured result from parsing a job description."""
    title: str = Field(default="", description="Job title if found")
    requirements: List[JDRequirement] = Field(default_factory=list)
    all_skills: List[str] = Field(default_factory=list, description="Flat list of all skills found")


class RewriteSuggestion(BaseModel):
    """A single resume rewrite suggestion."""
    skill_gap: str = Field(description="The missing skill this addresses")
    original_context: str = Field(default="", description="What the resume currently says, if relevant")
    suggested_bullet: str = Field(description="The rewritten or new resume bullet point")
    reasoning: str = Field(default="", description="Why this change helps")


class RewriteResult(BaseModel):
    """Structured result from the suggest_rewrites node."""
    suggestions: List[RewriteSuggestion] = Field(default_factory=list)
    general_tips: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """State flowing through the LangGraph agent."""
    # Inputs
    resume_text: str
    jd_text: str

    # Node outputs
    jd_requirements: Optional[Dict[str, Any]]     # From parse_jd
    resume_skills: Optional[List[str]]             # From extract_resume_skills
    jd_skills: Optional[List[str]]                 # From parse_jd / extract_resume_skills
    gaps: Optional[Dict[str, Any]]                 # From analyze_gaps
    suggestions: Optional[Dict[str, Any]]          # From suggest_rewrites

    # Meta
    error: Optional[str]
    node_errors: Optional[Dict[str, str]]          # Per-node error tracking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _safe_parse_json(text: str) -> Optional[dict]:
    """Try to parse JSON from LLM text, stripping fences first."""
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to find the first { ... } or [ ... ] block
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                pass
    return None


def _get_api_key() -> tuple[Optional[str], str]:
    """
    Retrieve API key and provider type from environment.
    Returns (api_key, provider) where provider is 'openrouter' or 'google'.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return openrouter_key, "openrouter"

    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if google_key:
        if google_key.startswith("sk-or-"):
            return google_key, "openrouter"
        return google_key, "google"

    return None, "none"


def _get_llm():
    """Create a chat model instance (supporting OpenRouter or Google AI Studio)."""
    api_key, provider = _get_api_key()
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY or OPENROUTER_API_KEY environment variable is not set. "
            "Set your API key in your .env file."
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        model_name = (
            os.environ.get("OPENROUTER_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or "google/gemini-2.5-flash"
        )
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
            timeout=45,
            max_retries=2,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        timeout=45,
        max_retries=2,
    )


def _ensure_models_loaded():
    """Ensure existing ML models (SentenceTransformers, NER) are loaded once."""
    from models import ml_models
    if ml_models.embedding_model is None or ml_models.ner_pipeline is None:
        logger.info("Initializing ML models for agent...")
        ml_models.load_models()


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def parse_jd(state: AgentState) -> dict:
    """
    LLM call #1: Extract structured requirements and skills from the JD.
    """
    jd_text = state.get("jd_text", "")
    node_errors = dict(state.get("node_errors") or {})

    if not jd_text.strip():
        return {
            "jd_requirements": {"title": "", "requirements": [], "all_skills": []},
            "jd_skills": [],
            "node_errors": {**node_errors, "parse_jd": "Empty JD text"},
        }

    prompt = f"""Analyze this job description and extract structured information.

Return ONLY valid JSON (no markdown fences, no explanation) with this exact structure:
{{
  "title": "job title if found, or empty string",
  "requirements": [
    {{
      "category": "technical",
      "requirement": "the requirement text",
      "skills": ["skill1", "skill2"]
    }}
  ],
  "all_skills": ["flat", "list", "of", "every", "skill", "mentioned"]
}}

Job Description:
{jd_text}"""

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        parsed = _safe_parse_json(raw)
        if parsed is None:
            logger.warning("Failed to parse JD LLM output as JSON, using fallback")
            return {
                "jd_requirements": {"title": "", "requirements": [], "all_skills": []},
                "jd_skills": [],
                "node_errors": {**node_errors, "parse_jd": "LLM returned unparseable JSON"},
            }

        try:
            result = JDParseResult(**parsed)
        except ValidationError as ve:
            logger.warning("JD parse validation error: %s", ve)
            all_skills = parsed.get("all_skills", [])
            skills_list = [s for s in all_skills if isinstance(s, str)] if isinstance(all_skills, list) else []
            return {
                "jd_requirements": parsed if isinstance(parsed, dict) else {},
                "jd_skills": skills_list,
                "node_errors": {**node_errors, "parse_jd": f"Partial parse: {ve}"},
            }

        return {
            "jd_requirements": result.model_dump(),
            "jd_skills": result.all_skills,
        }

    except EnvironmentError:
        raise
    except Exception as e:
        logger.error("parse_jd error: %s", e)
        return {
            "jd_requirements": {"title": "", "requirements": [], "all_skills": []},
            "jd_skills": [],
            "node_errors": {**node_errors, "parse_jd": str(e)},
        }


def extract_resume_skills(state: AgentState) -> dict:
    """
    No LLM call. Reuses existing extract_skills() from analyzer.py.
    """
    from analyzer import extract_skills

    resume_text = state.get("resume_text", "")
    node_errors = dict(state.get("node_errors") or {})

    if not resume_text.strip():
        return {
            "resume_skills": [],
            "node_errors": {**node_errors, "extract_resume_skills": "Empty resume text"},
        }

    try:
        _ensure_models_loaded()
        skills = extract_skills(resume_text)
        return {"resume_skills": sorted(skills)}
    except Exception as e:
        logger.error("extract_resume_skills error: %s", e)
        return {
            "resume_skills": [],
            "node_errors": {**node_errors, "extract_resume_skills": str(e)},
        }


def analyze_gaps(state: AgentState) -> dict:
    """
    No LLM call. Wraps existing analyzer functions as a LangChain tool and
    invokes it. Reuses compute_similarity and analyze_gap from analyzer.py.
    """
    from langchain_core.tools import tool

    from analyzer import analyze_gap, compute_similarity, compute_section_scores
    from parser import segment_resume

    resume_text = state.get("resume_text", "")
    jd_text = state.get("jd_text", "")
    node_errors = dict(state.get("node_errors") or {})

    @tool
    def resume_jd_scorer(resume: str, jd: str) -> dict:
        """Score a resume against a job description using semantic similarity,
        skill gap detection, and section-level analysis."""
        similarity = compute_similarity(resume, jd)
        gap_data = analyze_gap(resume, jd)
        sections = segment_resume(resume)
        section_scores = compute_section_scores(
            resume_text=resume,
            jd_text=jd,
            resume_sections=sections,
            resume_skills=gap_data["resume_skills"],
            jd_skills=gap_data["jd_skills"],
            similarity_score=similarity,
        )
        return {
            "similarity_score": similarity,
            "matched_skills": gap_data["matched_skills"],
            "missing_skills": gap_data["missing_skills"],
            "transferable_skills": gap_data["transferable_skills"],
            "resume_skills": gap_data["resume_skills"],
            "jd_skills": gap_data["jd_skills"],
            "section_scores": section_scores,
        }

    if not resume_text.strip() or not jd_text.strip():
        return {
            "gaps": {
                "similarity_score": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "transferable_skills": [],
                "resume_skills": [],
                "jd_skills": [],
                "section_scores": {},
            },
            "node_errors": {**node_errors, "analyze_gaps": "Empty input"},
        }

    try:
        _ensure_models_loaded()
        result = resume_jd_scorer.invoke({"resume": resume_text, "jd": jd_text})
        return {"gaps": result}
    except Exception as e:
        logger.error("analyze_gaps error: %s", e)
        return {
            "gaps": {
                "similarity_score": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "transferable_skills": [],
                "resume_skills": [],
                "jd_skills": [],
                "section_scores": {},
            },
            "node_errors": {**node_errors, "analyze_gaps": str(e)},
        }


def suggest_rewrites(state: AgentState) -> dict:
    """
    LLM call #2: Generate concrete resume bullet rewrites targeting the gaps.
    """
    gaps = state.get("gaps") or {}
    jd_requirements = state.get("jd_requirements") or {}
    resume_text = state.get("resume_text", "")
    missing_skills = gaps.get("missing_skills", [])
    node_errors = dict(state.get("node_errors") or {})

    if not missing_skills:
        return {
            "suggestions": {
                "suggestions": [],
                "general_tips": ["Your resume already covers key skills in this job description. Focus on quantifying your measurable business impact."],
            }
        }

    top_gaps = missing_skills[:5]

    prompt = f"""You are a professional resume coach. A candidate's resume is missing these skills that the job requires:

Missing skills: {json.dumps(top_gaps)}

Job requirements summary: {json.dumps(jd_requirements.get("requirements", [])[:5])}

Resume excerpt:
{resume_text[:1500]}

For each missing skill, suggest a concrete resume bullet point the candidate could add or rewrite.
Each bullet should be specific, start with an action verb, and include a measurable outcome where possible.

Return ONLY valid JSON (no markdown fences, no explanation) with this exact structure:
{{
  "suggestions": [
    {{
      "skill_gap": "the missing skill",
      "original_context": "what the resume currently says about this area, or empty string",
      "suggested_bullet": "a rewritten or new bullet point for the resume",
      "reasoning": "brief explanation of why this change helps"
    }}
  ],
  "general_tips": ["1-2 general resume improvement tips"]
}}"""

    fallback_suggestions = [
        {
            "skill_gap": skill,
            "original_context": "",
            "suggested_bullet": f"Demonstrate {skill}: cite a concrete project, tools used, and measurable outcome (e.g. reduced latency by 20%, improved coverage).",
            "reasoning": f"Job description requires {skill}, which was missing from the parsed resume.",
        }
        for skill in top_gaps
    ]

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        parsed = _safe_parse_json(raw)
        if parsed is None:
            logger.warning("Failed to parse rewrites LLM output, using fallback")
            return {
                "suggestions": {
                    "suggestions": fallback_suggestions,
                    "general_tips": ["Quantify achievements with metrics. Mirror keywords from the job description."],
                },
                "node_errors": {**node_errors, "suggest_rewrites": "LLM returned unparseable JSON, used fallback"},
            }

        try:
            result = RewriteResult(**parsed)
        except ValidationError as ve:
            logger.warning("Rewrite validation error: %s", ve)
            suggestions_raw = parsed.get("suggestions", [])
            valid_suggs = []
            for s in (suggestions_raw if isinstance(suggestions_raw, list) else []):
                if isinstance(s, dict) and "skill_gap" in s and "suggested_bullet" in s:
                    valid_suggs.append(s)
            tips = parsed.get("general_tips", [])
            return {
                "suggestions": {
                    "suggestions": valid_suggs or fallback_suggestions,
                    "general_tips": [t for t in tips if isinstance(t, str)] or ["Quantify achievements. Mirror JD keywords."],
                },
                "node_errors": {**node_errors, "suggest_rewrites": f"Partial parse: {ve}"},
            }

        return {"suggestions": result.model_dump()}

    except EnvironmentError:
        raise
    except Exception as e:
        logger.error("suggest_rewrites error: %s", e)
        return {
            "suggestions": {
                "suggestions": fallback_suggestions,
                "general_tips": ["Focus on quantifying impact and mirroring JD keywords."],
            },
            "node_errors": {**node_errors, "suggest_rewrites": str(e)},
        }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph StateGraph."""
    from langgraph.graph import StateGraph

    workflow = StateGraph(AgentState)

    workflow.add_node("parse_jd", parse_jd)
    workflow.add_node("extract_resume_skills", extract_resume_skills)
    workflow.add_node("analyze_gaps", analyze_gaps)
    workflow.add_node("suggest_rewrites", suggest_rewrites)

    workflow.set_entry_point("parse_jd")
    workflow.add_edge("parse_jd", "extract_resume_skills")
    workflow.add_edge("extract_resume_skills", "analyze_gaps")
    workflow.add_edge("analyze_gaps", "suggest_rewrites")

    return workflow.compile()


# Module-level compiled graph (lazy singleton)
_compiled_graph = None


def get_graph():
    """Return a cached compiled graph instance."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
