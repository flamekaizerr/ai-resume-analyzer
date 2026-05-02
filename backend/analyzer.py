import json
import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Set

from sentence_transformers import util

from models import ml_models

taxonomy_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "data", "skill_taxonomy.json")

SOFT_SKILLS = {
    "communication",
    "collaboration",
    "teamwork",
    "leadership",
    "problem solving",
    "problem-solving",
    "critical thinking",
    "adaptability",
    "ownership",
    "stakeholder",
    "mentoring",
    "presentation",
    "time management",
    "organization",
    "customer focus",
    "empathy",
    "agile",
    "scrum",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "for", "with", "on", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "as", "it", "its", "we", "our", "you", "your", "they", "their", "i", "me",
    "my", "us", "them", "will", "can", "should", "must", "may", "might", "not", "no",
    "do", "does", "did", "done", "if", "then", "than", "but", "so", "such", "also",
}

def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-/+#.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _chunk_text(text: str, max_chars: int = 900) -> Iterable[str]:
    if len(text) <= max_chars:
        yield text
        return

    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        if end < len(text):
            last_break = max(chunk.rfind(". "), chunk.rfind("\n"), chunk.rfind("; "))
            if last_break > max_chars * 0.6:
                end = start + last_break + 1
                chunk = text[start:end]
        yield chunk
        start = end

@lru_cache(maxsize=1)
def load_taxonomy() -> Dict[str, str]:
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {_normalize_text(k): v for k, v in raw.items()}

def _match_taxonomy_in_text(text: str, taxonomy: Dict[str, str]) -> Set[str]:
    normalized_text = _normalize_text(text)
    matches: Set[str] = set()
    for key, canonical in taxonomy.items():
        if not key:
            continue
        pattern = rf"(?<!\w){re.escape(key)}(?!\w)"
        if re.search(pattern, normalized_text):
            matches.add(canonical)
    return matches

def extract_skills(text: str) -> Set[str]:
    """Extract skills using NER and match them against the taxonomy."""
    if not text.strip():
        return set()

    taxonomy = load_taxonomy()
    skills: Set[str] = set()

    if ml_models.ner_pipeline is None:
        raise RuntimeError("NER pipeline is not loaded")

    for chunk in _chunk_text(text):
        ner_results = ml_models.ner_pipeline(chunk)
        for entity in ner_results:
            candidate = entity.get("word", "")
            if not candidate:
                continue
            candidate = candidate.replace("##", "")
            normalized = _normalize_text(candidate)
            if normalized in taxonomy:
                skills.add(taxonomy[normalized])
                continue
            if normalized.endswith("s") and normalized[:-1] in taxonomy:
                skills.add(taxonomy[normalized[:-1]])

    skills.update(_match_taxonomy_in_text(text, taxonomy))
    return skills

def compute_similarity(resume_text: str, jd_text: str) -> float:
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    if ml_models.embedding_model is None:
        raise RuntimeError("Embedding model is not loaded")

    embeddings = ml_models.embedding_model.encode(
        [resume_text, jd_text],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return float(max(0.0, min(1.0, similarity)))

def analyze_gap(resume_text: str, jd_text: str) -> dict:
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched = sorted(resume_skills.intersection(jd_skills))
    missing = sorted(jd_skills.difference(resume_skills))
    transferable = sorted(resume_skills.difference(jd_skills))[:5]

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "transferable_skills": transferable,
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
    }

def _keyword_set(text: str) -> Set[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {w for w in words if w not in STOPWORDS}

def _soft_skill_score(resume_text: str, jd_text: str) -> float:
    resume_norm = _normalize_text(resume_text)
    jd_norm = _normalize_text(jd_text)

    resume_found = {skill for skill in SOFT_SKILLS if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", resume_norm)}
    jd_found = {skill for skill in SOFT_SKILLS if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", jd_norm)}

    if jd_found:
        return len(resume_found.intersection(jd_found)) / max(1, len(jd_found))
    return min(1.0, len(resume_found) / max(1, len(SOFT_SKILLS)))

def _education_score(education_text: str, jd_text: str) -> float:
    edu_norm = _normalize_text(education_text or "")
    jd_norm = _normalize_text(jd_text)
    jd_requires = bool(re.search(r"\b(bachelor|master|phd|degree|mba)\b", jd_norm))
    has_degree = bool(re.search(r"\b(b\.s\.|bachelor|b\.a\.|master|m\.s\.|phd|doctorate|mba)\b", edu_norm))

    if not edu_norm:
        return 0.2 if jd_requires else 0.4
    if has_degree:
        return 1.0
    return 0.7 if jd_requires else 0.6

def compute_section_scores(
    resume_text: str,
    jd_text: str,
    resume_sections: Dict[str, str],
    resume_skills: List[str],
    jd_skills: List[str],
    similarity_score: float,
) -> Dict[str, float]:
    resume_skill_set = set(resume_skills)
    jd_skill_set = set(jd_skills)

    if jd_skill_set:
        technical = len(resume_skill_set.intersection(jd_skill_set)) / max(1, len(jd_skill_set))
    else:
        technical = similarity_score

    resume_keywords = _keyword_set(resume_text)
    jd_keywords = _keyword_set(jd_text)
    keywords = len(resume_keywords.intersection(jd_keywords)) / max(1, len(jd_keywords)) if jd_keywords else 0.0

    experience_text = resume_sections.get("experience") or resume_text
    experience_match = compute_similarity(experience_text, jd_text) if experience_text.strip() else similarity_score

    soft_skills = _soft_skill_score(resume_text, jd_text)
    education = _education_score(resume_sections.get("education", ""), jd_text)

    overall = (technical + soft_skills + experience_match + education + keywords) / 5

    return {
        "technical_skills": float(max(0.0, min(1.0, technical))),
        "soft_skills": float(max(0.0, min(1.0, soft_skills))),
        "experience_match": float(max(0.0, min(1.0, experience_match))),
        "education": float(max(0.0, min(1.0, education))),
        "keywords": float(max(0.0, min(1.0, keywords))),
        "overall": float(max(0.0, min(1.0, overall))),
    }
