import re
from typing import List

import torch

from models import ml_models

MAX_BULLET_CHARS = 160
GENERIC_FALLBACKS = [
    "Quantify impact: add metrics for scope, speed, revenue, quality, or cost improvements.",
    "Match the role: mirror priority keywords from the job description in relevant experience bullets.",
    "Show delivery: connect tools, projects, and outcomes so recruiters can verify hands-on experience.",
]

def _parse_bullets(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [line.lstrip("-•* ").strip() for line in lines if line.lstrip().startswith(("-", "•", "*"))]
    if len(bullet_lines) >= 3:
        return bullet_lines[:3]

    numbered = []
    for line in lines:
        if re.match(r"^\d+[\).]", line):
            numbered.append(re.sub(r"^\d+[\).]\s*", "", line))
    if len(numbered) >= 3:
        return numbered[:3]

    sentences = re.split(r"[.;]\s+", text.strip())
    parsed = [s.strip().lstrip("-•* ") for s in sentences if s.strip()]
    return parsed[:3]

def _generate_text(
    prompt: str,
    min_new_tokens: int = 0,
    max_new_tokens: int = 48,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_beams: int = 2,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
) -> str:
    if ml_models.t5_model is None or ml_models.t5_tokenizer is None:
        raise RuntimeError("T5 model is not loaded")

    tokenizer = ml_models.t5_tokenizer
    model = ml_models.t5_model
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    if do_sample:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["do_sample"] = False
        generate_kwargs["num_beams"] = num_beams
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", text.lower()).strip()

def _clean_bullet(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().strip("\"'").lstrip("-•* ")).strip()
    cleaned = re.sub(r"^(resume bullet|bullet|tip)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned

def _contains_obvious_garbage(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9+#.]+", text)
    if len(words) < 6:
        return True
    if re.search(r"\b(tell the candidate|write|start exactly|output|under 160 characters|direct imperative|after the colon)\b", text, re.IGNORECASE):
        return True
    unique_ratio = len(set(word.lower() for word in words)) / len(words)
    if unique_ratio < 0.55:
        return True
    uppercase_tokens = re.findall(r"\b[A-Z0-9+#.]{2,}\b", text)
    if len(uppercase_tokens) >= 5 and len(uppercase_tokens) / max(len(words), 1) > 0.35:
        return True
    if re.search(r"(.)\1{5,}", text):
        return True
    return False

def _mentions_skill(text: str, skill: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_skill = _normalize_text(skill)
    if not normalized_skill:
        return False
    if normalized_skill in normalized_text:
        return True
    skill_tokens = [token for token in normalized_skill.split() if len(token) > 1]
    return bool(skill_tokens) and all(token in normalized_text for token in skill_tokens)

def _is_valid_bullet(text: str, skill: str = None) -> bool:
    if not text or len(text) > MAX_BULLET_CHARS:
        return False
    if _contains_obvious_garbage(text):
        return False
    if skill and not _mentions_skill(text, skill):
        return False
    if skill and not _normalize_text(text).startswith(f"demonstrate {_normalize_text(skill)}"):
        return False
    if not re.search(r"\b(project|tool|tools|metric|measurable|outcome|impact|delivered|improved|reduced|increased)\b", text, re.IGNORECASE):
        return False
    return True

def _skill_fallback(skill: str) -> str:
    return f"Demonstrate {skill}: cite a concrete project, tools used, and measurable outcome."

def _valid_or_fallback(text: str, fallback: str, skill: str = None) -> str:
    bullet = _clean_bullet(text)
    if _is_valid_bullet(bullet, skill):
        return bullet
    fallback = _clean_bullet(fallback)
    if len(fallback) <= MAX_BULLET_CHARS:
        return fallback
    return fallback[:MAX_BULLET_CHARS - 1].rstrip(" ,.;:") + "…"

def generate_feedback(missing_skills: list, score: float) -> list:
    if missing_skills:
        bullets: List[str] = []
        for skill in missing_skills[:3]:
            prompt = (
                f"Resume advice for missing skill {skill}: "
                f"Demonstrate {skill}: highlight a project using {skill}, relevant tools, and a measurable outcome."
            )
            try:
                text = _generate_text(prompt, max_new_tokens=36, do_sample=False, num_beams=2)
            except RuntimeError:
                text = ""
            parsed = _parse_bullets(text)
            bullet = parsed[0] if parsed else text.strip().lstrip("-•* ")
            bullets.append(_valid_or_fallback(bullet, _skill_fallback(skill), skill))
        return bullets

    prompt = (
        "Write exactly three concise resume improvement tips. "
        "Each bullet must be under 160 characters and include recruiter guidance about measurable impact, keywords, or delivery. "
        "Output bullets only."
    )

    try:
        text = _generate_text(prompt, max_new_tokens=96, do_sample=False, num_beams=2)
    except RuntimeError:
        text = ""
    bullets = _parse_bullets(text)
    feedback: List[str] = []
    for index, fallback in enumerate(GENERIC_FALLBACKS):
        candidate = bullets[index] if index < len(bullets) else ""
        feedback.append(_valid_or_fallback(candidate, fallback))
    return feedback