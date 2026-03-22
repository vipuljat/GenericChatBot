"""Helpers for retrieval-oriented chunk metadata and query normalization."""

import re
from typing import Any, Dict, List, Optional


QUERY_STOPWORDS = {
    "a", "an", "and", "are", "can", "could", "do", "does", "for", "how",
    "i", "in", "is", "me", "of", "on", "or", "our", "should", "tell",
    "the", "to", "we", "what", "who", "why", "with", "you"
}

SECTION_PATTERNS = {
    "team_members": [
        r"\bteam members?\b",
        r"\bmembers?\s*:",
    ],
    "team_lead": [
        r"\bteam lead\b",
        r"\blead\b\s*:",
        r"\bheaded by\b",
    ],
    "leadership": [
        r"\bleadership team\b",
        r"\bchief\b",
        r"\bco[\-\s]?founder\b",
        r"\bhead\b",
        r"\bmanager\b",
    ],
    "department": [
        r"\bdepartment\b",
        r"\bteam breakdown\b",
        r"\bhuman resources\b",
        r"\bfrontend\b",
        r"\bbackend\b",
        r"\bfinance\b",
        r"\badmin(?:istration)?\b",
        r"\bai\s*/\s*ml\b",
        r"\bmachine learning\b",
    ],
    "core_team": [
        r"\bcore teams?\b",
        r"\bcross[\-\s]?functional\b",
    ],
    "qa_pair": [
        r"\bq\d+\.",
        r"\bans?\s*:",
    ],
}


def normalize_match_text(text: str) -> str:
    """Lowercase and normalize punctuation and whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stem_token(token: str) -> str:
    """Very light stemming to reduce obvious inflection noise."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def extract_content_tokens(text: str) -> List[str]:
    """Return lightly normalized content tokens suitable for retrieval."""
    tokens: List[str] = []
    for token in normalize_match_text(text).split():
        if len(token) < 2 or token in QUERY_STOPWORDS:
            continue
        tokens.append(stem_token(token))
    return tokens


def looks_like_heading(text: str) -> bool:
    """Best-effort heading detection for section-oriented documents."""
    candidate = re.sub(r"\s+", " ", (text or "").strip())
    if not candidate or len(candidate) > 100:
        return False

    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    if len(lines) > 2:
        return False

    words = candidate.split()
    if not words or len(words) > 10:
        return False

    if candidate.endswith((".", "?", "!")):
        return False

    capitalized = sum(1 for word in words if word[:1].isupper() or word.isupper())
    return capitalized >= max(1, len(words) - 1)


def infer_section_heading(chunk_text: str) -> Optional[str]:
    """
    Infer a representative heading from the chunk itself.
    We prefer explicit heading-like first lines over generic prose.
    """
    lines = [line.strip() for line in re.split(r"\n+", chunk_text or "") if line.strip()]
    for line in lines[:4]:
        if looks_like_heading(line):
            return line[:120]
    return None


def infer_chunk_retrieval_metadata(chunk_text: str) -> Dict[str, Any]:
    """
    Infer lightweight retrieval metadata from chunk text for better reranking.
    The goal is generic structure detection, not company-specific labeling.
    """
    text = chunk_text or ""
    normalized = normalize_match_text(text)
    content_tokens = extract_content_tokens(text)
    unique_keywords: List[str] = []
    seen = set()
    for token in content_tokens:
        if token in seen:
            continue
        seen.add(token)
        unique_keywords.append(token)
        if len(unique_keywords) >= 32:
            break

    section_labels: List[str] = []
    for label, patterns in SECTION_PATTERNS.items():
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns):
            section_labels.append(label)

    has_name_role_pairs = bool(
        re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*[–:\-]\s*[A-Za-z]", text)
    )
    looks_like_bulleted_people = bool(
        re.search(r"(?:^|\n)\s*[•\-]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text)
    )
    section_heading = infer_section_heading(text)
    section_heading_tokens = extract_content_tokens(section_heading or "")

    return {
        "retrieval_keywords": unique_keywords,
        "section_heading": section_heading,
        "section_heading_tokens": section_heading_tokens,
        "section_labels": section_labels,
        "contains_team_members": "team_members" in section_labels,
        "contains_team_lead": "team_lead" in section_labels,
        "contains_leadership": "leadership" in section_labels,
        "contains_department": "department" in section_labels,
        "contains_core_team": "core_team" in section_labels,
        "contains_qa_pair": "qa_pair" in section_labels,
        "contains_name_role_pairs": has_name_role_pairs,
        "contains_people_list": looks_like_bulleted_people or has_name_role_pairs,
    }
