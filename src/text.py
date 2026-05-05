import re
from typing import Iterable

_WORD_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"(?<![A-Za-z])(?:\d+[\d,./:-]*)")

# Negation patterns: sentences containing strong clinical negation without diagnosis
# terms likely describe the absence of findings, not ICD-codable information.
_NEGATION_RE = re.compile(
    r"\b(?:no\s+(?:evidence|sign|significant|acute|focal|definite|convincing|obvious)\s+(?:of|for)|"
    r"negative\s+for|"
    r"denies|"
    r"ruled\s+out|"
    r"without\s+(?:evidence|any)\s+(?:of|for)|"
    r"not\s+(?:seen|identified|present|appreciated|visualized|demonstrated)|"
    r"excluded|"
    r"unlikely|"
    r"absent)\b",
    re.IGNORECASE,
)

# Strong ICD-codable signals: words that override negation
_OVERRIDE_RE = re.compile(
    r"\b(?:diagnosis|history\s+of|continue|treatment|started|given|administered|"
    r"prescribed|diagnosed|admitted\s+for|confirmed|management|plan)\b",
    re.IGNORECASE,
)


def has_strong_negation(text: str) -> bool:
    """Check if sentence has strong clinical negation WITHOUT an override signal."""
    if not _NEGATION_RE.search(text):
        return False
    if _OVERRIDE_RE.search(text):
        return False
    return True


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r", "\n")
    text = text.strip().lower()
    text = _WORD_RE.sub(" ", text)
    return text


def normalize_for_vectorizer(text: str) -> str:
    text = normalize_text(text)
    return text


def truncate_words(text: str, max_words: int = 128) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def word_count(text: str) -> int:
    return len(text.split())


def deduplicate_texts(texts: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for text in texts:
        norm = normalize_text(text)
        if norm not in seen:
            seen.add(norm)
            out.append(text)
    return out
