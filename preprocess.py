"""Reusable, stateless IMDB text preparation, also usable on unlabeled input."""

import re
import unicodedata
from html.parser import HTMLParser
from numbers import Real

import pandas as pd

from load_data import LABELS


class _VisibleText(HTMLParser):
    """Extract text, separating tags and ignoring script/style contents."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        self.parts.append(" ")
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        self.parts.append(" ")
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def preprocess_text(text: object) -> str:
    """Normalize one review without needing labels or fitting corpus statistics.

    Non-string inputs, unparseable markup, and strings without letters/digits return "".
    Callers must reject that sentinel instead of classifying an empty input.
    Tags become spaces, HTML entities are decoded, and script/style content is
    removed. NFKC, lowercase, apostrophe normalization, control-character removal,
    and whitespace collapse follow. Keep negation, numbers, and punctuation;
    no stemming, stopword removal, or learned vocabulary is used.
    """
    if not isinstance(text, str):
        return ""
    parser = _VisibleText()
    try:
        parser.feed(text)
        parser.close()
    except (AssertionError, ValueError):
        # HTMLParser tolerates most broken tags, but rejects some declarations.
        return ""
    result = unicodedata.normalize("NFKC", "".join(parser.parts)).lower()
    result = result.translate(str.maketrans({"’": "'", "‘": "'"}))
    result = "".join(" " if unicodedata.category(char).startswith("C") else char
                     for char in result)
    result = re.sub(r"\s+", " ", result).strip()
    return result if any(char.isalnum() for char in result) else ""


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prepared, rejected) rows without mutating input or changing indexes.

    The prepared `text` replaces raw text; all other columns stay aligned. Labels
    retain IMDB's existing meaning and use int64. Rejected rows contain original
    values and a `rejection_reason` (text problems take precedence over labels).
    The strict Part 1 loader still validates the downloaded dataset; this boundary
    also supports imperfect frames supplied directly by subsequent callers.
    """
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    cleaned = frame["text"].map(preprocess_text)
    valid_labels = frame["label"].map(
        lambda value: isinstance(value, Real) and not isinstance(value, bool)
        and value in LABELS
    )
    reasons = []
    for raw, text, valid_label in zip(frame["text"], cleaned, valid_labels):
        if not isinstance(raw, str):
            reasons.append("non_string_text")
        elif not raw.strip():
            reasons.append("blank_text")
        elif not text:
            reasons.append("empty_after_preprocessing")
        elif not valid_label:
            reasons.append("invalid_label")
        else:
            reasons.append("")
    keep = [not reason for reason in reasons]
    prepared = frame.loc[keep].copy()
    prepared["text"] = cleaned.loc[keep]
    prepared["label"] = prepared["label"].astype("int64")
    rejected = frame.loc[[bool(reason) for reason in reasons]].copy()
    rejected["rejection_reason"] = [reason for reason in reasons if reason]
    return prepared, rejected
