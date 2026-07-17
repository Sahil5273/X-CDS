"""Deterministic verification of markdown citations against source passages."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from backend.app.llm.citations import (
    CITATION_PATTERN,
    SourceContext,
    extract_citation_indices,
)
from backend.app.search.tokenize import tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One deterministic citation or alignment failure."""

    code: str
    message: str
    citation_index: int | None = None
    claim: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """Outcome of algorithmic citation verification."""

    is_valid: bool
    issues: list[ValidationIssue]
    cited_indices: list[int]
    claim_alignments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "cited_indices": list(self.cited_indices),
            "claim_alignments": list(self.claim_alignments),
        }

    def feedback_message(self) -> str:
        """Render correction instructions for a regeneration attempt."""

        if self.is_valid:
            return ""
        lines = ["Citation validation failed. Fix every issue below:"]
        for issue in self.issues:
            prefix = f"[{issue.citation_index}] " if issue.citation_index else ""
            lines.append(f"- {prefix}{issue.message}")
        lines.append(
            "Rewrite the answer so every factual claim is grounded in the "
            "cited source passage and only uses valid citation numbers."
        )
        return "\n".join(lines)


def validate_citation_alignment(
    answer_markdown: str,
    contexts: list[SourceContext],
    *,
    min_token_overlap: float = 0.25,
    min_content_tokens: int = 2,
) -> CitationValidationResult:
    """Verify citation indices and claim-to-source string/token alignment."""

    issues: list[ValidationIssue] = []
    alignments: list[dict[str, Any]] = []

    answer = (answer_markdown or "").strip()
    if not answer:
        issues.append(
            ValidationIssue(
                code="empty_answer",
                message="Generated answer is empty.",
            )
        )
        return CitationValidationResult(
            is_valid=False,
            issues=issues,
            cited_indices=[],
            claim_alignments=alignments,
        )

    cited_indices = extract_citation_indices(answer)
    if not cited_indices:
        issues.append(
            ValidationIssue(
                code="missing_citations",
                message="Answer must include explicit markdown citations like [1].",
            )
        )

    by_index = {context.index: context for context in contexts}
    valid_indices = set(by_index)

    for index in cited_indices:
        if index not in valid_indices:
            issues.append(
                ValidationIssue(
                    code="unknown_citation",
                    message=f"Citation [{index}] does not map to a provided source.",
                    citation_index=index,
                )
            )

    claims = _extract_cited_claims(answer)
    if cited_indices and not claims:
        issues.append(
            ValidationIssue(
                code="unanchored_citations",
                message="Citations were found but no claim text could be aligned.",
            )
        )

    for claim_text, index in claims:
        context = by_index.get(index)
        if context is None:
            continue

        overlap = _token_overlap_ratio(claim_text, context.text)
        substring_hit = _has_significant_substring(claim_text, context.text)
        aligned = substring_hit or overlap >= min_token_overlap
        alignments.append(
            {
                "citation_index": index,
                "claim": claim_text,
                "chunk_id": context.chunk_id,
                "overlap": overlap,
                "substring_hit": substring_hit,
                "aligned": aligned,
            }
        )
        if not aligned:
            claim_tokens = _content_tokens(claim_text)
            if len(claim_tokens) < min_content_tokens and not substring_hit:
                # Very short claims still need at least one content token in source.
                if not set(claim_tokens).intersection(_content_tokens(context.text)):
                    issues.append(
                        ValidationIssue(
                            code="context_misalignment",
                            message=(
                                f"Claim cited as [{index}] is not supported by "
                                f"source passage {context.chunk_id}."
                            ),
                            citation_index=index,
                            claim=claim_text,
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        code="context_misalignment",
                        message=(
                            f"Claim cited as [{index}] is not supported by "
                            f"source passage {context.chunk_id}."
                        ),
                        citation_index=index,
                        claim=claim_text,
                    )
                )

    return CitationValidationResult(
        is_valid=not issues,
        issues=issues,
        cited_indices=cited_indices,
        claim_alignments=alignments,
    )


def _extract_cited_claims(answer_markdown: str) -> list[tuple[str, int]]:
    """Pair claim snippets with the citation index that closes them."""

    claims: list[tuple[str, int]] = []
    for sentence in _SENTENCE_SPLIT.split(answer_markdown):
        piece = sentence.strip()
        if not piece:
            continue
        matches = list(CITATION_PATTERN.finditer(piece))
        if not matches:
            continue
        cursor = 0
        for match in matches:
            claim = piece[cursor : match.start()].strip(" \t-*:;")
            claim = CITATION_PATTERN.sub("", claim).strip()
            if claim:
                claims.append((claim, int(match.group(1))))
            cursor = match.end()
    return claims


def _content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in _STOPWORDS and len(token) > 1]


def _token_overlap_ratio(claim: str, source: str) -> float:
    claim_tokens = set(_content_tokens(claim))
    if not claim_tokens:
        return 0.0
    source_tokens = set(_content_tokens(source))
    if not source_tokens:
        return 0.0
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def _has_significant_substring(claim: str, source: str) -> bool:
    """True when a multi-token claim fragment appears verbatim in the source."""

    normalized_claim = " ".join(tokenize(claim))
    normalized_source = " ".join(tokenize(source))
    if not normalized_claim or not normalized_source:
        return False
    if normalized_claim in normalized_source:
        return True

    claim_tokens = _content_tokens(claim)
    if len(claim_tokens) < 3:
        return False
    window = " ".join(claim_tokens[:3])
    return window in normalized_source
