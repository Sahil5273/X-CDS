"""Fetch and normalize PubMed Central articles represented as BioC JSON."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BIOC_API_URL = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
    "pmcoa.cgi/BioC_json/{pmcid}/unicode"
)
MAX_REQUESTS_PER_SECOND = 3.0


class BioCError(RuntimeError):
    """Raised when BioC data cannot be fetched or parsed."""


@dataclass(frozen=True, slots=True)
class BiomedicalChunk:
    """A passage-level text unit normalized from a BioC document."""

    chunk_id: str
    pmcid: str
    text: str
    section: str
    passage_type: str
    offset: int
    source_url: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable representation."""

        return asdict(self)


class RequestRateLimiter:
    """Thread-safe limiter that spaces requests at a fixed minimum interval."""

    def __init__(self, requests_per_second: float = MAX_REQUESTS_PER_SECOND) -> None:
        if not 0 < requests_per_second <= MAX_REQUESTS_PER_SECOND:
            raise ValueError(
                f"requests_per_second must be greater than 0 and at most "
                f"{MAX_REQUESTS_PER_SECOND:g}"
            )
        self._minimum_interval = 1.0 / requests_per_second
        self._last_request_at: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until issuing another request complies with the rate limit."""

        with self._lock:
            now = time.monotonic()
            if self._last_request_at is not None:
                delay = self._minimum_interval - (now - self._last_request_at)
                if delay > 0:
                    time.sleep(delay)
            self._last_request_at = time.monotonic()


class BioCClient:
    """Small BioC API client capped at NCBI's three-request-per-second limit."""

    def __init__(
        self,
        requests_per_second: float = MAX_REQUESTS_PER_SECOND,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        self._rate_limiter = RequestRateLimiter(requests_per_second)
        self._timeout_seconds = timeout_seconds

    def fetch(self, pmcid: str) -> Any:
        """Fetch one PMC article and return its decoded BioC JSON payload."""

        normalized_pmcid = _normalize_pmcid(pmcid)
        url = BIOC_API_URL.format(pmcid=normalized_pmcid)
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "X-CDS/1.0 (clinical decision support research)",
            },
        )
        self._rate_limiter.wait()

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                return json.load(response)
        except HTTPError as exc:
            raise BioCError(
                f"BioC API returned HTTP {exc.code} for {normalized_pmcid}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise BioCError(
                f"BioC API request failed for {normalized_pmcid}: {exc}"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BioCError(
                f"BioC API returned invalid JSON for {normalized_pmcid}"
            ) from exc

    def fetch_chunks(self, pmcid: str) -> list[BiomedicalChunk]:
        """Fetch one PMC article and normalize its passages."""

        return parse_bioc_json(self.fetch(pmcid))


def parse_bioc_json(payload: Any) -> list[BiomedicalChunk]:
    """Normalize BioC collection, document, or collection-list JSON shapes."""

    documents = list(_iter_documents(payload))
    if not documents:
        raise BioCError("BioC payload contains no documents")

    chunks: list[BiomedicalChunk] = []
    for document in documents:
        pmcid = _normalize_pmcid(str(document.get("id", "")))
        passages = document.get("passages", [])
        if not _is_sequence(passages):
            raise BioCError(f"BioC document {pmcid} has invalid passages")

        for passage_index, passage in enumerate(passages):
            if not isinstance(passage, Mapping):
                continue
            text = str(passage.get("text", "")).strip()
            if not text:
                continue

            infons = passage.get("infons", {})
            if not isinstance(infons, Mapping):
                infons = {}
            offset = _parse_offset(passage.get("offset", 0), pmcid)
            chunks.append(
                BiomedicalChunk(
                    chunk_id=f"{pmcid}:passage:{passage_index}",
                    pmcid=pmcid,
                    text=text,
                    section=str(
                        infons.get("section")
                        or infons.get("title")
                        or infons.get("type")
                        or "unknown"
                    ),
                    passage_type=str(infons.get("type") or "unknown"),
                    offset=offset,
                    source_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                )
            )

    return chunks


def _iter_documents(payload: Any):
    if isinstance(payload, Mapping):
        documents = payload.get("documents")
        if _is_sequence(documents):
            yield from (
                document for document in documents if isinstance(document, Mapping)
            )
            return
        if "id" in payload and "passages" in payload:
            yield payload
            return

    if _is_sequence(payload):
        for item in payload:
            yield from _iter_documents(item)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _normalize_pmcid(pmcid: str) -> str:
    normalized = pmcid.strip().upper()
    if not normalized:
        raise BioCError("PMC identifier cannot be empty")
    if not normalized.startswith("PMC"):
        normalized = f"PMC{normalized}"
    if not normalized[3:].isdigit():
        raise BioCError(f"Invalid PMC identifier: {pmcid!r}")
    return normalized


def _parse_offset(value: Any, pmcid: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BioCError(f"BioC document {pmcid} contains an invalid offset") from exc
