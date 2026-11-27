"""
Retrieval over the hostel knowledge base.

The knowledge base is a folder of Markdown files. This module splits them into
heading-scoped sections, indexes the sections with BM25 (a standard lexical
ranking function), and returns the best matching sections for a question
together with their source file and heading.

BM25 is a pure-Python, offline algorithm, so retrieval needs no network call and
no heavyweight dependency. That keeps the app fast, cheap and reproducible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
MAX_CHUNK_CHARS = 1400
MIN_CHUNK_CHARS = 40
BM25_K1 = 1.5
BM25_B = 0.75

# Very common English words plus a few generic hostel words that carry little
# signal on their own. Removing them stops short questions being matched by
# filler words instead of the actual topic.
_STOPWORDS = set(
    """a an the and or of to in on at for with is are was were be been being do does did
    i you he she it we they me my your our their this that these those what which who whom
    how when where why can could should would will shall may might must if then than as by
    from about into over under again further once here there all any both each few more most
    other some such no nor not only own same so too very s t don now
    time timings timing please want tell know give hello hi thanks thank help""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with hyphens squashed so 'Wi-Fi' matches 'wifi'."""
    lowered = (text or "").lower()
    words = _TOKEN_RE.findall(lowered)
    squashed = _TOKEN_RE.findall(lowered.replace("-", ""))
    return [w for w in words + squashed if w not in _STOPWORDS and len(w) > 1]


@dataclass
class Chunk:
    """One retrievable section of a knowledge base document."""

    source: str
    heading: str
    text: str
    score: float = field(default=0.0, compare=False)

    @property
    def snippet(self) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= 260 else text[:257].rstrip() + "..."


def _split_into_chunks(source: str, raw: str) -> list[Chunk]:
    """Split one Markdown document into heading-scoped chunks."""
    chunks: list[Chunk] = []
    current_heading = source
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if len(body) < MIN_CHUNK_CHARS:
            return
        # A section that is very long is split into paragraph-sized pieces so a
        # single huge section does not dominate retrieval.
        if len(body) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(source=source, heading=current_heading, text=body))
            return
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        piece = ""
        for para in paragraphs:
            if piece and len(piece) + len(para) > MAX_CHUNK_CHARS:
                chunks.append(Chunk(source=source, heading=current_heading, text=piece))
                piece = para
            else:
                piece = f"{piece}\n\n{para}".strip() if piece else para
        if piece and len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(source=source, heading=current_heading, text=piece))

    for line in raw.splitlines():
        if line.startswith("#"):
            flush()
            buffer = []
            current_heading = line.lstrip("#").strip() or current_heading
        else:
            buffer.append(line)
    flush()
    return chunks


def load_chunks(kb_dir: Path | str = DEFAULT_KB_DIR) -> list[Chunk]:
    """Load and chunk every Markdown file in the knowledge base folder."""
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_path}")

    chunks: list[Chunk] = []
    for path in sorted(kb_path.glob("*.md")):
        if path.name.upper().startswith("SOURCES"):
            continue
        raw = path.read_text(encoding="utf-8")
        chunks.extend(_split_into_chunks(path.name, raw))
    if not chunks:
        raise ValueError(f"No usable content found in knowledge base: {kb_path}")
    return chunks


class Retriever:
    """BM25 retriever over the knowledge base chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = BM25_K1, b: float = BM25_B):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk.")
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        # The heading and file name are indexed alongside the body: they are
        # short but very informative about what a section is about.
        self._doc_tokens = [
            _tokenize(f"{c.heading} {c.source} {c.text}") for c in chunks
        ]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avg_len = sum(self._doc_len) / len(self._doc_tokens) or 1.0
        self._term_freq = [Counter(t) for t in self._doc_tokens]

        doc_freq: Counter[str] = Counter()
        for counter in self._term_freq:
            doc_freq.update(counter.keys())
        total = len(self.chunks)
        self._idf = {
            term: math.log(1 + (total - n + 0.5) / (n + 0.5))
            for term, n in doc_freq.items()
        }

    def _score(self, tokenized_query: list[str]) -> list[float]:
        scores = [0.0] * len(self.chunks)
        for term in set(tokenized_query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self._term_freq):
                freq = freqs.get(term, 0)
                if freq == 0:
                    continue
                norm = 1 - self.b + self.b * self._doc_len[i] / self._avg_len
                scores[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
        return scores

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[Chunk]:
        """Return the top-k chunks for a query, best first."""
        tokenized = _tokenize(query)
        if not tokenized:
            return []
        scores = self._score(tokenized)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)

        results: list[Chunk] = []
        for i in ranked:
            if len(results) >= max(k, 1):
                break
            if scores[i] <= min_score:
                break
            chunk = self.chunks[i]
            results.append(
                Chunk(
                    source=chunk.source,
                    heading=chunk.heading,
                    text=chunk.text,
                    score=float(scores[i]),
                )
            )
        return results


def build_retriever(kb_dir: Path | str = DEFAULT_KB_DIR) -> Retriever:
    """Convenience helper: load the knowledge base and build a retriever."""
    return Retriever(load_chunks(kb_dir))
