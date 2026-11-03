"""Manual retrieval: BM25 keyword search over the product manuals.

Each `manuals/<stem>.md` file is split into chunks at its `##` headings.
Chunk ids look like ``aero/troubleshooting#beeping``:

* ``aero`` is the file stem (= the product),
* the heading ``Troubleshooting: Beeping`` becomes ``troubleshooting#beeping``
  (each part is slugified, and the ``:`` between "section" and "topic" becomes ``#``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

MANUALS_DIR = Path(__file__).resolve().parent.parent / "manuals"

# Extra words added to every chunk of a product so that queries like
# "my fan is beeping" match the Aero manual even if the chunk never says "fan".
PRODUCT_ALIASES: dict[str, str] = {
    "aero": "hawa aero fan ceiling pankha",
    "lock": "hawa lock s1 door smart lock",
    "pure": "hawa pure ro water purifier",
    "general": "hawa home policy",
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "to", "of", "and", "or", "in", "on", "for", "my", "me",
    "i", "it", "do", "does", "how", "what", "can", "you", "your", "with", "be", "this", "that",
    "at", "by", "from", "if", "when", "will", "should", "please", "hai", "ka", "ki", "ke", "kya",
    "kaise", "mera", "meri", "mujhe", "hota", "ho", "raha", "rahi", "se",
}


@dataclass
class Chunk:
    """One `##` section of a manual."""

    id: str
    product: str
    title: str
    text: str


def slugify(heading: str) -> str:
    """Turn a heading into an id slug. 'Troubleshooting: Beeping' -> 'troubleshooting#beeping'."""
    parts = []
    for part in heading.split(":"):
        slug = re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-")
        if slug:
            parts.append(slug)
    return "#".join(parts)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def load_chunks(manuals_dir: Path = MANUALS_DIR) -> list[Chunk]:
    """Read every manual and split it into chunks at `##` headings."""
    chunks: list[Chunk] = []
    for path in sorted(manuals_dir.glob("*.md")):
        product = path.stem
        title: str | None = None
        lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines() + ["## __end__"]:
            if line.startswith("## "):
                if title is not None:
                    chunks.append(Chunk(id=f"{product}/{slugify(title)}", product=product,
                                        title=title, text="\n".join(lines).strip()))
                title, lines = line[3:].strip(), []
            elif title is not None:
                lines.append(line)
    return chunks


class Retriever:
    """BM25 index over all manual chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(f"{PRODUCT_ALIASES.get(c.product, '')} {c.title} {c.title} {c.text}")
                  for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        """Return the top-k chunks for the query (chunks with score 0 are skipped)."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Build the index once and reuse it."""
    return Retriever(load_chunks())


def retrieve(query: str, k: int = 4) -> list[Chunk]:
    """Contract 2: top-k manual chunks for a user query."""
    return get_retriever().search(query, k)


def all_chunks() -> list[Chunk]:
    """All chunks (useful for evals and data checks)."""
    return list(get_retriever().chunks)
