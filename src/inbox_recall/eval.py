"""Retrieval benchmark on the fabricated demo archive.

Each case is a search_messages call an agent might make, with the demo
messages that answer it. We report Recall@5, Recall@10 and MRR@10 per search
mode and per category:

- keyword: the query uses the message's own words
- paraphrase: same meaning, different words (the case embeddings exist for)
- filtered: people, chats, networks and plain-English time ranges narrow it down
- fuzzy: typos and partial names
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg

from .demo import message_ids
from .embedders import Embedder
from .query import SearchRequest
from .search import Searcher


@dataclass
class Case:
    category: str
    query: str
    relevant: list[str]
    filters: dict[str, Any] = field(default_factory=dict)


CASES = [
    Case("keyword", "timing belt", ["im-sam:car-1"]),
    Case("keyword", "sweet potato casserole", ["wa-family:tg-1"]),
    Case("keyword", "Glass Orchard", ["sig-book:club-1"]),
    Case("keyword", "logrotate", ["dc-lena:disk-3"]),
    Case("keyword", "Manteigaria pastel de nata", ["wa-maya:lisbon-9"]),
    Case("keyword", "ceramic planter", ["im-sam:gift-2"]),
    Case("paraphrase", "how much did the car repair cost", ["im-sam:car-1"]),
    Case("paraphrase", "where are we staying in Portugal", ["wa-maya:lisbon-5", "wa-maya:lisbon-7"]),
    Case("paraphrase", "best pastry shop in Lisbon", ["wa-maya:lisbon-9", "wa-maya:lisbon-12"]),
    Case("paraphrase", "how much more will the apartment cost after we renew", ["sig-priya:lease-1"]),
    Case("paraphrase", "recommendation for getting my teeth cleaned", ["tg-theo:dentist-1"]),
    Case("paraphrase", "the test environment crashed", ["dc-lena:disk-1", "dc-lena:disk-2"]),
    Case("paraphrase", "what novel did someone tell me to read", ["tg-theo:rec-1", "sig-book:club-1"]),
    Case(
        "filtered",
        "lisbon flights",
        ["wa-maya:lisbon-1", "wa-maya:lisbon-3", "wa-maya:lisbon-4"],
        {"sender": ["Maya"], "when": "last spring"},
    ),
    Case("filtered", "lisbon", ["tg-theo:cousin-1"], {"person": ["Theo"]}),
    Case("filtered", "flight", ["wa-family:tg-2"], {"chat": ["Lee Family"]}),
    Case("filtered", "pay you back", ["ig-kai:show-2"], {"sender": ["me"]}),
    Case("filtered", "tickets", ["ig-kai:show-1", "ig-kai:show-3"], {"network": ["instagram"], "when": "last summer"}),
    Case(
        "filtered", "class", ["ig-maya:pottery-1", "ig-maya:pottery-3"], {"sender": ["Maya"], "network": ["instagram"]}
    ),
    Case("filtered", "kitchen leak", ["sig-priya:leak-1", "sig-priya:leak-2"], {"when": "past 6 months"}),
    Case("fuzzy", "manteigria", ["wa-maya:lisbon-9"]),
    Case("fuzzy", "alfma balcony", ["wa-maya:lisbon-5"]),
    Case("fuzzy", "Lantern Hal", ["ig-kai:show-1"]),
    Case("fuzzy", "okafor dentst", ["tg-theo:dentist-1"]),
]


def _metrics(ranked: list[int], relevant: set[int]) -> dict[str, float]:
    def recall(k: int) -> float:
        return len(relevant & set(ranked[:k])) / len(relevant)

    rr = next((1.0 / (i + 1) for i, mid in enumerate(ranked[:10]) if mid in relevant), 0.0)
    return {"recall@5": recall(5), "recall@10": recall(10), "mrr@10": rr}


def run(
    conn: psycopg.Connection, embedder: Embedder | None, now: datetime, modes: tuple[str, ...] | None = None
) -> dict[str, Any]:
    ids = message_ids(conn, sorted({key for case in CASES for key in case.relevant}))
    missing = sorted({key for case in CASES for key in case.relevant} - set(ids))
    if missing:
        raise RuntimeError(f"demo archive is missing {len(missing)} benchmark messages; run `inbox-recall demo` first")
    searcher = Searcher(conn, embedder, tz=now.tzinfo, now=now)
    modes = modes or (("keyword", "semantic", "hybrid") if embedder else ("keyword",))
    per_mode: dict[str, dict[str, list[dict[str, float]]]] = {m: defaultdict(list) for m in modes}
    misses: list[dict[str, Any]] = []
    for case in CASES:
        relevant = {ids[key] for key in case.relevant}
        for mode in modes:
            req = SearchRequest(query=case.query, mode=mode, limit=10, context=0, **case.filters)
            result = searcher.search(req)
            ranked = [hit["id"] for hit in result["hits"]]
            scores = _metrics(ranked, relevant)
            per_mode[mode][case.category].append(scores)
            per_mode[mode]["all"].append(scores)
            if mode == modes[-1] and scores["recall@10"] < 1:
                misses.append({"query": case.query, "category": case.category, "recall@10": scores["recall@10"]})

    def mean(rows: list[dict[str, float]]) -> dict[str, float]:
        return {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in rows[0]}

    return {
        "model": embedder.name if embedder else None,
        "cases": len(CASES),
        "results": {mode: {cat: mean(rows) for cat, rows in cats.items()} for mode, cats in per_mode.items()},
        "misses": misses,
    }


def table(report: dict[str, Any]) -> str:
    categories = ["all", "keyword", "paraphrase", "filtered", "fuzzy"]
    lines = [
        f"Benchmark: {report['cases']} queries, embeddings: {report['model'] or 'none'}",
        "",
        "| mode | " + " | ".join(f"{c} R@10" for c in categories) + " | all R@5 | all MRR@10 |",
        "| --- |" + " ---: |" * (len(categories) + 2),
    ]
    for mode, cats in report["results"].items():
        cells = [f"{cats[c]['recall@10']:.2f}" if c in cats else "-" for c in categories]
        lines.append(
            f"| {mode} | " + " | ".join(cells) + f" | {cats['all']['recall@5']:.2f} | {cats['all']['mrr@10']:.2f} |"
        )
    return "\n".join(lines)
