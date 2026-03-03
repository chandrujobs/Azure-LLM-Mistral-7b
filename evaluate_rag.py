#!/usr/bin/env python3
"""
Evaluate retrieval quality for a JSONL chunk index produced by build_training_data.py.

Inputs:
1) rag_chunks.jsonl
2) eval_questions.jsonl

Question file format (JSONL, one object per line):
{
  "id": "q1",
  "question": "How do I create an Azure Data Factory pipeline?",
  "expected_keywords": ["pipeline", "activity", "trigger"],
  "expected_sources": ["azure-data-factory.pdf"],
  "required_facts": ["Create a pipeline", "Add activities", "Create a trigger"],
  "forbidden_facts": ["Use Azure ML Studio to schedule ADF pipelines"],
  "reference_answer": "Create pipeline, add activities, add trigger, publish."
}

All fields except "question" are optional but strongly recommended for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    metadata: dict
    token_counts: Counter
    length: int


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_jsonl(path: Path) -> Iterable[dict]:
    # utf-8-sig handles files with or without BOM.
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc


def load_chunks(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for obj in load_jsonl(path):
        text = str(obj.get("text", "")).strip()
        if not text:
            continue
        chunk_id = str(obj.get("chunk_id", ""))
        metadata = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}
        source_file = str(metadata.get("source_file", "unknown"))
        tokens = tokenize(text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                source_file=source_file,
                metadata=metadata,
                token_counts=Counter(tokens),
                length=max(1, len(tokens)),
            )
        )
    return chunks


def build_idf(chunks: List[Chunk]) -> Dict[str, float]:
    doc_freq: Dict[str, int] = defaultdict(int)
    n_docs = len(chunks)
    for c in chunks:
        for token in c.token_counts.keys():
            doc_freq[token] += 1
    # Smoothed IDF
    return {t: math.log((1 + n_docs) / (1 + df)) + 1.0 for t, df in doc_freq.items()}


def score_query(query: str, chunks: List[Chunk], idf: Dict[str, float]) -> List[Tuple[float, Chunk]]:
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    q_counts = Counter(q_tokens)
    scored: List[Tuple[float, Chunk]] = []
    for c in chunks:
        score = 0.0
        for token, q_tf in q_counts.items():
            c_tf = c.token_counts.get(token, 0)
            if c_tf == 0:
                continue
            score += (q_tf * c_tf / c.length) * idf.get(token, 1.0)
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def normalize_source_name(s: str) -> str:
    return s.strip().lower()


def keyword_coverage(expected_keywords: List[str], top_chunks: List[Chunk]) -> float:
    if not expected_keywords:
        return float("nan")
    corpus = " ".join(c.text.lower() for c in top_chunks)
    hits = 0
    for kw in expected_keywords:
        if kw and kw.lower() in corpus:
            hits += 1
    return hits / len(expected_keywords)


def source_hit_and_rank(expected_sources: List[str], top_chunks: List[Chunk]) -> Tuple[int, int]:
    if not expected_sources:
        return 0, 0
    expected = {normalize_source_name(s) for s in expected_sources if s}
    for i, chunk in enumerate(top_chunks, start=1):
        if normalize_source_name(chunk.source_file) in expected:
            return 1, i
    return 0, 0


def phrase_matches(
    phrases: List[str],
    top_chunks: List[Chunk],
    *,
    allow_fuzzy: bool = False,
    min_overlap_ratio: float = 0.67,
    min_overlap_tokens: int = 2,
) -> List[str]:
    corpus = " ".join(c.text.lower() for c in top_chunks)
    corpus_tokens = set(tokenize(corpus))
    matched: List[str] = []
    for phrase in phrases:
        p = phrase.strip().lower()
        if not p:
            continue
        if p in corpus:
            matched.append(phrase)
            continue
        if not allow_fuzzy:
            continue

        phrase_tokens = tokenize(p)
        if not phrase_tokens:
            continue
        phrase_token_set = set(phrase_tokens)
        overlap = len(phrase_token_set & corpus_tokens)
        ratio = overlap / len(phrase_token_set)
        if overlap >= min_overlap_tokens and ratio >= min_overlap_ratio:
            matched.append(phrase)
    return matched


def phrase_coverage(
    expected_phrases: List[str],
    top_chunks: List[Chunk],
    *,
    allow_fuzzy: bool = False,
    min_overlap_ratio: float = 0.67,
    min_overlap_tokens: int = 2,
) -> float:
    if not expected_phrases:
        return float("nan")
    matched = phrase_matches(
        expected_phrases,
        top_chunks,
        allow_fuzzy=allow_fuzzy,
        min_overlap_ratio=min_overlap_ratio,
        min_overlap_tokens=min_overlap_tokens,
    )
    return len(matched) / len(expected_phrases)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate retrieval quality against labeled questions.")
    p.add_argument("--rag-file", default="output/rag_chunks.jsonl", help="Path to RAG chunk JSONL.")
    p.add_argument("--questions-file", default="eval_questions.jsonl", help="Path to labeled eval questions JSONL.")
    p.add_argument(
        "--source-filter",
        action="append",
        default=[],
        help=(
            "Evaluate only questions whose expected_sources include one of these source files. "
            "Can be passed multiple times and/or comma-separated."
        ),
    )
    p.add_argument("--top-k", type=int, default=5, help="Evaluate top-k retrieved chunks.")
    p.add_argument("--out-json", default="output/retrieval_eval_summary.json", help="Summary output JSON path.")
    p.add_argument(
        "--out-details-jsonl",
        default="output/retrieval_eval_details.jsonl",
        help="Per-question details output JSONL path.",
    )
    p.add_argument("--min-source-hit-rate", type=float, default=None, help="Fail if source_hit_rate_at_k is below this value.")
    p.add_argument("--min-mrr", type=float, default=None, help="Fail if mrr_at_k is below this value.")
    p.add_argument(
        "--min-keyword-coverage",
        type=float,
        default=None,
        help="Fail if avg_keyword_coverage_at_k is below this value.",
    )
    p.add_argument(
        "--min-required-fact-coverage",
        type=float,
        default=None,
        help="Fail if avg_required_fact_coverage_at_k is below this value.",
    )
    p.add_argument(
        "--max-forbidden-fact-hit-rate",
        type=float,
        default=None,
        help="Fail if forbidden_fact_hit_rate_at_k is above this value.",
    )
    return p.parse_args()


def parse_source_filters(values: List[str]) -> set[str]:
    parsed: set[str] = set()
    for v in values:
        if not v:
            continue
        for part in v.split(","):
            name = normalize_source_name(part)
            if name:
                parsed.add(name)
    return parsed


def main() -> int:
    args = parse_args()
    rag_path = Path(args.rag_file)
    questions_path = Path(args.questions_file)
    out_json = Path(args.out_json)
    out_details = Path(args.out_details_jsonl)

    if not rag_path.exists():
        print(f"RAG file not found: {rag_path}", file=sys.stderr)
        return 1
    if not questions_path.exists():
        print(f"Questions file not found: {questions_path}", file=sys.stderr)
        return 1

    chunks = load_chunks(rag_path)
    if not chunks:
        print("No chunks loaded from RAG file.", file=sys.stderr)
        return 1

    questions = list(load_jsonl(questions_path))
    if not questions:
        print("No questions loaded.", file=sys.stderr)
        return 1
    source_filters = parse_source_filters(args.source_filter)
    if source_filters:
        filtered_questions: List[dict] = []
        for q in questions:
            expected_sources = q.get("expected_sources", [])
            if not isinstance(expected_sources, list):
                continue
            normalized_expected = {normalize_source_name(s) for s in expected_sources if s}
            if normalized_expected & source_filters:
                filtered_questions.append(q)
        questions = filtered_questions
        if not questions:
            print(
                f"No questions matched --source-filter: {sorted(source_filters)}",
                file=sys.stderr,
            )
            return 1

    idf = build_idf(chunks)

    source_hits = 0
    source_labeled = 0
    mrr_sum = 0.0
    keyword_sum = 0.0
    keyword_labeled = 0
    required_fact_sum = 0.0
    required_fact_labeled = 0
    forbidden_fact_total = 0
    forbidden_fact_hits_total = 0
    forbidden_fact_labeled = 0
    forbidden_questions_with_hits = 0
    per_source: Dict[str, dict] = defaultdict(
        lambda: {
            "questions": 0,
            "source_labeled": 0,
            "source_hits": 0,
            "mrr_sum": 0.0,
            "keyword_labeled": 0,
            "keyword_sum": 0.0,
            "required_fact_labeled": 0,
            "required_fact_sum": 0.0,
            "forbidden_fact_labeled": 0,
            "forbidden_fact_total": 0,
            "forbidden_fact_hits_total": 0,
            "forbidden_questions_with_hits": 0,
        }
    )
    details: List[dict] = []

    for i, q in enumerate(questions, start=1):
        qid = str(q.get("id", f"q{i}"))
        question = str(q.get("question", "")).strip()
        expected_sources = q.get("expected_sources", [])
        expected_keywords = q.get("expected_keywords", [])
        required_facts = q.get("required_facts", [])
        forbidden_facts = q.get("forbidden_facts", [])
        reference_answer = q.get("reference_answer")
        if not question:
            continue
        if not isinstance(expected_sources, list):
            expected_sources = []
        if not isinstance(expected_keywords, list):
            expected_keywords = []
        if not isinstance(required_facts, list):
            required_facts = []
        if not isinstance(forbidden_facts, list):
            forbidden_facts = []
        primary_source = normalize_source_name(expected_sources[0]) if expected_sources else "unlabeled"
        per_source[primary_source]["questions"] += 1

        ranked = score_query(question, chunks, idf)
        top = [c for _, c in ranked[: args.top_k]]
        top_scores = [s for s, _ in ranked[: args.top_k]]

        hit, rank = source_hit_and_rank(expected_sources, top)
        if expected_sources:
            source_labeled += 1
            source_hits += hit
            per_source[primary_source]["source_labeled"] += 1
            per_source[primary_source]["source_hits"] += hit
            if rank > 0:
                mrr_sum += 1.0 / rank
                per_source[primary_source]["mrr_sum"] += 1.0 / rank

        kw_cov = keyword_coverage(expected_keywords, top)
        if not math.isnan(kw_cov):
            keyword_labeled += 1
            keyword_sum += kw_cov
            per_source[primary_source]["keyword_labeled"] += 1
            per_source[primary_source]["keyword_sum"] += kw_cov

        req_cov = phrase_coverage(required_facts, top, allow_fuzzy=True)
        if not math.isnan(req_cov):
            required_fact_labeled += 1
            required_fact_sum += req_cov
            per_source[primary_source]["required_fact_labeled"] += 1
            per_source[primary_source]["required_fact_sum"] += req_cov
        matched_required_facts = phrase_matches(required_facts, top, allow_fuzzy=True)

        # Keep forbidden-fact matching strict to reduce false positives.
        matched_forbidden_facts = phrase_matches(forbidden_facts, top, allow_fuzzy=False)
        if forbidden_facts:
            forbidden_fact_labeled += 1
            forbidden_fact_total += len(forbidden_facts)
            forbidden_fact_hits_total += len(matched_forbidden_facts)
            per_source[primary_source]["forbidden_fact_labeled"] += 1
            per_source[primary_source]["forbidden_fact_total"] += len(forbidden_facts)
            per_source[primary_source]["forbidden_fact_hits_total"] += len(matched_forbidden_facts)
            if matched_forbidden_facts:
                forbidden_questions_with_hits += 1
                per_source[primary_source]["forbidden_questions_with_hits"] += 1

        details.append(
            {
                "id": qid,
                "question": question,
                "expected_sources": expected_sources,
                "expected_keywords": expected_keywords,
                "required_facts": required_facts,
                "forbidden_facts": forbidden_facts,
                "reference_answer": reference_answer,
                "source_hit_at_k": bool(hit),
                "first_source_rank": rank if rank > 0 else None,
                "keyword_coverage": None if math.isnan(kw_cov) else round(kw_cov, 4),
                "required_fact_coverage": None if math.isnan(req_cov) else round(req_cov, 4),
                "matched_required_facts": matched_required_facts,
                "matched_forbidden_facts": matched_forbidden_facts,
                "retrieved": [
                    {
                        "rank": r,
                        "score": round(score, 6),
                        "chunk_id": chunk.chunk_id,
                        "source_file": chunk.source_file,
                        "page_start": chunk.metadata.get("page_start"),
                        "page_end": chunk.metadata.get("page_end"),
                    }
                    for r, (score, chunk) in enumerate(zip(top_scores, top), start=1)
                ],
            }
        )

    summary = {
        "rag_file": str(rag_path),
        "questions_file": str(questions_path),
        "source_filter": sorted(source_filters) if source_filters else None,
        "questions_total": len(questions),
        "top_k": args.top_k,
        "source_hit_rate_at_k": round(source_hits / source_labeled, 4) if source_labeled else None,
        "mrr_at_k": round(mrr_sum / source_labeled, 4) if source_labeled else None,
        "avg_keyword_coverage_at_k": round(keyword_sum / keyword_labeled, 4) if keyword_labeled else None,
        "avg_required_fact_coverage_at_k": (
            round(required_fact_sum / required_fact_labeled, 4) if required_fact_labeled else None
        ),
        "forbidden_fact_hit_rate_at_k": (
            round(forbidden_fact_hits_total / forbidden_fact_total, 4) if forbidden_fact_total else None
        ),
        "forbidden_question_hit_rate_at_k": (
            round(forbidden_questions_with_hits / forbidden_fact_labeled, 4) if forbidden_fact_labeled else None
        ),
        "labels_used": {
            "source_labeled_questions": source_labeled,
            "keyword_labeled_questions": keyword_labeled,
            "required_fact_labeled_questions": required_fact_labeled,
            "forbidden_fact_labeled_questions": forbidden_fact_labeled,
        },
        "per_source": {},
    }
    for source_name, stats in sorted(per_source.items()):
        source_labeled_q = stats["source_labeled"]
        keyword_labeled_q = stats["keyword_labeled"]
        required_labeled_q = stats["required_fact_labeled"]
        forbidden_labeled_q = stats["forbidden_fact_labeled"]
        forbidden_total = stats["forbidden_fact_total"]
        summary["per_source"][source_name] = {
            "questions": stats["questions"],
            "source_hit_rate_at_k": (
                round(stats["source_hits"] / source_labeled_q, 4) if source_labeled_q else None
            ),
            "mrr_at_k": round(stats["mrr_sum"] / source_labeled_q, 4) if source_labeled_q else None,
            "avg_keyword_coverage_at_k": (
                round(stats["keyword_sum"] / keyword_labeled_q, 4) if keyword_labeled_q else None
            ),
            "avg_required_fact_coverage_at_k": (
                round(stats["required_fact_sum"] / required_labeled_q, 4) if required_labeled_q else None
            ),
            "forbidden_fact_hit_rate_at_k": (
                round(stats["forbidden_fact_hits_total"] / forbidden_total, 4) if forbidden_total else None
            ),
            "forbidden_question_hit_rate_at_k": (
                round(stats["forbidden_questions_with_hits"] / forbidden_labeled_q, 4)
                if forbidden_labeled_q
                else None
            ),
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_details.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with out_details.open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    gate_failures: List[str] = []
    if args.min_source_hit_rate is not None:
        actual = summary.get("source_hit_rate_at_k")
        if actual is None or actual < args.min_source_hit_rate:
            gate_failures.append(f"source_hit_rate_at_k={actual} < min_source_hit_rate={args.min_source_hit_rate}")
    if args.min_mrr is not None:
        actual = summary.get("mrr_at_k")
        if actual is None or actual < args.min_mrr:
            gate_failures.append(f"mrr_at_k={actual} < min_mrr={args.min_mrr}")
    if args.min_keyword_coverage is not None:
        actual = summary.get("avg_keyword_coverage_at_k")
        if actual is None or actual < args.min_keyword_coverage:
            gate_failures.append(
                f"avg_keyword_coverage_at_k={actual} < min_keyword_coverage={args.min_keyword_coverage}"
            )
    if args.min_required_fact_coverage is not None:
        actual = summary.get("avg_required_fact_coverage_at_k")
        if actual is None or actual < args.min_required_fact_coverage:
            gate_failures.append(
                "avg_required_fact_coverage_at_k="
                f"{actual} < min_required_fact_coverage={args.min_required_fact_coverage}"
            )
    if args.max_forbidden_fact_hit_rate is not None:
        actual = summary.get("forbidden_fact_hit_rate_at_k")
        if actual is None or actual > args.max_forbidden_fact_hit_rate:
            gate_failures.append(
                "forbidden_fact_hit_rate_at_k="
                f"{actual} > max_forbidden_fact_hit_rate={args.max_forbidden_fact_hit_rate}"
            )

    print(json.dumps(summary, indent=2))
    if gate_failures:
        print("QUALITY GATE FAILURE:", file=sys.stderr)
        for failure in gate_failures:
            print(f"- {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
