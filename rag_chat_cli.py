#!/usr/bin/env python3
"""
Query Chroma-indexed documentation with strict grounded response behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

FALLBACK = "This information is not found in the indexed official Microsoft documentation."
SERVICE_HINTS = {
    "azure data factory": "Azure Data Factory",
    "data factory": "Azure Data Factory",
    "azure blob": "Azure Blob Storage",
    "blob storage": "Azure Blob Storage",
    "azure storage": "Azure Blob Storage",
    "azure databricks": "Azure Databricks",
    "databricks": "Azure Databricks",
    "azure synapse": "Azure Synapse Analytics",
    "synapse": "Azure Synapse Analytics",
    "azure devops": "Azure DevOps",
    "devops": "Azure DevOps",
    "power bi": "Power BI",
    "fabric": "Microsoft Fabric",
    "purview": "Microsoft Purview",
    "microsoft purview": "Microsoft Purview",
    "adf": "Azure Data Factory",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
# Accept both inline citations like [source_file p12-13] and compact source-list markers like [p12-13].
CITATION_RE = re.compile(r"\[(?:[^\]]+\s+p\d+(?:-\d+)?|p\d+(?:-\d+)?)\]", flags=re.IGNORECASE)
HEADING_RE = re.compile(r"^###\s+(.+?)\s*$")
CANONICAL_SECTIONS = [
    "Overview",
    "Architecture Explanation",
    "Implementation Steps",
    "Best Practices",
    "Security & Governance Considerations",
    "Performance Considerations",
    "Reference",
]


def env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG query CLI for Azure documentation index.")
    p.add_argument("--question", default="", help="Single question. If omitted, starts interactive mode.")
    p.add_argument("--persist-dir", default="output/chroma_db", help="Chroma persistence folder.")
    p.add_argument("--collection", default="azure_docs", help="Chroma collection name.")
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers embedding model.",
    )
    p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    p.add_argument(
        "--retrieval-mode",
        choices=["similarity", "mmr"],
        default="mmr",
        help="Retrieval strategy. MMR improves diversity and reduces duplicate intro chunks.",
    )
    p.add_argument(
        "--fetch-k",
        type=int,
        default=40,
        help="Candidate pool size for MMR retrieval.",
    )
    p.add_argument("--min-context-chars", type=int, default=120, help="Minimum context size to answer.")
    p.add_argument(
        "--llm-url",
        default=env_first("LLM_URL", "RUNPOD_LLM_URL"),
        help="OpenAI-compatible base URL (env: LLM_URL or RUNPOD_LLM_URL).",
    )
    p.add_argument("--llm-model", default=env_first("LLM_MODEL", "RUNPOD_LLM_MODEL"), help="LLM model name.")
    p.add_argument(
        "--llm-api-key",
        default=env_first("LLM_API_KEY", "RUNPOD_API_TOKEN"),
        help="Bearer token/API key (env: LLM_API_KEY or RUNPOD_API_TOKEN).",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature.")
    p.add_argument("--max-tokens", type=int, default=1200, help="Maximum completion tokens.")
    p.add_argument("--request-timeout", type=int, default=240, help="LLM request timeout in seconds.")
    p.add_argument(
        "--min-relevance",
        type=float,
        default=0.01,
        help="Minimum lexical relevance score required to answer; otherwise fallback.",
    )
    p.add_argument(
        "--stability-attempts",
        type=int,
        default=2,
        help="Number of LLM generation attempts before final fallback.",
    )
    p.add_argument(
        "--debug-stability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print stability diagnostics (context/relevance/fallback decisions).",
    )
    p.add_argument(
        "--strict-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force exact fallback sentence when model returns partial/invalid fallback.",
    )
    p.add_argument(
        "--require-citations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require citation markers in model output; otherwise fallback.",
    )
    p.add_argument(
        "--strict-sentence-citations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require every non-heading content line to include citation(s); otherwise fallback.",
    )
    p.add_argument(
        "--response-style",
        choices=["strict", "chatgpt"],
        default="chatgpt",
        help="Response formatting style.",
    )
    p.add_argument(
        "--show-retrieved-refs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print retrieved context references JSON (debug mode).",
    )
    p.add_argument(
        "--retrieve-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only retrieve and print context references; skip LLM generation.",
    )
    p.add_argument(
        "--cross-source-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retrieve context from multiple source PDFs (per-source + global merge).",
    )
    p.add_argument(
        "--retrieval-profile",
        choices=["auto", "focused", "coverage"],
        default="auto",
        help=(
            "Retrieval strategy profile: "
            "'focused' = single-service focused retrieval, "
            "'coverage' = cross-source retrieval, "
            "'auto' = choose based on question intent."
        ),
    )
    p.add_argument(
        "--coverage-source-dir",
        default="data",
        help="Directory of source PDFs used to build source list for cross-source retrieval.",
    )
    p.add_argument(
        "--coverage-source",
        action="append",
        default=[],
        help="Specific source file name(s) to include in cross-source retrieval. Can repeat or use comma-separated values.",
    )
    p.add_argument(
        "--coverage-per-source-k",
        type=int,
        default=1,
        help="Chunks to retrieve per source when cross-source coverage is enabled.",
    )
    p.add_argument(
        "--coverage-max-sources",
        type=int,
        default=14,
        help="Maximum number of sources to include in cross-source retrieval.",
    )
    p.add_argument(
        "--coverage-global-k",
        type=int,
        default=10,
        help="Additional global chunks to merge with per-source retrieval.",
    )
    p.add_argument(
        "--coverage-total-k",
        type=int,
        default=28,
        help="Final merged chunk count for context when cross-source retrieval is enabled.",
    )
    return p.parse_args()


def build_context(retrieved_docs) -> Tuple[str, List[dict]]:
    blocks = []
    refs = []
    for i, d in enumerate(retrieved_docs, start=1):
        md = d.metadata or {}
        source = md.get("source_file", "unknown")
        p_start = md.get("page_start", "?")
        p_end = md.get("page_end", "?")
        service = md.get("service", "")
        topic = md.get("topic", "")
        refs.append(
            {
                "rank": i,
                "source_file": source,
                "page_start": p_start,
                "page_end": p_end,
                "service": service,
                "topic": topic,
            }
        )
        blocks.append(
            f"[Chunk {i}]\n"
            f"source_file: {source}\n"
            f"pages: {p_start}-{p_end}\n"
            f"service: {service}\n"
            f"topic: {topic}\n"
            f"content:\n{d.page_content}"
        )
    return "\n\n".join(blocks).strip(), refs


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def relevance_score(question: str, context: str) -> float:
    q_tokens = {t for t in tokenize(question) if len(t) > 2}
    if not q_tokens:
        return 0.0
    c_tokens = set(tokenize(context))
    if not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)


def context_is_strong(question: str, context: str, refs: List[dict], min_context_chars: int, min_relevance: float) -> bool:
    if len(context) < max(80, min_context_chars // 2):
        return False
    if relevance_score(question, context) < max(0.005, min_relevance * 0.8):
        return False
    services = detect_service_mentions(question)
    if not services:
        return True
    ref_services = {str(r.get("service", "")).strip() for r in refs if str(r.get("service", "")).strip()}
    return bool(ref_services & services)


def call_openai_compatible(
    llm_url: str,
    llm_model: str,
    llm_api_key: str,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if not llm_url or not llm_model:
        return FALLBACK
    endpoint = llm_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"
    payload = {
        "model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=request_timeout)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body = ""
        try:
            body = exc.response.text[:300]
        except Exception:
            pass
        raise RuntimeError(
            f"LLM request failed (HTTP). Verify --llm-url/--llm-model/token. status={getattr(exc.response, 'status_code', 'unknown')} body={body}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            "LLM request failed. Verify --llm-url/--llm-model and Bearer token."
        ) from exc
    data = resp.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def infer_service_filter(question: str) -> dict | None:
    services = detect_service_mentions(question)
    if len(services) == 1:
        return {"service": next(iter(services))}
    q = question.lower()
    # Heuristic: pipeline-run troubleshooting questions are typically ADF in this corpus.
    if (
        ("pipeline" in q and ("run" in q or "trigger" in q or "monitor" in q or "troubleshoot" in q))
        and "azure devops" not in q
        and "devops" not in q
    ):
        return {"service": "Azure Data Factory"}
    return None


def detect_service_mentions(question: str) -> Set[str]:
    q = question.lower()
    found: Set[str] = set()
    for key, service in SERVICE_HINTS.items():
        if key in q:
            found.add(service)
    return found


def should_use_coverage_mode(question: str) -> bool:
    q = question.lower()
    services = detect_service_mentions(question)
    broad_intent_terms = (
        "end to end",
        "end-to-end",
        "architecture",
        "across",
        "compare",
        "vs ",
        "versus",
        "multi",
        "integration",
        "platform",
        "pipeline from",
    )
    if len(services) >= 2:
        return True
    if any(term in q for term in broad_intent_terms):
        return True
    return False


def detect_question_type(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ("difference", "different", "compare", "comparison", "versus", " vs ", "choose between", "which should i use", "which is better")):
        return "comparison"
    if q.startswith("what is ") or q.startswith("what's ") or q.startswith("what are "):
        return "role"
    if any(x in q for x in ("role", "used for", "purpose", "where does", "how does it fit")):
        return "role"
    if any(x in q for x in ("how do i", "how to", "steps", "implement", "build")):
        return "implementation"
    return "general"


def is_definition_question(question: str) -> bool:
    q = question.lower().strip()
    return q.startswith("what is ") or q.startswith("what's ") or q.startswith("what are ")


def is_pipeline_question(question: str) -> bool:
    q = question.lower()
    markers = (
        "pipeline",
        "ingestion",
        "transform",
        "orchestration",
        "governance",
        "deployment",
        "end to end",
        "end-to-end",
        "data platform",
    )
    return any(m in q for m in markers)


def source_priority_boost(question: str, source_file: str) -> float:
    src = str(source_file).lower()
    if not is_pipeline_question(question):
        return 0.0
    # Boost core architecture docs for pipeline questions.
    if src in {
        "azure-data-factory.pdf",
        "azure-synapse-analytics.pdf",
        "azure-databricks.pdf",
        "azure-storage-blobs.pdf",
        "purview.pdf",
        "azure-devops-pipelines-azure-devops.pdf",
    }:
        return 0.18
    # Slightly de-prioritize noisy/general docs.
    if src in {
        "azure-devops-release-notes.pdf",
        "azure-devops-dev-resources-azure-devops.pdf",
    }:
        return -0.08
    return 0.0


def chunk_relevance_boost(question: str, chunk_text: str) -> float:
    q = question.lower()
    t = chunk_text.lower()
    boost = 0.0
    # Definition-style questions should favor overview/introduction chunks.
    if q.startswith("what is ") or q.startswith("what's ") or q.startswith("what are "):
        definition_terms = ("what is", "overview", "introduction", "is a", "is an", "used for")
        niche_terms = ("assistant", "mcp", "release notes", "preview", "permissions required")
        if any(x in t for x in definition_terms):
            boost += 0.10
        if any(x in t for x in niche_terms):
            boost -= 0.06
    # Purview/data-governance-focused reweighting.
    if "purview" in q or "data governance" in q or "governance" in q:
        gov_terms = ("data map", "lineage", "catalog", "scan", "classification", "glossary")
        sec_terms = ("insider risk", "dlp", "information protection", "endpoint", "m365")
        if any(x in t for x in gov_terms):
            boost += 0.16
        if any(x in t for x in sec_terms):
            boost -= 0.12
    # Comparison questions should prefer conceptual/decision-oriented chunks.
    if any(x in q for x in ("difference", "compare", "versus", " vs ", "choose between", "which should i use")):
        cmp_terms = ("when to use", "best for", "tradeoff", "difference", "comparison", "architecture", "choose")
        setup_terms = ("quickstart", "tutorial", "step-by-step", "install", "create a workspace")
        if any(x in t for x in cmp_terms):
            boost += 0.10
        if any(x in t for x in setup_terms):
            boost -= 0.06
    # Role-in-pipeline questions should prioritize integration/choreography language.
    if any(x in q for x in ("role", "used for", "fits", "pipeline")):
        role_terms = ("orchestrate", "integrate", "pipeline", "workflow", "control plane", "compute", "storage", "governance")
        if any(x in t for x in role_terms):
            boost += 0.08
    # ADF scheduling procedural reweighting.
    if ("data factory" in q or "adf" in q) and ("schedule" in q or "trigger" in q):
        ui_terms = ("trigger", "schedule", "studio", "portal", "pipeline")
        cli_terms = ("powershell", "invoke-azdatafactory", "cmdlet")
        if any(x in t for x in ui_terms):
            boost += 0.08
        if any(x in t for x in cli_terms):
            boost -= 0.03
    # ADF monitor/troubleshoot questions should bias monitoring and troubleshooting chunks.
    if ("data factory" in q or "adf" in q or "pipeline" in q) and any(
        x in q for x in ("monitor", "troubleshoot", "failed", "failure", "error", "debug", "run")
    ):
        monitor_terms = ("monitor", "troubleshoot", "failed", "failure", "error", "activity run", "pipeline run", "alert")
        if any(x in t for x in monitor_terms):
            boost += 0.12
    return boost


def rerank_docs(question: str, docs, limit: int):
    scored = sorted(
        docs,
        key=lambda d: (
            lexical_overlap_score(question, d.page_content)
            + source_priority_boost(question, (d.metadata or {}).get("source_file", ""))
            + chunk_relevance_boost(question, d.page_content)
        ),
        reverse=True,
    )
    return scored[: max(1, limit)]


def dynamic_max_tokens(args: argparse.Namespace, question: str, question_type: str) -> int:
    base = max(256, int(args.max_tokens))
    q = question.lower()
    complexity = 0
    if is_definition_question(question):
        # Keep "what is" answers concise and stable.
        return min(700, base)
    if question_type == "comparison":
        complexity += 420
    if question_type == "implementation":
        complexity += 180
    if any(x in q for x in ("end to end", "end-to-end", "architecture", "across", "multi")):
        complexity += 220
    # Ensure richer answers for architect-level comparison prompts.
    if question_type == "comparison":
        return max(1500, min(1800, base + complexity))
    return min(1800, base + complexity)


def normalize_response(text: str, strict_fallback: bool) -> str:
    if not strict_fallback:
        return text.strip()
    cleaned = text.strip()
    if cleaned == FALLBACK:
        return FALLBACK
    # Do not collapse a full answer to fallback just because fallback text appears in one line.
    # Some models include the sentence as a disclaimer while still providing grounded content.
    if cleaned.lower().startswith(FALLBACK.lower()) and len(cleaned) <= len(FALLBACK) + 20:
        return FALLBACK
    cleaned = cleaned.replace(FALLBACK, "").strip()
    return cleaned


def has_citations(text: str) -> bool:
    return bool(CITATION_RE.search(text))


def canonicalize_citations(text: str) -> str:
    text = re.sub(r"\[\s*chunk\s*\d+[^\]]*\]", "", text, flags=re.IGNORECASE)
    # Normalize common model citation variants to: [file pX-Y]
    text = re.sub(
        r"\[source_file:\s*([^\]\n]+?)\s+pages:\s*(\d+)\s*-\s*(\d+)\]",
        r"[\1 p\2-\3]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[source_file:\s*([^\]\n]+?)\s+pages:\s*(\d+)\]",
        r"[\1 p\2]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[([^\]\n]+?)\s+pages?\s*:?\s*(\d+)\s*-\s*(\d+)\]",
        r"[\1 p\2-\3]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[([^\]\n]+?)\s+pages?\s*:?\s*(\d+)\]",
        r"[\1 p\2]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[\s*source_file\s*:\s*([^\],\n]+)\s+p(\d+(?:-\d+)?)\s*\]",
        r"[\1 p\2]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[\s*source_file\s*:\s*([a-z0-9_\-]+\.pdf)\s+p(\d+(?:-\d+)?)\s*,\s*p(\d+(?:-\d+)?)\s*\]",
        r"[\1 p\2] [\1 p\3]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[\s*source_file\s*:\s*([^\],\n]+)\s+pages?\s*:\s*(\d+)\s*-\s*(\d+)\s*\]",
        r"[\1 p\2-\3]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[\s*source_file\s*:\s*([^\],\n]+)\s+pages?\s*:\s*(\d+)\s*\]",
        r"[\1 p\2]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[\s*([a-z0-9_\-]+\.pdf)\s*,\s*p(\d+)(?:-(\d+))?\s*\]",
        lambda m: f"[{m.group(1)} p{m.group(2)}" + (f"-{m.group(3)}" if m.group(3) else "") + "]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\[\d+\s*:\s*\d+\]",
        "",
        text,
    )
    return text


def split_sections(text: str) -> List[Tuple[str, List[str]]]:
    sections: List[Tuple[str, List[str]]] = []
    current_title = ""
    current_lines: List[str] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            if current_title:
                sections.append((current_title, current_lines))
            current_title = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title:
        sections.append((current_title, current_lines))
    return sections


def render_sections(sections: List[Tuple[str, List[str]]]) -> str:
    out: List[str] = []
    for title, lines in sections:
        out.append(f"### {title}")
        content = "\n".join(lines).strip()
        out.append(content if content else "This information is not found in the indexed official Microsoft documentation.")
    return "\n\n".join(out).strip()


def build_reference_lines(refs: List[dict]) -> List[str]:
    seen = set()
    lines: List[str] = []
    for r in refs:
        src = str(r.get("source_file", "unknown"))
        ps = str(r.get("page_start", "?"))
        pe = str(r.get("page_end", "?"))
        key = (src, ps, pe)
        if key in seen:
            continue
        seen.add(key)
        pages = f"p{ps}" if ps == pe else f"p{ps}-{pe}"
        lines.append(f"- {src} [{pages}]")
    return lines


def ensure_sources_section(response: str, refs: List[dict]) -> str:
    if not response.strip():
        return FALLBACK
    if re.search(r"^###\s+Sources\s*$", response, flags=re.IGNORECASE | re.MULTILINE):
        return response
    lines = build_reference_lines(refs)
    if not lines:
        return response
    return response.strip() + "\n\n### Sources\n" + "\n".join(lines)


def clean_chatgpt_response(response: str, refs: List[dict]) -> str:
    response = canonicalize_citations(response).strip()
    if not response or response == FALLBACK:
        return FALLBACK
    # Normalize plain-label section headers to markdown headers.
    response = re.sub(r"(?im)^summary:\s*$", "### Summary", response)
    response = re.sub(r"(?im)^architecture:\s*$", "### Architecture", response)
    response = re.sub(r"(?im)^implementation steps:\s*$", "### Implementation Steps", response)
    response = re.sub(r"(?im)^security\s*&\s*governance:\s*$", "### Security & Governance", response)
    response = re.sub(r"(?im)^performance:\s*$", "### Performance", response)
    # Remove noisy non-grounded citation placeholders.
    response = re.sub(r"\[(?:chunk|source_)\d+[^\]]*\]", "", response, flags=re.IGNORECASE)
    response = re.sub(r"\[(?:\d+\s*,\s*)+\d+\]", "", response)
    response = re.sub(r"\[\d+\]", "", response)
    # Remove non-canonical source breadcrumbs emitted by some models.
    response = re.sub(r"(?im)^\s*source_file\s*:\s*.+$", "", response)
    response = re.sub(r"(?im)^\s*\[/?note\]\s*:?.*$", "", response)
    response = re.sub(r"(?im)^\s*\[source_file\]\s*$", "", response)
    response = re.sub(r"\s+([.,;:!?])", r"\1", response)
    response = re.sub(r"(?m)^\s*[.,;:!?]+\s*$", "", response)

    # Strip any model-generated sources/reference tail; we append canonical sources later.
    lines = response.splitlines()
    cut_idx = None
    for i, line in enumerate(lines):
        if re.match(r"(?i)^\s*(sources|reference)\s*:\s*$", line.strip()):
            cut_idx = i
            break
        if re.match(r"(?i)^\s*###\s*(sources|reference)\s*$", line.strip()):
            cut_idx = i
            break
        if re.match(r"(?i)^\s*\[\s*sources?\s*\]\s*$", line.strip()):
            cut_idx = i
            break
    if cut_idx is not None:
        lines = lines[:cut_idx]

    # Remove repetitive fallback filler lines from model output for cleaner UX.
    cleaned_lines = []
    for ln in lines:
        s = ln.strip()
        if s == FALLBACK:
            continue
        if re.match(r"(?i)^\[\s*sources?\s*\]$", s):
            continue
        if re.match(r"(?i)^sources?\s*:\s*$", s):
            continue
        if s in {".", "..", "..."}:
            continue
        cleaned_lines.append(ln)
    response = "\n".join(cleaned_lines).strip()
    response = re.sub(r"\n{3,}", "\n\n", response)
    response = ensure_sources_section(response, refs)
    return response if response else FALLBACK


def enforce_definition_conciseness(response: str, refs: List[dict], max_paragraphs: int = 3) -> str:
    if not response or response == FALLBACK:
        return response
    text = response.strip()
    # Remove tables for simple definition questions; they add noise and instability.
    text = re.sub(r"(?ms)^\|.*?$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Split body from sources, then keep only the first few meaningful paragraphs.
    parts = re.split(r"(?im)^###\s+Sources\s*$", text, maxsplit=1)
    body = parts[0].strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    concise = []
    for p in paras:
        concise.append(p)
        if len(concise) >= max_paragraphs:
            break
    if not concise and paras:
        concise = paras[:max_paragraphs]
    out = "\n\n".join(concise).strip() if concise else body
    out = ensure_sources_section(out, refs)
    return out if out else FALLBACK


def has_strict_sentence_citations(response: str) -> bool:
    in_sources = False
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"(?i)^###\s+sources\s*$", line):
            in_sources = True
            continue
        if re.match(r"^###\s+", line):
            in_sources = False
            continue
        if in_sources:
            continue
        if line == FALLBACK:
            continue
        # Skip plain list markers, but not list content.
        if line in {"-", "*"}:
            continue
        # Any substantial content line must contain a normalized citation.
        if not CITATION_RE.search(line):
            return False
    return True


def ensure_required_sections(response: str, required_titles: List[str]) -> str:
    if not response or response == FALLBACK:
        return response
    existing = {m.group(1).strip().lower() for m in re.finditer(r"(?im)^###\s+(.+?)\s*$", response)}
    missing = [t for t in required_titles if t.lower() not in existing]
    if not missing:
        return response
    blocks = [response.rstrip()]
    for t in missing:
        blocks.append(f"\n### {t}\nThis information is not found in the indexed official Microsoft documentation.")
    return "\n".join(blocks).strip()


def parse_source_args(values: List[str]) -> Set[str]:
    parsed: Set[str] = set()
    for v in values:
        if not v:
            continue
        for part in v.split(","):
            name = part.strip()
            if name:
                parsed.add(name)
    return parsed


def list_sources_from_dir(source_dir: str) -> List[str]:
    p = Path(source_dir)
    if not p.exists() or not p.is_dir():
        return []
    return sorted(x.name for x in p.glob("*.pdf") if x.is_file())


def lexical_overlap_score(question: str, chunk_text: str) -> float:
    q_tokens = {t for t in tokenize(question) if len(t) > 2}
    if not q_tokens:
        return 0.0
    c_tokens = set(tokenize(chunk_text))
    if not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)


def build_extractive_answer(question: str, docs, refs: List[dict], max_points: int = 5) -> str:
    q_tokens = {t for t in tokenize(question) if len(t) > 2}
    candidates: List[Tuple[float, str]] = []
    seen_sentences: Set[str] = set()

    for d in docs:
        md = d.metadata or {}
        src = str(md.get("source_file", "unknown"))
        ps = str(md.get("page_start", "?"))
        pe = str(md.get("page_end", "?"))
        page_tag = f"p{ps}" if ps == pe else f"p{ps}-{pe}"
        text = re.sub(r"\s+", " ", d.page_content).strip()
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s = s.strip()
            if len(s) < 50:
                continue
            normalized = s.lower()
            if normalized in seen_sentences:
                continue
            seen_sentences.add(normalized)
            s_tokens = set(tokenize(s))
            score = len(q_tokens & s_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
            candidates.append((score, f"{s} [{src} {page_tag}]"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    lines = [c for score, c in candidates if score > 0][:max_points]
    if not lines:
        lines = [c for _, c in candidates[:max_points]]
    if not lines:
        return FALLBACK

    out = "Based on the indexed documentation:\n\n"
    for i, line in enumerate(lines, start=1):
        out += f"{i}. {line}\n"
    src_lines = build_reference_lines(refs)
    if src_lines:
        out += "\n### Sources\n" + "\n".join(src_lines)
    return out.strip()


def merge_filters(base_filter: dict | None, source_file: str | None) -> dict | None:
    if base_filter is None and source_file is None:
        return None
    # Chroma where-filter in this stack is strict with operators.
    # For per-source coverage retrieval, prioritize source_file filter only.
    if source_file:
        return {"source_file": source_file}
    merged: Dict[str, str] = {}
    if isinstance(base_filter, dict):
        for k, v in base_filter.items():
            if isinstance(v, str):
                merged[k] = v
    if source_file:
        merged["source_file"] = source_file
    return merged if merged else None


def default_retrieve(
    args: argparse.Namespace,
    vectorstore: Chroma,
    question: str,
    metadata_filter: dict | None,
    k_override: int | None = None,
):
    k = args.top_k if k_override is None else max(1, k_override)
    if args.retrieval_mode == "mmr":
        return vectorstore.max_marginal_relevance_search(
            question,
            k=k,
            fetch_k=max(args.fetch_k, k),
            filter=metadata_filter,
        )
    return vectorstore.similarity_search(question, k=k, filter=metadata_filter)


def retrieve_with_cross_source_coverage(
    args: argparse.Namespace,
    vectorstore: Chroma,
    question: str,
    metadata_filter: dict | None,
):
    sources = parse_source_args(args.coverage_source)
    if not sources:
        sources = set(list_sources_from_dir(args.coverage_source_dir))
    if not sources:
        return default_retrieve(args, vectorstore, question, metadata_filter)

    source_list = sorted(sources)[: max(1, args.coverage_max_sources)]
    per_source_docs = []
    for source in source_list:
        filt = merge_filters(metadata_filter, source)
        if args.retrieval_mode == "mmr":
            docs = vectorstore.max_marginal_relevance_search(
                question,
                k=max(1, args.coverage_per_source_k),
                fetch_k=max(args.fetch_k, args.coverage_per_source_k),
                filter=filt,
            )
        else:
            docs = vectorstore.similarity_search(
                question,
                k=max(1, args.coverage_per_source_k),
                filter=filt,
            )
        per_source_docs.extend(docs)

    global_k = max(args.top_k, args.coverage_global_k)
    global_docs = default_retrieve(args, vectorstore, question, metadata_filter, k_override=global_k)

    combined = per_source_docs + global_docs
    deduped = []
    seen = set()
    for d in combined:
        md = d.metadata or {}
        key = str(md.get("chunk_id") or f"{md.get('source_file')}:{md.get('page_start')}:{md.get('page_end')}:{hash(d.page_content)}")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)

    scored = sorted(
        deduped,
        key=lambda d: (
            lexical_overlap_score(question, d.page_content)
            + source_priority_boost(question, (d.metadata or {}).get("source_file", ""))
        ),
        reverse=True,
    )
    limit = max(args.top_k, args.coverage_total_k)
    return scored[:limit]


def enforce_structure_and_grounding(response: str, refs: List[dict]) -> str:
    response = canonicalize_citations(response)
    sections = split_sections(response)
    if not sections:
        return response

    weak_headings = {
        "Best Practices",
        "Security & Governance Considerations",
        "Performance Considerations",
    }
    rebuilt: List[Tuple[str, List[str]]] = []
    present = {title for title, _ in sections}

    for title, lines in sections:
        if title == "Reference":
            continue
        content = "\n".join(lines).strip()
        if title in weak_headings and not has_citations(content):
            content = "This information is not found in the indexed official Microsoft documentation."
        rebuilt.append((title, [content]))

    # Ensure required headings exist at least with fallback text.
    for title in CANONICAL_SECTIONS:
        if title == "Reference":
            continue
        if title not in present:
            rebuilt.append((title, ["This information is not found in the indexed official Microsoft documentation."]))

    ref_lines = build_reference_lines(refs)
    rebuilt.append(("Reference", ref_lines if ref_lines else ["This information is not found in the indexed official Microsoft documentation."]))
    return render_sections(rebuilt)


def answer_question(args: argparse.Namespace, vectorstore: Chroma, question: str) -> str:
    if not question.strip():
        return FALLBACK

    metadata_filter = infer_service_filter(question)
    use_coverage = args.cross_source_coverage
    if args.retrieval_profile == "coverage":
        use_coverage = True
    elif args.retrieval_profile == "focused":
        use_coverage = False
    else:
        use_coverage = should_use_coverage_mode(question)

    question_type = detect_question_type(question)
    if use_coverage:
        docs = retrieve_with_cross_source_coverage(args, vectorstore, question, metadata_filter)
    else:
        docs = default_retrieve(args, vectorstore, question, metadata_filter)
        docs = rerank_docs(question, docs, args.top_k)
    active_docs = docs
    context, refs = build_context(docs)
    if args.show_retrieved_refs:
        print("\n--- Retrieved Context References ---")
        print(json.dumps(refs, indent=2))

    if args.retrieve_only:
        return FALLBACK
    # Second-chance retrieval before hard fallback: relax service filter and increase recall.
    if len(context) < args.min_context_chars or relevance_score(question, context) < args.min_relevance:
        if use_coverage:
            docs = retrieve_with_cross_source_coverage(args, vectorstore, question, None)
        else:
            docs = default_retrieve(
                args,
                vectorstore,
                question,
                None,
                k_override=max(args.top_k + 4, args.top_k * 2),
            )
            docs = rerank_docs(question, docs, max(args.top_k + 4, args.top_k * 2))
        context, refs = build_context(docs)
        active_docs = docs
        if args.show_retrieved_refs:
            print("\n--- Retrieved Context References (retry) ---")
            print(json.dumps(refs, indent=2))
        if len(context) < args.min_context_chars or relevance_score(question, context) < args.min_relevance:
            return FALLBACK
    strong_context = context_is_strong(question, context, refs, args.min_context_chars, args.min_relevance)
    if args.debug_stability:
        ref_services = sorted({str(r.get("service", "")).strip() for r in refs if str(r.get("service", "")).strip()})
        print(
            f"[debug] context_len={len(context)} relevance={relevance_score(question, context):.4f} "
            f"strong_context={strong_context} services={sorted(detect_service_mentions(question))} "
            f"ref_services={ref_services}",
            file=sys.stderr,
        )

    if args.response_style == "strict":
        system_prompt = (
            "You are an Enterprise Azure Data Platform Architect and Senior Data Engineer.\n"
            "Use ONLY the provided context from official Microsoft documentation.\n"
            "Do NOT use external or pre-trained knowledge.\n"
            "Do NOT assume missing facts.\n"
            "If answer is not supported by context, return exactly:\n"
            f"\"{FALLBACK}\"\n\n"
            "Citations are mandatory.\n"
            "For every factual statement, append a citation in this format: [source_file pX-Y].\n"
            "Use only source_file and page ranges present in context.\n"
            "Required response format:\n"
            "### Overview\n"
            "### Architecture Explanation\n"
            "### Implementation Steps\n"
            "### Best Practices\n"
            "### Security & Governance Considerations\n"
            "### Performance Considerations\n"
            "### Reference\n"
            "In Reference section, use only sources present in context.\n"
        )
    else:
        comparison_directive = ""
        if question_type == "comparison":
            comparison_directive = (
                "For comparison questions, prefer these sections when evidence exists:\n"
                "### Architectural Positioning\n"
                "### Strengths\n"
                "### Trade-offs\n"
                "### When to Use Each\n"
                "### Key Differences\n"
                "### Architecture Differences\n"
                "### Decision Matrix\n"
                "### Comparison Table\n"
                "In Comparison Table, include at least 4 rows and no empty cells when possible.\n"
                "Be opinionated and explicit about recommended fit by workload type.\n"
            )
        howto_directive = ""
        if question_type == "implementation" and (
            "schedule" in question.lower() or "trigger" in question.lower() or "pipeline" in question.lower()
        ):
            howto_directive = (
                "For procedural questions, include these sections only if context supports them:\n"
                "### Portal-Based Method (UI)\n"
                "### CLI/PowerShell Method\n"
                "### Scheduling via Trigger\n"
                "### Best Practices\n"
            )
        section_mode = (
            "Use adaptive sections based on question intent. "
            "Do not force irrelevant sections. "
            "If a section is not supported by context, omit it instead of adding placeholder text.\n"
        )
        system_prompt = (
            "You are a senior Azure enterprise data architect.\n"
            "Use ONLY the provided context from official Microsoft documentation.\n"
            "Write a clean, direct answer without meta-commentary.\n"
            "Synthesize across sources: do not list products independently; explain how they connect in one architecture.\n"
            "Prefer concrete, implementation-level guidance over generic descriptions.\n"
            "For each section, include actionable details (service choice + purpose + integration point).\n"
            f"If context is insufficient, return exactly: \"{FALLBACK}\"\n"
            "Always explain where each service fits in the end-to-end data lifecycle.\n"
            "Explicitly label whether each service acts as storage, compute, orchestration, governance, or CI/CD layer.\n"
            "Explain integration paths between services instead of isolated descriptions.\n"
            f"{section_mode}"
            "Always include:\n"
            "### Sources\n"
            "Use other sections only when relevant (e.g., Summary, Architecture, Implementation Steps, "
            "Security & Governance, Performance, Comparison Table).\n"
            "For implementation questions, provide a stage-wise plan (ingest, transform, orchestrate, deploy, govern) if relevant.\n"
            "For role/purpose questions, emphasize role in pipeline lifecycle.\n"
            f"{comparison_directive}"
            f"{howto_directive}"
            "Always clarify whether each service is control plane, compute plane, storage plane, or governance plane.\n"
            "Include citations as [source_file pX-Y] in factual lines and list sources at end."
        )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Question type detected: {question_type}\n\n"
        f"Context:\n{context}\n\n"
        "Synthesize cross-documently: connect services into one end-to-end flow.\n"
        "Avoid generic statements; each major claim should map to cited context.\n"
        "For comparison questions, include a comparison table when context supports it.\n"
        "For role/purpose questions, include explicit role-in-pipeline positioning.\n"
        "Do not include irrelevant sections.\n"
        "Do not use generic references. Use only citations grounded in provided chunks."
    )

    def run_generation(prompt_suffix: str = "", temperature_override: float | None = None) -> str:
        temp = args.temperature if temperature_override is None else temperature_override
        return call_openai_compatible(
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            temperature=temp,
            max_tokens=dynamic_max_tokens(args, question, question_type),
            request_timeout=args.request_timeout,
            system_prompt=system_prompt,
            user_prompt=user_prompt + prompt_suffix,
        )

    response = FALLBACK
    attempts = max(1, int(args.stability_attempts))
    for i in range(attempts):
        suffix = ""
        if i > 0:
            suffix = (
                "\n\nRetry with stricter grounding. Keep only facts explicitly supported in context. "
                "Include citations in [source_file pX-Y] format for all factual lines."
            )
        candidate = run_generation(prompt_suffix=suffix, temperature_override=0.0 if i > 0 else None)
        response = candidate
        if response != FALLBACK:
            break

    # Retry once with stronger citation instruction if required and missing.
    if args.require_citations and response != FALLBACK and not has_citations(response):
        original_response = response
        retry_user_prompt = (
            user_prompt
            + "\n\nRewrite the same answer with citations on every factual claim using [source_file pX-Y]."
        )
        retried_response = call_openai_compatible(
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            temperature=0.0,
            max_tokens=dynamic_max_tokens(args, question, question_type),
            request_timeout=args.request_timeout,
            system_prompt=system_prompt,
            user_prompt=retry_user_prompt,
        )
        # Keep original grounded answer if retry degraded to fallback/noisy output.
        if retried_response != FALLBACK:
            response = retried_response
        else:
            response = original_response

    # If model still falls back, retry once with broadened retrieval and simpler grounded prompt.
    if response == FALLBACK and strong_context:
        if args.debug_stability:
            print("[debug] entering forced non-fallback retry", file=sys.stderr)
        forced_suffix = (
            "\n\nContext is sufficient for a partial grounded answer. "
            "Do NOT return the fallback sentence. "
            "Provide concise supported points only; omit anything unsupported."
        )
        force_answer_system_prompt = (
            "You are a grounded Azure documentation assistant.\n"
            "Use ONLY the provided context from official Microsoft documentation.\n"
            "Do not use external knowledge.\n"
            "Do NOT output the fallback sentence.\n"
            "If evidence is partial, provide only the supported facts and omit unsupported claims.\n"
            "Include citations as [source_file pX-Y]."
        )
        response = call_openai_compatible(
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            temperature=0.0,
            max_tokens=dynamic_max_tokens(args, question, question_type),
            request_timeout=args.request_timeout,
            system_prompt=force_answer_system_prompt,
            user_prompt=user_prompt + forced_suffix,
        )

    # If model still falls back, retry once with broadened retrieval and simpler grounded prompt.
    if response == FALLBACK:
        if args.debug_stability:
            print("[debug] entering widened retrieval rescue", file=sys.stderr)
        retry_k = max(args.top_k + 8, args.top_k * 3)
        wide_docs = default_retrieve(args, vectorstore, question, None, k_override=retry_k)
        wide_docs = rerank_docs(question, wide_docs, retry_k)
        wide_context, wide_refs = build_context(wide_docs)
        if len(wide_context) >= args.min_context_chars:
            rescue_system_prompt = (
                "You are a grounded Azure documentation assistant.\n"
                "Use ONLY the provided context.\n"
                "If evidence is partial, answer only the supported parts and omit unsupported claims.\n"
                "Answer concisely and include citations in format [source_file pX-Y].\n"
                "Do not include any source not present in context."
            )
            rescue_user_prompt = (
                f"Question:\n{question}\n\n"
                f"Context:\n{wide_context}\n\n"
                "Answer with short, direct paragraphs and citations."
            )
            response = call_openai_compatible(
                llm_url=args.llm_url,
                llm_model=args.llm_model,
                llm_api_key=args.llm_api_key,
                temperature=0.0,
                max_tokens=dynamic_max_tokens(args, question, question_type),
                request_timeout=args.request_timeout,
                system_prompt=rescue_system_prompt,
                user_prompt=rescue_user_prompt,
            )
            if response != FALLBACK:
                refs = wide_refs
                active_docs = wide_docs

    if response == FALLBACK and strong_context:
        if args.debug_stability:
            print("[debug] using extractive recovery (pre-normalize)", file=sys.stderr)
        response = build_extractive_answer(question, active_docs, refs)

    response = normalize_response(response, args.strict_fallback)
    if response != FALLBACK:
        if args.response_style == "strict":
            response = enforce_structure_and_grounding(response, refs)
        else:
            response = clean_chatgpt_response(response, refs)
            if is_definition_question(question):
                response = enforce_definition_conciseness(response, refs)
    if response == FALLBACK and strong_context:
        if args.debug_stability:
            print("[debug] using extractive recovery (post-normalize)", file=sys.stderr)
        response = build_extractive_answer(question, active_docs, refs)
    if args.require_citations and response != FALLBACK and not has_citations(response):
        # Last-mile grounding recovery: append canonical sources if model omitted inline markers.
        response = ensure_sources_section(response, refs)
        if not has_citations(response):
            return FALLBACK
    if args.strict_sentence_citations and response != FALLBACK and not has_strict_sentence_citations(response):
        return FALLBACK
    return response


def interactive_loop(args: argparse.Namespace, vectorstore: Chroma) -> int:
    print("Interactive RAG mode. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in {"exit", "quit"}:
            return 0
        try:
            print(answer_question(args, vectorstore, q))
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)


def main() -> int:
    # Auto-load local .env for RUNPOD/LLM settings.
    load_dotenv()
    args = parse_args()
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    vectorstore = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.persist_dir,
    )

    if args.question:
        try:
            print(answer_question(args, vectorstore, args.question))
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    return interactive_loop(args, vectorstore)


if __name__ == "__main__":
    raise SystemExit(main())
