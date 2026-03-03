#!/usr/bin/env python3
"""
Build training artifacts from Azure PDF documentation.

Outputs:
1) rag_chunks.jsonl  -> retrieval-ready chunks with metadata/citations
2) sft_train.jsonl   -> optional supervised fine-tuning records (Q/A)
3) build_stats.json  -> run statistics

Usage examples:
  python build_training_data.py --input-dir Data --output-dir output
  python build_training_data.py --input-dir Data --output-dir output --build-sft --llm-url http://localhost:8000/v1 --llm-model llama-3.1-8b-instruct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import requests
except ImportError:
    requests = None


@dataclass
class ChunkRecord:
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    service: str
    topic: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create RAG/SFT training data from a folder of Azure documentation PDFs."
    )
    parser.add_argument("--input-dir", default="Data", help="Folder containing PDF files.")
    parser.add_argument("--output-dir", default="output", help="Where output files are written.")
    parser.add_argument("--chunk-size", type=int, default=1400, help="Target chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap size in characters.")
    parser.add_argument("--max-chunks-per-doc", type=int, default=400, help="Upper bound chunks per document.")
    parser.add_argument("--build-sft", action="store_true", help="Also generate SFT Q/A samples.")
    parser.add_argument("--qa-per-chunk", type=int, default=2, help="Target Q/A count per chunk.")
    parser.add_argument("--llm-url", default="", help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--llm-model", default="", help="Model name for the OpenAI-compatible endpoint.")
    parser.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", ""), help="API key if required.")
    parser.add_argument("--min-chars", type=int, default=500, help="Skip chunks shorter than this many chars.")
    return parser.parse_args()


def list_pdfs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.pdf") if p.is_file()])


def infer_service(file_name: str) -> str:
    name = file_name.lower()
    mapping = {
        "data-factory": "Azure Data Factory",
        "databricks": "Azure Databricks",
        "synapse": "Azure Synapse Analytics",
        "storage-blobs": "Azure Blob Storage",
        "devops": "Azure DevOps",
        "power-bi": "Power BI",
        "fabric": "Microsoft Fabric",
        "purview": "Microsoft Purview",
        "dax": "DAX",
    }
    for key, value in mapping.items():
        if key in name:
            return value
    return "Azure"


def infer_topic(file_name: str) -> str:
    name = file_name.lower()
    if "release-notes" in name:
        return "release_notes"
    if "pipelines" in name:
        return "pipelines"
    if "get-started" in name:
        return "getting_started"
    if "organizations" in name:
        return "governance"
    if "storage" in name or "blobs" in name:
        return "storage"
    if "devops" in name:
        return "ci_cd"
    if "purview" in name:
        return "data_governance"
    if "synapse" in name or "databricks" in name:
        return "transformation"
    if "data-factory" in name:
        return "orchestration"
    return "general"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> List[str]:
    if PdfReader is None:
        raise RuntimeError("Missing dependency: pypdf. Install with `pip install pypdf`.")
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(normalize_text(text))
    return pages


def chunk_pages(
    pages: List[str],
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
) -> List[tuple[str, int, int]]:
    """Return list of (chunk_text, page_start, page_end)."""
    full = []
    for i, page in enumerate(pages, start=1):
        if page:
            full.append(f"[PAGE {i}]\n{page}\n")
    document = "\n".join(full)
    if not document.strip():
        return []

    chunks: List[tuple[str, int, int]] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(document), step):
        end = min(len(document), start + chunk_size)
        block = document[start:end]
        if not block.strip():
            continue
        page_nums = [int(n) for n in re.findall(r"\[PAGE (\d+)\]", block)]
        page_start = min(page_nums) if page_nums else 1
        page_end = max(page_nums) if page_nums else page_start
        cleaned = re.sub(r"\[PAGE \d+\]\n?", "", block).strip()
        if cleaned:
            chunks.append((cleaned, page_start, page_end))
        if len(chunks) >= max_chunks:
            break
    return chunks


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def heuristic_qa(chunk: ChunkRecord, qa_per_chunk: int) -> List[dict]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.text) if len(s.strip()) > 40]
    if not sentences:
        return []
    seed = " ".join(sentences[: min(5, len(sentences))])
    records = []
    for i in range(qa_per_chunk):
        if i % 2 == 0:
            question = f"How is {chunk.service} used for {chunk.topic} in an Azure data engineering pipeline?"
            answer = seed
        else:
            question = f"What are best practices from this {chunk.service} documentation section?"
            answer = " ".join(sentences[: min(7, len(sentences))])
        records.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an Azure data engineering assistant. "
                            "Answer using only official Microsoft documentation context."
                        ),
                    },
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": f"{answer}\n\nSource: {chunk.source_file} (pages {chunk.page_start}-{chunk.page_end})",
                    },
                ],
                "metadata": {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "service": chunk.service,
                    "topic": chunk.topic,
                    "generator": "heuristic",
                },
            }
        )
    return records


def parse_json_from_text(raw: str) -> Optional[list]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        return None
    return None


def llm_qa(
    chunk: ChunkRecord,
    qa_per_chunk: int,
    llm_url: str,
    llm_model: str,
    llm_api_key: str,
    timeout_s: int = 90,
) -> List[dict]:
    if requests is None:
        raise RuntimeError("Missing dependency: requests. Install with `pip install requests`.")
    if not llm_url or not llm_model:
        return []

    system = (
        "Create grounded training data for an Azure data engineering assistant. "
        "Return JSON array only. Each element must contain keys: question, answer. "
        "Answer only from given context and include concrete steps when possible."
    )
    user = (
        f"Generate {qa_per_chunk} high-quality QA pairs from this context.\n"
        "Constraints:\n"
        "- Focus on building Azure data engineering pipelines end to end.\n"
        "- Questions must be practical and implementation-focused.\n"
        "- Answers must be concise and factual.\n"
        "- No markdown code fences.\n\n"
        f"Service: {chunk.service}\n"
        f"Topic: {chunk.topic}\n"
        f"Source: {chunk.source_file} pages {chunk.page_start}-{chunk.page_end}\n\n"
        f"Context:\n{chunk.text[:5000]}"
    )
    headers = {"Content-Type": "application/json"}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"
    payload = {
        "model": llm_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    endpoint = llm_url.rstrip("/") + "/chat/completions"
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = parse_json_from_text(content)
    if not parsed:
        return []

    out = []
    for item in parsed[:qa_per_chunk]:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        out.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an Azure data engineering assistant. "
                            "Answer using only official Microsoft documentation context."
                        ),
                    },
                    {"role": "user", "content": q},
                    {
                        "role": "assistant",
                        "content": f"{a}\n\nSource: {chunk.source_file} (pages {chunk.page_start}-{chunk.page_end})",
                    },
                ],
                "metadata": {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "service": chunk.service,
                    "topic": chunk.topic,
                    "generator": "llm",
                },
            }
        )
    return out


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    pdfs = list_pdfs(input_dir)
    if not pdfs:
        print(f"No PDF files found in: {input_dir}", file=sys.stderr)
        return 1

    chunks_for_sft: List[ChunkRecord] = []
    failures: List[dict] = []
    t0 = time.time()
    total_pdfs = len(pdfs)
    rag_path = output_dir / "rag_chunks.jsonl"
    rag_count = 0

    # Stream writes so progress is visible during long runs.
    with rag_path.open("w", encoding="utf-8") as rag_f:
        for idx_pdf, pdf in enumerate(pdfs, start=1):
            source = pdf.name
            service = infer_service(source)
            topic = infer_topic(source)
            print(f"[{idx_pdf}/{total_pdfs}] Processing {source} ...", flush=True)
            try:
                pages = extract_pdf_pages(pdf)
                chunked = chunk_pages(
                    pages=pages,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    max_chunks=args.max_chunks_per_doc,
                )
            except Exception as exc:
                failures.append({"file": source, "error": str(exc)})
                print(f"[{idx_pdf}/{total_pdfs}] Failed {source}: {exc}", flush=True)
                continue

            for idx, (text, p_start, p_end) in enumerate(chunked, start=1):
                if len(text) < args.min_chars:
                    continue
                chunk_id = f"{pdf.stem}-{idx:04d}"
                rec = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": {
                        "source_file": source,
                        "page_start": p_start,
                        "page_end": p_end,
                        "service": service,
                        "topic": topic,
                    },
                }
                rag_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rag_count += 1
                chunks_for_sft.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source_file=source,
                        page_start=p_start,
                        page_end=p_end,
                        service=service,
                        topic=topic,
                        text=text,
                    )
                )
            print(
                f"[{idx_pdf}/{total_pdfs}] Done {source} | pages={len(pages)} chunks={len(chunked)}",
                flush=True,
            )

    sft_count = 0
    sft_path = output_dir / "sft_train.jsonl"
    if args.build_sft:
        sft_records: List[dict] = []
        use_llm = bool(args.llm_url and args.llm_model)
        for chunk in chunks_for_sft:
            try:
                if use_llm:
                    qa = llm_qa(
                        chunk=chunk,
                        qa_per_chunk=args.qa_per_chunk,
                        llm_url=args.llm_url,
                        llm_model=args.llm_model,
                        llm_api_key=args.llm_api_key,
                    )
                    if not qa:
                        qa = heuristic_qa(chunk, args.qa_per_chunk)
                else:
                    qa = heuristic_qa(chunk, args.qa_per_chunk)
                sft_records.extend(qa)
            except Exception as exc:
                failures.append({"file": chunk.source_file, "chunk_id": chunk.chunk_id, "error": str(exc)})
        sft_count = write_jsonl(sft_path, sft_records)

    elapsed = round(time.time() - t0, 2)
    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pdf_count": len(pdfs),
        "rag_chunk_count": rag_count,
        "sft_record_count": sft_count,
        "failed_items": failures,
        "elapsed_seconds": elapsed,
    }
    stats_path = output_dir / "build_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
