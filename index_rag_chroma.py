#!/usr/bin/env python3
"""
Build a persistent ChromaDB index from rag_chunks.jsonl.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index RAG chunks into ChromaDB.")
    p.add_argument("--rag-file", default="output/rag_chunks.jsonl", help="Input chunk JSONL.")
    p.add_argument("--persist-dir", default="output/chroma_db", help="Chroma persistence folder.")
    p.add_argument("--collection", default="azure_docs", help="Chroma collection name.")
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name.",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for add_documents.")
    p.add_argument("--reset", action="store_true", help="Delete existing persisted DB before indexing.")
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc


def to_documents(records: Iterable[dict]) -> List[Document]:
    docs: List[Document] = []
    for rec in records:
        text = str(rec.get("text", "")).strip()
        if not text:
            continue
        md = rec.get("metadata", {})
        if not isinstance(md, dict):
            md = {}
        md = dict(md)
        md["chunk_id"] = rec.get("chunk_id", "")
        docs.append(Document(page_content=text, metadata=md))
    return docs


def batched(items: List[Document], batch_size: int) -> Iterable[List[Document]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> int:
    args = parse_args()
    rag_file = Path(args.rag_file)
    persist_dir = Path(args.persist_dir)

    if not rag_file.exists():
        print(f"RAG file not found: {rag_file}", file=sys.stderr)
        return 1

    if args.reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    records = list(iter_jsonl(rag_file))
    docs = to_documents(records)
    if not docs:
        print("No documents found in rag file. Build chunks first.", file=sys.stderr)
        return 1

    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    vectorstore = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    total = len(docs)
    for i, batch in enumerate(batched(docs, args.batch_size), start=1):
        vectorstore.add_documents(batch)
        done = min(i * args.batch_size, total)
        print(f"Indexed {done}/{total} chunks", flush=True)

    print(
        json.dumps(
            {
                "status": "ok",
                "collection": args.collection,
                "persist_dir": str(persist_dir.resolve()),
                "chunks_indexed": total,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
