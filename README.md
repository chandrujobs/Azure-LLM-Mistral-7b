Azure LLM – Mistral 7B Enterprise RAG System

Overview

This project demonstrates a production-ready Retrieval-Augmented Generation (RAG) system built using:

Self-hosted Mistral-7B-Instruct

GPU-backed inference

Azure documentation corpus

Automated evaluation benchmarking

The system is designed to simulate enterprise-grade AI + Data Engineering architecture, combining:

Large Language Models

Document indexing

Vector search

Retrieval evaluation

Production accuracy validation

🧠 Architecture
Core Components

LLM: Mistral-7B-Instruct

Vector Store: ChromaDB

Embeddings: Sentence Transformers

Data Source: Azure official documentation PDFs

Evaluation Framework: Custom JSONL benchmark (7000+ queries)

CLI-based RAG interface

📂 Project Structure
Azure-LLM-Mistral-7b/
│
├── build_training_data.py        # Generate evaluation dataset
├── index_rag_chroma.py           # Index documents into vector DB
├── evaluate_rag.py               # Run retrieval accuracy benchmarks
├── rag_chat_cli.py               # Interactive RAG CLI
├── eval_questions.jsonl          # Gold evaluation benchmark
├── requirements.txt
├── PRODUCTION_ACCURACY_RUNBOOK.md
└── Data/                         # Source documentation corpus
🔥 Key Features

✅ Self-hosted LLM inference
✅ Enterprise-style RAG pipeline
✅ 7,000+ evaluation benchmark questions
✅ Retrieval scoring & accuracy metrics
✅ Production validation runbook
✅ Modular indexing pipeline
✅ CLI-based conversational interface

📊 Impact Metrics

🧠 Model Size: 7B parameters

📄 Documents Indexed: 14+ Azure service docs

🧪 Evaluation Dataset: ~7000 structured questions

⚡ Context Window: 32K tokens

🎯 Retrieval Accuracy Optimization via iterative tuning

🏗️ System Flow

Ingest Azure documentation PDFs

Chunk and embed documents

Store embeddings in ChromaDB

Retrieve top-K context for queries

Pass retrieved context to Mistral 7B

Evaluate retrieval quality using benchmark JSONL

Iterate for production optimization

🧪 Evaluation Framework

Unlike typical RAG demos, this project includes:

Expected source validation

Keyword-based scoring

Required fact matching

Retrieval ranking evaluation

Per-document accuracy analysis

This simulates how enterprise AI systems are validated before production deployment.

💻 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Index Documents
python index_rag_chroma.py
3️⃣ Run CLI Chat
python rag_chat_cli.py
4️⃣ Run Evaluation
python evaluate_rag.py
