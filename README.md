**Azure LLM – Mistral 7B Enterprise RAG System**

Overview

This project demonstrates a production-ready Retrieval-Augmented Generation (RAG) system built using:

1) Self-hosted Mistral-7B-Instruct
2) GPU-backed inference
3) Azure documentation corpus
4) Automated evaluation benchmarking

The system is designed to simulate enterprise-grade AI + Data Engineering architecture, combining:

1) Large Language Models
2) Document indexing
3) Vector search
4) Retrieval evaluation
5) Production accuracy validation

🧠 Architecture

Core Components

1) LLM: Mistral-7B-Instruct
2) Vector Store: ChromaDB
3) Embeddings: Sentence Transformers
4) Data Source: Azure official documentation PDFs
5) Evaluation Framework: Custom JSONL benchmark (7000+ queries)

CLI-based RAG interface

📂 Project Structure

Azure-LLM-Mistral-7b/</br>
│</br>
├── build_training_data.py              # Generate evaluation dataset</br>
├── index_rag_chroma.py                 # Index documents into vector DB</br>
├── evaluate_rag.py                     # Run retrieval accuracy benchmarks</br>
├── rag_chat_cli.py                    # Interactive RAG CLI</br>
├── eval_questions.jsonl                # Gold evaluation benchmark</br>
├── requirements.txt</br> 
├── PRODUCTION_ACCURACY_RUNBOOK.md</br> 
└── Data/                               # Source documentation corpus</br>

🔥 Key Features

✅ Self-hosted LLM inference</br></br>
✅ Enterprise-style RAG pipeline</br></br>
✅ 7,000+ evaluation benchmark questions</br></br>
✅ Retrieval scoring & accuracy metrics</br></br>
✅ Production validation runbook</br></br>
✅ Modular indexing pipeline</br></br>
✅ CLI-based conversational interface</br></br>

📊 Impact Metrics

🧠 Model Size: 7B parameters
📄 Documents Indexed: 14+ Azure service docs
🧪 Evaluation Dataset: ~7000 structured questions
⚡ Context Window: 32K tokens
🎯 Retrieval Accuracy Optimization via iterative tuning

🏗️ System Flow

1) Ingest Azure documentation PDFs
2) Chunk and embed documents
3) Store embeddings in ChromaDB
4) Retrieve top-K context for queries
5) Pass retrieved context to Mistral 7B
6) Evaluate retrieval quality using benchmark JSONL
7) Iterate for production optimization

🧪 Evaluation Framework

Unlike typical RAG demos, this project includes:

1) Expected source validation
2) Keyword-based scoring
3) Required fact matching
4) Retrieval ranking evaluation
5) Per-document accuracy analysis

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
