**Azure LLM – Mistral 7B Enterprise RAG System**



<img width="1536" height="1024" alt="Azure LLM diagram" src="https://github.com/user-attachments/assets/fd227cc9-ed97-43f0-9585-e9208f6fc2b9" />

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

✅ Self-hosted LLM inference</br>
✅ Enterprise-style RAG pipeline</br>
✅ 7,000+ evaluation benchmark questions</br>
✅ Retrieval scoring & accuracy metrics</br>
✅ Production validation runbook</br>
✅ Modular indexing pipeline</br>
✅ CLI-based conversational interface</br>

📊 Impact Metrics

🧠 Model Size: 7B parameters</br>
📄 Documents Indexed: 14+ Azure service docs</br>
🧪 Evaluation Dataset: ~7000 structured questions</br>
⚡ Context Window: 32K tokens</br>
🎯 Retrieval Accuracy Optimization via iterative tuning</br>

🏗️ System Flow

1) Ingest Azure documentation PDFs</br>
2) Chunk and embed documents</br>
3) Store embeddings in ChromaDB</br>
4) Retrieve top-K context for queries</br>
5) Pass retrieved context to Mistral 7B</br>
6) Evaluate retrieval quality using benchmark JSONL</br>
7) Iterate for production optimization</br>

🧪 Evaluation Framework

Unlike typical RAG demos, this project includes:

1) Expected source validation</br>
2) Keyword-based scoring</br>
3) Required fact matching</br>
4) Retrieval ranking evaluation</br>
5) Per-document accuracy analysis</br>

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
