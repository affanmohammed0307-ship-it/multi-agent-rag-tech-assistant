# Multi-Agent RAG Technical Assistant (Pro Git)

An advanced Retrieval-Augmented Generation (RAG) system that provides grounded, expert-level technical support using the Pro Git manual.

## 🚀 Features

- Multi-agent routing for dynamic tool selection  
- LangChain + ChromaDB retrieval pipeline  
- Groq (Llama 3.1) powered responses  
- Multi-turn conversational memory  
- Automated LLM evaluation framework  

## 🧠 Architecture

- Router Agent — classifies user intent  
- Retriever Tool — semantic document search  
- Math Tool — computation engine  
- General Agent — fallback knowledge  
- LLM-as-a-Judge — evaluates response quality  

## ⚙️ Tech Stack

Python, LangChain, ChromaDB, Groq (Llama 3.1)

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/agent.py
