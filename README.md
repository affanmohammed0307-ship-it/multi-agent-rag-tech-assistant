Multi-Domain Engineering Intelligence Assistant (Auto / Aero / EMC) 🚀🤖
🎯 Project Overview
This is a high-precision Retrieval-Augmented Generation (RAG) system architected for complex systems engineering. It leverages a Multi-Agent Routing logic to provide grounded technical insights across three critical domains: Aerospace Standards, Automotive Architectures, and Electromagnetic Compatibility (EMC).
Developed as part of my Master’s specialization in Mechatronics at DIT Deggendorf, this tool bridges the gap between massive technical document repositories (1,138+ pages) and actionable engineering intelligence.
🚀 Key Technical Features
Multi-Agent Intent Routing: An intelligent decision-making layer that classifies user queries into specialized domain engines (AERO, AUTO, or MATH) to ensure zero-hallucination retrieval.
Aerospace Compliance (ECSS): Specifically pre-loaded with European Space Standards (ECSS) covering Space Engineering Testing, Cleanroom Particulate Control, and System Verification.
High-Density Knowledge Base: Ingests and indexes over 1,100 pages of specialized documentation from Airbus-relevant sources (ECSS, ISO), Vector Informatik, and Springer Research.
LLM-as-a-Judge Framework: Implements a secondary evaluation loop to verify the technical accuracy of generated responses against indexed standards.
Streamlit Web Interface: A professional dashboard designed for real-time technology scouting and benchmarking.
🧠 System Architecture
Data Ingestion Engine: Recursive character splitting with context-overlap optimized for technical manuals.
Vector Space: High-dimensional semantic indexing using ChromaDB and HuggingFace Embeddings.
Inference Layer: Near-instantaneous response generation powered by Groq (Llama 3.1 8B).
🛠️ Integrated Tech Stack
Languages: Python (Advanced OOP)
AI Frameworks: LangChain, HuggingFace
Vector Database: ChromaDB
Inference: Groq API
Deployment: Streamlit Cloud
📂 Featured Knowledge Base
Domain	Key Documents & Standards
Aerospace (AERO)	ECSS-E-ST-10-03C (Testing), ECSS-Q-ST-70-01C (Cleanrooms), ECSS-E-ST-10-02C (Verification)
Automotive (AUTO)	Vector Zonal Architecture, NXP Software-Defined Vehicle (SDV), Molex E/E Trends
Physics/EMC	Springer EMC Simulation Research, TI Isolation Technologies, ST Microelectronics Guides
▶️ Setup & Usage
code
Bash
# Clone the repository
git clone https://github.com/affanmohammed0307-ship-it/multi-agent-rag-tech-assistant.git

# Install high-precision dependencies
pip install -r requirements.txt

# Run the Ingestion Engine (Processes 1,100+ pages)
python 1_ingest.py

# Launch the Engineering Dashboard
streamlit run 6_app.py
🔗 Live Demo
CLICK HERE TO ACCESS THE LIVE ENGINEERING ASSISTANT
Developed by Mohammed Affan M | M.Sc. Mechatronics and Cyber-Physical Systems
