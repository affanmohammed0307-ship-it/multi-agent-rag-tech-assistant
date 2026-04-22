Automotive E/E & EMC Intelligence Assistant 🏎️🤖
🎯 Project Overview
This project is a specialized Multi-Agent Retrieval-Augmented Generation (RAG) system designed to automate technical scouting and benchmarking in the automotive sector. It allows engineers to navigate over 800 pages of high-density technical whitepapers and research (including Vector Informatik, NXP, and Springer) to extract precise insights on Zonal Architectures, Software-Defined Vehicles (SDV), and Electromagnetic Compatibility (EMC).
As a Mechatronics student, I developed this tool to bridge the gap between high-level AI capabilities and the rigorous technical requirements of automotive system validation.
🚀 Key Technical Features
Multi-Agent Routing: Implemented an intelligent router that classifies user intent into domain-specific categories (Automotive E/E, Engineering Mathematics, or General Knowledge) to ensure the highest retrieval precision.
High-Density Knowledge Base: Ingests and indexes 800+ pages of specialized documentation using ChromaDB and HuggingFace Embeddings (all-MiniLM-L6-v2).
LLM-as-a-Judge Framework: Developed an evaluation logic to cross-reference AI-generated responses against official documentation, significantly reducing hallucinations in technical answers.
Zero-Latency Inference: Powered by Groq (Llama 3.1 8B) to provide near-instantaneous responses for real-time engineering research.
Streamlit Web Dashboard: A professional-grade UI for interactive data querying and source-context visualization.
🧠 Software Architecture (OOP Approach)
The system is built on modular, Object-Oriented principles:
Ingestion Layer: Partitioning of complex PDFs into 1000-character chunks with recursive overlap for context retention.
Retrieval Layer: Semantic search optimized with a k=4 retrieval parameter for deep technical analysis.
Agentic Workflow: A decision-making loop that evaluates query complexity before selecting the appropriate tool for response synthesis.
🛠️ Tech Stack
Languages: Python (Advanced OOP)
Frameworks: LangChain, Streamlit
Vector Database: ChromaDB
LLMs: Groq (Llama 3.1), HuggingFace
Data Source: Technical Whitepapers from Vector Informatik, NXP, Molex, STMicroelectronics, and Springer Research.
📂 Knowledge Base Highlights
Vector Informatik: Zonal E/E Architectures and SDV Trends.
NXP: Electronics Architectures for Software-Defined Vehicles.
Springer: Academic research on EMC Simulation and Parasitic Inductance.
TI/ST: Conducted Emissions (CE) and Isolation Technologies.
▶️ Setup & Installation
code
Bash
# Clone the repository
git clone https://github.com/affanmohammed0307-ship-it/multi-agent-rag-tech-assistant.git

# Install dependencies
pip install -r requirements.txt

# Run the Ingestion Engine (Process 800+ pages)
python 1_ingest.py

# Launch the Streamlit Dashboard
streamlit run 6_app.py
🔗 Live Demo
CLICK HERE TO VIEW THE LIVE ASSISTANT
Developed by Mohammed Affan | M.Sc. Mechatronics and Cyber-Physical Systems
