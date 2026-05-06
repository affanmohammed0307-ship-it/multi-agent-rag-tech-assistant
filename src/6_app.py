import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Engineering Intelligence Assistant", 
    page_icon="🚀", 
    layout="wide"
)

# --- 2. TITLES & SIDEBAR ---
st.title("🚀 Multi-Domain Engineering Assistant (Auto / Aero / EMC)")

st.sidebar.markdown("### 👨‍💻 Developer Profile")
st.sidebar.info(
    "**Mohammed Affan**\n\n"
    "M.Sc. Mechatronics & Cyber-Physical Systems\n"
    "Deggendorf Institute of Technology (DIT)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Integrated Knowledge Base")
st.sidebar.write(
    "This RAG agent is architected for high-precision system engineering. "
    "Current Knowledge Base (1,138+ Pages):"
)
st.sidebar.caption("✅ **Aerospace:** ECSS Standards (Testing, Cleanrooms, Verification)")
st.sidebar.caption("✅ **Automotive:** Zonal Architecture & SDV (Vector, NXP, Bosch)")
st.sidebar.caption("✅ **EMC:** High-Frequency Parasitics & Simulation (Springer, TI, ST)")

# --- 3. SECURE API SETUP ---
load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    st.error("GROQ_API_KEY not found. Please set it in Secrets or api.env")
    st.stop()

# --- 4. LOAD RESOURCES (Cached) ---
@st.cache_resource
def init_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Points to the 1,138-page database including ECSS
    vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)
    return vector_db, llm

vector_db, llm = init_resources()
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# --- 5. USER INTERFACE ---
user_query = st.text_input(
    "Query the Engineering Intelligence Base:", 
    placeholder="e.g., Cleanroom protocols for optical instruments under ECSS..."
)

if user_query:
    with st.spinner("Analyzing cross-domain documentation..."):
        # STEP A: Strategic Routing (Updated for AERO)
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", "Categorize query as AUTO, AERO, MATH, or GENERAL.\n"
                       "AUTO: Cars, Zonal, CAN Bus.\n"
                       "AERO: Space, ECSS, Cleanrooms, AIT, Satellites.\n"
                       "Output ONLY the word."),
            ("human", "{user_query}")
        ])
        router_chain = router_prompt | llm | StrOutputParser()
        decision = router_chain.invoke({"user_query": user_query}).strip().upper()
        
        st.caption(f"⚡ Engine: {decision} System Activated")

        # STEP B: Execution
        if decision in ["AUTO", "AERO"]:
            persona = "Aerospace Systems Engineer" if decision == "AERO" else "Automotive Systems Engineer"
            docs = retriever.invoke(user_query)
            context = "\n\n".join([d.page_content for d in docs])
            
            final_prompt = f"""
            You are a Senior {persona}. Use the technical context to answer.
            Context: {context}
            Question: {user_query}
            Answer:"""
            
            response = llm.invoke(final_prompt).content
            st.markdown(f"### 🛠️ {persona} Analysis")
            st.write(response)
            
            with st.expander("🔍 View Source References"):
                st.write(context)
        
        elif "MATH" in decision:
            math_prompt = f"Solve precisely: {user_query}. Output only numeric answer and unit."
            response = llm.invoke(math_prompt).content
            st.success(f"📊 Result: {response}")
        
        else:
            response = llm.invoke(user_query).content
            st.write(response)

# --- 6. FOOTER ---
st.markdown("---")
st.caption("**Standards Compliant:** European Space Standards (ECSS), ISO-14644 (Cleanroom), and Automotive E/E Architectures.")