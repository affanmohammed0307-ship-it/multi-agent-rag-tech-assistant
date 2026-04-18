import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIG ---
st.set_page_config(page_title="Automotive E/E Assistant", page_icon="🏎️", layout="wide")

# --- TITLES & BIO ---
st.title("🏎️ Automotive E/E Innovation Assistant")
st.sidebar.markdown("### Developer Profile")
st.sidebar.info("**Mohammed Affan**\nM.Sc. Mechatronics & CPS\nDeggendorf Institute of Technology")
st.sidebar.markdown("---")
st.sidebar.markdown("This RAG agent uses Multi-Agent routing to analyze Zonal Architectures and SDV trends.")

# --- SECURE API SETUP ---
load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Please set it in Secrets or api.env")
    st.stop()

# --- LOAD RESOURCES (Cached for speed) ---
@st.cache_resource
def init_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)
    return vector_db, llm

vector_db, llm = init_resources()
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# --- USER INPUT ---
user_query = st.text_input("Enter your automotive technical query:", placeholder="e.g. Benefits of Zonal Architecture...")

if user_query:
    with st.spinner("Analyzing documentation..."):
        # 1. Routing Logic
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", "Categorize query as AUTO, MATH, or GENERAL. Output only the word."),
            ("human", "{user_query}")
        ])
        router_chain = router_prompt | llm | StrOutputParser()
        decision = router_chain.invoke({"user_query": user_query}).strip().upper()
        
        st.caption(f"Routing Decision: {decision} Engine activated.")

        # 2. Tool Execution
        if "AUTO" in decision:
            docs = retriever.invoke(user_query)
            context = "\n\n".join([d.page_content for d in docs])
            final_prompt = f"Using this context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
            response = llm.invoke(final_prompt).content
            
            st.markdown("### Technical Analysis")
            st.write(response)
            
            with st.expander("View Source Context"):
                st.info(context)
        
        elif "MATH" in decision:
            math_prompt = f"Solve precisely: {user_query}. Output only answer and unit."
            response = llm.invoke(math_prompt).content
            st.success(f"Calculation Result: {response}")
        
        else:
            response = llm.invoke(user_query).content
            st.write(response)

# --- FOOTER ---
st.markdown("---")
st.caption("Knowledge Base: Vector, NXP, and Molex E/E Architecture Whitepapers.")