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
    page_title="Automotive E/E & EMC Assistant", 
    page_icon="🏎️", 
    layout="wide"
)

# --- 2. TITLES & SIDEBAR ---
st.title("🏎️ Automotive E/E & EMC Intelligence Assistant")

st.sidebar.markdown("### 👨‍💻 Developer Profile")
st.sidebar.info(
    "**Mohammed Affan**\n\n"
    "M.Sc. Mechatronics & Cyber-Physical Systems\n"
    "Deggendorf Institute of Technology (DIT)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Research Scope")
st.sidebar.write(
    "This RAG agent is optimized for **Automotive Innovation Scouting** and **EMC Thesis Research**. "
    "It analyzes over 800 pages of technical documentation including:"
)
st.sidebar.caption("- Zonal E/E Architectures (Vector, Molex)")
st.sidebar.caption("- Software-Defined Vehicles (NXP, Bosch)")
st.sidebar.caption("- EMC Simulation & Parasitics (Springer, TI, ST)")

# --- 3. SECURE API SETUP ---
load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

# Fallback for Streamlit Cloud Secrets
if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    st.error("GROQ_API_KEY not found. Please set it in Secrets or api.env")
    st.stop()

# --- 4. LOAD RESOURCES (Cached) ---
@st.cache_resource
def init_resources():
    # Use the same embedding model as your ingest script
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Points to your updated 837-page database
    vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)
    return vector_db, llm

vector_db, llm = init_resources()
retriever = vector_db.as_retriever(search_kwargs={"k": 4}) # Increased k for deeper research

# --- 5. USER INTERFACE ---
user_query = st.text_input(
    "Enter your technical query:", 
    placeholder="e.g., How do parasitic inductances impact conducted emissions?"
)

if user_query:
    with st.spinner("Analyzing high-density technical documentation..."):
        # STEP A: Strategic Routing
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical router. Categorize the query as AUTO, MATH, or GENERAL. "
                       "Use 'AUTO' for: Zonal Architecture, SDV, EMC, Parasitics, Power Electronics, "
                       "Conducted Emissions, and Microcontrollers. Output ONLY the word."),
            ("human", "{user_query}")
        ])
        router_chain = router_prompt | llm | StrOutputParser()
        decision = router_chain.invoke({"user_query": user_query}).strip().upper()
        
        st.caption(f"⚡ System Status: {decision} Engine activated.")

        # STEP B: Execution
        if "AUTO" in decision:
            # Retrieve relevant chunks
            docs = retriever.invoke(user_query)
            context = "\n\n".join([d.page_content for d in docs])
            
            # Specialized Engineering Prompt
            final_prompt = f"""
            You are a Senior Automotive Systems Engineer specializing in E/E Architecture and EMC.
            Use the following technical context to answer the user query accurately.
            If the answer is not in the context, state that the current documentation does not provide a specific answer.
            
            Context:
            {context}
            
            Question: {user_query}
            
            Detailed Technical Answer:"""
            
            response = llm.invoke(final_prompt).content
            
            st.markdown("### 🛠️ Technical Analysis")
            st.write(response)
            
            with st.expander("🔍 View Source References"):
                st.info("The following document segments were used to synthesize this answer:")
                st.write(context)
        
        elif "MATH" in decision:
            math_prompt = f"Act as a precise engineering calculator. Solve: {user_query}. Output only numeric answer and unit."
            response = llm.invoke(math_prompt).content
            st.success(f"📊 Calculation Result: {response}")
        
        else:
            response = llm.invoke(user_query).content
            st.write(response)

# --- 6. FOOTER ---
st.markdown("---")
st.caption(
    "**Knowledge Base:** Springer EMC Research, TI Isolation Technologies, Vector Zonal Whitepapers, "
    "NXP SDV Architectures, Molex Technical Briefs, and STMicroelectronics EMC Guides."
)