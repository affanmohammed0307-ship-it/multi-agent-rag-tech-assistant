import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# We will increase K to 5 to see if "More Context" helps or hurts
retriever = vector_db.as_retriever(search_kwargs={"k": 5}) 
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)

# THE CHALLENGE: A question that requires connecting information from different places
complex_question = (
    "Explain the full workflow of resolving a merge conflict, "
    "from identifying the conflicted files to finalizing the commit."
)

print("🔍 Phase 4: Hunting for Silent Failures...")
print(f"Question: {complex_question}\n")

docs = retriever.invoke(complex_question)
context = "\n\n".join([d.page_content for d in docs])

answer = llm.invoke(f"Context: {context}\n\nQuestion: {complex_question}").content

print("🤖 AI Response:")
print(answer)

# --- MANUAL ERROR ANALYSIS ---
print("\n--- 📝 INVESTIGATION CHECKLIST ---")
print("1. Did it mention 'git status' to find files? (Search Check)")
print("2. Did it mention the '<<<<<<<', '=======', '>>>>>>>' markers? (Detail Check)")
print("3. Did it mention 'git add' to mark them as resolved? (Workflow Check)")
print("4. Did it mention 'git commit' to finish? (Conclusion Check)")