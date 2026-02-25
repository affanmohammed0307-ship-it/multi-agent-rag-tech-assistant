import os
from dotenv import load_dotenv
# Using the modern HuggingFace integration
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load the API Key from your specific file name
# Make sure "api.env" is in the same folder as this script
load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error: GROQ_API_KEY not found in api.env file.")
    exit()

# 2. Load the Vector Database
print("--- Loading Database ---")
# This model matches the one used during ingestion (Phase 0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to the existing chroma_db folder
vector_db = Chroma(
    persist_directory="chroma_db", 
    embedding_function=embeddings
)

# Create the retriever (fetches top 3 most relevant chunks)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Initialize the Groq LLM 
print("--- Initializing LLM ---")
# Updated model to Llama 3.1 (the previous 3.0 version was decommissioned)
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0,
    groq_api_key=api_key
)

# 4. Create the Prompt Template
system_prompt = (
    "You are an expert Git assistant. Use the following pieces of retrieved "
    "context to answer the question. If you don't know the answer or if it's "
    "not in the context, just say 'I do not know based on the manual'. "
    "Keep the answer concise and helpful.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Define the question
question = "How do I create a new branch and switch to it at the same time?"
print(f"\nQuestion: {question}")
print("Searching the manual...\n")

# ==========================================
# EXPLICIT RAG PIPELINE (Phase 1 Logic)
# ==========================================

# Step A: Retrieve the documents (The 'R' in RAG)
docs = retriever.invoke(question)

# Step B: Combine the document text together
# This joins the content of the 3 retrieved chunks into one string
context_text = "\n\n".join([doc.page_content for doc in docs])

# Step C: Create the LLM pipeline and run it (The 'G' in RAG)
generation_chain = prompt | llm | StrOutputParser()
answer = generation_chain.invoke({"context": context_text, "input": question})

# ==========================================

print("🤖 Answer:")
print(answer)

print("\n--- Sources Used ---")
for i, doc in enumerate(docs):
    # Print the source metadata (page number) and a snippet of the content
    page_num = doc.metadata.get('page', 'Unknown')
    print(f"\nSource {i+1} (Page {page_num}):")
    print(doc.page_content[:200] + "...")