import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup
load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

# 2. Components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)

# 3. THE JUDGE'S RUBRIC
# This is the most important part of Phase 3.
judge_system_prompt = (
    "You are a strict technical auditor. You will evaluate an AI's response "
    "based ONLY on the provided Context. Rate the response from 1 to 5:\n"
    "1: Hallucination (Answer is not in context or is wrong)\n"
    "3: Partial (Answer is mostly right but missing details)\n"
    "5: Perfect (Answer is accurate and fully supported by context)\n\n"
    "Provide your output exactly in this format:\n"
    "SCORE: [number]\n"
    "REASON: [one sentence explanation]"
)

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", judge_system_prompt),
    ("human", "QUESTION: {question}\nCONTEXT: {context}\nANSWER: {answer}")
])

judge_chain = judge_prompt | llm | StrOutputParser()

# 4. TEST QUESTIONS (The "Benchmark")
test_queries = [
    "How do I create a new branch?",
    "What command shows the commit history?",
    "How do I delete a branch?"
]

# 5. EXECUTION LOOP
print("🔬 Starting Evaluation Suite...\n")

for query in test_queries:
    print(f"Testing Question: {query}")
    
    # Step A: Retrieve (R)
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    
    # Step B: Generate (G)
    gen_prompt = f"Use this context to answer. Context: {context}\n\nQuestion: {query}"
    answer = llm.invoke(gen_prompt).content
    
    # Step C: Evaluate (The Judge)
    evaluation = judge_chain.invoke({
        "question": query,
        "context": context,
        "answer": answer
    })
    
    print("-" * 20)
    print(f"🤖 AI Answer: {answer[:100]}...") # Show just the start
    print(f"⚖️ Judge Result:\n{evaluation}")
    print("-" * 20 + "\n")

print("Evaluation Complete.")