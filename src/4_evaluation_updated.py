import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)

# 1. THE AUDITOR (Strict Judge)
judge_system_prompt = (
    "You are a technical auditor. Evaluate if the AI is 'Faithful' to the Context.\n"
    "CRITICAL RULE: If the answer contains info NOT in the context, it must be penalized "
    "UNLESS the AI explicitly says it is using its own knowledge.\n"
    "Rate 1-5. Format: SCORE: [num] | REASON: [text]"
)

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", judge_system_prompt),
    ("human", "CONTEXT: {context}\nQUESTION: {question}\nANSWER: {answer}")
])
judge_chain = judge_prompt | llm | StrOutputParser()

# 2. THE HARD TEST SET
test_queries = [
    "How do I create a branch?",                   # Easy (Baseline)
    "What is the command to change the color of the git UI to 'neon pink'?", # Negative Test (Not in manual)
    "What does the '-S' flag do in 'git commit'?", # Hard Detail (GPG signing)
    "How do I bake a chocolate cake?"              # Total Out-of-Bounds Test
]

all_scores = []

print("🧪 Starting Stress Test Evaluation...\n")

for query in test_queries:
    print(f"Testing: {query}")
    
    # RAG Step
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    
    # Generate Answer
    gen_prompt = (
        "You are a Git assistant. Answer ONLY using the context below. "
        "If the answer isn't there, say 'I do not know based on the manual'.\n\n"
        f"Context: {context}\nQuestion: {query}"
    )
    answer = llm.invoke(gen_prompt).content
    
    # Evaluate
    evaluation = judge_chain.invoke({"question": query, "context": context, "answer": answer})
    
    # Extract Score for math
    try:
        score = int(evaluation.split("SCORE:")[1].split("|")[0].strip())
        all_scores.append(score)
    except:
        pass

    print(f"🤖 Answer: {answer[:100]}...")
    print(f"⚖️ Judge: {evaluation}")
    print("-" * 30)

# 3. STATISTICAL SUMMARY
if all_scores:
    avg_score = sum(all_scores) / len(all_scores)
    print(f"\n📊 SYSTEM HEALTH REPORT")
    print(f"Average Reliability Score: {avg_score:.2f} / 5.0")
    if avg_score < 4.0:
        print("⚠️ Warning: System is prone to hallucinations or missing data.")
    else:
        print("✅ System is robust.")