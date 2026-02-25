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

# 2. Setup Vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Initialize LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)

# 4. UPDATED ROUTER: Now with 3 categories
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized router. Categorize the user request into exactly one of these labels:\n"
               "1. 'GIT' - Questions about Git commands, branching, or version control.\n"
               "2. 'MATH' - Questions involving calculations, arithmetic, or percentages.\n"
               "3. 'GENERAL' - Greetings or general knowledge unrelated to Git or Math.\n\n"
               "Output ONLY the word: GIT, MATH, or GENERAL."),
    ("human", "{user_query}")
])

router_chain = router_prompt | llm | StrOutputParser()

# 5. THE AGENT LOGIC
def run_enhanced_agent(question):
    print(f"\n--- Processing: {question} ---")
    
    # STEP 1: THINK (Routing)
    decision = router_chain.invoke({"user_query": question}).strip().upper()
    print(f"🤔 Agent Decision: Using the {decision} tool.")

    # STEP 2: ACT (Tool Execution)
    if "GIT" in decision:
        print("🔍 Tool: Git Manual Search")
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        final_prompt = f"Using this Git manual context:\n{context}\n\nAnswer: {question}"
        return llm.invoke(final_prompt).content

    elif "MATH" in decision:
        print("🔢 Tool: Calculator/Math Engine")
        # For this tool, we tell the LLM to be a strict calculator
        math_prompt = f"Act as a precise calculator. Solve this math problem: {question}. Provide only the answer."
        return llm.invoke(math_prompt).content

    else:
        print("💡 Tool: General Knowledge")
        return llm.invoke(question).content

# --- Step 2: Stress Test the Router ---
# We will run questions that might confuse the agent to see if it holds up.

test_queries = [
    "How do I create a branch?",             # Clear GIT
    "What is 15 percent of 200?",            # Clear MATH
    "Who is Linus Torvalds?",                # GENERAL (Related to Git, but a bio question)
    "If I have 3 branches and I delete 1, how many are left?", # TRICKY (Git + Math)
    "What is the capital of France?"         # Clear GENERAL
]

if __name__ == "__main__":
    for query in test_queries:
        answer = run_enhanced_agent(query)
        print(f"🤖 Agent Result: {answer}")
        print("-" * 50)