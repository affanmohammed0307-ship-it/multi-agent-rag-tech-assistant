import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Secure Setup
# This loads the key from your local api.env file. 
# GitHub will not see this file if it is in your .gitignore.
load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in api.env or system environment.")
    exit()

# 2. Setup Vector DB (Knowledge Base: 1,138 Pages)
print("🧠 Loading Engineering Intelligence Base (Auto, Aero, EMC)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# 3. Initialize LLM (Llama 3.1 via Groq)
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)

# 4. Multi-Domain Router
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical intent router. Categorize the user query into exactly one: \n"
               "1. 'AUTO' - Automotive, Zonal Architecture, CAN Bus, or SDV.\n"
               "2. 'AERO' - Space Standards (ECSS), Cleanrooms, Satellite AIT, or Aerospace hardware.\n"
               "3. 'MATH' - Engineering calculations, unit conversions, or formulas.\n"
               "4. 'GENERAL' - Greetings or non-technical info.\n\n"
               "Output ONLY the word: AUTO, AERO, MATH, or GENERAL."),
    ("human", "{user_query}")
])

router_chain = router_prompt | llm | StrOutputParser()

# 5. Agent Processing Logic
def run_enhanced_agent(question):
    # Step A: Intent Classification
    decision = router_chain.invoke({"user_query": question}).strip().upper()
    print(f"🤔 Decision: Using {decision} Engine.")

    # Step B: Domain-Specific Execution
    if decision in ["AUTO", "AERO"]:
        persona = "Senior Aerospace Systems Engineer" if decision == "AERO" else "Senior Automotive Systems Engineer"
        
        # Retrieval
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Context-Grounded Prompt
        final_prompt = f"""
        You are a {persona} specializing in high-precision validation. 
        Use the provided technical context to answer the question accurately.
        If the answer is not present in the documentation, say you do not know based on available data.
        
        Context:
        {context}
        
        Question: {question}
        
        Detailed Technical Answer:"""
        
        return llm.invoke(final_prompt).content

    elif "MATH" in decision:
        math_prompt = f"Act as a precise engineering calculator. Solve: {question}. Provide only the result and unit."
        return llm.invoke(math_prompt).content

    else:
        return llm.invoke(question).content

# 6. Interactive Testing Loop
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 ENGINEERING RAG AGENT: MULTI-DOMAIN (AERO/AUTO/EMC)")
    print("ECSS Space Standards & Automotive Architectures Integrated")
    print("Type 'exit' to quit.")
    print("="*60)

    while True:
        query = input("\n👤 Query: ")
        if query.lower() in ['exit', 'quit']:
            print("👋 Closing Agent.")
            break
        
        try:
            result = run_enhanced_agent(query)
            print(f"\n🤖 Agent Response:\n{result}")
            print("-" * 60)
        except Exception as e:
            print(f"❌ System Error: {e}")