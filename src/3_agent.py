import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup API Key Securely
# It will look for api.env first, then standard system environment variables
load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found. Ensure it is set in your environment or api.env file.")
    exit()

# 2. Setup Vector DB
print("🧠 Loading Automotive Knowledge Base...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Initialize LLM (Llama 3.1)
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)

# 4. ROUTER: Categorizes questions to ensure the right 'tool' is used
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized Automotive E/E Architecture router. Categorize the user request into exactly one of these labels:\n"
               "1. 'AUTO' - Questions about Zonal Architecture, CAN Bus, Ethernet, ECUs, or Software-Defined Vehicles.\n"
               "2. 'MATH' - Questions involving engineering calculations, percentages, or frequencies.\n"
               "3. 'GENERAL' - Greetings or non-technical general knowledge.\n\n"
               "Output ONLY the word: AUTO, MATH, or GENERAL."),
    ("human", "{user_query}")
])

router_chain = router_prompt | llm | StrOutputParser()

# 5. AGENT LOGIC
def run_enhanced_agent(question):
    decision = router_chain.invoke({"user_query": question}).strip().upper()
    
    if "AUTO" in decision:
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        final_prompt = f"""
        You are a Senior Automotive Systems Engineer. Use the following technical context to answer the query.
        Context: {context}
        Question: {question}
        Answer:"""
        return llm.invoke(final_prompt).content

    elif "MATH" in decision:
        math_prompt = f"Act as a precise engineering calculator. Solve: {question}. Provide numeric answer and unit."
        return llm.invoke(math_prompt).content

    else:
        return llm.invoke(question).content
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 AUTOMOTIVE E/E & EMC INTELLIGENCE AGENT")
    print("Knowledge Base: 837 Pages (Springer, NXP, Vector, TI, etc.)")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)

    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Closing Agent. Good luck with your studies!")
            break
            
        if not user_input.strip():
            continue

        try:
            answer = run_enhanced_agent(user_input)
            print(f"\n🤖 Agent Result:\n{answer}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ An error occurred: {e}")