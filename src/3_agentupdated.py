import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 1. Setup
load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")

# 2. Setup Vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Initialize LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)

# 4. MEMORY STORAGE
chat_history = [] 

# 5. ROUTER WITH MEMORY
# We add a placeholder for history so the router understands follow-up questions
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized router. Use the chat history to understand context. "
               "Categorize the latest user request as: 'GIT', 'MATH', or 'GENERAL'. "
               "Output ONLY the word."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_query}")
])

router_chain = router_prompt | llm | StrOutputParser()

# 6. THE AGENT LOGIC
def run_memory_agent(question):
    global chat_history
    print(f"\n--- User: {question} ---")
    
    # STEP 1: THINK (Routing with context)
    decision = router_chain.invoke({
        "history": chat_history,
        "user_query": question
    }).strip().upper()
    
    print(f"🤔 Thinking... (Context leads me to: {decision})")

    # STEP 2: ACT
    if "GIT" in decision:
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Generator prompt includes history
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a Git expert. Use the context and history to answer.\nContext: {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        gen_chain = gen_prompt | llm
        response = gen_chain.invoke({"history": chat_history, "input": question})
        
    elif "MATH" in decision:
        math_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise calculator."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        math_chain = math_prompt | llm
        response = math_chain.invoke({"history": chat_history, "input": question})
        
    else:
        general_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        general_chain = general_prompt | llm
        response = general_chain.invoke({"history": chat_history, "input": question})

    # STEP 3: UPDATE HISTORY
    # Store the exchange so the next turn remembers it
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))
    
    return response.content

# 7. INTERACTIVE CHAT LOOP
if __name__ == "__main__":
    print("Welcome to your Private Git AI! (Type 'exit' to stop)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        answer = run_memory_agent(user_input)
        print(f"🤖 AI: {answer}")