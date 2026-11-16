import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

question = "What is the minimum particle collection efficiency of a HEPA filter?"

results = db.similarity_search(question, k=4)

print(f"\nQuestion: {question}\n")

for position, r in enumerate(results, start=1):
    filename = os.path.basename(r.metadata["source"])
    page = r.metadata["page_label"]
    print(f"  {position}.  page {page:>4}   {filename[:45]}")
