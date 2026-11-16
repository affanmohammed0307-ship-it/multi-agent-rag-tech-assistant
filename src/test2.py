import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Each test: the question, which file the answer is in, and which page.
tests = [
    {
        "question": "What is the minimum particle collection efficiency of a HEPA filter?",
        "file": "70-01",
        "page": "15",
    },
    {
        "question": "What equipment shall be used for air monitoring of cleanrooms of class ISO 8 or better?",
        "file": "70-01",
        "page": "37",
    },
    {
        "question": "What are the three stages of the ECSS verification process?",
        "file": "10-02C",
        "page": "15",
    },
    {
        "question": "In ECSS standards, what is the difference between the words shall and should?",
        "file": "10-02C",
        "page": "10",
    },
    {
        "question": "Approximately how many ECUs does a modern vehicle contain?",
        "file": "Zonal",
        "page": "3",
    },
]

for test in tests:
    print(f"\nQ: {test['question']}")
    print(f"   looking for: {test['file']}, page {test['page']}")

    results = db.similarity_search(test["question"], k=20)

    pairs = []
    for r in results:
        pairs.append((test["question"], r.page_content))

    print(f"   built {len(pairs)} pairs")
    scores = reranker.predict(pairs)

    for position, r in enumerate(results, start=1):
        filename = os.path.basename(r.metadata["source"])
        page = r.metadata["page_label"]

        # Is this the chunk we were hoping for?
        if test["file"] in filename and page == test["page"]:
            mark = "  <-- FOUND IT"
        else:
            mark = ""
        score = scores[position - 1]
        print(f"   {position}.  score {score:>7.2f}   page {page:>4}   {filename[:40]}{mark}")
        
