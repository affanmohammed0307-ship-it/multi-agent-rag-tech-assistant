"""
RAG Evaluation Harness
======================
Measures two things separately:
  1. RETRIEVAL  - did the search find the chunk that actually contains the answer?
  2. GENERATION - given that chunk, did the model answer correctly and refuse
                  when the documents cannot answer?

Run 1_ingest.py first so chroma_db/ exists.
"""

import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# 1. THE GOLDEN SET
# ---------------------------------------------------------------------------
# Every entry states what the right answer is and where it lives. Without this
# there is no way to tell a correct answer from a confident wrong one.
#
#   question        - what we ask
#   expected_source - filename fragment that should appear in the retrieved chunks
#   expected_page   - 1-based page number where the fact appears
#   must_contain    - strings that a correct ANSWER should include
#   answerable      - False means the documents cannot answer it; correct
#                     behaviour is refusal, not a plausible guess

GOLDEN_SET = [
    {
        "question": "What is the minimum particle collection efficiency of a HEPA filter?",
        "expected_source": "70-01",
        "expected_page": 15,
        "must_contain": ["99,97", "0,3"],
        "answerable": True,
    },
    {
        "question": "What equipment shall be used for air monitoring of cleanrooms of class ISO 8 or better?",
        "expected_source": "70-01",
        "expected_page": 37,
        "must_contain": ["light scattering"],
        "answerable": True,
    },
    {
        "question": "What are the three stages of the ECSS verification process?",
        "expected_source": "10-02C",
        "expected_page": 15,
        "must_contain": ["planning", "execution", "close-out"],
        "answerable": True,
    },
    {
        "question": "In ECSS standards, what is the difference between the words 'shall' and 'should'?",
        "expected_source": "10-02C",
        "expected_page": 10,
        "must_contain": ["requirement", "recommendation"],
        "answerable": True,
    },
    {
        "question": "Approximately how many ECUs does a modern vehicle contain?",
        "expected_source": "Zonal",
        "expected_page": 3,
        "must_contain": ["100", "150"],
        "answerable": True,
    },
    # --- negative tests: the documents genuinely do not cover these ---
    {
        "question": "What is the recommended torque setting for a Tesla Model 3 wheel nut?",
        "expected_source": None,
        "expected_page": None,
        "must_contain": [],
        "answerable": False,
    },
    {
        "question": "How do I create a branch in Git?",
        "expected_source": None,
        "expected_page": None,
        "must_contain": [],
        "answerable": False,
    },
]

# Phrases that indicate the model correctly declined to answer.
REFUSAL_MARKERS = [
    "do not know", "don't know", "not know based",
    "not contain", "no information", "cannot be answered",
    "not covered", "not provided", "unable to find",
]

# ---------------------------------------------------------------------------
# 2. SETUP
# ---------------------------------------------------------------------------
load_dotenv("api.env")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found in api.env")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

K = 4                       # how many chunks we retrieve per question
PAGE_TOLERANCE = 1          # allow the answer to sit one page either side

llm = ChatGroq(model_name="openai/gpt-oss-20b", groq_api_key=api_key)

ANSWER_PROMPT = (
    "You are a systems engineering assistant. Answer using ONLY the context below.\n"
    "If the context does not contain the answer, reply exactly: "
    "'I do not know based on the provided documents.'\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------
def retrieval_rank(docs, case):
    """
    Position (1-based) of the first retrieved chunk that matches the expected
    source file and page. Returns None if no retrieved chunk is the right one.
    """
    for i, d in enumerate(docs, start=1):
        source = os.path.basename(d.metadata.get("source", ""))
        page = d.metadata.get("page", -1) + 1        # pypdf pages are 0-based
        if case["expected_source"] in source and \
           abs(page - case["expected_page"]) <= PAGE_TOLERANCE:
            return i
    return None


def answer_contains_all(answer, required):
    """True if every required string appears in the answer (case-insensitive)."""
    low = answer.lower()
    return all(term.lower() in low for term in required)


def is_refusal(answer):
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# 4. RUN THE EVALUATION
# ---------------------------------------------------------------------------
hits = 0                # retrieval found the right chunk
reciprocal_ranks = []   # for MRR
correct_answers = 0     # answerable questions answered correctly
correct_refusals = 0    # unanswerable questions correctly refused
n_answerable = sum(1 for c in GOLDEN_SET if c["answerable"])
n_unanswerable = len(GOLDEN_SET) - n_answerable

print("=" * 70)
print("RAG EVALUATION")
print("=" * 70)

for case in GOLDEN_SET:
    q = case["question"]
    print(f"\nQ: {q}")

    docs = vector_db.similarity_search(q, k=K)
    context = "\n\n".join(d.page_content for d in docs)
    answer = llm.invoke(ANSWER_PROMPT.format(context=context, question=q)).content

    if case["answerable"]:
        # ---- retrieval ----
        rank = retrieval_rank(docs, case)
        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            print(f"   retrieval : HIT at position {rank}/{K}")
        else:
            reciprocal_ranks.append(0.0)
            found = {os.path.basename(d.metadata.get('source', '?'))[:28]
                     for d in docs}
            print(f"   retrieval : MISS  (expected {case['expected_source']} "
                  f"p{case['expected_page']}, got {sorted(found)})")

        # ---- generation ----
        if answer_contains_all(answer, case["must_contain"]):
            correct_answers += 1
            print("   answer    : CORRECT")
        else:
            missing = [t for t in case["must_contain"]
                       if t.lower() not in answer.lower()]
            print(f"   answer    : WRONG  (missing {missing})")
    else:
        # ---- negative test: refusal is the correct behaviour ----
        if is_refusal(answer):
            correct_refusals += 1
            print("   refusal   : CORRECT (declined, as it should)")
        else:
            print("   refusal   : FAILED  (answered a question it cannot know)")

    print(f"   said      : {re.sub(r'\s+', ' ', answer)[:150]}")

# ---------------------------------------------------------------------------
# 5. REPORT
# ---------------------------------------------------------------------------
recall = hits / n_answerable
mrr = sum(reciprocal_ranks) / n_answerable
accuracy = correct_answers / n_answerable
refusal_rate = correct_refusals / n_unanswerable if n_unanswerable else 1.0

print("\n" + "=" * 70)
print(f"Retrieval recall@{K} : {recall:.0%}   ({hits}/{n_answerable} found the right page)")
print(f"MRR               : {mrr:.2f}   (1.00 = always ranked first)")
print(f"Answer accuracy   : {accuracy:.0%}   ({correct_answers}/{n_answerable})")
print(f"Refusal rate      : {refusal_rate:.0%}   ({correct_refusals}/{n_unanswerable} correctly declined)")
print("=" * 70)

if recall < 0.8:
    print("\nRetrieval is the bottleneck. Fix search before touching prompts:")
    print("  - add BM25 hybrid search (exact codes like ECSS-Q-ST-70-01C)")
    print("  - add a cross-encoder reranker")
    print("  - revisit chunk size for structured standards documents")
elif accuracy < recall:
    print("\nRetrieval works but generation is losing information.")
    print("  - increase k, or improve the answer prompt")
