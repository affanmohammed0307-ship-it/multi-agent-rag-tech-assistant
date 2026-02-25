from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the Document
pdf_path = "manual.pdf"
print(f"Loading {pdf_path}...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split the Document into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"Split document into {len(chunks)} chunks.")

# 3. Load the completely free, local Embedding Model
print("Downloading/Loading Embedding Model... (This takes a minute the very first time)")
embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create and Save the Vector Database
print("Storing chunks in the Vector Database... (This might take 1-3 minutes)")
persist_directory = "chroma_db"

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"Success! Database saved locally to the '{persist_directory}' folder.")