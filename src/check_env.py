try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("⏳ Loading embedding model... (this may take a moment)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ SUCCESS: Embeddings loaded correctly!")
except Exception as e:
    print(f"❌ STILL FAILING: {e}")