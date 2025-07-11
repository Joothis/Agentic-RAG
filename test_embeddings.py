from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("HuggingFaceEmbeddings initialized successfully!")
except Exception as e:
    print(f"Error initializing HuggingFaceEmbeddings: {e}")