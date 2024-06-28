from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def initialize_embeddings_model() -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings()
    return embeddings
