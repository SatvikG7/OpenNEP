from os import path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from helpers.config import get_vector_store_name
from lib.vector_store.load_documents import load_documents


VECTOR_STORE = get_vector_store_name()


def check_vector_store() -> bool:
    """
    Check if the vector store exists.

    Returns:
        bool: True if the vector store exists, False otherwise.
    """
    if path.exists(VECTOR_STORE):
        return True
    else:
        return False


def create_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    bulk = load_documents()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=200,
        chunk_overlap=20,
    )
    documents = text_splitter.split_documents(bulk)
    vector_store = FAISS.from_documents(embedding=embeddings, documents=documents)
    vector_store.save_local(VECTOR_STORE)
    return vector_store


def get_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    # check for the existence of the vector store
    if check_vector_store():
        # if the vector store exists, load it
        vector_store = FAISS.load_local(
            VECTOR_STORE, embeddings=embeddings, allow_dangerous_deserialization=True
        )
    else:
        # if the vector store does not exist, create it
        vector_store = create_vector_store(embeddings)

    return vector_store