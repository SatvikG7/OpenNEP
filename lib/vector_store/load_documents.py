from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader


def load_documents() -> list[Document]:
    """
    Load and return a list of documents from the specified directory.

    Returns:
        list[Document]: A list of Document objects loaded from the directory.
    """
    loader = DirectoryLoader("./data/", show_progress=True, use_multithreading=True, max_concurrency=12)
    docs = loader.load()
    return docs
