from helpers.config import get_vector_store_name
from lib.embeddings import initialize_embeddings_model
from lib.vector_store.init import (
    check_vector_store,
    get_text_splitter,
    get_vector_store,
)
import argparse
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from os import path


def load_file(file_path) -> list[Document]:
    """
    Load and return a list of documents from the specified file.

    Args:
        file_path (str): The path to the file.

    Returns:
        list[Document]: A list of Document objects loaded from the file.
    """

    # check if file_path exists
    if not path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    # determine the loader based on the file extension
    if file_path.endswith(".csv"):
        loader = CSVLoader(
            file_path,
            csv_args={
                "delimiter": ",",
                "quotechar": '"',
                "fieldnames": ["input", "output"],
            },
        )
    elif file_path.endswith(".pdf"):
        loader = UnstructuredPDFLoader(file_path, strategy="fast")
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format for file {file_path}")

    docs = loader.load()
    return docs


# parse the file path from the command line and add to vector store
def main():
    parser = argparse.ArgumentParser(description="Add a document to the vector store")
    parser.add_argument(
        "file_path", type=str, help="The path to the file to add to the vector store"
    )
    args = parser.parse_args()
    file_path = args.file_path

    # initialize the embeddings model
    embeddings_model = initialize_embeddings_model(
        "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )

    # check if vector store exists
    if not check_vector_store():
        print("Vector store does not exist. Please create it first.")
        return

    # get the vector store
    vector_store = get_vector_store(embeddings=embeddings_model)

    text_splitter = get_text_splitter()

    # load the documents from the file
    docs = load_file(file_path)

    # split the documents
    documents = text_splitter.split_documents(docs)

    # add the documents to the vector store
    vector_store.add_documents(documents)
    print(f"Documents from {file_path} added to the vector store.")
    vector_store.save_local(get_vector_store_name())


if __name__ == "__main__":
    main()
