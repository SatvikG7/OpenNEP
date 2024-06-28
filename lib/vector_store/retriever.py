from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS

def get_retriever(vector_store: FAISS, search_type: str, k: int) -> VectorStoreRetriever:
    """
    Returns a retriever object that can be used to retrieve the k most similar items
    from the vector store.

    Args:
        vector_store (FAISS): The vector store to search.
        search_type (str): The type of search to perform. Must be one of ["similarity", "similarity_score_threshold", "mmr"].

        k (int): The number of most similar items to retrieve.

    Returns:
        VectorStoreRetriever: A retriever object that can be used to retrieve the k most
            similar items from the vector store.
    """
    if search_type not in ["similarity", "similarity_score_threshold", "mmr"]:
        raise ValueError(f"search_type must be one of  ['similarity', 'similarity_score_threshold', 'mmr'], got '{search_type}'")
    if k < 1:
        raise ValueError(f"k must be greater than 0, got {k}")

    return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
