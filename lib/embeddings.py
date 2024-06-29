from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch


def initialize_embeddings_model(model_name, device="cpu") -> HuggingFaceEmbeddings:
    """
    Initializes and returns a HuggingFaceEmbeddings model.

    Args:
        model_name (str): The name of the Hugging Face model to initialize.
        device (str, optional): The device to use for the model. Defaults to "cpu".

    Returns:
        HuggingFaceEmbeddings: The initialized HuggingFaceEmbeddings model.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": device}
    )
    return embeddings
