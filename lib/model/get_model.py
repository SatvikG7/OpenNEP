from langchain_community.llms.gpt4all import GPT4All
import torch


def get_model(local_path, device=None) -> GPT4All:
    """
    Retrieves a GPT4All model.

    Args:
        local_path (str): The path to the model.
        device (str, optional): The device to use for inference. Defaults to None.

    Returns:
        GPT4All: The GPT4All model.

    Raises:
        None

    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    return GPT4All(model=local_path, device=device, streaming=True)  # type: ignore
