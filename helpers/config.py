import json


def get_models() -> dict[str, str]:
    """
    Returns a dictionary of models with their corresponding filenames.

    Returns:
        dict[str, str]: A dictionary mapping model names to their filenames.
    """

    with open("config.json", "r") as f:
        config = json.load(f)
        models = config["models"]

    return dict(models)


def set_model(model_name: str) -> None:
    """
    Sets the current model in the config file.

    Args:
        model_name (str): The name of the model to set as the current model.
    """
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in the config file.")

    with open("config.json", "r+") as f:
        config = json.load(f)
        config["current_model"] = model_name
        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()


def get_vector_store_name() -> str:
    """
    Returns the name of the vector store.

    Returns:
        str: The name of the vector store.
    """
    with open("config.json", "r") as f:
        config = json.load(f)
        return config["vectore_store"]
