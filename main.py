import argparse
from os import path
from time import time

import torch

from helpers.cli.init import init
from helpers.config import get_models

from lib.embeddings import initialize_embeddings_model

from lib.model.get_model import get_model
from lib.vector_store.init import get_vector_store
from lib.vector_store.retriever import get_retriever

from helpers import bot


def main():
    llm_model_name = init()
    embeddindgs_model = initialize_embeddings_model()
    vector_store = get_vector_store(embeddings=embeddindgs_model)
    retriever = get_retriever(vector_store=vector_store, search_type="similarity", k=3)
    local_path = "./models/" + get_models()[llm_model_name]
    print(torch.cuda.is_available())
    model = get_model(local_path=local_path, device="cuda")
    bot.run(llm=model, retriever=retriever)


if __name__ == "__main__":
    main()
