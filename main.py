import argparse
from os import path
from time import time

from colorama import Fore

# import torch

from helpers.cli.init import init
from helpers.config import get_models

from lib.embeddings import initialize_embeddings_model

from lib.model.get_model import get_model
from lib.vector_store.init import get_vector_store
from lib.vector_store.retriever import get_retriever

from helpers import bot

import google.generativeai as genai


def main():
    # llm_model_name = init()
    embeddindgs_model = initialize_embeddings_model(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )  # sentence-transformers/multi-qa-mpnet-base-dot-v1, sentence-transformers/all-mpnet-base-v2, sentence-transformers/multi-qa-MiniLM-L6-cos-v1
    print(Fore.LIGHTGREEN_EX + "Embeddings model initialized." + Fore.RESET)

    vector_store = get_vector_store(embeddings=embeddindgs_model)
    print(Fore.LIGHTGREEN_EX + "Vector store initialized." + Fore.RESET)

    retriever = get_retriever(vector_store=vector_store, search_type="similarity", k=3)
    print(Fore.LIGHTGREEN_EX + "Retriever initialized." + Fore.RESET)

    # local_path = "./models/" + get_models()[llm_model_name]
    # model = get_model(local_path=local_path, device="cuda")

    genai.configure(
        api_key="AIzaSyCsxb8J7Dp2pV21hmgnw3_WPPh6KYS_rbM",
    )

    generation_config = genai.GenerationConfig(
        temperature=0.9,
        top_p=1,
        top_k=0,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="You are a helpful assistant OpenNEP Chatbot that answers questions about India's National Education Policy (NEP 2020). Do not say according to the source, just provide the information. If you are unsure, you can say I am not sure. If you are unable to answer, you can say I am unable to answer.",
    )

    print(Fore.LIGHTGREEN_EX + "Model initialized." + Fore.RESET)

    bot.run(llm=model, retriever=retriever)


if __name__ == "__main__":
    main()
