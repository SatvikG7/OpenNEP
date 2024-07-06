from time import time
from colorama import Fore, Style
from langchain_core.vectorstores import VectorStoreRetriever
from helpers.prompts import default
from langchain_community.llms.gpt4all import GPT4All

def run(llm: GPT4All, retriever: VectorStoreRetriever) -> None:
    """
    Run the bot by continuously taking user input, retrieving relevant documents,
    generating a prompt, and invoking the language model to get a response.

    Args:
        llm: The language model used for generating responses.
        retriever: The retriever used for retrieving relevant documents.

    Returns:
        None
    """

    print(
        Fore.CYAN
        + "Chatbot initialized successfully! type exit to quit."
        + Style.RESET_ALL
    )
    print("-" * 100)
    while True:
        query_text = input(Style.BRIGHT + Fore.BLUE + "Question: ")
        if query_text == "exit":
            break

        start = time()
        results = retriever.invoke(query_text)
        context_text = "\n---\n".join([doc.page_content for doc in results])
        prompt = default.get_prompt(context=context_text, question=query_text)
        response = llm.invoke(prompt)

        formatted_response = f"Response: {response}"
        print(Style.BRIGHT + Fore.GREEN + formatted_response)


        sources = [doc.metadata.get("source", None) for doc in results]
        formatted_sources = f"Sources: {sources}"
        print(Style.BRIGHT + Fore.YELLOW + formatted_sources + Style.RESET_ALL)

        print(Fore.LIGHTRED_EX + f"Time taken: {time() - start:.2f}s" + Style.RESET_ALL)
        print("-" * 100)

    print(Style.DIM + Fore.YELLOW + "Exiting bot..." + Style.RESET_ALL)
    return
