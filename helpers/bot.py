from time import time
from colorama import Fore, Style
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.llms.gpt4all import GPT4All
from helpers.prompts import claude_qna_prompt_template, default
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


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
    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    print(
        Fore.CYAN
        + "Chatbot initialized successfully! type exit to quit."
        + Style.RESET_ALL
    )
    print("-" * 100)
    store = {}
    session_id = "default"

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    while True:
        query_text = input(Style.BRIGHT + Fore.BLUE + "Question: ")
        if query_text == "exit":
            break
        results = retriever.invoke(query_text)
        context_text = "\n---\n".join([doc.page_content for doc in results])
        qa_system_prompt = default.get_prompt()
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),

            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)
### Statefully manage chat history ###

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        start = time()
        response = conversational_rag_chain.invoke(
            {"input": query_text, "context": context_text},
            config={
                "configurable": {"session_id": session_id}
            },  # constructs a key "abc123" in `store`.
        )["answer"]
        formatted_response = f"Response: {response}"
        print(Style.BRIGHT + Fore.GREEN + formatted_response + Style.RESET_ALL)
        sources = [doc.metadata.get("source", None) for doc in results]
        formatted_sources = f"Sources: {sources}"
        print(Style.BRIGHT + Fore.YELLOW + formatted_sources + Style.RESET_ALL)

        print(Fore.LIGHTRED_EX +
              f"Time taken: {time() - start:.2f}s" + Style.RESET_ALL)
        print("-" * 100)

    print(Style.DIM + Fore.YELLOW + "Exiting bot..." + Style.RESET_ALL)
    return
