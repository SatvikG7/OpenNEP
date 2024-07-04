def get_prompt(
    question: str,
    context="",
) -> str:
    PROMPT_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:"""

    return PROMPT_TEMPLATE.format(context=context, question=question)
