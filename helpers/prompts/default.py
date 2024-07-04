def get_prompt(
    question: str,
    system="You are a helpful assistant OpenNEP Chatbot that answers questions about India's National Education Policy (NEP 2020)",
    context="",
) -> str:
    PROMPT_TEMPLATE = """{system}

  Documents: {context}

  <QUESTION_START>
  {input}
  <QUESTION_END>
  """

    return PROMPT_TEMPLATE
