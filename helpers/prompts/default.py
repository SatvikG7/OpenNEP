def get_prompt(
    question: str,
    system="You are a OpenNEP Chatbot answering questions about India's National Education Policy.",
    context="",
) -> str:
    PROMPT_TEMPLATE = """
  <SYSTEM_PROMPT_START>
  You are a OpenNEP Chatbot answering questions about India's National Education Policy.
  <SYSTEM_PROMPT_END>

  <CONTEXT_START>
  {context}
  <CONTEXT_END>

  <QUESTION_START>
  {input}
  <QUESTION_END>
  """

    return PROMPT_TEMPLATE
