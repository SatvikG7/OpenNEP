def get_prompt(
    question: str,
    system="You are a OpenNEP Chatbot answering questions about India's National Education Policy.",
    context="",
) -> str:
    PROMPT_TEMPLATE = """
  <SYSTEM_PROMPT_START>
  {system}
  <SYSTEM_PROMPT_END>

  <CONTEXT_START>
  {context}
  <CONTEXT_END>

  <QUESTION_START>
  {question}
  <QUESTION_END>
  """

    return PROMPT_TEMPLATE.format(system=system, context=context, question=question)
