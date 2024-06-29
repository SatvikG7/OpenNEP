HUMAN_PROMPT = """I'm going to give you a document. Then I'm going to ask you a question about it. I'd like you to first write down exact quotes of parts of the document that would help answer the question, and then I'd like you to answer the question using facts from the quoted content. Here is the document:

<document>
{CONTEXT}
</document>

First, find the quotes from the document that are most relevant to answering the question, and then print them in numbered order. Quotes should be relatively short.

If there are no relevant quotes, write "No relevant quotes" instead.

Then, answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer. Don't say "According to Quote [1]" when answering. Instead make references to quotes relevant to each section of the answer solely by adding their bracketed numbers at the end of relevant sentences.

Here is the first question: {QUESTION}

If the question cannot be answered by the document, say so.

Answer the question immediately without preamble."""


PROMPT_TEMPLATE = """System: {SYSTEM_PROMPT}

Human: {HUMAN_PROMPT}

Assistant:"""


def get_prompt(
    question: str,
    system="You are a helpful assistant answering questions about India's National Education Policy.",
    context="",
) -> str:
    HUMAN = HUMAN_PROMPT.format(CONTEXT=context, QUESTION=question)

    return PROMPT_TEMPLATE.format(SYSTEM_PROMPT=system, HUMAN_PROMPT=HUMAN)
