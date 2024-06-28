SYSTEM_PROMPT = """You are a helpful assistant."""
HUMAN_PROMPT = """I'm going to give you a document. Then I'm going to ask you a question about it. I'd like you to first write down exact quotes of parts of the document that would help answer the question, and then I'd like you to answer the question using facts from the quoted content. Here is the document:

<document>
{CONTEXT}
</document>
 
First, find the quotes from the document that are most relevant to answering the question, and then print them in numbered order. Quotes should be relatively short.
 
If there are no relevant quotes, write "No relevant quotes" instead.
 
Then, answer the question, starting with "Answer:". Do not include or reference quoted content verbatim in the answer. Don't say "According to Quote [1]" when answering. Instead make references to quotes relevant to each section of the answer solely by adding their bracketed numbers at the end of relevant sentences.
 
Thus, the format of your overall response should look like what's shown between the <example></example> tags. Make sure to follow the formatting and spacing exactly.
 
<example>
Relevant quotes:
[1] "Company X reported revenue of $12 million in 2021."
[2] "Almost 90% of revene came from widget sales, with gadget sales making up the remaining 10%."
 
Answer: 
Company X earned $12 million. [1]  Almost 90% of it was from widget sales. [2]
</example>
 
Here is the first question: {QUESTION}
 
If the question cannot be answered by the document, say so.
 
Answer the question immediately without preamble."""

PROMPT = """System: {SYSTEM_PROMPT}

Human: {HUMAN_PROMPT}
 
Assistant:"""
