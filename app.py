# create flask endpoint

from flask import Flask, request, jsonify

from lib.embeddings import initialize_embeddings_model
from lib.vector_store.init import get_vector_store
from lib.vector_store.retriever import get_retriever
from helpers.prompts import claude_qna_prompt_template, default

import google.generativeai as genai

app = Flask(__name__)


embeddindgs_model = initialize_embeddings_model(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
vector_store = get_vector_store(embeddings=embeddindgs_model)
retriever = get_retriever(vector_store=vector_store, search_type="similarity", k=3)

genai.configure(
    api_key="xxxxxxxxxxxxxxxxxxxxxxx",
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

chat_session = model.start_chat(history=[])

@app.route("/")
def index():
    return jsonify({"status": "ok"})

# post request to /ask endpoint with question in body to get answer from model
@app.route("/ask", methods=["POST"])
def ask():
    question = request.json["question"]  # type: ignore
    results = retriever.invoke(question)
    context_text = "\n---\n".join([doc.page_content for doc in results])
    prompt = default.get_prompt(question=question, context=context_text)
    # print(prompt)
    response = chat_session.send_message(prompt)
    sources = [doc.metadata.get("source", None) for doc in results]
    return jsonify({"response": response.text, "sources": sources})


if __name__ == "__main__":
    app.run(debug=False)
