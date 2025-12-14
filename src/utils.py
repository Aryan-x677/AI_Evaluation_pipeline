from loader import load_chat_json, load_context_json
from embeddings import embedd_text, embedd_context
from evaluators import retrieve_top_k

def get_latest_turn(chat_data):
    messages = chat_data.get("conversation_turns", [])

    user_query = None
    assistant_answer = None

    for msg in reversed(messages):
        role = msg.get("role")

        if role == "AI/Chatbot" and assistant_answer is None:
            assistant_answer = msg.get("message")
        elif role == "User" and assistant_answer is not None:
            user_query = msg.get("message")
            break

    return user_query, assistant_answer

    
def get_context(context_data_path: str):
    context_data = load_context_json(context_data_path)
    contexts = []

    data = context_data.get("data", [])
    vector_data = data.get("vector_data", [])

    for item in vector_data:
        text = item.get("text", "")
        if text.strip():
            contexts.append(text)

    return contexts


'''
data = load_chat_json("data/chat.json")
user_query, assistant_answer = get_latest_turn(data)
context_texts = get_context("data/context.json")

context_embeddings = embedd_context(context_texts)

query = "What is the cost of IVF treatment?"
query_embedding = embedd_text(query)

top_contexts = retrieve_top_k(context_embs=context_embeddings, query_emb=query_embedding, k=3)
'''