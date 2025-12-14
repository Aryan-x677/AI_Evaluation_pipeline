import time
from evaluators import retrieve_top_k, evaluate_grounding, evaluate_answer_relevance, evaluate_answer_completeness
from embeddings import embedd_text, embedd_context
from loader import load_chat_json, load_context_json
from utils import get_latest_turn, get_context

def run_pipeline():
    timestart = time.time()

    chat_data = load_chat_json("data/chat.json")
    user_query, assistant_answer = get_latest_turn(chat_data)
    context_texts = get_context("data/context.json")

    context_embeddings = embedd_context(context_texts)
    query_embedding = embedd_text(user_query)
    answer_embedding = embedd_text(assistant_answer)

    top_contexts = retrieve_top_k(context_embs=context_embeddings, query_emb=query_embedding, k=3)
    grounding_result = evaluate_grounding(contexts_embs=context_embeddings, query_emb=query_embedding, threshold=0.6)
    relevance_result = evaluate_answer_relevance(answer_emb=answer_embedding, query_emb=query_embedding)
    completeness_result = evaluate_answer_completeness(answer_emb=answer_embedding, context_embs=context_embeddings, threshold=0.5)

    timeend = time.time()

    return {
        "top_contexts": top_contexts,
        "grounding_result": grounding_result,
        "relevance_result": relevance_result,
        "completeness_result": completeness_result,
        "latency": timeend - timestart*1000,
        "cost": (timeend - timestart) * 0.0004
    }

if __name__ == "__main__":
    results = run_pipeline()
    for key, value in results.items():
        print(f"{key}: {value}")