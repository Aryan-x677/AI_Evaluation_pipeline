import torch
import torch.nn.functional as F


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    return F.cosine_similarity(emb1, emb2, dim=0).item()

def retrieve_top_k(
    context_embs: torch.Tensor,
    query_emb: torch.Tensor,
    k: int = 3
) -> list[dict]:
    
    scored_contexts = []

    for item in context_embs:
        context_text = item['text']
        context_embedding = item['embedding']
        score = cosine_similarity(query_emb, context_embedding)
        scored_contexts.append({
            'text': context_text,
            'score': score
        })

    scored_contexts.sort(key=lambda x: x['score'], reverse=True)
    return scored_contexts[:k]

def evaluate_grounding(
    contexts_embs: torch.Tensor,
    query_emb: torch.Tensor,
    threshold: float = 0.6
) -> dict:
    hallucinated = True
    for item in contexts_embs:
        context_embedding = item['embedding']
        score = cosine_similarity(query_emb, context_embedding)
        if score >= threshold:
            hallucinated = False
            break

    return {
        'hallucinated': hallucinated,
        'grounding_score': score
    }

def evaluate_answer_relevance(
    answer_emb: torch.Tensor,
    query_emb: torch.Tensor
) -> dict:
    score = cosine_similarity(answer_emb, query_emb)
    relevant = score >= 0.5

    return {
        'relevant': relevant,
        'relevance_score': score
    }

def evaluate_answer_completeness(
    answer_emb: torch.Tensor,
    context_embs: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    max_score = 0.0
    for item in context_embs:
        context_embedding = item['embedding']
        score = cosine_similarity(answer_emb, context_embedding)
        if score > max_score:
            max_score = score

    complete = max_score >= threshold

    return {
        'complete': complete,
        'completeness_score': max_score
    }
