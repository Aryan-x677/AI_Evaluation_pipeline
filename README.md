# **LLM Response Evaluation Pipeline**

## **Overview**
This Python project implements an automated evaluation pipeline to assess **AI assistant responses** against user queries. It measures:

- **Response Relevance & Completeness** – Does the answer address the query fully?  
- **Hallucination / Factual Accuracy** – Is the answer grounded in context?  
- **Latency & Estimated Cost** – Can the evaluation run efficiently in real-time?  

The pipeline is designed to process **conversation JSONs** and **context JSONs** and produce structured evaluation metrics.

---

## **Features**
- **Top-k Context Retrieval** using cosine similarity.  
- **Grounded Score** computation to flag hallucinations.  
- **Semantic Relevance** score between user query and assistant response.  
- **Completeness Score** measuring how much the answer covers available context.  
- **Latency Tracking** and **Estimated Cost Calculation**.  
- Modular functions, easy to extend for future metrics.

---

## **Architecture**

User Message
↓
Load Conversation JSON (chat.json)
↓
Load Context JSON (context.json)
↓
Compute Embeddings
├─ Query Embedding
├─ Answer Embedding
└─ Context Embeddings
↓
Top-K Context Retrieval (cosine similarity)
↓
Evaluation Module
├─ Grounding & Hallucination
├─ Relevance
└─ Completeness
↓
Output JSON / Console Report

**Explanation:**

- **Grounding:** Measures how well the answer is supported by context using cosine similarity. Flags hallucinations if the score falls below threshold.  
- **Relevance:** Evaluates semantic similarity between the answer and user query to ensure the response is on-topic.  
- **Completeness:** Checks if the answer sufficiently covers top-k relevant contexts.  
- **Latency / Cost:** Measures runtime and estimates API or computational cost.

---

## **Local Setup Instructions**
1. **Clone the repository:**
```bash
git clone https://github.com/Aryan-x677/AI_Evaluation_pipeline
```

## **Create a virtual environment:**

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


## **Place your JSON files:**
data/chat.json
data/context.json

## **Run the evaluation pipeline:**
python pipeline.py


## **Design Choices & Rationale:**
Cosine similarity was chosen for deterministic, explainable evaluation of embeddings.

Separate modular functions for grounding, relevance, and completeness make the pipeline flexible and maintainable.

Top-k context retrieval ensures the answer is compared against the most relevant context, improving accuracy.

Threshold-based flags allow quick identification of hallucinations and incomplete answers.

PyTorch is used for efficient vector operations; it scales easily for large datasets.

## **Why this design?**

Deterministic, vector-based evaluation is fast and interpretable, unlike purely LLM-based scoring.

Modular design allows each metric to be extended or replaced independently.

Using precomputed embeddings for context and caching query embeddings reduces latency and cost.

## **Scalability & Efficiency**

If the script is run at scale (millions of daily conversations):

Latency Minimization

Precompute and store context embeddings to avoid recomputation.

Batch process queries and embeddings to leverage GPU parallelism.

Use lightweight embedding models for real-time inference.

Cost Efficiency

Minimize API calls by reusing embeddings.

Cache results for frequently asked queries.

Approximate nearest neighbor search (e.g., FAISS) can replace brute-force similarity.

Real-Time Feasibility

All vector operations (cosine similarity, top-k retrieval) are optimized with PyTorch.

Modular architecture allows horizontal scaling across multiple worker processes.

## **Example Output**
```
{
  "top_contexts": [
    {"text": "Context 1", "score": 0.91},
    {"text": "Context 2", "score": 0.88},
    {"text": "Context 3", "score": 0.84}
  ],
  "grounding_result": {"grounding_score": 0.91, "hallucinated": false},
  "relevance_result": {"relevance_score": 0.87, "relevant": true},
  "completeness_result": {"completeness_score": 0.89, "complete": true},
  "latency": 0.12,
  "estimated_cost": 0.000048
}
