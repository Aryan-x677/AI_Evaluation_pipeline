LLM Response Evaluation Pipeline

Overview:
This Python project implements an automated evaluation pipeline to assess AI assistant responses against user queries. It measures:

Response Relevance & Completeness – Does the answer address the query fully?

Hallucination / Factual Accuracy – Is the answer grounded in context?

Latency & Estimated Cost – Can the evaluation run efficiently in real-time?

The pipeline is designed to process conversation JSONs and context JSONs and produce structured evaluation metrics.

Features:
Top-k context retrieval using cosine similarity.

Grounded score computation to flag hallucinations.

Semantic relevance score between user query and assistant response.

Completeness score measuring how much the answer covers available context.

Latency tracking and estimated cost calculation.

Modular functions, easy to extend for future metrics.

Architecture
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


Explanation:
Grounding: Measures how well the answer is supported by context using cosine similarity. Flags hallucinations if the score falls below threshold.

Relevance: Evaluates semantic similarity between the answer and user query to ensure the response is on-topic.

Completeness: Checks if the answer sufficiently covers top-k relevant contexts.

Latency / Cost: Measures runtime and estimates API or computational cost.

Local Setup Instructions

Clone the repository:
git clone https://github.com/Aryan-x677/AI_Evaluation_pipeline


Create a virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies:
pip install sentence_transformers

Place your JSON files:
data/chat.json
data/context.json

Run the evaluation pipeline:
python pipeline.py

Design Choices & Rationale

Cosine similarity was chosen for deterministic, explainable evaluation of embeddings.

Separate modular functions for grounding, relevance, and completeness make the pipeline flexible and maintainable.

Top-k context retrieval ensures the answer is compared against the most relevant context, improving accuracy.

Threshold-based flags allow quick identification of hallucinations and incomplete answers.

PyTorch is used for efficient vector operations and can scale to large embeddings.

Why this design?

Deterministic, vector-based evaluation is fast and interpretable, unlike purely LLM-based scoring.

Modularity allows individual metrics to be extended or replaced as needed.

Using precomputed embeddings for context and caching query embeddings reduces repeated computation, lowering latency and cost.

Scalability & Efficiency

If the script is run at scale (millions of daily conversations):

Latency Minimization

Precompute and store context embeddings in a vector database to avoid recomputing each time.

Batch process user queries and context embeddings to utilize GPU parallelism.

Use lightweight or smaller embedding models for real-time inference.

Cost Efficiency

Minimize API calls by reusing embeddings.

Cache results for frequently asked queries.

Optionally, perform approximate nearest neighbor search (e.g., FAISS) for top-k context retrieval instead of brute-force similarity.

Real-Time Feasibility

All vector operations (cosine similarity, top-k retrieval) are optimized for batch execution with PyTorch / NumPy.

Modular architecture allows scaling horizontally by distributing queries across multiple worker processes.

This ensures the evaluation remains fast, low-cost, and scalable even under heavy load.

Example Output
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
