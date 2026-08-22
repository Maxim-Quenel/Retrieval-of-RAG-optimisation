import math

def calculate_recall_at_k(retrieved_results, ground_truth_answer, k):
    """
    Calculate Recall@K.
    retrieved_results: list of dicts with 'text'
    ground_truth_answer: string
    k: int
    """
    top_k = retrieved_results[:k]
    # Simple substring matching for the answer in the retrieved chunks
    for res in top_k:
        if ground_truth_answer.lower() in res["text"].lower():
            return 1.0
    return 0.0

def calculate_ndcg_at_k(retrieved_results, ground_truth_answer, k):
    """
    Calculate NDCG@K.
    For question answering, relevance is often binary (1 if answer in chunk, 0 otherwise).
    DCG = sum(rel_i / log2(i + 1))
    IDCG = 1.0 (since max 1 relevant chunk for a single exact answer, usually)
    """
    top_k = retrieved_results[:k]
    dcg = 0.0
    idcg = 1.0 # Max possible DCG for binary relevance where there's at least one relevant document
    
    for i, res in enumerate(top_k):
        if ground_truth_answer.lower() in res["text"].lower():
            rel = 1.0
            dcg += rel / math.log2((i + 1) + 1)
            # Stop after finding the first one if we assume binary relevance or add all up
            # In QA context, we usually just care about the highest ranked correct answer
            break
            
    return dcg / idcg

def calculate_mrr_at_k(retrieved_results, ground_truth_answer, k):
    """
    Calculate Mean Reciprocal Rank (MRR) at K.
    """
    top_k = retrieved_results[:k]
    for i, res in enumerate(top_k):
        if ground_truth_answer.lower() in res["text"].lower():
            return 1.0 / (i + 1)
    return 0.0
