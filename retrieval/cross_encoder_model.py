import torch
from FlagEmbedding import FlagReranker

class CrossEncoderModel:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        """
        Initialize the Cross-Encoder Reranker model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Cross-Encoder {model_name} on {device}...")
        # use_fp16=True saves memory and speeds up inference on modern GPUs
        self.reranker = FlagReranker(model_name, use_fp16=True, device=device)
        print("Cross-Encoder loaded.")

    def rerank(self, query, retrieved_results):
        """
        Re-rank a list of retrieved results for a given query.
        retrieved_results: list of dicts, where each dict has a "text" key.
        Returns a sorted list of dicts with an updated "score" and "rank".
        """
        if not retrieved_results:
            return []

        # Prepare pairs: [ [query, doc_text_1], [query, doc_text_2], ... ]
        pairs = [[query, res["text"]] for res in retrieved_results]
        
        # Compute scores
        # Note: batch_size can be adjusted based on VRAM
        scores = self.reranker.compute_score(pairs, batch_size=16)
        
        # If compute_score returns a single float (when len(pairs) == 1), wrap it
        if isinstance(scores, float):
            scores = [scores]

        # Attach scores and sort
        reranked = []
        for i, res in enumerate(retrieved_results):
            new_res = res.copy()
            new_res["score"] = float(scores[i])
            reranked.append(new_res)
            
        reranked.sort(key=lambda x: x["score"], reverse=True)
        
        # Assign new ranks
        for i, res in enumerate(reranked):
            res["rank"] = i + 1
            
        return reranked
