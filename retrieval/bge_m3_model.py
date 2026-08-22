import torch
from FlagEmbedding import BGEM3FlagModel

class BGEM3EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-m3"):
        """
        Initialize the BGE-M3 model.
        It generates both dense embeddings and sparse (lexical weight) representations.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {device}...")
        self.model = BGEM3FlagModel(model_name, use_fp16=True, device=device)
        print("Model loaded.")

    def encode(self, texts, batch_size=12, max_length=8192):
        """
        Encode a list of texts into dense and sparse representations.
        Returns a dictionary with 'dense_vecs' and 'lexical_weights'.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            max_length=max_length, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        return embeddings

    def compute_lexical_matching_score(self, lexical_weights_1, lexical_weights_2):
        """
        Compute the similarity between two sparse representations (lexical weights).
        """
        return self.model.compute_lexical_matching_score(lexical_weights_1, lexical_weights_2)

    def encode_colbert(self, texts, batch_size=12, max_length=8192):
        """
        Encode a list of texts and return their colbert vectors.
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            max_length=max_length, 
            return_dense=False, 
            return_sparse=False, 
            return_colbert_vecs=True
        )
        return embeddings['colbert_vecs']

    def compute_colbert_score(self, q_vec, p_vec):
        """
        Compute the similarity between two colbert representations.
        """
        return self.model.colbert_score(q_vec, p_vec)
