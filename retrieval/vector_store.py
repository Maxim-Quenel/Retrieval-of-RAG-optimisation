import chromadb
import uuid
import json
import os
import numpy as np
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_directory="./chroma_db", sparse_index_path="./sparse_index.json"):
        self.persist_directory = persist_directory
        self.sparse_index_path = sparse_index_path
        
        # Initialize ChromaDB for Dense vectors
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {
            "default": self.client.get_or_create_collection(name="piaf_rag_collection_default", metadata={"hnsw:space": "cosine"}),
            "with_titles": self.client.get_or_create_collection(name="piaf_rag_collection_with_titles", metadata={"hnsw:space": "cosine"})
        }
        
        # Initialize Sparse Index in memory
        self.sparse_index = {"default": {}, "with_titles": {}}
        self.chunk_metadata = {"default": {}, "with_titles": {}}
        self._load_sparse_index()

    def _load_sparse_index(self):
        if os.path.exists(self.sparse_index_path):
            with open(self.sparse_index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.sparse_index = data.get("sparse_index", {"default": {}, "with_titles": {}})
                self.chunk_metadata = data.get("chunk_metadata", {"default": {}, "with_titles": {}})

    def _save_sparse_index(self):
        with open(self.sparse_index_path, 'w', encoding='utf-8') as f:
            json.dump({
                "sparse_index": self.sparse_index,
                "chunk_metadata": self.chunk_metadata
            }, f)

    def clear_all(self, namespace="default"):
        """Clear the ChromaDB collection and the sparse index for a given namespace."""
        collection_name = f"piaf_rag_collection_{namespace}"
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        self.collections[namespace] = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.sparse_index[namespace] = {}
        self.chunk_metadata[namespace] = {}
        self._save_sparse_index()

    def add_chunks(self, chunks, dense_vecs, sparse_vecs, ids=None, namespace="default"):
        """Add chunks and their embeddings to the stores."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Format for ChromaDB
        metadatas = [{"text": chunk} for chunk in chunks]
        
        # Add Dense to ChromaDB in batches to avoid max batch size error (5461)
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            self.collections[namespace].add(
                ids=ids[i:i+batch_size],
                embeddings=dense_vecs[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                documents=chunks[i:i+batch_size]
            )
        
        # Add Sparse to local index
        for i, doc_id in enumerate(ids):
            # Format sparse vector keys to strings if they aren't
            formatted_sparse = {str(k): float(v) for k, v in sparse_vecs[i].items()}
            self.sparse_index[namespace][doc_id] = formatted_sparse
            self.chunk_metadata[namespace][doc_id] = chunks[i]
            
        self._save_sparse_index()

    def search_dense(self, query_dense_vec, top_k=5, namespace="default", expand_adjacent=False):
        """Search in ChromaDB using dense vectors."""
        results = self.collections[namespace].query(
            query_embeddings=[query_dense_vec],
            n_results=top_k
        )
        
        retrieved = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                score = results['distances'][0][i] # It's a distance, so lower is better (usually 1 - cosine_similarity)
                # Convert distance to similarity for consistent ranking
                sim = 1.0 - score
                text = results['documents'][0][i]
                retrieved.append({"id": doc_id, "text": text, "score": float(sim), "rank": i + 1})
                
        if expand_adjacent:
            expanded_retrieved = []
            for res in retrieved:
                doc_id = res["id"]
                if doc_id.startswith("doc_") and "_chunk_" in doc_id:
                    try:
                        prefix = doc_id.split("_chunk_")[0]
                        chunk_idx = int(doc_id.split("_chunk_")[1])
                        
                        prev_id = f"{prefix}_chunk_{chunk_idx - 1}"
                        next_id = f"{prefix}_chunk_{chunk_idx + 1}"
                        
                        adj_results = self.collections[namespace].get(ids=[prev_id, next_id])
                        
                        prev_text = ""
                        next_text = ""
                        if adj_results and adj_results['ids']:
                            for idx, adj_id in enumerate(adj_results['ids']):
                                if adj_id == prev_id:
                                    prev_text = adj_results['documents'][idx]
                                elif adj_id == next_id:
                                    next_text = adj_results['documents'][idx]
                                    
                        new_text = f"{prev_text} {res['text']} {next_text}".strip()
                        # Clean up double spaces if any
                        new_text = " ".join(new_text.split())
                        
                        new_res = res.copy()
                        new_res['text'] = new_text
                        expanded_retrieved.append(new_res)
                    except Exception as e:
                        expanded_retrieved.append(res)
                else:
                    expanded_retrieved.append(res)
            return expanded_retrieved
            
        return retrieved

    def search_sparse(self, query_sparse_vec, compute_lexical_matching_score_fn, top_k=5, namespace="default", expand_adjacent=False):
        """Search using the in-memory sparse index."""
        scores = []
        for doc_id, doc_sparse_vec in self.sparse_index[namespace].items():
            # FlagEmbedding has a function to compute this: compute_lexical_matching_score
            # doc_sparse_vec has string keys, convert back if necessary, but FlagEmbedding handles it or we do it manually.
            score = compute_lexical_matching_score_fn(query_sparse_vec, doc_sparse_vec)
            scores.append((doc_id, float(score)))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        retrieved = []
        for i, (doc_id, score) in enumerate(scores[:top_k]):
            retrieved.append({
                "id": doc_id,
                "text": self.chunk_metadata[namespace][doc_id],
                "score": score,
                "rank": i + 1
            })
            
        if expand_adjacent:
            for res in retrieved:
                doc_id = res["id"]
                if doc_id.startswith("doc_") and "_chunk_" in doc_id:
                    try:
                        prefix = doc_id.split("_chunk_")[0]
                        chunk_idx = int(doc_id.split("_chunk_")[1])
                        
                        prev_id = f"{prefix}_chunk_{chunk_idx - 1}"
                        next_id = f"{prefix}_chunk_{chunk_idx + 1}"
                        
                        prev_text = self.chunk_metadata[namespace].get(prev_id, "")
                        next_text = self.chunk_metadata[namespace].get(next_id, "")
                        
                        new_text = f"{prev_text} {res['text']} {next_text}".strip()
                        new_text = " ".join(new_text.split())
                        res['text'] = new_text
                    except Exception as e:
                        pass
                        
        return retrieved

    def rrf_fusion(self, dense_results, sparse_results, k=60, top_k=5):
        """
        Reciprocal Rank Fusion (RRF).
        RRF_score(d) = sum(1 / (k + rank(d)))
        """
        rrf_scores = {}
        
        # Process dense results
        for res in dense_results:
            doc_id = res["id"]
            rank = res["rank"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0.0, "text": res["text"]}
            rrf_scores[doc_id]["score"] += 1.0 / (k + rank)
            
        # Process sparse results
        for res in sparse_results:
            doc_id = res["id"]
            rank = res["rank"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0.0, "text": res["text"]}
            rrf_scores[doc_id]["score"] += 1.0 / (k + rank)
            
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        retrieved = []
        for i, (doc_id, data) in enumerate(sorted_results[:top_k]):
            retrieved.append({
                "id": doc_id,
                "text": data["text"],
                "score": float(data["score"]),
                "rank": i + 1
            })
        return retrieved
