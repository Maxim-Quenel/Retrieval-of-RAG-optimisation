import requests
import time

def test_pipeline():
    print("Testing pipeline initialization and evaluation...")
    
    # 1. Initialize
    res = requests.post("http://localhost:5000/api/init")
    print("Init response:", res.json())
    
    # 2. Reindex
    print("Reindexing (Chunk size = 100)...")
    res = requests.post("http://localhost:5000/api/reindex", json={"chunk_size": 100})
    print("Reindex response:", res.json())
    
    # 3. Search
    print("Testing search...")
    res = requests.post("http://localhost:5000/api/search", json={
        "query": "Quel est le sport le plus populaire en France ?",
        "top_k": 3,
        "rrf_k": 60
    })
    print("Search status:", res.json()["status"])
    print(f"Got {len(res.json().get('dense_results', []))} dense results")
    print(f"Got {len(res.json().get('sparse_results', []))} sparse results")
    print(f"Got {len(res.json().get('rrf_results', []))} rrf results")
    
    # 4. Evaluate
    print("Running evaluation...")
    res = requests.post("http://localhost:5000/api/evaluate", json={"top_k": 5, "rrf_k": 60})
    print("Evaluate response:", res.json())

if __name__ == "__main__":
    test_pipeline()
