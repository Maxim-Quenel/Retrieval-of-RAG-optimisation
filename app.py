from flask import Flask, request, jsonify, render_template
import json
import os
import glob
import pandas as pd
import time
import csv
from retrieval.bge_m3_model import BGEM3EmbeddingModel
from retrieval.cross_encoder_model import CrossEncoderModel
from retrieval.vector_store import VectorStore
from evaluation.metrics import calculate_recall_at_k, calculate_ndcg_at_k, calculate_mrr_at_k
from transformers import AutoTokenizer
import logging

# Suppress the max length warning from the tokenizer since we chunk manually
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Suppress /api/status logs from werkzeug to avoid breaking tqdm progress bars
class NoStatusFilter(logging.Filter):
    def filter(self, record):
        return 'GET /api/status' not in record.getMessage()
logging.getLogger('werkzeug').addFilter(NoStatusFilter())

app = Flask(__name__)

import threading

# Initialize Global Models
bge_model = None
vector_store = None
cross_encoder = None
model_lock = threading.Lock()

# Global pause event for evaluation
eval_pause_event = threading.Event()
eval_pause_event.set()

def init_models():
    global bge_model, vector_store, cross_encoder
    with model_lock:
        if bge_model is None:
            bge_model = BGEM3EmbeddingModel()
        if vector_store is None:
            vector_store = VectorStore()
        if cross_encoder is None:
            cross_encoder = CrossEncoderModel()

# Initialize tokenizer for token-based chunking
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def chunk_text(text, chunk_size=256, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    if not tokens:
        return chunks
        
    if overlap >= chunk_size:
        overlap = chunk_size - 1
        
    step = chunk_size - overlap
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
        
        if i + chunk_size >= len(tokens):
            break
            
    return chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def init_system():
    init_models()
    return jsonify({"status": "success", "message": "Models initialized."})

@app.route('/api/pause', methods=['POST'])
def pause():
    eval_pause_event.clear()
    state = load_eval_state()
    if state and state["status"] == "running":
        state["status"] = "paused"
        save_eval_state(state)
    return jsonify({"status": "paused", "message": "Évaluation en pause."})

@app.route('/api/reindex', methods=['POST'])
def reindex():
    init_models()
    data = request.json
    chunk_size = int(data.get('chunk_size', 256))
    chunk_overlap = int(data.get('chunk_overlap', 50))
    data_source = data.get('data_source', 'wikipedia')
    
    all_chunks_default = []
    all_chunks_titles = []
    
    all_ids_default = []
    all_ids_titles = []
    
    import uuid
    
    if data_source == 'piaf':
        # Load PIAF
        try:
            piaf_path = os.path.join(os.path.dirname(__file__), '..', 'piaf-v1.2.json')
            with open(piaf_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                piaf_data = raw_data.get("data", [])
            for item in piaf_data:
                title = item.get('title', '')
                doc_idx = 0
                for paragraph in item['paragraphs']:
                    doc_uuid = str(uuid.uuid4())
                    context = paragraph['context']
                    chunks = chunk_text(context, chunk_size=chunk_size, overlap=chunk_overlap)
                    all_chunks_default.extend(chunks)
                    for i in range(len(chunks)):
                        all_chunks_titles.append(f"Titre : {title}. {chunks[i]}")
                        all_ids_default.append(f"doc_{doc_uuid}_chunk_{i}")
                        all_ids_titles.append(f"doc_{doc_uuid}_chunk_{i}")
                    doc_idx += 1
        except Exception as e:
            return jsonify({"status": "error", "message": f"Could not load PIAF data: {str(e)}"}), 500
    else:
        # Load Wikipedia
        wiki_dir = os.path.join(os.path.dirname(__file__), 'data-wikipedia')
        try:
            wiki_files = glob.glob(os.path.join(wiki_dir, '*.json'))
            for file_path in wiki_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    wiki_data = json.load(f)
                    text = wiki_data.get('text', '')
                    title = wiki_data.get('title', '')
                    if text:
                        doc_uuid = str(uuid.uuid4())
                        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
                        all_chunks_default.extend(chunks)
                        for i in range(len(chunks)):
                            all_chunks_titles.append(f"Titre : {title}. {chunks[i]}")
                            all_ids_default.append(f"doc_{doc_uuid}_chunk_{i}")
                            all_ids_titles.append(f"doc_{doc_uuid}_chunk_{i}")
        except Exception as e:
            return jsonify({"status": "error", "message": f"Could not load Wikipedia data: {str(e)}"}), 500
        
    vector_store.clear_all(namespace="default")
    vector_store.clear_all(namespace="with_titles")
            
    if not all_chunks_default:
        return jsonify({"status": "error", "message": "No text chunks found to index."}), 400
        
    # Encode and Add Default
    print(f"\n[INFO] Début de la vectorisation de {len(all_chunks_default)} chunks SANS titre...")
    embeddings_def = bge_model.encode(all_chunks_default)
    vector_store.add_chunks(all_chunks_default, embeddings_def['dense_vecs'], embeddings_def['lexical_weights'], ids=all_ids_default, namespace="default")
    
    # Encode and Add With Titles
    print(f"\n[INFO] Début de la vectorisation de {len(all_chunks_titles)} chunks AVEC titre...")
    embeddings_tit = bge_model.encode(all_chunks_titles)
    vector_store.add_chunks(all_chunks_titles, embeddings_tit['dense_vecs'], embeddings_tit['lexical_weights'], ids=all_ids_titles, namespace="with_titles")
    
    print("[INFO] Vectorisation terminée ! Ajout dans ChromaDB et l'index local terminé.")
    
    return jsonify({
        "status": "success", 
        "message": f"Successfully reindexed {len(all_chunks_default)} chunks for both methods.",
        "chunk_count": len(all_chunks_default)
    })

@app.route('/api/search', methods=['POST'])
def search():
    init_models()
    data = request.json
    query = data.get('query', '')
    top_k = int(data.get('top_k', 5))
    rrf_k = int(data.get('rrf_k', 60))
    
    if not query:
        return jsonify({"status": "error", "message": "Query cannot be empty"}), 400
        
    # Encode query
    q_embeddings = bge_model.encode(query)
    
    # Safely convert to list for ChromaDB and JSON serialization
    q_dense = q_embeddings['dense_vecs'][0]
    if hasattr(q_dense, 'tolist'):
        q_dense = q_dense.tolist()
        
    q_sparse = q_embeddings['lexical_weights'][0]
    # Ensure q_sparse values are standard floats
    q_sparse = {k: float(v) for k, v in q_sparse.items()}
    
    # Search Dense
    dense_results = vector_store.search_dense(q_dense, top_k=top_k)
    
    # Search Sparse
    sparse_results = vector_store.search_sparse(
        q_sparse, 
        bge_model.compute_lexical_matching_score, 
        top_k=top_k
    )
    
    # Search RRF
    rrf_results = vector_store.rrf_fusion(dense_results, sparse_results, k=rrf_k, top_k=top_k)
    
    return jsonify({
        "status": "success",
        "dense_results": dense_results,
        "sparse_results": sparse_results,
        "rrf_results": rrf_results
    })

EVAL_STATE_FILE = "eval_state.json"
eval_thread = None

def load_eval_state():
    if os.path.exists(EVAL_STATE_FILE):
        try:
            with open(EVAL_STATE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return None

def save_eval_state(state):
    with open(EVAL_STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def init_evaluation(top_k_list, rrf_k_list, num_questions, chunk_size):
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'questoin-reponse.csv')
    df = pd.read_csv(csv_path)
    if num_questions:
        n_samples = min(int(num_questions), len(df))
        df = df.sample(n=n_samples, random_state=42)
    sampled_indices = df.index.tolist()
    
    results_template = []
    for tk in top_k_list:
        for rk in rrf_k_list:
            results_template.append({
                "top_k": tk, "rrf_k": rk,
                "dense": {"recall": 0, "ndcg": 0, "mrr": 0},
                "sparse": {"recall": 0, "ndcg": 0, "mrr": 0},
                "rrf": {"recall": 0, "ndcg": 0, "mrr": 0},
                "colbert": {"recall": 0, "ndcg": 0, "mrr": 0},
                "cross_encoder": {"recall": 0, "ndcg": 0, "mrr": 0}
            })
            
    state = {
        "status": "idle",
        "params": {
            "top_k_list": top_k_list,
            "rrf_k_list": rrf_k_list,
            "num_questions": num_questions,
            "chunk_size": chunk_size,
            "sampled_indices": sampled_indices
        },
        "progress": {
            "current_method_idx": 0,
            "current_question_idx": 0,
            "total_questions": len(sampled_indices)
        },
        "results": {
            "default": json.loads(json.dumps(results_template)),
            "with_titles": json.loads(json.dumps(results_template)),
            "adjacent": json.loads(json.dumps(results_template))
        },
        "times": {
            "default": {"total": 0.0, "dense": 0.0, "sparse": 0.0, "rrf": 0.0, "colbert": 0.0, "cross_encoder": 0.0},
            "with_titles": {"total": 0.0, "dense": 0.0, "sparse": 0.0, "rrf": 0.0, "colbert": 0.0, "cross_encoder": 0.0},
            "adjacent": {"total": 0.0, "dense": 0.0, "sparse": 0.0, "rrf": 0.0, "colbert": 0.0, "cross_encoder": 0.0}
        }
    }
    save_eval_state(state)
    return state

def finalize_evaluation(state):
    n_queries = state["progress"]["total_questions"]
    chunk_size = state["params"]["chunk_size"]
    
    for method in ["default", "with_titles", "adjacent"]:
        for res_dict in state["results"][method]:
            for model in ["dense", "sparse", "rrf", "colbert", "cross_encoder"]:
                res_dict[model]["recall"] = round(res_dict[model]["recall"] / max(1, n_queries), 4)
                res_dict[model]["ndcg"] = round(res_dict[model]["ndcg"] / max(1, n_queries), 4)
                res_dict[model]["mrr"] = round(res_dict[model]["mrr"] / max(1, n_queries), 4)
                
    state["status"] = "finished"
    save_eval_state(state)
    
    csv_filename = f"results_chunk{chunk_size}_{n_queries}questions.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Top_K", "RRF_K", "Model", "Recall", "NDCG", "MRR", "Time_Taken_Sec", "Time_Dense", "Time_Sparse", "Time_RRF", "Time_Colbert", "Time_CrossEncoder"])
        for method in ["default", "with_titles", "adjacent"]:
            time_info = state["times"][method]
            if not isinstance(time_info, dict):
                time_info = {"total": time_info, "dense": 0, "sparse": 0, "rrf": 0, "colbert": 0, "cross_encoder": 0}
            
            for res_dict in state["results"][method]:
                tk = res_dict["top_k"]
                rk = res_dict["rrf_k"]
                for model in ["dense", "sparse", "rrf", "colbert", "cross_encoder"]:
                    writer.writerow([
                        method, tk, rk, model,
                        res_dict[model]["recall"],
                        res_dict[model]["ndcg"],
                        res_dict[model]["mrr"],
                        round(time_info["total"], 2),
                        round(time_info["dense"], 2),
                        round(time_info["sparse"], 2),
                        round(time_info["rrf"], 2),
                        round(time_info["colbert"], 2),
                        round(time_info["cross_encoder"], 2)
                    ])
    print(f"[INFO] Évaluation terminée. CSV exporté : {csv_filename}")

def evaluation_worker():
    global eval_pause_event
    state = load_eval_state()
    if not state: return
    
    try:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'questoin-reponse.csv')
        df_full = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}")
        return

    df = df_full.loc[state["params"]["sampled_indices"]]
    
    methods = ["default", "with_titles", "adjacent"]
    top_k_list = state["params"]["top_k_list"]
    max_top_k = max(top_k_list) if top_k_list else 10
    n_queries = len(df)
    
    while state["progress"]["current_method_idx"] < 3:
        m_idx = state["progress"]["current_method_idx"]
        method = methods[m_idx]
        q_idx = state["progress"]["current_question_idx"]
        
        namespace = "default" if method != "with_titles" else "with_titles"
        expand_adj = (method == "adjacent")
        
        if not isinstance(state["times"][method], dict):
            old_val = state["times"][method]
            state["times"][method] = {"total": old_val, "dense": 0.0, "sparse": 0.0, "rrf": 0.0, "colbert": 0.0, "cross_encoder": 0.0}
        
        start_time = time.time()
        
        for i in range(q_idx, n_queries):
            if not eval_pause_event.is_set():
                state["status"] = "paused"
                state["times"][method]["total"] += (time.time() - start_time)
                save_eval_state(state)
                print("[INFO] Évaluation mise en pause.")
                return
                
            print(f"[INFO] [{method}] Question traitée : {i+1} / {n_queries}")
            
            row = df.iloc[i]
            q = str(row['question'])
            ans = str(row['reponse'])
            
            q_embs = bge_model.encode(q)
            q_dense = q_embs['dense_vecs'][0]
            if hasattr(q_dense, 'tolist'): q_dense = q_dense.tolist()
            q_sparse = q_embs['lexical_weights'][0]
            q_sparse = {k: float(v) for k, v in q_sparse.items()}
            
            t0 = time.time()
            dense_res_all = vector_store.search_dense(q_dense, top_k=max_top_k, namespace=namespace, expand_adjacent=expand_adj)
            t1 = time.time()
            state["times"][method]["dense"] += (t1 - t0)
            
            t0 = time.time()
            sparse_res_all = vector_store.search_sparse(q_sparse, bge_model.compute_lexical_matching_score, top_k=max_top_k, namespace=namespace, expand_adjacent=expand_adj)
            t1 = time.time()
            state["times"][method]["sparse"] += (t1 - t0)
            
            t0 = time.time()
            rrf_res_all_max = vector_store.rrf_fusion(dense_res_all, sparse_res_all, k=60, top_k=max_top_k)
            t1 = time.time()
            state["times"][method]["rrf"] += (t1 - t0)
            
            texts_to_rerank = [res["text"] for res in rrf_res_all_max]
            
            t0 = time.time()
            if texts_to_rerank:
                q_colbert = bge_model.encode_colbert(q)[0]
                p_colberts = bge_model.encode_colbert(texts_to_rerank)
                colbert_res_all = []
                for j, res in enumerate(rrf_res_all_max):
                    score = bge_model.compute_colbert_score(q_colbert, p_colberts[j])
                    new_res = res.copy()
                    new_res["score"] = float(score)
                    colbert_res_all.append(new_res)
                colbert_res_all.sort(key=lambda x: x["score"], reverse=True)
            else:
                colbert_res_all = []
            t1 = time.time()
            state["times"][method]["colbert"] += (t1 - t0)
            
            t0 = time.time()
            if texts_to_rerank:
                ce_res_all = cross_encoder.rerank(q, rrf_res_all_max)
            else:
                ce_res_all = []
            t1 = time.time()
            state["times"][method]["cross_encoder"] += (t1 - t0)
            
            # Accumulate metrics
            for res_dict in state["results"][method]:
                tk = res_dict["top_k"]
                rk = res_dict["rrf_k"]
                
                dense_res = dense_res_all[:tk]
                sparse_res = sparse_res_all[:tk]
                rrf_res = vector_store.rrf_fusion(dense_res, sparse_res, k=rk, top_k=tk)
                colbert_res = colbert_res_all[:tk]
                ce_res = ce_res_all[:tk]
                
                res_dict["dense"]["recall"] += calculate_recall_at_k(dense_res, ans, tk)
                res_dict["dense"]["ndcg"] += calculate_ndcg_at_k(dense_res, ans, tk)
                res_dict["dense"]["mrr"] += calculate_mrr_at_k(dense_res, ans, tk)
                res_dict["sparse"]["recall"] += calculate_recall_at_k(sparse_res, ans, tk)
                res_dict["sparse"]["ndcg"] += calculate_ndcg_at_k(sparse_res, ans, tk)
                res_dict["sparse"]["mrr"] += calculate_mrr_at_k(sparse_res, ans, tk)
                res_dict["rrf"]["recall"] += calculate_recall_at_k(rrf_res, ans, tk)
                res_dict["rrf"]["ndcg"] += calculate_ndcg_at_k(rrf_res, ans, tk)
                res_dict["rrf"]["mrr"] += calculate_mrr_at_k(rrf_res, ans, tk)
                res_dict["colbert"]["recall"] += calculate_recall_at_k(colbert_res, ans, tk)
                res_dict["colbert"]["ndcg"] += calculate_ndcg_at_k(colbert_res, ans, tk)
                res_dict["colbert"]["mrr"] += calculate_mrr_at_k(colbert_res, ans, tk)
                res_dict["cross_encoder"]["recall"] += calculate_recall_at_k(ce_res, ans, tk)
                res_dict["cross_encoder"]["ndcg"] += calculate_ndcg_at_k(ce_res, ans, tk)
                res_dict["cross_encoder"]["mrr"] += calculate_mrr_at_k(ce_res, ans, tk)
                
            state["progress"]["current_question_idx"] = i + 1
            save_eval_state(state)
            
        state["times"][method]["total"] += (time.time() - start_time)
        state["progress"]["current_method_idx"] += 1
        state["progress"]["current_question_idx"] = 0
        save_eval_state(state)
        
    finalize_evaluation(state)

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    global eval_thread, eval_pause_event
    init_models()
    data = request.json
    
    action = data.get('action', 'start')
    if action == 'resume':
        state = load_eval_state()
        if not state or state["status"] == "finished":
            return jsonify({"status": "error", "message": "Aucune évaluation en pause."}), 400
        state["status"] = "running"
        save_eval_state(state)
    else:
        top_k_list = data.get('top_k_list', [10])
        rrf_k_list = data.get('rrf_k_list', [60])
        num_questions = data.get('num_questions')
        chunk_size = data.get('chunk_size', 256)
        if not isinstance(top_k_list, list): top_k_list = [top_k_list]
        if not isinstance(rrf_k_list, list): rrf_k_list = [rrf_k_list]
        
        state = init_evaluation(top_k_list, rrf_k_list, num_questions, chunk_size)
        state["status"] = "running"
        save_eval_state(state)
        
    eval_pause_event.set()
    eval_thread = threading.Thread(target=evaluation_worker)
    eval_thread.start()
    
    return jsonify({"status": "success", "message": "Évaluation démarrée/reprise."})

@app.route('/api/status', methods=['GET'])
def get_status():
    state = load_eval_state()
    if not state:
        return jsonify({"status": "idle"})
    return jsonify(state)

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Aucun fichier"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Aucun fichier selectionné"}), 400
        
    try:
        df = pd.read_csv(file)
        req_cols = ["Method", "Top_K", "RRF_K", "Model", "Recall", "NDCG", "MRR"]
        if not all(col in df.columns for col in req_cols):
            return jsonify({"status": "error", "message": "Format CSV invalide"}), 400
            
        results = {"default": [], "with_titles": [], "adjacent": []}
        grouped = df.groupby(["Method", "Top_K", "RRF_K"])
        for (method, tk, rk), group in grouped:
            if method not in results:
                continue
                
            res_dict = {
                "top_k": int(tk[0]) if isinstance(tk, tuple) else int(tk),
                "rrf_k": int(rk[0]) if isinstance(rk, tuple) else int(rk),
            }
            for _, row in group.iterrows():
                model = row["Model"]
                res_dict[model] = {
                    "recall": float(row["Recall"]),
                    "ndcg": float(row["NDCG"]),
                    "mrr": float(row["MRR"])
                }
            results[method].append(res_dict)
            
        return jsonify({
            "status": "success",
            "results_default": results.get("default", []),
            "results_titles": results.get("with_titles", []),
            "results_adjacent": results.get("adjacent", [])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
