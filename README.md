# NexusRAG : Système Avancé de RAG et de Recherche Sémantique Multi-Niveaux

Bienvenue dans le projet **NexusRAG**, un système performant de Recherche d'Information et de RAG (Retrieval-Augmented Generation). Ce projet a pour but d'indexer de grands corpus documentaires (comme Wikipédia ou le dataset francophone PIAF) et de fournir une interface d'évaluation pour différentes stratégies de vectorisation et de récupération (Retrieval).

## 🚀 Fonctionnalités Principales

- **Indexation Hybride** : Utilise le modèle `BAAI/bge-m3` pour générer simultanément trois types de représentations :
  - Vecteurs Denses (Sémantique globale) stockés dans ChromaDB.
  - Vecteurs "Sparse" (Poids lexicaux / mots-clés) gérés dans un index local.
  - Vecteurs ColBERT (Correspondance au niveau des tokens).
- **Reranking Puissant** : Utilisation du modèle Cross-Encoder `BAAI/bge-reranker-v2-m3` pour un réordonnancement ultra-précis des résultats.
- **Fusion des Rangs (RRF)** : Combinaison des résultats Dense et Sparse grâce à l'algorithme Reciprocal Rank Fusion pour une approche hybride robuste.
- **Pipeline d'Évaluation Complet** : Comparaison des performances de différentes méthodes de Retrieval (Dense, Sparse, Hybride RRF, ColBERT, Cross-Encoder) avec calcul des métriques **Recall@K** et **NDCG@K**.
- **Étude de l'Enrichissement Contextuel** : Évaluation de l'impact des métadonnées (ajout des titres aux chunks) et de l'expansion post-retrieval (ajout des chunks adjacents) sur la qualité des résultats.
- **Interface Web Interactive** : Application SPA (Single Page Application) avec un backend Flask permettant l'indexation, la recherche interactive et la visualisation des résultats et métriques.

## 📁 Structure du Projet

- `RAG_Project/` : Dossier principal contenant l'application RAG.
  - `app.py` : Le serveur backend (Flask) exposant l'API d'indexation, de recherche et d'évaluation.
  - `Etude result/` : Contient les résultats des évaluations (fichiers CSV, JSON) et les graphiques d'analyse.
  - `evaluation/` & `retrieval/` : Modules gérant la logique métier d'évaluation et de récupération.
  - `static/` & `templates/` : Fichiers du frontend (HTML, CSS, JS, Chart.js).
  - `chroma_db/` : Stockage persistant pour la base de données vectorielle.
  - `test_pipeline.py` : Script de test du pipeline RAG.
- `analyse_projet.md` : Document d'analyse détaillée de l'architecture et des expérimentations.

## 📊 Résultats de l'Étude (Graphiques d'Évaluation)

Le dossier `RAG_Project/Etude result/plots/` contient les visualisations issues des campagnes d'évaluation. Ces graphiques permettent d'analyser en détail les performances des différents modèles et approches :

### Comparaison des Méthodes de Retrieval
Ce graphique met en évidence les performances (Recall et NDCG) des différentes stratégies de recherche : Dense, Sparse, Hybride (RRF), ColBERT et Cross-Encoder.

![Comparaison des Méthodes](Etude%20result/plots/method_comparison.png)
### Performance en fonction du Top-K
Analyse de l'évolution du Recall et du NDCG en fonction du nombre de documents récupérés (K) pour les différentes méthodes.

![Performance vs Top K](Etude%20result/plots/performance_vs_top_k.png)

### Impact du paramètre K dans l'algorithme RRF
Étude de l'influence de la constante K de l'algorithme Reciprocal Rank Fusion sur la qualité globale des résultats hybrides.

![Impact RRF K](Etude%20result/plots/rrf_k_impact.png")


### Temps d'exécution Global (Pipeline)
Comparaison des temps de réponse (latence) pour chaque méthode de recherche, illustrant le compromis entre précision et coût computationnel (notamment pour le Cross-Encoder).

![Temps par Modèle](Etude%20result/plots/time_taken_model.png)

## 🛠️ Technologies Utilisées

- **Langage** : Python
- **Framework Web** : Flask (Backend), Vanilla JS / HTML / CSS (Frontend)
- **Base de Données Vectorielle** : ChromaDB
- **Modèles de Langue / Embeddings** : HuggingFace (Modèles BAAI BGE-M3 et BGE-Reranker)
- **Visualisation de données** : Chart.js (Frontend), Matplotlib/Seaborn (Génération des plots d'étude)

## 📝 Conclusion de l'Analyse

Le projet NexusRAG se présente comme un banc d'essai exhaustif et moderne pour l'étude des systèmes RAG. En couplant une interface utilisateur fluide pour la recherche visuelle à un pipeline d'évaluation scientifique rigoureux (multi-modèles et multi-métriques), il permet de déterminer avec précision l'architecture offrant le meilleur équilibre entre vitesse de traitement, coût computationnel, et pertinence des résultats.
