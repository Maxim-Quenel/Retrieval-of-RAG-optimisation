document.addEventListener('DOMContentLoaded', () => {
    
    // --- Elements ---
    const btnReindex = document.getElementById('btn-reindex');
    const chunkSizeInput = document.getElementById('chunk-size');
    const reindexLoader = document.getElementById('reindex-loader');
    const reindexStatus = document.getElementById('reindex-status');
    const reindexBtnText = btnReindex.querySelector('.btn-text');

    const searchInput = document.getElementById('search-input');
    const btnSearch = document.getElementById('btn-search');
    const searchTopK = document.getElementById('search-top-k');
    const searchRrfK = document.getElementById('search-rrf-k');
    const resultsContainer = document.getElementById('results-container');
    const searchLoader = document.getElementById('search-loader');
    
    const listDense = document.getElementById('list-dense');
    const listSparse = document.getElementById('list-sparse');
    const listRrf = document.getElementById('list-rrf');

    const btnEval = document.getElementById('btn-evaluate');
    const evalTopK = document.getElementById('eval-top-k');
    const evalRrfK = document.getElementById('eval-rrf-k');
    const evalLoader = document.getElementById('eval-loader');
    const evalResultsContainer = document.getElementById('eval-results-container');
    const evalBtnText = btnEval.querySelector('.btn-text');
    const btnPause = document.getElementById('btn-pause');
    const evalProgressContainer = document.getElementById('eval-progress-container');
    const evalProgressBar = document.getElementById('eval-progress-bar');
    const evalStatusText = document.getElementById('eval-status-text');
    const btnUploadCsv = document.getElementById('btn-upload-csv');
    const csvFileInput = document.getElementById('csv-file');

    let evalInterval = null;

    // --- Persist Helpers ---
    function renderTable(scores, tbodyId) {
        const tbody = document.getElementById(tbodyId);
        tbody.innerHTML = '';
        scores.forEach(s => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${s.top_k}</td>
                <td>${s.rrf_k}</td>
                <td>${s.dense.recall.toFixed(4)} / ${s.dense.ndcg.toFixed(4)} / ${s.dense.mrr.toFixed(4)}</td>
                <td>${s.sparse.recall.toFixed(4)} / ${s.sparse.ndcg.toFixed(4)} / ${s.sparse.mrr.toFixed(4)}</td>
                <td><strong>${s.rrf.recall.toFixed(4)} / ${s.rrf.ndcg.toFixed(4)} / ${s.rrf.mrr.toFixed(4)}</strong></td>
                <td><strong>${s.colbert.recall.toFixed(4)} / ${s.colbert.ndcg.toFixed(4)} / ${s.colbert.mrr.toFixed(4)}</strong></td>
                <td><strong>${s.cross_encoder.recall.toFixed(4)} / ${s.cross_encoder.ndcg.toFixed(4)} / ${s.cross_encoder.mrr.toFixed(4)}</strong></td>
            `;
            tbody.appendChild(tr);
        });
    }

    function renderTableAndCharts(scoresDefault, scoresTitles, scoresAdjacent) {
        renderTable(scoresDefault, 'eval-results-body-default');
        renderTable(scoresTitles, 'eval-results-body-titles');
        if (scoresAdjacent) {
            renderTable(scoresAdjacent, 'eval-results-body-adjacent');
        }
        
        updateCharts(scoresDefault, scoresTitles, scoresAdjacent);
        evalResultsContainer.classList.remove('hidden');
    }

    // Restore inputs
    if (localStorage.getItem('evalTopK')) evalTopK.value = localStorage.getItem('evalTopK');
    if (localStorage.getItem('evalRrfK')) evalRrfK.value = localStorage.getItem('evalRrfK');
    if (localStorage.getItem('evalNumQuestions')) document.getElementById('eval-num-questions').value = localStorage.getItem('evalNumQuestions');
    
    // Restore scores
    const savedScoresDefault = localStorage.getItem('evalScoresDefault');
    const savedScoresTitles = localStorage.getItem('evalScoresTitles');
    const savedScoresAdjacent = localStorage.getItem('evalScoresAdjacent');
    if (savedScoresDefault && savedScoresTitles) {
        try {
            const scoresDefault = JSON.parse(savedScoresDefault);
            const scoresTitles = JSON.parse(savedScoresTitles);
            const scoresAdjacent = savedScoresAdjacent ? JSON.parse(savedScoresAdjacent) : null;
            renderTableAndCharts(scoresDefault, scoresTitles, scoresAdjacent);
        } catch(e) {
            console.error(e);
        }
    }

    // --- Init ---
    // Initialize models in background just in case
    fetch('/api/init', { method: 'POST' }).catch(console.error);

    // --- Helpers ---
    function renderChunks(listElement, chunks) {
        listElement.innerHTML = '';
        if (!chunks || chunks.length === 0) {
            listElement.innerHTML = '<p class="chunk-text">Aucun résultat trouvé.</p>';
            return;
        }

        chunks.forEach(chunk => {
            const div = document.createElement('div');
            div.className = 'chunk-item';
            
            // Format score based on value
            let displayScore = typeof chunk.score === 'number' ? chunk.score.toFixed(4) : chunk.score;
            
            div.innerHTML = `
                <div class="chunk-meta">
                    <span class="chunk-rank">#${chunk.rank}</span>
                    <span class="chunk-score">Score: ${displayScore}</span>
                </div>
                <div class="chunk-text">${chunk.text}</div>
            `;
            listElement.appendChild(div);
        });
    }

    // --- Reindex ---
    btnReindex.addEventListener('click', async () => {
        const chunkSize = document.getElementById('chunk-size').value;
        const chunkOverlap = document.getElementById('chunk-overlap').value;
        const dataSource = document.getElementById('data-source').value;
        btnReindex.disabled = true;
        reindexLoader.classList.remove('hidden');
        reindexStatus.className = 'status-msg';
        reindexStatus.textContent = '';
        
        try {
            const res = await fetch('/api/reindex', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap, data_source: dataSource })
            });
            const data = await res.json();
            
            if (data.status === 'success') {
                reindexStatus.textContent = `Succès ! ${data.chunk_count} chunks vectorisés (Dense & Sparse).`;
                reindexStatus.className = 'status-msg success';
            } else {
                throw new Error(data.message);
            }
        } catch (err) {
            reindexStatus.textContent = `Erreur: ${err.message}`;
            reindexStatus.className = 'status-msg error';
        } finally {
            reindexBtnText.classList.remove('hidden');
            reindexLoader.classList.add('hidden');
            btnReindex.disabled = false;
        }
    });

    // --- Search ---
    const performSearch = async () => {
        const query = searchInput.value.trim();
        if (!query) return;

        const topK = parseInt(searchTopK.value) || 5;
        const rrfK = parseInt(searchRrfK.value) || 60;

        resultsContainer.classList.add('hidden');
        searchLoader.classList.remove('hidden');
        
        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: topK, rrf_k: rrfK })
            });
            
            const data = await res.json();
            
            if (data.status === 'success') {
                renderChunks(listDense, data.dense_results);
                renderChunks(listSparse, data.sparse_results);
                renderChunks(listRrf, data.rrf_results);
                
                searchLoader.classList.add('hidden');
                resultsContainer.classList.remove('hidden');
            } else {
                alert(`Erreur: ${data.message}`);
                searchLoader.classList.add('hidden');
            }
        } catch (err) {
            alert(`Erreur réseau: ${err.message}`);
            searchLoader.classList.add('hidden');
        }
    };

    btnSearch.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    // --- Evaluate ---
    function updateEvalUI(state) {
        if (!state || state.status === 'idle') {
            evalBtnText.classList.remove('hidden');
            evalLoader.classList.add('hidden');
            btnEval.disabled = false;
            btnPause.classList.add('hidden');
            evalProgressContainer.classList.add('hidden');
            return;
        }

        if (state.status === 'running' || state.status === 'paused') {
            evalBtnText.classList.add('hidden');
            evalLoader.classList.add('hidden'); 
            btnEval.disabled = true;
            btnPause.classList.remove('hidden');
            evalProgressContainer.classList.remove('hidden');

            const m_idx = state.progress.current_method_idx;
            const q_idx = state.progress.current_question_idx;
            const total = state.progress.total_questions;
            
            const methods = ["Sans Titre", "Avec Titre", "Adjacents"];
            const currentMethod = methods[m_idx] || "Terminé";
            
            const totalTasks = 3 * total;
            const currentTask = (m_idx * total) + q_idx;
            const pct = Math.round((currentTask / Math.max(1, totalTasks)) * 100);

            evalProgressBar.style.width = `${pct}%`;

            if (state.status === 'paused') {
                evalStatusText.textContent = `En pause (${pct}%) - Méthode : ${currentMethod}`;
                btnPause.querySelector('.btn-text').textContent = 'Reprendre l\'évaluation';
                btnPause.style.backgroundColor = '#10b981';
                btnPause.style.borderColor = '#10b981';
            } else {
                evalStatusText.textContent = `En cours (${pct}%) - Méthode : ${currentMethod} (${q_idx}/${total})`;
                btnPause.querySelector('.btn-text').textContent = 'Mettre en pause';
                btnPause.style.backgroundColor = '#f59e0b';
                btnPause.style.borderColor = '#f59e0b';
            }
        }

        if (state.status === 'finished') {
            clearInterval(evalInterval);
            evalInterval = null;
            evalBtnText.classList.remove('hidden');
            evalLoader.classList.add('hidden');
            btnEval.disabled = false;
            btnPause.classList.add('hidden');
            evalProgressContainer.classList.add('hidden');

            localStorage.setItem('evalScoresDefault', JSON.stringify(state.results.default));
            localStorage.setItem('evalScoresTitles', JSON.stringify(state.results.with_titles));
            localStorage.setItem('evalScoresAdjacent', JSON.stringify(state.results.adjacent));
            renderTableAndCharts(state.results.default, state.results.with_titles, state.results.adjacent);
            
            // Re-fetch status to idle if we want, or let user click evaluate again.
        }
    }

    async function pollStatus() {
        try {
            const res = await fetch('/api/status');
            const state = await res.json();
            updateEvalUI(state);
            if (state.status === 'finished' || state.status === 'idle') {
                if (evalInterval) {
                    clearInterval(evalInterval);
                    evalInterval = null;
                }
            } else if (state.status === 'running' && !evalInterval) {
                evalInterval = setInterval(pollStatus, 2000);
            }
        } catch(e) {
            console.error(e);
        }
    }

    pollStatus(); // Check on load

    btnEval.addEventListener('click', async () => {
        const topKStr = evalTopK.value || "10";
        const rrfKStr = evalRrfK.value || "60";
        const numQInput = document.getElementById('eval-num-questions').value;
        const chunkSize = document.getElementById('chunk-size').value || "256";
        const numQ = numQInput ? parseInt(numQInput) : null;
        
        localStorage.setItem('evalTopK', evalTopK.value);
        localStorage.setItem('evalRrfK', evalRrfK.value);
        localStorage.setItem('evalNumQuestions', numQInput);

        const topKList = topKStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        const rrfKList = rrfKStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

        if (topKList.length === 0) topKList.push(10);
        if (rrfKList.length === 0) rrfKList.push(60);
        
        try {
            const res = await fetch('/api/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    action: 'start',
                    top_k_list: topKList, 
                    rrf_k_list: rrfKList, 
                    num_questions: numQ,
                    chunk_size: parseInt(chunkSize)
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                pollStatus();
            } else {
                alert(`Erreur: ${data.message}`);
            }
        } catch (err) {
            alert(`Erreur réseau: ${err.message}`);
        }
    });

    btnPause.addEventListener('click', async () => {
        try {
            const resStatus = await fetch('/api/status');
            const state = await resStatus.json();
            
            if (state.status === 'paused') {
                const res = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'resume' })
                });
                const data = await res.json();
                if (data.status === 'success') pollStatus();
            } else if (state.status === 'running') {
                const res = await fetch('/api/pause', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'paused') {
                    if(evalInterval) clearInterval(evalInterval);
                    evalInterval = null;
                    pollStatus();
                }
            }
        } catch (err) {
            alert(`Erreur: ${err.message}`);
        }
    });

    if(btnUploadCsv) {
        btnUploadCsv.addEventListener('click', async () => {
            if(!csvFileInput.files || csvFileInput.files.length === 0) {
                alert('Veuillez sélectionner un fichier CSV.');
                return;
            }
            const formData = new FormData();
            formData.append('file', csvFileInput.files[0]);

            btnUploadCsv.disabled = true;
            btnUploadCsv.querySelector('.btn-text').textContent = 'Chargement...';

            try {
                const res = await fetch('/api/upload_csv', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.status === 'success') {
                    localStorage.setItem('evalScoresDefault', JSON.stringify(data.results_default));
                    localStorage.setItem('evalScoresTitles', JSON.stringify(data.results_titles));
                    localStorage.setItem('evalScoresAdjacent', JSON.stringify(data.results_adjacent));
                    renderTableAndCharts(data.results_default, data.results_titles, data.results_adjacent);
                } else {
                    alert(`Erreur: ${data.message}`);
                }
            } catch(e) {
                alert(`Erreur réseau: ${e.message}`);
            } finally {
                btnUploadCsv.disabled = false;
                btnUploadCsv.querySelector('.btn-text').textContent = 'Charger et Visualiser';
            }
        });
    }
});

// --- Chart Instances ---
let recallChartInstance = null;
let ndcgChartInstance = null;
let mrrChartInstance = null;
let modalChartInstance = null;
let lastScoresDefault = null;
let lastScoresTitles = null;
let lastScoresAdjacent = null;

const chartModal = document.getElementById('chart-modal');
const closeModal = document.getElementById('close-modal');

closeModal.addEventListener('click', () => {
    chartModal.classList.add('hidden');
});
chartModal.addEventListener('click', (e) => {
    if (e.target === chartModal) chartModal.classList.add('hidden');
});

function openModalChart(metricTitle, metricKey) {
    if (!lastScoresDefault || !lastScoresTitles) return;
    
    chartModal.classList.remove('hidden');
    
    const labels = lastScoresDefault.map(s => `${s.top_k} / ${s.rrf_k}`);
    
    const denseDataDef = lastScoresDefault.map(s => s.dense[metricKey]);
    const sparseDataDef = lastScoresDefault.map(s => s.sparse[metricKey]);
    const rrfDataDef = lastScoresDefault.map(s => s.rrf[metricKey]);
    const colbertDataDef = lastScoresDefault.map(s => s.colbert[metricKey]);
    const ceDataDef = lastScoresDefault.map(s => s.cross_encoder[metricKey]);
    
    const denseDataTit = lastScoresTitles.map(s => s.dense[metricKey]);
    const sparseDataTit = lastScoresTitles.map(s => s.sparse[metricKey]);
    const rrfDataTit = lastScoresTitles.map(s => s.rrf[metricKey]);
    const colbertDataTit = lastScoresTitles.map(s => s.colbert[metricKey]);
    const ceDataTit = lastScoresTitles.map(s => s.cross_encoder[metricKey]);
    
    if (modalChartInstance) modalChartInstance.destroy();
    
    const datasets = [
        { label: 'Dense (Sans Titre)', data: denseDataDef, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderDash: [5, 5] },
        { label: 'Sparse (Sans Titre)', data: sparseDataDef, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderDash: [5, 5] },
        { label: 'RRF (Sans Titre)', data: rrfDataDef, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderDash: [5, 5] },
        { label: 'ColBERT (Sans Titre)', data: colbertDataDef, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderDash: [5, 5] },
        { label: 'CrossEncoder (Sans Titre)', data: ceDataDef, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderDash: [5, 5] },
        
        { label: 'Dense (Avec Titre)', data: denseDataTit, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderWidth: 3 },
        { label: 'Sparse (Avec Titre)', data: sparseDataTit, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderWidth: 3 },
        { label: 'RRF (Avec Titre)', data: rrfDataTit, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderWidth: 3 },
        { label: 'ColBERT (Avec Titre)', data: colbertDataTit, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderWidth: 3 },
        { label: 'CrossEncoder (Avec Titre)', data: ceDataTit, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderWidth: 3 }
    ];

    if (lastScoresAdjacent) {
        const denseDataAdj = lastScoresAdjacent.map(s => s.dense[metricKey]);
        const sparseDataAdj = lastScoresAdjacent.map(s => s.sparse[metricKey]);
        const rrfDataAdj = lastScoresAdjacent.map(s => s.rrf[metricKey]);
        const colbertDataAdj = lastScoresAdjacent.map(s => s.colbert[metricKey]);
        const ceDataAdj = lastScoresAdjacent.map(s => s.cross_encoder[metricKey]);
        
        datasets.push(
            { label: 'Dense (Adjacents)', data: denseDataAdj, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderDash: [2, 2], borderWidth: 2 },
            { label: 'Sparse (Adjacents)', data: sparseDataAdj, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderDash: [2, 2], borderWidth: 2 },
            { label: 'RRF (Adjacents)', data: rrfDataAdj, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderDash: [2, 2], borderWidth: 2 },
            { label: 'ColBERT (Adjacents)', data: colbertDataAdj, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderDash: [2, 2], borderWidth: 2 },
            { label: 'CrossEncoder (Adjacents)', data: ceDataAdj, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderDash: [2, 2], borderWidth: 2 }
        );
    }
    
    const ctx = document.getElementById('modalChart').getContext('2d');
    modalChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false, 
            plugins: { 
                title: { display: true, text: metricTitle, color: '#f8fafc', font: { size: 18 } }, 
                legend: { labels: { color: '#f8fafc' } },
                zoom: {
                    zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' },
                    pan: { enabled: true, mode: 'xy' }
                }
            },
            scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' }, min: 0, max: 1 } }
        }
    });
}

document.getElementById('recallChart').addEventListener('click', () => openModalChart('Recall@K', 'recall'));
document.getElementById('ndcgChart').addEventListener('click', () => openModalChart('NDCG@10', 'ndcg'));
document.getElementById('mrrChart').addEventListener('click', () => openModalChart('MRR@K', 'mrr'));

function updateCharts(scoresDefault, scoresTitles, scoresAdjacent) {
    lastScoresDefault = scoresDefault;
    lastScoresTitles = scoresTitles;
    lastScoresAdjacent = scoresAdjacent;
    
    const labels = scoresDefault.map(s => `${s.top_k} / ${s.rrf_k}`);
    
    const denseRecallDef = scoresDefault.map(s => s.dense.recall);
    const sparseRecallDef = scoresDefault.map(s => s.sparse.recall);
    const rrfRecallDef = scoresDefault.map(s => s.rrf.recall);
    const colbertRecallDef = scoresDefault.map(s => s.colbert.recall);
    const ceRecallDef = scoresDefault.map(s => s.cross_encoder.recall);
    
    const denseRecallTit = scoresTitles.map(s => s.dense.recall);
    const sparseRecallTit = scoresTitles.map(s => s.sparse.recall);
    const rrfRecallTit = scoresTitles.map(s => s.rrf.recall);
    const colbertRecallTit = scoresTitles.map(s => s.colbert.recall);
    const ceRecallTit = scoresTitles.map(s => s.cross_encoder.recall);
    
    const denseNdcgDef = scoresDefault.map(s => s.dense.ndcg);
    const sparseNdcgDef = scoresDefault.map(s => s.sparse.ndcg);
    const rrfNdcgDef = scoresDefault.map(s => s.rrf.ndcg);
    const colbertNdcgDef = scoresDefault.map(s => s.colbert.ndcg);
    const ceNdcgDef = scoresDefault.map(s => s.cross_encoder.ndcg);
    
    const denseNdcgTit = scoresTitles.map(s => s.dense.ndcg);
    const sparseNdcgTit = scoresTitles.map(s => s.sparse.ndcg);
    const rrfNdcgTit = scoresTitles.map(s => s.rrf.ndcg);
    const colbertNdcgTit = scoresTitles.map(s => s.colbert.ndcg);
    const ceNdcgTit = scoresTitles.map(s => s.cross_encoder.ndcg);
    
    const denseMrrDef = scoresDefault.map(s => s.dense.mrr);
    const sparseMrrDef = scoresDefault.map(s => s.sparse.mrr);
    const rrfMrrDef = scoresDefault.map(s => s.rrf.mrr);
    const colbertMrrDef = scoresDefault.map(s => s.colbert.mrr);
    const ceMrrDef = scoresDefault.map(s => s.cross_encoder.mrr);
    
    const denseMrrTit = scoresTitles.map(s => s.dense.mrr);
    const sparseMrrTit = scoresTitles.map(s => s.sparse.mrr);
    const rrfMrrTit = scoresTitles.map(s => s.rrf.mrr);
    const colbertMrrTit = scoresTitles.map(s => s.colbert.mrr);
    const ceMrrTit = scoresTitles.map(s => s.cross_encoder.mrr);
    
    if (recallChartInstance) recallChartInstance.destroy();
    if (ndcgChartInstance) ndcgChartInstance.destroy();
    if (mrrChartInstance) mrrChartInstance.destroy();
    
    const datasetsRecall = [
        { label: 'Dense (Sans Titre)', data: denseRecallDef, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderDash: [5, 5] },
        { label: 'Sparse (Sans Titre)', data: sparseRecallDef, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderDash: [5, 5] },
        { label: 'RRF (Sans Titre)', data: rrfRecallDef, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderDash: [5, 5] },
        { label: 'ColBERT (Sans Titre)', data: colbertRecallDef, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderDash: [5, 5] },
        { label: 'CrossEncoder (Sans Titre)', data: ceRecallDef, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderDash: [5, 5] },
        
        { label: 'Dense (Avec Titre)', data: denseRecallTit, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderWidth: 3 },
        { label: 'Sparse (Avec Titre)', data: sparseRecallTit, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderWidth: 3 },
        { label: 'RRF (Avec Titre)', data: rrfRecallTit, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderWidth: 3 },
        { label: 'ColBERT (Avec Titre)', data: colbertRecallTit, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderWidth: 3 },
        { label: 'CrossEncoder (Avec Titre)', data: ceRecallTit, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderWidth: 3 }
    ];

    const datasetsNdcg = [
        { label: 'Dense (Sans Titre)', data: denseNdcgDef, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderDash: [5, 5] },
        { label: 'Sparse (Sans Titre)', data: sparseNdcgDef, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderDash: [5, 5] },
        { label: 'RRF (Sans Titre)', data: rrfNdcgDef, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderDash: [5, 5] },
        { label: 'ColBERT (Sans Titre)', data: colbertNdcgDef, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderDash: [5, 5] },
        { label: 'CrossEncoder (Sans Titre)', data: ceNdcgDef, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderDash: [5, 5] },
        
        { label: 'Dense (Avec Titre)', data: denseNdcgTit, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderWidth: 3 },
        { label: 'Sparse (Avec Titre)', data: sparseNdcgTit, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderWidth: 3 },
        { label: 'RRF (Avec Titre)', data: rrfNdcgTit, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderWidth: 3 },
        { label: 'ColBERT (Avec Titre)', data: colbertNdcgTit, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderWidth: 3 },
        { label: 'CrossEncoder (Avec Titre)', data: ceNdcgTit, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderWidth: 3 }
    ];

    const datasetsMrr = [
        { label: 'Dense (Sans Titre)', data: denseMrrDef, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderDash: [5, 5] },
        { label: 'Sparse (Sans Titre)', data: sparseMrrDef, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderDash: [5, 5] },
        { label: 'RRF (Sans Titre)', data: rrfMrrDef, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderDash: [5, 5] },
        { label: 'ColBERT (Sans Titre)', data: colbertMrrDef, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderDash: [5, 5] },
        { label: 'CrossEncoder (Sans Titre)', data: ceMrrDef, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderDash: [5, 5] },
        
        { label: 'Dense (Avec Titre)', data: denseMrrTit, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderWidth: 3 },
        { label: 'Sparse (Avec Titre)', data: sparseMrrTit, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderWidth: 3 },
        { label: 'RRF (Avec Titre)', data: rrfMrrTit, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderWidth: 3 },
        { label: 'ColBERT (Avec Titre)', data: colbertMrrTit, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderWidth: 3 },
        { label: 'CrossEncoder (Avec Titre)', data: ceMrrTit, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderWidth: 3 }
    ];

    if (lastScoresAdjacent) {
        const denseRecallAdj = lastScoresAdjacent.map(s => s.dense.recall);
        const sparseRecallAdj = lastScoresAdjacent.map(s => s.sparse.recall);
        const rrfRecallAdj = lastScoresAdjacent.map(s => s.rrf.recall);
        const colbertRecallAdj = lastScoresAdjacent.map(s => s.colbert.recall);
        const ceRecallAdj = lastScoresAdjacent.map(s => s.cross_encoder.recall);
        
        datasetsRecall.push(
            { label: 'Dense (Adjacents)', data: denseRecallAdj, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderDash: [2, 2], borderWidth: 2 },
            { label: 'Sparse (Adjacents)', data: sparseRecallAdj, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderDash: [2, 2], borderWidth: 2 },
            { label: 'RRF (Adjacents)', data: rrfRecallAdj, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderDash: [2, 2], borderWidth: 2 },
            { label: 'ColBERT (Adjacents)', data: colbertRecallAdj, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderDash: [2, 2], borderWidth: 2 },
            { label: 'CrossEncoder (Adjacents)', data: ceRecallAdj, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderDash: [2, 2], borderWidth: 2 }
        );

        const denseNdcgAdj = lastScoresAdjacent.map(s => s.dense.ndcg);
        const sparseNdcgAdj = lastScoresAdjacent.map(s => s.sparse.ndcg);
        const rrfNdcgAdj = lastScoresAdjacent.map(s => s.rrf.ndcg);
        const colbertNdcgAdj = lastScoresAdjacent.map(s => s.colbert.ndcg);
        const ceNdcgAdj = lastScoresAdjacent.map(s => s.cross_encoder.ndcg);
        
        datasetsNdcg.push(
            { label: 'Dense (Adjacents)', data: denseNdcgAdj, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderDash: [2, 2], borderWidth: 2 },
            { label: 'Sparse (Adjacents)', data: sparseNdcgAdj, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderDash: [2, 2], borderWidth: 2 },
            { label: 'RRF (Adjacents)', data: rrfNdcgAdj, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderDash: [2, 2], borderWidth: 2 },
            { label: 'ColBERT (Adjacents)', data: colbertNdcgAdj, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderDash: [2, 2], borderWidth: 2 },
            { label: 'CrossEncoder (Adjacents)', data: ceNdcgAdj, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderDash: [2, 2], borderWidth: 2 }
        );

        const denseMrrAdj = lastScoresAdjacent.map(s => s.dense.mrr);
        const sparseMrrAdj = lastScoresAdjacent.map(s => s.sparse.mrr);
        const rrfMrrAdj = lastScoresAdjacent.map(s => s.rrf.mrr);
        const colbertMrrAdj = lastScoresAdjacent.map(s => s.colbert.mrr);
        const ceMrrAdj = lastScoresAdjacent.map(s => s.cross_encoder.mrr);
        
        datasetsMrr.push(
            { label: 'Dense (Adjacents)', data: denseMrrAdj, borderColor: '#3b82f6', tension: 0.1, backgroundColor: '#3b82f6', borderDash: [2, 2], borderWidth: 2 },
            { label: 'Sparse (Adjacents)', data: sparseMrrAdj, borderColor: '#10b981', tension: 0.1, backgroundColor: '#10b981', borderDash: [2, 2], borderWidth: 2 },
            { label: 'RRF (Adjacents)', data: rrfMrrAdj, borderColor: '#8b5cf6', tension: 0.1, backgroundColor: '#8b5cf6', borderDash: [2, 2], borderWidth: 2 },
            { label: 'ColBERT (Adjacents)', data: colbertMrrAdj, borderColor: '#f59e0b', tension: 0.1, backgroundColor: '#f59e0b', borderDash: [2, 2], borderWidth: 2 },
            { label: 'CrossEncoder (Adjacents)', data: ceMrrAdj, borderColor: '#ef4444', tension: 0.1, backgroundColor: '#ef4444', borderDash: [2, 2], borderWidth: 2 }
        );
    }
    
    const ctxRecall = document.getElementById('recallChart').getContext('2d');
    recallChartInstance = new Chart(ctxRecall, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasetsRecall
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false, 
            plugins: { title: { display: true, text: 'Comparaison Recall@K', color: '#94a3b8' }, legend: { labels: { color: '#f8fafc' } } },
            scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' }, min: 0, max: 1 } }
        }
    });
    
    const ctxNdcg = document.getElementById('ndcgChart').getContext('2d');
    ndcgChartInstance = new Chart(ctxNdcg, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasetsNdcg
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false, 
            plugins: { title: { display: true, text: 'Comparaison NDCG@10', color: '#94a3b8' }, legend: { labels: { color: '#f8fafc' } } },
            scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' }, min: 0, max: 1 } }
        }
    });

    const ctxMrr = document.getElementById('mrrChart').getContext('2d');
    mrrChartInstance = new Chart(ctxMrr, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasetsMrr
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false, 
            plugins: { title: { display: true, text: 'Comparaison MRR@K', color: '#94a3b8' }, legend: { labels: { color: '#f8fafc' } } },
            scales: { x: { ticks: { color: '#94a3b8' } }, y: { ticks: { color: '#94a3b8' }, min: 0, max: 1 } }
        }
    });
}
