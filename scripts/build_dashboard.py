"""Build a self-contained, premium HTML/JS evaluation comparison dashboard."""

from __future__ import annotations

import base64
import json
from pathlib import Path


def load_json_report(path: Path) -> dict[str, any]:
    if not path.exists():
        return {"metrics": {}, "details": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_materialized_contexts(path: Path) -> dict[str, list[str]]:
    contexts_map: dict[str, list[str]] = {}
    if not path.exists():
        return contexts_map
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                contexts_map[record["question"]] = record.get("contexts", [])
            except Exception:
                continue
    return contexts_map


def get_base64_image(path: Path) -> str:
    """Encode image to base64 string for HTML embedding."""
    if not path.exists():
        return ""
    with path.open("rb") as f:
        data = f.read()
        return base64.b64encode(data).decode("utf-8")


def main() -> None:
    data_dir = Path("data")
    output_path = Path("docs/evaluation_dashboard.html")
    chart_path = Path("docs/threshold_sweep_chart.png")

    print("Loading evaluation reports...")
    # Load the 3 main comparisons
    xcds_report = load_json_report(data_dir / "ragas_report_t10.json")  # Using optimized T_min=0.10
    hybrid_report = load_json_report(data_dir / "baseline_ragas_report.json")  # Hybrid RAG
    naive_report = load_json_report(data_dir / "naive_ragas_report.json")  # Naive RAG

    print("Loading retrieved contexts...")
    # Contexts are captured from the hybrid run predictions
    contexts_map = load_materialized_contexts(data_dir / "baseline_materialized_predictions.jsonl")

    # Encode chart to base64
    print("Encoding threshold sweep chart to base64...")
    chart_base64 = get_base64_image(chart_path)

    # Merge data by question
    merged_details: list[dict[str, any]] = []
    
    hybrid_map = {item["question"]: item for item in hybrid_report.get("details", [])}
    naive_map = {item["question"]: item for item in naive_report.get("details", [])}

    for xcds_item in xcds_report.get("details", []):
        question = xcds_item["question"]
        hybrid_item = hybrid_map.get(question, {})
        naive_item = naive_map.get(question, {})

        merged_record = {
            "question": question,
            "ground_truth": xcds_item["ground_truth"],
            "xcds_answer": xcds_item["answer"],
            "xcds_scores": xcds_item.get("scores", {}),
            "hybrid_answer": hybrid_item.get("answer", "N/A"),
            "hybrid_scores": hybrid_item.get("scores", {}),
            "naive_answer": naive_item.get("answer", "N/A"),
            "naive_scores": naive_item.get("scores", {}),
            "contexts": contexts_map.get(question, []),
        }
        merged_details.append(merged_record)

    # Metrics
    x_metrics = xcds_report.get("metrics", {})
    h_metrics = hybrid_report.get("metrics", {})
    n_metrics = naive_report.get("metrics", {})

    html_template = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X-CDS Evaluation Benchmarks</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['"Plus Jakarta Sans"', 'sans-serif'],
                    },
                    colors: {
                        brand: {
                            50: '#f0f7ff',
                            100: '#e0effe',
                            500: '#007eff',
                            600: '#0062d9',
                            900: '#0a2540',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #070a13;
            background-image: 
                radial-gradient(at 0% 0%, hsla(217,76%,15%,0.3) 0, transparent 50%), 
                radial-gradient(at 100% 100%, hsla(240,60%,10%,0.3) 0, transparent 50%);
        }
        .glass {
            background: rgba(13, 18, 30, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .glass-hover:hover {
            background: rgba(20, 27, 45, 0.85);
            border-color: rgba(0, 126, 255, 0.25);
            transform: translateY(-2px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #070a13;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #334155;
        }
    </style>
</head>
<body class="text-slate-200 antialiased min-h-screen pb-12">

    <!-- Navbar -->
    <header class="sticky top-0 z-40 w-full glass border-b border-slate-800/50 backdrop-blur-md px-6 py-4 flex justify-between items-center">
        <div class="flex items-center space-x-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-blue-400 flex items-center justify-center shadow-lg shadow-brand-500/20">
                <i class="fa-solid fa-square-poll-vertical text-white text-lg"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold text-white tracking-tight">X-CDS</h1>
                <p class="text-xs text-slate-400">Advanced 3-Way RAG Benchmarks</p>
            </div>
        </div>
        <div class="flex items-center space-x-4">
            <span class="px-3 py-1 text-xs font-semibold rounded-full bg-brand-500/10 text-brand-500 border border-brand-500/20">
                <i class="fa-solid fa-server mr-1.5"></i>Dataset: Pathogens Benchmark (N=__DATA_COUNT__)
            </span>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 mt-8 space-y-8">

        <!-- Navigation Tabs -->
        <div class="flex border-b border-slate-800/60 space-x-6">
            <button onclick="switchTab('comparison')" id="tab-comparison-btn" class="pb-3 text-sm font-bold border-b-2 border-brand-500 text-white transition-all">
                <i class="fa-solid fa-table-list mr-2"></i>3-Way Query Comparison
            </button>
            <button onclick="switchTab('sweep')" id="tab-sweep-btn" class="pb-3 text-sm font-semibold border-b-2 border-transparent text-slate-400 hover:text-white transition-all">
                <i class="fa-solid fa-chart-line mr-2"></i>Threshold Sweep Chart
            </button>
        </div>

        <!-- TAB: 3-Way Comparison -->
        <div id="tab-comparison" class="space-y-8">
            <!-- Metrics Comparison Cards -->
            <section class="grid grid-cols-1 md:grid-cols-4 gap-6">
                <!-- Faithfulness Card -->
                <div class="glass rounded-2xl p-6 flex flex-col justify-between">
                    <div>
                        <span class="text-xs font-bold uppercase tracking-wider text-slate-400">Faithfulness</span>
                        <div class="text-3xl font-extrabold text-white mt-1">__X_FAITHFULNESS__%</div>
                        <span class="text-xs text-emerald-400 font-semibold flex items-center mt-1">
                            <i class="fa-solid fa-circle-check mr-1"></i>Peak Grounding Score
                        </span>
                    </div>
                    <div class="mt-4 space-y-1.5 text-xs text-slate-400">
                        <div class="flex justify-between"><span>Naive RAG:</span> <span class="font-mono text-slate-300 font-bold">__N_FAITHFULNESS__%</span></div>
                        <div class="flex justify-between"><span>Hybrid RAG:</span> <span class="font-mono text-slate-300 font-bold">__H_FAITHFULNESS__%</span></div>
                        <div class="flex justify-between"><span>X-CDS (0.10):</span> <span class="font-mono text-brand-500 font-bold">__X_FAITHFULNESS__%</span></div>
                    </div>
                </div>

                <!-- Relevancy Card -->
                <div class="glass rounded-2xl p-6 flex flex-col justify-between">
                    <div>
                        <span class="text-xs font-bold uppercase tracking-wider text-slate-400">Answer Relevancy</span>
                        <div class="text-3xl font-extrabold text-white mt-1">__X_RELEVANCY__%</div>
                        <span class="text-xs text-slate-400 font-semibold flex items-center mt-1">
                            Balanced Semantic Synthesis
                        </span>
                    </div>
                    <div class="mt-4 space-y-1.5 text-xs text-slate-400">
                        <div class="flex justify-between"><span>Naive RAG:</span> <span class="font-mono text-slate-300 font-bold">__N_RELEVANCY__%</span></div>
                        <div class="flex justify-between"><span>Hybrid RAG:</span> <span class="font-mono text-slate-300 font-bold">__H_RELEVANCY__%</span></div>
                        <div class="flex justify-between"><span>X-CDS (0.10):</span> <span class="font-mono text-brand-500 font-bold">__X_RELEVANCY__%</span></div>
                    </div>
                </div>

                <!-- Precision Card -->
                <div class="glass rounded-2xl p-6 flex flex-col justify-between">
                    <div>
                        <span class="text-xs font-bold uppercase tracking-wider text-slate-400">Context Precision</span>
                        <div class="text-3xl font-extrabold text-white mt-1">__H_PRECISION__%</div>
                        <span class="text-xs text-emerald-400 font-semibold flex items-center mt-1">
                            Advanced Retrieval Filter
                        </span>
                    </div>
                    <div class="mt-4 space-y-1.5 text-xs text-slate-400">
                        <div class="flex justify-between"><span>Naive RAG:</span> <span class="font-mono text-slate-300 font-bold">__N_PRECISION__%</span></div>
                        <div class="flex justify-between"><span>Hybrid RAG:</span> <span class="font-mono text-brand-500 font-bold">__H_PRECISION__%</span></div>
                        <div class="flex justify-between"><span>X-CDS (0.10):</span> <span class="font-mono text-slate-300 font-bold">__X_PRECISION__%</span></div>
                    </div>
                </div>

                <!-- Recall Card -->
                <div class="glass rounded-2xl p-6 flex flex-col justify-between">
                    <div>
                        <span class="text-xs font-bold uppercase tracking-wider text-slate-400">Context Recall</span>
                        <div class="text-3xl font-extrabold text-white mt-1">__H_RECALL__%</div>
                        <span class="text-xs text-emerald-400 font-semibold flex items-center mt-1">
                            High Coverage of Facts
                        </span>
                    </div>
                    <div class="mt-4 space-y-1.5 text-xs text-slate-400">
                        <div class="flex justify-between"><span>Naive RAG:</span> <span class="font-mono text-slate-300 font-bold">__N_RECALL__%</span></div>
                        <div class="flex justify-between"><span>Hybrid RAG:</span> <span class="font-mono text-brand-500 font-bold">__H_RECALL__%</span></div>
                        <div class="flex justify-between"><span>X-CDS (0.10):</span> <span class="font-mono text-slate-300 font-bold">__X_RECALL__%</span></div>
                    </div>
                </div>
            </section>

            <!-- Table Section -->
            <section class="glass rounded-3xl p-8 space-y-6">
                <div class="flex justify-between items-center flex-wrap gap-4">
                    <div>
                        <h2 class="text-2xl font-bold text-white">Clinical Queries Benchmarks</h2>
                        <p class="text-sm text-slate-400 mt-1">Select any case below to see raw output comparisons, ground truth, and context passages.</p>
                    </div>
                    <div class="flex items-center space-x-3 w-full md:w-auto">
                        <div class="relative w-full md:w-80">
                            <input type="text" id="search-input" oninput="filterQueries()" placeholder="Search clinical queries..." 
                                   class="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-900 border border-slate-800 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500">
                            <i class="fa-solid fa-search absolute left-3.5 top-3.5 text-slate-500 text-xs"></i>
                        </div>
                    </div>
                </div>

                <div class="overflow-x-auto rounded-xl border border-slate-800/80 bg-slate-950/20">
                    <table class="w-full border-collapse text-left text-sm text-slate-300">
                        <thead class="bg-slate-900/60 text-xs font-semibold uppercase tracking-wider text-slate-400">
                            <tr>
                                <th class="px-4 py-4">Clinical Question</th>
                                <th class="px-4 py-4 text-center">Faithfulness (N / H / X)</th>
                                <th class="px-4 py-4 text-center">Relevancy (N / H / X)</th>
                                <th class="px-4 py-4 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody id="queries-table-body" class="divide-y divide-slate-800/40">
                            <!-- Filled by JS -->
                        </tbody>
                    </table>
                </div>
            </section>
        </div>

        <!-- TAB: Threshold Sweep -->
        <div id="tab-sweep" class="hidden flex flex-col items-center justify-center p-8 glass rounded-3xl space-y-6">
            <div class="text-center">
                <h2 class="text-2xl font-bold text-white">Citation Overlap Parameter Sweep ($T_{min}$)</h2>
                <p class="text-sm text-slate-400 mt-1">Analysis of verification threshold strictly vs. natural semantic grounding ($N=100$ cases)</p>
            </div>
            <div class="p-4 bg-slate-900/40 rounded-2xl border border-slate-800">
                <img id="sweep-chart-img" class="max-w-3xl rounded-xl shadow-lg border border-slate-800/60" alt="Threshold Sweep Chart">
            </div>
        </div>

    </main>

    <!-- Side Drawer Details Panel -->
    <div id="detail-drawer" class="fixed inset-0 z-50 invisible transition-all duration-300" role="dialog" aria-modal="true">
        <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" onclick="closeDrawer()"></div>
        <div class="absolute right-0 top-0 bottom-0 w-full md:w-[85%] lg:w-[75%] glass border-l border-slate-800 text-slate-300 shadow-2xl p-8 flex flex-col justify-between overflow-y-auto transform translate-x-full transition-transform duration-300" id="drawer-content">
            
            <div class="space-y-6">
                <!-- Header -->
                <div class="flex justify-between items-start border-b border-slate-800/60 pb-4">
                    <div>
                        <span class="text-xs font-semibold tracking-wider uppercase text-brand-500">Evaluation Case Study</span>
                        <h3 class="text-xl font-bold text-white mt-1" id="drawer-question">Question Placeholder</h3>
                    </div>
                    <button onclick="closeDrawer()" class="text-slate-500 hover:text-white p-2 text-lg">
                        <i class="fa-solid fa-xmark"></i>
                    </button>
                </div>

                <!-- Ground Truth -->
                <div class="space-y-2">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center">
                        <i class="fa-solid fa-clipboard-check text-brand-500 mr-2"></i>Expert Ground Truth Recommendation
                    </h4>
                    <div class="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-sm italic text-slate-300" id="drawer-ground-truth">
                        Ground truth here...
                    </div>
                </div>

                <!-- Answers 3-Way Side-by-Side -->
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- Naive Answer -->
                    <div class="space-y-2">
                        <div class="flex justify-between items-center">
                            <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center">
                                <i class="fa-solid fa-bolt-lightning text-slate-500 mr-2"></i>1. Naive RAG (Dense Only)
                            </h4>
                            <span id="badge-naive" class="px-2 py-0.5 text-[10px] font-bold rounded bg-slate-800 text-slate-300 border border-slate-700">Score</span>
                        </div>
                        <div class="p-4 rounded-xl bg-slate-950/40 border border-slate-800/80 text-xs h-96 overflow-y-auto whitespace-pre-line leading-relaxed font-sans" id="drawer-naive-answer">
                            Naive answer...
                        </div>
                    </div>
                    <!-- Hybrid Answer -->
                    <div class="space-y-2">
                        <div class="flex justify-between items-center">
                            <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center">
                                <i class="fa-solid fa-layer-group text-slate-500 mr-2"></i>2. Hybrid RAG (No Guardrail)
                            </h4>
                            <span id="badge-hybrid" class="px-2 py-0.5 text-[10px] font-bold rounded bg-slate-800 text-slate-300 border border-slate-700">Score</span>
                        </div>
                        <div class="p-4 rounded-xl bg-slate-950/40 border border-slate-800/80 text-xs h-96 overflow-y-auto whitespace-pre-line leading-relaxed font-sans" id="drawer-hybrid-answer">
                            Hybrid answer...
                        </div>
                    </div>
                    <!-- X-CDS Answer -->
                    <div class="space-y-2">
                        <div class="flex justify-between items-center">
                            <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center">
                                <i class="fa-solid fa-circle-check text-brand-500 mr-2"></i>3. X-CDS RAG (Optimized 0.10)
                            </h4>
                            <span id="badge-xcds" class="px-2 py-0.5 text-[10px] font-bold rounded bg-brand-500/10 text-brand-500 border border-brand-500/20">Score</span>
                        </div>
                        <div class="p-4 rounded-xl bg-slate-900/40 border border-brand-500/20 text-xs h-96 overflow-y-auto whitespace-pre-line leading-relaxed font-sans shadow-inner" id="drawer-xcds-answer">
                            X-CDS answer...
                        </div>
                    </div>
                </div>

                <!-- Retrieved Contexts -->
                <div class="space-y-3">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center">
                        <i class="fa-solid fa-book-open text-brand-500 mr-2"></i>Retrieved Literature Contexts
                    </h4>
                    <div class="space-y-3 max-h-64 overflow-y-auto pr-1" id="drawer-contexts">
                        <!-- Passages go here -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Data Injection -->
    <script>
        const detailsData = __MERGED_DETAILS__;
        const chartBase64 = "__CHART_BASE64__";

        function switchTab(tab) {
            const comparisonTab = document.getElementById("tab-comparison");
            const sweepTab = document.getElementById("tab-sweep");
            const comparisonBtn = document.getElementById("tab-comparison-btn");
            const sweepBtn = document.getElementById("tab-sweep-btn");

            if (tab === "comparison") {
                comparisonTab.classList.remove("hidden");
                sweepTab.classList.add("hidden");
                comparisonBtn.className = "pb-3 text-sm font-bold border-b-2 border-brand-500 text-white transition-all";
                sweepBtn.className = "pb-3 text-sm font-semibold border-b-2 border-transparent text-slate-400 hover:text-white transition-all";
            } else {
                comparisonTab.classList.add("hidden");
                sweepTab.classList.remove("hidden");
                comparisonBtn.className = "pb-3 text-sm font-semibold border-b-2 border-transparent text-slate-400 hover:text-white transition-all";
                sweepBtn.className = "pb-3 text-sm font-bold border-b-2 border-brand-500 text-white transition-all";
                
                // Lazy-load base64 image
                const chartImg = document.getElementById("sweep-chart-img");
                if (chartBase64 && !chartImg.src) {
                    chartImg.src = "data:image/png;base64," + chartBase64;
                }
            }
        }
        
        function fillTable(data) {
            const tbody = document.getElementById("queries-table-body");
            tbody.innerHTML = "";
            
            data.forEach((item, index) => {
                const getScoreStr = (scores) => {
                    const f = scores.faithfulness !== undefined && scores.faithfulness !== null ? scores.faithfulness.toFixed(2) : "--";
                    const r = scores.answer_relevancy !== undefined && scores.answer_relevancy !== null ? scores.answer_relevancy.toFixed(2) : "--";
                    return { f, r };
                };

                const n = getScoreStr(item.naive_scores);
                const h = getScoreStr(item.hybrid_scores);
                const x = getScoreStr(item.xcds_scores);

                const tr = document.createElement("tr");
                tr.className = "hover:bg-slate-900/20 transition-colors";
                tr.innerHTML = `
                    <td class="px-4 py-4 font-medium text-white max-w-sm truncate">${item.question}</td>
                    <td class="px-4 py-4 text-center">
                        <span class="text-xs text-slate-500 font-mono">${n.f}</span>
                        <span class="text-xs text-slate-700 px-1">/</span>
                        <span class="text-xs text-slate-400 font-mono">${h.f}</span>
                        <span class="text-xs text-slate-700 px-1">/</span>
                        <span class="text-xs font-bold text-brand-500 font-mono">${x.f}</span>
                    </td>
                    <td class="px-4 py-4 text-center">
                        <span class="text-xs text-slate-500 font-mono">${n.r}</span>
                        <span class="text-xs text-slate-700 px-1">/</span>
                        <span class="text-xs text-slate-400 font-mono">${h.r}</span>
                        <span class="text-xs text-slate-700 px-1">/</span>
                        <span class="text-xs font-bold text-brand-500 font-mono">${x.r}</span>
                    </td>
                    <td class="px-4 py-4 text-right">
                        <button onclick="viewDetails(${index})" class="text-brand-500 hover:text-white px-3 py-1.5 rounded-lg border border-brand-500/20 bg-brand-500/5 hover:bg-brand-500 font-semibold text-xs tracking-tight transition-all">
                            <i class="fa-solid fa-magnifying-glass-plus mr-1"></i>Analyze
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }

        function viewDetails(index) {
            const item = detailsData[index];
            document.getElementById("drawer-question").innerText = item.question;
            document.getElementById("drawer-ground-truth").innerText = item.ground_truth;
            document.getElementById("drawer-naive-answer").innerText = item.naive_answer;
            document.getElementById("drawer-hybrid-answer").innerText = item.hybrid_answer;
            document.getElementById("drawer-xcds-answer").innerText = item.xcds_answer;

            // Badges
            const formatBadge = (scores) => {
                const f = scores.faithfulness !== undefined && scores.faithfulness !== null ? scores.faithfulness.toFixed(2) : "--";
                const r = scores.answer_relevancy !== undefined && scores.answer_relevancy !== null ? scores.answer_relevancy.toFixed(2) : "--";
                return `F: ${f} | R: ${r}`;
            };

            document.getElementById("badge-naive").innerText = formatBadge(item.naive_scores);
            document.getElementById("badge-hybrid").innerText = formatBadge(item.hybrid_scores);
            document.getElementById("badge-xcds").innerText = formatBadge(item.xcds_scores);

            // Context Passages
            const contextsContainer = document.getElementById("drawer-contexts");
            contextsContainer.innerHTML = "";
            
            if (item.contexts && item.contexts.length > 0) {
                item.contexts.forEach((passage, pIdx) => {
                    const div = document.createElement("div");
                    div.className = "p-3.5 rounded-xl bg-slate-900/40 border border-slate-800/60 text-xs text-slate-300 leading-relaxed";
                    div.innerHTML = `<span class="font-bold text-brand-500 block mb-1">Passage ${pIdx + 1}:</span>${passage}`;
                    contextsContainer.appendChild(div);
                });
            } else {
                contextsContainer.innerHTML = `<p class="text-xs text-slate-500 italic">No retrieved passages available.</p>`;
            }

            // Show drawer
            const drawer = document.getElementById("detail-drawer");
            const content = document.getElementById("drawer-content");
            drawer.classList.remove("invisible");
            setTimeout(() => {
                content.classList.remove("translate-x-full");
            }, 10);
        }

        function closeDrawer() {
            const drawer = document.getElementById("detail-drawer");
            const content = document.getElementById("drawer-content");
            content.classList.add("translate-x-full");
            setTimeout(() => {
                drawer.classList.add("invisible");
            }, 300);
        }

        function filterQueries() {
            const query = document.getElementById("search-input").value.toLowerCase();
            const filtered = detailsData.filter(item => 
                item.question.toLowerCase().includes(query) || 
                item.xcds_answer.toLowerCase().includes(query) || 
                item.hybrid_answer.toLowerCase().includes(query) ||
                item.naive_answer.toLowerCase().includes(query)
            );
            fillTable(filtered);
        }

        // Init
        document.addEventListener("DOMContentLoaded", () => {
            fillTable(detailsData);
        });
    </script>
</body>
</html>
"""

    # Replacements
    html_content = html_template.replace("__DATA_COUNT__", str(len(merged_details)))
    
    # X-CDS 0.10 Peak Metrics
    html_content = html_content.replace("__X_FAITHFULNESS__", f"{x_metrics.get('faithfulness', 0)*100:.1f}")
    html_content = html_content.replace("__X_RELEVANCY__", f"{x_metrics.get('answer_relevancy', 0)*100:.1f}")
    html_content = html_content.replace("__X_PRECISION__", f"{x_metrics.get('context_precision', 0)*100:.1f}")
    html_content = html_content.replace("__X_RECALL__", f"{x_metrics.get('context_recall', 0)*100:.1f}")

    # Hybrid Metrics
    html_content = html_content.replace("__H_FAITHFULNESS__", f"{h_metrics.get('faithfulness', 0)*100:.1f}")
    html_content = html_content.replace("__H_RELEVANCY__", f"{h_metrics.get('answer_relevancy', 0)*100:.1f}")
    html_content = html_content.replace("__H_PRECISION__", f"{h_metrics.get('context_precision', 0)*100:.1f}")
    html_content = html_content.replace("__H_RECALL__", f"{h_metrics.get('context_recall', 0)*100:.1f}")

    # Naive Metrics
    html_content = html_content.replace("__N_FAITHFULNESS__", f"{n_metrics.get('faithfulness', 0)*100:.1f}")
    html_content = html_content.replace("__N_RELEVANCY__", f"{n_metrics.get('answer_relevancy', 0)*100:.1f}")
    html_content = html_content.replace("__N_PRECISION__", f"{n_metrics.get('context_precision', 0)*100:.1f}")
    html_content = html_content.replace("__N_RECALL__", f"{n_metrics.get('context_recall', 0)*100:.1f}")

    html_content = html_content.replace("__CHART_BASE64__", chart_base64)
    html_content = html_content.replace("__MERGED_DETAILS__", json.dumps(merged_details, ensure_ascii=False))

    print(f"Writing dashboard to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print("Success! Evaluation dashboard created.")


if __name__ == "__main__":
    main()
