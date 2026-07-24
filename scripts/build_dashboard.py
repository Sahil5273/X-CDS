"""Build a self-contained, premium HTML/JS evaluation comparison dashboard."""

from __future__ import annotations

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


def main() -> None:
    data_dir = Path("data")
    output_path = Path("docs/evaluation_dashboard.html")

    print("Loading evaluation reports...")
    xcds_report = load_json_report(data_dir / "ragas_report.json")
    baseline_report = load_json_report(data_dir / "baseline_ragas_report.json")

    print("Loading retrieved contexts...")
    contexts_map = load_materialized_contexts(data_dir / "materialized_predictions.jsonl")

    # Merge data by question
    merged_details: list[dict[str, any]] = []

    # Map baseline details by question
    baseline_map = {item["question"]: item for item in baseline_report.get("details", [])}

    for xcds_item in xcds_report.get("details", []):
        question = xcds_item["question"]
        baseline_item = baseline_map.get(question, {})

        merged_record = {
            "question": question,
            "ground_truth": xcds_item["ground_truth"],
            "xcds_answer": xcds_item["answer"],
            "xcds_scores": xcds_item.get("scores", {}),
            "baseline_answer": baseline_item.get("answer", "N/A"),
            "baseline_scores": baseline_item.get("scores", {}),
            "contexts": contexts_map.get(question, []),
        }
        merged_details.append(merged_record)

    # Metrics
    x_metrics = xcds_report.get("metrics", {})
    b_metrics = baseline_report.get("metrics", {})

    xf = x_metrics.get("faithfulness", 0) * 100
    bf = b_metrics.get("faithfulness", 0) * 100
    df = xf - bf

    xr = x_metrics.get("answer_relevancy", 0) * 100
    br = b_metrics.get("answer_relevancy", 0) * 100
    dr = xr - br

    xp = x_metrics.get("context_precision", 0) * 100
    bp = b_metrics.get("context_precision", 0) * 100
    dp = xp - bp

    xc = x_metrics.get("context_recall", 0) * 100
    bc = b_metrics.get("context_recall", 0) * 100
    dc = xc - bc

    html_template = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X-CDS RAG Evaluation Benchmarks</title>
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
                <p class="text-xs text-slate-400">Clinical RAG Evaluator Dashboard</p>
            </div>
        </div>
        <div class="flex items-center space-x-4">
            <span class="px-3 py-1 text-xs font-semibold rounded-full bg-brand-500/10 text-brand-500 border border-brand-500/20">
                <i class="fa-solid fa-server mr-1.5"></i>Dataset: Zika Virus (N=__DATA_COUNT__)
            </span>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 mt-8 space-y-8">

        <!-- Metrics Overview Grid -->
        <section class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <!-- Faithfulness -->
            <div class="glass rounded-2xl p-6 relative overflow-hidden flex flex-col justify-between">
                <div>
                    <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Faithfulness</span>
                    <div class="flex items-baseline space-x-2 mt-2">
                        <span class="text-3xl font-extrabold text-white">__X_FAITHFULNESS__%</span>
                        <span class="text-xs text-emerald-400 font-semibold flex items-center">
                            <i class="fa-solid fa-caret-up mr-0.5"></i>
                            +__D_FAITHFULNESS__%
                        </span>
                    </div>
                </div>
                <div class="mt-4 space-y-2">
                    <div class="flex justify-between text-xs text-slate-400">
                        <span>Baseline: __B_FAITHFULNESS__%</span>
                        <span>X-CDS: __X_FAITHFULNESS__%</span>
                    </div>
                    <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden flex">
                        <div class="h-full bg-slate-500" style="width: __B_FAITHFULNESS__%"></div>
                        <div class="h-full bg-brand-500" style="width: __D_FAITHFULNESS__%"></div>
                    </div>
                </div>
            </div>

            <!-- Answer Relevancy -->
            <div class="glass rounded-2xl p-6 relative overflow-hidden flex flex-col justify-between">
                <div>
                    <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Answer Relevancy</span>
                    <div class="flex items-baseline space-x-2 mt-2">
                        <span class="text-3xl font-extrabold text-white">__X_RELEVANCY__%</span>
                        <span class="text-xs text-emerald-400 font-semibold flex items-center">
                            <i class="fa-solid fa-caret-up mr-0.5"></i>
                            +__D_RELEVANCY__%
                        </span>
                    </div>
                </div>
                <div class="mt-4 space-y-2">
                    <div class="flex justify-between text-xs text-slate-400">
                        <span>Baseline: __B_RELEVANCY__%</span>
                        <span>X-CDS: __X_RELEVANCY__%</span>
                    </div>
                    <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden flex">
                        <div class="h-full bg-slate-500" style="width: __B_RELEVANCY__%"></div>
                        <div class="h-full bg-brand-500" style="width: __D_RELEVANCY__%"></div>
                    </div>
                </div>
            </div>

            <!-- Context Precision -->
            <div class="glass rounded-2xl p-6 relative overflow-hidden flex flex-col justify-between">
                <div>
                    <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Context Precision</span>
                    <div class="flex items-baseline space-x-2 mt-2">
                        <span class="text-3xl font-extrabold text-white">__X_PRECISION__%</span>
                        <span class="text-xs text-emerald-400 font-semibold flex items-center">
                            <i class="fa-solid fa-caret-up mr-0.5"></i>
                            +__D_PRECISION__%
                        </span>
                    </div>
                </div>
                <div class="mt-4 space-y-2">
                    <div class="flex justify-between text-xs text-slate-400">
                        <span>Baseline: __B_PRECISION__%</span>
                        <span>X-CDS: __X_PRECISION__%</span>
                    </div>
                    <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden flex">
                        <div class="h-full bg-slate-500" style="width: __B_PRECISION__%"></div>
                        <div class="h-full bg-brand-500" style="width: __D_PRECISION__%"></div>
                    </div>
                </div>
            </div>

            <!-- Context Recall -->
            <div class="glass rounded-2xl p-6 relative overflow-hidden flex flex-col justify-between">
                <div>
                    <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Context Recall</span>
                    <div class="flex items-baseline space-x-2 mt-2">
                        <span class="text-3xl font-extrabold text-white">__X_RECALL__%</span>
                        <span class="text-xs text-emerald-400 font-semibold flex items-center">
                            <i class="fa-solid fa-caret-up mr-0.5"></i>
                            +__D_RECALL__%
                        </span>
                    </div>
                </div>
                <div class="mt-4 space-y-2">
                    <div class="flex justify-between text-xs text-slate-400">
                        <span>Baseline: __B_RECALL__%</span>
                        <span>X-CDS: __X_RECALL__%</span>
                    </div>
                    <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden flex">
                        <div class="h-full bg-slate-500" style="width: __B_RECALL__%"></div>
                        <div class="h-full bg-brand-500" style="width: __D_RECALL__%"></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Main Query Section -->
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

            <!-- Table -->
            <div class="overflow-x-auto rounded-xl border border-slate-800/80 bg-slate-950/20">
                <table class="w-full border-collapse text-left text-sm text-slate-300">
                    <thead class="bg-slate-900/60 text-xs font-semibold uppercase tracking-wider text-slate-400">
                        <tr>
                            <th class="px-4 py-4">Clinical Question</th>
                            <th class="px-4 py-4 text-center">Faithfulness (B / X)</th>
                            <th class="px-4 py-4 text-center">Relevancy (B / X)</th>
                            <th class="px-4 py-4 text-center">Precision (B / X)</th>
                            <th class="px-4 py-4 text-center">Recall (B / X)</th>
                            <th class="px-4 py-4 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody id="queries-table-body" class="divide-y divide-slate-800/40">
                        <!-- Filled by JS -->
                    </tbody>
                </table>
            </div>
        </section>
    </main>

    <!-- Side Drawer Details Panel -->
    <div id="detail-drawer" class="fixed inset-0 z-50 invisible transition-all duration-300" role="dialog" aria-modal="true">
        <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" onclick="closeDrawer()"></div>
        <div class="absolute right-0 top-0 bottom-0 w-full md:w-[75%] lg:w-[65%] glass border-l border-slate-800 text-slate-300 shadow-2xl p-8 flex flex-col justify-between overflow-y-auto transform translate-x-full transition-transform duration-300" id="drawer-content">
            
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

                <!-- Scores Comparison -->
                <div class="grid grid-cols-4 gap-4 bg-slate-900/30 p-4 rounded-2xl border border-slate-800/40">
                    <div class="text-center flex flex-col items-center">
                        <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Faithfulness</span>
                        <div id="score-faithfulness"></div>
                    </div>
                    <div class="text-center flex flex-col items-center">
                        <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Relevancy</span>
                        <div id="score-relevancy"></div>
                    </div>
                    <div class="text-center flex flex-col items-center">
                        <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Precision</span>
                        <div id="score-precision"></div>
                    </div>
                    <div class="text-center flex flex-col items-center">
                        <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Recall</span>
                        <div id="score-recall"></div>
                    </div>
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

                <!-- Answers Side-by-Side -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="space-y-2">
                        <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center">
                            <i class="fa-solid fa-bolt-lightning text-slate-500 mr-2"></i>Baseline RAG Answer
                        </h4>
                        <div class="p-4 rounded-xl bg-slate-950/40 border border-slate-800/80 text-sm h-64 overflow-y-auto whitespace-pre-line" id="drawer-baseline-answer">
                            Baseline answer...
                        </div>
                    </div>
                    <div class="space-y-2">
                        <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center">
                            <i class="fa-solid fa-circle-check text-brand-500 mr-2"></i>X-CDS Answer (Stateful Guardrails)
                        </h4>
                        <div class="p-4 rounded-xl bg-slate-900/40 border border-brand-500/10 text-sm h-64 overflow-y-auto whitespace-pre-line border-brand-500/20 shadow-inner" id="drawer-xcds-answer">
                            X-CDS answer...
                        </div>
                    </div>
                </div>

                <!-- Retrieved Contexts -->
                <div class="space-y-3">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center">
                        <i class="fa-solid fa-book-open text-brand-500 mr-2"></i>Retrieved literature passages
                    </h4>
                    <div class="space-y-3 max-h-80 overflow-y-auto pr-1" id="drawer-contexts">
                        <!-- Passages go here -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Data Injection -->
    <script>
        const detailsData = __MERGED_DETAILS__;
        
        function fillTable(data) {
            const tbody = document.getElementById("queries-table-body");
            tbody.innerHTML = "";
            
            data.forEach((item, index) => {
                const f_b = item.baseline_scores.faithfulness !== undefined && item.baseline_scores.faithfulness !== null ? item.baseline_scores.faithfulness.toFixed(2) : "--";
                const f_x = item.xcds_scores.faithfulness !== undefined && item.xcds_scores.faithfulness !== null ? item.xcds_scores.faithfulness.toFixed(2) : "--";
                
                const r_b = item.baseline_scores.answer_relevancy !== undefined && item.baseline_scores.answer_relevancy !== null ? item.baseline_scores.answer_relevancy.toFixed(2) : "--";
                const r_x = item.xcds_scores.answer_relevancy !== undefined && item.xcds_scores.answer_relevancy !== null ? item.xcds_scores.answer_relevancy.toFixed(2) : "--";

                const p_b = item.baseline_scores.context_precision !== undefined && item.baseline_scores.context_precision !== null ? item.baseline_scores.context_precision.toFixed(2) : "--";
                const p_x = item.xcds_scores.context_precision !== undefined && item.xcds_scores.context_precision !== null ? item.xcds_scores.context_precision.toFixed(2) : "--";

                const c_b = item.baseline_scores.context_recall !== undefined && item.baseline_scores.context_recall !== null ? item.baseline_scores.context_recall.toFixed(2) : "--";
                const c_x = item.xcds_scores.context_recall !== undefined && item.xcds_scores.context_recall !== null ? item.xcds_scores.context_recall.toFixed(2) : "--";

                const tr = document.createElement("tr");
                tr.className = "hover:bg-slate-900/20 transition-colors";
                tr.innerHTML = `
                    <td class="px-4 py-4 font-medium text-white max-w-xs truncate">${item.question}</td>
                    <td class="px-4 py-4 text-center">
                        <span class="text-xs text-slate-400 font-mono">${f_b}</span>
                        <span class="text-xs text-slate-600 px-1">/</span>
                        <span class="text-xs font-bold text-brand-500 font-mono">${f_x}</span>
                    </td>
                    <td class="px-4 py-4 text-center">
                        <span class="text-xs text-slate-400 font-mono">${r_b}</span>
                        <span class="text-xs text-slate-600 px-1">/</span>
                        <span class="text-xs font-bold text-brand-500 font-mono">${r_x}</span>
                    </td>
                    <td class="px-4 py-4 text-center">
                        <span class="text-xs text-slate-400 font-mono">${p_b}</span>
                        <span class="text-xs text-slate-600 px-1">/</span>
                        <span class="text-xs font-bold text-brand-500 font-mono">${p_x}</span>
                    </td>
                    <td class="px-4 py-4 text-center">
                        <span class="text-xs text-slate-400 font-mono">${c_b}</span>
                        <span class="text-xs text-slate-600 px-1">/</span>
                        <span class="text-xs font-bold text-brand-500 font-mono">${c_x}</span>
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
            document.getElementById("drawer-baseline-answer").innerText = item.baseline_answer;
            document.getElementById("drawer-xcds-answer").innerText = item.xcds_answer;

            // Scores
            const showScore = (b, x) => {
                const bs = b !== undefined && b !== null ? b.toFixed(2) : "--";
                const xs = x !== undefined && x !== null ? x.toFixed(2) : "--";
                return `
                    <div class="mt-1 flex flex-col items-center space-y-1">
                        <div class="text-[11px] text-slate-400 font-semibold bg-slate-950/40 px-2 py-0.5 rounded border border-slate-800/60 w-24 flex justify-between">
                            <span>Base:</span> <span class="font-mono text-slate-300 font-bold">${bs}</span>
                        </div>
                        <div class="text-[11px] text-brand-400 font-semibold bg-brand-500/5 px-2 py-0.5 rounded border border-brand-500/10 w-24 flex justify-between">
                            <span>X-CDS:</span> <span class="font-mono text-brand-500 font-bold">${xs}</span>
                        </div>
                    </div>
                `;
            };

            document.getElementById("score-faithfulness").innerHTML = showScore(item.baseline_scores.faithfulness, item.xcds_scores.faithfulness);
            document.getElementById("score-relevancy").innerHTML = showScore(item.baseline_scores.answer_relevancy, item.xcds_scores.answer_relevancy);
            document.getElementById("score-precision").innerHTML = showScore(item.baseline_scores.context_precision, item.xcds_scores.context_precision);
            document.getElementById("score-recall").innerHTML = showScore(item.baseline_scores.context_recall, item.xcds_scores.context_recall);

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
                item.baseline_answer.toLowerCase().includes(query)
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
    html_content = html_content.replace("__X_FAITHFULNESS__", f"{xf:.1f}")
    html_content = html_content.replace("__B_FAITHFULNESS__", f"{bf:.1f}")
    html_content = html_content.replace("__D_FAITHFULNESS__", f"{df:+.1f}")

    html_content = html_content.replace("__X_RELEVANCY__", f"{xr:.1f}")
    html_content = html_content.replace("__B_RELEVANCY__", f"{br:.1f}")
    html_content = html_content.replace("__D_RELEVANCY__", f"{dr:+.1f}")

    html_content = html_content.replace("__X_PRECISION__", f"{xp:.1f}")
    html_content = html_content.replace("__B_PRECISION__", f"{bp:.1f}")
    html_content = html_content.replace("__D_PRECISION__", f"{dp:+.1f}")

    html_content = html_content.replace("__X_RECALL__", f"{xc:.1f}")
    html_content = html_content.replace("__B_RECALL__", f"{bc:.1f}")
    html_content = html_content.replace("__D_RECALL__", f"{dc:+.1f}")

    html_content = html_content.replace("__MERGED_DETAILS__", json.dumps(merged_details, ensure_ascii=False))

    print(f"Writing dashboard to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print("Success! Evaluation dashboard created.")


if __name__ == "__main__":
    main()
