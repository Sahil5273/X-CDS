import json
from pathlib import Path
from scipy.stats import wilcoxon

data_dir = Path("d:/X-CDS/data")

def compute_stats():
    # Load Ragas reports
    naive_path = data_dir / "naive_ragas_report.json"
    hybrid_path = data_dir / "baseline_ragas_report.json"
    xcds_path = data_dir / "ragas_report_t10.json"
    
    if not all(p.exists() for p in [naive_path, hybrid_path, xcds_path]):
        raise FileNotFoundError("Ragas reports not found in data/ directory.")
        
    with naive_path.open("r", encoding="utf-8") as f:
        naive = json.load(f)
    with hybrid_path.open("r", encoding="utf-8") as f:
        hybrid = json.load(f)
    with xcds_path.open("r", encoding="utf-8") as f:
        xcds = json.load(f)
        
    xcds_details = xcds.get("details", [])
    
    results = {}
    
    for label, ref_details in [("naive", naive.get("details", [])), ("hybrid", hybrid.get("details", []))]:
        ref_faith = []
        xcds_faith = []
        improved = 0
        tied = 0
        worse = 0
        
        for r, x in zip(ref_details, xcds_details, strict=True):
            r_val = r.get("scores", {}).get("faithfulness")
            x_val = x.get("scores", {}).get("faithfulness")
            
            if r_val is None: r_val = 1.0
            if x_val is None: x_val = 1.0
                
            ref_faith.append(r_val)
            xcds_faith.append(x_val)
            
            if x_val > r_val:
                improved += 1
            elif x_val == r_val:
                tied += 1
            else:
                worse += 1
                
        mean_ref = sum(ref_faith) / len(ref_faith)
        mean_xcds = sum(xcds_faith) / len(xcds_faith)
        mean_diff = mean_xcds - mean_ref
        
        stat, p_val = wilcoxon(xcds_faith, ref_faith)
        p_val_float = float(p_val)
        
        results[label] = {
            "comparison_label": f"X-CDS vs {label.capitalize()} RAG",
            "ref_mean_faithfulness": float(mean_ref),
            "xcds_mean_faithfulness": float(mean_xcds),
            "mean_difference": float(mean_diff),
            "improved_count": int(improved),
            "tied_count": int(tied),
            "worse_count": int(worse),
            "wilcoxon_p_value": p_val_float,
            "statistically_significant_05": bool(p_val_float < 0.05)
        }
        
    output_path = data_dir / "faithfulness_wilcoxon_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Wilcoxon analysis saved to {output_path}:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    compute_stats()
