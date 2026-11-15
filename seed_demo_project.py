"""
Seeds the database with a complete, fully populated research demo project
so the user immediately sees all 10 stages populated upon launching.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_FACTORS, DEFAULT_RESPONSES
from database import (
    save_project, save_design, save_experimental_data,
    save_model, save_optimization, save_validation
)
from doe import create_design_matrix
from utils import generate_synthetic_benchmark_dataset
from analysis import fit_rsm_model
from optimization import optimize_formulation
from validation import evaluate_validation_results

def seed():
    proj = {
        "name": "Diclofenac Nanoemulgel Optimization",
        "drug_name": "Diclofenac Sodium",
        "dosage_form": "Nanoemulgel",
        "route": "Topical / Transdermal",
        "oil_phase": "Clove Oil",
        "surfactant": "Tween 80 (Polysorbate 80)",
        "cosurfactant": "Transcutol P",
        "gelling_agent": "Carbopol 940 (1.0% w/w)",
        "other_excipients": "Triethanolamine (q.s. pH 6.0), Methylparaben, Purified Water"
    }

    pid = save_project(proj, DEFAULT_FACTORS, DEFAULT_RESPONSES)
    df_design, _, _ = create_design_matrix("Box-Behnken Design (BBD)", DEFAULT_FACTORS, num_center=5, random_seed=42)
    des_id = save_design(pid, "Box-Behnken Design (BBD)", 3, 5, 42, df_design)

    df_data = generate_synthetic_benchmark_dataset(DEFAULT_FACTORS, DEFAULT_RESPONSES, random_seed=42)
    save_experimental_data(des_id, df_data)

    factor_codes = ["A", "B", "C"]
    X_coded = df_data[[f"Factor_{c}_Coded" for c in factor_codes]].values
    models_all = {}

    for r in DEFAULT_RESPONSES:
        col = f"{r['name']} ({r['unit']})"
        if col in df_data.columns:
            y = df_data[col].values
            m = fit_rsm_model(X_coded, y, DEFAULT_FACTORS, r["name"], r["unit"], "Quadratic")
            models_all[r["name"]] = m
            save_model(pid, r["name"], "Quadratic", m["coded_equation"], m["coefficients"], m["anova"], m["fit_statistics"], m["diagnostics"])

    criteria = {
        "Particle Size": {"goal": "minimize", "low": 50.0, "high": 200.0, "importance": 5},
        "Polydispersity Index (PDI)": {"goal": "minimize", "low": 0.1, "high": 0.45, "importance": 4},
        "Zeta Potential": {"goal": "range", "low": -40.0, "high": -20.0, "importance": 3},
        "Entrapment Efficiency": {"goal": "maximize", "low": 70.0, "high": 95.0, "importance": 5},
        "Drug Content": {"goal": "target", "low": 95.0, "high": 105.0, "target": 100.0, "importance": 4},
        "Viscosity": {"goal": "range", "low": 12000.0, "high": 25000.0, "importance": 3},
        "Spreadability": {"goal": "range", "low": 4.5, "high": 6.5, "importance": 3},
        "In-vitro Drug Release (8h)": {"goal": "maximize", "low": 70.0, "high": 95.0, "importance": 5}
    }

    opt_res = optimize_formulation(models_all, DEFAULT_FACTORS, criteria)
    save_optimization(pid, opt_res["composite_desirability"], opt_res["optimal_actual_factors"], opt_res["predicted_responses"], opt_res["candidate_solutions"], criteria)

    obs_sim = {r_name: pinfo["predicted_mean"] * 1.015 for r_name, pinfo in opt_res["predicted_responses"].items()}
    df_val, val_sum = evaluate_validation_results(opt_res["predicted_responses"], obs_sim)
    save_validation(pid, opt_res["predicted_responses"], obs_sim, df_val.to_dict(orient="records"), val_sum["overall_status"], val_sum["scientific_assessment"])

    print(f"Project seeded successfully with ID: {pid}")

if __name__ == "__main__":
    seed()
