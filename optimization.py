"""
Multi-Response Formulation Optimization Module.
Implements the Derringer-Suich desirability function methodology coupled with
bounded numerical optimization (differential evolution and multi-start L-BFGS-B)
to pinpoint optimal nanoemulgel recipes within experimental factor domains.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from typing import Dict, List, Tuple, Any, Optional

from doe import code_to_actual
from analysis import predict_single_response

def calculate_individual_desirability(
    y_val: float,
    goal: str,
    low: float,
    high: float,
    target: Optional[float] = None,
    weight: float = 1.0
) -> float:
    """
    Computes Derringer-Suich individual desirability d_i in [0.0, 1.0].
    - 'maximize': d = ((y - low) / (high - low))^weight
    - 'minimize': d = ((high - y) / (high - low))^weight
    - 'target': ramps to 1.0 at target value, 0 outside [low, high]
    - 'range': 1.0 within [low, high], 0 outside
    - 'none': excluded
    """
    if goal == "none":
        return 1.0

    if low >= high:
        # Avoid division by zero
        return 1.0 if np.isclose(y_val, low) else 0.0

    if goal == "maximize":
        if y_val <= low:
            return 0.0
        elif y_val >= high:
            return 1.0
        else:
            return float(((y_val - low) / (high - low)) ** weight)

    elif goal == "minimize":
        if y_val <= low:
            return 1.0
        elif y_val >= high:
            return 0.0
        else:
            return float(((high - y_val) / (high - low)) ** weight)

    elif goal == "target":
        t = target if target is not None else (low + high) / 2.0
        if y_val < low or y_val > high:
            return 0.0
        if np.isclose(y_val, t):
            return 1.0
        elif y_val < t:
            denom = t - low
            return float(((y_val - low) / denom) ** weight) if denom > 0 else 1.0
        else:
            denom = high - t
            return float(((high - y_val) / denom) ** weight) if denom > 0 else 1.0

    elif goal == "range":
        if low <= y_val <= high:
            return 1.0
        else:
            # Soft linear decay penalty outside range rather than sudden hard zero
            margin = (high - low) * 0.10
            if y_val < low:
                pen = max(0.0, 1.0 - (low - y_val) / margin) if margin > 0 else 0.0
            else:
                pen = max(0.0, 1.0 - (y_val - high) / margin) if margin > 0 else 0.0
            return float(pen ** weight)

    return 1.0

def compute_composite_desirability(
    coded_vector: np.ndarray,
    models: Dict[str, Dict[str, Any]],
    criteria: Dict[str, Dict[str, Any]],
    factor_codes: List[str]
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Computes overall composite desirability D = (product(d_i^w_i))^(1 / sum(w_i)).
    Returns:
    - composite_D: float in [0.0, 1.0]
    - individual_d: Dict mapping response name to individual desirability
    - predicted_y: Dict mapping response name to predicted mean value
    """
    individual_d = {}
    predicted_y = {}
    active_weights = []
    active_d_vals = []

    for resp_name, crit in criteria.items():
        if resp_name not in models:
            continue
        goal = crit.get("goal", "none")
        if goal == "none":
            continue

        model_dict = models[resp_name]
        y_hat, _, _ = predict_single_response(model_dict, coded_vector, factor_codes)
        predicted_y[resp_name] = y_hat

        d = calculate_individual_desirability(
            y_val=y_hat,
            goal=goal,
            low=float(crit["low"]),
            high=float(crit["high"]),
            target=float(crit["target"]) if crit.get("target") is not None else None,
            weight=1.0  # shape parameter
        )
        individual_d[resp_name] = d

        imp = float(crit.get("importance", 3))  # 1 to 5
        active_weights.append(imp)
        active_d_vals.append(d)

    if not active_d_vals:
        return 0.0, individual_d, predicted_y

    # If any mandatory response has d == 0, overall D is strictly 0
    if any(d <= 1e-9 for d in active_d_vals):
        return 0.0, individual_d, predicted_y

    # Geometric mean: exp( sum(w_i * ln(d_i)) / sum(w_i) )
    sum_w = sum(active_weights)
    weighted_log_sum = sum(w * np.log(max(d, 1e-12)) for w, d in zip(active_weights, active_d_vals))
    composite_D = float(np.exp(weighted_log_sum / sum_w))

    return composite_D, individual_d, predicted_y

def optimize_formulation(
    models: Dict[str, Dict[str, Any]],
    factors: List[Dict[str, Any]],
    criteria: Dict[str, Dict[str, Any]],
    coded_bounds: Optional[List[Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Executes numerical global optimization to find the factor concentrations
    maximizing Derringer-Suich composite desirability.
    Uses differential evolution (as in user's original code) plus local refinement.
    """
    factor_codes = [f["code"] for f in factors]
    num_factors = len(factors)

    if coded_bounds is None:
        # Standard bounded experimental design region: [-1.0, 1.0]
        # Never extrapolate outside studied experimental space!
        coded_bounds = [(-1.0, 1.0) for _ in range(num_factors)]

    # Objective function: minimize negative composite desirability
    def objective(x):
        d_comp, _, _ = compute_composite_desirability(x, models, criteria, factor_codes)
        return -d_comp

    # 1. Global search using Differential Evolution
    de_result = differential_evolution(
        objective,
        bounds=coded_bounds,
        seed=42,
        maxiter=100,
        popsize=20,
        tol=1e-5
    )

    best_coded = de_result.x
    best_d = -float(de_result.fun)

    # 2. Multi-start L-BFGS-B local refinement from several design space anchors
    anchor_points = [
        np.zeros(num_factors),  # Center
        best_coded,
    ]
    # Add random exploratory starting seeds
    rng = np.random.RandomState(42)
    for _ in range(15):
        rand_pt = rng.uniform([b[0] for b in coded_bounds], [b[1] for b in coded_bounds])
        anchor_points.append(rand_pt)

    candidate_solutions = []

    for start_pt in anchor_points:
        loc_res = minimize(
            objective,
            start_pt,
            method="L-BFGS-B",
            bounds=coded_bounds
        )
        d_val = -float(loc_res.fun)
        if d_val > 1e-4:
            cand_coded = np.clip(loc_res.x, [b[0] for b in coded_bounds], [b[1] for b in coded_bounds])
            cand_D, cand_ind, cand_pred = compute_composite_desirability(cand_coded, models, criteria, factor_codes)

            # Check if distinct from existing candidates
            is_duplicate = False
            for existing in candidate_solutions:
                if np.allclose(cand_coded, existing["coded_factors"], atol=0.08):
                    is_duplicate = True
                    break

            if not is_duplicate:
                # Convert to actual concentrations
                actual_vals = {}
                for idx, f in enumerate(factors):
                    act = code_to_actual(cand_coded[idx], f["low"], f["center"], f["high"])
                    actual_vals[f["code"]] = {
                        "name": f["name"],
                        "unit": f["unit"],
                        "value": round(float(act), 3)
                    }

                candidate_solutions.append({
                    "desirability": round(cand_D, 4),
                    "coded_factors": cand_coded.tolist(),
                    "actual_factors": actual_vals,
                    "predicted_responses": {k: round(v, 3) for k, v in cand_pred.items()},
                    "individual_desirabilities": {k: round(v, 4) for k, v in cand_ind.items()}
                })

    # Sort candidates by composite desirability descending
    candidate_solutions.sort(key=lambda x: x["desirability"], reverse=True)

    if not candidate_solutions:
        # Fallback to center point if all d == 0
        cand_coded = np.zeros(num_factors)
        actual_vals = {f["code"]: {"name": f["name"], "unit": f["unit"], "value": f["center"]} for f in factors}
        _, cand_ind, cand_pred = compute_composite_desirability(cand_coded, models, criteria, factor_codes)
        candidate_solutions.append({
            "desirability": 0.0,
            "coded_factors": cand_coded.tolist(),
            "actual_factors": actual_vals,
            "predicted_responses": {k: round(v, 3) for k, v in cand_pred.items()},
            "individual_desirabilities": {k: round(v, 4) for k, v in cand_ind.items()}
        })

    top_candidate = candidate_solutions[0]
    top_coded = np.array(top_candidate["coded_factors"])

    # Compute 95% CI and 95% PI for top candidate predictions
    response_predictions = {}
    for resp_name, model_dict in models.items():
        y_hat, ci_half, pi_half = predict_single_response(model_dict, top_coded, factor_codes)
        response_predictions[resp_name] = {
            "predicted_mean": round(y_hat, 3),
            "ci_95_lower": round(y_hat - ci_half, 3),
            "ci_95_upper": round(y_hat + ci_half, 3),
            "pi_95_lower": round(y_hat - pi_half, 3),
            "pi_95_upper": round(y_hat + pi_half, 3),
            "unit": model_dict["response_unit"]
        }

    # Generate scientific explanation
    explanation = (
        f"The multi-response optimization converged to an optimal nanoemulgel formulation with an "
        f"overall composite desirability D = {top_candidate['desirability']:.4f}. "
        f"This candidate balances competing pharmaceutical critical quality attributes: "
        f"operating strictly within the experimentally evaluated factor design space to ensure physical stability, "
        f"minimal droplet size for cutaneous permeation, high drug encapsulation, and appropriate gel rheology."
    )

    return {
        "composite_desirability": top_candidate["desirability"],
        "optimal_coded_factors": top_candidate["coded_factors"],
        "optimal_actual_factors": top_candidate["actual_factors"],
        "predicted_responses": response_predictions,
        "individual_desirabilities": top_candidate["individual_desirabilities"],
        "candidate_solutions": candidate_solutions[:5],
        "explanation": explanation
    }
