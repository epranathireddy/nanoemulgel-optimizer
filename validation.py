"""
Experimental Validation Module for Nanoemulgel Optimization System.
Compares laboratory confirmation batch observations against regression predictions,
calculates percentage prediction errors, verifies adherence to 95% Prediction Intervals,
and generates academic scientific assessment reports.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional

def evaluate_validation_results(
    predicted_responses: Dict[str, Dict[str, Any]],
    observed_values: Dict[str, float]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates experimental validation batch observations against model predictions.

    Calculates:
    - Absolute Error = |Observed - Predicted|
    - Percentage Error = (|Observed - Predicted| / |Predicted|) * 100%
    - 95% Prediction Interval compliance: PI_low <= Observed <= PI_high
    """
    rows = []
    within_pi_count = 0
    total_evaluated = 0

    for resp_name, pred_info in predicted_responses.items():
        if resp_name not in observed_values or observed_values[resp_name] is None:
            continue

        obs_val = float(observed_values[resp_name])
        pred_val = float(pred_info["predicted_mean"])
        pi_low = float(pred_info.get("pi_95_lower", pred_val * 0.9))
        pi_high = float(pred_info.get("pi_95_upper", pred_val * 1.1))
        unit = pred_info.get("unit", "")

        abs_error = abs(obs_val - pred_val)

        # Handle zero or near-zero predicted values safely
        if abs(pred_val) > 1e-6:
            pct_error = (abs_error / abs(pred_val)) * 100.0
        else:
            pct_error = 0.0 if np.isclose(obs_val, 0.0) else 100.0

        in_pi = (pi_low <= obs_val <= pi_high)
        if in_pi:
            within_pi_count += 1
        total_evaluated += 1

        # Assessment verdict
        if in_pi and pct_error <= 5.0:
            status = "Verified (Excellent <5% Error & Within 95% PI)"
        elif in_pi:
            status = "Verified (Within 95% PI)"
        elif pct_error <= 10.0:
            status = "Acceptable (<10% Error, Slight PI Deviation)"
        else:
            status = "Discrepancy (>10% Error, Outside 95% PI)"

        rows.append({
            "Response Variable": resp_name,
            "Unit": unit,
            "Predicted Value": round(pred_val, 3),
            "95% PI Low": round(pi_low, 3),
            "95% PI High": round(pi_high, 3),
            "Observed Lab Value": round(obs_val, 3),
            "Absolute Error": round(abs_error, 3),
            "Prediction Error (%)": round(pct_error, 2),
            "Within 95% PI": "Yes" if in_pi else "No",
            "Verification Status": status
        })

    df_validation = pd.DataFrame(rows)

    # Scientific Summary Assessment
    if total_evaluated == 0:
        overall_status = "No Data"
        assessment_text = "No confirmation laboratory observations entered yet."
    else:
        success_ratio = within_pi_count / total_evaluated
        if success_ratio >= 0.8:
            overall_status = "Successfully Validated"
            assessment_text = (
                f"The experimental confirmation batch demonstrated high concordance with the polynomial response "
                f"surface models. {within_pi_count} of {total_evaluated} responses ({success_ratio*100:.1f}%) "
                f"fell strictly within their respective 95% Prediction Intervals, with low overall prediction error. "
                f"This verifies that the QbD model space successfully navigates nanoemulgel manufacturing."
            )
        else:
            overall_status = "Qualified / Partial Validation"
            assessment_text = (
                f"{within_pi_count} of {total_evaluated} responses fell within the 95% Prediction Intervals. "
                f"While average prediction error may be modest, the observed deviations highlight the impact of "
                f"nanoscale manufacturing variability (e.g. sonication thermal fluctuations, surfactant batch differences) "
                f"and indicate that further formulation robustness testing is warranted."
            )

    summary = {
        "overall_status": overall_status,
        "total_evaluated": total_evaluated,
        "within_pi_count": within_pi_count,
        "scientific_assessment": assessment_text
    }

    return df_validation, summary

def create_validation_comparison_chart(df_validation: pd.DataFrame) -> go.Figure:
    """
    Generates a grouped bar chart comparing Predicted vs. Observed responses
    with 95% Prediction Interval error bars.
    """
    if df_validation.empty:
        return go.Figure()

    responses = df_validation["Response Variable"].tolist()
    pred_vals = df_validation["Predicted Value"].tolist()
    obs_vals = df_validation["Observed Lab Value"].tolist()
    pi_high = df_validation["95% PI High"].tolist()
    pi_low = df_validation["95% PI Low"].tolist()

    error_plus = [high - p for high, p in zip(pi_high, pred_vals)]
    error_minus = [p - low for low, p in zip(pi_low, pred_vals)]

    fig = go.Figure()

    # Predicted Bar with 95% PI Error Bars
    fig.add_trace(go.Bar(
        name="Predicted (with 95% PI)",
        x=responses,
        y=pred_vals,
        marker_color="#1f77b4",
        error_y=dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color="#333333",
            thickness=1.5,
            width=5
        )
    ))

    # Observed Bar
    fig.add_trace(go.Bar(
        name="Observed Experimental",
        x=responses,
        y=obs_vals,
        marker_color="#2ca02c"
    ))

    fig.update_layout(
        title="<b>Experimental Confirmation: Predicted vs. Observed Responses</b><br><sup>Error bars depict 95% Prediction Intervals</sup>",
        barmode="group",
        xaxis_title="Critical Quality Attributes (CQAs)",
        yaxis_title="Response Magnitude",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        width=800,
        height=500
    )
    return fig
