"""
Visualization Module for Nanoemulgel Formulation Optimization System.
Generates 2D contour plots, interactive 3D response surface topologies using Plotly,
actual vs. predicted parity plots, residual diagnostic suites, and perturbation curves.
Also generates Matplotlib static figures for embedding into PDF research reports.
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

from doe import code_to_actual, actual_to_code

def create_contour_plot(
    model_dict: Dict[str, Any],
    factors: List[Dict[str, Any]],
    factor_x_code: str,
    factor_y_code: str,
    hold_values_actual: Dict[str, float],
    grid_points: int = 50
) -> go.Figure:
    """
    Generates an interactive 2D contour plot in Plotly for two active factors
    while holding other factors at specified constant values.
    """
    model = model_dict["model_object"]
    response_name = model_dict["response_name"]
    response_unit = model_dict["response_unit"]
    selected_features = model_dict["feature_names"]
    factor_codes = [f["code"] for f in factors]
    factors_by_code = {f["code"]: f for f in factors}

    f_x = factors_by_code[factor_x_code]
    f_y = factors_by_code[factor_y_code]

    x_actual = np.linspace(f_x["low"], f_x["high"], grid_points)
    y_actual = np.linspace(f_y["low"], f_y["high"], grid_points)
    X_mesh, Y_mesh = np.meshgrid(x_actual, y_actual)

    # Convert to coded grid
    X_coded = (X_mesh - f_x["center"]) / ((f_x["high"] - f_x["low"]) / 2.0)
    Y_coded = (Y_mesh - f_y["center"]) / ((f_y["high"] - f_y["low"]) / 2.0)

    # Hold values in coded units
    coded_hold = {}
    for code, act_val in hold_values_actual.items():
        fc = factors_by_code[code]
        coded_hold[code] = (act_val - fc["center"]) / ((fc["high"] - fc["low"]) / 2.0)

    # Predict Z across mesh
    Z = np.zeros_like(X_mesh)
    for i in range(grid_points):
        for j in range(grid_points):
            row_dict = {}
            for code in factor_codes:
                if code == factor_x_code:
                    row_dict[code] = X_coded[i, j]
                elif code == factor_y_code:
                    row_dict[code] = Y_coded[i, j]
                else:
                    row_dict[code] = coded_hold[code]

            # Interactions and quadratic
            for a_idx, a_code in enumerate(factor_codes):
                for b_idx in range(a_idx + 1, len(factor_codes)):
                    b_code = factor_codes[b_idx]
                    row_dict[f"{a_code}*{b_code}"] = row_dict[a_code] * row_dict[b_code]
                row_dict[f"{a_code}^2"] = row_dict[a_code] ** 2

            vec = [1.0] + [row_dict[feat] for feat in selected_features]
            Z[i, j] = float(model.params.values @ np.array(vec, dtype=float))

    fig = go.Figure(data=go.Contour(
        z=Z,
        x=x_actual,
        y=y_actual,
        colorscale="Viridis",
        contours=dict(
            coloring='heatmap',
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorbar=dict(title=f"{response_name} ({response_unit})")
    ))

    hold_str = ", ".join([f"{c}={hold_values_actual[c]:.2f} {factors_by_code[c]['unit']}" for c in hold_values_actual if c not in [factor_x_code, factor_y_code]])
    title_text = f"<b>2D Contour: {response_name} ({response_unit})</b>"
    if hold_str:
        title_text += f"<br><sup>Holding: {hold_str}</sup>"

    fig.update_layout(
        title=title_text,
        xaxis_title=f"{f_x['code']}: {f_x['name']} ({f_x['unit']})",
        yaxis_title=f"{f_y['code']}: {f_y['name']} ({f_y['unit']})",
        template="plotly_white",
        width=750,
        height=550
    )
    return fig

def create_3d_surface_plot(
    model_dict: Dict[str, Any],
    factors: List[Dict[str, Any]],
    factor_x_code: str,
    factor_y_code: str,
    hold_values_actual: Dict[str, float],
    grid_points: int = 40,
    experimental_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Generates an interactive 3D Response Surface topology using Plotly
    with overlaid actual experimental points.
    """
    model = model_dict["model_object"]
    response_name = model_dict["response_name"]
    response_unit = model_dict["response_unit"]
    selected_features = model_dict["feature_names"]
    factor_codes = [f["code"] for f in factors]
    factors_by_code = {f["code"]: f for f in factors}

    f_x = factors_by_code[factor_x_code]
    f_y = factors_by_code[factor_y_code]

    x_actual = np.linspace(f_x["low"], f_x["high"], grid_points)
    y_actual = np.linspace(f_y["low"], f_y["high"], grid_points)
    X_mesh, Y_mesh = np.meshgrid(x_actual, y_actual)

    X_coded = (X_mesh - f_x["center"]) / ((f_x["high"] - f_x["low"]) / 2.0)
    Y_coded = (Y_mesh - f_y["center"]) / ((f_y["high"] - f_y["low"]) / 2.0)

    coded_hold = {}
    for code, act_val in hold_values_actual.items():
        fc = factors_by_code[code]
        coded_hold[code] = (act_val - fc["center"]) / ((fc["high"] - fc["low"]) / 2.0)

    Z = np.zeros_like(X_mesh)
    for i in range(grid_points):
        for j in range(grid_points):
            row_dict = {}
            for code in factor_codes:
                if code == factor_x_code:
                    row_dict[code] = X_coded[i, j]
                elif code == factor_y_code:
                    row_dict[code] = Y_coded[i, j]
                else:
                    row_dict[code] = coded_hold[code]

            for a_idx, a_code in enumerate(factor_codes):
                for b_idx in range(a_idx + 1, len(factor_codes)):
                    b_code = factor_codes[b_idx]
                    row_dict[f"{a_code}*{b_code}"] = row_dict[a_code] * row_dict[b_code]
                row_dict[f"{a_code}^2"] = row_dict[a_code] ** 2

            vec = [1.0] + [row_dict[feat] for feat in selected_features]
            Z[i, j] = float(model.params.values @ np.array(vec, dtype=float))

    fig = go.Figure()

    # 3D Surface
    fig.add_trace(go.Surface(
        z=Z,
        x=x_actual,
        y=y_actual,
        colorscale="Turbo",
        colorbar=dict(title=f"{response_name} ({response_unit})"),
        opacity=0.92,
        name="RSM Surface"
    ))

    # Overlay experimental design points if provided
    if experimental_df is not None:
        col_x = [c for c in experimental_df.columns if c.startswith(f"{f_x['code']}_")]
        col_y = [c for c in experimental_df.columns if c.startswith(f"{f_y['code']}_")]
        col_resp = [c for c in experimental_df.columns if response_name in c]

        if col_x and col_y and col_resp:
            pts_x = experimental_df[col_x[0]].values
            pts_y = experimental_df[col_y[0]].values
            pts_z = experimental_df[col_resp[0]].values
            valid = ~np.isnan(pts_z)

            fig.add_trace(go.Scatter3d(
                x=pts_x[valid],
                y=pts_y[valid],
                z=pts_z[valid],
                mode="markers",
                marker=dict(size=5, color="red", symbol="circle", line=dict(color="black", width=1)),
                name="Experimental Observations"
            ))

    hold_str = ", ".join([f"{c}={hold_values_actual[c]:.2f} {factors_by_code[c]['unit']}" for c in hold_values_actual if c not in [factor_x_code, factor_y_code]])
    title_text = f"<b>3D Response Surface: {response_name}</b>"
    if hold_str:
        title_text += f"<br><sup>Holding: {hold_str}</sup>"

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title=f"{f_x['code']}: {f_x['name']} ({f_x['unit']})",
            yaxis_title=f"{f_y['code']}: {f_y['name']} ({f_y['unit']})",
            zaxis_title=f"{response_name} ({response_unit})"
        ),
        template="plotly_white",
        width=850,
        height=650
    )
    return fig

def create_actual_vs_predicted_plot(model_dict: Dict[str, Any]) -> go.Figure:
    """
    Plots experimentally observed vs regression-predicted response values
    with an ideal 45-degree reference parity line.
    """
    actual = np.array(model_dict["diagnostics"]["actual"])
    predicted = np.array(model_dict["diagnostics"]["predicted"])
    response_name = model_dict["response_name"]
    response_unit = model_dict["response_unit"]
    r2 = model_dict["fit_statistics"]["R2"]

    min_val = min(np.min(actual), np.min(predicted))
    max_val = max(np.max(actual), np.max(predicted))
    padding = (max_val - min_val) * 0.08
    line_x = [min_val - padding, max_val + padding]

    fig = go.Figure()
    # Parity Line
    fig.add_trace(go.Scatter(
        x=line_x, y=line_x,
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        name="Ideal Parity (Y = Y_hat)"
    ))
    # Points
    fig.add_trace(go.Scatter(
        x=actual, y=predicted,
        mode="markers",
        marker=dict(size=9, color="#1f77b4", symbol="circle", line=dict(color="black", width=1)),
        text=[f"Obs: {a:.2f}<br>Pred: {p:.2f}" for a, p in zip(actual, predicted)],
        name="Formulation Runs"
    ))

    fig.update_layout(
        title=f"<b>Actual vs. Predicted: {response_name}</b><br><sup>R² = {r2:.4f}</sup>",
        xaxis_title=f"Actual Experimental Value ({response_unit})",
        yaxis_title=f"Model Predicted Value ({response_unit})",
        template="plotly_white",
        width=650,
        height=500
    )
    return fig

def create_residual_diagnostics_suite(model_dict: Dict[str, Any]) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """
    Constructs 4 standard regression diagnostic plots:
    1. Residuals vs. Predicted
    2. Normal Probability Q-Q Plot
    3. Residual Distribution Histogram
    4. Cook's Distance Plot (Outlier Detection)
    """
    predicted = np.array(model_dict["diagnostics"]["predicted"])
    stud_res = np.array(model_dict["diagnostics"]["studentized_residuals"])
    residuals = np.array(model_dict["diagnostics"]["residuals"])
    cooks_d = np.array(model_dict["diagnostics"]["cooks_distance"])
    runs = np.arange(1, len(predicted) + 1)

    # 1. Residuals vs Predicted
    fig_res_pred = go.Figure()
    fig_res_pred.add_hline(y=0, line_dash="dash", line_color="black")
    fig_res_pred.add_hline(y=3, line_dash="dot", line_color="red")
    fig_res_pred.add_hline(y=-3, line_dash="dot", line_color="red")
    fig_res_pred.add_trace(go.Scatter(
        x=predicted, y=stud_res,
        mode="markers",
        marker=dict(size=8, color="#2ca02c", line=dict(color="black", width=1)),
        name="Studentized Residuals"
    ))
    fig_res_pred.update_layout(
        title="<b>Residuals vs. Predicted</b><br><sup>Checks Homoscedasticity / Constant Error Variance</sup>",
        xaxis_title="Predicted Response",
        yaxis_title="Internally Studentized Residuals",
        template="plotly_white"
    )

    # 2. Normal Probability Q-Q Plot
    sorted_res = np.sort(stud_res)
    n = len(sorted_res)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))

    fig_qq = go.Figure()
    # Parity line
    qq_line = np.linspace(np.min(theoretical_quantiles), np.max(theoretical_quantiles), 100)
    fig_qq.add_trace(go.Scatter(x=qq_line, y=qq_line, mode="lines", line=dict(color="red", dash="dash"), name="Normal Line"))
    fig_qq.add_trace(go.Scatter(
        x=theoretical_quantiles, y=sorted_res,
        mode="markers",
        marker=dict(size=8, color="#9467bd", line=dict(color="black", width=1)),
        name="Observed Quantiles"
    ))
    fig_qq.update_layout(
        title=f"<b>Normal Probability Q-Q Plot</b><br><sup>Shapiro-Wilk p = {model_dict['diagnostics']['shapiro_p']}</sup>",
        xaxis_title="Theoretical Quantiles (Normal Distribution)",
        yaxis_title="Studentized Residuals",
        template="plotly_white"
    )

    # 3. Residual Distribution Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=residuals,
        nbinsx=10,
        marker=dict(color="#17becf", line=dict(color="white", width=1)),
        name="Residuals"
    ))
    fig_hist.update_layout(
        title="<b>Residual Distribution</b>",
        xaxis_title="Residual",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    # 4. Cook's Distance Plot
    fig_cooks = go.Figure()
    fig_cooks.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Threshold (D=1.0)")
    fig_cooks.add_trace(go.Bar(
        x=runs, y=cooks_d,
        marker=dict(color="#e377c2"),
        name="Cook's Distance"
    ))
    fig_cooks.update_layout(
        title="<b>Cook's Distance by Run Order</b><br><sup>Identifies Highly Influential Experimental Outliers</sup>",
        xaxis_title="Run Number",
        yaxis_title="Cook's Distance",
        template="plotly_white"
    )

    return fig_res_pred, fig_qq, fig_hist, fig_cooks

def create_perturbation_plot(
    model_dict: Dict[str, Any],
    factors: List[Dict[str, Any]]
) -> go.Figure:
    """
    Plots the response perturbation curves showing how the response varies
    as each individual factor moves from -1 to +1 (coded scale) while all
    other factors are fixed at the reference center point (0).
    """
    model = model_dict["model_object"]
    response_name = model_dict["response_name"]
    response_unit = model_dict["response_unit"]
    selected_features = model_dict["feature_names"]
    factor_codes = [f["code"] for f in factors]

    coded_points = np.linspace(-1.0, 1.0, 50)
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, f in enumerate(factors):
        y_curve = []
        for dev in coded_points:
            row_dict = {c: 0.0 for c in factor_codes}
            row_dict[f["code"]] = dev

            for a_i, a_c in enumerate(factor_codes):
                for b_i in range(a_i + 1, len(factor_codes)):
                    b_c = factor_codes[b_i]
                    row_dict[f"{a_c}*{b_c}"] = row_dict[a_c] * row_dict[b_c]
                row_dict[f"{a_c}^2"] = row_dict[a_c] ** 2

            vec = [1.0] + [row_dict[feat] for feat in selected_features]
            y_curve.append(float(model.params.values @ np.array(vec, dtype=float)))

        fig.add_trace(go.Scatter(
            x=coded_points,
            y=y_curve,
            mode="lines",
            line=dict(width=3, color=colors[idx % len(colors)]),
            name=f"Factor {f['code']}: {f['name']}"
        ))

    fig.add_vline(x=0.0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"<b>Perturbation Plot: {response_name}</b><br><sup>Deviation from Reference Center Formulation (Coded -1 to +1)</sup>",
        xaxis_title="Deviation from Reference Center (Coded Factor Level)",
        yaxis_title=f"{response_name} ({response_unit})",
        template="plotly_white",
        width=700,
        height=500
    )
    return fig

def generate_static_contour_png(
    model_dict: Dict[str, Any],
    factors: List[Dict[str, Any]],
    factor_x_code: str,
    factor_y_code: str,
    hold_values_actual: Dict[str, float]
) -> bytes:
    """
    Renders a static 2D contour plot via Matplotlib and returns raw PNG bytes
    for direct integration into PDF reports.
    """
    model = model_dict["model_object"]
    response_name = model_dict["response_name"]
    response_unit = model_dict["response_unit"]
    selected_features = model_dict["feature_names"]
    factor_codes = [f["code"] for f in factors]
    factors_by_code = {f["code"]: f for f in factors}

    f_x = factors_by_code[factor_x_code]
    f_y = factors_by_code[factor_y_code]

    x_actual = np.linspace(f_x["low"], f_x["high"], 60)
    y_actual = np.linspace(f_y["low"], f_y["high"], 60)
    X_mesh, Y_mesh = np.meshgrid(x_actual, y_actual)

    X_coded = (X_mesh - f_x["center"]) / ((f_x["high"] - f_x["low"]) / 2.0)
    Y_coded = (Y_mesh - f_y["center"]) / ((f_y["high"] - f_y["low"]) / 2.0)

    coded_hold = {}
    for code, act_val in hold_values_actual.items():
        fc = factors_by_code[code]
        coded_hold[code] = (act_val - fc["center"]) / ((fc["high"] - fc["low"]) / 2.0)

    Z = np.zeros_like(X_mesh)
    for i in range(60):
        for j in range(60):
            row_dict = {}
            for code in factor_codes:
                if code == factor_x_code:
                    row_dict[code] = X_coded[i, j]
                elif code == factor_y_code:
                    row_dict[code] = Y_coded[i, j]
                else:
                    row_dict[code] = coded_hold[code]

            for a_idx, a_code in enumerate(factor_codes):
                for b_idx in range(a_idx + 1, len(factor_codes)):
                    b_code = factor_codes[b_idx]
                    row_dict[f"{a_code}*{b_code}"] = row_dict[a_code] * row_dict[b_code]
                row_dict[f"{a_code}^2"] = row_dict[a_code] ** 2

            vec = [1.0] + [row_dict[feat] for feat in selected_features]
            Z[i, j] = float(model.params.values @ np.array(vec, dtype=float))

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=150)
    cs = ax.contourf(X_mesh, Y_mesh, Z, levels=14, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(f"{response_name} ({response_unit})", fontsize=10)
    lines = ax.contour(X_mesh, Y_mesh, Z, levels=14, colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(lines, inline=True, fontsize=8, fmt="%.1f")

    ax.set_title(f"Contour Plot: {response_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{f_x['code']}: {f_x['name']} ({f_x['unit']})", fontsize=10)
    ax.set_ylabel(f"{f_y['code']}: {f_y['name']} ({f_y['unit']})", fontsize=10)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
