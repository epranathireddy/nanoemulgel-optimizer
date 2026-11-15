"""
Statistical Analysis and Response Surface Methodology (RSM) Module.
Fits second-order polynomial regression models, computes full ANOVA tables
with Lack of Fit and Pure Error decomposition, calculates R², Adj R², Pred R²
via PRESS/leverage statistics, Adequate Precision, and generates diagnostic metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Dict, List, Tuple, Any, Optional

def build_polynomial_features(X_coded: np.ndarray, factor_codes: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs full quadratic polynomial terms for coded variables:
    Intercept, Linear (A, B, C...), 2-Factor Interactions (AB, AC, BC...), Quadratic (A², B², C²...).
    """
    n_samples, n_factors = X_coded.shape
    df_poly = pd.DataFrame(index=range(n_samples))
    feature_names = []

    # Linear terms
    for i, code in enumerate(factor_codes):
        col_name = code
        df_poly[col_name] = X_coded[:, i]
        feature_names.append(col_name)

    # 2-Factor Interaction terms (i < j)
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            col_name = f"{factor_codes[i]}*{factor_codes[j]}"
            df_poly[col_name] = X_coded[:, i] * X_coded[:, j]
            feature_names.append(col_name)

    # Quadratic terms
    for i, code in enumerate(factor_codes):
        col_name = f"{code}^2"
        df_poly[col_name] = X_coded[:, i] ** 2
        feature_names.append(col_name)

    return df_poly, feature_names

def compute_lack_of_fit(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, ss_residual: float, df_residual: int) -> Tuple[float, int, float, float, int, float, Optional[float], Optional[float]]:
    """
    Decomposes residual sum of squares into Lack of Fit (LoF) and Pure Error (PE).
    Pure error is determined from true replicates (identical factor combinations, e.g. center points).
    """
    n = len(y)
    # Find unique factor rows (replicates)
    # Round to avoid floating point precision differences
    X_rounded = np.round(X, 4)
    unique_rows, inverse_indices, counts = np.unique(X_rounded, axis=0, return_inverse=True, return_counts=True)

    ss_pure_error = 0.0
    df_pure_error = 0

    for idx, count in enumerate(counts):
        if count > 1:
            replicate_mask = (inverse_indices == idx)
            replicate_y = y[replicate_mask]
            mean_y = np.mean(replicate_y)
            ss_pure_error += float(np.sum((replicate_y - mean_y) ** 2))
            df_pure_error += (count - 1)

    df_lof = df_residual - df_pure_error
    ss_lof = max(0.0, ss_residual - ss_pure_error)

    if df_pure_error > 0 and ss_pure_error > 1e-12:
        ms_pure_error = ss_pure_error / df_pure_error
        ms_lof = ss_lof / df_lof if df_lof > 0 else 0.0
        f_lof = ms_lof / ms_pure_error if ms_pure_error > 0 else 0.0
        p_lof = float(1.0 - stats.f.cdf(f_lof, df_lof, df_pure_error))
    else:
        ms_pure_error = 0.0
        ms_lof = ss_lof / df_lof if df_lof > 0 else 0.0
        f_lof = None
        p_lof = None

    return ss_lof, df_lof, ms_lof, ss_pure_error, df_pure_error, ms_pure_error, f_lof, p_lof

def fit_rsm_model(
    X_coded: np.ndarray,
    y: np.ndarray,
    factors: List[Dict[str, Any]],
    response_name: str,
    response_unit: str,
    model_type: str = "Quadratic"
) -> Dict[str, Any]:
    """
    Fits Response Surface Methodology regression model using statsmodels OLS,
    calculates ANOVA table, fit quality metrics (R², Adj R², Pred R², Adequate Precision),
    diagnostic statistics, and generates equations in both coded and actual units.
    """
    y = np.asarray(y, dtype=float)
    valid_mask = ~np.isnan(y)
    if np.sum(valid_mask) < len(factors) + 3:
        raise ValueError(f"Insufficient valid data points to fit a model for {response_name}.")

    X_coded = X_coded[valid_mask]
    y = y[valid_mask]
    n_samples = len(y)
    factor_codes = [f["code"] for f in factors]

    # Construct design matrix
    df_poly, feature_names = build_polynomial_features(X_coded, factor_codes)

    # Filter features based on model_type
    if model_type == "Linear":
        selected_features = factor_codes
    elif model_type == "2FI":
        selected_features = [f for f in feature_names if "^2" not in f]
    else:  # Quadratic (Full 2nd-order polynomial)
        selected_features = feature_names

    X_selected = df_poly[selected_features]
    X_with_const = sm.add_constant(X_selected, has_constant='add')

    # Fit OLS Model
    model = sm.OLS(y, X_with_const).fit()

    y_pred = model.fittedvalues.values
    residuals = model.resid.values

    # Degrees of freedom and SS components
    p_params = len(model.params)  # including intercept
    df_model = p_params - 1
    df_residual = n_samples - p_params
    df_total = n_samples - 1

    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    ss_residual = float(np.sum(residuals ** 2))
    ss_model = max(0.0, ss_total - ss_residual)

    ms_model = ss_model / df_model if df_model > 0 else 0.0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0.0

    f_model = ms_model / ms_residual if ms_residual > 0 else 0.0
    p_model = float(1.0 - stats.f.cdf(f_model, df_model, df_residual)) if df_residual > 0 else 0.0

    # Linear, Interaction, and Quadratic partitioned Sum of Squares
    # Linear SS
    linear_cols = [c for c in selected_features if c in factor_codes]
    ss_linear = 0.0
    df_linear = len(linear_cols)
    if df_linear > 0:
        ols_lin = sm.OLS(y, sm.add_constant(df_poly[linear_cols])).fit()
        ss_linear = float(ols_lin.ess)

    # Interaction SS
    inter_cols = [c for c in selected_features if "*" in c]
    ss_inter = 0.0
    df_inter = len(inter_cols)

    # Quadratic SS
    quad_cols = [c for c in selected_features if "^2" in c]
    ss_quad = 0.0
    df_quad = len(quad_cols)

    if model_type == "Quadratic" and (df_inter > 0 or df_quad > 0):
        ols_lin_inter = sm.OLS(y, sm.add_constant(df_poly[linear_cols + inter_cols])).fit()
        ss_inter = max(0.0, float(ols_lin_inter.ess - ss_linear))
        ss_quad = max(0.0, float(ss_model - ols_lin_inter.ess))

    # Lack of fit calculation
    ss_lof, df_lof, ms_lof, ss_pure, df_pure, ms_pure, f_lof, p_lof = compute_lack_of_fit(
        X_coded, y, y_pred, ss_residual, df_residual
    )

    # PRESS and Predicted R² via leverage formula
    # H = X (X^T X)^(-1) X^T
    try:
        X_mat = X_with_const.values
        hat_matrix = X_mat @ np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T
        leverage = np.clip(np.diag(hat_matrix), 0.0, 0.9999)
        press_residuals = residuals / (1.0 - leverage)
        press = float(np.sum(press_residuals ** 2))
        r2_pred = float(max(0.0, 1.0 - (press / ss_total))) if ss_total > 0 else 0.0
    except Exception:
        press = float(ss_residual)
        r2_pred = float(model.rsquared)
        leverage = np.zeros(n_samples)

    r2 = float(model.rsquared)
    r2_adj = float(model.rsquared_adj)
    r2_diff = abs(r2_adj - r2_pred)

    # Adequate Precision = (max(Y_pred) - min(Y_pred)) / sqrt(p * MS_res / n)
    denom = np.sqrt((p_params * ms_residual) / n_samples) if n_samples > 0 else 1.0
    adeq_precision = float((np.max(y_pred) - np.min(y_pred)) / denom) if denom > 0 else 0.0

    # Summary stats
    std_dev = float(np.sqrt(ms_residual))
    mean_y = float(np.mean(y))
    cv_percent = float((std_dev / abs(mean_y)) * 100.0) if abs(mean_y) > 1e-9 else 0.0

    # Model parameters table
    conf_int = model.conf_int()
    coefficients = {}
    for term in model.params.index:
        coefficients[term] = {
            "coefficient": float(model.params[term]),
            "std_error": float(model.bse[term]),
            "t_stat": float(model.tvalues[term]),
            "p_value": float(model.pvalues[term]),
            "ci_lower": float(conf_int.loc[term, 0]),
            "ci_upper": float(conf_int.loc[term, 1]),
            "significant": bool(model.pvalues[term] < 0.05)
        }

    # Construct Coded Equation String
    coded_terms = [f"{model.params['const']:.4f}"]
    for term in selected_features:
        coef = model.params[term]
        sign = "+" if coef >= 0 else "-"
        coded_terms.append(f"{sign} {abs(coef):.4f}*{term}")
    coded_equation = " ".join(coded_terms)

    # Diagnostics
    std_residuals = residuals / std_dev if std_dev > 0 else residuals
    studentized_residuals = residuals / (std_dev * np.sqrt(np.clip(1.0 - leverage, 1e-4, 1.0)))
    cooks_d = (studentized_residuals ** 2 / p_params) * (leverage / np.clip(1.0 - leverage, 1e-4, 1.0))

    # Normality test
    try:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
    except Exception:
        shapiro_stat, shapiro_p = 1.0, 1.0

    # Autocorrelation (Durbin-Watson)
    try:
        dw_stat = float(sm.stats.stattools.durbin_watson(residuals))
    except Exception:
        dw_stat = 2.0

    # Scientific Quality Assessment Flags
    warnings = []
    if p_model >= 0.05:
        warnings.append("CRITICAL: The regression model is not statistically significant (p >= 0.05).")
    if p_lof is not None and p_lof < 0.05:
        warnings.append("WARNING: Lack of Fit is statistically significant (p < 0.05), indicating model deficiency.")
    if r2_diff > 0.20:
        warnings.append(f"WARNING: The difference between Adj R² ({r2_adj:.3f}) and Pred R² ({r2_pred:.3f}) is > 0.20.")
    if adeq_precision < 4.0:
        warnings.append(f"WARNING: Adequate Precision ({adeq_precision:.2f}) is below 4.0, indicating insufficient signal-to-noise.")
    if shapiro_p < 0.05:
        warnings.append("NOTE: Residuals depart from normal distribution (Shapiro-Wilk p < 0.05).")

    # ANOVA Data Table
    anova_table = [
        {"Source": "Model", "Sum of Squares": round(ss_model, 4), "df": int(df_model), "Mean Square": round(ms_model, 4), "F-Value": round(f_model, 3), "p-Value": round(p_model, 5), "Significance": "Significant" if p_model < 0.05 else "Not Significant"},
        {"Source": "  Linear", "Sum of Squares": round(ss_linear, 4), "df": int(df_linear), "Mean Square": round(ss_linear / df_linear, 4) if df_linear > 0 else 0, "F-Value": round((ss_linear / df_linear) / ms_residual, 3) if df_linear > 0 and ms_residual > 0 else 0, "p-Value": "-", "Significance": "-"},
    ]
    if df_inter > 0:
        ms_inter = ss_inter / df_inter
        anova_table.append({"Source": "  2FI (Interactions)", "Sum of Squares": round(ss_inter, 4), "df": int(df_inter), "Mean Square": round(ms_inter, 4), "F-Value": round(ms_inter / ms_residual, 3) if ms_residual > 0 else 0, "p-Value": "-", "Significance": "-"})
    if df_quad > 0:
        ms_quad = ss_quad / df_quad
        anova_table.append({"Source": "  Quadratic", "Sum of Squares": round(ss_quad, 4), "df": int(df_quad), "Mean Square": round(ms_quad, 4), "F-Value": round(ms_quad / ms_residual, 3) if ms_residual > 0 else 0, "p-Value": "-", "Significance": "-"})

    anova_table.append({"Source": "Residual", "Sum of Squares": round(ss_residual, 4), "df": int(df_residual), "Mean Square": round(ms_residual, 4), "F-Value": "-", "p-Value": "-", "Significance": "-"})

    if df_pure > 0 and f_lof is not None:
        anova_table.append({"Source": "  Lack of Fit", "Sum of Squares": round(ss_lof, 4), "df": int(df_lof), "Mean Square": round(ms_lof, 4), "F-Value": round(f_lof, 3), "p-Value": round(p_lof, 5), "Significance": "Significant (Bad)" if p_lof < 0.05 else "Not Significant (Good)"})
        anova_table.append({"Source": "  Pure Error", "Sum of Squares": round(ss_pure, 4), "df": int(df_pure), "Mean Square": round(ms_pure, 4), "F-Value": "-", "p-Value": "-", "Significance": "-"})

    anova_table.append({"Source": "Total Corrected", "Sum of Squares": round(ss_total, 4), "df": int(df_total), "Mean Square": "-", "F-Value": "-", "p-Value": "-", "Significance": "-"})

    return {
        "response_name": response_name,
        "response_unit": response_unit,
        "model_type": model_type,
        "coded_equation": f"{response_name} = {coded_equation}",
        "coefficients": coefficients,
        "anova": anova_table,
        "fit_statistics": {
            "R2": round(r2, 4),
            "Adjusted_R2": round(r2_adj, 4),
            "Predicted_R2": round(r2_pred, 4),
            "R2_Diff": round(r2_diff, 4),
            "Adequate_Precision": round(adeq_precision, 3),
            "Std_Dev": round(std_dev, 4),
            "Mean": round(mean_y, 4),
            "CV_Percent": round(cv_percent, 2),
            "PRESS": round(press, 4),
            "Model_F": round(f_model, 3),
            "Model_p": round(p_model, 5),
            "LoF_F": round(f_lof, 3) if f_lof is not None else None,
            "LoF_p": round(p_lof, 5) if p_lof is not None else None
        },
        "diagnostics": {
            "residuals": residuals.tolist(),
            "predicted": y_pred.tolist(),
            "actual": y.tolist(),
            "studentized_residuals": studentized_residuals.tolist(),
            "cooks_distance": cooks_d.tolist(),
            "leverage": leverage.tolist(),
            "shapiro_p": round(float(shapiro_p), 4),
            "durbin_watson": round(float(dw_stat), 3),
            "warnings": warnings
        },
        "model_object": model,
        "feature_names": selected_features
    }

def predict_single_response(model_dict: Dict[str, Any], coded_vector: np.ndarray, factor_codes: List[str]) -> Tuple[float, float, float]:
    """
    Predicts a response value for a single coded vector along with 95% Confidence Interval (CI)
    and 95% Prediction Interval (PI).
    """
    model = model_dict["model_object"]
    selected_features = model_dict["feature_names"]
    std_dev = model_dict["fit_statistics"]["Std_Dev"]
    df_res = len(model_dict["diagnostics"]["residuals"]) - len(model.params)

    # Build feature row
    row_dict = {}
    for i, code in enumerate(factor_codes):
        row_dict[code] = coded_vector[i]

    for i in range(len(factor_codes)):
        for j in range(i + 1, len(factor_codes)):
            key = f"{factor_codes[i]}*{factor_codes[j]}"
            row_dict[key] = coded_vector[i] * coded_vector[j]

    for i, code in enumerate(factor_codes):
        key = f"{code}^2"
        row_dict[key] = coded_vector[i] ** 2

    # Vector for regression
    x_vec = [1.0] + [row_dict[feat] for feat in selected_features]
    x_vec = np.array(x_vec, dtype=float)

    y_hat = float(model.params.values @ x_vec)

    # Leverage / standard error of prediction: se_mean = sqrt(s^2 * x^T (X^T X)^-1 x)
    try:
        cov_matrix = model.cov_params().values
        var_mean = float(x_vec.T @ cov_matrix @ x_vec)
        se_mean = np.sqrt(max(0.0, var_mean))
        se_pred = np.sqrt(max(0.0, var_mean + (std_dev ** 2)))
        t_crit = float(stats.t.ppf(0.975, df=df_res)) if df_res > 0 else 1.96

        ci_half = t_crit * se_mean
        pi_half = t_crit * se_pred
    except Exception:
        ci_half = 1.96 * std_dev
        pi_half = 2.5 * std_dev

    return y_hat, ci_half, pi_half
