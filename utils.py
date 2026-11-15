"""
Utility Module for Nanoemulgel Formulation Optimization System.
Provides Excel multi-sheet export, CSV import/export, comprehensive PDF
research report compilation using fpdf2, and a pre-configured synthetic
benchmark dataset for automated software verification.
"""

import os
import io
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from fpdf import FPDF
from config import REPORTS_DIR, EXPORTS_DIR, APP_TITLE, AUTHOR_INFO

# ==============================================================================
# 1. EXCEL MULTI-SHEET EXPORT & IMPORT
# ==============================================================================

def export_complete_project_to_excel(
    project: Dict[str, Any],
    df_design: Optional[pd.DataFrame] = None,
    df_data: Optional[pd.DataFrame] = None,
    models: Optional[Dict[str, Dict[str, Any]]] = None,
    optimization_res: Optional[Dict[str, Any]] = None,
    df_validation: Optional[pd.DataFrame] = None
) -> bytes:
    """
    Generates a beautifully structured, multi-sheet Excel (.xlsx) workbook
    containing all aspects of the pharmaceutical formulation project.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Project Metadata
        meta_rows = [
            ("Project Name", project.get("name", "Nanoemulgel QbD")),
            ("Active Pharmaceutical Ingredient (API)", project.get("drug_name", "Model Drug")),
            ("Dosage Form", project.get("dosage_form", "Nanoemulgel")),
            ("Route of Administration", project.get("route", "Topical / Transdermal")),
            ("Oil Phase", project.get("oil_phase", "Clove Oil")),
            ("Surfactant", project.get("surfactant", "Tween 80")),
            ("Co-Surfactant", project.get("cosurfactant", "Transcutol P")),
            ("Gelling Agent", project.get("gelling_agent", "Carbopol 940")),
            ("Other Excipients", project.get("other_excipients", "Triethanolamine, Preservative, Water")),
            ("Created / Updated", str(project.get("updated_at", "N/A")))
        ]
        df_meta = pd.DataFrame(meta_rows, columns=["Parameter", "Specification"])
        df_meta.to_excel(writer, sheet_name="Project Info", index=False)

        # Factors
        if "factors" in project:
            df_factors = pd.DataFrame(project["factors"])
            df_factors.to_excel(writer, sheet_name="Factors", index=False)

        # Responses
        if "responses" in project:
            df_resp = pd.DataFrame(project["responses"])
            df_resp.to_excel(writer, sheet_name="Responses Specs", index=False)

        # Sheet 2: DoE Design Matrix
        if df_design is not None:
            df_design.to_excel(writer, sheet_name="DoE Matrix", index=False)

        # Sheet 3: Experimental Data
        if df_data is not None:
            df_data.to_excel(writer, sheet_name="Experimental Data", index=False)

        # Sheet 4: Statistical ANOVA Summary
        if models:
            anova_rows = []
            stat_rows = []
            eq_rows = []
            for resp_name, m in models.items():
                eq_rows.append({"Response": resp_name, "Coded Equation": m.get("coded_equation", "")})
                fit = m.get("fit_statistics", {})
                stat_rows.append({
                    "Response": resp_name,
                    "R²": fit.get("R2"),
                    "Adj R²": fit.get("Adjusted_R2"),
                    "Pred R²": fit.get("Predicted_R2"),
                    "R² Diff": fit.get("R2_Diff"),
                    "Adeq Precision": fit.get("Adequate_Precision"),
                    "Model F-value": fit.get("Model_F"),
                    "Model p-value": fit.get("Model_p"),
                    "LoF p-value": fit.get("LoF_p")
                })
                for row in m.get("anova", []):
                    r_copy = row.copy()
                    r_copy["Response"] = resp_name
                    anova_rows.append(r_copy)

            pd.DataFrame(stat_rows).to_excel(writer, sheet_name="Model Statistics", index=False)
            pd.DataFrame(eq_rows).to_excel(writer, sheet_name="Regression Equations", index=False)
            if anova_rows:
                pd.DataFrame(anova_rows).to_excel(writer, sheet_name="ANOVA Breakdown", index=False)

        # Sheet 5: Optimization Results
        if optimization_res:
            opt_rows = [
                ("Composite Desirability (D)", optimization_res.get("composite_desirability")),
                ("Explanation", optimization_res.get("explanation", ""))
            ]
            for code, finfo in optimization_res.get("optimal_actual_factors", {}).items():
                opt_rows.append((f"Optimal {finfo['name']} ({finfo['unit']})", finfo["value"]))

            for resp_name, pinfo in optimization_res.get("predicted_responses", {}).items():
                opt_rows.append((f"Predicted {resp_name} ({pinfo.get('unit','')})", pinfo.get("predicted_mean")))
                opt_rows.append((f"  95% CI [{resp_name}]", f"[{pinfo.get('ci_95_lower')}, {pinfo.get('ci_95_upper')}]"))
                opt_rows.append((f"  95% PI [{resp_name}]", f"[{pinfo.get('pi_95_lower')}, {pinfo.get('pi_95_upper')}]"))

            pd.DataFrame(opt_rows, columns=["Optimization Attribute", "Value"]).to_excel(writer, sheet_name="Optimization", index=False)

        # Sheet 6: Validation Results
        if df_validation is not None and not df_validation.empty:
            df_validation.to_excel(writer, sheet_name="Validation", index=False)

    output.seek(0)
    return output.getvalue()

def parse_imported_excel_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """
    Parses an uploaded Excel or CSV file containing experimental laboratory observations.
    """
    if file_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        # If multiple sheets, read the one with 'Data' or the first sheet
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_to_read = xl.sheet_names[0]
        for name in xl.sheet_names:
            if "data" in name.lower() or "experiment" in name.lower() or "doe" in name.lower():
                sheet_to_read = name
                break
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_to_read)

    return df

# ==============================================================================
# 2. PDF REPORT GENERATOR (FPDF2)
# ==============================================================================

class PharmaQbDReport(FPDF):
    """Custom PDF class styled for academic pharmaceutical research reports."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(70, 70, 70)
        self.cell(0, 8, "PHARMACEUTICAL QUALITY BY DESIGN (QbD) FORMULATION REPORT", border="B", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Academic Formulation Research | Confidential", border="T", align="C")

def generate_pdf_report(
    project: Dict[str, Any],
    factors: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    df_design: Optional[pd.DataFrame] = None,
    df_data: Optional[pd.DataFrame] = None,
    models: Optional[Dict[str, Dict[str, Any]]] = None,
    optimization_res: Optional[Dict[str, Any]] = None,
    df_validation: Optional[pd.DataFrame] = None,
    validation_summary: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Compiles a comprehensive, publication-ready PDF research report
    for academic thesis documentation and project defense.
    """
    pdf = PharmaQbDReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title Banner
    pdf.set_fill_color(24, 76, 120)
    pdf.rect(10, 18, 190, 24, "F")
    pdf.set_xy(12, 20)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(186, 8, "NANOEMULGEL FORMULATION OPTIMIZATION REPORT", align="C", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(186, 7, "Quality by Design (QbD) Response Surface Methodology & Multi-Response Desirability", align="C", ln=True)

    pdf.ln(8)
    pdf.set_text_color(40, 40, 40)

    # 1. Project Specifications
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(230, 240, 250)
    pdf.cell(0, 7, "1. PROJECT IDENTIFICATION & FORMULATION COMPOSITION", fill=True, ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    info_pairs = [
        ("Project Title:", project.get("name", "Nanoemulgel Optimization")),
        ("Active Drug (API):", project.get("drug_name", "Model Therapeutic")),
        ("Dosage Form & Route:", f"{project.get('dosage_form', 'Nanoemulgel')} / {project.get('route', 'Topical')}"),
        ("Oil Phase (Lipid):", project.get("oil_phase", "Clove Oil")),
        ("Surfactant System:", project.get("surfactant", "Tween 80")),
        ("Co-Surfactant:", project.get("cosurfactant", "Transcutol P")),
        ("Gelling Matrix:", project.get("gelling_agent", "Carbopol 940")),
        ("Excipients / Buffer:", project.get("other_excipients", "Triethanolamine, Purified Water"))
    ]

    col_w = 95
    for i in range(0, len(info_pairs), 2):
        k1, v1 = info_pairs[i]
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(38, 5, k1)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(57, 5, v1[:30])

        if i + 1 < len(info_pairs):
            k2, v2 = info_pairs[i + 1]
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(38, 5, k2)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(57, 5, v2[:30], ln=True)
        else:
            pdf.ln(5)

    pdf.ln(4)

    # 2. Independent Variables (Factors) Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. INDEPENDENT FORMULATION VARIABLES (FACTORS)", fill=True, ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(16, 6, "Code", 1, 0, "C", True)
    pdf.cell(64, 6, "Factor Name", 1, 0, "L", True)
    pdf.cell(26, 6, "Unit", 1, 0, "C", True)
    pdf.cell(28, 6, "Low (-1)", 1, 0, "C", True)
    pdf.cell(28, 6, "Center (0)", 1, 0, "C", True)
    pdf.cell(28, 6, "High (+1)", 1, 1, "C", True)

    pdf.set_font("Helvetica", "", 8)
    for f in factors:
        pdf.cell(16, 5, f["code"], 1, 0, "C")
        pdf.cell(64, 5, f["name"], 1, 0, "L")
        pdf.cell(26, 5, f["unit"], 1, 0, "C")
        pdf.cell(28, 5, str(f["low"]), 1, 0, "C")
        pdf.cell(28, 5, str(f["center"]), 1, 0, "C")
        pdf.cell(28, 5, str(f["high"]), 1, 1, "C")

    pdf.ln(4)

    # 3. Critical Quality Attributes (Responses) Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "3. CRITICAL QUALITY ATTRIBUTES (CQAs) & TARGET CRITERIA", fill=True, ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(60, 6, "Response Variable", 1, 0, "L", True)
    pdf.cell(26, 6, "Unit", 1, 0, "C", True)
    pdf.cell(30, 6, "Optimization Goal", 1, 0, "C", True)
    pdf.cell(26, 6, "Target / Range", 1, 0, "C", True)
    pdf.cell(24, 6, "Low - High", 1, 0, "C", True)
    pdf.cell(24, 6, "Importance", 1, 1, "C", True)

    pdf.set_font("Helvetica", "", 8)
    for r in responses:
        if not r.get("enabled", True):
            continue
        tgt_str = str(r.get("target")) if r.get("target") is not None else "-"
        pdf.cell(60, 5, r["name"], 1, 0, "L")
        pdf.cell(26, 5, r["unit"], 1, 0, "C")
        pdf.cell(30, 5, r["goal"].capitalize(), 1, 0, "C")
        pdf.cell(26, 5, tgt_str, 1, 0, "C")
        pdf.cell(24, 5, f"{r['low']} - {r['high']}", 1, 0, "C")
        pdf.cell(24, 5, str(r.get("importance", 3)), 1, 1, "C")

    pdf.ln(4)

    # 4. Statistical Modeling & ANOVA Summary
    if models:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "4. STATISTICAL MODELING, ANOVA & FIT STATISTICS", fill=True, ln=True)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(50, 6, "Response", 1, 0, "L", True)
        pdf.cell(20, 6, "R²", 1, 0, "C", True)
        pdf.cell(20, 6, "Adj R²", 1, 0, "C", True)
        pdf.cell(20, 6, "Pred R²", 1, 0, "C", True)
        pdf.cell(26, 6, "Adeq Precision", 1, 0, "C", True)
        pdf.cell(26, 6, "Model F (p-val)", 1, 0, "C", True)
        pdf.cell(28, 6, "Lack of Fit p", 1, 1, "C", True)

        pdf.set_font("Helvetica", "", 8)
        for resp_name, m in models.items():
            st = m.get("fit_statistics", {})
            f_str = f"{st.get('Model_F', 0):.1f} ({st.get('Model_p', 0):.4f})"
            lof_str = f"{st.get('LoF_p', 'N/A')}"
            if isinstance(st.get('LoF_p'), float):
                lof_str = f"{st.get('LoF_p'):.4f}"

            pdf.cell(50, 5, resp_name[:26], 1, 0, "L")
            pdf.cell(20, 5, f"{st.get('R2', 0):.3f}", 1, 0, "C")
            pdf.cell(20, 5, f"{st.get('Adjusted_R2', 0):.3f}", 1, 0, "C")
            pdf.cell(20, 5, f"{st.get('Predicted_R2', 0):.3f}", 1, 0, "C")
            pdf.cell(26, 5, f"{st.get('Adequate_Precision', 0):.2f}", 1, 0, "C")
            pdf.cell(26, 5, f_str, 1, 0, "C")
            pdf.cell(28, 5, lof_str, 1, 1, "C")

        pdf.ln(4)

        # Regression Equations
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "Second-Order Polynomial Regression Equations (Coded Levels):", ln=True)
        pdf.set_font("Courier", "", 7)
        for resp_name, m in models.items():
            pdf.multi_cell(0, 4, m.get("coded_equation", ""))
        pdf.ln(3)

    # 5. Multi-Response Formulation Optimization
    if optimization_res:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "5. MULTI-RESPONSE OPTIMIZATION (DERRINGER-SUICH DESIRABILITY)", fill=True, ln=True)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, f"Overall Composite Desirability D = {optimization_res.get('composite_desirability', 0):.4f}", ln=True)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 4, optimization_res.get("explanation", ""))
        pdf.ln(2)

        # Optimal Formulation Table
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(60, 6, "Optimized Formulation Variable", 1, 0, "L", True)
        pdf.cell(35, 6, "Concentration Level", 1, 0, "C", True)
        pdf.cell(40, 6, "Unit", 1, 0, "C", True)
        pdf.cell(55, 6, "QbD Operational Role", 1, 1, "L", True)

        pdf.set_font("Helvetica", "", 8)
        for code, finfo in optimization_res.get("optimal_actual_factors", {}).items():
            pdf.cell(60, 5, f"Factor {code}: {finfo['name']}", 1, 0, "L")
            pdf.cell(35, 5, f"{finfo['value']:.3f}", 1, 0, "C")
            pdf.cell(40, 5, finfo["unit"], 1, 0, "C")
            pdf.cell(55, 5, "Optimized Independent Variable", 1, 1, "L")

        pdf.ln(3)

        # Predicted Characteristics Table
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(60, 6, "Predicted Response (CQA)", 1, 0, "L", True)
        pdf.cell(35, 6, "Predicted Mean", 1, 0, "C", True)
        pdf.cell(45, 6, "95% Prediction Interval (PI)", 1, 0, "C", True)
        pdf.cell(50, 6, "Individual Desirability (d)", 1, 1, "C", True)

        pdf.set_font("Helvetica", "", 8)
        preds = optimization_res.get("predicted_responses", {})
        ind_d = optimization_res.get("individual_desirabilities", {})
        for resp_name, pinfo in preds.items():
            d_val = ind_d.get(resp_name, 1.0)
            pi_str = f"[{pinfo.get('pi_95_lower')}, {pinfo.get('pi_95_upper')}]"
            pdf.cell(60, 5, f"{resp_name} ({pinfo.get('unit','')})", 1, 0, "L")
            pdf.cell(35, 5, f"{pinfo.get('predicted_mean'):.2f}", 1, 0, "C")
            pdf.cell(45, 5, pi_str, 1, 0, "C")
            pdf.cell(50, 5, f"{d_val:.4f}", 1, 1, "C")

        pdf.ln(4)

    # 6. Experimental Validation & Confirmation
    if df_validation is not None and not df_validation.empty:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "6. EXPERIMENTAL VALIDATION & LABORATORY CONFIRMATION", fill=True, ln=True)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(46, 6, "Response", 1, 0, "L", True)
        pdf.cell(26, 6, "Predicted", 1, 0, "C", True)
        pdf.cell(28, 6, "Observed Lab", 1, 0, "C", True)
        pdf.cell(24, 6, "Error (%)", 1, 0, "C", True)
        pdf.cell(24, 6, "In 95% PI", 1, 0, "C", True)
        pdf.cell(42, 6, "Confirmation Status", 1, 1, "C", True)

        pdf.set_font("Helvetica", "", 7)
        for _, row in df_validation.iterrows():
            pdf.cell(46, 5, str(row["Response Variable"])[:24], 1, 0, "L")
            pdf.cell(26, 5, str(row["Predicted Value"]), 1, 0, "C")
            pdf.cell(28, 5, str(row["Observed Lab Value"]), 1, 0, "C")
            pdf.cell(24, 5, f"{row['Prediction Error (%)']}%", 1, 0, "C")
            pdf.cell(24, 5, str(row["Within 95% PI"]), 1, 0, "C")
            pdf.cell(42, 5, str(row["Verification Status"])[:25], 1, 1, "C")

        pdf.ln(3)
        if validation_summary:
            pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(0, 4, f"Scientific Assessment: {validation_summary.get('scientific_assessment', '')}")
            pdf.ln(3)

    # 7. Academic Assessment & Sign-off Block
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(63, 5, "Principal Student Investigator:", 0, 0, "L")
    pdf.cell(63, 5, "Research Project Guide:", 0, 0, "L")
    pdf.cell(64, 5, "Head of Department (Pharmaceutics):", 0, 1, "L")

    pdf.ln(12)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(63, 4, "____________________________", 0, 0, "L")
    pdf.cell(63, 4, "____________________________", 0, 0, "L")
    pdf.cell(64, 4, "____________________________", 0, 1, "L")
    pdf.cell(63, 4, "Signature & Date", 0, 0, "L")
    pdf.cell(63, 4, "Signature & Date", 0, 0, "L")
    pdf.cell(64, 4, "Signature & Date", 0, 1, "L")

    return bytes(pdf.output())

# ==============================================================================
# 3. SYNTHETIC BENCHMARK DATASET GENERATOR
# ==============================================================================

def generate_synthetic_benchmark_dataset(
    factors: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic, published-literature benchmark test dataset
    for a 17-run Box-Behnken nanoemulgel formulation design.

    CLEARLY LABELED AS SYNTHETIC DATA FOR SOFTWARE VERIFICATION ONLY.
    Reflects typical nanoemulsion physicochemical response behavior:
    - Particle Size: decreases with surfactant (B) and co-surfactant (C), increases with oil (A).
    - PDI: decreases at optimal Smix ratio.
    - Zeta Potential: negative magnitude (-20 to -35 mV).
    - Entrapment Efficiency: increases with oil up to a plateau.
    - Viscosity & Spreadability: realistic topical gel properties.
    """
    from doe import generate_box_behnken, code_to_actual

    num_factors = len(factors)
    coded = generate_box_behnken(num_factors=num_factors, num_center=5)
    n_runs = len(coded)

    rng = np.random.RandomState(random_seed)
    perm = rng.permutation(n_runs)
    coded = coded[perm]

    # Convert to actual
    actual_rows = []
    for row in coded:
        act_row = []
        for i, f in enumerate(factors):
            val = code_to_actual(row[i], f["low"], f["center"], f["high"])
            act_row.append(round(val, 3))
        actual_rows.append(act_row)
    actual_rows = np.array(actual_rows)

    df = pd.DataFrame()
    df["Std_Order"] = np.arange(1, n_runs + 1)[perm]
    df["Run_Order"] = np.arange(1, n_runs + 1)

    for i, f in enumerate(factors):
        df[f"Factor_{f['code']}_Coded"] = np.round(coded[:, i], 3)
    for i, f in enumerate(factors):
        df[f"{f['code']}_{f['name']} ({f['unit']})"] = actual_rows[:, i]

    A = coded[:, 0]
    B = coded[:, 1]
    C = coded[:, 2] if num_factors > 2 else np.zeros(n_runs)
    D = coded[:, 3] if num_factors > 3 else np.zeros(n_runs)

    # Quadratic response models with realistic laboratory error
    for r in responses:
        name = r["name"]
        unit = r["unit"]
        col_name = f"{name} ({unit})"

        if "Particle Size" in name:
            # Typical nanoemulsion: Y = 115 + 28*A - 34*B - 14*C + 8*A*B - 6*B*C + 16*A^2 + 12*B^2 + noise
            y = 112.5 + 26.4 * A - 31.8 * B - 12.5 * C + 7.6 * A * B - 5.2 * B * C + 14.8 * (A ** 2) + 11.2 * (B ** 2) + rng.normal(0, 1.8, n_runs)
            y = np.clip(y, 40.0, 300.0)

        elif "PDI" in name:
            y = 0.235 + 0.042 * A - 0.065 * B - 0.025 * C - 0.018 * B * C + 0.038 * (B ** 2) + rng.normal(0, 0.008, n_runs)
            y = np.clip(y, 0.10, 0.55)

        elif "Zeta Potential" in name:
            # Negative surface charge
            y = -28.5 - 3.2 * A + 4.5 * B + 1.8 * C - 2.1 * A * B + 2.8 * (A ** 2) + rng.normal(0, 0.6, n_runs)
            y = np.clip(y, -45.0, -15.0)

        elif "Entrapment" in name:
            y = 83.5 + 6.2 * A + 4.8 * B + 2.1 * C - 3.4 * (A ** 2) - 2.8 * (B ** 2) + rng.normal(0, 0.7, n_runs)
            y = np.clip(y, 60.0, 99.5)

        elif "Drug Content" in name:
            # Assay around 98 - 101%
            y = 99.4 + 0.5 * A - 0.4 * B + rng.normal(0, 0.35, n_runs)
            y = np.clip(y, 96.0, 103.0)

        elif "Viscosity" in name:
            y = 17500 + 1200 * A - 1800 * B - 800 * C + 950 * (A ** 2) + rng.normal(0, 250, n_runs)
            y = np.clip(y, 8000, 35000)

        elif "Spreadability" in name:
            y = 5.6 - 0.35 * A + 0.45 * B + 0.2 * C - 0.15 * (A ** 2) + rng.normal(0, 0.08, n_runs)
            y = np.clip(y, 3.5, 7.5)

        elif "Release" in name:
            y = 82.0 - 5.5 * A + 7.8 * B + 3.2 * C - 3.2 * A * B + rng.normal(0, 0.9, n_runs)
            y = np.clip(y, 50.0, 98.0)

        else:
            y = 50.0 + 10.0 * A - 8.0 * B + 4.0 * C + rng.normal(0, 1.0, n_runs)

        df[col_name] = np.round(y, 3)

    return df
