"""
PharmaQbD: Complete Python-Based Nanoemulgel Formulation Optimization System.
Main Streamlit Research Application.
"""

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st

# Configure Matplotlib backend and suppress warnings
import config
from config import (
    APP_TITLE, APP_SUBTITLE, APP_VERSION, AUTHOR_INFO,
    STANDARD_EXCIPIENTS, DEFAULT_FACTORS, DEFAULT_RESPONSES,
    STAT_THRESHOLDS, EDUCATIONAL_GLOSSARY
)
from database import (
    init_db, save_project, load_project, list_projects, delete_project,
    save_design, load_design, save_experimental_data, load_experimental_data,
    save_model, load_models, save_optimization, load_optimization,
    save_validation, load_validation
)
from doe import (
    create_design_matrix, get_design_guidance, code_to_actual, actual_to_code
)
from analysis import fit_rsm_model, predict_single_response
from visualization import (
    create_contour_plot, create_3d_surface_plot, create_actual_vs_predicted_plot,
    create_residual_diagnostics_suite, create_perturbation_plot, generate_static_contour_png
)
from optimization import optimize_formulation, compute_composite_desirability
from validation import evaluate_validation_results, create_validation_comparison_chart
from utils import (
    export_complete_project_to_excel, parse_imported_excel_data,
    generate_pdf_report, generate_synthetic_benchmark_dataset
)

# Set page configuration
st.set_page_config(
    page_title="PharmaQbD - Nanoemulgel Optimizer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a446c;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1a446c;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #e8f4fd;
        border-radius: 6px;
        padding: 12px;
        color: #0c5460;
        margin-bottom: 12px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-radius: 6px;
        padding: 12px;
        color: #856404;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

from pathlib import Path

# Initialize Session State
if "current_project_id" not in st.session_state:
    st.session_state["current_project_id"] = None

# Sidebar Navigation
logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=72)
else:
    st.sidebar.markdown("# 🧪")
st.sidebar.markdown(f"### **PharmaQbD v{APP_VERSION}**")
st.sidebar.markdown("*Nanoemulgel Optimization Platform*")

menu_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. 🏠 Dashboard",
        "2. 🧪 Project Setup",
        "3. 📐 Experimental Design (DoE)",
        "4. 📝 Experimental Data Entry",
        "5. 📊 Statistical Analysis",
        "6. 📋 ANOVA & Regression",
        "7. 📈 Response Surface Plots",
        "8. 🎯 Formulation Optimization",
        "9. 🔬 Experimental Validation",
        "10. 📄 Reports & Downloads"
    ]
)

# Sidebar Educational Drawer
st.sidebar.markdown("---")
with st.sidebar.expander("📚 B.Pharm Scientific Glossary", expanded=False):
    selected_term = st.selectbox("Select Concept:", list(EDUCATIONAL_GLOSSARY.keys()))
    st.info(EDUCATIONAL_GLOSSARY[selected_term])

# Active Project Loader
projects_list = list_projects()
project_options = {p["id"]: f"{p['name']} ({p['drug_name']})" for p in projects_list}

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Project:**")
if projects_list:
    selected_pid = st.sidebar.selectbox(
        "Switch Project",
        options=list(project_options.keys()),
        format_func=lambda x: project_options[x],
        index=0 if st.session_state["current_project_id"] not in project_options else list(project_options.keys()).index(st.session_state["current_project_id"])
    )
    st.session_state["current_project_id"] = selected_pid
else:
    st.sidebar.warning("No projects saved yet.")
    st.session_state["current_project_id"] = None

# Load current project
current_project = None
if st.session_state["current_project_id"]:
    current_project = load_project(st.session_state["current_project_id"])


# ==============================================================================
# SECTION 1: DASHBOARD
# ==============================================================================
if "1. 🏠 Dashboard" in menu_choice:
    st.markdown('<div class="main-title">🧪 Pharmaceutical QbD Nanoemulgel Optimization Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Systematic Quality by Design (ICH Q8 R2) Framework for Formulation Engineering</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Saved Projects", len(projects_list))
    with col2:
        num_factors = len(current_project["factors"]) if current_project else 0
        st.metric("Active Factors", num_factors)
    with col3:
        num_resp = len([r for r in current_project["responses"] if r.get("enabled", True)]) if current_project else 0
        st.metric("Active Responses (CQAs)", num_resp)
    with col4:
        st.metric("QbD Methodology", "RSM (BBD / CCD)")

    st.markdown("---")

    col_a, col_b = st.columns([1.2, 0.8])
    with col_a:
        st.markdown("### 📋 Systematic QbD Nanoemulgel Workflow")
        st.markdown("""
        1. **Define Target Product Quality Profile (QTPP):** Specify topical nanoemulgel target attributes (droplet size < 100 nm, high entrapment, appropriate gel spreadability).
        2. **Identify Critical Material Attributes (CMAs):** Screen Oil phase, Surfactant, Co-surfactant (Smix), and Gelling polymer concentration ranges.
        3. **Formulate Experimental Design (DoE):** Select 3-factor Box-Behnken Design (17 runs) or Central Composite Design to map quadratic response surfaces without risk of phase cracking.
        4. **Conduct Laboratory Batches & Enter Data:** Prepare randomized batches and measure droplet size, PDI, zeta potential, entrapment, and drug release.
        5. **Statistical Modeling & ANOVA Decomposition:** Partition variance into linear, interaction, and curvature components with Lack of Fit validation.
        6. **Multi-Response Desirability Optimization:** Apply Derringer-Suich numerical optimization to find the balanced global compromise recipe.
        7. **Laboratory Confirmation & Validation:** Fabricate the optimal formulation and compare experimental findings against 95% Prediction Intervals.
        """)

    with col_b:
        st.markdown("### 📌 Active Project Summary")
        if current_project:
            st.markdown(f"""
            - **Project Name:** `{current_project['name']}`
            - **Drug (API):** `{current_project['drug_name']}`
            - **Dosage Form:** `{current_project['dosage_form']}`
            - **Oil Phase:** `{current_project['oil_phase']}`
            - **Surfactant:** `{current_project['surfactant']}`
            - **Co-Surfactant:** `{current_project['cosurfactant']}`
            - **Gelling Agent:** `{current_project['gelling_agent']}`
            - **Route:** `{current_project['route']}`
            """)
        else:
            st.info("No active project loaded. Go to **2. 🧪 Project Setup** to create or initialize a project.")


# ==============================================================================
# SECTION 2: PROJECT SETUP
# ==============================================================================
elif "2. 🧪 Project Setup" in menu_choice:
    st.markdown('<div class="main-title">🧪 Project Setup & Formulation Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Define API, Excipient System, Independent Variables (Factors), and Critical Quality Attributes</div>', unsafe_allow_html=True)

    mode = st.radio("Project Mode", ["Create New Project", "Edit Active Project", "Quick-Load Default BBD Project"], horizontal=True)

    if mode == "Quick-Load Default BBD Project":
        st.info("Quickly populates the standard 3-Factor Box-Behnken Nanoemulgel setup matching your original sample research code.")
        if st.button("Initialize Default BBD Research Project"):
            default_proj = {
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
            new_id = save_project(default_proj, DEFAULT_FACTORS, DEFAULT_RESPONSES)
            st.session_state["current_project_id"] = new_id
            st.success("Default Nanoemulgel Project created and set as active!")
            st.rerun()

    else:
        with st.form("project_form"):
            st.subheader("1. General Pharmaceutical Information")
            c1, c2, c3 = st.columns(3)
            with c1:
                p_name = st.text_input("Project Title", value=current_project["name"] if current_project and mode == "Edit Active Project" else "Nanoemulgel QbD Optimization")
                drug_name = st.text_input("Active Drug (API)", value=current_project["drug_name"] if current_project and mode == "Edit Active Project" else "Model Drug")
            with c2:
                dosage_form = st.text_input("Dosage Form", value=current_project["dosage_form"] if current_project and mode == "Edit Active Project" else "Nanoemulgel")
                route = st.text_input("Route of Administration", value=current_project["route"] if current_project and mode == "Edit Active Project" else "Topical")
            with c3:
                gelling_agent = st.selectbox(
                    "Gelling Polymer",
                    STANDARD_EXCIPIENTS["gelling_agents"],
                    index=0 if not current_project or mode != "Edit Active Project" else (
                        STANDARD_EXCIPIENTS["gelling_agents"].index(current_project["gelling_agent"])
                        if current_project["gelling_agent"] in STANDARD_EXCIPIENTS["gelling_agents"] else 0
                    )
                )
                other_excipients = st.text_input("Other Excipients (Neutralizer / Water)", value=current_project.get("other_excipients", "") if current_project and mode == "Edit Active Project" else "Triethanolamine, Water")

            st.subheader("2. Nanoemulsion Composition (Lipid & Surfactant System)")
            e1, e2, e3 = st.columns(3)
            with e1:
                oil_phase = st.selectbox("Oil Phase (Lipid Core)", STANDARD_EXCIPIENTS["oils"], index=0)
            with e2:
                surfactant = st.selectbox("Surfactant", STANDARD_EXCIPIENTS["surfactants"], index=0)
            with e3:
                cosurfactant = st.selectbox("Co-Surfactant", STANDARD_EXCIPIENTS["cosurfactants"], index=0)

            st.subheader("3. Independent Formulation Variables (Factors)")
            st.caption("Define concentration levels. Preserves original sample ranges: Oil (5-15%), Surfactant (2-6%), Co-surfactant (1-3%).")

            num_factors_input = st.number_input("Number of Independent Factors", min_value=2, max_value=4, value=len(current_project["factors"]) if current_project and mode == "Edit Active Project" else 3)

            factor_data = []
            f_cols = st.columns(int(num_factors_input))
            default_codes = ["A", "B", "C", "D"]
            default_names = ["Oil concentration", "Surfactant concentration", "Co-surfactant concentration", "Gelling agent concentration"]
            default_lows = [5.0, 2.0, 1.0, 0.5]
            default_centers = [10.0, 4.0, 2.0, 1.0]
            default_highs = [15.0, 6.0, 3.0, 1.5]

            for idx, col in enumerate(f_cols):
                with col:
                    st.markdown(f"**Factor {default_codes[idx]}**")
                    f_name = st.text_input(f"Name ({default_codes[idx]})", value=current_project["factors"][idx]["name"] if current_project and mode == "Edit Active Project" and idx < len(current_project["factors"]) else default_names[idx], key=f"fn_{idx}")
                    f_unit = st.text_input(f"Unit ({default_codes[idx]})", value="% w/w", key=f"fu_{idx}")
                    c_low = st.number_input(f"Low (-1)", value=float(current_project["factors"][idx]["low"]) if current_project and mode == "Edit Active Project" and idx < len(current_project["factors"]) else default_lows[idx], key=f"fl_{idx}")
                    c_center = st.number_input(f"Center (0)", value=float(current_project["factors"][idx]["center"]) if current_project and mode == "Edit Active Project" and idx < len(current_project["factors"]) else default_centers[idx], key=f"fc_{idx}")
                    c_high = st.number_input(f"High (+1)", value=float(current_project["factors"][idx]["high"]) if current_project and mode == "Edit Active Project" and idx < len(current_project["factors"]) else default_highs[idx], key=f"fh_{idx}")

                    factor_data.append({
                        "code": default_codes[idx],
                        "name": f_name,
                        "unit": f_unit,
                        "low": c_low,
                        "center": c_center,
                        "high": c_high
                    })

            st.subheader("4. Dependent Responses (Critical Quality Attributes - CQAs)")
            st.caption("Select and configure response limits and targets.")
            resp_data = []
            for r_idx, r in enumerate(DEFAULT_RESPONSES):
                rc1, rc2, rc3, rc4, rc5 = st.columns([0.4, 1.2, 0.8, 0.8, 0.8])
                with rc1:
                    is_en = st.checkbox("Enable", value=True, key=f"re_en_{r_idx}")
                with rc2:
                    r_name = st.text_input("Response Name", value=r["name"], key=f"rn_{r_idx}")
                with rc3:
                    r_goal = st.selectbox("Goal", ["minimize", "maximize", "target", "range", "none"], index=["minimize", "maximize", "target", "range", "none"].index(r["goal"]), key=f"rg_{r_idx}")
                with rc4:
                    r_low = st.number_input("Lower Bound", value=float(r["low"]), key=f"rl_{r_idx}")
                with rc5:
                    r_high = st.number_input("Upper Bound", value=float(r["high"]), key=f"rh_{r_idx}")

                resp_data.append({
                    "name": r_name,
                    "unit": r["unit"],
                    "goal": r_goal,
                    "target": r["target"],
                    "low": r_low,
                    "high": r_high,
                    "importance": r["importance"],
                    "enabled": is_en
                })

            submitted = st.form_submit_button("Save Project Configuration")
            if submitted:
                proj_dict = {
                    "id": current_project["id"] if current_project and mode == "Edit Active Project" else None,
                    "name": p_name,
                    "drug_name": drug_name,
                    "dosage_form": dosage_form,
                    "route": route,
                    "oil_phase": oil_phase,
                    "surfactant": surfactant,
                    "cosurfactant": cosurfactant,
                    "gelling_agent": gelling_agent,
                    "other_excipients": other_excipients
                }
                new_pid = save_project(proj_dict, factor_data, resp_data)
                st.session_state["current_project_id"] = new_pid
                st.success("Project saved successfully!")
                st.rerun()


# ==============================================================================
# SECTION 3: EXPERIMENTAL DESIGN (DoE)
# ==============================================================================
elif "3. 📐 Experimental Design (DoE)" in menu_choice:
    st.markdown('<div class="main-title">📐 Design of Experiments (DoE) Matrix Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Generate, Randomize, and Export Box-Behnken or Central Composite Experimental Designs</div>', unsafe_allow_html=True)

    if not current_project:
        st.warning("Please create or select an active project in **2. 🧪 Project Setup** first.")
        st.stop()

    factors = current_project["factors"]
    st.markdown(f"**Current Factors ({len(factors)}):** " + " | ".join([f"**{f['code']}**: {f['name']} ({f['low']} - {f['high']} {f['unit']})" for f in factors]))

    st.markdown("---")
    st.subheader("1. Design Selection Guidance")

    c_g1, c_g2 = st.columns([1, 1])
    with c_g1:
        is_mixture = st.checkbox("Ingredients are constrained to a fixed proportion (e.g. Sum = 100%)", value=False)
        can_extremes = st.checkbox("Can safely fabricate extreme simultaneous factor combinations without emulsion breaking", value=False)

    guidance = get_design_guidance(len(factors), is_mixture_constrained=is_mixture, can_test_extremes=can_extremes)
    with c_g2:
        st.markdown(f"**Recommended Design:** `{guidance['recommended']}`")
        if guidance["warning"]:
            st.error(guidance["warning"])
        else:
            st.success(guidance["rationale"])

    st.markdown("---")
    st.subheader("2. Configure & Generate Design")

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        design_type = st.selectbox(
            "Design Methodology",
            ["Box-Behnken Design (BBD)", "Central Composite Design (CCD)", "Simplex Lattice Mixture Design"],
            index=0
        )
    with col_d2:
        num_center = st.number_input("Center Point Replicates", min_value=3, max_value=8, value=5, help="5 center points is standard for 3-factor BBD (17 runs total).")
    with col_d3:
        rand_seed = st.number_input("Reproducible Random Seed", min_value=1, max_value=99999, value=42, help="Seed 42 preserves the randomization from your original script.")

    if st.button("Generate Experimental Design Matrix"):
        try:
            df_design, coded_mat, actual_mat = create_design_matrix(
                design_type=design_type,
                factors=factors,
                num_center=num_center,
                random_seed=rand_seed
            )
            save_design(current_project["id"], design_type, len(factors), num_center, rand_seed, df_design)
            st.success(f"Generated {len(df_design)} experimental runs successfully!")
        except Exception as e:
            st.error(f"Error generating design: {str(e)}")

    # Display Saved Design
    saved_design = load_design(current_project["id"])
    if saved_design:
        df_display = saved_design["dataframe"]
        st.markdown(f"### 📋 Experimental Matrix ({saved_design['design_type']} - {len(df_display)} Runs)")

        st.dataframe(df_display, use_container_width=True)

        # Download Buttons
        c_down1, c_down2 = st.columns(2)
        with c_down1:
            csv_data = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Design (CSV)",
                data=csv_data,
                file_name=f"DoE_Design_{current_project['name'].replace(' ', '_')}.csv",
                mime="text/csv"
            )
        with c_down2:
            excel_bytes = export_complete_project_to_excel(current_project, df_design=df_display)
            st.download_button(
                "📥 Download Design in Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"DoE_Design_{current_project['name'].replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ==============================================================================
# SECTION 4: EXPERIMENTAL DATA ENTRY
# ==============================================================================
elif "4. 📝 Experimental Data Entry" in menu_choice:
    st.markdown('<div class="main-title">📝 Experimental Observations Entry & Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter Laboratory Observations for Each Batch or Import via Excel/CSV</div>', unsafe_allow_html=True)

    if not current_project:
        st.warning("Please create or select an active project first.")
        st.stop()

    saved_design = load_design(current_project["id"])
    if not saved_design:
        st.warning("No experimental design found! Please generate a design in **3. 📐 Experimental Design** first.")
        st.stop()

    df_design = saved_design["dataframe"]
    active_responses = [r for r in current_project["responses"] if r.get("enabled", True)]

    # Load existing experimental data if present
    df_existing_data = load_experimental_data(saved_design["id"])

    # Option to populate with Synthetic Benchmark Data
    st.markdown("### 🧪 Laboratory Data Source")
    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        st.markdown("**Option A: Synthetic Benchmark Dataset**")
        st.caption("Load a pre-configured, realistic 17-run Box-Behnken dataset to verify all software regression and optimization algorithms.")
        if st.button("Load Synthetic Benchmark Data (For Testing Only)"):
            df_synth = generate_synthetic_benchmark_dataset(current_project["factors"], active_responses)
            save_experimental_data(saved_design["id"], df_synth)
            st.success("Loaded synthetic benchmark dataset! (Clearly labeled as synthetic software test data).")
            st.rerun()

    with c_btn2:
        st.markdown("**Option B: Upload Laboratory Spreadsheet**")
        uploaded_file = st.file_uploader("Upload Completed Excel / CSV", type=["xlsx", "xls", "csv"])
        if uploaded_file is not None:
            try:
                df_imported = parse_imported_excel_data(uploaded_file.getvalue(), uploaded_file.name)
                # Verify rows match
                if len(df_imported) == len(df_design):
                    save_experimental_data(saved_design["id"], df_imported)
                    st.success("Imported experimental observations successfully!")
                    st.rerun()
                else:
                    st.error(f"Row count mismatch! Design requires {len(df_design)} runs, but uploaded file contains {len(df_imported)} rows.")
            except Exception as e:
                st.error(f"Error parsing file: {str(e)}")

    st.markdown("---")
    st.markdown("### 📊 Interactive Data Editor")
    st.caption("You can edit values directly in the table below and click Save.")

    # Prepare DataFrame for editing
    if df_existing_data is not None:
        df_editor = df_existing_data.copy()
    else:
        df_editor = df_design.copy()
        for r in active_responses:
            col_name = f"{r['name']} ({r['unit']})"
            if col_name not in df_editor.columns:
                df_editor[col_name] = np.nan

    edited_df = st.data_editor(df_editor, num_rows="fixed", use_container_width=True)

    if st.button("💾 Save Experimental Observations"):
        save_experimental_data(saved_design["id"], edited_df)
        st.success("Experimental observations saved successfully to SQLite database!")
        st.rerun()

    # Data Quality Check
    st.markdown("---")
    st.markdown("### 🔍 Data Completeness Inspection")
    missing_counts = {}
    for r in active_responses:
        col_name = f"{r['name']} ({r['unit']})"
        if col_name in edited_df.columns:
            missing = edited_df[col_name].isna().sum()
            missing_counts[r['name']] = missing
        else:
            missing_counts[r['name']] = len(edited_df)

    has_missing = any(c > 0 for c in missing_counts.values())
    if has_missing:
        st.warning("⚠️ Incomplete Observations Detected:")
        for r_name, cnt in missing_counts.items():
            if cnt > 0:
                st.write(f"- **{r_name}:** {cnt} missing values out of {len(edited_df)} runs.")
        st.info("Ensure all runs have measured response values before fitting statistical regression models.")
    else:
        st.success("✅ All experimental runs have complete response observations! Ready for Statistical Modeling.")


# ==============================================================================
# SECTION 5: STATISTICAL ANALYSIS
# ==============================================================================
elif "5. 📊 Statistical Analysis" in menu_choice:
    st.markdown('<div class="main-title">📊 Statistical Modeling & Response Surface Equations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Fit Second-Order Polynomial Regression Models & Evaluate Term Significance</div>', unsafe_allow_html=True)

    if not current_project:
        st.warning("Please select a project first.")
        st.stop()

    saved_design = load_design(current_project["id"])
    if not saved_design:
        st.warning("No design found.")
        st.stop()

    df_data = load_experimental_data(saved_design["id"])
    if df_data is None:
        st.warning("No experimental data found. Please enter observations in **4. 📝 Experimental Data Entry** first.")
        st.stop()

    factors = current_project["factors"]
    factor_codes = [f["code"] for f in factors]
    active_responses = [r for r in current_project["responses"] if r.get("enabled", True)]

    # Extract coded matrix from columns
    coded_cols = [f"Factor_{c}_Coded" for c in factor_codes]
    if not all(col in df_data.columns for col in coded_cols):
        st.error("Coded factor columns not found in data table.")
        st.stop()

    X_coded = df_data[coded_cols].values

    st.subheader("1. Fit Response Surface Models")
    model_type_selected = st.selectbox("Polynomial Model Hierarchy", ["Quadratic", "2FI", "Linear"], index=0)

    if st.button("Fit RSM Models for All Active Responses"):
        fitted_count = 0
        for r in active_responses:
            col_name = f"{r['name']} ({r['unit']})"
            if col_name in df_data.columns:
                y = df_data[col_name].values
                if not np.all(np.isnan(y)):
                    try:
                        res = fit_rsm_model(
                            X_coded=X_coded,
                            y=y,
                            factors=factors,
                            response_name=r["name"],
                            response_unit=r["unit"],
                            model_type=model_type_selected
                        )
                        save_model(
                            current_project["id"],
                            r["name"],
                            model_type_selected,
                            res["coded_equation"],
                            res["coefficients"],
                            res["anova"],
                            res["fit_statistics"],
                            res["diagnostics"]
                        )
                        fitted_count += 1
                    except Exception as e:
                        st.error(f"Error modeling {r['name']}: {str(e)}")
        st.success(f"Successfully fitted and saved RSM models for {fitted_count} responses!")
        st.rerun()

    # Display Fitted Models
    fitted_models = load_models(current_project["id"])
    if fitted_models:
        st.markdown("---")
        st.subheader("2. Regression Coefficients & Significance")
        selected_resp = st.selectbox("Select Response to Inspect:", list(fitted_models.keys()))
        m = fitted_models[selected_resp]

        st.markdown(f"**Regression Equation (Coded Levels):**")
        st.code(m["equation"], language="text")

        # Coefficients DataFrame
        coef_rows = []
        for term, cdata in m["coefficients"].items():
            coef_rows.append({
                "Model Factor / Term": term,
                "Coefficient Estimate": round(cdata["coefficient"], 4),
                "Standard Error": round(cdata["std_error"], 4),
                "t-Statistic": round(cdata["t_stat"], 3),
                "p-Value": round(cdata["p_value"], 5),
                "95% CI Lower": round(cdata["ci_lower"], 4),
                "95% CI Upper": round(cdata["ci_upper"], 4),
                "Statistical Significance": "Significant (p < 0.05)" if cdata["significant"] else "Not Significant"
            })
        df_coef = pd.DataFrame(coef_rows)
        st.dataframe(df_coef, use_container_width=True)

        # Warnings / Diagnostics
        diag_warns = m.get("diagnostics", {}).get("warnings", [])
        if diag_warns:
            st.markdown("#### ⚠️ Statistical Diagnostics & Academic Alerts:")
            for w in diag_warns:
                st.warning(w)


# ==============================================================================
# SECTION 6: ANOVA & REGRESSION
# ==============================================================================
elif "6. 📋 ANOVA & Regression" in menu_choice:
    st.markdown('<div class="main-title">📋 Analysis of Variance (ANOVA) & Fit Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluate Fisher F-tests, Lack of Fit, R-Squared Metrics, and Adequate Precision</div>', unsafe_allow_html=True)

    if not current_project:
        st.warning("Please select a project first.")
        st.stop()

    fitted_models = load_models(current_project["id"])
    if not fitted_models:
        st.warning("No fitted models found. Please fit models in **5. 📊 Statistical Analysis** first.")
        st.stop()

    selected_resp = st.selectbox("Select Response Variable:", list(fitted_models.keys()))
    m = fitted_models[selected_resp]
    fit_stats = m["fit_statistics"]

    st.markdown(f"### 🧪 ANOVA Table: `{selected_resp}`")
    df_anova = pd.DataFrame(m["anova"])
    st.dataframe(df_anova, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Model Fit Quality & Precision Statistics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("R² (Determination)", f"{fit_stats['R2']:.4f}")
        st.metric("Model F-Value", f"{fit_stats['Model_F']:.2f}")
    with c2:
        st.metric("Adjusted R²", f"{fit_stats['Adjusted_R2']:.4f}")
        st.metric("Model p-Value", f"{fit_stats['Model_p']:.5f}")
    with c3:
        st.metric("Predicted R² (PRESS)", f"{fit_stats['Predicted_R2']:.4f}")
        lof_p = fit_stats.get("LoF_p")
        st.metric("Lack of Fit p-Value", f"{lof_p:.4f}" if lof_p is not None else "N/A")
    with c4:
        st.metric("Adequate Precision", f"{fit_stats['Adequate_Precision']:.2f}")
        st.metric("C.V. %", f"{fit_stats['CV_Percent']:.2f}%")

    st.markdown("---")
    st.markdown("### 📋 Design-Expert Style Statistical Compliance Checklist")

    # Checklist Checks
    c_check1, c_check2 = st.columns(2)
    with c_check1:
        # Check 1: Model significance
        if fit_stats["Model_p"] < 0.05:
            st.success("✅ **Model Significance:** p < 0.05. The polynomial model explains significant variance.")
        else:
            st.error("❌ **Model Significance:** p >= 0.05. The model is statistically insignificant.")

        # Check 2: Lack of fit
        if lof_p is not None:
            if lof_p > 0.05:
                st.success("✅ **Lack of Fit:** p > 0.05 (Non-significant). Good! The model fits the experimental variation.")
            else:
                st.warning("⚠️ **Lack of Fit:** p < 0.05 (Significant). Possible unaccounted curvature or higher-order terms.")
        else:
            st.info("ℹ️ Pure error replicates not available to calculate Lack of Fit.")

    with c_check2:
        # Check 3: Adj R2 vs Pred R2
        r2_diff = fit_stats["R2_Diff"]
        if r2_diff <= 0.20:
            st.success(f"✅ **R² Consistency:** |Adj R² - Pred R²| = {r2_diff:.4f} (<= 0.20). Excellent predictive agreement.")
        else:
            st.warning(f"⚠️ **R² Consistency:** Difference ({r2_diff:.4f}) is > 0.20. Watch for influential outliers.")

        # Check 4: Adequate precision
        if fit_stats["Adequate_Precision"] >= 4.0:
            st.success(f"✅ **Adequate Precision:** {fit_stats['Adequate_Precision']:.2f} (> 4.0). Signal is adequate to navigate the design space.")
        else:
            st.warning(f"⚠️ **Adequate Precision:** {fit_stats['Adequate_Precision']:.2f} (< 4.0). Signal-to-noise ratio is low.")


# ==============================================================================
# SECTION 7: RESPONSE SURFACE PLOTS
# ==============================================================================
elif "7. 📈 Response Surface Plots" in menu_choice:
    st.markdown('<div class="main-title">📈 Response Surface Visualizations & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive 2D Contour Topologies, 3D Plotly Surfaces, Parity Plots, and Diagnostics</div>', unsafe_allow_html=True)

    if not current_project:
        st.warning("Please select a project first.")
        st.stop()

    saved_design = load_design(current_project["id"])
    df_data = load_experimental_data(saved_design["id"]) if saved_design else None
    fitted_models = load_models(current_project["id"])

    if not fitted_models:
        st.warning("No fitted models found. Please fit models in **5. 📊 Statistical Analysis** first.")
        st.stop()

    factors = current_project["factors"]
    factor_codes = [f["code"] for f in factors]
    factors_by_code = {f["code"]: f for f in factors}

    resp_selected = st.selectbox("Select Response:", list(fitted_models.keys()))
    m = fitted_models[resp_selected]

    # Reconstruct statsmodels OLS model object for predictions
    col_name = f"{resp_selected} ({current_project['responses'][0]['unit']})"
    for r in current_project["responses"]:
        if r["name"] == resp_selected:
            col_name = f"{r['name']} ({r['unit']})"
            break

    coded_cols = [f"Factor_{c}_Coded" for c in factor_codes]
    X_coded = df_data[coded_cols].values
    y = df_data[col_name].values
    model_obj_dict = fit_rsm_model(X_coded, y, factors, resp_selected, r["unit"], m["model_type"])

    plot_tab1, plot_tab2, plot_tab3, plot_tab4, plot_tab5 = st.tabs([
        "2D Contour Plots", "3D Interactive Surface", "Actual vs. Predicted",
        "Residual Diagnostics Suite", "Perturbation Curve"
    ])

    with plot_tab1:
        st.subheader("2D Contour Map")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            fx_code = st.selectbox("X-Axis Factor", factor_codes, index=0, key="cont_x")
        with c_p2:
            remaining = [c for c in factor_codes if c != fx_code]
            fy_code = st.selectbox("Y-Axis Factor", remaining, index=0, key="cont_y")

        # Sliders for holding other factors
        hold_vals = {}
        other_factors = [c for c in factor_codes if c not in [fx_code, fy_code]]
        if other_factors:
            st.markdown("**Hold Remaining Variables at:**")
            h_cols = st.columns(len(other_factors))
            for i, c in enumerate(other_factors):
                f = factors_by_code[c]
                with h_cols[i]:
                    hold_vals[c] = st.slider(f"{f['code']}: {f['name']} ({f['unit']})", float(f["low"]), float(f["high"]), float(f["center"]))
        else:
            hold_vals = {}

        fig_contour = create_contour_plot(model_obj_dict, factors, fx_code, fy_code, hold_vals)
        st.plotly_chart(fig_contour, use_container_width=True)

    with plot_tab2:
        st.subheader("3D Interactive Response Surface")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            s_fx = st.selectbox("Surface X-Axis Factor", factor_codes, index=0, key="surf_x")
        with c_s2:
            rem_s = [c for c in factor_codes if c != s_fx]
            s_fy = st.selectbox("Surface Y-Axis Factor", rem_s, index=0, key="surf_y")

        s_holds = {}
        other_s = [c for c in factor_codes if c not in [s_fx, s_fy]]
        if other_s:
            s_h_cols = st.columns(len(other_s))
            for i, c in enumerate(other_s):
                f = factors_by_code[c]
                with s_h_cols[i]:
                    s_holds[c] = st.slider(f"Hold {f['code']}: {f['name']}", float(f["low"]), float(f["high"]), float(f["center"]), key=f"sh_{c}")

        fig_3d = create_3d_surface_plot(model_obj_dict, factors, s_fx, s_fy, s_holds, experimental_df=df_data)
        st.plotly_chart(fig_3d, use_container_width=True)

    with plot_tab3:
        st.subheader("Actual Experimental vs. Model Predicted Values")
        fig_parity = create_actual_vs_predicted_plot(model_obj_dict)
        st.plotly_chart(fig_parity, use_container_width=True)

    with plot_tab4:
        st.subheader("Statistical Residual Diagnostics Suite")
        fig_rp, fig_qq, fig_hist, fig_cooks = create_residual_diagnostics_suite(model_obj_dict)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.plotly_chart(fig_rp, use_container_width=True)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_d2:
            st.plotly_chart(fig_qq, use_container_width=True)
            st.plotly_chart(fig_cooks, use_container_width=True)

    with plot_tab5:
        st.subheader("Factor Perturbation Analysis")
        st.caption("Displays relative response sensitivity as each factor deviates from the reference center formulation.")
        fig_pert = create_perturbation_plot(model_obj_dict, factors)
        st.plotly_chart(fig_pert, use_container_width=True)


# ==============================================================================
# SECTION 8: FORMULATION OPTIMIZATION
# ==============================================================================
elif "8. 🎯 Formulation Optimization" in menu_choice:
    st.markdown('<div class="main-title">🎯 Multi-Response Numerical Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Derringer-Suich Desirability Optimization & Predicted Optimum Formulation Specifications</div>', unsafe_allow_html=True)

    if not current_project:
        st.warning("Please select a project first.")
        st.stop()

    fitted_models = load_models(current_project["id"])
    if not fitted_models:
        st.warning("No fitted models found. Please fit models first.")
        st.stop()

    saved_design = load_design(current_project["id"])
    df_data = load_experimental_data(saved_design["id"])
    factors = current_project["factors"]
    factor_codes = [f["code"] for f in factors]

    # Reconstruct model dicts with model_objects
    models_dict_full = {}
    for r_name in fitted_models:
        for r in current_project["responses"]:
            if r["name"] == r_name:
                unit = r["unit"]
                break
        col_name = f"{r_name} ({unit})"
        X_coded = df_data[[f"Factor_{c}_Coded" for c in factor_codes]].values
        y = df_data[col_name].values
        models_dict_full[r_name] = fit_rsm_model(X_coded, y, factors, r_name, unit, fitted_models[r_name]["model_type"])

    st.subheader("1. Optimization Goals & Critical Acceptance Criteria")
    st.caption("Adjust optimization criteria, goals (Minimize, Maximize, Target, Range), and importance weights (1 to 5).")

    criteria = {}
    for r in current_project["responses"]:
        if r["name"] in models_dict_full:
            c1, c2, c3, c4, c5 = st.columns([1.2, 0.8, 0.8, 0.8, 0.6])
            with c1:
                st.markdown(f"**{r['name']} ({r['unit']})**")
            with c2:
                g = st.selectbox(f"Goal", ["minimize", "maximize", "target", "range", "none"], index=["minimize", "maximize", "target", "range", "none"].index(r["goal"]), key=f"opt_g_{r['name']}")
            with c3:
                low_b = st.number_input(f"Lower", value=float(r["low"]), key=f"opt_l_{r['name']}")
            with c4:
                high_b = st.number_input(f"Upper", value=float(r["high"]), key=f"opt_h_{r['name']}")
            with c5:
                imp = st.slider(f"Weight", 1, 5, int(r.get("importance", 3)), key=f"opt_w_{r['name']}")

            target_val = None
            if g == "target":
                target_val = (low_b + high_b) / 2.0

            criteria[r["name"]] = {
                "goal": g,
                "low": low_b,
                "high": high_b,
                "target": target_val,
                "importance": imp
            }

    st.markdown("---")
    if st.button("🚀 Run Multi-Response Desirability Optimization"):
        with st.spinner("Executing Differential Evolution and multi-start gradient search across factor design space..."):
            opt_result = optimize_formulation(models_dict_full, factors, criteria)
            save_optimization(
                current_project["id"],
                opt_result["composite_desirability"],
                opt_result["optimal_actual_factors"],
                opt_result["predicted_responses"],
                opt_result["candidate_solutions"],
                criteria
            )
            st.success("Formulation optimization converged successfully!")
            st.rerun()

    # Display Saved Optimization
    saved_opt = load_optimization(current_project["id"])
    if saved_opt:
        st.markdown("---")
        st.subheader("2. Optimal Nanoemulgel Formulation Solution")

        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#1a446c; margin-top:0px;">🌟 Global Composite Desirability: D = {saved_opt['composite_desirability']:.4f}</h3>
            <p>Operating strictly within experimental factor boundaries. No extrapolation beyond studied concentration ranges.</p>
        </div>
        """, unsafe_allow_html=True)

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.markdown("#### 🧪 Optimized Ingredient Concentrations")
            opt_rows = []
            for code, finfo in saved_opt["optimal_factors"].items():
                opt_rows.append({
                    "Factor": f"Factor {code}",
                    "Ingredient Name": finfo["name"],
                    "Optimized Concentration": finfo["value"],
                    "Unit": finfo["unit"]
                })
            st.dataframe(pd.DataFrame(opt_rows), use_container_width=True)

        with col_opt2:
            st.markdown("#### 🎯 Predicted Response Characteristics & 95% Intervals")
            pred_rows = []
            for r_name, pinfo in saved_opt["predicted_responses"].items():
                pred_rows.append({
                    "Response (CQA)": r_name,
                    "Predicted Mean": pinfo["predicted_mean"],
                    "Unit": pinfo["unit"],
                    "95% CI Range": f"[{pinfo['ci_95_lower']}, {pinfo['ci_95_upper']}]",
                    "95% PI Range": f"[{pinfo['pi_95_lower']}, {pinfo['pi_95_upper']}]"
                })
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)

        st.markdown("#### 📋 Alternative High-Desirability Candidate Formulations")
        cands = saved_opt.get("candidate_solutions", [])
        if cands:
            cand_table = []
            for idx, c in enumerate(cands):
                row = {"Candidate #": idx + 1, "Desirability (D)": c["desirability"]}
                for code, finfo in c["actual_factors"].items():
                    row[f"{code}_{finfo['name']} ({finfo['unit']})"] = finfo["value"]
                cand_table.append(row)
            st.dataframe(pd.DataFrame(cand_table), use_container_width=True)


# ==============================================================================
# SECTION 9: EXPERIMENTAL VALIDATION
# ==============================================================================
elif "9. 🔬 Experimental Validation" in menu_choice:
    st.markdown('<div class="main-title">🔬 Experimental Confirmation & Model Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Fabricate the Optimized Formulation in the Laboratory and Compare Against 95% Prediction Intervals</div>', unsafe_allow_html=True)

    if not current_project:
        st.warning("Please select a project first.")
        st.stop()

    saved_opt = load_optimization(current_project["id"])
    if not saved_opt:
        st.warning("No optimization results found. Please run optimization in **8. 🎯 Formulation Optimization** first.")
        st.stop()

    st.subheader("1. Predicted Optimized Formulation Recipe")
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        st.markdown("**Prepare the following laboratory confirmation batch:**")
        for code, finfo in saved_opt["optimal_factors"].items():
            st.write(f"- **{finfo['name']} ({code}):** `{finfo['value']} {finfo['unit']}`")

    with c_f2:
        st.markdown("**Expected Response Characteristics:**")
        for r_name, pinfo in saved_opt["predicted_responses"].items():
            st.write(f"- **{r_name}:** `{pinfo['predicted_mean']} {pinfo['unit']}` *(95% PI: [{pinfo['pi_95_lower']}, {pinfo['pi_95_upper']}])*")

    st.markdown("---")
    st.subheader("2. Enter Experimentally Measured Confirmation Results")
    st.caption("Measure the prepared formulation in triplicates and enter average observed laboratory values.")

    obs_inputs = {}
    v_cols = st.columns(len(saved_opt["predicted_responses"]))
    for i, (r_name, pinfo) in enumerate(saved_opt["predicted_responses"].items()):
        with v_cols[i % len(v_cols)]:
            obs_inputs[r_name] = st.number_input(
                f"Observed {r_name} ({pinfo['unit']})",
                value=float(pinfo["predicted_mean"]),
                key=f"obs_in_{r_name}"
            )

    if st.button("Evaluate Validation Concordance"):
        df_val, val_sum = evaluate_validation_results(saved_opt["predicted_responses"], obs_inputs)
        save_validation(
            current_project["id"],
            saved_opt["predicted_responses"],
            obs_inputs,
            df_val.to_dict(orient="records"),
            val_sum["overall_status"],
            val_sum["scientific_assessment"]
        )
        st.success("Validation evaluated and recorded!")
        st.rerun()

    # Display Saved Validation
    saved_val = load_validation(current_project["id"])
    if saved_val:
        st.markdown("---")
        st.subheader("3. Validation Summary & Error Analysis Table")

        df_v_display = pd.DataFrame(saved_val["error_table"])
        st.dataframe(df_v_display, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
            <b>Scientific Assessment & Dissertation Evaluation:</b><br>
            {saved_val['scientific_assessment']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 Predicted vs. Observed Comparison Chart")
        fig_v = create_validation_comparison_chart(df_v_display)
        st.plotly_chart(fig_v, use_container_width=True)


# ==============================================================================
# SECTION 10: REPORTS & DOWNLOADS
# ==============================================================================
elif "10. 📄 Reports & Downloads" in menu_choice:
    st.markdown('<div class="main-title">📄 Pharmaceutical Research Reports & Data Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Export Academic PDF Thesis Report, Multi-Sheet Excel Workbooks, and Project Data</div>', unsafe_allow_html=True)

    if not current_project:
        st.warning("Please select a project first.")
        st.stop()

    saved_design = load_design(current_project["id"])
    df_data = load_experimental_data(saved_design["id"]) if saved_design else None
    fitted_models = load_models(current_project["id"])
    saved_opt = load_optimization(current_project["id"])
    saved_val = load_validation(current_project["id"])

    st.subheader("1. Export Multi-Sheet Research Excel Workbook (.xlsx)")
    st.write("Generates a complete multi-sheet spreadsheet containing: Project Details, Factors, Responses, Design Matrix, Experimental Data, ANOVA Decomposition, Regression Equations, Optimization, and Validation.")

    excel_bytes = export_complete_project_to_excel(
        project=current_project,
        df_design=saved_design["dataframe"] if saved_design else None,
        df_data=df_data,
        models=fitted_models,
        optimization_res=saved_opt,
        df_validation=pd.DataFrame(saved_val["error_table"]) if saved_val else None
    )

    st.download_button(
        "📥 Download Multi-Sheet Excel Workbook (.xlsx)",
        data=excel_bytes,
        file_name=f"Nanoemulgel_QbD_{current_project['name'].replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")
    st.subheader("2. Generate Academic PDF Research Report (.pdf)")
    st.write("Compiles a publication-ready research report formatted for B.Pharm project dissertation submission, defense presentations, and academic faculty review.")

    if st.button("Generate Academic PDF Report"):
        with st.spinner("Compiling academic PDF document..."):
            try:
                pdf_bytes = generate_pdf_report(
                    project=current_project,
                    factors=current_project["factors"],
                    responses=current_project["responses"],
                    df_design=saved_design["dataframe"] if saved_design else None,
                    df_data=df_data,
                    models=fitted_models,
                    optimization_res=saved_opt,
                    df_validation=pd.DataFrame(saved_val["error_table"]) if saved_val else None,
                    validation_summary=saved_val
                )
                st.download_button(
                    "📥 Download Formal Academic PDF Report (.pdf)",
                    data=pdf_bytes,
                    file_name=f"PharmaQbD_Dissertation_Report_{current_project['name'].replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
                st.success("PDF generated successfully!")
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

    st.markdown("---")
    st.subheader("3. Project JSON Backup & Restore")
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        backup_dict = {
            "project": current_project,
            "design": saved_design["dataframe"].to_dict(orient="records") if saved_design else None,
            "data": df_data.to_dict(orient="records") if df_data is not None else None,
            "models": fitted_models,
            "optimization": saved_opt,
            "validation": saved_val
        }
        json_str = json.dumps(backup_dict, indent=2, default=str)
        st.download_button(
            "📥 Backup Full Project State (JSON)",
            data=json_str.encode('utf-8'),
            file_name=f"Project_Backup_{current_project['name'].replace(' ', '_')}.json",
            mime="application/json"
        )
