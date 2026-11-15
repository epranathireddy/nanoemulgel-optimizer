"""
Comprehensive Automated Test Suite for Nanoemulgel Formulation Optimization System.
Tests all scientific, statistical, optimization, visualization, and reporting modules.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_FACTORS, DEFAULT_RESPONSES, STAT_THRESHOLDS
from database import (
    init_db, save_project, load_project, list_projects,
    save_design, load_design, save_experimental_data, load_experimental_data,
    save_model, load_models, save_optimization, load_optimization,
    save_validation, load_validation
)
from doe import (
    generate_box_behnken, generate_central_composite, generate_simplex_lattice_mixture,
    create_design_matrix, code_to_actual, actual_to_code, get_design_guidance
)
from analysis import fit_rsm_model, predict_single_response
from visualization import (
    create_contour_plot, create_3d_surface_plot, create_actual_vs_predicted_plot,
    create_residual_diagnostics_suite, create_perturbation_plot, generate_static_contour_png
)
from optimization import optimize_formulation, compute_composite_desirability
from validation import evaluate_validation_results
from utils import (
    export_complete_project_to_excel, generate_pdf_report,
    generate_synthetic_benchmark_dataset
)

def run_all_tests():
    print("=" * 70)
    print("RUNNING AUTOMATED SYSTEM QUALITY CONTROL & SCIENTIFIC TEST SUITE")
    print("=" * 70)

    # 1. Test Database Initialization & CRUD
    print("\n[TEST 1/10] Database Engine & Project CRUD...")
    init_db()
    test_proj = {
        "name": "Automated Test Project",
        "drug_name": "Diclofenac Sodium",
        "dosage_form": "Nanoemulgel",
        "route": "Topical",
        "oil_phase": "Clove Oil",
        "surfactant": "Tween 80",
        "cosurfactant": "Transcutol P",
        "gelling_agent": "Carbopol 940",
        "other_excipients": "TEA, Purified Water"
    }
    pid = save_project(test_proj, DEFAULT_FACTORS, DEFAULT_RESPONSES)
    loaded_proj = load_project(pid)
    assert loaded_proj is not None, "Failed to load saved project."
    assert loaded_proj["name"] == test_proj["name"], "Project name mismatch."
    assert len(loaded_proj["factors"]) == 3, "Factor count mismatch."
    assert len(loaded_proj["responses"]) == len(DEFAULT_RESPONSES), "Response count mismatch."
    print("  -> Database CRUD passed successfully!")

    # 2. Test Experimental Design Generation (BBD & CCD)
    print("\n[TEST 2/10] Experimental Design Algorithms...")
    bbd_3 = generate_box_behnken(3, num_center=5)
    assert bbd_3.shape == (17, 3), f"3-Factor BBD must be 17x3, got {bbd_3.shape}"

    bbd_4 = generate_box_behnken(4, num_center=5)
    assert bbd_4.shape == (29, 4), f"4-Factor BBD must be 29x4, got {bbd_4.shape}"

    ccd_3, alpha = generate_central_composite(3, num_center=6, alpha_type="face_centered")
    assert ccd_3.shape == (20, 3), f"3-Factor CCD must be 20x3, got {ccd_3.shape}"
    assert alpha == 1.0, f"Face-centered alpha must be 1.0, got {alpha}"

    mix_3 = generate_simplex_lattice_mixture(3, degree=2)
    assert len(mix_3) >= 6, "Simplex lattice mixture points missing."

    # Test Coded to Actual Conversion
    act_val = code_to_actual(1.0, 5.0, 10.0, 15.0)
    assert np.isclose(act_val, 15.0), f"Expected 15.0, got {act_val}"
    act_val_center = code_to_actual(0.0, 5.0, 10.0, 15.0)
    assert np.isclose(act_val_center, 10.0), f"Expected 10.0, got {act_val_center}"
    code_val = actual_to_code(15.0, 5.0, 10.0, 15.0)
    assert np.isclose(code_val, 1.0), f"Expected 1.0, got {code_val}"

    # Test full matrix generation
    df_design, coded_mat, actual_mat = create_design_matrix(
        "Box-Behnken Design (BBD)", DEFAULT_FACTORS, num_center=5, random_seed=42
    )
    assert len(df_design) == 17, "Full design matrix length mismatch."
    save_design(pid, "Box-Behnken Design (BBD)", 3, 5, 42, df_design)
    print("  -> DoE generation and level conversion passed!")

    # 3. Test Synthetic Benchmark Dataset Generation
    print("\n[TEST 3/10] Synthetic Benchmark Dataset...")
    df_data = generate_synthetic_benchmark_dataset(DEFAULT_FACTORS, DEFAULT_RESPONSES, random_seed=42)
    assert len(df_data) == 17, "Data rows mismatch."
    assert "Particle Size (nm)" in df_data.columns, "Particle Size column missing."
    assert "Entrapment Efficiency (%)" in df_data.columns, "Entrapment column missing."
    save_experimental_data(1, df_data)
    print("  -> Synthetic benchmark dataset generation passed!")

    # 4. Test Response Surface Modeling, ANOVA, & PRESS
    print("\n[TEST 4/10] Statistical Modeling, ANOVA & PRESS Statistics...")
    factor_codes = ["A", "B", "C"]
    X_coded = df_data[[f"Factor_{c}_Coded" for c in factor_codes]].values
    y_ps = df_data["Particle Size (nm)"].values

    rsm_res = fit_rsm_model(
        X_coded=X_coded,
        y=y_ps,
        factors=DEFAULT_FACTORS,
        response_name="Particle Size",
        response_unit="nm",
        model_type="Quadratic"
    )
    fit_st = rsm_res["fit_statistics"]
    print(f"  -> Particle Size: R2={fit_st['R2']:.4f}, Adj R2={fit_st['Adjusted_R2']:.4f}, Pred R2={fit_st['Predicted_R2']:.4f}")
    print(f"  -> Adequate Precision: {fit_st['Adequate_Precision']:.2f}")
    assert fit_st["R2"] > 0.90, "Model R2 below expected threshold for quadratic synthetic data."
    assert fit_st["Predicted_R2"] > 0.80, "Predicted R2 below expected threshold."
    assert fit_st["Model_p"] < 0.05, "Model must be statistically significant."
    assert len(rsm_res["anova"]) >= 5, "ANOVA decomposition incomplete."

    # Save model
    save_model(
        pid, "Particle Size", "Quadratic", rsm_res["coded_equation"],
        rsm_res["coefficients"], rsm_res["anova"], rsm_res["fit_statistics"], rsm_res["diagnostics"]
    )

    # Fit all responses for full pipeline test
    fitted_models_all = {"Particle Size": rsm_res}
    for r in DEFAULT_RESPONSES[1:]:
        col_name = f"{r['name']} ({r['unit']})"
        if col_name in df_data.columns:
            y_r = df_data[col_name].values
            m_r = fit_rsm_model(X_coded, y_r, DEFAULT_FACTORS, r["name"], r["unit"], "Quadratic")
            fitted_models_all[r["name"]] = m_r
            save_model(pid, r["name"], "Quadratic", m_r["coded_equation"], m_r["coefficients"], m_r["anova"], m_r["fit_statistics"], m_r["diagnostics"])
    print("  -> Full ANOVA decomposition & regression passed!")

    # 5. Test 95% Confidence & Prediction Intervals
    print("\n[TEST 5/10] Prediction Intervals (95% CI / 95% PI)...")
    center_pt = np.array([0.0, 0.0, 0.0])
    y_hat, ci_half, pi_half = predict_single_response(rsm_res, center_pt, factor_codes)
    print(f"  -> Center Prediction: {y_hat:.2f} nm, 95% CI: [{y_hat-ci_half:.2f}, {y_hat+ci_half:.2f}], 95% PI: [{y_hat-pi_half:.2f}, {y_hat+pi_half:.2f}]")
    assert pi_half > ci_half, "Prediction interval must be wider than confidence interval."
    print("  -> Prediction interval calculations passed!")

    # 6. Test Visualization Generation
    print("\n[TEST 6/10] Visualization Topologies & Diagnostics...")
    fig_cont = create_contour_plot(rsm_res, DEFAULT_FACTORS, "A", "B", {"C": 2.0})
    assert fig_cont is not None, "Contour plot generation failed."

    fig_3d = create_3d_surface_plot(rsm_res, DEFAULT_FACTORS, "A", "B", {"C": 2.0}, experimental_df=df_data)
    assert fig_3d is not None, "3D Surface plot generation failed."

    fig_parity = create_actual_vs_predicted_plot(rsm_res)
    assert fig_parity is not None, "Parity plot generation failed."

    rp, qq, hist, cooks = create_residual_diagnostics_suite(rsm_res)
    assert rp is not None and qq is not None, "Residual diagnostics suite failed."

    fig_pert = create_perturbation_plot(rsm_res, DEFAULT_FACTORS)
    assert fig_pert is not None, "Perturbation plot generation failed."

    png_bytes = generate_static_contour_png(rsm_res, DEFAULT_FACTORS, "A", "B", {"C": 2.0})
    assert len(png_bytes) > 1000, "Static PNG contour generation failed."
    print("  -> All 2D, 3D, and diagnostic visualization figures passed!")

    # 7. Test Multi-Response Desirability Optimization
    print("\n[TEST 7/10] Derringer-Suich Multi-Response Optimization...")
    criteria = {
        "Particle Size": {"goal": "minimize", "low": 50.0, "high": 200.0, "importance": 5},
        "Entrapment Efficiency": {"goal": "maximize", "low": 70.0, "high": 95.0, "importance": 5},
        "Viscosity": {"goal": "range", "low": 12000.0, "high": 25000.0, "importance": 3}
    }
    opt_res = optimize_formulation(fitted_models_all, DEFAULT_FACTORS, criteria)
    d_comp = opt_res["composite_desirability"]
    print(f"  -> Optimal Composite Desirability D = {d_comp:.4f}")
    assert d_comp > 0.0, "Composite desirability must be greater than zero."
    assert -1.05 <= opt_res["optimal_coded_factors"][0] <= 1.05, "Optimizer exceeded factor domain!"

    save_optimization(
        pid, d_comp, opt_res["optimal_actual_factors"],
        opt_res["predicted_responses"], opt_res["candidate_solutions"], criteria
    )
    print("  -> Multi-response optimization passed!")

    # 8. Test Experimental Confirmation Validation
    print("\n[TEST 8/10] Laboratory Validation Evaluation...")
    # Simulate confirmation lab measurement within PI
    obs_test = {}
    for r_name, pinfo in opt_res["predicted_responses"].items():
        obs_test[r_name] = pinfo["predicted_mean"] * 1.02  # 2% error

    df_val, val_sum = evaluate_validation_results(opt_res["predicted_responses"], obs_test)
    assert not df_val.empty, "Validation table is empty."
    assert val_sum["within_pi_count"] >= 1, "Validation PI compliance failure."
    print(f"  -> Validation Status: {val_sum['overall_status']}")
    print(f"  -> Assessment: {val_sum['scientific_assessment']}")

    save_validation(
        pid, opt_res["predicted_responses"], obs_test,
        df_val.to_dict(orient="records"), val_sum["overall_status"], val_sum["scientific_assessment"]
    )
    print("  -> Validation module passed!")

    # 9. Test Excel Multi-Sheet Export
    print("\n[TEST 9/10] Multi-Sheet Excel Workbook Export...")
    excel_bytes = export_complete_project_to_excel(
        project=loaded_proj,
        df_design=df_design,
        df_data=df_data,
        models=load_models(pid),
        optimization_res=opt_res,
        df_validation=df_val
    )
    assert len(excel_bytes) > 2000, "Excel export generated insufficient bytes."
    print(f"  -> Excel workbook generated ({len(excel_bytes)} bytes) successfully!")

    # 10. Test PDF Research Report Export
    print("\n[TEST 10/10] Academic PDF Research Report Compilation...")
    pdf_bytes = generate_pdf_report(
        project=loaded_proj,
        factors=DEFAULT_FACTORS,
        responses=DEFAULT_RESPONSES,
        df_design=df_design,
        df_data=df_data,
        models=load_models(pid),
        optimization_res=opt_res,
        df_validation=df_val,
        validation_summary=val_sum
    )
    assert len(pdf_bytes) > 1000, "PDF report compilation generated insufficient bytes."
    print(f"  -> PDF research report compiled ({len(pdf_bytes)} bytes) successfully!")

    print("\n" + "=" * 70)
    print("ALL 10/10 SYSTEM VERIFICATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
