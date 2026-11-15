# PharmaQbD: Nanoemulgel Formulation Optimization System

A comprehensive, research-grade, Python-based pharmaceutical formulation optimization and statistical modeling platform developed under the **Quality by Design (QbD) / ICH Q8(R2)** framework.

Designed specifically for undergraduate and postgraduate pharmacy research projects (B.Pharm / M.Pharm), this application replaces commercial formulation software (such as Design-Expert®) with a completely transparent, modular, and independent scientific computing platform.

---

## 🌟 Key Capabilities & Scientific Architecture

1. **Experimental Design (DoE):**
   - **Box–Behnken Design (BBD):** Standard 3-factor (17 runs with 5 center points) and 4-factor designs avoiding extreme multi-factor combinations that cause nanoemulsion cracking or phase separation.
   - **Central Composite Design (CCD):** 2, 3, and 4-factor designs with Face-Centered ($\alpha = 1.0$), Rotatable ($\alpha = (2^k)^{1/4}$), and Spherical ($\alpha = \sqrt{k}$) options.
   - **Simplex Mixture Design:** Handles constrained formulation proportions ($\sum X_i = 100\%$) with Scheffé canonical polynomial guidance.
   - **Reproducible Run Randomization:** Configurable random seed ensuring laboratory reproducibility.

2. **Statistical Modeling & Response Surface Methodology (RSM):**
   - Fits second-order polynomial response surface regression models:
     $$Y = \beta_0 + \sum \beta_i X_i + \sum \beta_{ii} X_i^2 + \sum \sum_{i < j} \beta_{ij} X_i X_j$$
   - Complete **ANOVA Decomposition**: Model, Linear, 2FI (Interactions), Quadratic, Residual, Lack of Fit, Pure Error, and Total Corrected.
   - **Fit Quality Statistics:** $R^2$, Adjusted $R^2$, Predicted $R^2$ (via PRESS and studentized leverage statistics), Difference $|R^2_{adj} - R^2_{pred}| \le 0.20$, and Adequate Precision (Signal-to-Noise $> 4.0$).
   - Regression equations generated in both coded and actual physical concentration units.

3. **Multi-Response Desirability Optimization:**
   - Numerical optimization using the **Derringer–Suich Desirability Function** ($D = (\prod d_i^{w_i})^{1/\sum w_i}$).
   - Differential Evolution coupled with multi-start gradient search strictly bounded within experimental factor limits $[-1.0, +1.0]$.
   - Calculates predicted mean values, 95% Confidence Intervals (CI) for the mean, and 95% Prediction Intervals (PI) for future confirmation batches.

4. **Experimental Validation & Confirmation:**
   - Compares laboratory confirmation batch results against model predictions.
   - Percentage Prediction Error:
     $$\text{Prediction Error \%} = \frac{|\text{Experimental Value} - \text{Predicted Value}|}{|\text{Predicted Value}|} \times 100\%$$
   - 95% Prediction Interval compliance verification with academic dissertation-ready scientific assessment.

5. **Reporting & Publication Exports:**
   - Multi-sheet Excel workbook (`.xlsx`) containing metadata, factors, responses, design matrix, laboratory observations, ANOVA, equations, optimization, and validation.
   - Formal Academic PDF Research Report (`.pdf`) generated with FPDF2, including signature blocks for student investigator, research guide, and department head.

---

## 📁 Project Directory Structure

```text
nanoemulgel_optimizer/
├── app.py              # Main Streamlit interactive research dashboard
├── config.py           # Excipient catalogues, default factors/responses, thresholds, glossary
├── database.py         # SQLite persistent storage engine (nanoemulgel.db)
├── doe.py              # Experimental design generation algorithms (BBD, CCD, Mixture)
├── analysis.py         # RSM polynomial regression, ANOVA, PRESS, and diagnostics
├── visualization.py    # 2D contours, 3D Plotly response surfaces, diagnostics, perturbations
├── optimization.py     # Derringer-Suich desirability optimization (Differential Evolution)
├── validation.py       # Laboratory confirmation error evaluation and 95% PI verification
├── utils.py            # Excel multi-sheet export, PDF compiler (FPDF2), synthetic benchmark data
├── requirements.txt    # Python library dependencies
└── README.md           # System documentation and instructions
```

---

## 🚀 Installation & Launch Instructions

### 1. Prerequisites
Ensure Python 3.10+ (tested up to Python 3.14) is installed.

### 2. Install Required Dependencies
Navigate to the project directory and install the packages:
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit research dashboard locally:
```bash
streamlit run app.py
```
The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 🔬 Step-by-Step B.Pharm Research Walkthrough

1. **Dashboard (Section 1):** View project status and QbD workflow milestones.
2. **Project Setup (Section 2):** Enter drug name (e.g. *Diclofenac Sodium*), excipients (Clove oil, Tween 80, Transcutol P, Carbopol 940), factor ranges, and critical quality attributes.
3. **Experimental Design (Section 3):** Select Box-Behnken Design with 5 center points (17 runs). Review randomized run order and download the Excel protocol for laboratory batch preparation.
4. **Data Entry (Section 4):** Prepare experimental batches and record observed droplet size (nm), PDI, zeta potential (mV), entrapment efficiency (%), drug content (%), viscosity (cP), spreadability (cm), and release (%). *(A one-click synthetic benchmark dataset is provided for testing).*
5. **Statistical Analysis (Section 5):** Fit second-order quadratic response surface equations. Review coefficient estimates, standard errors, and p-values.
6. **ANOVA & Regression (Section 6):** Verify model significance ($p < 0.05$), non-significant Lack of Fit ($p > 0.05$), high $R^2$, and Adequate Precision ($> 4.0$).
7. **Response Surface Plots (Section 7):** Explore interactive 2D contour maps and 3D Plotly surfaces showing interactions between oil, surfactant, and co-surfactant.
8. **Formulation Optimization (Section 8):** Set goals (Minimize Particle Size, Minimize PDI, Maximize Entrapment, Target Viscosity). Execute Derringer-Suich optimization to discover the global optimal recipe.
9. **Experimental Validation (Section 9):** Prepare the predicted optimal formulation in the lab, enter observed values, and confirm adherence to 95% Prediction Intervals.
10. **Reports & Downloads (Section 10):** Download complete Excel workbooks and formal academic PDF reports ready for dissertation binding.

---

## 🧪 Scientific Rigor & Ethics Note
- **No Fabricated Data:** The application never generates fake experimental results for research conclusions.
- **Testing Benchmark Data:** The included benchmark dataset is explicitly labeled `[SYNTHETIC TEST BENCHMARK - FOR VERIFICATION ONLY]` for software validation.
- **Model Hierarchy:** Lower-order parent terms are preserved when higher-order interaction or quadratic terms are retained.
- **Domain Bounded:** The optimization engine enforces strict boundaries within the evaluated factor domain $[-1.0, +1.0]$, preventing unverified mathematical extrapolation.
