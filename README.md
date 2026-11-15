# Nanoemulgel Formulation Design & Optimization Web App

A beginner-friendly Streamlit starter app for a 3-factor Box–Behnken experimental design, lab response entry, quadratic response-surface modeling, ANOVA, and desirability-based multi-response optimization.

## Important
- The app generates the experimental design; it does not generate laboratory measurements.
- Factor ranges and response goals must be chosen scientifically for your drug/formulation.
- Model predictions are provisional and require experimental validation.
- A 3-factor, 10-term quadratic model needs adequate observations; use the full 17-run design and actual results.

## Folder structure

```text
nanoemulgel_webapp/
├── app.py
├── requirements.txt
└── README.md
```

## Run locally (Windows)

1. Install Python 3.10+ and Visual Studio Code.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install packages:

   ```powershell
   pip install -r requirements.txt
   ```

6. Start the web app:

   ```powershell
   streamlit run app.py
   ```

7. Your browser should open the local app (usually `http://localhost:8501`).

## Workflow

1. Set low, center, and high factor concentrations in the sidebar.
2. Generate the randomized 17-run Box–Behnken design and download its Excel template.
3. Conduct experiments and enter measured response values (or upload completed Excel).
4. Review response-specific regression and ANOVA.
5. Select responses, goals, and acceptable response ranges, then run desirability optimization.
6. Prepare the predicted optimum and compare actual lab results with predictions.
