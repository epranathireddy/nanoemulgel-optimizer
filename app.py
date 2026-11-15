import io
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nanoemulgel Optimizer", page_icon="🧪", layout="wide")
st.title("🧪 Nanoemulgel Formulation Design & Optimization")
st.caption("Python-assisted DoE, response modelling, and optimization. Experimental measurements must come from your laboratory.")

FACTORS = [
    ("A", "Oil concentration (%)"),
    ("B", "Surfactant concentration (%)"),
    ("C", "Co-surfactant concentration (%)"),
]
RESPONSES = [
    "Particle_Size_nm", "PDI", "Zeta_Potential_mV", "Entrapment_Efficiency_pct",
    "Drug_Content_pct", "Viscosity_cP", "Spreadability", "Drug_Release_pct"
]


def make_bbd():
    rows = []
    for a, b in itertools.product([-1, 1], repeat=2):
        rows.append([a, b, 0])
    for a, c in itertools.product([-1, 1], repeat=2):
        rows.append([a, 0, c])
    for b, c in itertools.product([-1, 1], repeat=2):
        rows.append([0, b, c])
    rows.extend([[0, 0, 0] for _ in range(5)])
    return np.array(rows, dtype=float)


def coded_to_actual(code, low, center, high):
    return low if code == -1 else center if code == 0 else high


def make_design(levels, seed=42):
    coded = make_bbd()
    rng = np.random.default_rng(seed)
    coded = coded[rng.permutation(len(coded))]
    actual = []
    for row in coded:
        actual.append([coded_to_actual(row[i], *levels[i]) for i in range(3)])
    df = pd.DataFrame(actual, columns=[name for _, name in FACTORS])
    df.insert(0, "Run", np.arange(1, len(df) + 1))
    df["A_coded"], df["B_coded"], df["C_coded"] = coded[:, 0], coded[:, 1], coded[:, 2]
    for response in RESPONSES:
        df[response] = np.nan
    return df


def coded_formula(response):
    return (f'{response} ~ A_coded + B_coded + C_coded + '
            'A_coded:B_coded + A_coded:C_coded + B_coded:C_coded + '
            'I(A_coded**2) + I(B_coded**2) + I(C_coded**2)')


def response_desirability(value, goal, lower, upper):
    if not np.isfinite(value) or upper <= lower:
        return 0.0
    x = np.clip((value - lower) / (upper - lower), 0, 1)
    if goal == "Maximize":
        return float(x)
    if goal == "Minimize":
        return float(1 - x)
    # Target: peak at target value (center of selected range)
    return float(1 - abs(2 * x - 1))


with st.sidebar:
    st.header("1. Define formulation factors")
    st.write("Enter the low, center, and high levels that you selected from preliminary studies.")
    levels = []
    for code, name in FACTORS:
        st.subheader(f"{code}: {name}")
        low = st.number_input(f"{code} low", value={"A": 5.0, "B": 2.0, "C": 1.0}[code], key=f"{code}_low")
        center = st.number_input(f"{code} center", value={"A": 10.0, "B": 4.0, "C": 2.0}[code], key=f"{code}_center")
        high = st.number_input(f"{code} high", value={"A": 15.0, "B": 6.0, "C": 3.0}[code], key=f"{code}_high")
        if not low < center < high:
            st.warning(f"For {code}, values must satisfy low < center < high.")
        levels.append((low, center, high))
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    create = st.button("Generate / reset 17-run BBD", type="primary", use_container_width=True)

if create:
    if any(not (lo < ce < hi) for lo, ce, hi in levels):
        st.error("Please ensure low < center < high for all factors.")
    else:
        st.session_state["data"] = make_design(levels, int(seed))
        st.success("Design generated. The experimental run order is randomized.")

if "data" not in st.session_state:
    st.session_state["data"] = make_design(levels, int(seed))

data = st.session_state["data"]

page = st.radio("Choose workflow step", ["1 — Experimental design", "2 — Enter lab results", "3 — Statistical analysis", "4 — Optimization"], horizontal=True)

if page == "1 — Experimental design":
    st.subheader("Box–Behnken experimental design")
    st.info("This page creates the batch plan only. It does not produce laboratory results or claim an optimized formulation.")
    st.dataframe(data[["Run", *[name for _, name in FACTORS], "A_coded", "B_coded", "C_coded"]], use_container_width=True, hide_index=True)
    excel = io.BytesIO()
    data.to_excel(excel, index=False)
    st.download_button("Download design and response template (.xlsx)", excel.getvalue(), "nanoemulgel_BBD_template.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("Prepare the batches in the randomized run order, measure responses, then enter the results on the next page.")

elif page == "2 — Enter lab results":
    st.subheader("Enter measured response values")
    st.write("Enter actual results from laboratory testing. Leave unmeasured response cells blank.")
    edited = st.data_editor(data, use_container_width=True, num_rows="fixed", key="response_editor")
    st.session_state["data"] = edited
    up = st.file_uploader("Or upload a completed Excel template", type=["xlsx"])
    if up is not None:
        try:
            uploaded = pd.read_excel(up)
            required = {"Run", "A_coded", "B_coded", "C_coded"}
            if required.issubset(uploaded.columns):
                st.session_state["data"] = uploaded
                st.success("Uploaded spreadsheet loaded into the app. Review the values above.")
            else:
                st.error("Spreadsheet must include Run, A_coded, B_coded, and C_coded columns.")
        except Exception as exc:
            st.error(f"Could not read Excel file: {exc}")

elif page == "3 — Statistical analysis":
    st.subheader("Quadratic response-surface model and ANOVA")
    response = st.selectbox("Choose response to analyze", RESPONSES)
    current = st.session_state["data"].copy()
    current[response] = pd.to_numeric(current[response], errors="coerce")
    model_data = current.dropna(subset=[response, "A_coded", "B_coded", "C_coded"])
    if len(model_data) < 10:
        st.warning(f"At least 10 complete measured rows are recommended for the 10-term quadratic model. Current complete rows: {len(model_data)}.")
    else:
        try:
            model = smf.ols(coded_formula(response), data=model_data).fit()
            st.metric("R-squared", f"{model.rsquared:.4f}")
            st.metric("Adjusted R-squared", f"{model.rsquared_adj:.4f}")
            st.write("**Regression coefficients**")
            st.dataframe(model.params.rename("Coefficient").to_frame(), use_container_width=True)
            st.write("**ANOVA table**")
            st.dataframe(anova_lm(model, typ=2), use_container_width=True)
            st.write("**Model summary**")
            st.text(model.summary().as_text())
            fig, ax = plt.subplots()
            ax.scatter(model.fittedvalues, model_data[response])
            limits = [min(model.fittedvalues.min(), model_data[response].min()), max(model.fittedvalues.max(), model_data[response].max())]
            ax.plot(limits, limits, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Observed")
            ax.set_title(f"Observed vs predicted: {response}")
            st.pyplot(fig)
        except Exception as exc:
            st.error(f"Model fitting failed: {exc}")

elif page == "4 — Optimization":
    st.subheader("Multi-response desirability optimization")
    st.warning("Optimization is based on fitted models and user-defined goals. Validate the predicted optimum experimentally before making research conclusions.")
    current = st.session_state["data"].copy()
    goals = {}
    models = {}
    response_choices = st.multiselect("Responses to include", ["Particle_Size_nm", "PDI", "Entrapment_Efficiency_pct", "Drug_Content_pct", "Drug_Release_pct", "Viscosity_cP", "Spreadability", "Zeta_Potential_mV"], default=["Particle_Size_nm", "PDI", "Entrapment_Efficiency_pct"])
    for response in response_choices:
        values = pd.to_numeric(current[response], errors="coerce")
        complete = current.dropna(subset=[response, "A_coded", "B_coded", "C_coded"]).copy()
        if len(complete) < 10:
            st.warning(f"{response}: need at least 10 complete experimental results for fitting; skipping this response.")
            continue
        models[response] = smf.ols(coded_formula(response), data=complete).fit()
        default_goal = "Minimize" if response in ["Particle_Size_nm", "PDI"] else "Maximize"
        goal = st.selectbox(f"Goal for {response}", ["Minimize", "Maximize", "Target"], index=["Minimize", "Maximize", "Target"].index(default_goal), key=f"goal_{response}")
        lower_default = float(values.min())
        upper_default = float(values.max())
        if upper_default <= lower_default:
            st.warning(f"{response} has no observed range; skipped.")
            models.pop(response, None)
            continue
        col1, col2 = st.columns(2)
        lower = col1.number_input(f"{response} acceptable lower", value=lower_default, key=f"lower_{response}")
        upper = col2.number_input(f"{response} acceptable upper", value=upper_default, key=f"upper_{response}")
        goals[response] = (goal, lower, upper)
    if st.button("Find best factor combination", type="primary"):
        if not models:
            st.error("Enter at least 10 complete experimental results for one or more selected responses.")
        elif any(lo >= hi for _, lo, hi in goals.values()):
            st.error("For each selected response, acceptable lower must be less than acceptable upper.")
        else:
            def objective(x):
                predictions = {r: float(m.predict(pd.DataFrame({"A_coded": [x[0]], "B_coded": [x[1]], "C_coded": [x[2]]}))[0]) for r, m in models.items()}
                ds = [response_desirability(predictions[r], *goals[r]) for r in models]
                overall = float(np.prod(ds) ** (1 / len(ds))) if ds else 0.0
                return -overall
            result = differential_evolution(objective, [(-1, 1), (-1, 1), (-1, 1)], seed=42, polish=True)
            coded_opt = result.x
            actual_opt = [levels[i][1] + coded_opt[i] * (levels[i][2] - levels[i][0]) / 2 for i in range(3)]
            st.success(f"Best overall desirability found: {-result.fun:.4f}")
            st.write("**Predicted optimum factor levels**")
            st.dataframe(pd.DataFrame({"Factor": [name for _, name in FACTORS], "Coded level": coded_opt, "Actual concentration": actual_opt}), use_container_width=True, hide_index=True)
            pred_rows = []
            for r, m in models.items():
                pred = float(m.predict(pd.DataFrame({"A_coded": [coded_opt[0]], "B_coded": [coded_opt[1]], "C_coded": [coded_opt[2]]}))[0])
                pred_rows.append({"Response": r, "Predicted value": pred, "Goal": goals[r][0], "Acceptable lower": goals[r][1], "Acceptable upper": goals[r][2]})
            st.write("**Predicted responses (must be verified experimentally)**")
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("Research note: choose factor ranges and response acceptance criteria based on your drug, preliminary screening, literature, and supervisor guidance. Statistical output is not a substitute for experimental validation.")
