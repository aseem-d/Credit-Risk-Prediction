import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model import predict_probability

ROOT = Path(__file__).resolve().parent
GITHUB_URL = "https://github.com/aseem-d/Credit-Risk-Prediction"

st.set_page_config(page_title="Credit Risk Prediction", page_icon="📊", layout="wide")
st.title("Credit Risk Prediction")
st.caption("A project by Aseem Deshpande")
st.write("Explore an estimated probability of serious delinquency within the next two years.")

with st.form("risk_inputs"):
    st.subheader("Your information")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=18, max_value=120, value=40)
        income_unknown = st.checkbox("I don't know my monthly income")
        income = st.number_input("Monthly income", min_value=0.0, value=5000.0, step=500.0, disabled=income_unknown, help="Gross income per month. Use the checkbox if you do not know this amount.")
        utilization = st.number_input("Revolving credit utilization", min_value=0.0, value=0.30, step=0.05, help="Balance on credit cards and personal credit lines divided by their available limits. It can be above 1.")
        debt_ratio = st.number_input("Debt ratio", min_value=0.0, value=0.35, step=0.05, help="Monthly debt payments relative to monthly income. Very large values occur in the source data.")
        open_lines = st.number_input("Open credit lines and loans", min_value=0, value=5, step=1)
    with right:
        real_estate = st.number_input("Real estate loans or lines", min_value=0, value=1, step=1)
        dependents = st.number_input("Number of dependents", min_value=0, value=0, step=1)
        late_30 = st.number_input("Times 30–59 days past due", min_value=0, value=0, step=1, help="Enter the number of recorded occurrences.")
        late_60 = st.number_input("Times 60–89 days past due", min_value=0, value=0, step=1)
        late_90 = st.number_input("Times 90+ days past due", min_value=0, value=0, step=1)
    submitted = st.form_submit_button("Estimate Risk", type="primary")

if submitted:
    raw = {
        "age": age, "MonthlyIncome": None if income_unknown else income,
        "RevolvingUtilizationOfUnsecuredLines": utilization, "DebtRatio": debt_ratio,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberRealEstateLoansOrLines": real_estate, "NumberOfDependents": dependents,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60,
        "NumberOfTimes90DaysLate": late_90,
    }
    try:
        probability = predict_probability(raw)
        st.subheader(f"Estimated probability of serious delinquency within the next two years: {probability:.1%}")
        st.progress(probability)
        st.caption("This is the probability estimated by the XGBoost model based on the information entered.")
    except FileNotFoundError as exc:
        st.error(f"The model artifact is missing. {exc}")

with (ROOT / "results" / "model_metrics.json").open() as f:
    evaluation = json.load(f)
metrics = pd.DataFrame(evaluation["models"])

with st.expander("About the model"):
    st.markdown("""The deployed model is an **unweighted XGBoost** classifier selected in the analysis. Logistic Regression and Random Forest were also compared. The target is serious delinquency within two years. Evaluation included ROC-AUC, Average Precision, calibration curves and Brier score. This is a predictive and educational project, not a real lending decision system.""")

st.divider()
st.subheader("Model performance")
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(metrics, x="name", y="roc_auc", title="ROC-AUC", labels={"name": "Model", "roc_auc": "ROC-AUC"}, color="name")
    for trace in fig.data:
        if trace.name == "XGBoost":
            trace.name = "<b>XGBoost</b>"
    fig.update_xaxes(
        tickmode="array",
        tickvals=["XGBoost", "Logistic Regression", "Random Forest"],
        ticktext=["<b>XGBoost</b>", "Logistic Regression", "Random Forest"],
    )
    st.plotly_chart(fig, width="stretch")
with col2:
    fig = px.bar(metrics, x="name", y="average_precision", title="Average Precision", labels={"name": "Model", "average_precision": "Average Precision"}, color="name")
    for trace in fig.data:
        if trace.name == "XGBoost":
            trace.name = "<b>XGBoost</b>"
    fig.update_xaxes(
        tickmode="array",
        tickvals=["XGBoost", "Logistic Regression", "Random Forest"],
        ticktext=["<b>XGBoost</b>", "Logistic Regression", "Random Forest"],
    )
    st.plotly_chart(fig, width="stretch")

# Calibration coordinates are saved from the notebook's shared held-out predictions.
curve_path = ROOT / "results" / "calibration_curves.json"
if curve_path.exists():
    curves = json.loads(curve_path.read_text())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line={"dash": "dash", "color": "gray"}))
    for model_name, values in curves.items():
        label = "<b>XGBoost</b>" if model_name == "XGBoost" else model_name
        fig.add_trace(go.Scatter(x=values["mean_predicted"], y=values["observed_frequency"], mode="lines+markers", name=label))
    fig.update_layout(title="Calibration on the held-out test set", xaxis_title="Mean predicted probability", yaxis_title="Observed frequency")
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Calibration points are exported when the notebook's final evaluation section is run.")
display_metrics = metrics.set_index("name").reindex(["XGBoost", "Logistic Regression", "Random Forest"])
display_metrics.index.name = "Model"
display_style = display_metrics.rename(columns={"roc_auc": "ROC-AUC", "average_precision": "Average Precision", "brier_score": "Brier Score"}).style.format("{:.4f}")
display_style = display_style.apply(
    lambda row: ["font-weight: bold" if row.name == "XGBoost" else "" for _ in row],
    axis=1,
)
st.dataframe(display_style, width="stretch")
if GITHUB_URL != "YOUR_REPOSITORY_URL":
    st.sidebar.markdown(f"[View the project on GitHub]({GITHUB_URL})")
else:
    st.sidebar.caption("GitHub link: set GITHUB_URL in app.py after repository creation.")
