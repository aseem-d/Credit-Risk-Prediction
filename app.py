"""Lender-facing credit-risk decision support dashboard."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.model import explain_prediction, load_model, predict_probability

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

ROOT = Path(__file__).resolve().parent
GITHUB_URL = "https://github.com/aseem-d/Credit-Risk-Prediction"
GITHUB_PROFILE_URL = "https://github.com/aseem-d"
LINKEDIN_URL = "https://www.linkedin.com/in/aseem-deshpande"
CONTACT_EMAIL = "aseemad.mds2026@cmi.ac.in"

FEATURES = {
    "RevolvingUtilizationOfUnsecuredLines": {
        "label": "Revolving credit utilization",
        "description": "Balance on credit cards and personal unsecured credit lines divided by available limits. Values can exceed 1. The source data does not specify when or over what period this was measured.",
        "unit": "Ratio",
    },
    "age": {
        "label": "Age",
        "description": "Borrower age in years. The analysis excludes the single source record with age 0.",
        "unit": "Years",
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "label": "Times 30–59 days past due",
        "description": "Number of recorded occasions when an account was 30–59 days past due, and not worse.",
        "unit": "Count",
    },
    "DebtRatio": {
        "label": "Debt ratio",
        "description": "Monthly debt payments relative to gross monthly income. The dataset contains unusually large values and has a documented data-quality anomaly.",
        "unit": "Ratio",
    },
    "MonthlyIncome": {
        "label": "Monthly income",
        "description": "Reported gross monthly income. Missing values are median-imputed by the fitted pipeline, with a separate model-generated missing-income indicator.",
        "unit": "Dataset currency per month",
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "label": "Open credit lines and loans",
        "description": "Number of open loans and credit lines, including cards and other accounts.",
        "unit": "Count",
    },
    "NumberOfTimes90DaysLate": {
        "label": "Times 90+ days late",
        "description": "Number of recorded occasions when an account was 90 or more days past due. Source values 96 and 98 are treated as suspicious/missing by the model pipeline.",
        "unit": "Count",
    },
    "NumberRealEstateLoansOrLines": {
        "label": "Real estate loans or lines",
        "description": "Number of mortgage and real-estate loans or lines.",
        "unit": "Count",
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "label": "Times 60–89 days past due",
        "description": "Number of recorded occasions when an account was 60–89 days past due, and not worse. Source values 96 and 98 are treated as suspicious/missing by the model pipeline.",
        "unit": "Count",
    },
    "NumberOfDependents": {
        "label": "Number of dependents",
        "description": "Number of dependents reported for the borrower. Missing values are filled with the most frequent training-set value.",
        "unit": "Count",
    },
    "MonthlyIncome_missing": {
        "label": "Monthly-income missing indicator (model-generated)",
        "description": "1 when monthly income was missing in the input record; otherwise 0. The dashboard generates this feature automatically.",
        "unit": "Binary indicator",
    },
    "SuspiciousDelinqValue": {
        "label": "Suspicious delinquency-value indicator (model-generated)",
        "description": "1 when any delinquency counter contained 96 or 98 before those values were treated as missing; otherwise 0. The dashboard generates this feature automatically.",
        "unit": "Binary indicator",
    },
}

st.set_page_config(page_title="Credit Risk Decision Support", page_icon="📊", layout="wide")
st.title("Credit Risk Decision Support")
st.caption("Delinquency Risk Modelling Project by Aseem Deshpande")
assessment_tab, dictionary_tab, insights_tab, results_tab, about_tab = st.tabs(
    ["Risk assessment", "Data dictionary", "Variable insights", "Model results", "About & information"]
)


@st.cache_resource
def get_model():
    return load_model()


def risk_bucket(probability):
    if probability < 0.30:
        return "Low risk", "0% to under 30%"
    if probability < 0.60:
        return "Medium risk", "30% to under 60%"
    return "High risk", "60% to 100%"


def percentile_of(value, quantiles):
    return float(np.interp(value, quantiles, np.linspace(0, 100, len(quantiles))))


with assessment_tab:
    st.subheader("Applicant information")
    with st.form("risk_inputs"):
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
            late_90 = st.number_input("Times 90+ days late", min_value=0, value=0, step=1)
        submitted = st.form_submit_button("Estimate Risk", type="primary")

    if submitted:
        raw = {
            "age": age,
            "MonthlyIncome": None if income_unknown else income,
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "DebtRatio": debt_ratio,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfDependents": dependents,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60,
            "NumberOfTimes90DaysLate": late_90,
        }
        try:
            deployed_model = get_model()
            probability = predict_probability(raw, deployed_model)
            try:
                explanation = explain_prediction(raw, deployed_model)
            except Exception:
                # The probability is useful even if the runtime cannot load
                # the optional SHAP explanation path.
                explanation = None
                st.warning("The probability was calculated, but case-specific SHAP details are unavailable in this runtime.")
            st.session_state["risk_case"] = {"inputs": raw, "probability": probability, "explanation": explanation}
        except Exception as exc:
            st.error(f"The risk estimate could not be calculated ({type(exc).__name__}). Check the app logs for details.")

    case = st.session_state.get("risk_case")
    if case:
        probability = case["probability"]
        bucket, interval = risk_bucket(probability)
        st.divider()
        st.subheader("Risk estimate")
        st.metric("Estimated probability of serious delinquency within two years", f"{probability:.1%}")
        st.progress(probability)
        st.markdown(f"**Risk band: {bucket}** · {interval}")
        st.caption("This band describes the model estimate under the displayed probability ranges. It is not an approval or rejection recommendation.")

    st.markdown("### Lending decision context")
    st.write("A risk probability is one input to a lender's decision. The economic outcome also depends on the lender's product terms, recovery assumptions, and risk appetite:")
    st.markdown(
        "- **Approve and the customer repays:** expected interest and fees can generate a profit.\n"
        "- **Approve and the customer becomes seriously delinquent:** the lender may incur unpaid principal, servicing, and recovery losses.\n"
        "- **Decline a customer who would have repaid:** the lender forgoes the expected return from that loan (opportunity cost)."
    )
    st.info("The lending decision should weigh these outcomes using the lender's own relative costs and risk appetite. The low, medium, and high bands organize model probabilities; they do not encode those costs or make the decision.")


with dictionary_tab:
    st.subheader("Model variable dictionary")
    st.write("The model predicts the probability of serious delinquency (an account 90 or more days past due) within the next two years.")
    dictionary_rows = [{"Variable": "SeriousDlqin2yrs (target)", "Description": "1 if serious delinquency occurred within two years; otherwise 0. Used as the historical outcome, not an applicant input.", "Unit / handling": "Binary outcome"}]
    for key, info in FEATURES.items():
        dictionary_rows.append({"Variable": info["label"], "Description": info["description"], "Unit / handling": info["unit"]})
    st.dataframe(pd.DataFrame(dictionary_rows), hide_index=True, width="stretch")
    st.caption("The two indicator variables are engineered automatically. They are not requested in the applicant input form.")
    st.caption("The Give Me Some Credit data does not specify the observation date for revolving utilization, and the analysis identifies unusual debt-ratio and delinquency values as data-quality limitations.")


with insights_tab:
    st.subheader("Population distribution and model contribution")
    st.write("Choose one model feature to compare its reference-population distribution with the current case and inspect its SHAP contribution.")
    profiles_path = ROOT / "results" / "feature_profiles.json"
    if not profiles_path.exists():
        st.warning("Feature insight summaries are not available. Run the notebook's final dashboard-artifact export section.")
    else:
        profiles = json.loads(profiles_path.read_text())["features"]
        available_features = [name for name in FEATURES if name in profiles]
        selected_feature = st.selectbox("Select a model variable", available_features, format_func=lambda name: FEATURES[name]["label"])
        feature_info = FEATURES[selected_feature]
        profile = profiles[selected_feature]
        st.caption(feature_info["description"])
        st.caption("Reference values show the full cleaned analysis sample after the model's fitted preprocessing. SHAP summaries use a reproducible sample from the held-out test set.")
        st.info("**How to read SHAP:** Each feature's contribution shows how it shifts this case's model score from the model's baseline. A positive value pushes the estimated risk higher; a negative value pushes it lower. A larger absolute value means a stronger influence on this estimate. This describes how the model used the feature, not cause and effect in the real world.")

        if case and case.get("explanation"):
            current_feature = case["explanation"]["features"][selected_feature]
            current_value = current_feature["model_value"]
            current_shap = current_feature["shap_value"]
            value_percentile = percentile_of(current_value, profile["value_quantiles"])
            shap_percentile = percentile_of(current_shap, profile["shap_quantiles"])
            p1, p2, p3 = st.columns(3)
            p1.metric("Model-side value", f"{current_value:,.3g}")
            p2.metric("Reference percentile", f"{value_percentile:.0f}th")
            p3.metric("This case's SHAP contribution", f"{current_shap:+.3f}")
        else:
            current_feature = None
            if case:
                st.info("The reference distributions are shown above, but a case-specific SHAP value is unavailable for this runtime.")
            else:
                st.info("Submit a risk estimate in the Risk assessment tab to show where that case falls and its individual SHAP contribution.")

        distribution, contribution = st.columns(2)
        with distribution:
            st.markdown("#### Reference distribution")
            hist = profile["value_histogram"]
            edges = hist["edges"]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
            hist_frame = pd.DataFrame({"Model-side value": centers, "Records": hist["counts"]})
            if go is not None:
                fig = go.Figure(go.Bar(x=centers, y=hist["counts"], marker_color="#6885A3"))
                if current_feature and edges[0] <= current_value <= edges[-1]:
                    fig.add_vline(x=current_value, line_color="#D55E00", line_width=3, annotation_text="Current case")
                fig.update_layout(xaxis_title="Model-side value", yaxis_title="Reference records", margin={"t": 20})
                st.plotly_chart(fig, width="stretch")
            else:
                st.bar_chart(hist_frame.set_index("Model-side value"), horizontal=False, x_label="Model-side value")
            st.caption(f"Reference sample: {profile['reference_rows']:,} records. The chart displays the central 1st–99th percentile; {hist['underflow']:,} records fall below and {hist['overflow']:,} above that range.")
            if current_feature:
                if current_value < edges[0] or current_value > edges[-1]:
                    st.caption("The current value is outside the chart's central range; use the percentile above to locate it relative to the full reference sample.")
                if selected_feature == "MonthlyIncome" and case["inputs"]["MonthlyIncome"] is None:
                    st.caption("Income was left unknown: the model imputes the training median and separately uses the missing-income indicator.")
                if selected_feature in ("NumberOfTimes90DaysLate", "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse") and case["inputs"][selected_feature] in (96, 98):
                    st.caption("The entered 96/98 value is treated as missing and median-imputed; the suspicious-value indicator is generated separately.")

        with contribution:
            st.markdown("#### SHAP contribution distribution")
            shap_hist = profile["shap_histogram"]
            shap_edges = shap_hist["edges"]
            shap_centers = [(shap_edges[i] + shap_edges[i + 1]) / 2 for i in range(len(shap_edges) - 1)]
            if go is not None:
                fig = go.Figure(go.Bar(x=shap_centers, y=shap_hist["counts"], marker_color="#789A75"))
                if current_feature and shap_edges[0] <= current_shap <= shap_edges[-1]:
                    fig.add_vline(x=current_shap, line_color="#D55E00", line_width=3, annotation_text="Current case")
                fig.update_layout(xaxis_title="SHAP contribution (log-odds scale)", yaxis_title="Held-out cases", margin={"t": 20})
                st.plotly_chart(fig, width="stretch")
            else:
                shap_frame = pd.DataFrame({"SHAP contribution": shap_centers, "Held-out cases": shap_hist["counts"]})
                st.bar_chart(shap_frame.set_index("SHAP contribution"), horizontal=False, x_label="SHAP contribution (log-odds scale)")
            st.caption(f"The horizontal scale is in log-odds. Mean absolute contribution across the reference cases: {profile['mean_absolute_shap']:.3f}.")
            if current_feature:
                direction = "pushes the score upward" if current_shap > 0 else "pushes the score downward" if current_shap < 0 else "has no net directional contribution"
                st.write(f"For this case, this feature's SHAP value is **{current_shap:+.3f}**, which {direction} relative to the model baseline.")
                if current_shap < shap_edges[0] or current_shap > shap_edges[-1]:
                    st.caption(f"This case's SHAP value is around the {shap_percentile:.0f}th percentile of the held-out SHAP reference values, outside the plotted range.")


with results_tab:
    st.subheader("Model evaluation results")
    st.write("All reported model comparisons come from the notebook's shared stratified held-out test set. The deployed prediction model is the unweighted XGBoost pipeline.")
    result_metrics = json.loads((ROOT / "results" / "model_metrics.json").read_text())
    metric_table = pd.DataFrame(result_metrics["models"]).set_index("name").reindex(["XGBoost", "Logistic Regression", "Random Forest"])
    metric_table.index.name = "Model"
    metric_style = metric_table.rename(columns={"roc_auc": "ROC-AUC", "average_precision": "Average Precision", "brier_score": "Brier Score"}).style.format("{:.4f}")
    metric_style = metric_style.apply(lambda row: ["font-weight: bold" if row.name == "XGBoost" else "" for _ in row], axis=1)
    st.dataframe(metric_style, width="stretch")

    expl_col, avg_col, brier_col = st.columns(3)
    with expl_col:
        st.caption("**ROC-AUC:** How well the model ranks serious-delinquency cases above other cases across thresholds. 0.5 is roughly random ranking; higher is better.")
    with avg_col:
        st.caption("**Average Precision:** Summarizes precision across recall levels. It is useful with a rare positive outcome; the test-set event rate is about 6.7%.")
    with brier_col:
        st.caption("**Brier score:** Average squared difference between predicted probabilities and outcomes. Lower indicates probabilities closer to observed outcomes.")

    st.markdown("### ROC curves")
    roc_image = ROOT / "assets" / "roc_curve.png"
    if roc_image.exists():
        st.image(str(roc_image), caption="Notebook ROC comparison on the shared held-out test set")
    else:
        st.info("ROC comparison figure is not available. Run the notebook's dashboard-artifact export section.")
    st.caption("The ROC curve compares true-positive rate with false-positive rate as the classification threshold changes. With an imbalanced target, read it alongside Average Precision.")

    st.markdown("### Precision–Recall curves")
    pr_image = ROOT / "assets" / "pr_curve.png"
    if pr_image.exists():
        st.image(str(pr_image), caption="Notebook precision–recall comparison on the shared held-out test set")
    else:
        st.info("Precision–Recall comparison figure is not available. Run the notebook's dashboard-artifact export section.")
    st.caption("The curve shows precision among flagged cases as recall increases. Average Precision summarizes the curve; the dashed line is the positive-case rate, a useful baseline when serious delinquency is uncommon.")

    st.markdown("### XGBoost recall across thresholds")
    threshold_path = ROOT / "results" / "xgboost_threshold_metrics.json"
    if threshold_path.exists():
        threshold_data = pd.DataFrame(json.loads(threshold_path.read_text())["rows"])
        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=threshold_data["threshold"], y=threshold_data["recall"], mode="lines+markers", name="Recall"))
            fig.add_trace(go.Scatter(x=threshold_data["threshold"], y=threshold_data["precision"], mode="lines+markers", name="Precision"))
            fig.update_layout(xaxis_title="Classification threshold", yaxis_title="Score", yaxis_range=[0, 1], margin={"t": 20})
            st.plotly_chart(fig, width="stretch")
        else:
            st.line_chart(threshold_data.set_index("threshold")[["recall", "precision"]], x_label="Classification threshold", y_label="Score")
        st.dataframe(threshold_data[["threshold", "recall", "precision"]].rename(columns={"threshold": "Threshold", "recall": "Recall", "precision": "Precision"}).style.format("{:.3f}"), hide_index=True, width="stretch")
    else:
        st.info("Threshold recall values are not available. Run the notebook's dashboard-artifact export section.")
    st.caption("Lowering the threshold usually captures more serious-delinquency cases (higher recall) while also flagging more non-events (lower precision). Changing a threshold changes the operating point, not the model's underlying ranking.")

    st.markdown("### XGBoost confusion matrix")
    confusion_path = ROOT / "results" / "xgboost_confusion_matrix.json"
    if confusion_path.exists():
        confusion = json.loads(confusion_path.read_text())
        matrix = pd.DataFrame(confusion["matrix"], index=["Actual: no serious delinquency", "Actual: serious delinquency"], columns=["Predicted: no serious delinquency", "Predicted: serious delinquency"])
        st.write(f"At threshold **{confusion['threshold']:.2f}** on the held-out test set:")
        st.dataframe(matrix, width="stretch")
        st.caption("True positives are serious-delinquency cases correctly identified; false negatives are missed cases. False positives are flagged cases that did not experience serious delinquency; true negatives are correctly unflagged cases.")
    else:
        st.info("The XGBoost confusion matrix is not available. Run the notebook's dashboard-artifact export section.")

    calibration_path = ROOT / "results" / "calibration_curves.json"
    if calibration_path.exists():
        st.markdown("### Calibration")
        st.caption("Calibration compares predicted probabilities with observed event rates within groups. A well-calibrated model follows the diagonal; the Brier score summarizes probability error.")
        calibration = json.loads(calibration_path.read_text())
        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line={"dash": "dash", "color": "gray"}))
            for model_name, values in calibration.items():
                fig.add_trace(go.Scatter(x=values["mean_predicted"], y=values["observed_frequency"], mode="lines+markers", name=model_name))
            fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed frequency", margin={"t": 20})
            st.plotly_chart(fig, width="stretch")
        else:
            rows = [{"Model": name, "Mean predicted probability": x, "Observed frequency": y} for name, values in calibration.items() for x, y in zip(values["mean_predicted"], values["observed_frequency"])]
            st.line_chart(pd.DataFrame(rows), x="Mean predicted probability", y="Observed frequency", color="Model")


with about_tab:
    st.subheader("Project overview")
    st.write("This project uses the Give Me Some Credit dataset to estimate serious-delinquency risk within two years and present information that can support a lender's review. It compares Logistic Regression, Random Forest, and XGBoost; unweighted XGBoost is the selected model.")
    st.write("The probability bands and model explanations are analytical aids. A lender should apply its own economics, policy, fairness review, and risk appetite before making any lending decision.")
    st.subheader("How to use the dashboard")
    st.markdown(
        "1. Enter an applicant profile in **Risk assessment** and select **Estimate Risk**.\n"
        "2. Review the estimated probability and its low, medium, or high band.\n"
        "3. Use **Data dictionary** to check variable definitions and preprocessing.\n"
        "4. Use **Variable insights** to compare one input with the reference distribution and inspect its SHAP contribution.\n"
        "5. Use **Model results** to review discrimination, thresholds, errors, and calibration."
    )
    st.subheader("Project Author Details")
    st.markdown(
        f"**Aseem Deshpande**  \n"
        f"Email: [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL})  \n"
        f"GitHub profile: [github.com/aseem-d]({GITHUB_PROFILE_URL})  \n"
        f"Project repository: [Credit Risk Prediction]({GITHUB_URL})  \n"
        f"LinkedIn: [linkedin.com/in/aseem-deshpande]({LINKEDIN_URL})"
    )
