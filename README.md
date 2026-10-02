# Credit Risk Prediction

**Live dashboard:** [Open the Streamlit app](https://credit-risk-prediction-aseemd.streamlit.app)

## Overview

A lender-facing analytics project using the Give Me Some Credit dataset to estimate serious-delinquency risk within the next two years. It includes the full analysis notebook and a Streamlit dashboard backed by the notebook's selected unweighted XGBoost pipeline. The dashboard presents risk bands and model evidence to support review; it does not issue an approval or rejection decision.

## Objective

Estimate the probability of serious delinquency within two years and provide model context for a lender's review. The application does not make approval or rejection decisions; lenders must weigh their own product economics and risk appetite.

## Dataset

The Kaggle Give Me Some Credit dataset contains borrower characteristics and a binary serious delinquency target. The target indicates whether a borrower experienced serious delinquency (90 or more days past due) within two years. Data files are not included; see [data/README.md](data/README.md).

## Methodology

The analysis investigates missing values, suspicious 96/98 delinquency values, anomalous DebtRatio and MonthlyIncome patterns, skewness, outliers, and multicollinearity (VIF). It compares preprocessing choices and multiple Logistic Regression specifications, including class weighting and a severity-weighted delinquency feature, then evaluates Random Forest and XGBoost. XGBoost uses median imputation for numeric features, most-frequent imputation for dependents, and the notebook's missing-income and suspicious-value indicators. It is compared using a shared stratified train/test split. ROC-AUC, Average Precision, Brier score and calibration curves are reported; SHAP is used to interpret the selected tree model.

Class weighting did not improve XGBoost discrimination and slightly reduced Average Precision. The hyperparameter-tuning experiment did not materially improve the initial XGBoost model. The unweighted XGBoost remains the selected model.

## Results

Held-out test-set results exported from the analysis notebook:

| Model | ROC-AUC | Average Precision | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.836128 | 0.348515 | 0.052968 |
| Random Forest | 0.856446 | 0.380701 | 0.049861 |
| XGBoost (selected) | 0.869623 | 0.404935 | 0.048735 |

At the default 0.5 threshold, the selected XGBoost achieved precision 0.5942, recall 0.2045, and F1 0.3043. The project focuses on probability estimation; these results are not an operational lending threshold.

## Dashboard

Install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `models/final_xgb_model.joblib` and reads evaluation summaries from `results/`. It does not train or reevaluate models at startup. The dashboard includes applicant risk assessment, a data dictionary, per-variable distributions with SHAP explanations, model performance, recall across thresholds, a confusion matrix, and calibration. Its low, medium, and high probability bands organize results but do not determine lending decisions. The About section links to the project repository, Aseem Deshpande's GitHub profile, and LinkedIn.

## Reproducing the notebook

Place `cs-training.csv` in the repository root (or adjust the notebook's input path), install the notebook dependencies listed in `requirements-notebook.txt`, and run `notebooks/credit_risk_analysis.ipynb` from the repository root. Its final export section writes the model and evaluation artifacts. The original notebook is preserved byte-for-byte at `notebooks/credit_risk_analysis_original.ipynb`.

## Repository Structure

```text
app.py                         Streamlit dashboard
src/model.py                   Model loading and raw-input feature construction
models/final_xgb_model.joblib  Selected fitted XGBoost pipeline
results/                       Notebook-exported metrics and calibration points
notebooks/                     Cleaned analysis and untouched original
data/                          Dataset retrieval instructions
```

## Limitations

- The analysis uses one historical dataset and has no temporal or external validation.
- The source data contains suspicious delinquency values and other data-quality anomalies.
- Predictive associations are not causal explanations.
- No fairness analysis has been performed.
- This is not a lending approval system and does not implement lender-specific cost-sensitive decisions.
- The model's probabilities and performance may not generalize to current populations or other institutions.

## Future Scope

Temporal and external validation, further probability calibration, fairness analysis, deployment monitoring, lender-specific cost-sensitive thresholds, and exposure/LGD modeling are possible extensions.
