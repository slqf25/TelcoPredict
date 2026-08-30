# Telco Customer Churn Prediction

A CRISP-DM based machine learning project predicting customer churn for a telecommunications provider, using the IBM Telco Customer Churn dataset.

Four classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost) are trained and compared on a redundancy-checked,
23-predictor feature set, with a Streamlit prototype for real-time churn risk scoring. 
Full methodology, findings, and discussion are in theaccompanying project report.

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 75.23% | 52.44% | 71.93% | 60.65% | 0.833 |
| Decision Tree | 73.74% | 50.35% | **77.01%** | 60.89% | 0.823 |
| **Random Forest** | **75.87%** | **53.31%** | **73.26%** | **61.71%** | **0.839** |
| XGBoost | 75.23% | 52.40% | 72.99% | 61.01% | 0.830 |

Two engineered features — `ContractRiskScore` and `ChargesToTenureRatio` — outrank every raw column in feature importance for both Random Forest and XGBoost. 
See the report for the full four-round multicollinearity audit behind the final feature set, and the eight-feature engineering audit
(five features tested, three rejected with evidence).

## Repository Structure

```
telco-customer-churn/
├── README.md
├── requirements.txt             # All Python dependencies
├── data/
│   └── Telco_Cusomer_Churn.csv  # Raw dataset (IBM Sample Data Sets, via Kaggle)
├── src/                         # Analysis package
│   ├── data_prep.py             # Cleaning, feature engineering, encoding
│   ├── eda.py                   # EDA tables, outlier/VIF checks, feature audit
│   ├── eda_plots.py             # EDA figures (report Figures 1–5 + supporting charts)
│   ├── modelling.py             # Train/test split, scaling, SMOTE, model training
│   └── evaluation.py            # Metrics, plots, feature importance, threshold tuning
├── notebooks/
│   └── analysis.ipynb           # Full analysis calling the modules above (Sections 2–5)
│
└── streamlit/                   # Deployment prototype (all four models)
    ├── app.py                   # Streamlit churn predictor — page flow + data wiring
    ├── train_model.py           # Trains all four models, saves models.pkl/scaler/columns
    ├── ui_style.py              # Colours, icon maps, CSS, HTML/markdown helpers
    ├── plotly_charts.py         # Interactive Plotly chart layer
    ├── model_interactions.py    # Threshold tuning, Sankey and comparison visuals
    ├── prediction_visuals.py    # Single-customer Predict result visuals
    ├── tower_3d.py              # Interactive 3D customer-profile visual
    └── requirement.txt          # Prototype-only dependencies
```

## Setup

```bash
git clone https://github.com/slqf25/TelcoPredict.git
cd TelcoPredict
pip install -r requirements.txt
```

## Running the Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

Runs the full pipeline — data cleaning, feature engineering, EDA, the four-round multicollinearity audit, model training with GridSearchCV, 
and all evaluation plots — using the functions in `data_prep.py`, `eda.py`, `eda_plots.py`, `modelling.py`, and `evaluation.py`.

## Running the Prototype

```bash
cd streamlit
pip install -r requirement.txt
python train_model.py      # trains all four models, saves models.pkl / scaler.pkl / feature_columns.pkl
streamlit run app.py       # launches the web app at localhost:8501
```

Enter a customer's profile (contract type, tenure, charges, services, etc.) to get a churn probability and risk category (Low / Medium / High). 
Use the model selector to switch between all four trained models and compare their predictions for the same customer.

## Four-Member Streamlit Presentation Guide

The Streamlit prototype is designed to serve as the presentation interface, so a separate slide deck is not required. The recommended narrative is:

> Business problem -> data evidence -> four individual models -> fair model
> comparison -> customer-level prediction -> retention decision.

For a 30-minute session including questions, aim for approximately 24-25
minutes of presentation and reserve the remaining 5-6 minutes for Q&A.

| Section | Presenter | Suggested time |
|---|---|---:|
| Problem, dataset, proposed solution | Member 1 | 3-4 min |
| Logistic Regression | Member 1 | 4 min |
| Decision Tree | Member 2 | 4 min |
| Random Forest | Member 3 | 4 min |
| XGBoost and overall comparison | Member 4 | 5-6 min |
| Customer prediction demo | Group / one operator | 3 min |

### Shared opening: problem and solution

Start in **Data Analysis -> Executive Overview** and establish the business problem:

- The dataset contains 7,043 customers and 21 original attributes.
- The observed churn rate is 26.54%.
- Churn causes recurring-revenue loss and additional customer-acquisition costs.
- The proposed solution is an early-warning classifier that converts an existing customer profile into a churn probability and risk category.

The target is imbalanced. A classifier predicting `No Churn` for every customer would obtain about 73.46% accuracy but zero churn recall. 
Therefore, the presentation should not treat Accuracy as the main success criterion.
Discuss Precision, Recall, F1 and AUC, with particular attention to Recall and F1 because failing to identify a real churner has a direct business cost.

Next, use **Data Analysis -> Patterns & Segments** to show only the most actionable patterns rather than every available graph:

- Month-to-month contract: 42.7% churn, versus 2.8% for a two-year contract.
- Electronic check: 45.3% churn, versus approximately 15-17% for automatic payment methods.
- Fiber optic: 41.9% churn, versus 19.0% for DSL.
- First year: 47.4% churn, showing the importance of early retention.

Then open **Data Analysis -> Feature Decisions**. 
Explain that `ChargesToTenureRatio` (r = 0.412) and `ContractRiskScore` (r = 0.397) were retained 
because both were more strongly correlated with churn than any raw numeric feature. 
They later ranked above the original variables in the tree-based models' feature importance.

### Individual model presentation pattern

Before each member speaks, use the model cards at the top of the application to select that member's model. The selection is shared across **Predict** 
and all model-specific analysis views. Each member should answer the same five questions:

1. Why was this model selected?
2. How does it make a classification?
3. How was its complexity configured or controlled?
4. What do its test results mean in business terms?
5. What is its main advantage and limitation?

#### Member 1 - Logistic Regression baseline

Navigation: select **Logistic Regression**, then use **Models -> Performance**
and **Models -> Explainability -> Selected Model Detail**.

- Role: interpretable baseline from a different model family.
- Configuration: `max_iter=2000`; numerical variables are standardised.
- Results: Accuracy 75.23%, Precision 52.44%, Recall 71.93%, F1 60.65%, AUC 0.833.
- Business meaning: it identifies about 72 of every 100 actual churners.
- Strength: coefficients provide the direction and relative influence of each predictor.
- Limitation: its additive log-odds structure may miss nonlinear relationships and complex customer interactions.

The coefficient view should be interpreted with the encoding and scaling in mind. 
Avoid comparing a scaled numerical coefficient directly with an unscaled binary coefficient as if their units were identical.

#### Member 2 - Decision Tree

Navigation: select **Decision Tree**, then use **Models -> Performance**, **Models -> Explainability -> Selected Model Detail**, 
and optionally **Models -> Decision Threshold**.

- Role: interpretable nonlinear model expressed as decision rules.
- Configuration: `max_depth=5`, `min_samples_split=10` to limit complexity.
- Results: Accuracy 73.74%, Precision 50.35%, Recall 77.01%, F1 60.89%, AUC 0.823.
- Business meaning: it catches about 77 of every 100 actual churners, the highest Recall among the four models, but raises more false alerts.
- Strength: the tree structure is relatively easy to explain.
- Limitation: a single tree can be unstable and obtained the lowest Accuracy and AUC in this comparison.

Do not narrate the entire tree. Explain only the first few important split and connect them to the earlier churn-pattern analysis.

#### Member 3 - Random Forest

Navigation: select **Random Forest**, then use **Models -> Performance**, **Models -> Explainability -> Selected Model Detail**, 
and **Models -> Reliability -> Generalisation Gap**.

- Role: reduce the variance of a single Decision Tree by combining many trees.
- Tuned configuration: 200 trees, `max_depth=12`, `min_samples_leaf=5`.
- Results: Accuracy 75.87%, Precision 53.31%, Recall 73.26%, F1 61.71%, AUC 0.839.
- Business meaning: it catches about 73 of every 100 actual churners and has the strongest overall balance at the default threshold.
- Strength: best point estimates for Accuracy, Precision, F1 and AUC.
- Limitation: less transparent than Logistic Regression or a single tree and still shows a train-to-test performance gap.

The reliability view should mention the modelling iteration: 
the initial unconstrained Random Forest severely overfit (train F1 0.999 versus test F1 0.584), which motivated regularising GridSearchCV settings. 
The displayed generalisation gap is evidence to discuss, not standalone proof of overfitting,
because the training score uses SMOTE-balanced data while the test score uses the natural held-out distribution.

#### Member 4 - XGBoost and final comparison

Navigation: select **XGBoost**, then use **Models -> Performance** and **Models -> Explainability -> Selected Model Detail**. 
Finish in **Models -> Decision Threshold -> Compare All Models**.

- Role: sequentially add trees that correct errors made by earlier trees.
- Tuned configuration: `max_depth=4`, `learning_rate=0.1`, 150 trees, `min_child_weight=5`, row/column subsampling of 0.8 and L2 regularisation.
- Results: Accuracy 75.30%, Precision 52.53%, Recall 72.19%, F1 60.81%, AUC 0.830.
- Business meaning: it remains competitive with the other strong models but does not lead the held-out test ranking.
- Strength: captures nonlinear relationships and interactions through boosting.
- Limitation: requires careful tuning and is less interpretable than the baseline or a small Decision Tree.

For the final comparison, avoid claiming that one model is best for every purpose:

- Decision Tree has the highest Recall and misses the fewest churners.
- Logistic Regression offers the clearest coefficient-based interpretation.
- Random Forest has the best overall point-estimate balance and is the recommended default.
- XGBoost is a competitive nonlinear benchmark but does not outperform Random Forest on this hold-out set.

McNemar's paired tests found a statistically significant difference only between Decision Tree and Random Forest (p = 0.009). 
Random Forest was not significantly different from Logistic Regression or XGBoost at the 5% level.
The defensible conclusion is therefore that Random Forest leads the observed point estimates, not that it is conclusively superior to every alternative.

### Final prediction demo

Finish in **Predict** with **Random Forest** selected. Use the three **Demo presets** buttons rather than manually entering every field during the presentation:

1. Click **Highest-risk segment** and press **Predict Churn Risk**.
2. Click **Dataset average** and press **Predict Churn Risk**.
3. Click **Lowest-risk segment** and press **Predict Churn Risk**.
4. Explain the churn probability, risk band and model-consensus view for each, going from highest to lowest risk.
5. Connect each result to an action: prioritised retention contact for high risk, personalised support for medium risk (the dataset-average profile), and routine engagement for low risk.

The prediction is decision support, not an automatic decision. A business can adjust the classification threshold according to retention capacity and the relative costs of missed churners and false alerts.

### Presentation operating checklist

- Use one laptop operator throughout; speakers should not exchange control of the mouse between sections.
- Visit all required pages once before presenting so cached model evaluation, cross-validation and VIF calculations are already warm.
- Fix the browser zoom at approximately 80-90% and rehearse the scroll position for each view.
- Show two or three relevant visuals per model, not every available output.
- Use the demo presets for repeatable customer examples.
- Keep screenshots of the key pages as a backup in case the live application cannot be loaded.
- Rehearse the exact sequence of model-card and page selections so every click answers a stated analytical or business question.

## Methodology Summary

- **Data preparation**: logic-based imputation for `TotalCharges` (not statistical), 
  outlier detection via mean ± 3σ, and a four-round Variance Inflation Factor audit that removed exact and redundant features. 
  The deployed 23-predictor matrix still has severe residual multicollinearity (live maximum VIF 369.2 for `MonthlyCharges`), 
  so individual Logistic Regression coefficients are not interpreted as independent effects.  
  The tree ensembles are less sensitive, although importance may be shared across correlated predictors.
- **Feature engineering**: 8 features engineered from the original 21 columns; each checked directly against the data rather than assumeduseful — 2 kept for modelling, 
  3 kept for description only, 3 rejected outright with evidence.
- **Class imbalance**: SMOTE applied to the training set only, after the train-test split, and not combined with algorithmic class weighting (which would double-correct).
- **Models**: Logistic Regression (baseline), Decision Tree, Random Forest, XGBoost — 
  each regularised via GridSearchCV after an initial unconstrained Random Forest was found to overfit severely (train F1 0.999 vs test F1 0.584).
- **Evaluation**: Accuracy/Precision/Recall/F1/AUC, 5-fold stratified cross-validation, Precision-Recall curves, Logistic Regression odds ratios, Decision Tree structure, 
  and classification-threshold tuning.

## Tech Stack

Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, imbalanced-learn, statsmodels, matplotlib, seaborn, Jupyter — 
for the analysis pipeline in `src/` and `notebooks/analysis.ipynb`.

Streamlit, Plotly, and Three.js (WebGL, loaded client-side via a Streamlit custom component) — 
for the interactive prototype in `streamlit/`, including the model comparison charts and the 3D customer-profile visual.

