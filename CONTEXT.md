# CONTEXT.md — Telco Customer Churn Prediction (BMDS2003, Group 6)

> ⚠️ **REMINDER — READ BEFORE SUBMISSION:** Remove all Claude-related files before
> submitting the ZIP / pushing the final version. That means: this `CONTEXT.md` file,
> the `.claude/` directory (contains local session settings), and any other AI-tool
> artifacts. Markers/tutors should not see AI assistant scaffolding in the submission.
> Keep this file in the working repo for the team's own reference, but strip it (and
> `.claude/`) out of the final submission ZIP.

This file exists so every group member has the same context Claude built up across a
multi-session working history on this project. It captures decisions made, why they
were made, bugs found and fixed, and what is still outstanding. Read this before you
touch the code or the report so you don't redo work or reintroduce fixed bugs.

---

## 1. Assignment essentials

- **Course:** BMDS2003 Data Science, TAR UMT, Group 6 (Tutorial Group 6).
- **Members:** Stephanie Lo Qian Fui (2612532), Chan Yi Herng (2612476), Chia Zhen Yang
  (2612480), Ham Guan Quan (2612490). Tutor: Ms. Yan Yen Wei.
- **Dataset:** IBM Telco Customer Churn (Kaggle), 7,043 customers, 21 original
  attributes. Already in repo at `data/Telco_Cusomer_Churn.csv`.
- **Deliverables:**
  1. Written report (Google Docs format) — currently being edited as a local `.docx`
     at `C:\Users\Alex\Downloads\DS Assignment.docx` (**not** in this repo — see §6).
  2. Code ZIP named `GroupX_RSWY1S2_DataScienceProject.zip` containing all `.py`/
     `.ipynb` files and a working deployment prototype (Streamlit — compulsory).
- **Rubric weight (of 100)** — verified against the official
  `202605 BMDS2003 Marking Rubrics.pdf` (2026-08-07): Model Selection 25 ·
  Data Preprocessing 10 · Descriptive/Exploratory Data Analysis **15** ·
  Graphing & Visualisation 10 · **Advanced Analytics & Discussion 30** ·
  Report Structure 5 · Presentation 5.
  (Correction: two earlier reads of the rubric had EDA and Advanced Analytics
  swapped — EDA was wrongly recorded as 30 and Advanced Analytics as 10. In
  reality **Advanced Analytics & Discussion (30) is the single biggest line
  item**, not EDA (15). This means the report's discussion should weight
  literature-grounded model comparison, honest limitations/improvements
  discussion, and the prototype's real-world-deployment framing more heavily
  than further EDA breadth — the current 19-analysis EDA section already
  exceeds what its 15-mark weight calls for.)
- **Spec requirements satisfied:** 4 models (4-member group requires 4, one must be
  baseline), ≥3 required — done. Model choice justified with ≥2 literature references
  per model, in-text cited — done (report Table 5). Compulsory deployment prototype —
  done (Streamlit, all 4 models selectable).

---

## 2. Repository structure (this repo)

```
Asgm_DS/
├── CONTEXT.md                  # this file — REMOVE before submission
├── README.md                   # project overview, setup instructions
├── requirements.txt             # full dependency list (root)
├── .gitignore                   # ignores *.pkl, __pycache__, etc.
├── data/
│   └── Telco_Cusomer_Churn.csv  # raw dataset
├── src/                         # analysis package (single source of truth)
│   ├── data_prep.py             # cleaning, feature engineering, encoding
│   ├── eda.py                   # EDA tables + statistical tests (chi-sq, point-biserial, VIF)
│   ├── eda_plots.py             # EDA figures (matplotlib), report Figures 1-6b
│   ├── modelling.py             # train/test split, scaling, SMOTE, model training
│   └── evaluation.py            # metrics, plots, feature importance, threshold tuning
├── notebooks/
│   └── analysis.ipynb           # full analysis notebook, calls src/ modules (Sections 2-5)
├── reports/figures/             # exported PNGs for the report (19 figures)
└── streamlit/                   # deployment prototype
    ├── app.py                   # the Streamlit app (see §5)
    ├── plotly_charts.py         # interactive Plotly chart layer, app-only (see §4)
    ├── train_model.py           # trains + saves all 4 models for the app
    └── requirement.txt          # prototype-only dependencies
```

**Not in this repo** (lives on the maintainer's machine, in `Downloads/`):
`DS Assignment.docx` (the report) and a Chinese explainer HTML
(`Telco_Churn_报告讲解_Valorant.html`). If you need these, ask whoever has been running
the Claude session, or work from the notebook/report structure described below to
reconstruct sections.

---

## 3. Locked decisions (do not relitigate without discussion)

- **Models:** Logistic Regression (baseline) / Decision Tree / Random Forest / XGBoost.
  Deliberate progression: linear → single tree → bagging → boosting. Not changed to
  SVM/KNN/etc. — the four chosen are defensible and already justified with literature.
- **Recommended default model:** Random Forest (best Accuracy/Precision/F1/AUC).
  Decision Tree has the highest Recall (77%) — kept as the interpretable/high-recall
  alternative. All 4 are deployed and selectable in the prototype, not just the winner.
- **Class imbalance:** SMOTE on the training set only, after the train/test split.
  Deliberately NOT combined with `class_weight='balanced'` (would double-correct).
- **Feature set:** 23 predictors after a 4-round multicollinearity (VIF) audit
  (worst-case VIF 2,346 → ~20). See `src/data_prep.py:build_feature_matrix` for the
  exact drop list and reasoning in the code comments.
- **Feature engineering:** 8 features engineered, audited honestly — 2 kept for
  modelling (`ContractRiskScore`, `ChargesToTenureRatio`), 3 kept for description only
  (`TenureGroup`, `TotalServicesSubscribed`, `IsAutoPay`), 3 rejected with evidence
  (`AvgChargePerMonth`, `HasInternetService`, `HasPartnerOrDependents`).

---

## 4. Session history — what was done, in order

### P0 — Verification
Confirmed the pre-existing codebase (inherited at session start, already ~80% built)
reproduces the report's numbers exactly on current library versions (pandas 3.0,
numpy 2.4, sklearn 1.9). Installed missing deps (`xgboost`, `imbalanced-learn`,
`statsmodels`).

### P1 — Bug fixes
- `streamlit/app.py` had a broken import (`from codebase.eda import ...` — neither
  the module nor the function existed there). Fixed.
- `streamlit/train_model.py` used to train **only Random Forest**. Changed to train
  and save **all four models** into a single `models.pkl` dict, so the prototype can
  expose all of them (user explicitly asked "why not them all?" instead of one).
- Added root `requirements.txt`; added missing `xgboost`/`matplotlib` to
  `streamlit/requirement.txt`.

### P2 — Repository restructure + EDA visualisations
- Reorganised a flat, messy root into `data/ src/ notebooks/ reports/figures/
  streamlit/` (git mv, history preserved). Made CSV path resolution `__file__`-relative
  so it works regardless of caller's working directory.
- Added `src/eda_plots.py` — generates the report's Figures 1–5 plus supporting charts
  (categorical small-multiples, numeric distributions/boxplots by churn, correlation
  heatmap). Wired into the notebook, exported to `reports/figures/`.

### P3 — Statistical rigor (CLO3)
- Added `eda.chi_square_tests()` (+ Cramér's V effect size) and
  `eda.point_biserial_tests()` to `src/eda.py`. Added `eda_plots.plot_cramers_v()`
  (Figure 6). Wired into notebook as report Section 2.8. This directly targets the
  CLO3 "statistical methods" rubric criterion.
- Analysis inventory: **19 total analyses** across Sections 2–3 (well above the ~15
  target / ~4 per member). Full table of what's where is in the report §2–3 and
  mirrored in `src/eda.py`/`src/eda_plots.py` docstrings (each function cites its
  report section number).

### P4 — Report finalisation (the `.docx`, edited via python-docx)
- **Numbers reconciled**: report's model metrics had drifted slightly from a
  reproducible `tune=True` run (sklearn version differences). Fixed §4.2 XGBoost
  params (`min_child_weight` 5→10 to match the report's GridSearchCV result), updated
  all metric tables (§5.2, §5.3, §5.6, §5.7, §5.8) to the reconciled numbers.
- Filled every `[PLACEHOLDER]` in the draft (Executive Summary closing, §5.9 final
  model recommendation, §6.1 prototype description, §6.3 screenshot section).
- Fixed §6 contradiction: report said "trained XGBoost model" while code deployed
  Random Forest — rewritten to describe the actual multi-model prototype.
- **Content audit against the codebase** (user explicitly asked: make sure report
  content matches the codebase, rewrite what doesn't):
  - §2.7's second interaction paragraph was **factually wrong** — claimed "longer-
    tenured customers with more services churn less," but the actual data shows the
    opposite: *within every tenure band, churn rises with service count* (the
    original "more services = less churn" pattern is a Simpson's-paradox artefact of
    tenure, not a real service-count effect). Rewrote the paragraph, added
    `eda_plots.plot_tenure_service_interaction()`, added it to the notebook, added
    **Figure 5b** to the report with correct interpretation (first-year customers
    with 4–5 services churn at 56.9%, the peak).
  - §6.4 tech table claimed "Matplotlib/Seaborn" for the prototype's visualisation —
    wrong, the app uses Streamlit native widgets only. Fixed.
  - §6.2 claimed "SHAP values" as an option — the app never implemented SHAP (uses
    `feature_importances_`/coefficients). Removed the claim.
- Report now has **19 embedded images** (13 original modelling figures the group had
  already inserted + 5 supporting EDA charts + Figure 5b added this session), 15
  tables, a Member Contributions table, and a Statistical Association Tests section
  (§2.8) with 2 new tables (chi-square/Cramér's V, point-biserial).
- **Still outstanding for the group:** §6.3 has an italic placeholder asking for 2
  actual screenshots of the running app (a high-risk and a low-risk customer
  prediction) — these need to be taken manually and pasted in. Automated screenshot
  capture failed in this session due to an in-app browser tooling limitation, not an
  app bug.

### Streamlit app — 3-tab rebuild + critical bug fix

The prototype (`streamlit/app.py`) was rebuilt from a single predict-only page into
three tabs:

- **🔮 Predict** — customer profile form, churn probability + risk badge/gauge, top
  contributing features, all-4-models comparison table.
- **📊 Data Analysis** — 7 sub-tabs covering **every one of the 19 report analyses**,
  computed **live** by importing `src/data_prep.py`, `eda.py`, `eda_plots.py` directly
  (not reimplemented, not static screenshots — same source of truth as the notebook
  and report).
- **🤖 Models** — 8 sub-tabs covering the full §4–5 evaluation suite live: performance
  table, ROC/PR curves, confusion matrices, feature importance + LR odds ratios,
  decision tree structure, overfitting check, threshold tuning (with model picker),
  5-fold cross-validation.

**🔴 Critical bug found and fixed this session:** `build_feature_matrix()` (in both
`streamlit/train_model.py` and `src/data_prep.py`) used
`pd.get_dummies(..., drop_first=True)` on the categorical columns. On a **single-row
prediction** (the Predict tab always sends exactly one row), only one category is ever
present in that row, so `drop_first` always drops it — producing **zero dummy columns**
for `InternetService` and `PaymentMethod` regardless of which option the user picked.
After the code's `reindex(fill_value=0)` step, every choice silently collapsed to
whatever category was the training-time reference (alphabetically first: DSL /
"Bank transfer (automatic)"). **This meant the Predict tab's Internet Service and
Payment Method dropdowns had never actually affected any prediction** — verified
empirically (before fix: DSL/Fiber/No all gave identical 61.2% for the same customer;
after fix: DSL 61.2% / Fiber optic 80.6% / No 48.4%, matching the report's known
direction). Fixed by declaring `pd.Categorical(col, categories=[...fixed list...])`
before calling `get_dummies` in both files — this makes pandas always generate all
trained dummy columns regardless of how many rows/categories are present in the
current batch. Verified zero regression on full-dataset encoding (still 23 columns,
retrained model numbers identical). **Models were retrained after this fix — if you
pull this repo, run `python streamlit/train_model.py` again before demoing.**

Also fixed/added this session:
- Predict tab moved off `st.form` to live widgets — inside a form, a `disabled=` flag
  derived from another in-form widget only updates after submit, which was why
  add-on services didn't visually grey out immediately when Internet Service was set
  to "No". Now: add-ons stay live-editable, and an amber warning banner explains when
  the app is coercing the input to match the dataset's encoding (e.g. Internet
  Service = No forces all six add-ons to "No internet service" for the model, exactly
  as all 1,526 real such customers are encoded, zero exceptions).
- Added a "how these inputs relate" expander answering exactly the questions a
  teammate is likely to ask: Contract is the customer's *current* contract, not a
  lifetime commitment (any tenure+contract combination is valid — verified 1,144 real
  customers are Month-to-month with tenure > 24 months); Total Charges vs
  tenure×MonthlyCharges (median deviation 0.0%, 94.6% of real customers within ±10%,
  deviations are genuine price changes over time, not errors).
- Added a live "how common is this combination" callout using real dataset stats for
  the selected tenure-band × contract cell.
- Added sidebar "demo presets" (Highest-risk segment / Lowest-risk segment / Dataset
  average) that one-click populate the whole form via `session_state` + `st.rerun()`.
- Added engineered-feature preview (ContractRiskScore, ChargesToTenureRatio) and a
  model-agreement-spread metric (how much the 4 models disagree on one customer).

Note: a glassmorphism CSS redesign of the app was applied locally (outside this
Claude session) at some point during the work — it was preserved as-is; new features
were added around it via targeted edits, not a full file rewrite.

### Plotly interactive charts + friendly error handling (later session)

`streamlit/plotly_charts.py` (new file, app-only — `src/eda_plots.py` and
`src/evaluation.py` stay matplotlib, since they feed the notebook and the report's
already-graded static PNG exports; nothing there was touched). It reimplements the
rendering layer only, reusing the exact same underlying tables/computations
(`eda.py`, `evaluation.py`, plain sklearn calls against the deployed models) so all
numbers match the matplotlib originals exactly — only hover/zoom/legend interactivity
changed. **18 of 19 report charts are now Plotly** (`st.plotly_chart`); only the
**Decision Tree structure** stays matplotlib (`sklearn.tree.plot_tree` has no
reasonable Plotly equivalent). Two bugs found and fixed during this conversion,
both worth knowing about if you touch `plotly_charts.py` again:

1. **Title/legend overlap** — an internal Plotly `title=` was set on every chart on
   top of an ALREADY-present markdown header in `app.py` (e.g. "**Figure 6 — ROC
   curves**" printed just above the chart) — the two collided visually because
   Plotly's default title position and the horizontal legend (`y=1.02`) both sit in
   the same top-left band. Fixed by dropping the internal Plotly title entirely
   (`_base_layout()` no longer accepts `title=`) and reserving explicit top margin
   (`margin.t`) for the legend instead of letting it float above the plot.
2. **Negative-bar label collision** — `plot_correlation_with_target_plotly` (Figure 4)
   put its value labels `textposition="outside"` unconditionally; for a *negative*
   horizontal bar, "outside" renders further left than the bar tip, and for the
   longest negative bar (`tenure`, r=-0.352) that collided with the y-axis category
   label itself, rendering as garbled text ("tenure).352"). Fixed by putting negative
   bars' labels `inside` (white text) and positive bars' labels `outside` — general
   lesson: **never hardcode `textposition="outside"` on a horizontal bar chart that
   can have negative values**, pick per-bar based on sign.

Verification method used since this environment's browser pane cannot reliably
screenshot: installed `kaleido` (`pip install kaleido`, dev-only, NOT added to
`requirements.txt` since the app itself never calls `fig.write_image`) and rendered
each new chart to a static PNG (`fig.write_image(...)`) to visually inspect layout
bugs directly, rather than trying to screenshot the live app.

Also added: `load_artifacts()` (in `app.py`) now wraps its three `.pkl` reads in
try/except — a missing-file or version-mismatch error shows a friendly `st.error()`
with the exact fix (`python train_model.py`) and calls `st.stop()`, instead of a raw
Python traceback. This was deliberately scoped to just this one function (the app's
only external-file dependency) rather than added everywhere — a full test suite is
overkill for a one-off assignment demo, but a raw traceback during the graded
Presentation (5% of the mark) is a real, cheap-to-prevent risk.

Two things were explicitly investigated and found to be **inherent, not bugs**:
transition animation on chart re-render only has something to animate on the
**Threshold tab slider** (the only place a chart's underlying data actually changes
after first load — it's wrapped in `st.fragment` so only that section reruns) and the
**Predict tab's risk gauge/badge** (CSS `transition` on `width`/`background`, changes
per prediction). The Model Comparison / ROC / PR / VIF / Cramér's V / heatmap charts
show the same all-models or all-features data on every render, so there is nothing
for Plotly's `transition` to animate between — this is a data-shape fact, not a
missed implementation.

---

## 5. How to run things

```bash
# Install everything
pip install -r requirements.txt

# Run the full analysis notebook (regenerates all tables/figures)
jupyter notebook notebooks/analysis.ipynb

# Run the Streamlit prototype
cd streamlit
pip install -r requirement.txt
python train_model.py       # trains all 4 models, writes models.pkl/scaler.pkl/feature_columns.pkl
streamlit run app.py        # opens at localhost:8501
```

`models.pkl` (~13MB), `scaler.pkl`, `feature_columns.pkl` are **committed to the repo**
(`streamlit/*.pkl`, explicit exception in `.gitignore`) so teammates can run the app
immediately without retraining. If you change anything in `src/data_prep.py`,
`streamlit/train_model.py`, or the feature engineering, **re-run `train_model.py` and
commit the refreshed `.pkl` files** — otherwise the app will silently keep using stale
models trained on the old code.

---

## 6. Outstanding work / TODO for the team

1. **§6.3 screenshots** — take 2 real screenshots of the running app (high-risk and
   low-risk customer predictions, per the italic instruction already in that section
   of the report) and paste them in.
2. **P5 — package the submission ZIP** — not yet done as of this file being written.
   Needs: all `.py`/`.ipynb` files, the `streamlit/` prototype, correctly named per
   spec (`GroupX_RSWY1S2_DataScienceProject.zip`). **Remember to strip `.claude/` and
   this `CONTEXT.md` out of that ZIP** (see reminder at top of this file).
3. **Report → Google Docs** — the report is currently a local `.docx`. Spec requires
   Google Docs format for submission; someone needs to upload/paste it into Google
   Docs and re-check formatting survived the conversion (especially the 15 tables and
   19 images).
4. **Plagiarism statement form / signatures** — still has blank signature/date fields
   in the report's front matter (Section: Plagiarism Statement Form) — each member
   needs to fill in their own signature and date before submission.
5. If anyone re-runs `train_model.py`, the model numbers will be near-identical but
   not bit-identical to what's quoted in the report (library-version-dependent
   GridSearchCV micro-differences) — this is expected and was already reconciled once;
   don't chase sub-0.5pp differences again unless something looks structurally wrong.

---

## 7. Key gotchas for anyone continuing this work

- **pandas 3.0** stores string columns as `StringDtype`, not `object` — code that does
  `df[col].dtype == object` to detect string columns will silently misbehave. Use
  `pd.api.types.is_numeric_dtype()` instead (see `src/eda.py:_churn_binary` and
  `src/eda_plots.py:_churn_binary` for the pattern already in use).
- **matplotlib 3.11** removed `plt.cm.get_cmap()` — use `plt.get_cmap()`.
- **One-hot encoding on small batches** — see the critical bug in §4 above. Any new
  code that does `pd.get_dummies()` on fewer rows than there are categories needs
  `pd.Categorical(col, categories=[...])` declared first, or it will silently drop
  columns.
- **python-docx + `add_table`** — if you get a `ValueError: invalid literal for int()`
  on a fractional twips value when adding tables to the report `.docx`, the document's
  page margins are stored as fractional twips (a Google-Docs-export quirk). Round them
  first (search this session's history / the docx-editing scripts for
  `fix_fractional_margins` if you need the helper again).
- **Streamlit + this repo's `streamlit/` folder name** — if you `import streamlit` from
  the repo root (not from inside `streamlit/`), Python may resolve it to the project's
  own `streamlit/` directory (a namespace package) instead of the real installed
  package, especially if the real package isn't installed. Always launch with
  `cd streamlit && streamlit run app.py`, and if `import streamlit` ever behaves
  strangely, check `streamlit.__file__` — `None` means you've hit the namespace-package
  shadow, not a real import.
- **Plotly horizontal bar charts with negative values** — never hardcode
  `textposition="outside"`; for negative bars the label renders past the bar tip
  (further left), which can collide with y-axis category labels for the longest bar.
  Pick per-bar: `"inside"` for negative, `"outside"` for positive (see
  `plotly_charts.py:plot_correlation_with_target_plotly`).
- **Verifying Plotly chart layout without a working screenshot tool** — `pip install
  kaleido` then `fig.write_image("out.png", width=..., height=..., scale=2)` and read
  the PNG directly. Much more reliable in this environment than trying to screenshot
  the live Streamlit app (the in-app browser pane here cannot reliably composite/
  screenshot). `kaleido` is dev-only — don't add it to `requirements.txt`, the app
  itself never calls `write_image`.
