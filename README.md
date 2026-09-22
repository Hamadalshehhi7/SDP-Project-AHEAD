# ✚ AHEAD — Advanced Health Early Awareness and Disease Detection System

**Senior Design Project** · Python · Streamlit · scikit-learn · XGBoost · Plotly · Gemini

AHEAD is an educational early-awareness screening platform. A user (or a clinician) enters
lifestyle, medical-history and clinical values and receives a machine-learning screening
estimate for **diabetes**, **cardiovascular disease** or **chronic kidney disease** — together
with a gauge, next steps, the factors AHEAD noticed in the inputs and an optional AI-generated
explanation.

> AHEAD does not diagnose disease and must not be used to start, stop or change treatment.

---

## Table of contents

1. [Features](#features)
2. [Project structure](#project-structure)
3. [Setup](#setup)
4. [Running the app](#running-the-app)
5. [Demo accounts](#demo-accounts)
6. [AI features (Gemini)](#ai-features-gemini)
7. [Datasets](#datasets)
8. [Re-training the models](#re-training-the-models)
9. [How the machine learning works](#how-the-machine-learning-works)
10. [Deploying on Streamlit Cloud](#deploying-on-streamlit-cloud)
11. [Common errors](#common-errors)

---

## Features

| Area | What it does |
|---|---|
| **Sign-in** | Patient or Doctor/Admin demo accounts, public About / Contact / Help pages |
| **Overview** | Welcome hero with quick-start, KPI cards, session risk distribution, dataset and model-quality charts |
| **Screenings** | Guided three-section form per condition, six selectable model families (recommended first), gauge, next steps, risk-factor chips, urgent-care notice, AHEAD Insight (Gemini), model interpretation |
| **Data & Analytics** | Dataset explorer, final test metrics, threshold sweep (heart), algorithm comparison table |
| **Clinical Dashboard** *(doctors)* | Upload a CSV/XLSX of patient records, score the whole cohort, download results, review a single record |
| **AI Assistant** | "Ask AHEAD" chat with suggested prompts; answers via Gemini (offline answers for the suggestions) |
| **Settings** | Profile, Light/Dark appearance, notification & privacy preferences, account actions |

---

## Project structure

```
SDP-Project-AHEAD/
├── app.py                  ← entry point: page config, routing, sidebar
├── ahead/
│   ├── config.py           ← disease registry, form layout, labels, navigation, demo accounts
│   ├── resources.py        ← cached models / metadata / datasets / images / Gemini client
│   ├── theme.py            ← colour tokens (light & dark), global CSS, Plotly styling
│   ├── components.py       ← shared UI pieces + session-state helpers
│   ├── screening.py        ← the guided screening flow (form → prediction → guidance)
│   └── pages/
│       ├── public.py       ← login, about, contact, help
│       ├── overview.py     ← dashboard
│       ├── screenings.py   ← condition picker + predictor
│       ├── analytics.py    ← data & model explorer
│       ├── clinical.py     ← doctor-only cohort screening
│       ├── assistant.py    ← Ask AHEAD chat
│       └── settings.py     ← settings tabs
├── train_model.py          ← benchmarks six model families per disease, saves models + metadata
├── login.css               ← styles for the public (pre-login) pages
├── assets/                 ← favicon and SVG icons (inlined as data URIs)
├── static/                 ← large background images, served by Streamlit at app/static/…
├── data/                   ← diabetes.csv · heart.csv · kidney_disease.csv
├── models/
│   ├── <disease>_model.pkl ← recommended pipeline per disease
│   ├── all_models/         ← every benchmarked pipeline (<disease>_<model>.pkl)
│   └── model_meta.json     ← features, metrics, thresholds, benchmark results
├── .streamlit/config.toml  ← Streamlit theme + server settings
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 is recommended.

---

## Running the app

```bash
streamlit run app.py
```

Open the printed URL (usually `http://localhost:8501`).

---

## Demo accounts

| Role | Email | Password |
|---|---|---|
| Patient | `user@ahead.demo` | `user123` |
| Doctor / Admin | `doctor@ahead.demo` | `doctor123` |

Doctor accounts additionally see the **Clinical Dashboard**. Accounts are defined in
`ahead/config.py` (`DEMO_ACCOUNTS`) — this is a prototype, not a real authentication system.

---

## AI features (Gemini)

AHEAD Insight (personalised result explanation) and the AI Assistant use Google Gemini
through the `google-genai` SDK. Provide an API key in **either** place:

```bash
export GEMINI_API_KEY="your-key"          # environment variable
```

or in `.streamlit/secrets.toml` (git-ignored):

```toml
GEMINI_API_KEY = "your-key"
```

Without a key the app still works: Insight shows a notice and the Assistant answers the
suggested questions from built-in text. The model name lives in `ahead/config.py`
(`GEMINI_MODEL`).

---

## Datasets

| Condition | File | Records | Target |
|---|---|---|---|
| Diabetes | `data/diabetes.csv` | 100,000 | `diabetes` |
| Cardiovascular | `data/heart.csv` | 246,022 | `HadHeartAttack` |
| Chronic kidney disease | `data/kidney_disease.csv` | 1,659 | `Diagnosis` |

Two of the datasets are strongly imbalanced (8.5 % positive for diabetes, 5.5 % for heart,
92 % positive for kidney), which is why the app reports disease recall, F1/F2 and ROC-AUC
alongside accuracy.

---

## Re-training the models

The trained pipelines are included. To rebuild everything:

```bash
python train_model.py
```

For each disease the script:

1. splits the data 64 % train / 16 % validation / 20 % test (stratified),
2. trains Random Forest, Logistic Regression, Gradient Boosting, SVM, Decision Tree and XGBoost
   inside a `Pipeline(preprocessor → classifier)`,
3. saves every pipeline to `models/all_models/` so the app can compare them,
4. picks the winner on the validation set (macro-F1 for diabetes/kidney, F2 for heart),
5. tunes the heart model's probability threshold on the validation set,
6. reports final metrics once on the untouched test set and writes `models/model_meta.json`.

---

## How the machine learning works

* **Preprocessing** — numeric columns are median-imputed and standardised; categorical columns
  are most-frequent-imputed and one-hot encoded (`ColumnTransformer`).
* **Prediction** — `predict_proba()` gives the probability of the disease class; the result is
  "elevated" when it reaches the model's decision threshold (0.5, or the tuned value for the
  recommended heart model).
* **Interpretation** — tree models expose `feature_importances_`, Logistic Regression uses
  absolute coefficient magnitude; RBF-SVM has no per-feature importance.
* **Rule-based factors** — independent of the model, the app highlights entered values that are
  commonly discussed with a clinician (e.g. HbA1c ≥ 5.7 %, BMI ≥ 30, eGFR < 60).

---

## Deploying on Streamlit Cloud

1. Push the repository to GitHub — `app.py`, the whole `ahead/` package, `assets/`, `static/`,
   `models/` and `data/` must all be committed (note: `data/heart.csv` is 82 MB, above GitHub's
   50 MB warning threshold; consider Git LFS).
2. On [share.streamlit.io](https://share.streamlit.io) create a new app pointing at `app.py`.
3. Add `GEMINI_API_KEY` under *Advanced settings → Secrets* if you want the AI features.

---

## Common errors

| Error | Fix |
|---|---|
| "No trained model is available" | Run `python train_model.py` to create `models/` |
| `ModuleNotFoundError` | Activate the virtual environment and `pip install -r requirements.txt` |
| A selected model says "has not been saved yet" | Run `python train_model.py` to generate `models/all_models/` |
| AI Assistant / Insight unavailable | Provide `GEMINI_API_KEY` (see above) |
| Background images missing | `server.enableStaticServing = true` must stay in `.streamlit/config.toml` and the PNGs in `static/` |
| Signed out after refreshing the page | Expected: the prototype keeps the session in memory only (no cookies) |
| Blank page after editing | Hard-refresh the browser (`Ctrl/Cmd + Shift + R`) |
