# 🏥 HealthAI — AI-Based Healthcare Analytics & Disease Risk Prediction Platform

**Senior Design Project** | Python · Streamlit · Scikit-learn · Plotly

---

## 📋 Table of Contents

1. [What This Project Does](#-what-this-project-does)
2. [Project Structure](#-project-structure)
3. [Datasets Used](#-datasets-used)
4. [Setup Instructions (Windows / VS Code)](#-setup-instructions-windows--vs-code)
5. [Running the App](#-running-the-app)
6. [Re-training the Models](#-re-training-the-models)
7. [How the Machine Learning Works](#-how-the-machine-learning-works)
8. [Understanding Each File](#-understanding-each-file)
9. [Pushing to GitHub](#-pushing-to-github)
10. [Deploying on Streamlit Cloud](#-deploying-on-streamlit-cloud)
11. [Common Errors & Fixes](#-common-errors--fixes)

---

## 🎯 What This Project Does

This platform lets a user enter their **clinical measurements** (like blood glucose, cholesterol, BMI, etc.) and receive a **machine-learning-based disease risk score** — either for Diabetes or Heart Disease.

Under the hood:
- Two **Random Forest** machine learning models are pre-trained and saved as `.pkl` files
- When the user submits the form, the app loads the model and predicts the probability of disease
- Results are shown as a **gauge chart**, a **risk label** (High/Low), and **personalised health tips**
- A **Dataset Explorer** page lets you visualise the training data with interactive charts

---

## 📂 Project Structure

```
healthcare_platform/
│
├── app.py                  ← Main Streamlit app (the website)
├── train_model.py          ← Script to train and save ML models
├── requirements.txt        ← All Python packages needed
├── .gitignore              ← Files to exclude from GitHub
│
├── .streamlit/
│   └── config.toml         ← Streamlit theme settings
│
├── models/
│   ├── diabetes_model.pkl  ← Trained diabetes model
│   ├── heart_model.pkl     ← Trained heart disease model
│   └── model_meta.json     ← Model metadata (features, accuracy)
│
├── data/
│   ├── diabetes.csv        ← Pima Indians Diabetes Dataset (768 records)
│   └── heart.csv           ← Cleveland Heart Disease Dataset (297 records)
│
└── utils/
    └── helpers.py          ← Utility/helper functions
```

---

## 📊 Datasets Used

### Pima Indians Diabetes Dataset
- Records: 768
- Target column: `Outcome`
- Features: pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age
- Task: Binary classification: diabetic or not diabetic

### Cleveland Heart Disease Dataset
- Records: 297
- Target column: `condition`
- Features: 13 clinical factors including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, maximum heart rate, exercise-induced angina, oldpeak, slope, ca, and thal
- Task: Binary classification: heart disease present or not present

---

## 🛠️ Setup Instructions (Windows / VS Code)

### Step 1 — Open the project folder in VS Code

1. Extract the downloaded zip file (or folder) anywhere on your computer, e.g. `C:\Users\YourName\Desktop\healthcare_platform`
2. Open **VS Code**
3. Click `File → Open Folder` and select the `healthcare_platform` folder

### Step 2 — Open the Terminal in VS Code

- Press `` Ctrl + ` `` (backtick key, top-left of keyboard)
- You should see a terminal panel at the bottom

### Step 3 — Create a Virtual Environment

In the terminal, type:
```bash
python -m venv venv
```

This creates a folder called `venv` — it's an isolated Python environment for this project.

### Step 4 — Activate the Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

You should now see `(venv)` at the start of your terminal prompt. That means the environment is active.

> ⚠️ If you get an error like "execution policy", run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then try activating again.

### Step 5 — Install All Required Packages

```bash
pip install -r requirements.txt
```

This installs Streamlit, scikit-learn, pandas, plotly, and everything else. It may take 2–4 minutes.

---

## 🚀 Running the App

Make sure your virtual environment is **activated** (you see `(venv)` in the terminal), then run:

```bash
streamlit run app.py
```

Streamlit will print a URL like:
```
Local URL: http://localhost:8501
```

Open that in your browser and the app will load!

> Press `Ctrl + C` in the terminal to stop the app.

---

## 🔁 Re-training the Models

The `.pkl` files (trained models) are already included, so you don't need to run this. But if you want to re-train from scratch:

```bash
python train_model.py
```

This will:
1. Load `data/diabetes.csv` and `data/heart.csv`
2. Split into 80% training / 20% test data
3. Train a Random Forest model for each disease
4. Print accuracy and a classification report
5. Save new `.pkl` files to the `models/` folder

---

## 🤖 How the Machine Learning Works

### What is a Random Forest?
A Random Forest is a **collection of decision trees**. Each tree learns patterns from the data, like:
> "If Glucose > 140 AND BMI > 30 AND Age > 50 → likely diabetic"

With 200 trees, the model takes a **majority vote** to make a final prediction. This makes it more accurate and less prone to errors than a single tree.

### Pipeline (Scaler + Classifier)
The model uses a scikit-learn **Pipeline** — two steps chained together:

```
Raw input → StandardScaler → RandomForestClassifier → Probability
```

**StandardScaler** normalises all values to the same range (mean=0, std=1) so that large numbers (like cholesterol=240) don't unfairly dominate small ones (like pregnancies=2).

### Probability vs Prediction
- `predict_proba()` → gives a probability between 0 and 1 (e.g. `0.73` = 73% chance of disease)
- `predict()` → gives 0 or 1 (the binary decision)

### Feature Importance
After training, `clf.feature_importances_` tells you how much each feature contributed to the model's accuracy. This is shown in the bar chart inside the predictor pages.

---

## 📄 Understanding Each File

### `app.py`
The main file. Uses Streamlit to create the web interface. Key sections:
- `st.set_page_config(...)` — sets page title, icon, layout
- `@st.cache_resource` — loads the model once and caches it (so it doesn't reload on every click)
- `with st.form(...)` — creates the input form for the user
- `model.predict_proba(input_data)` — runs the ML prediction
- `st.plotly_chart(...)` — renders interactive charts

### `train_model.py`
Standalone training script. Run this to rebuild models. Uses:
- `train_test_split()` — splits data into training and test sets
- `Pipeline([...])` — chains scaler + classifier
- `pickle.dump(...)` — saves the trained model to disk

### `models/model_meta.json`
A JSON file that stores metadata about each model:
```json
{
  "diabetes": {
    "features": ["Pregnancies", "Glucose", ...],
    "accuracy": 0.734
  }
}
```
`app.py` reads this to know which feature names to use when building the input DataFrame.

### `requirements.txt`
Lists all Python packages. Streamlit Cloud reads this when deploying.

### `.streamlit/config.toml`
Sets the colour theme for the app.

---

## 📤 Pushing to GitHub

### Step 1 — Create a GitHub account
Go to [github.com](https://github.com) and sign up (free).

### Step 2 — Create a new repository
1. Click the `+` button → `New repository`
2. Name it e.g. `healthcare-ai-platform`
3. Set it to **Public**
4. **Do NOT** tick "Add README" (we already have one)
5. Click `Create repository`

### Step 3 — Connect your local folder to GitHub

In your VS Code terminal (with venv active):

```bash
git init
git add .
git commit -m "Initial commit: Healthcare AI platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/healthcare-ai-platform.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

> If Git asks for login, use your GitHub username and a **Personal Access Token** (not password).
> Create one at: GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)

### Step 4 — Verify
Go to `https://github.com/YOUR_USERNAME/healthcare-ai-platform` — you should see all your files!

---

## ☁️ Deploying on Streamlit Cloud

### Step 1 — Sign up for Streamlit Cloud
Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your **GitHub account**.

### Step 2 — Deploy your app
1. Click **"New app"**
2. Select your repository (`healthcare-ai-platform`)
3. Branch: `main`
4. Main file path: `app.py`
5. Click **"Deploy!"**

Streamlit Cloud will:
- Read `requirements.txt` and install all packages
- Run `streamlit run app.py`
- Give you a public URL like `https://yourname-healthcare-ai.streamlit.app`

> ⏱️ First deploy takes about 3–5 minutes.

---

## 🐛 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `'streamlit' is not recognized` | Streamlit not installed or venv not active | Activate venv and run `pip install -r requirements.txt` |
| `FileNotFoundError: models/diabetes_model.pkl` | Models not generated | Run `python train_model.py` |
| `ModuleNotFoundError: No module named 'plotly'` | Package missing | Run `pip install plotly` |
| `venv\Scripts\activate` fails on Windows | Execution policy blocked | Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| App shows blank page | Browser cache | Hard refresh: `Ctrl + Shift + R` |
| Git push fails | Wrong remote URL | Re-run `git remote set-url origin <your-url>` |

---

## 🔮 What to Add Next (Suggested Improvements)

1. **SHAP explanations** — Install `shap` and add `shap.TreeExplainer` to show per-patient feature contributions
2. **More diseases** — Add kidney disease, liver disease datasets
3. **Model comparison** — Show accuracy side-by-side for Random Forest vs Logistic Regression vs XGBoost
4. **Export report** — Allow users to download their results as a PDF
5. **User history** — Store predictions in a database (SQLite)

---

## 📚 Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [Pima Diabetes Dataset Info](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Heart Disease Cleveland UCI Dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

---

*Built with ❤️ for Senior Design Project | HealthAI Platform*
