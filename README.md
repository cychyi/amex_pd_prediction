# AMEX Probability of Default (PD) — Local Thesis Pipeline

**Comparative Evaluation of Machine Learning and Deep Learning Models for Predicting Probability of Default**

Cham Ying Chyi (23076054) · University of Malaya · Supervisor: Prof. Dr Rafidah Md Noor

The whole pipeline now lives in **one self-contained notebook** you run locally (built for a Mac Mini M4):

```
amex_pd_pipeline.ipynb
```

It covers everything end to end — environment checks, data landing & EDA, preprocessing, feature engineering, feature selection, all models, evaluation and interpretability — and **prints a timestamp + elapsed time at every modelling step** so you can watch progress.

The standalone `src/*.py` scripts and `configs/`, `utils/` are kept for reference, but you do **not** need them: the notebook is fully self-contained, which avoids the import/path issues that come up when running scripts from inside VS Code.

---

## How to run (VS Code)

1. **Get the data.** Download the AMEX Default Prediction files and place them here:
   ```
   data/raw/train_data.csv
   data/raw/train_labels.csv
   ```
2. **Create a virtual environment and install dependencies** (VS Code terminal):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   On a Mac, LightGBM also needs the OpenMP runtime once: `brew install libomp`.
3. **Open `amex_pd_pipeline.ipynb`** and select the `.venv` kernel (top-right).
4. **Run the cells top to bottom.** Stage 0 checks the environment; Stages 1–4 prepare data; Stages 5a–5d train and evaluate.

Quick dry run on fake data: `python3 make_synthetic.py` creates a tiny synthetic `data/raw/`. Replace with the real CSVs and delete `data/parquet/` to switch.

---

## What the pipeline does

| Stage | Output |
|------|--------|
| 1 Landing & EDA | `data/parquet/`, `data/reports/eda_*.csv` |
| 2 Preprocessing | `data/interim/clean.parquet`, `kept_features.json` |
| 3 Feature engineering | `snapshot.parquet` (ML, 6 aggregations), `sequential.npz` (DL, top-15 features) |
| 4 Feature selection | `snapshot_selected.parquet` |
| 5a ML | Logistic Regression, XGBoost, LightGBM |
| 5b DL | LSTM, GRU, Transformer (13 epochs) |
| 5c Ensembles | LightGBM+GRU, XGBoost+LSTM, LightGBM+LSTM, XGBoost+GRU |
| 5d Evaluation | `comparison_train.csv`, `comparison_validate.csv`, `best_model_deciles.csv`, `shap_top10.csv`, `ig_top10.csv` |

**Key memory choices (16 GB):** snapshot keeps all 6 aggregations (~1,000 ML features); the **sequential DL tensor uses only the top-15 features** (LightGBM gain), so it is ~`(458913, 13, 15)` ≈ 0.35 GB instead of ~4 GB, with column-by-column standardisation. This prevents the out-of-memory ("killed") crash. DL device auto-detects **MPS (Apple GPU)** → CUDA → CPU.

Metrics follow the slides: ROC-AUC, KS, Gini (`2·AUC−1`), log-loss, 10-decile risk-ranking, and the train-vs-validate **Diff Gini %** overfit rule (>30% ⇒ overfit).

---

## Export for your presentation

- VS Code: Command Palette → **Jupyter: Export to HTML** (or PDF), or
- Terminal: `jupyter nbconvert --to html amex_pd_pipeline.ipynb`

---

## Push to GitLab

Raw data and generated artefacts are git-ignored — only code, the notebook and small report CSVs are committed.

```bash
git init
git add .
git commit -m "AMEX PD thesis: full local pipeline (ML + DL + ensembles)"
git branch -M main
git remote add origin https://gitlab.com/<your-username>/<your-repo>.git
git push -u origin main
```

If GitLab rejects the password, use a **Personal Access Token** (GitLab → Settings → Access Tokens, scope `write_repository`) as the password, or use SSH. Wrong remote? `git remote remove origin`, then add the correct URL.

---

## Repository layout

```
amex-pd-thesis/
├── amex_pd_pipeline.ipynb     # <- run this (self-contained)
├── make_synthetic.py          # tiny fake dataset for a quick dry run
├── requirements.txt
├── README.md
├── data/                      # raw/ (you add CSVs); everything else generated
├── configs/ utils/ src/       # reference implementation (optional)
└── notebooks/                 # earlier Kaggle templates (optional)
```
