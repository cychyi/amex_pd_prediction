#!/usr/bin/env bash
# Run the full PD pipeline end-to-end.
# Usage:  bash run_all.sh
set -e

echo ">>> Stage 1  data landing & EDA"
python src/01_data_landing_eda.py

echo ">>> Stage 2  preprocessing"
python src/02_data_preprocessing.py

echo ">>> Stage 3  feature engineering (snapshot + sequential)"
python src/03_feature_engineering.py

echo ">>> Stage 4  feature selection"
python src/04_feature_selection.py

echo ">>> Stage 5a  ML models (LogReg / XGBoost / LightGBM)"
python src/05a_modelling_ml.py

echo ">>> Stage 5b  DL models (LSTM / GRU / Transformer)"
python src/05b_modelling_dl.py

echo ">>> Stage 5c  ensembles (LGBM+GRU / XGB+LSTM / LGBM+LSTM / XGB+GRU)"
python src/05c_modelling_ensemble.py

echo ">>> Stage 5d  evaluation, comparison tables & interpretability"
python src/05d_evaluation.py --shap --ig

echo ">>> DONE. See data/reports/ for the comparison tables."
