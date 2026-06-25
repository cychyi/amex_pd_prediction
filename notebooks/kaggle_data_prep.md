# Kaggle Notebook 1 — Data Prep & Feature Engineering (CPU)

Run Stages 1–4 on Kaggle where the AMEX CSVs are already mounted, then save the
engineered features so the GPU notebook (Notebook 2) can train on them.

**Settings:** Accelerator = *None* (CPU), Internet = *On* (only needed if you
`pip install` anything; torch/lightgbm/xgboost are pre-installed).
**Attach data:** the `amex-default-prediction` competition dataset.

```python
# Cell 1 — get the code (clone your GitHub repo, or upload it as a Dataset)
!git clone https://github.com/<your-username>/amex-pd-thesis.git
%cd amex-pd-thesis
```

```python
# Cell 2 — Kaggle is auto-detected (config.py sees /kaggle/input). Run stages 1-4.
!python src/01_data_landing_eda.py
!python src/02_data_preprocessing.py
!python src/03_feature_engineering.py
!python src/04_feature_selection.py
```

```python
# Cell 3 — collect the engineered features into /kaggle/working so they persist
import shutil, os
os.makedirs('/kaggle/working/features', exist_ok=True)
for f in ['snapshot_selected.parquet', 'sequential.npz', 'split.npz',
          'seq_feature_list.json', 'seq_scaler.npz', 'selected_features.json']:
    src = f'/kaggle/working/features/{f}'
    if os.path.exists(src):
        print('ok', f)
# Also copy the labels the modelling stages need.
shutil.copy('/kaggle/working/interim/labels.parquet',
            '/kaggle/working/features/labels.parquet')
```

After the notebook finishes, click **"Save Version"**. Then **create a Dataset
from the notebook output** (the `features/` folder), e.g. named
`amex-pd-features`. Notebook 2 attaches that Dataset for GPU training.

> Tip: if memory is tight during Stage 3, the sequential tensor is the biggest
> object (~4–5 GB at full scale). It still fits in Kaggle's 30 GB RAM. If you
> ever hit limits, lower `MAX_SEQ_LEN` or process customers in batches.
