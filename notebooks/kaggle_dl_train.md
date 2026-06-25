# Kaggle Notebook 2 — Modelling & Evaluation (GPU)

Trains the ML, DL and ensemble models on the engineered features from Notebook 1
and produces the comparison tables. The DL models use the free Kaggle GPU.

**Settings:** Accelerator = *GPU T4 x2* (or P100), Internet = *On*.
**Attach data:** your `amex-pd-features` Dataset (output of Notebook 1).

```python
# Cell 1 — get the code
!git clone https://github.com/<your-username>/amex-pd-thesis.git
%cd amex-pd-thesis
!pip install captum shap -q        # interpretability extras (torch/lgb/xgb pre-installed)
```

```python
# Cell 2 — point the pipeline at the attached features Dataset.
# config.py writes to /kaggle/working by default; we symlink the input features in.
import os
os.makedirs('/kaggle/working/features', exist_ok=True)
os.makedirs('/kaggle/working/interim', exist_ok=True)
FEAT = '/kaggle/input/amex-pd-features'      # <-- your Dataset path
for f in os.listdir(FEAT):
    dst = f'/kaggle/working/features/{f}'
    if not os.path.exists(dst):
        os.symlink(f'{FEAT}/{f}', dst)
# labels.parquet is needed by the ensemble/eval stages
if os.path.exists(f'{FEAT}/labels.parquet'):
    os.symlink(f'{FEAT}/labels.parquet', '/kaggle/working/interim/labels.parquet')
```

```python
# Cell 3 — confirm the GPU is visible
import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

```python
# Cell 4 — optional: enable XGBoost GPU for a big speedup
# Edit configs/config.py -> XGB_PARAMS["tree_method"] = "gpu_hist"
```

```python
# Cell 5 — run modelling + evaluation
!python src/05a_modelling_ml.py
!python src/05b_modelling_dl.py
!python src/05c_modelling_ensemble.py
!python src/05d_evaluation.py --shap --ig
```

```python
# Cell 6 — view the slide-style comparison tables
import pandas as pd
print('TRAIN'); display(pd.read_csv('/kaggle/working/reports/comparison_train.csv'))
print('VALIDATE'); display(pd.read_csv('/kaggle/working/reports/comparison_validate.csv'))
```

Click **"Save Version"** to persist the report CSVs and trained models as the
notebook output — you can attach these to your thesis or download them.

> The DL stage prints validation AUC per epoch with early stopping. If a session
> nears the 9-hour cap, reduce `DL_PARAMS["epochs"]` or train one model at a time.
