# House Price Lab

House Price Lab is an interactive machine-learning workspace for estimating California house values and understanding how regression models behave. It turns a basic prediction project into a small experiment journal: compare models, inspect errors, explain feature influence, and save each run.

## What it includes

- Reproducible California Housing dataset with no manual download.
- Leakage-resistant preprocessing and a fixed 80/20 train-test split.
- Linear Regression, Random Forest, and Gradient Boosting models.
- MAE, RMSE, and R2 evaluation.
- Interactive Streamlit views for overview, training, prediction, explanations, and error analysis.
- Saved JSON experiment history and the best trained model under `artifacts/` and `experiments/`.
- Focused tests for the data, model, metrics, and experiment store.

The dataset target is measured in units of `$100,000`. For example, a prediction of `2.5` is displayed as approximately `$250k`. Predictions are educational estimates, not property appraisals or financial advice. Feature importance describes model association, not causation.

## Setup

The repository includes a `genz` environment from the original scaffold. It is ignored by Git, but a fresh environment is recommended:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell blocks activation, run the commands with the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the dashboard

```powershell
streamlit run app.py
```

The app opens with an overview of the data. Use **Train & compare** before visiting **Predict**, **Explain**, or **Error analysis**. Each training run is saved in the experiment journal.

## Run tests

```powershell
pytest -q
```

## Project structure

```text
app.py                         Streamlit dashboard
src/house_price_lab/data.py   Dataset loading and deterministic splitting
src/house_price_lab/modeling.py
							   Model pipelines, evaluation, and persistence
src/house_price_lab/experiments.py
							   JSON experiment history
tests/test_core.py             Core behavior tests
artifacts/                     Generated model files, ignored by Git
experiments/                   Generated run records, ignored by Git
```

## Design notes

The application favors transparent comparisons over a large hyperparameter search. All models see the same split and preprocessing pipeline. The California Housing dataset is convenient for a local demo, but it is historical and aggregated; results should not be interpreted as current market valuations.