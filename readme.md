## NN Team Project

Predictive modeling of **hourly energy consumption (MW)** using gated recurrent architectures (LSTM) on the PJM AEP dataset.

## Running `model_update.py`

`model_update.py` is the “halfway update” training/evaluation script. It:
- builds next-hour forecasting sequences,
- trains an LSTM,
- evaluates via walk-forward validation (time-series splits),
- reports **RMSE** and **MAE**,
- optionally saves plots (predicted vs. actual, loss curve).

### Expected CSV location
- **Default**: `data/AEP_hourly.csv`
- Override with `--csv <path>`

### Expected CSV columns
This repository’s `data/AEP_hourly.csv` schema (as provided) has **two columns**:
- **`Datetime`**: hourly timestamp (parseable by pandas)
- **`AEP_MW`**: target load in MW (float)

No other columns are expected.

### Example commands
Run with defaults:

```bash
python3 model_update.py
```

Run on the real dataset with time-derived seasonality features:

```bash
source .venv/bin/activate
python model_update.py --csv data/AEP_hourly.csv --add-time-features --epochs 1 --splits 3 --seq-len 24
```

Outputs:
- metrics printed to stdout (per fold + mean)
- plots saved to `outputs/` (if `matplotlib` is installed)