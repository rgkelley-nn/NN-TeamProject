## NN Team Project

Predictive modeling of **hourly energy consumption (MW)** using gated recurrent architectures (LSTM) on the PJM AEP dataset.

## Running `model_update.py`

`model_update.py` trains an LSTM to predict the next hour’s `AEP_MW` and evaluates with walk-forward splits. It prints RMSE/MAE and can save a couple plots.

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
python model_update.py --add-time-features --epochs 1 --splits 3 --seq-len 24
```

Outputs:
- metrics printed to stdout (per fold + mean)
- plots saved to `outputs/` (if `matplotlib` is installed)