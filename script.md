## Halfway Progress Update (5 min)

Hi everyone—this is our halfway progress update for **Predicting Hourly Energy Consumption with LSTMs**.

Our goal is to predict the next hour’s energy consumption in **MW** using the PJM **AEP hourly** dataset (~121k hourly points, 2004–2018). The CSV we’re using has two columns: `Datetime` and `AEP_MW`. So the model uses the past load values (and optionally some time-of-day / day-of-week / day-of-year features derived from `Datetime`).

---

### Motivation

A key challenge is that load depends on what happened recently (last few hours / last day) and also longer seasonal patterns. A “memoryless” model (like plain linear regression on a single hour) tends to miss that.

A big part of the signal is seasonality: daily cycles, weekly patterns, and annual trends. We can give the model time-based features from `Datetime`, and the LSTM learns dependencies from the load sequence.

---

### Method

We’re using an **LSTM**. At each time step $t$, it keeps a hidden state $h_t$ and a cell state $c_t$.

* **Forget gate** decides what portion of the previous memory to retain:
    $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
* **Input gate** decides what new information to write:
    $$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad \tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
* **Cell update** combines forgetting and writing:
    $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
* **Output gate** decides what to reveal as the hidden state:
    $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(c_t)$$

---

### Validation

For evaluation, we use walk-forward splits (`TimeSeriesSplit`): train on earlier data and validate on later data. We report **RMSE** and **MAE**:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{j=1}^{n} (y_j - \hat{y}_j)^2}, \quad \text{MAE} = \frac{1}{n} \sum_{j=1}^{n} |y_j - \hat{y}_j|$$

---

### Team prior work

One of our team members already worked with this dataset using a different architecture, so we’re building off that baseline workflow and switching to an LSTM to better handle time dependencies.

---

### References

* **PJM AEP Dataset**: Kaggle.
* **Course Lecture**: Lobaton, E. RNN/LSTM Lecture, NC State University, 2026.