## Halfway Progress Update (5-minute script)

Hi everyone—this is our halfway progress update for **Predictive Modeling of Hourly Energy Consumption using Gated Recurrent Architectures**

Our goal is to predict the next hour’s energy consumption in **Megawatts (MW)** using the PJM **AEP hourly** dataset, which contains about **121,273 hourly observations** from 2004 to 2018. In the version of the dataset we’re using right now, the CSV includes only two columns: `Datetime` and `AEP_MW`. So our current inputs are historical MW load, plus optional calendar/seasonality features derived from `Datetime` (hour-of-day, day-of-week, and day-of-year encodings).

---

### Motivation (Why not “memoryless” models?)

A key challenge is that energy load is strongly **temporal**: it depends not just on the current hour, but on patterns across the last day, week, and season. A “memoryless” model—like a plain linear regression—can’t naturally represent these lagged dependencies.

A major driver of short-term dynamics is regular seasonality: daily cycles, weekly patterns, and longer annual trends. By deriving calendar features from `Datetime`, we give the model signals for these periodic effects, while the LSTM learns longer-range dependencies from the historical load sequence itself.

---

### Methodology (LSTM + gates)

We’re using an **LSTM**, which is a gated recurrent architecture designed to maintain and update a memory state over time. At each time step $t$, the LSTM maintains a hidden state $h_t$ and a cell state $c_t$.

* **Forget gate** decides what portion of the previous memory to retain:
    $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
* **Input gate** decides what new information to write:
    $$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad \tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
* **Cell update** combines forgetting and writing:
    $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
* **Output gate** decides what to reveal as the hidden state:
    $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(c_t)$$

---

### Validation (Walk-forward, no leakage)

For evaluation, we use a walk-forward validation strategy with `TimeSeriesSplit`. The key rule is: **train on the past, validate on the future**. We evaluate with **RMSE** and **MAE**:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{j=1}^{n} (y_j - \hat{y}_j)^2}, \quad \text{MAE} = \frac{1}{n} \sum_{j=1}^{n} |y_j - \hat{y}_j|$$

---

### Team Prior Work

A team member previously worked with this exact dataset using a different architecture, which gave us a baseline workflow for cleaning and feature alignment. We’re building on that foundation but switching to an LSTM to capture longer-range seasonal patterns.

---

### References

* **PJM AEP Dataset**: Kaggle.
* **Course Lecture**: Lobaton, E. RNN/LSTM Lecture, NC State University, 2026.