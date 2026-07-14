# 🧪 HDPE Reactor MFI Soft Sensor

Predicting plastic quality *before* the lab does — using AI on live reactor sensors.

🔗 **[Try the live app](https://hdpe-mfi-soft-sensor-cdweh8t7qnnzknatg4tfqf.streamlit.app)**

---

## The Problem

MFI (Melt Flow Index) tells us if a batch of HDPE plastic came out right. Right now, we only find out the MFI after the lab tests a sample — and that takes hours. By the time the result comes back, the reactor may have already made a bad batch.

## The Idea: A "Soft Sensor"

A soft sensor isn't a physical device — it's a model that predicts the lab result instantly, using sensor readings we already have: temperature, pressure, flow rates, and more. Think of it like predicting tomorrow's weather from today's clouds and wind, instead of waiting until tomorrow to find out.

## The Challenge: No Cheating Allowed

Every batch has a product "Grade" label that's an easy giveaway of what MFI to expect. We deliberately excluded it — the model has to predict MFI purely from sensor behavior, not from being told the answer's category in disguise. It's a harder test, but a more honest one: if sensors alone can predict quality, this becomes a true early-warning system, useful on any grade, not just ones seen before.

## How It Works

1. **Clean the data** — fix gaps, remove duplicates, drop bad readings
2. **Engineer features** — 2-hour rolling averages and rate-of-change trends for every sensor
3. **Fix the timing** — shift the lab result forward so sensor readings align with the lab's reporting delay
4. **Train & tune** — XGBoost, auto-tuned via GridSearchCV

## Results

| Metric | Value |
|---|---|
| RMSE | 36.78 |
| R² Score | 0.571 |
| Accuracy within 10% | 29.9% |

**Verdict:** Moderate correlation — the model finds a real signal and tracks overall trends well, but isn't yet accurate enough to fully replace lab testing. Sudden sharp spikes in MFI are harder to catch; the model tends to smooth over big swings.

### Top sensor drivers
1. **H2/C2 Ratio** — controls chain length, directly tied to melt flow
2. **Mother Liquor Flow (T1)** — affects catalyst concentration in the reactor

These two alone explain over a third of the model's decisions.

## Tech Stack

- **Streamlit** — interactive web app
- **XGBoost** — gradient-boosted regression model
- **scikit-learn** — train/test splitting, GridSearchCV, metrics
- **pandas / numpy** — data cleaning and feature engineering
- **matplotlib** — visualizations

## Run It Yourself

```bash
git clone https://github.com/ayanpanda2006/hdpe-mfi-soft-sensor.git
cd hdpe-mfi-soft-sensor
pip install -r requirements.txt
streamlit run app.py
```

Upload your own reactor sensor Excel file (same format as the notebook), adjust settings in the sidebar, and click **Run Model**.

## What's Next

- Collect more data to improve accuracy
- Better catch sudden spikes
- Test other models alongside XGBoost
- Pilot it live alongside real lab testing

---

Built by **Ayan Panda & Adarsh Taru**
