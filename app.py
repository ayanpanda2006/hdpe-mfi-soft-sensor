import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title="HDPE Reactor MFI Soft Sensor", layout="wide")

st.title("🧪 HDPE Reactor MFI Soft Sensor")
st.caption(
    "Upload your reactor sensor data (the same Excel file used in the notebook) "
    "to train an XGBoost soft sensor that predicts Reactor MFI at 5 kg from "
    "sensor readings alone — no GRADE column used."
)

# ----------------------------------------------------------------------------
# Sidebar: inputs & settings
# ----------------------------------------------------------------------------
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Excel file (e.g. 'BASE DATA REACTOR MFI 5 KG.xlsx')", type=["xlsx", "xls"]
)

st.sidebar.header("2. Feature Engineering Settings")
window_size = st.sidebar.selectbox(
    "Rolling window size", ["2h", "4h", "6h"], index=0,
    help="Historical window used to compute rolling mean/std of each sensor. "
         "1h was removed — sensor logging intervals are close to or longer than "
         "1 hour, so a 1h window often captures only a single reading, which "
         "breaks the rolling standard deviation calculation and wipes out the dataset."
)
shift_periods = st.sidebar.number_input(
    "Target shift (rows)", min_value=0, max_value=10, value=1,
    help="Shifts the target this many rows into the future to align sensor "
         "readings with the lab result delay (dead-time)."
)

st.sidebar.header("3. Training Settings")
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
tolerance_pct = st.sidebar.slider("Tolerance band for accuracy (%)", 5, 20, 10, 1)
quick_mode = st.sidebar.checkbox(
    "Quick mode (smaller hyperparameter grid)", value=True,
    help="Uncheck to run the full grid search from the notebook (slower)."
)

run_button = st.sidebar.button("🚀 Run Model", type="primary", use_container_width=True)


# ----------------------------------------------------------------------------
# Cached pipeline steps
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(file):
    df = pd.read_excel(file)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").set_index("Timestamp")

    target_col = "Reactor MFI at 5 kg"
    grade_col = "GRADE"

    df[grade_col] = df[grade_col].astype(str).str.strip()

    sensor_cols = [c for c in df.columns if c not in [target_col, grade_col]]
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[sensor_cols] = df[sensor_cols].ffill()

    df[target_col] = df[target_col].astype(str).str.rstrip(".")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    initial_count = len(df)
    df = df.drop_duplicates()
    n_dupes = initial_count - len(df)

    df = df.dropna(subset=[target_col])

    return df, sensor_cols, target_col, grade_col, n_dupes


@st.cache_data(show_spinner=False)
def engineer_features(df, sensor_cols, target_col, window_size, shift_periods):
    rolling_mean = df[sensor_cols].rolling(window=window_size).mean().add_suffix("_mean")
    rolling_std = df[sensor_cols].rolling(window=window_size).std().add_suffix("_std")
    rate_of_change = df[sensor_cols].diff().add_suffix("_trend")

    df_features = pd.concat([df[[target_col]], rolling_mean, rolling_std, rate_of_change], axis=1)
    df_features["Target_Shifted"] = df_features[target_col].shift(-shift_periods)

    df_model_ready = df_features.dropna()

    X = df_model_ready.drop(columns=[target_col, "Target_Shifted"])
    y = df_model_ready["Target_Shifted"]
    return X, y


def train_model(X, y, test_size, quick_mode):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=42
    )

    xgb_base = XGBRegressor(random_state=42, objective="reg:squarederror")

    if quick_mode:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        }
    else:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "reg_alpha": [0.1, 1.0],
        }

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    predictions = best_xgb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return best_xgb, grid_search.best_params_, X_train, X_test, y_test, predictions, rmse


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload your Excel file in the sidebar to get started.")
    st.stop()

with st.spinner("Loading and cleaning data..."):
    df, sensor_cols, target_col, grade_col, n_dupes = load_and_clean(uploaded_file)

col1, col2, col3 = st.columns(3)
col1.metric("Rows after cleaning", len(df))
col2.metric("Duplicate rows removed", n_dupes)
col3.metric("Sensor columns", len(sensor_cols))

with st.expander("Preview cleaned data"):
    st.dataframe(df.head(20), use_container_width=True)

if not run_button:
    st.info("Adjust settings in the sidebar, then click **Run Model**.")
    st.stop()

with st.spinner("Engineering features (rolling windows, trends, target shift)..."):
    X, y = engineer_features(df, sensor_cols, target_col, window_size, shift_periods)
st.success(f"Feature engineering complete! Training matrix shape: {X.shape}")

with st.spinner("Training and tuning XGBoost (GridSearchCV)... this can take a minute"):
    best_xgb, best_params, X_train, X_test, y_test, predictions, rmse = train_model(
        X, y, test_size, quick_mode
    )

st.subheader("🏆 Best Model Settings")
st.json(best_params)

# --- Performance metrics ---
r2 = r2_score(y_test, predictions)
within_tolerance = np.mean(
    np.abs((y_test - predictions) / y_test) <= (tolerance_pct / 100)
) * 100

st.subheader("📈 Model Performance Summary")
m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:.3f}")
m2.metric("R² (variance explained)", f"{r2:.3f}")
m3.metric(f"Accuracy within {tolerance_pct}%", f"{within_tolerance:.1f}%")

if r2 <= 0.0:
    st.error("⚠️ CRITICAL: R² is 0 or negative — the model isn't finding signal. Check for target leakage or flatline data.")
elif r2 < 0.5:
    st.warning("⚠️ Low correlation — the model is struggling to find patterns without the GRADE column.")
elif r2 > 0.8:
    st.success("✅ High correlation — the model is tracking the physics well!")
else:
    st.info("Moderate correlation.")

# --- Feature importance ---
st.subheader("🔍 Sensor Priority Ranking")
importances = best_xgb.feature_importances_
raw_priority_df = pd.DataFrame(
    {"Engineered_Feature": X_train.columns, "Priority Impact": importances * 100}
)


def get_original_name(col):
    return col.replace("_mean", "").replace("_std", "").replace("_trend", "")


raw_priority_df["Original Sensor"] = raw_priority_df["Engineered_Feature"].apply(get_original_name)
final_priority_df = (
    raw_priority_df.groupby("Original Sensor")["Priority Impact"]
    .sum()
    .sort_values(ascending=False)
)

fig_imp, ax_imp = plt.subplots(figsize=(10, max(3, 0.35 * len(final_priority_df))))
final_priority_df.sort_values().plot(kind="barh", ax=ax_imp, color="#2b7de9")
ax_imp.set_xlabel("Priority Impact (%)")
ax_imp.set_title("Sensor Importance (aggregated across mean/std/trend features)")
plt.tight_layout()
st.pyplot(fig_imp)

with st.expander("See raw importance table"):
    st.dataframe(
        final_priority_df.reset_index().rename(columns={"Priority Impact": "Priority Impact (%)"}),
        use_container_width=True,
    )

# --- Predictions table ---
st.subheader("📋 Predictions vs Actual (Test Set)")
showdown_df = pd.DataFrame({"Actual Lab MFI": y_test.values, "Predicted MFI": predictions})
st.dataframe(showdown_df.head(20), use_container_width=True)

# --- Plot ---
st.subheader("📊 Actual vs Predicted MFI")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test.values, label="Actual MFI (Lab Result)", alpha=0.8, color="blue")
ax.plot(predictions, label="XGBoost Soft Sensor", alpha=0.8, color="orange", linestyle="--")
ax.legend()
ax.set_title("Reactor MFI: Actual Lab Results vs XGBoost Predictions")
ax.set_xlabel("Chronological Test Samples")
ax.set_ylabel("MFI (g/10 min)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

st.caption("Built from your HDPE reactor MFI notebook — XGBoost soft sensor, GRADE column intentionally excluded.")
