import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Aircraft Engine RUL Prediction",
    page_icon="✈️",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("✈️ Aircraft Engine Remaining Useful Life Prediction")

st.write(
    "Upload NASA C-MAPSS FD001 test data to estimate the "
    "Remaining Useful Life (RUL) of an aircraft engine."
)


# ---------------------------------------------------------
# Load trained model and scaler
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    # Load XGBoost model from native JSON format
    model = xgb.XGBRegressor()
    model.load_model("engine_rul_model.json")

    # Load StandardScaler
    scaler = joblib.load("scaler.pkl")

    return model, scaler


try:
    model, scaler = load_models()

except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# ---------------------------------------------------------
# FD001 column names
# ---------------------------------------------------------

columns = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensors_1",
    "sensors_2",
    "sensors_3",
    "sensors_4",
    "sensors_5",
    "sensors_6",
    "sensors_7",
    "sensors_8",
    "sensors_9",
    "sensors_10",
    "sensors_11",
    "sensors_12",
    "sensors_13",
    "sensors_14",
    "sensors_15",
    "sensors_16",
    "sensors_17",
    "sensors_18",
    "sensors_19",
    "sensors_20",
    "sensors_21"
]


# ---------------------------------------------------------
# Features used during model training
# ---------------------------------------------------------

feature_columns = [
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "sensors_2",
    "sensors_3",
    "sensors_4",
    "sensors_6",
    "sensors_7",
    "sensors_8",
    "sensors_9",
    "sensors_11",
    "sensors_12",
    "sensors_13",
    "sensors_14",
    "sensors_15",
    "sensors_17",
    "sensors_20",
    "sensors_21"
]


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("✈️ Aircraft Engine Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload FD001 test data (.txt)",
    type=["txt"]
)


# ---------------------------------------------------------
# Wait for file upload
# ---------------------------------------------------------

if uploaded_file is None:

    st.info(
        "👈 Upload the NASA C-MAPSS FD001 test data "
        "from the sidebar to begin."
    )

    st.stop()


# ---------------------------------------------------------
# Read FD001 dataset
# ---------------------------------------------------------

try:

    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+",
        header=None
    )

    # FD001 test data should contain 26 columns
    if df.shape[1] != 26:

        st.error(
            f"Expected 26 columns in FD001 data, "
            f"but found {df.shape[1]}."
        )

        st.stop()

    df.columns = columns

except Exception as e:

    st.error(
        f"Could not read the uploaded FD001 file: {e}"
    )

    st.stop()


# ---------------------------------------------------------
# Engine selection
# ---------------------------------------------------------

engine_ids = sorted(
    df["unit_id"].unique()
)

selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_ids
)


# ---------------------------------------------------------
# Select engine data
# ---------------------------------------------------------

engine_df = df[
    df["unit_id"] == selected_engine
].copy()

engine_df = engine_df.sort_values(
    "cycle"
).reset_index(drop=True)


# ---------------------------------------------------------
# Prepare features
# ---------------------------------------------------------

X = engine_df[
    feature_columns
].copy()


# ---------------------------------------------------------
# Scale and predict
# ---------------------------------------------------------

try:

    X_scaled = scaler.transform(X)

    predictions = model.predict(
        X_scaled
    )

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()


# Prevent negative RUL predictions
predictions = np.maximum(
    predictions,
    0
)

engine_df["Predicted_RUL"] = predictions


# ---------------------------------------------------------
# Latest prediction
# ---------------------------------------------------------

latest_prediction = float(
    engine_df["Predicted_RUL"].iloc[-1]
)

latest_cycle = int(
    engine_df["cycle"].iloc[-1]
)


# ---------------------------------------------------------
# Engine status
# ---------------------------------------------------------

if latest_prediction > 50:

    status = "🟢 Healthy"

elif latest_prediction > 20:

    status = "🟡 Degrading"

else:

    status = "🔴 Critical"


# ---------------------------------------------------------
# Dashboard header
# ---------------------------------------------------------

st.subheader(
    f"Engine {selected_engine} — RUL Analysis"
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Current Cycle",
        latest_cycle
    )


with col2:

    st.metric(
        "Predicted RUL",
        f"{latest_prediction:.1f} cycles"
    )


with col3:

    st.metric(
        "Engine Status",
        status
    )


# ---------------------------------------------------------
# RUL trend
# ---------------------------------------------------------

st.subheader(
    "📈 Remaining Useful Life Trend"
)


fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    engine_df["cycle"],
    engine_df["Predicted_RUL"],
    linewidth=2
)

ax.set_xlabel(
    "Engine Cycle"
)

ax.set_ylabel(
    "Predicted Remaining Useful Life"
)

ax.set_title(
    f"Engine {selected_engine} — "
    "Predicted RUL vs Cycle"
)

ax.grid(
    True,
    alpha=0.3
)

st.pyplot(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# Prediction table
# ---------------------------------------------------------

st.subheader(
    "🔍 Engine Prediction Data"
)


display_df = engine_df[
    [
        "cycle",
        "Predicted_RUL"
    ]
].copy()


display_df["Predicted_RUL"] = (
    display_df["Predicted_RUL"]
    .round(2)
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------

with st.expander(
    "📊 Dataset Information"
):

    st.write(
        f"**Total engines:** {len(engine_ids)}"
    )

    st.write(
        f"**Selected engine:** Engine {selected_engine}"
    )

    st.write(
        f"**Cycles available:** {len(engine_df)}"
    )

    st.write(
        f"**Features used by model:** {len(feature_columns)}"
    )