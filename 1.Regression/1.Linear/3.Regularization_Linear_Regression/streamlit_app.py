import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------
# Load trained models
# -------------------------
with open("models/all_regression_models.pkl", "rb") as f:
    models = pickle.load(f)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Car MPG Predictor",
    page_icon="🚗",
    layout="centered"
)

# -------------------------
# Header Branding
# -------------------------
st.title("🚗 Car MPG Prediction")
st.markdown("### Powered by **Regularization Models**")
st.markdown("---")

# -------------------------
# Sidebar for Model Selection
# -------------------------
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Choose Mode", ["Single Model", "Compare All Models"])

if mode == "Single Model":
    model_choice = st.sidebar.selectbox("Select Regression Model", list(models.keys()))
    model = models[model_choice]
else:
    st.sidebar.info("All models will be compared simultaneously.")

# -------------------------
# User Input Section
# -------------------------
st.subheader("🔧 Input Car Attributes")

col1, col2 = st.columns(2)
with col1:
    cyl = st.number_input("Cylinders", min_value=3, max_value=12, step=1, value=4)
    disp = st.number_input("Displacement", min_value=50.0, max_value=500.0, step=1.0, value=150.0)
    hp = st.number_input("Horsepower", min_value=40.0, max_value=250.0, step=1.0, value=90.0)
    wt = st.number_input("Weight", min_value=1500.0, max_value=5000.0, step=10.0, value=2500.0)

with col2:
    acc = st.number_input("Acceleration", min_value=5.0, max_value=30.0, step=0.1, value=15.0)
    yr = st.number_input("Model Year", min_value=60, max_value=100, step=1, value=76)
    car_type = st.selectbox("Car Type", ["sedan", "hatchback"])
    origin = st.selectbox("Origin", ["america", "europe", "asia"])

# -------------------------
# Prepare Input
# -------------------------
origin_america, origin_asia, origin_europe = 0, 0, 0
if origin == "america":
    origin_america = 1
elif origin == "asia":
    origin_asia = 1
else:
    origin_europe = 1

car_type_val = 1 if car_type == "sedan" else 0

# ✅ Correct order of features (matches training dataset)
expected_order = [
    "cyl", "disp", "hp", "wt", "acc", "yr",
    "car_type", "origin_america", "origin_asia", "origin_europe"
]

input_data = pd.DataFrame([[cyl, disp, hp, wt, acc, yr, car_type_val,
                            origin_america, origin_asia, origin_europe]],
                          columns=expected_order)

# -------------------------
# Prediction Logic
# -------------------------
if st.button("🚀 Predict MPG"):
    if mode == "Single Model":
        mpg_pred = float(model.predict(input_data)[0])   # ✅ Ensure scalar
        st.success(f"Estimated Mileage (MPG) using **{model_choice}**: **{mpg_pred:.2f}**")

    else:
        results = {}
        for name, mdl in models.items():
            pred_value = float(mdl.predict(input_data)[0])   # ✅ Ensure scalar
            results[name] = round(pred_value, 2)

        st.subheader("📊 Comparison of All Models")
        df_results = pd.DataFrame(results.items(), columns=["Model", "Predicted MPG"])
        st.table(df_results)

        best_model = df_results.loc[df_results["Predicted MPG"].idxmax()]
        st.info(f"✅ Best predicted MPG: **{best_model['Predicted MPG']}** by **{best_model['Model']}**")

# -------------------------
# Footer Branding
# -------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:14px;">
    🚀 Developed by <b>Mubasshir Ahmed</b> | 
    <a href="https://github.com/mubasshirahmed-3712" target="_blank">GitHub</a> | 
    <a href="https://www.linkedin.com/in/mubasshir3712/" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
