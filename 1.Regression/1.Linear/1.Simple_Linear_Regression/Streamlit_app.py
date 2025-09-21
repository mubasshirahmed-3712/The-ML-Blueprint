import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load the saved model
# -------------------------------
MODEL_PATH = os.path.join("models", "linear_regression_model.pkl")

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"❌ Model file not found at {MODEL_PATH}! Please train and save the model first.")
    st.stop()

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Salary Prediction App", page_icon="💰", layout="centered")

# Title & Description
st.title("💼 Salary Prediction App")
st.markdown(
    """
    This app predicts **employee salary** based on their **years of experience**.  
    Simply enter the years of experience below and click **Predict Salary**.
    """
)

# -------------------------------
# User Input
# -------------------------------
years_of_experience = st.number_input(
    "📊 Enter years of experience:",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5,
    help="Use decimals if needed (e.g., 2.5 years)"
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔮 Predict Salary"):
    experience_input = np.array([[years_of_experience]])
    prediction = model.predict(experience_input)
    st.success(
        f"💡 The predicted salary for **{years_of_experience} years** of experience is: "
        f"**${prediction[0]:,.2f}**"
    )

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📈 Regression Visualization")

    # Generate regression line
    x_range = np.linspace(0, 20, 100).reshape(-1, 1)
    y_range = model.predict(x_range)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(x_range, y_range, color="blue", label="Regression Line")
    ax.scatter(years_of_experience, prediction, color="red", s=100, label="Your Prediction")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# Model Info Section
# -------------------------------
st.markdown("---")
st.subheader("ℹ️ About the Model")
st.write(
    """
    - The model was trained on a dataset containing employee salaries and years of experience.  
    - It learns the relationship between **experience** and **salary** to make predictions.  
    - This simple regression-based model can be extended with more features (education, skills, industry, etc.) for improved accuracy.  
    
    **~ Mubasshir Ahmed**
    """
)

# Footer
st.markdown("---")
st.markdown("🔗 [GitHub Repository](https://github.com/mubasshirahmed-3712) | ✨ Built by *Mubasshir Ahmed*")
