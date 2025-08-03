 # app.py
import streamlit as st
import pandas as pd
import joblib
import os
from model_utils import train_model, predict_model

st.set_page_config(page_title="📊 Generic ML Predictor", layout="centered")
st.title("📊 Generic ML Predictor App")

# ✅ Updated menu with "Manual Input"
menu = st.sidebar.selectbox("Choose Action", ["Train Model", "Predict Price", "Manual Input"])

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

# =======================
# ✅ TRAINING SECTION
# =======================
if menu == "Train Model":
    st.header("🧠 Upload Training Dataset (CSV or Excel)")
    train_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"], key="train")

    if train_file:
        df = load_file(train_file)
        if df is not None:
            st.write("📄 Preview of Training Data", df.head())

            target_col = st.selectbox("🎯 Select Target Column", df.columns)
            cat_cols = st.multiselect("🔤 Select Categorical Columns", df.columns.difference([target_col]))

            if st.button("🚀 Train Model"):
                train_model(df, target_col, cat_cols)
                st.success("✅ Model Trained and Saved Successfully!")

# =======================
# ✅ MANUAL INPUT SECTION
# =======================
elif menu == "Manual Input":
    st.header("📝 Enter Feature Values Manually")

    if os.path.exists("model.pkl") and os.path.exists("pipeline.pkl") and os.path.exists("metadata.pkl"):
        metadata = joblib.load("metadata.pkl")
        features = metadata["features"]
        cat_attribs = metadata["cat_attribs"]
        num_attribs = metadata["num_attribs"]

        input_data = {}
        with st.form("manual_form"):
            for col in features:
                if col in cat_attribs:
                    input_data[col] = st.text_input(f"{col}", key=col)  # Optional: improve by saving actual unique values
                else:
                    input_data[col] = st.number_input(f"{col}", value=0.0, key=col)

            submitted = st.form_submit_button("🔮 Predict")

        if submitted:
            df = pd.DataFrame([input_data])
            result_df = predict_model(df)
            st.success("✅ Prediction Complete")
            st.write(result_df)
    else:
        st.warning("⚠️ Please train the model first to enable manual input.")

# =======================
# ✅ FILE-BASED PREDICTION
# =======================
elif menu == "Predict Price":
    st.header("📂 Upload Data for Prediction")
    predict_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"], key="predict")

    if predict_file:
        df = load_file(predict_file)
        if df is not None:
            st.write("📄 Preview of Input Data", df.head())

            if os.path.exists("model.pkl") and os.path.exists("pipeline.pkl"):
                result_df = predict_model(df)
                st.success("✅ Prediction Complete")
                st.write(result_df)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
            else:
                st.error("❌ Please train the model first.")
