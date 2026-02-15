import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from model.loading_data import load_data
from model.model_training import train_models
from model.model_evaluation import evaluate
from sklearn.metrics import confusion_matrix
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV_PATH = os.path.join(BASE_DIR, "data", "test.csv")

st.set_page_config(page_title="Mobile Price Classification", layout="centered")
st.title("📱 Mobile Price Classification App")

# Load data
X_train, X_val, y_train, y_val, X_test_final = load_data()

# Train models
models = train_models(X_train, y_train)
st.subheader("📂 Download Test Dataset")

with open('./data/test.csv', "rb") as file:
    st.download_button(
        label="Download test.csv",
        data=file,
        file_name="test.csv",
        mime="text/csv"
    )
# Model selection
model_name = st.selectbox("Select ML Model", list(models.keys()))
model = models[model_name]

# Evaluate on validation set
if st.button("Evaluate Model"):
    metrics = evaluate(model, X_val, y_val)

    st.subheader("📊 Evaluation Metrics (Validation Set)")
    for k, v in metrics.items():
        st.write(f"**{k}**: {v}")

    # Confusion Matrix
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    st.subheader("🔁 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Predict on final test data
st.subheader("📥 Generate Predictions on Test Dataset")

if st.button("Run Prediction on test.csv"):
    test_preds = model.predict(X_test_final)
    st.write("Sample predictions:", test_preds[:10])

st.subheader("📤 Upload CSV for Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file (same format as test.csv)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Read uploaded CSV
        upload_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(upload_df.head())

        # Drop id if present
        if "id" in upload_df.columns:
            upload_df = upload_df.drop("id", axis=1)

        # Load scaler + model data
        X_train, X_val, y_train, y_val, _ = load_data()

        # IMPORTANT: reuse scaler by refitting on train features
        scaler = joblib.load("scaler.pkl")
        upload_scaled = scaler.transform(upload_df)


        # Predict
        upload_preds = model.predict(upload_scaled)

        st.success("Prediction completed!")

        # Show sample predictions
        result_df = upload_df.copy()
        result_df["Predicted_Price_Range"] = upload_preds
        st.dataframe(result_df.head())

        # Download predictions
        st.download_button(
            label="Download Predictions CSV",
            data=result_df.to_csv(index=False),
            file_name="uploaded_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

st.subheader("📊 Model Comparison Table")

if st.button("Compare All Models"):
    comparison_results = []

    for name, model in models.items():
        metrics = evaluate(model, X_val, y_val)

        comparison_results.append({
            "ML Model Name": name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "AUC": round(metrics["AUC"], 4) if metrics["AUC"] != "Not Supported" else "NA",
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1": round(metrics["F1 Score"], 4),
            "MCC": round(metrics["MCC"], 4)
        })

    comparison_df = pd.DataFrame(comparison_results)

    st.dataframe(
        comparison_df,
        use_container_width=True
    )
