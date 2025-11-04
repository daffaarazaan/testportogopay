# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(page_title="Paysim Fraud Predictor", layout="wide")

@st.cache_resource
def load_model_and_meta():
    model = joblib.load("xgb_fraud_model.joblib")
    with open("model_features.json", "r") as f:
        features = json.load(f)
    return model, features

def build_feature_row(amount, old_org, new_org, old_dest, new_dest, step, tx_type, feature_cols):
    # Basic engineered features (same names as training)
    row = {}
    row['amount_log'] = np.log1p(amount)
    row['delta_org'] = old_org - new_org
    row['delta_dest'] = old_dest - new_dest
    row['hour_sim'] = int(step % 24)

    # Type dummies — ensure names match training one-hot columns
    type_names = [c for c in feature_cols if c.startswith("type_")]
    for t in type_names:
        # t looks like "type_TRANSFER" or "type_CASH_OUT"
        t_label = t.split("type_",1)[1]
        row[t] = 1 if t_label == tx_type else 0

    # Build DataFrame and reindex to feature order
    row_df = pd.DataFrame([row])
    row_df = row_df.reindex(columns=feature_cols, fill_value=0)
    return row_df

def pretty_explain(prob):
    if prob >= 0.9:
        return "🚨 Very high risk — escalate for investigation."
    if prob >= 0.5:
        return "⚠️ Medium risk — review or queue for analyst."
    return "✅ Low risk — likely safe."

def load_sample_test(path="sample_test.csv"):
    # optional helper to load a sample row from a saved CSV of test set
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Load model + feature list
try:
    model, feature_cols = load_model_and_meta()
except Exception as e:
    st.error(f"Failed to load model or feature list. Ensure 'xgb_fraud_model.joblib' and 'model_features.json' are in the app folder. Error: {e}")
    st.stop()

st.title("Paysim Fraud Predictor — Demo")
st.markdown("""
This demo predicts the probability that a transaction is fraudulent using a model trained on the PaySim synthetic dataset.
- Enter transaction details in the left panel, then press **Predict**.
- Alternatively upload a CSV with transactions (columns: amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, step, type).
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input transaction")
    amount = st.number_input("Amount", min_value=0.0, value=100.0, step=1.0, format="%.2f")
    old_org = st.number_input("Old balance (origin)", min_value=0.0, value=1000.0, step=1.0, format="%.2f")
    new_org = st.number_input("New balance (origin)", min_value=0.0, value=900.0, step=1.0, format="%.2f")
    old_dest = st.number_input("Old balance (dest)", min_value=0.0, value=500.0, step=1.0, format="%.2f")
    new_dest = st.number_input("New balance (dest)", min_value=0.0, value=600.0, step=1.0, format="%.2f")
    step = st.number_input("Step (sim time)", min_value=0, value=100, step=1)
    tx_type = st.selectbox("Transaction type", options=[c.split("type_",1)[1] for c in feature_cols if c.startswith("type_")])

    if st.button("Predict"):
        # Build features and predict
        sample_df = build_feature_row(amount, old_org, new_org, old_dest, new_dest, step, tx_type, feature_cols)
        prob = model.predict_proba(sample_df.values.reshape(1, -1))[:,1][0]
        st.metric("Predicted fraud probability", f"{prob:.4f}")
        st.write(pretty_explain(prob))

with col2:
    st.header("Model insights")
    st.write("Top features used by the model (approximate importance):")
    try:
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
        st.bar_chart(feat_imp)
    except Exception:
        st.info("Feature importances not available for this model.")

    st.markdown("### Quick CSV prediction")
    uploaded = st.file_uploader("Upload CSV (optional). Columns required: amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, step, type", type=['csv'])
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            # Build feature matrix
            rows = []
            for _, r in df_upload.iterrows():
                rows.append(build_feature_row(
                    amount=float(r['amount']),
                    old_org=float(r['oldbalanceOrg']),
                    new_org=float(r['newbalanceOrig']),
                    old_dest=float(r['oldbalanceDest']),
                    new_dest=float(r['newbalanceDest']),
                    step=int(r.get('step', 0)),
                    tx_type=str(r['type']),
                    feature_cols=feature_cols
                ).iloc[0])
            X_batch = pd.DataFrame(rows)[feature_cols]  # ensure columns order
            probs = model.predict_proba(X_batch.values)[:,1]
            df_upload['pred_fraud_prob'] = probs
            st.dataframe(df_upload.head(50))
            st.download_button("Download predictions CSV", df_upload.to_csv(index=False).encode('utf-8'), file_name="predictions.csv")
        except Exception as e:
            st.error(f"Could not process uploaded file. Error: {e}")

st.markdown("---")
st.caption("Model demo — ensure that the model and feature list were generated with the same preprocessing pipeline as this app.")
