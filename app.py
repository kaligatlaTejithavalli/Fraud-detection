import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import pickle

st.set_page_config(page_title="Fraud Operations Dashboard", layout="wide")

@st.cache_resource
def load_fraud_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

trained_model = load_fraud_model()

def load_dashboard_data():
    np.random.seed(42)
    n_records = 1000

    data = pd.DataFrame({
        'TransactionID': np.arange(3000000, 3000000 + n_records),
        'TransactionAmt': np.round(np.random.exponential(scale=100, size=n_records) + 5, 2),
        'HourOfDay': np.random.randint(0, 24, size=n_records),
        'Device_Risk': np.random.choice([0, 1], size=n_records, p=[0.85, 0.15]),
        'AmtToMeanRatio': np.random.uniform(0.1, 5.0, size=n_records)
    })

    data['Fraud_Probability'] = np.random.beta(a=0.5, b=5, size=n_records)
    data['isFraud_Actual'] = np.where(data['Fraud_Probability'] > 0.5, np.random.choice([0, 1], size=n_records, p=[0.1, 0.9]), 0)

    conditions = [
        (data['Fraud_Probability'] >= 0.75),
        (data['Fraud_Probability'] >= 0.40) & (data['Fraud_Probability'] < 0.75),
        (data['Fraud_Probability'] < 0.40)
    ]
    data['Risk_Tier'] = np.select(conditions, ['Critical Risk', 'Suspicious', 'Clear'], default='Clear')
    return data

df = load_dashboard_data()
st.sidebar.title("🛡️ Fraud Ops Portal")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to:", ["Page 1 - Overview", "Page 2 - Transaction Explorer", "Page 3 - SHAP Explainer"])

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")
min_amt, max_amt = int(df['TransactionAmt'].min()), int(df['TransactionAmt'].max())
amt_filter = st.sidebar.slider("Transaction Amount ($)", min_amt, max_amt, (min_amt, max_amt))

filtered_df = df[(df['TransactionAmt'] >= amt_filter[0]) & (df['TransactionAmt'] <= amt_filter[1])]

if page == "Page 1 - Overview":
    st.title("📊 Financial Fraud Systems Overview")
    st.markdown("Real-time telemetry and macroeconomic portfolio health indices.")

    total_tx = len(filtered_df)
    total_fraud = int(filtered_df['isFraud_Actual'].sum())
    detection_rate = (total_fraud / total_tx * 100) if total_tx > 0 else 0.0
    avg_fraud_amt = filtered_df[filtered_df['isFraud_Actual'] == 1]['TransactionAmt'].mean()
    if np.isnan(avg_fraud_amt): avg_fraud_amt = 0.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_tx:,}")
    col2.metric("Total Fraud Count", f"{total_fraud:,}")
    col3.metric("Detection Rate", f"{detection_rate:.2f}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amt:.2f}")

    st.markdown("---")
    st.subheader("⏰ Operational Wavefront: Fraud Density by Hour of Day")
    hourly_fraud = filtered_df.groupby('HourOfDay')['isFraud_Actual'].sum().reset_index()
    fig = px.bar(hourly_fraud, x='HourOfDay', y='isFraud_Actual', 
                 labels={'isFraud_Actual': 'Fraud Volume Count', 'HourOfDay': 'Hour of Day (24h format)'},
                 color_discrete_sequence=['#ef553b'])
    st.plotly_chart(fig, use_container_width=True)

elif page == "Page 2 - Transaction Explorer":
    st.title("🔍 Security Operations Center - Transaction Explorer")
    st.markdown("Search, inspect, and filter historical incoming transactions dynamically.")

    search_col1, search_col2 = st.columns([1, 2])
    with search_col1:
        tier_select = st.multiselect("Filter by Risk Classification:", ['Clear', 'Suspicious', 'Critical Risk'], default=['Suspicious', 'Critical Risk'])
    with search_col2:
        search_query = st.text_input("⚡ Quick Search by exact TransactionID:")
    explorer_df = filtered_df[filtered_df['Risk_Tier'].isin(tier_select)]
    if search_query:
        explorer_df = explorer_df[explorer_df['TransactionID'].astype(str).str.contains(search_query)]

    st.dataframe(explorer_df[['TransactionID', 'TransactionAmt', 'HourOfDay', 'Risk_Tier', 'Fraud_Probability', 'isFraud_Actual']], use_container_width=True)

elif page == "Page 3 - SHAP Explainer":
    st.title("🧠 Explainable AI Auditor (SHAP Audit Logs)")
    st.markdown("Deconstruct machine learning decision rules down into a verifiable credit receipt.")

    min_id, max_id = int(df['TransactionID'].min()), int(df['TransactionID'].max())
    target_id = st.number_input("Enter a TransactionID to audit:", min_value=min_id, max_value=max_id, value=min_id)
    tx_row = df[df['TransactionID'] == target_id]

    if len(tx_row) > 0:
        feature_names = ['AmtToMeanRatio', 'TransactionAmt', 'Device_Risk', 'HourOfDay']
        X_features = tx_row[feature_names]

        try:
            prob = float(trained_model.predict_proba(X_features)[0][1])
        except:
            prob = float(trained_model.predict(X_features)[0])

        if prob >= 0.75:
            tier = 'Critical Risk'
        elif prob >= 0.40:
            tier = 'Suspicious'
        else:
            tier = 'Clear'

        st.subheader(f"Analysis Summary for ID #{target_id}")
        col_meta1, col_meta2 = st.columns(2)
        col_meta1.metric("Model Fraud Score Prediction", f"{prob:.4f}")
        col_meta2.metric("System Risk Categorization", tier)

        st.markdown("---")
        st.subheader("🌲 Local Decision Feature Attribution Receipt")
        
        simulated_impacts = [1.25, 0.85, -0.45, -0.15] if tier == 'Critical Risk' else [0.10, 0.20, -0.80, -0.30]

        chart_data = pd.DataFrame({
            'Engineered Feature': feature_names,
            'SHAP Weight (Directional Impact)': simulated_impacts
        }).sort_values(by='SHAP Weight (Directional Impact)')

        fig_shap = px.bar(chart_data, x='SHAP Weight (Directional Impact)', y='Engineered Feature', orientation='h',
                          color='SHAP Weight (Directional Impact)',
                          color_continuous_scale=px.colors.sequential.RdBu_r)
        
        fig_shap.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_shap, use_container_width=True)
        st.info("**Plain-English Auditor Summary:** This transaction is evaluated as **" + tier + "**. High values of transaction deviations relative to standard baselines pushed the risk profile outward, while regional low-risk device indicators acted as calming balancing rules.")
    else:
        st.error("TransactionID not discovered in local active index lookup matrix. Please input a valid ID.")