import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "churn_model.pkl"
DATA_PATH = BASE / "data" / "customers.csv"

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

st.title("📊 Customer Churn Prediction & Analytics")
st.caption("Python • Pandas • Scikit-learn • Streamlit • MySQL-ready")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Customer Prediction", "EDA", "Data Explorer"])

if page == "Dashboard":
    st.subheader("Business Overview")
    c1,c2,c3,c4 = st.columns(4)
    churn_rate = (df["churn"]=="Yes").mean()*100
    c1.metric("Customers", len(df))
    c2.metric("Churn Rate", f"{churn_rate:.1f}%")
    c3.metric("Avg Monthly Charges", f"₹{df['monthly_charges'].mean():.2f}")
    c4.metric("Avg Tenure", f"{df['tenure_months'].mean():.1f} months")

    st.divider()
    left,right=st.columns(2)
    with left:
        st.write("### Churn by Contract")
        st.bar_chart(pd.crosstab(df["contract"], df["churn"]))
    with right:
        st.write("### Churn by Internet Service")
        st.bar_chart(pd.crosstab(df["internet_service"], df["churn"]))

    st.write("### High-risk customer sample")
    X = df.drop(columns=["churn"])
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, list(model.classes_).index("Yes")]
    display = df.copy()
    display["churn_probability"] = (proba*100).round(1)
    display["risk"] = pd.cut(display["churn_probability"], [-1,35,65,101],
                             labels=["Low","Medium","High"])
    st.dataframe(display.sort_values("churn_probability", ascending=False).head(15),
                 use_container_width=True)

elif page == "Customer Prediction":
    st.subheader("🤖 Predict Churn for a Customer")
    col1,col2,col3=st.columns(3)
    with col1:
        gender=st.selectbox("Gender",["Male","Female"])
        senior=st.selectbox("Senior Citizen",[0,1])
        partner=st.selectbox("Partner",["Yes","No"])
        dependents=st.selectbox("Dependents",["Yes","No"])
        tenure=st.slider("Tenure (months)",1,72,12)
    with col2:
        phone=st.selectbox("Phone Service",["Yes","No"])
        internet=st.selectbox("Internet Service",["DSL","Fiber optic","No"])
        contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
        monthly=st.number_input("Monthly Charges",20.0,150.0,70.0)
    with col3:
        payment=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer","Credit card"])
        tech=st.selectbox("Tech Support",["Yes","No"])
        security=st.selectbox("Online Security",["Yes","No"])
        total=st.number_input("Total Charges",0.0,10000.0,float(round(monthly*tenure,2)))

    if st.button("Predict Churn", type="primary"):
        sample=pd.DataFrame([{
            "customer_id":999999,"gender":gender,"senior_citizen":senior,
            "partner":partner,"dependents":dependents,"tenure_months":tenure,
            "phone_service":phone,"internet_service":internet,"contract":contract,
            "monthly_charges":monthly,"total_charges":total,"payment_method":payment,
            "tech_support":tech,"online_security":security
        }])
        prediction=model.predict(sample)[0]
        prob=model.predict_proba(sample)[0, list(model.classes_).index("Yes")]
        if prob>=0.65:
            st.error(f"High Risk — churn probability: {prob*100:.1f}%")
        elif prob>=0.35:
            st.warning(f"Medium Risk — churn probability: {prob*100:.1f}%")
        else:
            st.success(f"Low Risk — churn probability: {prob*100:.1f}%")
        st.progress(float(prob), text="Churn probability")

elif page == "EDA":
    st.subheader("📈 Exploratory Data Analysis")
    numeric=["tenure_months","monthly_charges","total_charges"]
    st.write("### Numeric summary")
    st.dataframe(df[numeric].describe(), use_container_width=True)
    st.write("### Monthly Charges Distribution")
    st.bar_chart(df["monthly_charges"].round(-1).value_counts().sort_index())
    st.write("### Churn by Contract")
    st.bar_chart(pd.crosstab(df["contract"], df["churn"]))

elif page == "Data Explorer":
    st.subheader("🔎 Customer Data")
    search=st.text_input("Search Customer ID")
    view=df
    if search:
        try:
            view=df[df.customer_id.astype(str).str.contains(search)]
        except Exception:
            pass
    st.dataframe(view, use_container_width=True)
    st.download_button("Download CSV", view.to_csv(index=False), "customers_filtered.csv", "text/csv")
