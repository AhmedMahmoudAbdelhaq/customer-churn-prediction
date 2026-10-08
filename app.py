import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #312e81);
}

.title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 35px;
}

.section-title {
    color: #ffffff;
    font-size: 23px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
}

[data-testid="stNumberInput"] input {
    color: #111827 !important;
    background-color: #ffffff !important;
    font-weight: 600 !important;
}

[data-baseweb="select"] {
    background-color: #ffffff !important;
}

[data-baseweb="select"] div {
    color: #111827 !important;
    font-weight: 600 !important;
}

.stButton button {
    width: 100%;
    height: 58px;
    border: none;
    border-radius: 14px;
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed,
        #db2777
    );
    color: #ffffff !important;
    font-size: 18px;
    font-weight: 700;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35);
}

.stButton button:hover {
    background: linear-gradient(
        90deg,
        #7c3aed,
        #db2777
    );
    color: #ffffff !important;
}

.result {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
}

.low {
    background: linear-gradient(
        135deg,
        #047857,
        #059669
    );
    border: 2px solid #34d399;
}

.high {
    background: linear-gradient(
        135deg,
        #b91c1c,
        #dc2626
    );
    border: 2px solid #f87171;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff !important;
}

.result-text {
    color: #ffffff !important;
    font-size: 16px;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
}

[data-testid="stAlert"] {
    background: rgba(255, 255, 255, 0.12) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

[data-testid="stAlert"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    padding: 15px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🏦 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning system for predicting bank customer churn'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        300,
        850,
        650
    )

    age = st.number_input(
        "Age",
        18,
        100,
        40
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

with col3:
    tenure = st.number_input(
        "Tenure",
        0,
        10,
        5
    )

    products = st.number_input(
        "Number of Products",
        1,
        4,
        1
    )

st.markdown(
    '<div class="section-title">💰 Financial Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    balance = st.number_input(
        "Account Balance",
        0.0,
        300000.0,
        50000.0
    )

with col2:
    salary = st.number_input(
        "Estimated Salary",
        0.0,
        250000.0,
        50000.0
    )

st.markdown(
    '<div class="section-title">🏦 Banking Activity</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

with col2:
    active = st.selectbox(
        "Active Member?",
        ["Yes", "No"]
    )

st.write("")

predict = st.button(
    "🔮 Predict Customer Risk",
    use_container_width=True
)

if predict:

    data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [0 if gender == "Female" else 1],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [products],
        "HasCrCard": [1 if card == "Yes" else 0],
        "IsActiveMember": [1 if active == "Yes" else 0],
        "EstimatedSalary": [salary],
        "Geography_Germany": [
            1 if geography == "Germany" else 0
        ],
        "Geography_Spain": [
            1 if geography == "Spain" else 0
        ]
    })

    numerical = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary"
    ]

    data[numerical] = scaler.transform(
        data[numerical]
    )

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    st.divider()

    if prediction == 1:

        st.markdown("""
        <div class="result high">
            <div class="result-title">
                🔴 High Churn Risk
            </div>
            <div class="result-text">
                This customer is likely to leave the bank.
            </div>
        </div>
        """, unsafe_allow_html=True)

        result = "Churn"
        risk = "High"

        recommendation = (
            "Consider offering the customer a retention "
            "offer or personalized service."
        )

    else:

        st.markdown("""
        <div class="result low">
            <div class="result-title">
                🟢 Low Churn Risk
            </div>
            <div class="result-text">
                This customer is likely to stay with the bank.
            </div>
        </div>
        """, unsafe_allow_html=True)

        result = "Stay"
        risk = "Low"

        recommendation = (
            "The customer currently shows low churn risk. "
            "Continue providing good service."
        )

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prediction",
        result
    )

    col2.metric(
        "Churn Probability",
        f"{probability:.1%}"
    )

    col3.metric(
        "Risk Level",
        risk
    )

    st.info(
        f"💡 Recommendation: {recommendation}"
    )

st.markdown("""
<div class="footer">
    🏦 Customer Churn Prediction |
    Machine Learning Project |
    KNN Classification
</div>
""", unsafe_allow_html=True)