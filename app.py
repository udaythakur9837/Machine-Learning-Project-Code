from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Loan Approval Prediction System")
st.write(
    "A machine-learning application that predicts loan approval status "
    "from applicant financial and credit information."
)


# -----------------------------
# Load and clean dataset
# -----------------------------
@st.cache_data
def load_and_clean_data(file_source):
    df = pd.read_csv(file_source)

    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    if "Dependents" in df.columns:
        df["Dependents"] = (
            df["Dependents"].astype(str).str.replace("3+", "3", regex=False)
        )
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")

    categorical_columns = [
        "Gender", "Married", "Education", "Self_Employed", "Property_Area"
    ]
    numeric_columns = [
        "Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History",
    ]

    for col in categorical_columns:
        if col in df.columns:
            mode = df[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value).astype(str)

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    required_columns = categorical_columns + numeric_columns + ["Loan_Status"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    df["Dependents"] = df["Dependents"].astype(int)
    df["Credit_History"] = df["Credit_History"].astype(int)

    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]:
        df[col] = df[col].astype(float)

    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    if df["Loan_Status"].dtype == "object":
        df["Loan_Status"] = (
            df["Loan_Status"].astype(str).str.strip().map({"Y": 1, "N": 0})
        )

    df = df.dropna(subset=["Loan_Status"])
    df["Loan_Status"] = df["Loan_Status"].astype(int)

    return df


# Load local dataset or allow CSV upload
dataset_path = Path("data") / "loan_data.csv"

if dataset_path.exists():
    try:
        df = load_and_clean_data(dataset_path)
    except Exception as error:
        st.error(f"Could not load the dataset: {error}")
        st.stop()
else:
    st.info("Upload your loan dataset CSV to get started.")
    uploaded_file = st.file_uploader("Upload loan_data.csv", type=["csv"])
    if uploaded_file is None:
        st.stop()
    try:
        df = load_and_clean_data(uploaded_file)
    except Exception as error:
        st.error(f"Could not process the uploaded dataset: {error}")
        st.stop()


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Model Configuration")

algorithm = st.sidebar.selectbox(
    "Choose ML Algorithm",
    ["Random Forest Classifier", "Logistic Regression"],
)

test_size = st.sidebar.slider(
    "Test Data Split (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
) / 100

if algorithm == "Random Forest Classifier":
    n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 150, step=50)
    max_depth = st.sidebar.slider("Maximum Tree Depth", 2, 12, 5)
else:
    n_estimators = None
    max_depth = None

st.sidebar.markdown("---")
st.sidebar.header("🧾 Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
credit_history = st.sidebar.selectbox("Credit History (1 = Good, 0 = Bad)", [1, 0])
applicant_income = st.sidebar.number_input(
    "Applicant Monthly Income", min_value=0, max_value=100000, value=5000, step=500
)
coapplicant_income = st.sidebar.number_input(
    "Co-applicant Monthly Income", min_value=0, max_value=100000, value=1500, step=500
)
loan_amount = st.sidebar.number_input(
    "Loan Amount (in thousands)", min_value=10, max_value=1000, value=140, step=10
)
loan_term = st.sidebar.selectbox("Loan Term (in months)", [360, 180, 240, 120, 60])


# -----------------------------
# Prepare data and train model
# -----------------------------
feature_columns = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area", "Total_Income",
]

categorical_features = [
    "Gender", "Married", "Education", "Self_Employed", "Property_Area"
]
numeric_features = [
    "Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Total_Income",
]

X = df[feature_columns]
y = df["Loan_Status"]

if y.nunique() < 2:
    st.error("The dataset must contain both approved and rejected loan records.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
except ValueError as error:
    st.error(f"Unable to split the dataset: {error}")
    st.stop()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ]
)

if algorithm == "Random Forest Classifier":
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
else:
    classifier = LogisticRegression(max_iter=1000, random_state=42)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier),
])

model.fit(X_train, y_train)


# -----------------------------
# Model evaluation
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred) * 100
try:
    auc_score = roc_auc_score(y_test, y_prob)
except ValueError:
    auc_score = float("nan")


# -----------------------------
# Live prediction
# -----------------------------
input_data = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Dependents": int(dependents),
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": float(applicant_income),
    "CoapplicantIncome": float(coapplicant_income),
    "LoanAmount": float(loan_amount),
    "Loan_Amount_Term": float(loan_term),
    "Credit_History": int(credit_history),
    "Property_Area": property_area,
    "Total_Income": float(applicant_income + coapplicant_income),
}])

prediction = model.predict(input_data)[0]
approval_probability = model.predict_proba(input_data)[0][1] * 100

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Dataset Rows", len(df))
metric2.metric("Model Accuracy", f"{accuracy:.2f}%")
metric3.metric("ROC-AUC Score", f"{auc_score:.3f}")
metric4.metric("Approval Probability", f"{approval_probability:.1f}%")

st.markdown("---")
st.subheader("🎯 Live Loan Prediction Result")

if prediction == 1:
    st.success(f"**Predicted: LOAN APPROVED** — Probability: {approval_probability:.1f}%")
else:
    st.error(f"**Predicted: LOAN NOT APPROVED** — Approval probability: {approval_probability:.1f}%")

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Rejected (0)", "Actual: Approved (1)"],
        columns=["Predicted: Rejected (0)", "Predicted: Approved (1)"],
    )
    st.dataframe(cm_df, use_container_width=True)

with right_column:
    st.subheader("🔍 Key Decision Factors")
    if algorithm == "Random Forest Classifier":
        encoded_columns = model.named_steps["preprocessor"].get_feature_names_out()
        clean_names = [
            name.replace("num__", "").replace("cat__", "") for name in encoded_columns
        ]
        importances = model.named_steps["classifier"].feature_importances_
        importance_df = (
            pd.DataFrame({"Feature": clean_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(7)
        )
        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.info("Feature importance chart is available with Random Forest.")

with st.expander("📂 Preview Cleaned Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

st.caption(
    "Educational project demo only. Predictions are not financial advice "
    "or actual loan decisions."
)
