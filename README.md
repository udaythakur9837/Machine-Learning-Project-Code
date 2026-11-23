# Loan Approval Prediction System

A Streamlit machine-learning dashboard that predicts whether a loan application may be approved based on applicant details and credit information.

## Features

- Upload a CSV dataset or load `data/loan_data.csv`
- Clean missing values and prepare categorical/numerical features
- Select Random Forest or Logistic Regression
- Adjust train/test split
- Random Forest controls: number of trees and maximum depth
- Live prediction using applicant inputs
- Model metrics: accuracy and ROC-AUC
- Confusion matrix
- Random Forest feature-importance chart
- Cleaned dataset preview

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## Project Structure

```text
loan-approval-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Create and activate a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Put your dataset CSV at:

```text
data/loan_data.csv
```

Or launch the app and upload a CSV through the file uploader.

The dataset should include these columns:

- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`
- `Loan_Status` (Y/N or 1/0)

### 5. Run the app

```bash
streamlit run app.py
```

Streamlit will show a local URL in the terminal, usually `http://localhost:8501`.

## Disclaimer

This project is for educational and demonstration purposes only. It should not be used to make real lending decisions.
