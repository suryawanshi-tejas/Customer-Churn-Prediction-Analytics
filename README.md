# Customer Churn Prediction & Analytics System

A resume-ready end-to-end Python project using Pandas, Scikit-learn, Streamlit and MySQL.

## Features
- Data preprocessing
- Exploratory data analysis
- Random Forest churn prediction
- Probability-based risk classification
- Interactive Streamlit dashboard
- MySQL-ready database integration
- SQL analytics queries

## Run locally

### 1. Open terminal in this folder
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

This creates `models/churn_model.pkl`.

### 4. Start the application
```bash
streamlit run app.py
```

Open the URL shown by Streamlit.

## Optional MySQL setup

1. Install and start MySQL.
2. Open `database.sql` in MySQL Workbench and create the database.
3. Edit the password in `database.py`.
4. Run:
```bash
python database.py
```

## Project flow

CSV → Pandas preprocessing → Random Forest → churn probability → Streamlit dashboard.

## Resume
Project title: Customer Churn Prediction & Analytics System

Suggested stack: Python, Pandas, NumPy, Scikit-learn, Streamlit, MySQL, SQL, Matplotlib.
