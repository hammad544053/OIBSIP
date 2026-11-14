# AdClick & Sales Conversion Prediction (AdClick AI)

**Oasis Infobyte Internship (OIBSIP)**  
**Track:** Data Science  
**Task:** Task 5 — Sales & Advertisement Click Prediction with Machine Learning  
**Author:** Hammad Ullah  

---

## 📌 Project Overview
**AdClick AI** is an end-to-end Machine Learning web application that predicts whether an online visitor will click on an advertisement and convert based on demographic, behavioral, and timestamp-derived session features. 

The application pairs a serialized Scikit-Learn classification and scaling pipeline (`best_advertising_model.pkl`) with a responsive dark-mode Flask analytics dashboard featuring live **Chart.js** conversion trends and persistent CSV prediction logging.

---

## 🛠️ Tech Stack
* **Machine Learning & Data Processing:** Python, Pandas, Scikit-Learn (`StandardScaler`, Binary Classification), Joblib
* **Backend Web Framework:** Flask
* **Frontend & Data Visualization:** HTML5, CSS3 (Custom Dark-Navy Responsive UI), JavaScript, Chart.js

---

## 🧠 Model Input Schema
The serialized model bundle (`models/best_advertising_model.pkl`) enforces exact feature alignment and applies the fitted `StandardScaler` prior to inference:

1. `Daily Time Spent on Site` (minutes)
2. `Age` (years)
3. `Area Income` (regional household income)
4. `Daily Internet Usage` (minutes per day)
5. `Male` (`1` = Male, `0` = Female)
6. `Year` (visit year)
7. `Month` (visit month, `1–12`)
8. `Day` (visit day, `1–31`)
9. `Hour` (visit hour in 24-hour format, `0–23`)

---

## ✨ Key Application Features
* **Interactive Analytics Dashboard (`/`):** Displays KPI cards (Total Analyses, Likely Clicks, No-Clicks, Average Click Probability), a **Chart.js** Click Probability Trend line chart, a Conversion Split doughnut chart, and a recent prediction history table.
* **Real-Time Inference Engine (`/predict`):** Validates all 9 visitor inputs, scales features automatically, calculates class probability (`predict_proba`), and renders an instant confidence gauge alongside a profile summary.
* **Model Intelligence Modal:** Built-in architectural overview displaying the active classifier, reported F1-score, scaler details, and expected feature schema.
* **Persistent CSV Logging:** Automatically records every evaluation to `prediction_history.csv` with one-click log clearing.

---

## 📁 Project Structure
```text
DataScience-Task5-SalesPrediction/
├── app.py
├── requirements.txt
├── README.md
├── prediction_history.csv
├── models/
│   └── best_advertising_model.pkl
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
└── templates/
    ├── base.html
    ├── index.html
    └── predict.html