# Used Car Price Prediction with Machine Learning

**Oasis Infobyte Internship (OIBSIP)**  
**Track:** Data Science  
**Task:** Task 3 — Car Price Prediction with Machine Learning  
**Author:** Hammad Ullah  

---

## 📌 Executive Project Summary
This project delivers an end-to-end machine learning system that estimates the resale market value of used cars based on vehicle specifications, age, brand, and usage history. Using the CarDekho dataset (8,128 original rows), 1,202 duplicate records were removed, leaving 6,926 unique records for modeling. 

Features like mileage, engine capacity, and maximum power were extracted from text strings into clean continuous metrics. The project compares multiple algorithms (Linear Regression, Extra Trees, Random Forest) and implements a Random Forest Regressor pipeline achieving an **R² score of 0.932** and a **Test MAE of ₹71,620.89**.

---

## 📊 Model Benchmarks & Metrics

| Model Stage | Test MAE (₹) | Test RMSE (₹) | Test R² Score |
| :--- | :---: | :---: | :---: |
| **Median Baseline** | ₹259,546.73 | ₹478,963.27 | -0.046 |
| **Linear Regression** | ₹133,059.85 | ₹261,886.82 | 0.687 |
| **Extra Trees Regressor** | ₹75,776.48 | ₹154,414.17 | 0.891 |
| **Random Forest Regressor (Selected)** | **₹71,620.89** | **₹121,710.68** | **0.932** |

---

## 📁 Subfolder Directory Layout
```text
DataScience-Task3-CarPricePrediction/
├── app.py
├── requirements.txt
├── README.md
├── Car details v3.csv
├── car_price_prediction_model.joblib
├── car_price_model_metadata.json
├── Task_3_Car_Price_Prediction.ipynb
├── Oasis_Infobyte_Task_3_Car_Price_Prediction_Report.docx
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/
│       └── logo.jpg
└── templates/
    └── index.html