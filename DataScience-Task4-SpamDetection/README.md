# Email Spam Detection with Machine Learning (NLP)

**Oasis Infobyte Internship (OIBSIP)**  
**Track:** Data Science  
**Task:** Task 4 — Email Spam Detection with Machine Learning  
**Author:** Hammad Ullah  

---

## 📌 Project Overview
This project constructs a Natural Language Processing (NLP) binary classification pipeline to identify spam messages. Using a TF-IDF vectorizer paired with a Logistic Regression classifier, the system achieves **96.51% overall accuracy** and **100% precision** (0 False Positives on the test dataset).

---

## 📊 Model Evaluation Summary

| Classifier Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 96.32% | 100.0% | 70.31% | 0.8257 |
| **Logistic Regression (Selected)** | **96.51%** | **100.0%** | **71.88%** | **0.8364** |

---

## 📁 Subfolder Directory Structure
```text
DataScience-Task4-SpamDetection/
├── app.py
├── requirements.txt
├── README.md
├── spam_classifier_pipeline.joblib
├── Oasis_Infobyte_Task_4_Email_Spam_Detection.ipynb
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/
│       └── logo.jpg
└── templates/
    └── index.html