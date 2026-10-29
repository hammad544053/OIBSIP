<div align="center">

  <img src="static/images/oasis_logo.png" alt="Oasis Infobyte Logo" width="120" />

  # 🌸 Iris Flower Classification Web App

  **Oasis Infobyte Virtual Internship (OIBSIP) • Data Science Task 1**

  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
  [![Flask](https://img.shields.io/badge/Flask-3.0.2-000000.svg?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F79A3E.svg?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
  [![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.0-38B2AC.svg?style=for-the-badge&logo=tailwind-css)](https://tailwindcss.com/)

  <p align="center">
    A machine learning web application featuring a modern cyber-themed UI built with Flask, Scikit-Learn, and Tailwind CSS to predict Iris flower species from physical measurements.
  </p>

</div>

---

## 📌 Project Overview

This project implements an end-to-end Machine Learning pipeline to classify Iris flowers into three species (**Setosa**, **Versicolor**, or **Virginica**) using physical measurements ($cm$):
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

The underlying model uses a **K-Nearest Neighbors (KNN)** pipeline trained on the classic Iris dataset, wrapped with standard preprocessing scales, and deployed via a dynamic Flask web application.

---

## 🛠️ Tech Stack & Tools

- **Machine Learning:** `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Backend Framework:** `Flask`, `Gunicorn`
- **Frontend UI:** `HTML5`, `Tailwind CSS`, `Custom Cyber CSS`, `JavaScript (Fetch API)`
- **Environment & Hosting:** `Python 3.10+`, `Git`, `GitHub`

---

## 📂 Folder Structure

```text
OIBSIP/
└── DataScience-Task1-IrisFlowerClassification/
    ├── app.py                          # Main Flask Application
    ├── iris_classification_pipeline.joblib # Saved ML Model & Preprocessing Pipeline
    ├── Iris_Classification.ipynb       # Jupyter Notebook (EDA & Model Training)
    ├── requirements.txt                # Python Dependencies
    ├── Procfile                        # Production Deployment Commands
    ├── README.md                       # Documentation
    ├── static/
    │   ├── css/
    │   │   └── style.css               # Cyber Theme Styling
    │   └── images/                     # Oasis Logo & Flower Visuals
    └── templates/
        └── index.html                  # Glassmorphism Frontend Layout