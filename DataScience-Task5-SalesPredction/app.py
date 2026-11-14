import csv
from datetime import datetime
from pathlib import Path
from flask import Flask, flash, redirect, render_template, request, url_for
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "best_advertising_model.pkl"
HISTORY_PATH = BASE / "prediction_history.csv"

app = Flask(__name__)
app.secret_key = "oibsip-task5-adclick-secret-key"

if not MODEL_PATH.exists():
  raise FileNotFoundError(f"Place your updated PKL bundle at: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
model, scaler = bundle["model"], bundle["scaler"]
FEATURES = bundle["feature_names"]
MODEL_NAME = bundle.get("model_name", type(model).__name__)
MODEL_F1 = float(bundle.get("f1_score", 0))

EXPECTED = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
    "Year",
    "Month",
    "Day",
    "Hour",
]
if list(FEATURES) != EXPECTED:
  raise ValueError(f"Unexpected feature order: {FEATURES}")

COLUMNS = [
    "timestamp",
    "age",
    "gender",
    "daily_time",
    "area_income",
    "daily_internet",
    "year",
    "month",
    "day",
    "hour",
    "prediction",
    "click_probability",
]


def history_rows():
  if not HISTORY_PATH.exists():
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
      csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
  with HISTORY_PATH.open("r", newline="", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def save_row(row):
  history_rows()
  with HISTORY_PATH.open("a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)


def get_inputs(form):
  try:
    v = {
        "daily_time": float(form["daily_time"]),
        "age": float(form["age"]),
        "area_income": float(form["area_income"]),
        "daily_internet": float(form["daily_internet"]),
        "gender": int(form["gender"]),
        "year": int(form["year"]),
        "month": int(form["month"]),
        "day": int(form["day"]),
        "hour": int(form["hour"]),
    }
  except (KeyError, TypeError, ValueError):
    raise ValueError("Please enter valid numeric values in every field.")
  if not 0 <= v["daily_time"] <= 1440:
    raise ValueError("Daily time must be 0–1440 minutes.")
  if not 1 <= v["age"] <= 110:
    raise ValueError("Age must be between 1 and 110.")
  if v["area_income"] < 0:
    raise ValueError("Area income cannot be negative.")
  if not 0 <= v["daily_internet"] <= 1440:
    raise ValueError("Internet usage must be 0–1440 minutes.")
  if v["gender"] not in (0, 1):
    raise ValueError("Select Female (0) or Male (1).")
  if not 2000 <= v["year"] <= 2100:
    raise ValueError("Enter a valid year.")
  if not 1 <= v["month"] <= 12:
    raise ValueError("Month must be 1–12.")
  if not 1 <= v["day"] <= 31:
    raise ValueError("Day must be 1–31.")
  if not 0 <= v["hour"] <= 23:
    raise ValueError("Hour must be 0–23.")
  return v


@app.route("/")
def index():
  rows = history_rows()
  clicked = sum(r["prediction"] == "1" for r in rows)
  not_clicked = len(rows) - clicked
  recent = list(reversed(rows[-8:]))
  avg = (
      sum(float(r["click_probability"]) for r in rows) / len(rows)
      if rows
      else 0
  )

  # Data for Dashboard Charts (last 10 predictions in chronological order)
  chart_slice = rows[-10:]
  chart_labels = [
      r["timestamp"].split(" ")[1][:5] if " " in r["timestamp"] else f"#{i+1}"
      for i, r in enumerate(chart_slice)
  ]
  chart_probs = [
      round(float(r["click_probability"]) * 100, 1) for r in chart_slice
  ]

  return render_template(
      "index.html",
      active="dashboard",
      total=len(rows),
      clicked=clicked,
      not_clicked=not_clicked,
      avg=avg,
      recent=recent,
      chart_labels=chart_labels,
      chart_probs=chart_probs,
      model_name=MODEL_NAME,
      model_f1=MODEL_F1,
      features=FEATURES,
  )


@app.route("/predict", methods=["GET", "POST"])
def predict():
  now = datetime.now()
  if request.method == "GET":
    return render_template(
        "predict.html",
        active="predict",
        now=now,
        model_name=MODEL_NAME,
        model_f1=MODEL_F1,
        features=FEATURES,
    )
  try:
    v = get_inputs(request.form)
    values = [
        v["daily_time"],
        v["age"],
        v["area_income"],
        v["daily_internet"],
        v["gender"],
        v["year"],
        v["month"],
        v["day"],
        v["hour"],
    ]
    scaled = scaler.transform(pd.DataFrame([values], columns=FEATURES))
    pred = int(model.predict(scaled)[0])
    probs = (
        model.predict_proba(scaled)[0]
        if hasattr(model, "predict_proba")
        else [1 - pred, pred]
    )
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
    click_prob = float(probs[classes.index(1)]) if 1 in classes else 0.0

    row = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "age": v["age"],
        "gender": "Male" if v["gender"] == 1 else "Female",
        "daily_time": v["daily_time"],
        "area_income": v["area_income"],
        "daily_internet": v["daily_internet"],
        "year": v["year"],
        "month": v["month"],
        "day": v["day"],
        "hour": v["hour"],
        "prediction": pred,
        "click_probability": round(click_prob, 6),
    }
    save_row(row)

    return render_template(
        "predict.html",
        active="predict",
        now=now,
        details=v,
        prediction=pred,
        click_probability=click_prob,
        confidence=click_prob if pred == 1 else 1 - click_prob,
        model_name=MODEL_NAME,
        model_f1=MODEL_F1,
        features=FEATURES,
    )
  except ValueError as e:
    flash(str(e), "error")
    return redirect(url_for("predict"))


@app.route("/history/clear", methods=["POST"])
def clear_history():
  with HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
  flash("Prediction history cleared.", "success")
  return redirect(url_for("index"))


if __name__ == "__main__":
  history_rows()
  app.run(debug=True)