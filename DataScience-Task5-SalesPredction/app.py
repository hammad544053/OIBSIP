import json
import math
from pathlib import Path
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_sales_prediction_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

if not MODEL_PATH.exists():
  raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("Sales prediction model loaded successfully!")

if METADATA_PATH.exists():
  with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)
else:
  metadata = {}


@app.route("/", methods=["GET", "POST"])
def home():
  prediction = None
  error = None
  tv = ""
  radio = ""
  newspaper = ""
  submitted_budgets = None

  model_name = metadata.get("model_name", "Polynomial Regression (Degree 2)")
  r2_score = metadata.get("test_r2", "98.69%")
  rmse_score = metadata.get("test_rmse", "0.6426")

  if request.method == "POST":
    try:
      tv = request.form.get("tv", "").strip()
      radio = request.form.get("radio", "").strip()
      newspaper = request.form.get("newspaper", "").strip()

      tv_value = float(tv)
      radio_value = float(radio)
      newspaper_value = float(newspaper)

      values = [tv_value, radio_value, newspaper_value]

      if not all(math.isfinite(v) for v in values):
        raise ValueError("Please enter valid finite numbers.")

      if any(v < 0 for v in values):
        raise ValueError("Advertising budgets cannot be negative.")

      input_data = pd.DataFrame({
          "tv": [tv_value],
          "radio": [radio_value],
          "newspaper": [newspaper_value],
      })

      predicted_sales = model.predict(input_data)
      prediction = round(float(predicted_sales[0]), 2)

      submitted_budgets = {
          "tv": tv_value,
          "radio": radio_value,
          "newspaper": newspaper_value,
          "total": round(tv_value + radio_value + newspaper_value, 2),
      }

    except ValueError as e:
      error = str(e)
    except Exception:
      error = (
          "An unexpected error occurred. Please check your input and try again."
      )

  return render_template(
      "index.html",
      prediction=prediction,
      error=error,
      tv=tv,
      radio=radio,
      newspaper=newspaper,
      submitted_budgets=submitted_budgets,
      model_name=model_name,
      r2_score=r2_score,
      rmse_score=rmse_score,
  )


if __name__ == "__main__":
  app.run(debug=True, host="127.0.0.1", port=5000)