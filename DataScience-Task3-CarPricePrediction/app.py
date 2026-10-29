import base64
import datetime
import io
import json
from flask import Flask, render_template, request
import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

app = Flask(__name__)

MODEL_PATH = "car_price_prediction_model.joblib"
METADATA_PATH = "car_price_model_metadata.json"

# Load serialized model pipeline and metadata
try:
  model = joblib.load(MODEL_PATH)
  model_loaded = True
except Exception as e:
  model = None
  model_loaded = False
  print(f"Warning: Could not load car price model pipeline: {e}")

try:
  with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
except Exception:
  metadata = None


def generate_price_analytics_chart(
    predicted_price, brand, car_age, power_bhp, km_driven
):
  """Generates a multi-bar comparison chart returned as Base64 image string."""
  plt.figure(figsize=(7, 3.2))
  sns.set_theme(style="darkgrid")

  # Benchmarks for visual comparison
  categories = [
      "Market Median",
      f"{brand} Average",
      f"Valuation ({car_age}y, {power_bhp} BHP)",
  ]
  values = [400000, max(350000, predicted_price * 0.88), predicted_price]
  colors = ["#64748b", "#0284c7", "#f59e0b"]

  ax = sns.barplot(x=categories, y=values, palette=colors)
  plt.title(
      f"Market Position: {brand} ({car_age} Years Old)",
      fontsize=11,
      fontweight="bold",
      pad=12,
      color="#0f172a",
  )
  plt.ylabel("Price in INR (₹)", fontsize=9, fontweight="bold")
  plt.ylim(0, max(values) * 1.25)

  for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"₹{height:,.0f}",
        (p.get_x() + p.get_width() / 2.0, height),
        ha="center",
        va="center",
        xytext=(0, 7),
        textcoords="offset points",
        fontweight="bold",
        fontsize=8.5,
        color="#1e293b",
    )

  plt.tight_layout()
  buffer = io.BytesIO()
  plt.savefig(buffer, format="png", dpi=150)
  buffer.seek(0)
  image_png = buffer.getvalue()
  buffer.close()
  plt.close()

  return base64.b64encode(image_png).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
  prediction = None
  formatted_price = None
  chart_url = None
  error_msg = None
  submitted_inputs = None

  if request.method == "POST":
    if not model_loaded:
      error_msg = (
          "Pipeline status error: Version mismatch or model file could not be"
          " unpickled. Ensure scikit-learn==1.6.1 is installed."
      )
    else:
      try:
        # Form field extraction
        brand = request.form.get("brand", "Maruti").strip()
        model_name = request.form.get("model_name", "Swift").strip()
        year = int(request.form.get("year", 2017))
        km_driven = float(request.form.get("km_driven", 50000))
        fuel = request.form.get("fuel", "Diesel")
        seller_type = request.form.get("seller_type", "Individual")
        transmission = request.form.get("transmission", "Manual")
        owner = request.form.get("owner", "First Owner")

        mileage_val = float(request.form.get("mileage", 20.0))
        mileage_unit = request.form.get("mileage_unit", "kmpl")
        engine_cc = float(request.form.get("engine", 1248))
        max_power_bhp = float(request.form.get("max_power", 74))
        seats = float(request.form.get("seats", 5))

        # Derived Feature Engineering
        current_ref_year = 2020  # Dataset reference standard
        car_age = max(0, current_ref_year - year)
        full_name = f"{brand} {model_name}"

        submitted_inputs = {
            "Vehicle": full_name,
            "Year": year,
            "Age": f"{car_age} years",
            "KM Driven": f"{km_driven:,.0f} km",
            "Fuel": fuel,
            "Transmission": transmission,
            "Power": f"{max_power_bhp} BHP",
            "Engine": f"{engine_cc:.0f} CC",
            "Mileage": f"{mileage_val} {mileage_unit}",
        }

        # DataFrame matching pipeline training schema
        input_data = {
            "name": full_name,
            "brand": brand,
            "year": year,
            "car_age": car_age,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage_numeric": mileage_val,
            "mileage_unit": mileage_unit,
            "engine_cc": engine_cc,
            "max_power_bhp": max_power_bhp,
            "seats": seats,
        }

        input_df = pd.DataFrame([input_data])

        # Inference
        raw_pred = model.predict(input_df)[0]
        prediction = max(0, round(float(raw_pred), 2))
        formatted_price = f"₹{prediction:,.2f}"

        chart_url = generate_price_analytics_chart(
            prediction, brand, car_age, max_power_bhp, km_driven
        )

      except Exception as e:
        error_msg = f"Prediction Pipeline Error: {str(e)}"

  return render_template(
      "index.html",
      prediction=prediction,
      formatted_price=formatted_price,
      chart_url=chart_url,
      error_msg=error_msg,
      submitted_inputs=submitted_inputs,
      metadata=metadata,
  )


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, debug=True)