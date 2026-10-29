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

MODEL_PATH = "unemployment_forecasting_pipeline.joblib"
METADATA_PATH = "unemployment_forecasting_metadata.json"

# Load model pipeline
try:
  model = joblib.load(MODEL_PATH)
  model_loaded = True
except Exception as e:
  model = None
  model_loaded = False
  print(f"Warning: Could not load model pipeline: {e}")

# Load metadata
try:
  with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
except Exception:
  metadata = None


def generate_prediction_chart(prediction_val, region_name):
  """Generates a dynamic Seaborn bar plot converted to Base64 image string."""
  plt.figure(figsize=(7, 3.5))
  sns.set_theme(style="whitegrid")

  categories = ["National Avg (~10%)", f"{region_name} Predicted"]
  values = [10.2, prediction_val]
  colors = ["#6c757d", "#6366f1" if prediction_val < 15 else "#ef4444"]

  ax = sns.barplot(x=categories, y=values, palette=colors)
  plt.title(
      f"Unemployment Rate Comparison: {region_name}",
      fontsize=12,
      fontweight="bold",
      pad=12,
  )
  plt.ylabel("Unemployment Rate (%)", fontsize=10)
  plt.ylim(0, max(values) + 10)

  for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}%",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 8),
        textcoords="offset points",
        fontweight="bold",
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
  chart_url = None
  error_msg = None

  if request.method == "POST":
    if not model_loaded:
      error_msg = "Model error: Pipeline file not loaded."
    else:
      try:
        # User input fields
        labor_rate = float(request.form.get("labor_rate", 0))
        employed = float(request.form.get("employed", 0))
        region = request.form.get("region", "Selected Region")
        area = request.form.get("area", "Rural")

        # Current date components for time features
        now = datetime.datetime.now()
        current_year = now.year
        current_month = now.month

        # Estimated lag feature defaults derived from user inputs
        lag1_unemployment = 10.0  # Historical baseline average
        lag2_unemployment = 10.0
        lag1_employed = employed
        lag1_labour_participation = labor_rate

        # Construct full DataFrame with all expected pipeline columns
        input_data = {
            "Estimated Labour Participation Rate (%)": labor_rate,
            "Estimated Employed": employed,
            "Region": region,
            "Area": area,
            "Year": current_year,
            "Month": current_month,
            "Lag1_Unemployment": lag1_unemployment,
            "Lag2_Unemployment": lag2_unemployment,
            "Lag1_Employed": lag1_employed,
            "Lag1_Labour_Participation": lag1_labour_participation,
        }

        input_df = pd.DataFrame([input_data])

        # Run inference
        raw_pred = model.predict(input_df)[0]
        prediction = round(float(raw_pred), 2)
        chart_url = generate_prediction_chart(prediction, region)

      except Exception as e:
        error_msg = f"Inference Error: {str(e)}"

  return render_template(
      "index.html",
      prediction=prediction,
      chart_url=chart_url,
      error_msg=error_msg,
      metadata=metadata,
  )


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, debug=True)