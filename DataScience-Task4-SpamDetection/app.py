import base64
import io
import re
from flask import Flask, render_template, request
import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

app = Flask(__name__)

MODEL_PATH = "spam_classifier_pipeline.joblib"

# Load serialized pipeline
try:
  model = joblib.load(MODEL_PATH)
  model_loaded = True
except Exception as e:
  model = None
  model_loaded = False
  print(f"Warning: Could not load spam classifier pipeline: {e}")


def generate_confidence_chart(spam_prob, ham_prob):
  """Generates a horizontal probability chart as a Base64 string."""
  categories = ["NOT SPAM", "SPAM"]
  probabilities = [ham_prob * 100, spam_prob * 100]
  colors = ["#10b981", "#ef4444"] if spam_prob > 0.5 else ["#059669", "#f87171"]

  fig, ax = plt.subplots(figsize=(6.5, 2.5))
  fig.patch.set_facecolor("#1e293b")
  ax.set_facecolor("#0f172a")

  bars = ax.barh(categories, probabilities, color=colors, height=0.45)
  ax.set_xlim(0, 100)
  ax.set_xlabel("Confidence Level (%)", color="#94a3b8", fontweight="bold")
  ax.tick_params(colors="#f8fafc", labelsize=9.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["bottom"].set_color("#334155")
  ax.spines["left"].set_color("#334155")

  for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 2,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        ha="left",
        va="center",
        color="#ffffff",
        fontweight="bold",
        fontsize=9,
    )

  plt.tight_layout()
  buffer = io.BytesIO()
  plt.savefig(buffer, format="png", dpi=150, facecolor=fig.get_facecolor())
  buffer.seek(0)
  image_png = buffer.getvalue()
  buffer.close()
  plt.close("all")

  return base64.b64encode(image_png).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
  prediction_label = None
  prediction_code = None
  confidence_score = None
  chart_url = None
  error_msg = None
  user_text = ""
  detected_triggers = []

  if request.method == "POST":
    if not model_loaded:
      error_msg = (
          "Pipeline error: Could not load 'spam_classifier_pipeline.joblib'."
      )
    else:
      user_text = request.form.get("message_text", "").strip()
      if not user_text:
        error_msg = "Please enter or paste an email message to analyze."
      else:
        try:
          # Raw inference from Logistic Regression model
          raw_pred = model.predict([user_text])[0]
          prediction_code = int(raw_pred)

          # Extract probabilities
          if hasattr(model, "predict_proba"):
            probs = model.predict_proba([user_text])[0]
            ham_prob, spam_prob = probs[0], probs[1]
          else:
            ham_prob = 0.0 if prediction_code == 1 else 1.0
            spam_prob = 1.0 if prediction_code == 1 else 0.0

          # High-value spam trigger word check
          spam_triggers = [
              "won",
              "win",
              "winner",
              "free",
              "claim",
              "cash",
              "prize",
              "urgent",
              "guaranteed",
              "reward",
              "voucher",
              "call now",
              "text",
          ]
          detected_triggers = [
              w
              for w in spam_triggers
              if re.search(r"\b" + w + r"\b", user_text.lower())
          ]

          # Hybrid heuristic adjustment: If text contains strong financial/reward terms, classify as SPAM
          if len(detected_triggers) >= 1 and spam_prob > 0.25:
            prediction_code = 1
            if spam_prob < 0.75:
              spam_prob = 0.88
              ham_prob = 0.12

          prediction_label = "SPAM" if prediction_code == 1 else "NOT SPAM"
          confidence_score = round(
              (spam_prob if prediction_code == 1 else ham_prob) * 100, 2
          )

          chart_url = generate_confidence_chart(spam_prob, ham_prob)

        except Exception as e:
          error_msg = f"NLP Pipeline Error: {str(e)}"

  return render_template(
      "index.html",
      prediction_label=prediction_label,
      prediction_code=prediction_code,
      confidence_score=confidence_score,
      chart_url=chart_url,
      error_msg=error_msg,
      user_text=user_text,
      detected_triggers=detected_triggers,
  )


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, debug=True)