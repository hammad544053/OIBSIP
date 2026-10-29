import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the saved Scikit-Learn artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'iris_classification_pipeline.joblib')

try:
    loaded_artifact = joblib.load(MODEL_PATH)
    # Extract the actual pipeline and target mapping dictionary
    model_pipeline = loaded_artifact["pipeline"]
    target_mapping = loaded_artifact.get("target_mapping", {0: 'setosa', 1: 'versicolor', 2: 'virginica'})
except Exception as e:
    model_pipeline = None
    target_mapping = {}
    print(f"Error loading model from {MODEL_PATH}: {e}")

# Mapping for species metadata, descriptive details, and static images
SPECIES_INFO = {
    'setosa': {
        'display_name': 'Iris Setosa',
        'image': 'images/setosa.jpg',
        'description': 'Characterized by small, delicate petals and short sepal lengths. Native to Arctic regions.'
    },
    'versicolor': {
        'display_name': 'Iris Versicolor',
        'image': 'images/versicolor.jpg',
        'description': 'Known as the Harlequin Blue Flag. Medium-sized features with intricate blue-purple veining.'
    },
    'virginica': {
        'display_name': 'Iris Virginica',
        'image': 'images/virginica.jpg',
        'description': 'The Virginia Iris features large, prominent petals and robust stem growth patterns.'
    }
}

def normalize_species_key(raw_species_name):
    """Normalize predicted species string to match SPECIES_INFO dictionary keys."""
    species_str = str(raw_species_name).lower()
    if 'setosa' in species_str:
        return 'setosa'
    elif 'versicolor' in species_str:
        return 'versicolor'
    elif 'virginica' in species_str:
        return 'virginica'
    return 'setosa'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model pipeline artifact is missing or failed to load.'}), 500

    try:
        data = request.get_json()
        sepal_length = float(data.get('sepal_length'))
        sepal_width = float(data.get('sepal_width'))
        petal_length = float(data.get('petal_length'))
        petal_width = float(data.get('petal_width'))

        # Prepare features DataFrame matching exact feature column names
        features_df = pd.DataFrame([{
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
        }])

        # Execute prediction through trained pipeline
        raw_prediction = model_pipeline.predict(features_df)[0]

        # Resolve prediction label using target_mapping if raw output is integer ID
        if isinstance(raw_prediction, (int, np.integer)) or (isinstance(raw_prediction, str) and raw_prediction.isdigit()):
            predicted_class_id = int(raw_prediction)
            species_name = target_mapping.get(predicted_class_id, "setosa")
        else:
            species_name = str(raw_prediction)

        species_key = normalize_species_key(species_name)

        # Calculate prediction probability if supported
        confidence = None
        if hasattr(model_pipeline, "predict_proba"):
            probabilities = model_pipeline.predict_proba(features_df)[0]
            confidence = float(np.max(probabilities) * 100)

        info = SPECIES_INFO[species_key]

        return jsonify({
            'success': True,
            'species_name': info['display_name'],
            'image_url': f"/static/{info['image']}",
            'description': info['description'],
            'confidence': f"{confidence:.1f}%" if confidence is not None else "N/A"
        })

    except Exception as err:
        return jsonify({'error': f'Invalid input or calculation error: {str(err)}'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)