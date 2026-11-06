
from flask import Flask, render_template, request, jsonify
import pickle
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("model/spam_classifier.pkl")

# Load the trained ML model
model = None

def load_model():
    global model

    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

load_model()


@app.route("/")
def home():
    return render_template("index.html")


# API endpoint to check server status
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Spam Email Classifier API is running",
        "model_loaded": model is not None
    })


# API endpoint for spam prediction
@app.route("/api/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not found. Train the model first."
        }), 503

    data = request.get_json(silent=True) or {}

    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Please enter a valid email message."
        }), 400

    try:
        prediction = model.predict([message])[0]

        probabilities = model.predict_proba([message])[0]
        confidence = float(max(probabilities) * 100)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "result": (
                "Spam Message"
                if prediction == "spam"
                else "Not Spam"
            ),
            "confidence": round(confidence, 2)
        })

    except Exception:
        return jsonify({
            "error": "Unable to process this message."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
      
