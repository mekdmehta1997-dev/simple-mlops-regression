from flask import Flask, request, jsonify
import joblib
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

MODEL_PATH = os.path.join("artifacts", "model.pkl")

app = Flask(__name__)

# Prometheus metrics
PREDICTION_COUNT = Counter("prediction_total", "Total number of predictions")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time spent for prediction in seconds")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Make sure artifacts/model.pkl exists.")
    return joblib.load(MODEL_PATH)

model = None
try:
    model = load_model()
    print("Model loaded at startup.")
except Exception as e:
    print("Model NOT loaded at startup:", e)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = load_model()

    data = request.get_json() or {}
    size = data.get("size")
    if size is None:
        return jsonify({"error": "Please send JSON with 'size'"}), 400

    PREDICTION_COUNT.inc()
    with PREDICTION_LATENCY.time():
        prediction = model.predict([[float(size)]])[0]

    return jsonify({"size": float(size), "predicted_price": float(prediction)})

@app.route("/metrics")
def metrics():
    # Endpoint for Prometheus scraping
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
