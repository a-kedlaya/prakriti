"""
PrakritiAI Flask backend — production ready.
  POST /predict  — image upload → dosha prediction
  GET  /health   — health check
Note: camera bot (real_time.py) runs locally only, not on cloud.
"""

import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app, origins="*", allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading prakriti model...")
prakriti_model = tf.keras.models.load_model("prakriti_model.h5", compile=False)
IMG_SIZE = 224
CLASSES  = ['Kapha', 'Pitta', 'Pitta-Kapha', 'Vata', 'Vata-Kapha', 'Vata-Pitta']
prakriti_model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)), verbose=0)
print("✅ Model ready.")


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    try:
        inp    = preprocess(request.files["image"].read())
        scores = prakriti_model.predict(inp, verbose=0)[0]
        idx    = int(np.argmax(scores))
        top3   = [{"dosha": CLASSES[i], "confidence": round(float(scores[i]) * 100, 1)}
                  for i in np.argsort(scores)[::-1][:3]]
        return jsonify({
            "dosha":      CLASSES[idx],
            "confidence": round(float(scores[idx]) * 100, 1),
            "top3":       top3
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n🌿 PrakritiAI server running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
