from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "model/densenet_final.h5"
model = load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    "Corn Blight",
    "Corn Rust",
    "Healthy Corn",
    "Potato Early Blight",
    "Potato Late Blight",
    "Healthy Potato"
]

REMEDIES = {
    "Corn Blight": "Use copper-based fungicides, rotate crops, and avoid overhead irrigation.",
    "Corn Rust": "Apply sulfur-based fungicides and ensure good air circulation.",
    "Healthy Corn": "No disease detected. Maintain regular watering and balanced fertilization.",
    "Potato Early Blight": "Apply chlorothalonil fungicide and remove infected leaves.",
    "Potato Late Blight": "Use mancozeb fungicide and avoid watering foliage directly.",
    "Healthy Potato": "No disease detected. Continue normal plant care routine."
}

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = "temp.jpg"
    file.save(filepath)

    # Preprocess
    img = preprocess_image(filepath)

    # Predict
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds)) * 100

    disease_name = CLASS_NAMES[class_idx]
    remedy = REMEDIES[disease_name]

    return jsonify({
        "disease": disease_name,
        "confidence": round(confidence, 2),
        "remedy": remedy
    })

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)