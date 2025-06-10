from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = "models/Xception_best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
classes = ["Normal", "Diabetic Retinopathy", "Cataract", "Glaucoma"]

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file:
            img_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(img_path)

            # Preprocess and predict
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            return render_template(
                "result.html",
                img_path=img_path,
                prediction=predicted_class,
                confidence=confidence,
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
