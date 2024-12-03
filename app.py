from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "vgg19_fine_tuned_block5_91.keras"
model = load_model(MODEL_PATH)

# Define class labels and confidence threshold
CLASS_LABELS = ['NORMAL', 'PNEUMONIA']
CONFIDENCE_THRESHOLD = 0.7

def preprocess_image(file_path):
    """
    Preprocesses the input image for the model.
    Args:
        file_path (str): Path to the input image.
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.
    """
    img = Image.open(file_path).convert('RGB')  # Ensure the image is RGB
    img = img.resize((128, 128))  # Resize to model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(file_path):
    """
    Predicts the class of the input image with confidence-based filtering.
    Args:
        file_path (str): Path to the input image.
    Returns:
        str: Predicted class label or uncertainty message.
        float: Confidence score (if applicable).
    """
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    confidence = np.max(prediction)

    # Apply confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain: Low confidence", confidence

    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    return predicted_class, confidence

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    if file:
        # Save the uploaded file temporarily
        upload_path = os.path.join("static/uploads", file.filename)
        os.makedirs("static/uploads", exist_ok=True)
        file.save(upload_path)

        # Make prediction
        predicted_class, confidence = predict_image(upload_path)

        # Format the result based on prediction type
        if "Uncertain" in predicted_class:
            message = "The model is uncertain about the prediction. Please try another image."
            return render_template(
                "result.html",
                prediction=message,
                confidence=f"{confidence*100:.2f}%",
                image_path=upload_path,
            )
        else:
            return render_template(
                "result.html",
                prediction=predicted_class,
                confidence=f"{confidence*100:.2f}%",
                image_path=upload_path,
            )

if __name__ == "__main__":
    app.run(debug=True)