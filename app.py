import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
MODEL_PATH = "vgg19_fine_tuned_block5_91.keras"
model = load_model(MODEL_PATH)

# Define class labels and confidence threshold
CLASS_LABELS = ["NORMAL", "PNEUMONIA"]
CONFIDENCE_THRESHOLD = 0.7


def preprocess_image(image):
    """
    Preprocesses the input image for the model.
    Args:
        image (PIL.Image): Input image.
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.
    """
    img = image.convert("RGB")  # Ensure the image is RGB
    img = img.resize((128, 128))  # Resize to model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict_image(image):
    """
    Predicts the class of the input image with confidence-based filtering.
    Args:
        image (PIL.Image): Input image.
    Returns:
        str: Predicted class label or uncertainty message.
        float: Confidence score (if applicable).
    """
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    confidence = np.max(prediction)

    # Apply confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain: Low confidence", confidence

    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    return predicted_class, confidence


# Create a Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Predicted Class"), gr.Textbox(label="Confidence")],
    title="Pneumonia Detection CNN",
    description="Upload an image to classify it as NORMAL or PNEUMONIA.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
