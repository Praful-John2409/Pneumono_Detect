# Work Distribution:
# Function preprocess_image, predict_image written by Tingfei Gu
# Function acknowledge written by Ranyi Zhang
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
    img = image.convert("RGB")  # Ensure the image is RGB
    img = img.resize((128, 128))  # Resize to model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    confidence = np.max(prediction)

    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain: Low confidence", confidence

    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    return predicted_class, confidence


def acknowledge(agree):
    if agree:
        return (
            gr.update(visible=True),
            "Thank you for acknowledging the disclaimer. You may now use the app.",
        )
    else:
        return gr.update(visible=False), "You must accept the disclaimer to proceed."


# Create a Gradio interface using Blocks
with gr.Blocks() as app:
    gr.Markdown(
        "**Disclaimer:** This application is a student project developed as part of coursework and is intended solely for educational and experimental purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. The results provided by this application should not be relied upon for medical decision-making. Use at your own discretion."
    )

    agree = gr.Checkbox(
        label="I acknowledge that this application is for experimental use only and not suitable for medical purposes."
    )
    message = gr.Textbox(interactive=False)

    with gr.Row(visible=False) as interface_row:
        image_input = gr.Image(type="pil")
        submit_button = gr.Button("Submit")
        predicted_class = gr.Textbox(label="Predicted Class")
        confidence = gr.Textbox(label="Confidence")

        submit_button.click(
            fn=predict_image, inputs=image_input, outputs=[predicted_class, confidence]
        )

    agree.change(acknowledge, agree, [interface_row, message])

# Launch the app
if __name__ == "__main__":
    app.launch()
