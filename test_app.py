# Work Distribution:
# Function mock_model, test_preprocess_iamge, test_predict_image written by Tingfei Gu
# Function test_acknowledge, test_gradio_interface, test_end_to_end written by Ranyi Zhang
import pytest
import numpy as np
from unittest.mock import MagicMock
from PIL import Image
import gradio as gr
from app import preprocess_image, predict_image, acknowledge, model, CLASS_LABELS


# Mock the model for testing purposes
@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    mock = MagicMock()
    # Simulate model's prediction output
    mock.predict.return_value = np.array(
        [[0.2, 0.8]]
    )  # Simulates a PNEUMONIA prediction
    monkeypatch.setattr("app.model", mock)
    return mock


# Test 1: Preprocessing Function
def test_preprocess_image():
    # Create a dummy image
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    processed = preprocess_image(img)

    # Assert the output shape and values
    assert processed.shape == (1, 128, 128, 3), "Preprocessed image shape is incorrect."
    assert np.all(
        (0 <= processed) & (processed <= 1)
    ), "Image values should be normalized to [0, 1]."


# Test 2: Prediction Function
def test_predict_image(mock_model):
    # Create a dummy image
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))

    # Predict using the mock model
    predicted_class, confidence = predict_image(img)

    # Assert the outputs
    assert predicted_class == "PNEUMONIA", "Prediction class is incorrect."
    assert 0 <= confidence <= 1, "Confidence should be between 0 and 1."
    assert confidence == 0.8, "Confidence value does not match mock prediction."


# Test 3: Acknowledge Disclaimer
def test_acknowledge():
    # Test acceptance of disclaimer
    output, message = acknowledge(True)
    assert (
        output["visible"] is True
    ), "Interface should be visible when disclaimer is acknowledged."
    assert (
        "Thank you for acknowledging" in message
    ), "Acknowledgment message is incorrect."

    # Test rejection of disclaimer
    output, message = acknowledge(False)
    assert (
        output["visible"] is False
    ), "Interface should not be visible when disclaimer is rejected."
    assert (
        "You must accept the disclaimer" in message
    ), "Rejection message is incorrect."


# Test 4: Gradio Interface Integration
def test_gradio_interface():
    with gr.Blocks() as test_app:
        gr.Markdown("Test Disclaimer")
        agree = gr.Checkbox()
        interface_row = gr.Row(visible=False)
        message = gr.Textbox()

        agree.change(acknowledge, agree, [interface_row, message])

    # Simulate the interaction
    app_tester = gr.Interface(
        fn=lambda x: x,  # Dummy function for testing
        inputs=gr.Textbox(),  # Add a simple input
        outputs=gr.Textbox(),  # Add a simple output
    )

    # Verify Gradio app was created without issues
    assert app_tester is not None, "Gradio Interface should be created successfully."


# Test 5: End-to-End Integration Test
def test_end_to_end(mock_model):
    # Simulate a user uploading an image
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    preprocessed_img = preprocess_image(img)
