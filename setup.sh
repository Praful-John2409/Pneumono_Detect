#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv tf_test_env

echo "Activating virtual environment..."
source tf_test_env/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Upgrading TensorFlow and Keras..."
pip install --upgrade tensorflow keras

echo "Setup complete! To activate the environment, run:"
echo "source tf_test_env/bin/activate" 