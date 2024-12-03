# Pneumonia Detection System

A Flask-based web application that uses a fine-tuned VGG19 model to detect pneumonia from chest X-ray images.

## Model Links
- [CNN Model](https://drive.google.com/file/d/1-4L-8HJ79W5k-0l8FchG4HH1SI2dLi2W/view?usp=sharing)
<!-- - [Dataset](https://drive.google.com/drive/folders/1BhxsscDaVBamuyUv1HoyXsixeGvTWHt5?usp=sharing) -->

## Setup Instructions

### Prerequisites
- Python 3.8 or higher

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Pneumono_Detect.git
cd Pneumono_Detect
```

2. Set up the environment:

#### Windows:

```bash
./setup.bat
```

#### Linux/Mac:

```bash
chmod +x setup.sh
./setup.sh
```

### Activating the Environment

#### Windows:

```bash
tf_test_env\Scripts\activate
```

#### Linux/Mac:

```bash
source tf_test_env/bin/activate
```

## Running the Application

1. Ensure your virtual environment is activated
2. Run the Flask application:

```bash
python app.py
```
3. Open a web browser and navigate to `http://localhost:5000`

## Usage

1. Upload a chest X-ray image through the web interface
2. Click "Predict" to get the classification result
3. View the prediction result and confidence score

## Project Structure
```
Pneumono_Detect/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── setup.bat              # Windows setup script
├── setup.sh               # Linux/Mac setup script
├── static/
│   └── uploads/           # Folder for uploaded images
└── templates/
    ├── index.html         # Upload page
    └── result.html        # Results page
```

## Model Information
- Architecture: VGG19 (fine-tuned)
- Input Size: 128x128x3
- Classes: NORMAL, PNEUMONIA
- Confidence Threshold: 0.7

## Dependencies
- Flask 3.1.0
- TensorFlow 2.12.0
- Pillow 10.2.0
- NumPy 1.23.5

## License
[Your chosen license]

## Acknowledgments
- Dataset source: [Add source]
- Base VGG19 model: [Reference]
```

This README provides:
1. Clear setup instructions for both Windows and Linux/Mac
2. Project structure overview
3. Usage instructions
4. Model information
5. Dependencies list
6. Placeholders for license and acknowledgments

Remember to:
1. Replace `yourusername` with your actual GitHub username
2. Add appropriate license information
3. Fill in the acknowledgments section
4. Update any specific details about your implementation