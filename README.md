# Pneumono Detect: Pneumonia Detection System

Welcome to the Pneumono Detect repository, a deep learning-based application for detecting pneumonia from chest X-ray images using a fine-tuned VGG19 model. This repository contains the code, setup scripts, and resources to run the project locally.

---

## **Repository Contents**

| File/Folder                     | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `.gitattributes`                | Configures file attributes for the Git repository.                          |
| `.gitignore`                    | Specifies files and folders to be ignored by Git.                          |
| `272 Project PPT.pdf`           | Presentation file summarizing the project.                                 |
| `CNNforPneumoniaDetection.ipynb`| Jupyter Notebook with the model training code for pneumonia detection.      |
| `README.md`                     | This README file containing project details.                               |
| `app.py`                        | The main application script for running the Gradio-based interface.        |
| `requirements.txt`              | A list of Python dependencies required to run the application.             |
| `setup.bat`                     | Windows setup script to install dependencies and configure the environment.|
| `setup.sh`                      | Linux/Mac setup script to install dependencies and configure the environment.|
| `test_app.py`                   | Script containing tests for verifying the application functionality.        |
| `vgg19_fine_tuned_block5_91.keras`| The pre-trained VGG19 model file fine-tuned for pneumonia detection.      |

---

## **Steps to Set Up and Run the Project**

### **1. Clone the Repository**
Use the following command to clone the repository to your local machine:

```bash
git clone https://github.com/Praful-John2409/Pneumono_Detect.git
```

### **2. Open the Repository in an IDE**
1. Open your preferred IDE (e.g., **Visual Studio Code**).
2. Navigate to the cloned folder and open the project.

### **3. Install Dependencies**
Install the required dependencies using the `requirements.txt` file. Ensure you are using Python 3.8 or above.

#### On Windows:
1. Run the setup script to install dependencies:
   ```bash
   ./setup.bat
   ```

#### On Linux/Mac:
1. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```
2. Run the setup script:
   ```bash
   ./setup.sh
   ```

Alternatively, manually install dependencies using `pip`:
```bash
pip install -r requirements.txt
```

---

### **4. Dependencies Included**
The following dependencies are required and installed via `requirements.txt`:
- `flask==3.1.0`: For handling server-side operations.
- `tensorflow`: For building and loading the pre-trained VGG19 model.
- `Pillow==10.2.0`: For image processing.
- `numpy==1.23.5`: For numerical operations.
- `keras`: For deep learning workflows.
- `gunicorn`: For deploying the app in production.
- `gradio`: For creating the user interface.
- `pytest`: For writing and executing tests.

---

### **5. Run the Application**
Start the application using the following command:

```bash
python app.py
```

This will start a local server, and the application can be accessed via your web browser at `http://127.0.0.1:7860`.

---

### **6. Test the Application**
To verify the functionality of the application, run the test script:

```bash
python test_app.py
```

This will execute test cases to ensure the application is working as expected.

---

## **How It Works**
1. The application leverages a pre-trained **VGG19** model fine-tuned for pneumonia detection.
2. A **Gradio-based interface** allows users to upload chest X-ray images and view predictions.
3. The system preprocesses uploaded images, uses the TensorFlow model for inference, and displays results with confidence scores.

---
## **Live Project**

Click [here](https://huggingface.co/spaces/SoulMind01/PneumoniaDetection) to check out the live project.
---
## **Powerpoint Presentation link**

Click [here](https://docs.google.com/presentation/d/10pRh_B9fRCb__6gbEdFi43rkW2RWirDEH3OKZFdAKNI/edit?usp=sharing) to access the link to the powerpoint presentation with the updated reference to the images used in the presentation.
---
## **Acknowledgments**
This project was developed as part of coursework, with special thanks to **Prof. Andrew Bond** for guidance and support. The dataset used is from Kaggle's Chest X-Ray Images dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

