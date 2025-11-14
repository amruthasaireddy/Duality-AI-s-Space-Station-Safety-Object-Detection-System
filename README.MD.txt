📦 YOLOv8 Object Detection Web App
This project provides a Streamlit web application to load, test, and validate custom-trained Ultralytics YOLOv8 models.

You can perform two main actions:

Live Prediction: Upload any image to get instant object detection results from your model.

Model Validation: Run the official model.val() on your test dataset (as defined in yolo_params.yaml) and see the performance metrics and graphs.

📂 Project Structure
Your project must follow this directory structure for the app to find your models and data:

your_project_root/
│
├── 🚀 app.py                 # The Streamlit Web Application
├── 📝 yolo_params.yaml       # Your YOLO configuration file (must link to your dataset)
│
├── 📦 runs/
│   └── detect/
│       ├── train/           # Your YOLO training output folder
│       │   └── weights/
│       │       └── best.pt  # The model to be loaded
│       ├── train2/
│       │   └── weights/
│       │       └── best.pt
│       └── ...
│
└── 🖼️ your_dataset/          # Your dataset folder (as defined in yolo_params.yaml)
    ├── test/
    │   └── images/
    └── ...
🛠️ Setup & Installation
This guide assumes you have Conda installed.

Create a Conda Environment Open your terminal and create a new environment.

Bash

conda create -n yolo_app python=3.10
Activate the Environment You must activate the environment every time you run the app.

Bash

conda activate yolo_app
Install Required Packages Install all the necessary libraries into your new environment.

Bash

pip install streamlit ultralytics opencv-python-headless PyYAML pillow
🏃 How to Run the App
Navigate to Your Project Open your terminal and change to the project's root directory (where app.py is located).

Bash

cd path/to/your_project_root
Activate Your Environment If you haven't already, activate your conda environment.

Bash

conda activate yolo_app
Run the Streamlit App This is the main command to start the web server.

Bash

streamlit run app.py
Open in Browser Your terminal will show you a "Local URL". Copy this URL (usually http://localhost:8501) and paste it into your web browser.

The application is now running!