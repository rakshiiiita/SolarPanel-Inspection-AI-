# Solar Panel Inspection AI

An AI-based solar panel inspection system that uses deep learning to classify solar panel conditions and generate inspection reports.

## Features

- Solar panel image classification
- Three condition classes: Clean, Dusty, Visible Damage
- Confidence scores and class probabilities
- Grad-CAM visual explainability
- Image quality assessment
- Maintenance recommendations
- Automated PDF inspection reports
- Flask-based web dashboard

## Model Performance

Test accuracy: 86.96%

Evaluated on a 69-image test dataset using the TensorFlow evaluation pipeline.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Flask
- Grad-CAM
- ReportLab

## How to Run

1. Install Python.
2. Install dependencies:

   pip install -r requirements.txt

3. Run the application:

   python app.py

4. Open your browser and visit:

   http://127.0.0.1:5000

## Project Scope

The current implementation is an image-upload-based AI inspection system.

Physical robot movement, autonomous navigation and camera integration are planned as future development stages.

## Disclaimer

Predictions are AI-assisted assessments and should not replace physical inspection by a qualified person.
