
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    abort
)

from werkzeug.utils import secure_filename
from tensorflow import keras

from generate_heatmap import (
    create_gradcam,
    make_colored_overlay
)

from report_generator import generate_report


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = OUTPUT_DIR / "solar_panel_model_cleaned.keras"

CLASS_NAMES_PATH = OUTPUT_DIR / "class_names_cleaned.txt"

UPLOAD_DIR = BASE_DIR / "static" / "uploads"

RESULT_DIR = BASE_DIR / "static" / "results"

REPORT_DIR = OUTPUT_DIR / "web_app_reports"

IMAGE_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

MAX_FILE_SIZE = 10 * 1024 * 1024

for folder in (
    UPLOAD_DIR,
    RESULT_DIR,
    REPORT_DIR
):
    folder.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. FLASK APPLICATION
# ==================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ==================================================
# 3. LOAD EXISTING CLEANED MODEL
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

print("Loading cleaned solar panel model...")

model = keras.models.load_model(MODEL_PATH)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]

expected_classes = [
    "clean",
    "dusty",
    "visible_damage"
]

if class_names != expected_classes:
    raise ValueError(
        f"Unexpected class names: {class_names}"
    )

print("Model loaded successfully.")
print("Classes:", class_names)


# ==================================================
# 4. INSPECTION SESSION STORAGE
# ==================================================

# Stores successful inspection information
# while the Flask application is running.

inspection_reports = {}


# ==================================================
# 5. IMAGE VALIDATION
# ==================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==================================================
# 6. IMAGE QUALITY ASSESSMENT
# ==================================================

def assess_image_quality(image):

    width, height = image.size

    grayscale = image.convert("L").resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    gray_array = np.asarray(grayscale)

    brightness = float(
        np.mean(gray_array)
    )

    contrast = float(
        np.std(gray_array)
    )

    sharpness = float(
        cv2.Laplacian(
            gray_array,
            cv2.CV_64F
        ).var()
    )

    warnings = []

    if min(width, height) < 224:
        warnings.append(
            "Image resolution is relatively low."
        )

    if brightness < 25:
        warnings.append(
            "Image is too dark."
        )

    elif brightness > 230:
        warnings.append(
            "Image is too bright."
        )

    if contrast < 12:
        warnings.append(
            "Image has low contrast."
        )

    if sharpness < 20:
        warnings.append(
            "Image may be blurred."
        )

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "warnings": warnings
    }


# ==================================================
# 7. MAINTENANCE RECOMMENDATIONS
# ==================================================

def generate_recommendation(
    predicted_class,
    confidence,
    warnings
):

    if confidence < 65 or warnings:

        return {
            "status": "Manual Review Required",

            "recommendation": (
                "The prediction may be unreliable. "
                "Capture a clearer image under suitable "
                "lighting and arrange manual inspection."
            )
        }

    if predicted_class == "clean":

        return {
            "status": "No Visible Issue Detected",

            "recommendation": (
                "The model did not detect visible dust "
                "or damage. This is not a guarantee of "
                "panel health. Continue routine physical "
                "inspection."
            )
        }

    if predicted_class == "dusty":

        return {
            "status": "Dust / Soiling Suspected",

            "recommendation": (
                "Dust or soiling may be present. "
                "Arrange panel inspection and cleaning "
                "if required."
            )
        }

    if predicted_class == "visible_damage":

        return {
            "status": "Visible Damage Suspected",

            "recommendation": (
                "Possible physical damage was detected. "
                "Arrange a human physical inspection "
                "before taking maintenance action."
            )
        }

    return {
        "status": "Manual Review Required",

        "recommendation": (
            "The result requires human verification."
        )
    }


# ==================================================
# 8. MAIN DASHBOARD ROUTE
# ==================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:

            error = "Please select an image."

        else:

            uploaded_file = request.files["image"]

            if uploaded_file.filename == "":

                error = "No image was selected."

            elif not allowed_file(uploaded_file.filename):

                error = (
                    "Please upload JPG, JPEG, PNG, or WEBP."
                )

            else:

                unique_id = uuid4().hex

                safe_name = secure_filename(
                    uploaded_file.filename
                )

                upload_path = (
                    UPLOAD_DIR
                    / f"{unique_id}_{safe_name}"
                )

                try:

                    # ----------------------------------
                    # READ AND VALIDATE IMAGE
                    # ----------------------------------

                    image = Image.open(
                        uploaded_file.stream
                    )

                    image.load()

                    image = image.convert("RGB")

                    # ----------------------------------
                    # SAVE ORIGINAL IMAGE
                    # ----------------------------------

                    image.save(
                        upload_path,
                        format="JPEG",
                        quality=95
                    )

                    # ----------------------------------
                    # MODEL PREPROCESSING
                    # ----------------------------------

                    resized_image = keras.utils.load_img(
                        upload_path,
                        target_size=IMAGE_SIZE,
                        color_mode="rgb",
                        interpolation="bilinear"
                    )

                    image_array = keras.utils.img_to_array(
                        resized_image
                    )

                    model_input = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    # ----------------------------------
                    # AI PREDICTION
                    # ----------------------------------

                    probabilities = model.predict(
                        model_input,
                        verbose=0
                    )[0]

                    predicted_index = int(
                        np.argmax(probabilities)
                    )

                    predicted_class = class_names[
                        predicted_index
                    ]

                    confidence = float(
                        probabilities[predicted_index] * 100
                    )

                    class_probabilities = {
                        class_names[i]: round(
                            float(probabilities[i] * 100),
                            2
                        )
                        for i in range(len(class_names))
                    }

                    # ----------------------------------
                    # IMAGE QUALITY ASSESSMENT
                    # ----------------------------------

                    quality = assess_image_quality(image)

                    # ----------------------------------
                    # GRAD-CAM GENERATION
                    # ----------------------------------

                    heatmap = create_gradcam(
                        model,
                        image_array.astype(np.float32),
                        predicted_index
                    )

                    original_array = np.asarray(image)

                    overlay, resized_heatmap = (
                        make_colored_overlay(
                            original_array,
                            heatmap
                        )
                    )

                    overlay_filename = (
                        f"{unique_id}_overlay.jpg"
                    )

                    overlay_path = (
                        RESULT_DIR / overlay_filename
                    )

                    Image.fromarray(overlay).save(
                        overlay_path,
                        format="JPEG",
                        quality=95
                    )

                    # ----------------------------------
                    # MAINTENANCE RECOMMENDATION
                    # ----------------------------------

                    recommendation = generate_recommendation(
                        predicted_class,
                        confidence,
                        quality["warnings"]
                    )

                    # ----------------------------------
                    # INSPECTION RESULT
                    # ----------------------------------

                    result = {

                        "report_id": unique_id,

                        "original_image": (
                            f"uploads/{upload_path.name}"
                        ),

                        "overlay_image": (
                            f"results/{overlay_filename}"
                        ),

                        "predicted_class": predicted_class,

                        "confidence": round(
                            confidence,
                            2
                        ),

                        "probabilities": class_probabilities,

                        "quality": quality,

                        "recommendation": recommendation
                    }

                    # ----------------------------------
                    # STORE DATA FOR PDF GENERATION
                    # ----------------------------------

                    inspection_reports[unique_id] = {

                        "report_data": {

                            "report_id": unique_id,

                            "predicted_class": predicted_class,

                            "confidence": round(
                                confidence,
                                2
                            ),

                            "probabilities": class_probabilities,

                            "quality": quality,

                            "recommendation": recommendation
                        },

                        "original_path": upload_path,

                        "overlay_path": overlay_path
                    }

                except Exception:

                    if upload_path.exists():
                        upload_path.unlink()

                    app.logger.exception(
                        "Image inspection failed"
                    )

                    error = (
                        "Could not process this image. "
                        "Please try another valid image."
                    )

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ==================================================
# 9. PDF INSPECTION REPORT DOWNLOAD
# ==================================================

@app.route("/download-report/<report_id>")
def download_report(report_id):

    inspection = inspection_reports.get(report_id)

    if inspection is None:

        abort(
            404,
            description="Inspection report not found."
        )

    pdf_path = (
        REPORT_DIR
        / f"inspection_report_{report_id}.pdf"
    )

    try:

        generated_path = generate_report(

            report_data=inspection["report_data"],

            original_path=inspection["original_path"],

            overlay_path=inspection["overlay_path"],

            output_path=pdf_path
        )

        return send_file(

            generated_path,

            as_attachment=True,

            download_name=(
                f"SolarInspect_Report_{report_id[:8]}.pdf"
            ),

            mimetype="application/pdf"
        )

    except Exception:

        app.logger.exception(
            "PDF report generation failed"
        )

        abort(
            500,
            description="Could not generate the PDF report."
        )


# ==================================================
# 10. RUN FLASK APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
