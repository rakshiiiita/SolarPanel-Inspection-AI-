from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "solar_panel_model_cleaned.keras"
CLASS_PATH = BASE_DIR / "outputs" / "class_names_cleaned.txt"
OUTPUT_DIR = BASE_DIR / "outputs" / "heatmap_results"

IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER = "Conv_1"
ALPHA = 0.38


def load_image(image_path):
    original = Image.open(image_path).convert("RGB")
    original_array = np.array(original)

    resized = original.resize(IMAGE_SIZE)
    image_array = np.array(resized).astype(np.float32)

    # The trained model includes preprocessing internally.
    # Do not apply MobileNetV2 preprocess_input again.
    return original_array, image_array



def create_gradcam(model, image_array, class_index):
    backbone = model.get_layer("mobilenetv2_1.00_224")

    # Reproduce the preprocessing found in the saved model.
    normalized = image_array / 127.5 - 1.0

    batch = tf.convert_to_tensor(
        normalized[None, ...],
        dtype=tf.float32,
    )

    # Extract the final convolutional feature maps.
    feature_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[
            backbone.get_layer("Conv_1").output,
            backbone.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = feature_model(
            batch, training=False
        )

        pooled = model.get_layer("global_average_pooling2d")(
            backbone_features
        )
        dropped = model.get_layer("dropout")(
            pooled, training=False
        )
        predictions = model.get_layer("dense")(dropped)

        target_score = predictions[:, class_index]

    # Compare with the original model's prediction.
    original_predictions = model(
        tf.convert_to_tensor(
            image_array[None, ...],
            dtype=tf.float32,
        ),
        training=False,
    )

    print("\n=== GRAD-CAM DIAGNOSTICS ===")
    print("Original model scores:",
          original_predictions.numpy()[0])
    print("Reconstructed scores:",
          predictions.numpy()[0])
    print("Maximum score difference:",
          float(tf.reduce_max(
              tf.abs(original_predictions - predictions)
          ).numpy()))

    if not np.allclose(
        original_predictions.numpy(),
        predictions.numpy(),
        atol=1e-4,
    ):
        raise RuntimeError(
            "Predictions do not match. "
            "Do not trust this heatmap yet."
        )

    gradients = tape.gradient(target_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable."
        )

    weights = tf.reduce_mean(
        gradients, axis=(1, 2), keepdims=True
    )

    heatmap = tf.reduce_sum(
        weights * conv_outputs, axis=-1
    )[0]

    heatmap = tf.nn.relu(heatmap)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, maximum)

    return heatmap.numpy()




def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python generate_heatmap.py "path\\to\\image.jpg"'
        )
        return

    image_path = Path(sys.argv[1]).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_names = CLASS_PATH.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    original, image_array = load_image(image_path)

    # Disable training-time augmentation for inference.
    probabilities = model(
        tf.convert_to_tensor(image_array[None, ...]),
        training=False,
    ).numpy()[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]

    heatmap = create_gradcam(
        model,
        image_array,
        predicted_index,
    )

    overlay, resized_heatmap = make_colored_overlay(
        original,
        heatmap,
    )

    prefix = image_path.stem
    original_path = OUTPUT_DIR / f"{prefix}_original.jpg"
    heatmap_path = OUTPUT_DIR / f"{prefix}_gradcam.jpg"
    overlay_path = OUTPUT_DIR / f"{prefix}_overlay.jpg"

    Image.fromarray(original).save(original_path, quality=95)
    Image.fromarray(overlay).save(overlay_path, quality=95)

    plt.figure(figsize=(7, 7))
    plt.imshow(resized_heatmap, cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(
        heatmap_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    print("\n=== SOLAR PANEL IMAGE ANALYSIS ===")
    print("Predicted class:", predicted_class)
    print(f"Class score: {probabilities[predicted_index] * 100:.2f}%")

    for name, score in zip(class_names, probabilities):
        print(f"{name}: {score * 100:.2f}%")

    print("\nSaved outputs:")
    print(original_path)
    print(heatmap_path)
    print(overlay_path)

    print(
        "\nInterpretation: red indicates relatively strong model "
        "activation, not a confirmed physical defect. "
        "Green indicates lower activation, not guaranteed cleanliness."
    )



def make_colored_overlay(original_array, heatmap):
    """
    Resize a Grad-CAM heatmap and blend it with the original image.
    Returns the colored overlay and resized heatmap.
    """
    original_height, original_width = original_array.shape[:2]

    resized_heatmap = cv2.resize(
        heatmap,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    heatmap_uint8 = np.uint8(resized_heatmap * 255)

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    # OpenCV uses BGR; convert to RGB for PIL/matplotlib.
    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    original_uint8 = np.clip(
        original_array, 0, 255
    ).astype(np.uint8)

    overlay = cv2.addWeighted(
        original_uint8,
        1.0 - ALPHA,
        colored_heatmap,
        ALPHA,
        0,
    )

    return overlay, resized_heatmap



if __name__ == "__main__":
    main()
