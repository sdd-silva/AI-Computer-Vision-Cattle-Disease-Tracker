# Whole-cow camera:
# python animal_camera.py --mode cow
#
# Udder close-up camera:
# python animal_camera.py --mode udder

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import time

import cv2
from PIL import Image
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO



# Configuration


BASE_DIR = Path(__file__).parent

MODEL_PATHS = {
    "cow": BASE_DIR / "best_model.pth",
    "udder": BASE_DIR / "udder_model.pth",
}

YOLO_PATH = BASE_DIR / "yolo11n.pt"

FIREBASE_API_KEY = os.getenv(
    "FIREBASE_API_KEY",
    "AIzaSyDSHECu0Qzc6oJQ4jtpcj3bqTssq79dLzI"
)

FIREBASE_DATABASE_URL = os.getenv(
    "FIREBASE_DATABASE_URL",
    "https://ai-powered-smart-livestock-default-rtdb.asia-southeast1.firebasedatabase.app"
).rstrip("/")

FIREBASE_PATH = os.getenv(
    "FIREBASE_PATH",
    "animal_camera/latest"
)

FIREBASE_UPDATE_INTERVAL = float(
    os.getenv(
        "FIREBASE_UPDATE_INTERVAL",
        "5"
    )
)

HISTORY_SIZE = 20
COW_DETECTION_THRESHOLD = 0.50



# Arguments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Cattle disease camera"
    )

    parser.add_argument(
        "--mode",
        choices=["cow", "udder"],
        default="cow",
        help=(
            "Use 'cow' for whole-cow disease detection "
            "or 'udder' for mastitis detection."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index, normally 0"
    )

    parser.add_argument(
        "--no-firebase",
        action="store_true",
        help="Do not upload predictions to Firebase"
    )

    return parser.parse_args()



# Device


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Firebase


def save_prediction_to_firebase(
    prediction,
    confidence,
    model_mode
):
    if not FIREBASE_DATABASE_URL:
        raise RuntimeError(
            "FIREBASE_DATABASE_URL is not configured"
        )

    endpoint = (
        f"{FIREBASE_DATABASE_URL}/"
        f"{FIREBASE_PATH}.json"
    )

    payload = {
        "prediction": prediction,
        "confidence": confidence,
        "confidence_percent": confidence * 100,
        "model_mode": model_mode,
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    request_parameters = {}

    if FIREBASE_API_KEY:
        request_parameters["key"] = FIREBASE_API_KEY

    response = requests.put(
        endpoint,
        params=request_parameters,
        json=payload,
        timeout=10,
    )

    if not response.ok:
        raise RuntimeError(
            f"Firebase returned HTTP "
            f"{response.status_code}: "
            f"{response.text}"
        )



# Model transform


def create_transform(checkpoint):
    image_size = checkpoint.get(
        "image_size",
        224
    )

    normalization_mean = checkpoint.get(
        "normalization_mean"
    )

    normalization_std = checkpoint.get(
        "normalization_std"
    )

    # New udder model
    if (
        normalization_mean is not None
        and normalization_std is not None
    ):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization_mean,
                std=normalization_std
            ),
        ])

    # Original cow model
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
    ])



# Load disease model


def load_disease_model(
    model_path,
    device
):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    classes = checkpoint["classes"]

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    transform = create_transform(
        checkpoint
    )

    return model, classes, transform



# Disease prediction


def classify_image(
    image,
    model,
    classes,
    transform,
    device
):
    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = classes[
        prediction.item()
    ]

    confidence_value = float(
        confidence.item()
    )

    return predicted_class, confidence_value



# Smooth predictions


def smooth_prediction(history):
    labels = [
        label
        for label, _ in history
    ]

    final_prediction = Counter(
        labels
    ).most_common(1)[0][0]

    matching_confidences = [
        confidence
        for label, confidence in history
        if label == final_prediction
    ]

    final_confidence = sum(
        matching_confidences
    ) / len(matching_confidences)

    return final_prediction, final_confidence



# Find best cow


def find_best_cow(
    frame,
    animal_detector
):
    results = animal_detector(
        frame,
        verbose=False
    )

    best_box = None
    best_confidence = 0.0

    frame_height, frame_width = frame.shape[:2]

    for result in results:
        for box in result.boxes:
            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            # COCO class 19 is cow
            if (
                class_id == 19
                and confidence
                > COW_DETECTION_THRESHOLD
                and confidence
                > best_confidence
            ):
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                x1 = max(0, min(x1, frame_width))
                x2 = max(0, min(x2, frame_width))
                y1 = max(0, min(y1, frame_height))
                y2 = max(0, min(y2, frame_height))

                if x2 > x1 and y2 > y1:
                    best_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    best_confidence = confidence

    return best_box



# Main camera program


def main():
    args = parse_arguments()

    device = get_device()

    print("Device:", device)
    print("Camera mode:", args.mode)

    model_path = MODEL_PATHS[
        args.mode
    ]

    disease_model, classes, transform = (
        load_disease_model(
            model_path,
            device
        )
    )

    print("Disease model loaded:", model_path.name)
    print("Classes:", classes)

    animal_detector = None

    if args.mode == "cow":
        if not YOLO_PATH.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {YOLO_PATH}"
            )

        animal_detector = YOLO(
            YOLO_PATH
        )

        print("YOLO cow detector loaded")

    camera = cv2.VideoCapture(
        args.camera
    )

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}"
        )

    prediction_history = []
    last_firebase_update = 0.0

    window_title = (
        "Whole Cow Disease AI"
        if args.mode == "cow"
        else "Udder Mastitis AI"
    )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame")
                break

            region = None
            display_box = None

            # Whole-cow mode uses YOLO first
            if args.mode == "cow":
                display_box = find_best_cow(
                    frame,
                    animal_detector
                )

                if display_box is not None:
                    x1, y1, x2, y2 = display_box
                    region = frame[y1:y2, x1:x2]
                else:
                    prediction_history.clear()

                    cv2.putText(
                        frame,
                        "No cow detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )

            # Udder mode expects the udder to fill the frame
            else:
                region = frame

                cv2.putText(
                    frame,
                    "Point camera closely at the udder",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2
                )

            if region is not None and region.size > 0:
                rgb = cv2.cvtColor(
                    region,
                    cv2.COLOR_BGR2RGB
                )

                image = Image.fromarray(
                    rgb
                )

                prediction, confidence = (
                    classify_image(
                        image,
                        disease_model,
                        classes,
                        transform,
                        device
                    )
                )

                prediction_history.append(
                    (prediction, confidence)
                )

                if (
                    len(prediction_history)
                    > HISTORY_SIZE
                ):
                    prediction_history.pop(0)

                final_prediction, final_confidence = (
                    smooth_prediction(
                        prediction_history
                    )
                )

                final_percentage = (
                    final_confidence * 100
                )

                label = (
                    f"{final_prediction} "
                    f"({final_percentage:.1f}%)"
                )

                if args.mode == "cow":
                    x1, y1, x2, y2 = display_box

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    text_position = (
                        x1,
                        max(30, y1 - 10)
                    )
                else:
                    text_position = (
                        20,
                        80
                    )

                cv2.putText(
                    frame,
                    label,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                now = time.monotonic()

                if (
                    not args.no_firebase
                    and now - last_firebase_update
                    >= FIREBASE_UPDATE_INTERVAL
                ):
                    last_firebase_update = now

                    try:
                        save_prediction_to_firebase(
                            final_prediction,
                            final_confidence,
                            args.mode,
                        )

                        print(
                            "Saved to Firebase:",
                            args.mode,
                            final_prediction,
                            f"({final_percentage:.1f}%)"
                        )

                    except requests.RequestException as error:
                        print(
                            "Firebase connection failed:",
                            error
                        )

                    except RuntimeError as error:
                        print(
                            "Firebase save failed:",
                            error
                        )

            cv2.putText(
                frame,
                f"Mode: {args.mode} | Press Q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                window_title,
                frame
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

