# Examples:
# python predict.py cow.jpg --mode cow
# python predict.py udder.jpg --mode udder

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from urllib.parse import quote

from PIL import Image, ImageOps
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATHS = {
    "cow": BASE_DIR / "best_model.pth",
    "udder": BASE_DIR / "udder_model.pth",
}

FIREBASE_PROJECT_ID = os.getenv(
    "FIREBASE_PROJECT_ID",
    ""
)

FIREBASE_API_KEY = os.getenv(
    "FIREBASE_API_KEY",
    ""
)


# --------------------------------------------------
# Arguments
# --------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Predict a cattle disease and optionally save "
            "the result to Cloud Firestore."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the image to classify"
    )

    parser.add_argument(
        "--mode",
        choices=["cow", "udder"],
        default="cow",
        help=(
            "Use 'cow' for the original whole-cow model or "
            "'udder' for healthy-udder versus mastitis."
        )
    )

    parser.add_argument(
        "--collection",
        default=os.getenv(
            "FIREBASE_COLLECTION",
            "predictions"
        ),
        help=(
            "Firestore collection name "
            "(default: predictions)"
        )
    )

    parser.add_argument(
        "--no-firebase",
        action="store_true",
        help="Print the result without saving to Firestore"
    )

    return parser.parse_args(), parser


# --------------------------------------------------
# Firestore
# --------------------------------------------------

def save_prediction(
    collection,
    prediction,
    confidence,
    image_path,
    model_mode
):
    if not FIREBASE_PROJECT_ID:
        raise RuntimeError(
            "FIREBASE_PROJECT_ID is not configured"
        )

    if not FIREBASE_API_KEY:
        raise RuntimeError(
            "FIREBASE_API_KEY is not configured"
        )

    collection_name = quote(
        collection,
        safe=""
    )

    endpoint = (
        "https://firestore.googleapis.com/v1/"
        f"projects/{FIREBASE_PROJECT_ID}/"
        "databases/(default)/"
        f"documents/{collection_name}"
    )

    payload = {
        "fields": {
            "prediction": {
                "stringValue": prediction
            },
            "confidence": {
                "doubleValue": confidence
            },
            "confidence_percent": {
                "doubleValue": confidence * 100
            },
            "image_name": {
                "stringValue": image_path.name
            },
            "model_mode": {
                "stringValue": model_mode
            },
            "created_at": {
                "timestampValue": datetime.now(
                    timezone.utc
                ).isoformat()
            },
        }
    }

    response = requests.post(
        endpoint,
        params={
            "key": FIREBASE_API_KEY
        },
        json=payload,
        timeout=20,
    )

    if not response.ok:
        try:
            firebase_message = (
                response.json()["error"]["message"]
            )
        except (KeyError, TypeError, ValueError):
            firebase_message = response.text

        raise RuntimeError(
            f"Firestore returned HTTP "
            f"{response.status_code}: "
            f"{firebase_message}"
        )

    document_name = response.json()["name"]

    return document_name.rsplit("/", 1)[-1]


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model(model_path):
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

    model.eval()

    return model, classes, checkpoint


# --------------------------------------------------
# Transform selection
# --------------------------------------------------

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

    # New udder model uses ImageNet normalization
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
            )
        ])

    # Original model was trained without normalization
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
    ])



# Prediction


def predict_image(
    image_path,
    model,
    classes,
    transform
):
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
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



# Main


def main():
    args, parser = parse_arguments()

    image_path = Path(
        args.image
    ).expanduser().resolve()

    if not image_path.is_file():
        parser.error(
            f"Image does not exist: {image_path}"
        )

    model_path = MODEL_PATHS[args.mode]

    if not model_path.is_file():
        parser.error(
            f"Model does not exist: {model_path}"
        )

    model, classes, checkpoint = load_model(
        model_path
    )

    transform = create_transform(
        checkpoint
    )

    predicted_class, confidence_value = (
        predict_image(
            image_path,
            model,
            classes,
            transform
        )
    )

    print("Model mode:", args.mode)
    print("Prediction:", predicted_class)
    print(
        "Confidence:",
        f"{confidence_value * 100:.2f}%"
    )

    if args.no_firebase:
        print("Firestore save skipped.")
        return 0

    try:
        document_id = save_prediction(
            args.collection,
            predicted_class,
            confidence_value,
            image_path,
            args.mode,
        )
    except Exception as error:
        print(
            f"Firebase save failed: {error}",
            file=sys.stderr
        )
        return 1

    print(
        f"Saved to Firestore collection "
        f"'{args.collection}' with document ID: "
        f"{document_id}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
