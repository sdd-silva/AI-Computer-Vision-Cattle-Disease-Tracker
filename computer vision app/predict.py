# Run with:
# python predict.py image_name.file_type

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from urllib.parse import quote

from PIL import Image
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_model.pth"
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
FIREBASE_API_KEY = os.getenv(
    "FIREBASE_API_KEY",
    "",
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Predict a cattle disease and save the result to Cloud Firestore."
        )
    )
    parser.add_argument("image", help="Path to the image to classify")
    parser.add_argument(
        "--collection",
        default=os.getenv("FIREBASE_COLLECTION", "predictions"),
        help="Firestore collection name (default: predictions)",
    )
    return parser.parse_args(), parser


def save_prediction(collection, prediction, confidence, image_path):
    collection_name = quote(collection, safe="")
    endpoint = (
        "https://firestore.googleapis.com/v1/"
        f"projects/{FIREBASE_PROJECT_ID}/databases/(default)/"
        f"documents/{collection_name}"
    )
    payload = {
        "fields": {
            "prediction": {"stringValue": prediction},
            "confidence": {"doubleValue": confidence},
            "confidence_percent": {"doubleValue": confidence * 100},
            "image_name": {"stringValue": image_path.name},
            "created_at": {
                "timestampValue": datetime.now(timezone.utc).isoformat()
            },
        }
    }
    response = requests.post(
        endpoint,
        params={"key": FIREBASE_API_KEY},
        json=payload,
        timeout=20,
    )
    if not response.ok:
        try:
            firebase_message = response.json()["error"]["message"]
        except (KeyError, TypeError, ValueError):
            firebase_message = response.text
        raise RuntimeError(
            f"Firestore returned HTTP {response.status_code}: "
            f"{firebase_message}"
        )

    document_name = response.json()["name"]
    return document_name.rsplit("/", 1)[-1]


def main():
    args, parser = parse_arguments()
    image_path = Path(args.image).expanduser().resolve()

    if not image_path.is_file():
        parser.error(f"Image does not exist: {image_path}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    predicted_class = classes[prediction.item()]
    confidence_value = float(confidence.item())

    print("Prediction:", predicted_class)
    print("Confidence:", f"{confidence_value * 100:.2f}%")

    try:
        document_id = save_prediction(
            args.collection,
            predicted_class,
            confidence_value,
            image_path,
        )
    except Exception as error:
        print(f"Firebase save failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved to Firestore collection '{args.collection}' "
        f"with document ID: {document_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
