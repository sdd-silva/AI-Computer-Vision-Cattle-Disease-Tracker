from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image



# Paths


BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "best_model.pth"



# Classes


classes = [
    "foot-and-mouth",
    "healthy",
    "lumpy"
]



# Image Transform


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



# Load Model


model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Model loaded")



# Get Image Path


if len(sys.argv) < 2:
    print(
        "Usage: python predict.py image.jpg"
    )
    exit()


image_path = sys.argv[1]



# Load Image


image = Image.open(
    image_path
).convert("RGB")


image = transform(image)


# Add batch dimension

image = image.unsqueeze(0)



# Prediction


with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(
        output,
        dim=1
    )


    confidence, prediction = torch.max(
        probabilities,
        1
    )



# Result


print()

print(
    "Prediction:",
    classes[prediction.item()]
)


print(
    "Confidence:",
    f"{confidence.item()*100:.2f}%"
)
