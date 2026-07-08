from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# Dataset Location


BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"


# Image Transformations


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Load Dataset


dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)


# Split Dataset


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

# NOTE:
# This changes the transform for the underlying dataset.
# We'll improve this in the next version.
val_dataset.dataset.transform = val_transform


# DataLoaders


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# Load Model


model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Replace final layer for 3 classes
model.fc = nn.Linear(
    model.fc.in_features,
    3
)


# Device


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = model.to(device)

print("Using device:", device)


# Loss + Optimizer


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Dataset Information


print("\nClasses:")
print(dataset.class_to_idx)

print("\nTraining Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))

print("\nTraining Batches:", len(train_loader))
print("Validation Batches:", len(val_loader))

images, labels = next(iter(train_loader))

print("\nImage Batch Shape:", images.shape)
print("Label Batch Shape:", labels.shape)


# Training


epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {running_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )


# Save Model


torch.save(
    model.state_dict(),
    "model.pth"
)

print("\nTraining Complete!")
print("Model saved as model.pth")
