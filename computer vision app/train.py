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

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


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
num_classes = len(dataset.classes)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
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

best_accuracy = 0

# Training


epochs = 15

for epoch in range(epochs):

    # -----------------------
    # Training
    # -----------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(outputs, 1)


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = 100 * correct / total



    # -----------------------
    # Validation
    # -----------------------

    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)


            val_correct += (
                predicted == labels
            ).sum().item()



    val_accuracy = (
        100 * val_correct / val_total
    )


    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {running_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


# Save Model


if val_accuracy > best_accuracy:

    best_accuracy = val_accuracy

    torch.save(
        model.state_dict(),
        "best_model.pth"
    )

    print("Best model saved!")
