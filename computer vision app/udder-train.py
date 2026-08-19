from pathlib import Path
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models



# Configuration


BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "udder_dataset"
MODEL_PATH = BASE_DIR / "udder_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_RATIO = 0.20
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 6



# Reproducibility


random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)



# Device


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)



# Image transformations


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.85, 1.0),
        ratio=(0.90, 1.10)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])



# Load datasets


if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

# Used only to read class names and targets
base_dataset = datasets.ImageFolder(DATASET_DIR)

# Separate objects are required because training and validation
# use different transformations.
train_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=val_transform
)

classes = base_dataset.classes
num_classes = len(classes)

print("Classes:", classes)
print("Number of classes:", num_classes)
print("Total images:", len(base_dataset))

if num_classes != 2:
    print(
        "Warning: This model is intended for two classes: "
        "healthy_udder and mastitis."
    )



# Stratified train/validation split


split_random = random.Random(RANDOM_SEED)

train_indices = []
val_indices = []

for class_id, class_name in enumerate(classes):
    class_indices = [
        index
        for index, target in enumerate(base_dataset.targets)
        if target == class_id
    ]

    split_random.shuffle(class_indices)

    val_count = max(
        1,
        int(len(class_indices) * VALIDATION_RATIO)
    )

    # Ensure at least one image remains for training
    val_count = min(
        val_count,
        len(class_indices) - 1
    )

    class_val_indices = class_indices[:val_count]
    class_train_indices = class_indices[val_count:]

    val_indices.extend(class_val_indices)
    train_indices.extend(class_train_indices)

    print(
        f"{class_name}: "
        f"{len(class_train_indices)} training, "
        f"{len(class_val_indices)} validation"
    )

train_subset = Subset(
    train_dataset,
    train_indices
)

val_subset = Subset(
    val_dataset,
    val_indices
)



# Data loaders


train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)



# Calculate class weights
# 

train_class_counts = [0] * num_classes

for index in train_indices:
    class_id = base_dataset.targets[index]
    train_class_counts[class_id] += 1

total_training_images = sum(train_class_counts)

class_weights = [
    total_training_images / (num_classes * count)
    for count in train_class_counts
]

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

print("Training class counts:", train_class_counts)
print("Class weights:", class_weights)


# 
# Model
# 

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze the earlier feature extraction layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Fine-tune the final ResNet block
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Replace the classifier
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)



# Loss and optimizer


criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)

optimizer = optim.AdamW([
    {
        "params": model.layer4.parameters(),
        "lr": 0.0001
    },
    {
        "params": model.fc.parameters(),
        "lr": 0.001
    }
], weight_decay=0.0001)



# Training variables


best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())

epochs_without_improvement = 0



# Training loop


for epoch in range(EPOCHS):

    # Training
    model.train()

    training_loss = 0.0
    training_correct = 0
    training_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        training_loss += loss.item() * batch_size
        training_total += batch_size

        predictions = outputs.argmax(dim=1)

        training_correct += (
            predictions == labels
        ).sum().item()

    average_training_loss = (
        training_loss / training_total
    )

    training_accuracy = (
        100.0 * training_correct / training_total
    )

    # Validation
    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            validation_loss += loss.item() * batch_size
            validation_total += batch_size

            predictions = outputs.argmax(dim=1)

            validation_correct += (
                predictions == labels
            ).sum().item()

    average_validation_loss = (
        validation_loss / validation_total
    )

    validation_accuracy = (
        100.0 * validation_correct / validation_total
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {average_training_loss:.4f} | "
        f"Train Accuracy: {training_accuracy:.2f}% | "
        f"Val Loss: {average_validation_loss:.4f} | "
        f"Val Accuracy: {validation_accuracy:.2f}%"
    )

    # Save best model
    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy
        best_weights = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": best_weights,
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "normalization_mean": imagenet_mean,
                "normalization_std": imagenet_std,
                "best_validation_accuracy": best_accuracy
            },
            MODEL_PATH
        )

        print(
            f"Best model saved to: {MODEL_PATH}"
        )

    else:
        epochs_without_improvement += 1

    # Early stopping
    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):
        print("Early stopping triggered.")
        break



# Finished


model.load_state_dict(best_weights)

print("\nTraining complete!")
print(
    f"Best validation accuracy: "
    f"{best_accuracy:.2f}%"
)
print(f"Saved model: {MODEL_PATH}")
