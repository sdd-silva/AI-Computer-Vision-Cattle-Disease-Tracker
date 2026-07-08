from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


# Dataset location
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"


# Training augmentation
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])


# Validation transform
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# Load dataset
dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)


# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


# Validation should not use augmentation
val_dataset.dataset.transform = val_transform


# Create loaders
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


print("Classes:")
print(dataset.class_to_idx)

print()

print("Training batches:",
      len(train_loader))

print("Validation batches:",
      len(val_loader))


# Test one batch

images, labels = next(iter(train_loader))

print()

print("Image batch shape:")
print(images.shape)

print("Label batch shape:")
print(labels.shape)