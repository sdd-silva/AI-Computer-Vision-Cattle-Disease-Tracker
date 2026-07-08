import cv2
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image


classes = [
    "foot-and-mouth",
    "healthy",
    "lumpy"
]


# Load model

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)


model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location="cpu"
    )
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


camera = cv2.VideoCapture(0)


while True:

    ret, frame = camera.read()

    if not ret:
        break


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    image = Image.fromarray(rgb)

    image = transform(image)

    image = image.unsqueeze(0)


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


    label = classes[prediction.item()]


    text = (
        f"{label} "
        f"{confidence.item()*100:.1f}%"
    )


    cv2.putText(
        frame,
        text,
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    cv2.imshow(
        "Disease Classifier",
        frame
    )


    if cv2.waitKey(1) == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
