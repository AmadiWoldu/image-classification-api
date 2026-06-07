import torch
import torchvision.models as models
import torch.nn as nn

from PIL import Image

from model.preprocess import preprocess_image

# =====================================
# Device 
# =====================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================================
# Class Names
# =====================================

CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


# =====================================
# Load Model
# =====================================

model = models.resnet18(weights = None)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

checkpoint = torch.load(f = 'model/artifacts/model_state_dict.pth')

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()


def predict_image(image: str):


    image_tensor = preprocess_image(image).to(device)

    with torch.inference_mode():

        logits = model(image_tensor)

        probabilities = torch.softmax(logits, dim = 1)

        confidence, prediction = torch.max(
            probabilities,
            dim= 1
        )

        return {
            "prediction": CLASS_NAMES[prediction.item()],
            'confidence': round(confidence.item() * 100, 2)
        }
    

if __name__ == "__main__":
    image_path = input("Enter image path: ")

    result = predict_image(image_path)

    print(result)