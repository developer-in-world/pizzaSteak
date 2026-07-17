from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from src.model import FoodClassifierCNN
from src.config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES


device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


model = FoodClassifierCNN().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()


def predict(image):

    image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():

        output = model(image)

        probability = F.softmax(output, dim=1)

        predicted = torch.argmax(probability, dim=1)

    label = CLASS_NAMES[predicted.item()]

    confidence = probability.max().item()

    return label, confidence