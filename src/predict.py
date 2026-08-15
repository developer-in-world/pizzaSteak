from PIL import Image

import torch
import torch.nn.functional as F

from src.model import FoodClassifierCNN
from src.config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES


device = "cuda" if torch.cuda.is_available() else "cpu"

model = FoodClassifierCNN().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()


def predict(image):

    image = image.convert("RGB").resize(IMAGE_SIZE)
    image = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    image = image.float().view(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    image = image.permute(2, 0, 1) / 255.0

    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():

        output = model(image)

        probability = F.softmax(output, dim=1)

        predicted = torch.argmax(probability, dim=1)

    label = CLASS_NAMES[predicted.item()]

    confidence = probability.max().item()

    return label, confidence
