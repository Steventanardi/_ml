import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

model = models.resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_path = "cat.jpg"  
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

_, predicted_idx = torch.max(output, 1)

import json
import urllib.request

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(LABELS_URL) as f:
    labels = [line.decode('utf-8').strip() for line in f.readlines()]

print(f"Predicted: {labels[predicted_idx.item()]}")
