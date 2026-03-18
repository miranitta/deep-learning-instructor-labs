import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path("src").resolve()))

from model_resnet import build_resnet18
from utils import load_checkpoint

# Class names
CLASSES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

MODEL_PATH = "results/models/resnet_best.pt"

st.title('CIFAR-10 Image Classifier')
st.write('Upload an image and get predictions from your trained ResNet model.')

# Load model once
@st.cache_resource
def load_model():
    model = build_resnet18(num_classes=10, pretrained=False)
    model = load_checkpoint(model, MODEL_PATH, map_location="cpu")
    model.eval()
    return model

model = load_model()

# Transform (IMPORTANT: must match training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])

uploaded = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption='Uploaded image', use_container_width=True)

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    st.success(f'Prediction: {CLASSES[pred]}')
    st.write(f'Confidence: {probs[pred]:.3f}')

    st.write('Top-3 predictions:')
    topk = torch.topk(probs, k=3)

    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        st.write({
            'class': CLASSES[idx],
            'probability': round(score, 4)
        })