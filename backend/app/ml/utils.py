from PIL import Image
import torch
from torchvision import transforms
import io
from app.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

# PyTorch transform pipeline for MobileNetV3 input
preprocess_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    Takes image bytes, converts to an RGB Pillow image, applies PyTorch transforms,
    and returns a preprocessed batch-dimension tensor.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess_transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension (1, C, H, W)

def preprocess_image_path(image_path: str) -> torch.Tensor:
    """
    Loads an image from disk, converts to RGB, and applies PyTorch transforms.
    """
    image = Image.open(image_path).convert("RGB")
    return preprocess_transform(image)
