import cv2
import numpy as np
import torch
from PIL import Image

from src.dataset import get_transforms
from src.model import build_model

_tf = get_transforms(train=False)  # resize + normalize only, no augmentation


def load_model(path, name="unet", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name, pretrained=False)  # no download needed, weights come from the checkpoint
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_proba(model, image):
    """image: PIL image or numpy array. Returns a float map (0-1) at the ORIGINAL size."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    h, w = image.shape[:2]
    device = next(model.parameters()).device

    x = _tf(image=image)["image"].unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def predict(model, image, threshold=0.5):
    """Returns a binary mask (uint8, 0 or 1) at the original image size."""
    return (predict_proba(model, image) > threshold).astype(np.uint8)