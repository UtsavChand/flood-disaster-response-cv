import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image

# --- edit these to match your training ---
CKPT = "checkpoints/unet_best.pth"   # path to your trained weights
ARCH = "Unet"                   # "Unet" or "DeepLabV3Plus"
ENCODER = "resnet34"
SIZE = 256                      # image size used in training (multiple of 32)
# -----------------------------------------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_model = None


def _load():
    global _model
    if _model is None:
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        if isinstance(ck, torch.nn.Module):
            _model = ck
        else:
            m = getattr(smp, ARCH)(encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=1)
            sd = ck.get("model_state_dict") or ck.get("state_dict") or ck
            m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
            _model = m
        _model.eval()
    return _model


@torch.no_grad()
def predict_flood_mask(img: Image.Image, thr=0.5):
    """Returns a boolean flood mask at the original image size."""
    w, h = img.size
    x = np.asarray(img.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)
    p = torch.sigmoid(_load()(t))[0, 0].numpy()
    m = Image.fromarray(((p > thr) * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    return np.asarray(m) > 0
