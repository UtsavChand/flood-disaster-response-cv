# Flood Disaster Relief System (Computer Vision Segmentation)

Web application that detects flooded areas in aerial, drone, or satellite images
using a U-Net (ResNet34 encoder) and reports the estimated flooded percentage.

## Project structure

```
data/
  Image/, Mask/      # Kaggle dataset (NOT in Git, download separately)
  split.csv          # fixed train/val/test split (in Git, do not regenerate)
notebooks/           # data exploration, preprocessing, training results
src/
  dataset.py         # FloodDataset + transforms
  model.py           # build_model("unet" | "deeplab")
  train.py           # training loop
  predict.py         # load_model, predict, predict_proba
checkpoints/         # trained weights (NOT in Git, download separately)
results/             # training curves and history
```

## Setup

```powershell
git clone <REPO_URL>
cd flood-disaster-response
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter/kernel (Ctrl+Shift+P, "Python: Select Interpreter").

## Get the data

1. Download the Kaggle "Flood Area Segmentation" dataset.
2. Put the images in `data/Image/` and the masks in `data/Mask/`.

Seven mismatched image/mask pairs (14, 15, 2052, 2053, 1061, 1079, 3059) were
removed from `split.csv`, leaving 283 images. **Do not regenerate `split.csv`**
(the split cell in the preprocessing notebook is for reference only). Everyone must
use the same file so results are comparable.

## Get the trained model

Download `unet_best.pth` from: **<PASTE LINK HERE>**
and place it at `checkpoints/unet_best.pth`. No training is needed.

## Use the trained model

Run from the **project root**:

```python
import numpy as np
from PIL import Image
from src.predict import load_model, predict, predict_proba

model = load_model("checkpoints/unet_best.pth", "unet")

img = Image.open("data/Image/1.jpg").convert("RGB")
mask = predict(model, img)            # uint8 array (0/1), same size as the image
prob = predict_proba(model, img)      # float map (0-1), same size as the image

flooded_pct = mask.mean() * 100
print(f"{flooded_pct:.1f}% flooded")

# overlay: blend blue onto flooded pixels
arr = np.array(img)
overlay = arr.copy()
overlay[mask == 1] = (0.5 * arr[mask == 1] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
Image.fromarray(overlay).save("overlay.png")
```

`predict` resizes the image to 256x256, normalizes it, runs the model, and returns
a binary mask at the original image size (threshold 0.5).

## Train (only if you want to retrain)

```powershell
python -m src.train --model unet --epochs 30
python -m src.train --model deeplab --epochs 30   # comparison model
```

Training overwrites `checkpoints/<model>_best.pth`. Curves and logs are saved in `results/`.

## Results

| Model | Val IoU | Val Dice |
|---|---|---|
| U-Net (ResNet34) | 0.827 | 0.905 |

Test-set results: _to be added by Person 3_.

## Team rules

- Work on branches and merge through pull requests.
- Never commit the `data/Image`, `data/Mask`, or `.pth` files.
- Evaluate only on the test split.