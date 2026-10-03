import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


def build_zones(masks, seed=42):
    """masks: dict name -> binary mask array (use model predictions or ground truth)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, (name, m) in enumerate(masks.items()):
        ff = float((np.asarray(m) > 0).mean())
        rows.append(dict(
            zone_id=f"Z{i+1}", source=name, flood_frac=round(ff, 3),
            population=int(rng.integers(3000, 40000)),
            road_access=round(float(np.clip(1 - 0.7 * ff + rng.normal(0, 0.1), 0.05, 1)), 2),
            hospital_km=round(float(rng.uniform(1, 25)), 1),
            x=round(float(rng.uniform(0, 50)), 1), y=round(float(rng.uniform(0, 50)), 1),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    files = sorted(Path("data/Mask").glob("*"))
    files = files[:: max(1, len(files) // n)][:n]
    masks = {f.stem: np.array(Image.open(f).convert("L")) > 127 for f in files}
    df = build_zones(masks)
    df.to_csv("data/zones.csv", index=False)
    print(df)