# Flood Response Planner — Prototype

A small, classroom-demonstration prototype that combines flood-image segmentation with a simulated emergency-resource allocation system. A coordinator can inspect a flood prediction, review a proposed allocation, and approve, edit, or reject it.

> **Prototype limitation:** the image model estimates flooded pixels. Population, road access, hospital distance, and zone coordinates are simulated in the starter zones or entered manually. This is not a validated operational emergency-response system.

## What is implemented

### Existing computer-vision work

- Binary flood segmentation using U-Net with a ResNet34 encoder; DeepLabV3+ is also available for comparison.
- Image/mask dataset loading, fixed train/validation/test splits, training augmentation, and BCE + Dice training loss.
- Training metrics include IoU, Dice, precision, and recall. The recorded validation result is approximately 0.827 IoU and 0.905 Dice; **test-split results still need to be reported**.
- Prediction helpers return a flood probability map and a thresholded mask.

### Aarohi's additions on the `aarohi` branch

- A Gymnasium flood-resource allocation environment and PPO training/evaluation workflow.
- Four heuristic policies for comparison: severity-first, nearest-first, largest-need-first, and random.
- Resource inventory parsing/validation, approval/rejection logging, and template/optional Claude briefings.
- A Streamlit dashboard for zone/resource inspection, strategy selection, allocation review, human approval, briefings, and evaluation results.
- An **Add from image** flow: upload an image, run the trained U-Net, preview the mask overlay, enter the non-image zone attributes, then add/update a zone used by the allocation planner.

The environment assigns one resource unit per decision step. It randomizes unit order and rewards useful allocations to higher-priority zones while applying a travel-distance discount. PPO is compared with the four heuristics using need coverage, severity-weighted coverage, worst-zone coverage, mean travel distance, and wasted units.

## Repository layout

```text
app/dashboard.py           Streamlit prototype
.streamlit/config.toml     Dashboard theme
data/split.csv             Fixed dataset split; do not regenerate
data/resources.json        Starter resource inventory
data/zones.csv             Small starter scenario required by the dashboard
models/ppo_alloc.zip       Trained PPO resource-allocation policy
notebooks/                 Exploration, preprocessing, CV training, prediction demo
results/                   CV history/curves and RL comparison results
src/approval.py            Plan generation, validation, decision logging
src/briefing.py            Template and optional Claude summaries
src/dataset.py             CV dataset and transforms
src/env.py                 Resource-allocation simulator
src/evaluate.py            PPO-versus-heuristic evaluation
src/heuristics.py          Rule-based allocation policies
src/impact.py              Flood fraction and zone-table helpers
src/infer.py               U-Net inference used by the dashboard upload flow
src/model.py               Segmentation model builders
src/predict.py             General CV prediction helpers
src/resources.py           Resource/zone loading and validation
src/train.py               CV training
src/train_rl.py            PPO training
```

## Setup (Linux/macOS)

Clone the Aarohi branch to get the prototype files and PPO policy:

```bash
git clone -b aarohi https://github.com/UtsavChand/flood-disaster-response-cv.git
cd flood-disaster-response-cv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead.
In VS Code, select `.venv` as the Python interpreter.

## Dataset and U-Net checkpoint

The CV dataset is the Kaggle **Flood Area Segmentation** dataset. Download it separately and place paired images and masks here:

```text
data/Image/
data/Mask/
```

Use the committed `data/split.csv` so everyone uses the same train/validation/test assignment. Do not regenerate it. The dataset and U-Net weights are intentionally not tracked by Git.

For image-upload inference in the dashboard, obtain the trained `unet_best.pth` from the teammate who trained it and place it at:

```text
checkpoints/unet_best.pth
```

Alternatively, retrain the CV model with `python -m src.train --model unet --epochs 30`. The U-Net checkpoint is ignored by Git; **do not commit it**.

The small starter `data/zones.csv` and `data/resources.json` are committed so the dashboard has a demonstration scenario immediately after cloning. To rebuild zones from existing dataset masks, run `python -m src.impact 8`. That command measures ground-truth masks; the dashboard's uploaded-image flow instead uses model predictions from `src/infer.py`.

## Run the prototype

From the repository root with `.venv` activated:

```bash
python -m streamlit run app/dashboard.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). Suggested demo flow:

1. In **Add from image**, upload an image. This requires `checkpoints/unet_best.pth`.
2. Review the predicted flood overlay and flooded share.
3. Add it as a new zone or update an existing one. Enter/check population, road access, hospital distance, and schematic coordinates; the image alone cannot supply these values.
4. In the sidebar, select PPO or one of the heuristic strategies. The plan is recomputed for the current zones and available resources.
5. In **Plan**, inspect/edit the recommendation, then approve or reject it. Decisions append to `results/decisions.jsonl`.
6. In **Briefing**, generate a summary. Without an API key it uses the built-in template. To enable Claude, set `ANTHROPIC_API_KEY` in your shell; never commit the key.
7. In **Results**, compare the recorded policy evaluation metrics.

The PPO environment supports up to 12 zones. The map is a schematic x/y scatter plot, not a georeferenced map.

## Train and evaluate PPO (optional)

The trained policy is included at `models/ppo_alloc.zip`, so PPO does not need retraining to run the demo. To retrain it, which overwrites the policy:

```bash
python -m src.train_rl
```

The default training budget is 300,000 timesteps. To set a different budget, pass it as the first argument, for example `python -m src.train_rl 50000`.

Run a 200-scenario comparison of the available policies and refresh the results file:

```bash
python -m src.evaluate
```

PPO is not guaranteed to outperform the heuristics. Report the measured comparison as-is; training completion alone does not demonstrate superiority.

## Current limitations and next work

1. Evaluate the CV model over the held-out test split and record aggregate test metrics; validation metrics or a single sample are not test-set evidence.
2. Replace or clearly retain as synthetic the population, road, hospital, and coordinate fields. Real geographic claims require aligned GIS/population/infrastructure data and georeferenced imagery.
3. Re-run PPO and heuristic evaluation after final environment/reward changes; document the metrics and assumptions.
4. Make resource edits persistent if the demo needs them retained across browser sessions (the starter inventory is `data/resources.json`).
5. Keep the human coordinator in control; this prototype must not be used to dispatch real emergency resources.

There is currently **no FastAPI backend**; Streamlit calls the Python modules directly. A separate API service is optional for this prototype.

## Git and artifact rules

- Work on a feature branch and use small, regular commits.
- Do not commit `data/Image/`, `data/Mask/`, `.venv/`, or any `.pth` CV checkpoint.
- `models/ppo_alloc.zip` is the small PPO policy needed for the current demo and is included on this branch.
- Keep API keys and other secrets out of source control.