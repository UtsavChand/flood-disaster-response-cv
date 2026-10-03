import json
import time
import pandas as pd
from src.resources import TYPES


def plan_from_log(env, zones):
    df = pd.DataFrame(env.log, columns=["zone_idx", "type_idx", "km", "useful"])
    df = df[df.useful]
    g = df.groupby(["zone_idx", "type_idx"]).agg(qty=("km", "size"), avg_km=("km", "mean")).reset_index()
    g["zone_id"] = zones.zone_id.iloc[g.zone_idx].values
    g["type"] = [TYPES[i] for i in g.type_idx]
    return g[["zone_id", "type", "qty", "avg_km"]].round(1).reset_index(drop=True)


def validate(plan, available):
    used = plan.groupby("type").qty.sum().to_dict()
    return [f"{t}: {u} assigned, {available.get(t, 0)} available"
            for t, u in used.items() if u > available.get(t, 0)]


def log_decision(status, plan, note="", path="results/decisions.jsonl"):
    with open(path, "a") as f:
        f.write(json.dumps({"ts": time.time(), "status": status, "note": note,
                            "plan": plan.to_dict("records")}) + "\n")