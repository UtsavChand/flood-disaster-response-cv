import json
import pandas as pd

TYPES = ["boat", "medical", "truck"]


def units_from_records(records):
    units = []
    for r in records:
        if r["type"] not in TYPES or int(r["qty"]) < 0:
            raise ValueError(f"Bad resource row: {r}")
        units += [(TYPES.index(r["type"]), float(r["x"]), float(r["y"]))] * int(r["qty"])
    return units


def load_units(path="data/resources.json"):
    return units_from_records(json.load(open(path)))


def load_zones(path="data/zones.csv"):
    return pd.read_csv(path)