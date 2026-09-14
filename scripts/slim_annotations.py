"""Create slim annotation table + neurotransmitter join for the repo."""
import os as _os
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root
import pyarrow.feather as feather
import pandas as pd
import numpy as np

BASE = f"{_R}/data/malecns"

ann = feather.read_table(f"{BASE}/body-annotations-male-cns-v1.0-minconf-0.5.feather").to_pandas()
print("annotations:", ann.shape)

keep_cols = [c for c in ["bodyId", "type", "instance", "class", "subclass", "superclass",
                         "group", "supertype", "status", "statusLabel", "somaSide",
                         "somaNeuromere", "dimorphism", "flywireType", "hemibrainType",
                         "entryNerve", "exitNerve", "cropped"] if c in ann.columns]
slim = ann[keep_cols].copy()
slim.to_parquet(f"{BASE}/processed/annotations_slim.parquet")
print("slim saved:", slim.shape)
for c in ["class", "superclass", "statusLabel", "dimorphism"]:
    if c in slim.columns:
        print(c, "top:", slim[c].value_counts().head(6).to_dict())

nt = feather.read_table(f"{BASE}/body-neurotransmitters-male-cns-v1.0.feather").to_pandas()
print("\nnt table:", nt.shape, list(nt.columns))
print(nt.head(3).to_string())
nt.to_parquet(f"{BASE}/processed/neurotransmitters.parquet")
