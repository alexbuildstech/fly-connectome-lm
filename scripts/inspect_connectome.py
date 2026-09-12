"""Inspect the MaleCNS v1.0 connectome feather files: schema, size, sample rows."""
import pyarrow.feather as feather
import pandas as pd

base = "/home/z/my-project/data/malecns/"

print("=== connectome-weights (the edge list) ===")
w = feather.read_table(base + "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
print("rows:", w.num_rows, "cols:", w.column_names)
print(w.schema)
df = w.select(w.column_names[:8]).to_pandas()
print(df.head(10).to_string())

print("\n=== body-annotations ===")
a = feather.read_table(base + "body-annotations-male-cns-v1.0-minconf-0.5.feather")
print("rows:", a.num_rows, "cols:", len(a.column_names))
print("columns:", a.column_names[:40])
adf = a.select([c for c in a.column_names if c in
                ("bodyId", "bodyid", "id", "type", "class", "superclass", "group", "instance",
                 "status", "cropped", "size", "pre", "post", "somaSide", "somaNeuromere")]).to_pandas()
print(adf.head(10).to_string())

print("\n=== body-neurotransmitters ===")
nt = feather.read_table(base + "body-neurotransmitters-male-cns-v1.0.feather")
print("rows:", nt.num_rows, "cols:", nt.column_names)
print(nt.to_pandas().head(5).to_string())
