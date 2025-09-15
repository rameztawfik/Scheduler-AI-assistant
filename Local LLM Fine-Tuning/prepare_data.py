import pandas as pd
import json
from pathlib import Path

xlsx = "construction.xlsx"   # your Excel file
out_dir = Path("./data")
out_dir.mkdir(exist_ok=True, parents=True)

df = pd.read_excel(xlsx).fillna("").astype(str)

required = ["Predecessor_id","Successor_id","Relationship_typ","Predecessor_activ_status","Successor_activ_status","lag(d)","Predecessor_activ_name","Successor_activ_name"]]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

records = []
for _, r in df.iterrows():
    instr = "Given the predecessor details, predict the successor activity ID and name, and justify briefly."
    inp = (
        f"Predecessor_id: {r['Predecessor_id']}\n"
        f"Predecessor_activ_name: {r['Predecessor_activ_name']}\n"
        f"Relationship_typ: {r['Relationship_typ']}"
    )
    out = r["Successor_activ_name"].strip()
    if out == "":
        continue
    records.append({"instruction": instr, "input": inp, "output": out})

n = len(records)
val_n = max(1, int(n * 0.1))
val = records[:val_n]
train = records[val_n:]

def dump(lst, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in lst:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

dump(train, out_dir / "train.jsonl")
dump(val,   out_dir / "val.jsonl")
print(f"✅ Wrote {len(train)} train and {len(val)} val examples to ./data")