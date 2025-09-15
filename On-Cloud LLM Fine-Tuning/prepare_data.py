import pandas as pd, json
from pathlib import Path

xlsx = "construction.xlsx"
out = Path("./data"); out.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(xlsx).fillna("").astype(str)

records = []
for _, r in df.iterrows():
    instr = "Given the activity details, predict the successor activity."
    inp = f"Predecessor_id: {r['Predecessor_id']}\Successor_id: {r['Successor_id']}\Relationship_typ: {r['Relationship_typ']}\Predecessor_activ_status: {r['Predecessor_activ_status']}\Successor_activ_status: {r['Successor_activ_status']}\lag(d): {r['lag(d)']}\Predecessor_activ_name: {r['Predecessor_activ_name']}\Successor_activ_name: {r['Successor_activ_name']}"
    outp = r["Successor"].strip()
    if outp: records.append({"instruction": instr, "input": inp, "output": outp})

n = len(records)
val_n = max(1, int(n*0.1))
train, val = records[val_n:], records[:val_n]

def dump(lst, path): 
    with open(path,"w",encoding="utf-8") as f:
        for ex in lst: f.write(json.dumps(ex)+"\n")

dump(train,"data/train.jsonl"); dump(val,"data/val.jsonl")
print(f"✅ {len(train)} train / {len(val)} val examples ready")