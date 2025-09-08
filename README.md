# Scheduler-AI-assistant
Concept repo for the usage of an AI as an assistant chat bot for the the project manger, construction mangers and stakeholders to follow up on the planned and actual project status.
Data were exported from Primavera P6 from an older real construction project.
Headers naming are very case sensetive.

# Local LLM Fine-Tuning Guide (assumed 8 GB GPU)

## Requirements

- **OS**: Windows 10/11 (Admin rights required)  
- **GPU**: The Higher the GPU the better
- **RAM**: 16 GB+ (more is better)  
- **Disk**: 25–40 GB free (model + caches + outputs)  
- **Time**: A few hours for installs + training time (varies by dataset size) -- a simple excel was used as a proof case  

> ⚠️ If `bitsandbytes` (4-bit loader) fails on native Windows, switch to **WSL2 (Ubuntu)** *ASK AI Tool*.  
This avoids Windows build issues while still running locally.

---

## 1) Install the Basics (once)

- [Visual Studio Code](https://code.visualstudio.com/)  
  (Editor — open it when creating/editing files)  

- [Python 3.10](https://www.python.org/downloads/release/python-3100/)  
  *(During install, tick **“Add Python to PATH.””*)  

- [Git](https://git-scm.com/download/win)  

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)  

- **NVIDIA driver** up to date (via GeForce Experience or NVIDIA site).  
  *(No need to install CUDA toolkit; PyTorch will fetch the runtime.)*  

---

## 2) Create Project & Python Environment

Open **Anaconda Prompt** (or VS Code → Terminal): each line at a time,

```powershell
mkdir %USERPROFILE%\my_project
cd %USERPROFILE%\my_project

conda create -n llm_local python=3.10 -y
conda activate llm_local

## Install PyTorch & Libraries

```bash
# PyTorch (CUDA 12.1 wheel; works if your system driver is 12.x-compatible)
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core LLM tooling
pip install transformers datasets accelerate peft trl huggingface_hub

# Quantization + Excel + utilities
pip install bitsandbytes openpyxl pandas tqdm evaluate gradio

# (Optional) Axolotl trainer (we'll use it)
pip install axolotl

```
---

## 3) Put your Excel in the project folder


Place your file as:

```makefile

C:\Users\<YOU>\my_project\construction.xlsx


```

It must have headers in the exact order (naming could be changed accordinglly):**Predecessor_id, Successor_id, Relationship_typ, Predecessor_activ_status, Successor_activ_status, lag(d), Predecessor_activ_name, Successor_activ_name**

---

## 4) Make the data converter (Excel → JSONL) // so it can be trained 

In VS Code, create a file prepare_data.py in my_project and paste:

```python
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


```
Run it (Anaconda Prompt in my_project):

```bash
python prepare_data.py
```

You should now have data/train.jsonl and data/val.jsonl.