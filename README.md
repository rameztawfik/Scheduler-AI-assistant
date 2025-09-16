# Scheduler-AI-assistant
Concept repo for the usage of an AI as an assistant chat bot for the the project manger, construction mangers and stakeholders to follow up on the planned and actual project status.
Data were exported from Primavera P6 from an older real construction project.
Headers naming are very case sensetive.

# Local LLM Fine-Tuning Guide (assumed 8 GB GPU)

> yet not recommended on windows, better Linux or MAC

## Requirements on local user friendly Laptop/PC

- **OS**: Windows 10/11 (Admin rights required)  
- **GPU**: The Higher the GPU the better
- **RAM**: 16 GB+ (more is better)  
- **Disk**: 25–40 GB free (model + caches + outputs)  
- **Time**: A few hours for installs + training time (varies by dataset size) -- a simple excel was used as a proof case  

> ⚠️ If `bitsandbytes` (4-bit loader) fails on native Windows, try `pip uninstall bitsandbytes -y && pip install bitsandbytes` This avoids Windows build issues while still running locally.
> If still failing, use WSL2 (Ubuntu) and repeat the same steps there (still local).  

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
> It must have headers in the exact order (naming could be changed accordinglly):**Predecessor_id, Successor_id, Relationship_typ, Predecessor_activ_status, Successor_activ_status, lag(d), Predecessor_activ_name, Successor_activ_name**
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

required = ["Predecessor_id","Successor_id","Relationship_typ","Predecessor_activ_status","Successor_activ_status","lag(d)","Predecessor_activ_name","Successor_activ_name"]
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
---

## 5) Fine-tune pythia-2.8b with QLoRA (Axolotl) 
**this model was used due to my laptop capabilities, more stronger model is recommended to be used**
**either by renting online GPU or via your company server** 

Make `config.yaml` in the same folder:

```yaml
# config.yaml — tuned for ~8 GB VRAM
base_model: EleutherAI/pythia-2.8b

# 4-bit quantization (QLoRA)
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true

# LoRA
adapter_type: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
# Pythia/NeoX target modules:
target_modules: ["q_proj","k_proj","v_proj","o_proj","dense_h_to_4h","dense_4h_to_h"]

# Data
datasets:
  - path: ./data/train.jsonl
    type: alpaca
val_set_size: 0.1

# Train efficiency knobs for 8 GB
sequence_len: 512          # keep short to avoid OOM
sample_packing: true
gradient_checkpointing: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 32  # increases effective batch size without VRAM
num_train_epochs: 2
learning_rate: 2e-4
weight_decay: 0.0
lr_scheduler: cosine
warmup_ratio: 0.03

# Logging & output
output_dir: ./outputs/pythia2.8b-lora
logging_steps: 25
eval_steps: 200
save_steps: 200
```
Train:
```bash
axolotl train config.yaml
```
### If you see CUDA OOM:

- Drop sequence_len to 384 or 256
- Or increase gradient_accumulation_steps to 64
- Or reduce LoRA size: lora_r: 4, lora_alpha: 8
> You can attempt a 7B base (e.g., Pythia-6.9b) on 8 GB with sequence_len: 256, lora_r: 4, tighter settings, but 2.8B will be smoother and faster.
---

## 6) Quick sanity test / chat with the sheet (terminal UI)
Create `chat_with_excel.py`:
```python
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "EleutherAI/pythia-2.8b"
ADAPTER = "./outputs/pythia2.8b-lora"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token

print("Loading base model (4-bit)...")
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
device = next(model.parameters()).device
print("Model ready on", device)

df = pd.read_excel("construction.xlsx").fillna("").astype(str)

def prompt_from_row(row, question):
    parts = [f"{c}: {row[c]}" for c in ["ActivityID","ActivityName","Predecessor","Successor"] if c in row.index and row[c].strip()]
    ctx = "\n".join(parts)
    return f"### Instruction:\n{question}\n\n### Input:\n{ctx}\n\n### Response:\n"

while True:
    s = input(f"Row index 0..{len(df)-1} (or 'q'): ").strip()
    if s.lower() == "q": break
    try:
        i = int(s)
        row = df.iloc[i]
    except:
        print("Invalid index"); continue
    q = input("Question (default: 'Predict successor and rationale'): ").strip() or "Predict successor and provide a brief rationale."
    enc = tok(prompt_from_row(row, q), return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=160)
    print(tok.decode(out[0], skip_special_tokens=True))
    print("-----")
```
RUN:
```bash
python chat_with_excel.py
```
You can enter a row index and ask questions like:
what is the successor and provide a brief rationale.”
“If predecessor is missing, what’s the likely next activity?” etc.

---

# On-Cloud LLM Fine-Tuning Guide

## Requirements for on-cloud GPU servis

- For fine-tuning large models, you need at least 40–80 GB VRAM. 
  - use can use RUNPod.io, lambda cloud, paperspace or others.
  - **GPU**: A100 80GB is recommended since easiest setup, affordable, works with Hugging Face/Axolotl out of the box.

---
## 1) Connect to your cloud instance

- Usually all provided services provid a JupyterLab + SSH environment.
  - for coding cells → use JupyterLab (like a notebook).
  - If you prefer terminal commands → open “SSH” from the dashboard.
---

## 2) Environment setup

```bash
# Update system
sudo apt-get update -y && sudo apt-get upgrade -y

# Install Git + Miniconda
sudo apt-get install -y git wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create project folder + conda env
mkdir ~/my_project && cd ~/my_project
conda create -n llm_train python=3.10 -y
conda activate llm_train
```
Install the dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft trl huggingface_hub bitsandbytes
pip install pandas openpyxl tqdm evaluate gradio
pip install axolotl
```
---

## 3) Upload your Excel file

Use the JupyterLab file browser to upload your `construction.xlsx` into `~/my_project`.

Make sure it has the columns:
"Predecessor_id","Successor_id","Relationship_typ","Predecessor_activ_status","Successor_activ_status","lag(d)","Predecessor_activ_name","Successor_activ_name"
---

## 4) Convert Excel → JSONL (for training data)

Create `prepare_data.py` in JupyterLab and paste the following code>>>

```python
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
```

RUN
```bash
python prepare_data.py
```
---

## 5) Training config (GPT-NeoX-20B)

Create `config.yaml`:

```yaml
# Using GPT-NeoX 20B as base
base_model: EleutherAI/gpt-neox-20b

# QLoRA config
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true

# LoRA adapter
adapter_type: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: ["query_key_value"]

# Data
datasets:
  - path: ./data/train.jsonl
    type: alpaca
val_set_size: 0.1

# Training
sequence_len: 1024
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
num_train_epochs: 2
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.03

# Misc
gradient_checkpointing: true
sample_packing: true
output_dir: ./outputs/neox20b-lora
logging_steps: 20
save_steps: 200
eval_steps: 200

```
Train 
```bash
axolotl train config.yaml
```

Load GPT-NeoX-20B (quantized)
Train LoRA adapters on the Excel-derived dataset
Save to `outputs/neox20b-lora`
---

## 6) Chat with the Excel sheet

Create `chat_with_excel.py`:
```python
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE, ADAPTER = "EleutherAI/gpt-neox-20b", "./outputs/neox20b-lora"

bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True)

tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
device = next(model.parameters()).device

df = pd.read_excel("construction.xlsx").fillna("").astype(str)

def make_prompt(row, question):
    ctx = "\n".join([f"{c}: {row[c]}" for c in df.columns if row[c].strip()])
    return f"### Instruction:\n{question}\n\n### Input:\n{ctx}\n\n### Response:\n"

while True:
    s = input(f"Row index (0–{len(df)-1}) or q: ")
    if s=="q": break
    row = df.iloc[int(s)]
    q = input("Question: ") or "Predict successor and justify."
    enc = tok(make_prompt(row,q), return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=160)
    print(tok.decode(out[0], skip_special_tokens=True))
```
RUN
```bash
python chat_with_excel.py
```
---




