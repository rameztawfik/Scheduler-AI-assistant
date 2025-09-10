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