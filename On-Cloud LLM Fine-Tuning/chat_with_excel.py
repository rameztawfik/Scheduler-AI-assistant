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