# FedResPrompt at scale — GPU runbook (vast.ai)

Validates FedResPrompt vs Federated LoRA on a real frozen LLM + real task in a
non-i.i.d. federated setting (`exp12_fedresprompt_gpu.py`).

## 1. Rent the instance
- Template: **PyTorch (Vast)** (`vastai/pytorch`), CUDA 12.x, **SSH enabled**.
- GPU: 1× **H200 140 GB** (or A100 80 GB). Disk: **≥ 120 GB** for Qwen2.5-32B
  (≈ 64 GB download), **≥ 250 GB** for 72B (≈ 145 GB).

## 2. Connect
Vast shows an SSH command on the instance card:
```bash
ssh -p <PORT> root@<IP>          # e.g. ssh -p XXXXX root@52.24.227.223
```

## 3. Set up + get the script
```bash
pip install -U "transformers>=4.45" datasets peft accelerate
# only if using 8/4-bit (72B): pip install bitsandbytes

# fetch the experiment (public mirror)
wget https://raw.githubusercontent.com/daibeal/esnfed/main/experiments/exp12_fedresprompt_gpu.py
```
(If the file isn't on the public repo yet, just `scp` it up or paste it into a file.)

## 4. Run
Primary run — **Qwen2.5-32B, bf16, SST-2**:
```bash
python exp12_fedresprompt_gpu.py --model Qwen/Qwen2.5-32B --load bf16 \
    --task sst2 --clients 4 --rounds 12 --out results_exp12_32b_sst2.json
```
Headline — **Qwen2.5-72B, 8-bit**:
```bash
python exp12_fedresprompt_gpu.py --model Qwen/Qwen2.5-72B-Instruct --load 8bit \
    --task sst2 --clients 4 --rounds 12 --out results_exp12_72b_sst2.json
```
Topic classification (4-class) — **AG News**:
```bash
python exp12_fedresprompt_gpu.py --task agnews --rounds 15 --out results_exp12_agnews.json
```
Each run prints per-round accuracy and a final summary, and writes a JSON.

Rough time on 1×H200: model download a few minutes; each run ≈ 10–25 min.

## 5. Send results back
Paste the JSON(s) back here and I'll integrate them into the thesis, the results
database, the docs and the gallery:
```bash
cat results_exp12_32b_sst2.json
```

## 6. STOP the instance
**Destroy the instance in the vast.ai console when done** so billing stops.

---
### What it measures
- `fedresprompt.acc` vs `fedlora.acc` — accuracy on held-out test (non-iid clients).
- `fedresprompt.zero_shot` — accuracy with an untrained controller (floor).
- `comm_ratio` — how many× **less** communication FedResPrompt needs per round
  than Federated LoRA (measured from the actual parameter counts).
- `chance` — 1/num_classes baseline.
