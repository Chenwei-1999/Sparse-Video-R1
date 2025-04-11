<h1 style="text-align: center;">FramesR1: Avdance Video Understanding with Adaptive Frames Sampling</h1>

## 1. What is FramesR1?
FramesR1 is a lightweight extension for **Vision‑Language Models (VLMs)** that teaches them to reason about *which* video frames they actually need.  
Inspired by R1‑style fine‑tuning for LLMs, we introduce a **multi‑round dialogue loop**:

1. **Round *t*** – The VLM receives a small set of frames plus the question.  
2. The model *thinks* (“Is this enough to answer?”).  
3. If not, it *asks* for more evidence by selecting new timestamps.  
4. We sample those frames and repeat until the answer is confidently produced or the turn limit is reached.

This self‑curated framing leads to sharper grounding, lower latency, and better accuracy on video QA benchmarks.

---

## 2. Recent Changes
| Date       | Change                                               |
|------------|------------------------------------------------------|
| 2025‑04‑11 | Initial public release – code, docs, and NExT‑QA demo scripts. |

---

## 3. Installation

> **Prerequisites**  
> • CUDA 11.8+ (A100/H100 recommended)  
> • Python 3.10  
> • Conda (Miniconda or Anaconda)

```bash
# 1️⃣  Create a clean environment
conda create -n verl python=3.10 -y
conda activate verl

# 2️⃣  Clone and install VERL (base repo)
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .

# 3️⃣  Install the runtime stack
pip install vllm==0.8.2          # vLLM backend
pip install flash-attn --no-build-isolation
pip install tensordict==0.6.2
pip install --upgrade torchao    
```
3.1  Transformer & utility extras
```bash
# Latest transformers for Qwen‑VL 2.5 support
pip install git+https://github.com/huggingface/transformers
pip install accelerate

# Faster video decoding (strongly recommended)
pip install qwen-vl-utils[decord]
```

## 4. Preparing the NExT‑QA Demo Dataset
We use the open-sourced NExT-QA dataset from Google Drive.
Install gdown once if you don’t have it:
```bash
pip install gdown
```
Then download and unpack:
```bash
# 4️⃣  Fetch everything
cd /path/to/storage/data
gdown --folder https://drive.google.com/drive/folders/1gKRR2es8-gRTyP25CvrrVtV6aN5UxttF

# 5️⃣  Unzip
cd NExT-QA
unzip NExTVideo.zip
unzip nextqa.zip
unzip test-data-nextqa.zip
```

## 5. Quick Start
```bash
sh train.sh
```
The script will start the minimal training pipeline based on GRPO.