# Sparse-Video-R1: Advanced Video Understanding with Adaptive Frame Sampling

## 1. What is Sparse-Video-R1?
Sparse-Video-R1 is a lightweight extension for **Vision-Language Models (VLMs)** that teaches them to reason about *which* video frames they actually need.  
Inspired by R1-style fine-tuning for LLMs, we introduce a **multi-round dialogue loop**:

1. **Round t** – The VLM receives a small set of frames plus the question.  
2. The model *thinks* ("Is this enough to answer?").  
3. If not, it *asks* for more evidence by selecting new timestamps (frames).  
4. We sample those frames and repeat until the answer is confidently produced or the turn limit is reached.

This self-curated framing leads to sharper grounding, lower latency, and better accuracy on video QA benchmarks.

---

## 2. Recent Changes
| Date       | Change                                               |
|------------|------------------------------------------------------|
| 2025‑04‑13 | Initial public release – code, docs, and NExT‑QA demo scripts. |

---

## 3. Installation

> **Prerequisites**  
> • CUDA 11.8+ (A100/H100 recommended)  
> • Python 3.10  
> • Conda (Miniconda or Anaconda)  
> • 4x80G GPU (for 3B/7B Qwen-2.5-VL model)

```bash
# 1️⃣ Create a clean environment
conda create -n VideoR1 python=3.10 -y
conda activate VideoR1

# 2️⃣ Install VERL (base repo)
cd verl
pip install -e .

# 3️⃣ Install the runtime stack
pip install vllm==0.8.2          # vLLM backend
pip install flash-attn --no-build-isolation
pip install tensordict==0.6.2
pip install --upgrade torchao    

# 3.1 Transformer & utility extras
pip install git+https://github.com/huggingface/transformers  # Latest transformers for Qwen-VL 2.5 support
pip install accelerate
pip install qwen-vl-utils[decord]  # Faster video decoding (strongly recommended)
```

## 4. Preparing the NExT-QA Demo Dataset
We use the open-sourced NExT-QA dataset from Google Drive. Install gdown if you don't have it:
```bash
pip install gdown
```

Then download and unpack:
```bash
# 4️⃣ Fetch everything
cd /path/to/raw_data
gdown --folder https://drive.google.com/drive/folders/1gKRR2es8-gRTyP25CvrrVtV6aN5UxttF

# 5️⃣ Unzip
cd NExT-QA
unzip NExTVideo.zip
unzip nextqa.zip
unzip test-data-nextqa.zip
```

## 5. Quick Start

Before running the training script, update the following variables in the respective scripts:

- **HF_HOME:** Directory for storing your model.
- **DATA_DIR:** Directory for storing the data processed by `dataset/NExT_QA.py`.
- **RAW_DATA:** Path to where you downloaded the NExT-QA dataset.

Make sure to update these variables in both:
- `scripts/run_qwen2_5_vl_template.sh`
- `train.sh`

Then, launch the minimal training pipeline:
```bash
sh train.sh
```

