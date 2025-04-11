<h1 style="text-align: center;">VideoR1</h1>

### Installation
We follow this [guideline](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) in setting up our enviorments. 
```bash
# Create the conda environment
conda create -n verl python==3.10
conda activate verl

# Install verl
cd verl
pip3 install -e .

# Install the latest stable version of vLLM
pip3 install vllm==0.8.2

# Install flash-attn
pip3 install flash-attn --no-build-isolation

pip install tensordict==0.6.2
pip install --upgrade torchao
```

Please also consider install the lastest transformers to ensure support for QwenVL2.5
```bash
pip install git+https://github.com/huggingface/transformers accelerate
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-vl-utils[decord]
```