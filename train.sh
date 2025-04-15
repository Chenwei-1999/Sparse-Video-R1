#!/bin/bash

# Set environment variables
export RAW_DATA=/home/zye52/scr4_hlee283/zye52/NExT-QA
export DATA_DIR=/home/zye52/scr4_hlee283/zye52/NExT-QA-processed-data

# Process training and validation data
python dataset/NExT_QA.py --mode train --n 1000 --output_dir $DATA_DIR --parent_directory $RAW_DATA
python dataset/NExT_QA.py --mode val --n 1000 --output_dir $DATA_DIR --parent_directory $RAW_DATA

# Run the training script
bash ./scripts/run_qwen2_5_vl_uniform.sh
