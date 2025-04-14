#!/bin/bash

# Set environment variables
export RAW_DATA=/mnt/c/Users/Chenwei/Desktop/VideoAgent
export DATA_DIR=/mnt/c/Users/Chenwei/Desktop/VideoAgent/dataset

# Process training and validation data
python dataset/NExT_QA.py --mode train --n 1000 --output_dir $DATA_DIR --parent_directory $RAW_DATA
python dataset/NExT_QA.py --mode val --n 1000 --output_dir $DATA_DIR --parent_directory $RAW_DATA

# Run the training script
bash ./scripts/run_qwen2_5_vl_template.sh
