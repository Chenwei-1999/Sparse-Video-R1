#!/bin/bash

# Set environment variables
export RAW_DATA=/path/to/raw_data/NExT-QA
export DATA_DIR=/path/to/processed_data

# Process training and validation data
python dataset/NExT_QA.py --mode train --n 1000 --output_dir $DATA_DIR --parent_directory $RAW_DATA
python dataset/NExT_QA.py --mode val --n 1000 --output_dir $DATA_DIR --parent_directory $RAW_DATA

# Run the training script
bash ./scripts/run_qwen2_5_vl_random.sh
