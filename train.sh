
export ROW_DATA=/path/to/row_data/NExT-QA
export DATA_DIR=/path/to/pocessed_data

python dataset/NExT_QA.py --mode train --n 1000 --output_dir $DATA_DIR --parent_directory $ROW_DATA
python dataset/NExT_QA.py --mode val --n 1000 --output_dir $DATA_DIR --parent_directory $ROW_DATA

bash ./scripts/run_qwen2_5_vl_template.sh
