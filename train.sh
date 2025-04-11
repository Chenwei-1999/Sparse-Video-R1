
python dataset/NExT_QA.py --mode train --n 1000
python dataset/NExT_QA.py --mode val --n 1000 

# run random
bash ./scripts/run_qwen2_5_vl-3b_random.sh 

# run uniform
bash ./scripts/run_qwen2_5_vl-3b_uniform.sh 

# run all
bash ./scripts/run_qwen2_5_vl-3b_mix.sh 
# run importance
# bash ./scripts/run_qwen2_5_vl-3b_importance.sh

bash ./scripts/run_qwen2_5_vl-3b_random_resolution.sh

