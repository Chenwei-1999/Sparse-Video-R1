#!/bin/bash
set -x

# Environment variables
export HF_HOME=/path/to/huggingface_cache
export N_GPUS=4
export BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
export DATA_DIR=/path/to/processed_data
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=qwen2.5-3b
export SAMPLING_STRATEGY=all  # Options: "all", "random", "uniform"
# export VLLM_ATTENTION_BACKEND=XFORMERS

# Run training with PPO
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train/nextqa.json \
    data.val_files=$DATA_DIR/val/nextqa.json \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.val_batch_size=16 \
    data.max_response_length=256 \
    data.sampling_strategy=$SAMPLING_STRATEGY \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Sparse_Video_R1' \
    trainer.experiment_name=$SAMPLING_STRATEGY \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 | tee verl_demo.log
