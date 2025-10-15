#!/bin/sh

export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export HF_ENDPOINT=https://hf-mirror.com

# 如需严格离线（你已全量拉过），可解开下面两行
# export TRANSFORMERS_OFFLINE=1
# export HF_HUB_OFFLINE=1

python3 -m verl.trainer.main_ppo \
    data.train_files=../data/gsm8k/train.parquet \
    data.val_files=../data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-V2-Lite-Chat \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.trust_remote_code=true \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.trust_remote_code=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_gsm8k_dsv2_128' \
    trainer.experiment_name='original' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.fuse_enable=False \
    trainer.fuse_value=True \
    trainer.fuse_old_log_prob=True \
    trainer.total_epochs=5 \
    "$@" >> firstoutput.txt
