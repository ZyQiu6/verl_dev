#!/bin/sh

#SBATCH -J run_grpo_test
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:4

export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
# export RAY_TMPDIR=/data/qiuzy/tmp
export CUDA_VISIBLE_DEVICES='2,3'

python3 -m verl.trainer.main_ppo \
    data.train_files=/data/qiuzy/programs/verl_dev/data/gsm8k/train.parquet \
    data.val_files=/data/qiuzy/programs/verl_dev/data/gsm8k/test.parquet \
    data.train_batch_size=2048 \
    data.val_batch_size=1312 \
    data.max_prompt_length=256 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/data/qiuzy/Qwen2.5-1.5B-Instruct  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.use_history_spec_decode=True \
    actor_rollout_ref.rollout.temperature=0.3 \
    actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/data/qiuzy/Qwen2.5-1.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_gsm8k_0.5B_256' \
    trainer.experiment_name='original' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    +trainer.rollout_data_dir=/data/qiuzy/programs/verl_dev/dump \
    +trainer.rollout_length_dir=/data/qiuzy/programs/verl_dev/dump \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.fuse_enable=False \
    trainer.total_epochs=15 $@ >> output.txt
