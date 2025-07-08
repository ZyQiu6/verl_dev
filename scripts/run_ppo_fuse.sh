#!/bin/bash

#SBATCH -J run_ppo_fuse
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --gpus=4

set -x

module load miniconda/24.9.2
source activate verl_dev

module load cuda/12.4
module load nccl/2.23.4-1-cuda12.4
module load cudnn/8.8.1.3

export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export RAY_DEDUP_LOGS=0
# export NCCL_IB_DISABLE=1

python3 -m verl.trainer.main_ppo \
    data.train_files=/data/home/scyb166/zyqiu/data/dataset/sky_t1_math/train.parquet \
    data.val_files=/data/home/scyb166/zyqiu/data/dataset/sky_t1_math/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1204 \
    data.max_prompt_length=1536 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/data/home/scyb166/zyqiu/data/model/qwen2.5_3B/models--Qwen--Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    actor_rollout_ref.rollout.partial_rollout_save_steps=2098 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/data/home/scyb166/zyqiu/data/model/qwen2.5_3B/models--Qwen--Qwen2.5-3B \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name='verl_skyt1math_3B' \
    trainer.experiment_name='token-level' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    +trainer.rollout_data_dir=/data/home/scyb166/zyqiu/verl_dev/dump/3b \
    +trainer.rollout_length_dir=/data/home/scyb166/zyqiu/verl_dev/dump/3b \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.fuse_enable=False \
    trainer.fuse_value=True \
    trainer.fuse_old_log_prob=True \
    trainer.total_epochs=15 $@ >> output.txt