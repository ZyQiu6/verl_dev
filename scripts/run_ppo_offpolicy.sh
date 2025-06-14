#!/bin/sh

#SBATCH -J run_grpo_test
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:30:00
#SBATCH --gres=gpu:4

set -x

__conda_setup="$('/shared_ssd_storage/ziyiqiu/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/shared_ssd_storage/ziyiqiu/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/shared_ssd_storage/ziyiqiu/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/shared_ssd_storage/ziyiqiu/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate test

export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export RAY_DEDUP_LOGS=0
# export NCCL_IB_DISABLE=1

python3 -m verl.trainer.main_ppo_offpolicy \
    data.train_files=/shared_ssd_storage/ziyiqiu/programs/verl_dev/data/gsm8k/train.parquet \
    data.val_files=/shared_ssd_storage/ziyiqiu/programs/verl_dev/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=1312 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
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
    actor_rollout_ref.rollout.partial_rollout_save_steps=450 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.switch_role=True \
    trainer.offpolicy_sync_freq=4 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_gsm8k_0.5B_256' \
    trainer.experiment_name='sequence-level offpolicy switch+token-level offpolicy' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    +trainer.rollout_data_dir=/shared_ssd_storage/ziyiqiu/programs/verl_dev/ \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@ >> output.txt