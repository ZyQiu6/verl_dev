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
conda activate verl_dev

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/shared_ssd_storage/ziyiqiu/programs/verl_dev/data/gsm8k/train.parquet \
    data.val_files=/shared_ssd_storage/ziyiqiu/programs/verl_dev/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=256 \
    data.max_prompt_length=256 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_gsm8k_0.5B' \
    trainer.experiment_name='qwen2.5_0.5B' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.fuse_enable=False \
    trainer.total_epochs=15 $@ >> output.txt