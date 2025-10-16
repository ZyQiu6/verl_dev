#!/bin/sh
export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_IB_DISABLE=1
# allenai/OLMoE-1B-7B-0924-Instruct
# actor_rollout_ref.rollout.load_format='dummy_hf'\
#set ep
# +actor_rollout_ref.rollout.enable_expert_parallel=True\
# actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

python3 -m verl.trainer.main_ppo \
    data.train_files=/home/weijia/verl_dev/data/gsm8k/train.parquet \
    data.val_files=/home/weijia/verl_dev/data/gsm8k//test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path=allenai/OLMoE-1B-7B-0924-Instruct\
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.expert_parallel_size=4 \
    actor_rollout_ref.rollout.data_parallel_size=4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_gsm8k_olmoechat_dpep' \
    trainer.experiment_name='original' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@ >> olmoe-output_dpep.txt
