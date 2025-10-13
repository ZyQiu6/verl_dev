#!/usr/bin/env bash
set -Eeuo pipefail

### ---- CUDA / 链接必备 ----
# 把 CUDA 指到当前 conda 环境
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:${PATH}"
export CUDACXX="$CUDA_HOME/bin/nvcc"

# 运行时动态库搜索（程序“运行时”用）
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# 链接期搜索（ninja 调用 c++/ld “链接时”用）
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LDFLAGS="-L$CUDA_HOME/lib -L$CUDA_HOME/lib64 ${LDFLAGS:-}"

# 头文件搜索
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

# 可选：限制算力，避免 JIT/扩展遍历所有 arch（Hopper）
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"

### ---- 你已有的环境变量 ----
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com
export RAY_DEDUP_LOGS=0

### ---- 保障 JIT 重新编译，Ray 继承当前环境 ----
ray stop || true
rm -rf ~/.cache/flashinfer/*/cached_ops ~/.cache/flashinfer/noarch/cached_ops || true

# 若 ninja 只带了 -L$CONDA_PREFIX/lib64，而 cudart 只在 lib/ 下，则在 lib64 下补软链
mkdir -p "$CUDA_HOME/lib64"
if [ -f "$CUDA_HOME/lib/libcudart.so.12" ] && [ ! -e "$CUDA_HOME/lib64/libcudart.so" ]; then
  ln -sf ../lib/libcudart.so.12 "$CUDA_HOME/lib64/libcudart.so"
fi
if [ -f "$CUDA_HOME/lib/libcudart.so.12" ] && [ ! -e "$CUDA_HOME/lib64/libcudart.so.12" ]; then
  ln -sf ../lib/libcudart.so.12 "$CUDA_HOME/lib64/libcudart.so.12"
fi

#actor_rollout_ref.model.path=allenai/OLMoE-1B-7B-0924-Instruct\

### ---- 启动训练 ----
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.name=sglang \
    data.train_files=/home/weijia/verl_dev/data/gsm8k/train.parquet \
    data.val_files=/home/weijia/verl_dev/data/gsm8k//test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=allenai/OLMoE-1B-7B-0924-Instruct\
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 "$@" \
    >> olmoe-output_sglang.txt
