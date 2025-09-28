#!/bin/sh

# 1) 设镜像 & 默认缓存（不改你的脚本里的模型目录）
export HF_ENDPOINT="https://hf-mirror.com"           # 换成你的镜像域名
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # 可选
# 离线标志先关闭，允许下载
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

# 2) 只清这个仓库的坏快照（避免 0 字节残留）
rm -rf "$TRANSFORMERS_CACHE/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/"* 2>/dev/null || true

# 3) 用单进程预取到“默认缓存”，并用 flock 防并发首下
python - <<'PY'
from huggingface_hub import snapshot_download
repos = [
    "deepseek-ai/DeepSeek-V2-Lite-Chat",   # actor
    "Qwen/Qwen2.5-0.5B-Instruct",          # critic（你的脚本里用到）
]
allow = [
    "config.json","generation_config.json","special_tokens_map.json",
    "tokenizer*.json","tokenizer.model","model.safetensors.index.json","model-*.safetensors",
]
for r in repos:
    snapshot_download(r, allow_patterns=allow, resume_download=True)
print("prefetch done")
PY

# 4) 预取完成后再切回离线模式，训练阶段只读缓存
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 用新版环境变量，避免 deprecate 警告
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"      # 推荐
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"        # 可选
# 训练阶段离线读取（不再访问镜像）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# （可选）禁用 tokenizer 并行提示
export TOKENIZERS_PARALLELISM=false


export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
# export NCCL_IB_DISABLE=1
# allenai/OLMoE-1B-7B-0924-Instruct
#路径相对地址，进入scripts目录执行程序

python3 -m verl.trainer.main_ppo \
    data.train_files=../data/gsm8k/train.parquet \
    data.val_files=../data/gsm8k//test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-V2-Lite-Chat\
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_gsm8k_olmoe_128' \
    trainer.experiment_name='original' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.fuse_enable=False \
    trainer.fuse_value=True \
    trainer.fuse_old_log_prob=True \
    trainer.total_epochs=1 $@ >> firstoutput.txt
