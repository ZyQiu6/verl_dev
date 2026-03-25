#!/bin/sh
#
# Run a SMALL real GRPO training with HSpec enabled.
# Goal: validate that all *implemented* HSpec features in HSpec Tips.md work
# in an end-to-end training loop with vLLM-Ascend on NPU.
#
# What this script checks (by logs/metrics):
# - vLLM speculative_config resolves to method=hspec
# - online query path runs (HSpecProposer logs "HSpec online metrics")
# - hidden_states collection + transmission works (no "None hs" skips explosion)
# - table build runs + epoch-boundary swap happens
# - perf breakdown includes "hspec_build_wait" (detects training-side blocking)
#
# Notes:
# - This is NOT a benchmark; we keep dataset small for quick validation.
# - Requires: gsm8k parquet files exist under /workspace/verl_dev/data/gsm8k/
#
set -e

# debug
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export HSPEC_DEBUG=0
export HSPEC_TRACE=0

# super params
export PCA_COMPONENTS=64

# optim attempt
export HSPEC_ENTRY=1
export MATCH_WND=16

# Disable ad-hoc text timing by default; use torch_npu.profiler.profile instead.
export HSPEC_GEN="${HSPEC_GEN:-0}"
export HSPEC_GEN_REQ_IDX="${HSPEC_GEN_REQ_IDX:-0}"
export HSPEC_GEN_MAX_CALLS="${HSPEC_GEN_MAX_CALLS:-0}"

# torch_npu.profiler.profile for HSpec generate path.
# Only profile selected training steps and annotate sub-stages with mstx/profile ranges.
export HSPEC_PROFILE="${HSPEC_PROFILE:-1}"
export HSPEC_PROFILE_STEPS="${HSPEC_PROFILE_STEPS:-5,31}"
# export HSPEC_PROFILE_REQ_IDX="${HSPEC_PROFILE_REQ_IDX:-3}"
export HSPEC_PROFILE_DIR="${HSPEC_PROFILE_DIR:-/home/xy/hspec_profile-1}"
export HSPEC_PROFILE_METHOD="${HSPEC_PROFILE_METHOD:-mstx}"
export HSPEC_PROFILE_LEVEL="${HSPEC_PROFILE_LEVEL:-level_none}"
export HSPEC_PROFILE_ANALYSE="${HSPEC_PROFILE_ANALYSE:-1}"
export HSPEC_PROFILE_WITH_STACK="${HSPEC_PROFILE_WITH_STACK:-0}"
export HSPEC_PROFILE_MEMORY="${HSPEC_PROFILE_MEMORY:-0}"
export USE_HISTORY_SPEC_DECODE="${USE_HISTORY_SPEC_DECODE:-False}"
export USE_HSPEC_DECODE="${USE_HSPEC_DECODE:-True}"
if [ "${HSPEC_PROFILE}" = "1" ]; then
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    export ROLLOUT_N="${ROLLOUT_N:-2}"
else
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    export ROLLOUT_N="${ROLLOUT_N:-2}"
fi

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Make HSpec proposer print periodic online metrics.
export HSPEC_LOG_EVERY_CALLS="${HSPEC_LOG_EVERY_CALLS:-50}"
export HSPEC_LOG_EVERY_S="${HSPEC_LOG_EVERY_S:-5}"
export HSPEC_LOG_LEVEL="${HSPEC_LOG_LEVEL:-INFO}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

export MODEL_PATH="${MODEL_PATH:-/home/data/Qwen2.5-1.5B-Instruct}"

# Output log file
export OUT="${OUT:-/workspace/verl_dev/scripts/run_command/hspec/output-grpo-hspec-train.txt}"

python -m verl.trainer.main_ppo \
    ray_kwargs.ray_init.num_cpus=128 \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG='"'"${HSPEC_DEBUG}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG_MAX_REQS='"2"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG_MAX_SAMPLES='"4"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_DEBUG_MAX_VALUES='"8"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL='"INFO"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_TRACE='"'"${HSPEC_TRACE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_GEN='"'"${HSPEC_GEN}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_GEN_REQ_IDX='"'"${HSPEC_GEN_REQ_IDX}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_GEN_MAX_CALLS='"'"${HSPEC_GEN_MAX_CALLS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE='"'"${HSPEC_PROFILE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_STEPS='"'"${HSPEC_PROFILE_STEPS}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_METHOD='"'"${HSPEC_PROFILE_METHOD}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_DIR='"'"${HSPEC_PROFILE_DIR}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_LEVEL='"'"${HSPEC_PROFILE_LEVEL}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_ANALYSE='"'"${HSPEC_PROFILE_ANALYSE}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_WITH_STACK='"'"${HSPEC_PROFILE_WITH_STACK}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_PROFILE_MEMORY='"'"${HSPEC_PROFILE_MEMORY}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.HSPEC_LOG_LEVEL='"'"${HSPEC_LOG_LEVEL}"'"' \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL='"'"${VERL_LOGGING_LEVEL}"'"' \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/verl_dev/data/gsm8k/train.parquet \
    data.val_files=/workspace/verl_dev/data/gsm8k/test.parquet \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.use_history_spec_decode="${USE_HISTORY_SPEC_DECODE}" \
    +actor_rollout_ref.rollout.use_hspec_decode="${USE_HSPEC_DECODE}" \
    +actor_rollout_ref.rollout.hspec_num_speculative_tokens=5 \
    +actor_rollout_ref.rollout.hspec_similarity_threshold=0.85 \
    +actor_rollout_ref.rollout.hspec_min_match_len=1 \
    +actor_rollout_ref.rollout.hspec_n_components="${PCA_COMPONENTS}" \
    +actor_rollout_ref.rollout.hspec_max_entries_per_prompt=10000 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.device=npu \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_gsm8k_hspec_validate' \
    trainer.experiment_name='qwen_hspec_validate_small' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=2 \
    trainer.total_epochs=2 \
    > "${OUT}" 2>&1 $@

echo "OK: training finished. See log at ${OUT}"
