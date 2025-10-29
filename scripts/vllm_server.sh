#!/bin/bash

# 选择两张卡
export CUDA_VISIBLE_DEVICES=1,2

# 如果集群有 InfiniBand，避免初始化挂起（仅 IB 场景需要）
# export GLOO_SOCKET_IFNAME=eth0   # 官方建议，用以走以太网做进程发现

# 可选：先用通用 AgRs 后端；若已装好 pplx-kernels，可改成 pplx
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_IB_DISABLE=1 

vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
