#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在同一台机器上比较：
  - MoE 层 用 TP（enable_expert_parallel=False）
  - MoE 层 用 EP（enable_expert_parallel=True）

在不同 batch_size（≈ 每步 decode 的 token 数）下的吞吐差异，
帮你找到一个“大致的切换阈值”。

用法示例（8 卡）：
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
  python vllm_bench.py \
      --model /home/data/Qwen3-30B-A3B \
      --tp-size 16 \
      --batch-sizes 1,2,4,8,16,32,64,128,256 \
      --max-new-tokens 1024 \
      --output-csv bench_tp_vs_ep.csv

Ascend / vllm-ascend 类似，只要保证 vllm 的环境变量已经配好。
"""

import argparse
import time
import statistics
import os
import csv

from vllm import LLM, SamplingParams

try:
    import torch
except ImportError:
    torch = None


def build_llm(
    model_path: str,
    tp_size: int,
    enable_ep: bool,
    max_model_len: int,
    max_num_seqs: int,
    dtype: str = "bfloat16",
    gpu_mem_util: float = 0.9,
    enforce_eager: bool = False,
):
    """构建一个 vLLM LLM 实例。"""
    mode_name = "EP" if enable_ep else "TP"
    print(f"\n=== 初始化 LLM: mode={mode_name} ===")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_ep,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
        # 如果你是 vllm-ascend，可以在这里加自己训练时的配置：
        # distributed_executor_backend="external_launcher",
        # load_format="megatron",
    )
    return llm


def run_one_batch(
    llm: LLM,
    batch_size: int,
    max_new_tokens: int,
    repeat: int = 3,
):
    """在给定 batch_size 下跑 repeat 次生成，返回一次平均测量结果。"""
    base_prompt = "这是一个用于测试 MoE 并行策略（TP vs EP）的基准提示。"
    prompts = [base_prompt for _ in range(batch_size)]

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    latencies = []
    num_new_tokens = []

    for _ in range(repeat):
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        t1 = time.time()

        total_new = 0
        for out in outputs:
            total_new += len(out.outputs[0].token_ids)
        latencies.append(t1 - t0)
        num_new_tokens.append(total_new)

    avg_latency = statistics.mean(latencies)
    avg_tokens = statistics.mean(num_new_tokens)
    tps = avg_tokens / avg_latency

    tokens_per_step = batch_size  # decode 场景下，可近似为每步 token 数

    return {
        "batch_size": batch_size,
        "tokens_per_step": tokens_per_step,
        "avg_new_tokens": avg_tokens,
        "avg_latency": avg_latency,
        "tps": tps,
    }


def benchmark_mode(
    model_path: str,
    tp_size: int,
    batch_sizes,
    max_new_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
    dtype: str,
    gpu_mem_util: float,
    enforce_eager: bool,
    enable_ep: bool,
):
    """对某个模式（TP 或 EP）在一组 batch_sizes 上跑基准。

    返回：
      - results_tps: {batch_size: tokens_per_sec}
      - rows: List[dict]，每行带有 mode/batch_size/...，用于写 CSV
    """
    mode_name = "EP" if enable_ep else "TP"
    llm = build_llm(
        model_path=model_path,
        tp_size=tp_size,
        enable_ep=enable_ep,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        gpu_mem_util=gpu_mem_util,
        enforce_eager=enforce_eager,
    )

    print(f"[{mode_name}] warmup ...")
    _ = llm.generate(
        ["warmup"] * min(2, batch_sizes[0]),
        SamplingParams(max_tokens=8, temperature=0.0),
    )

    results_tps = {}
    rows = []

    print(f"\n[{mode_name}] 开始基准：")
    print("mode,batch_size,tokens_per_step,avg_new_tokens,avg_latency_sec,tokens_per_sec")
    for bs in batch_sizes:
        r = run_one_batch(
            llm=llm,
            batch_size=bs,
            max_new_tokens=max_new_tokens,
            repeat=3,
        )
        results_tps[bs] = r["tps"]

        print(
            f"{mode_name},{r['batch_size']},"
            f"{r['tokens_per_step']},"
            f"{r['avg_new_tokens']:.1f},"
            f"{r['avg_latency']:.4f},"
            f"{r['tps']:.2f}"
        )

        row = {
            "mode": mode_name,
            "batch_size": r["batch_size"],
            "tokens_per_step": r["tokens_per_step"],
            "avg_new_tokens": r["avg_new_tokens"],
            "avg_latency_sec": r["avg_latency"],
            "tokens_per_sec": r["tps"],
        }
        rows.append(row)

    # 尽量释放显存
    del llm
    if torch is not None:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()

    return results_tps, rows


def find_threshold(batch_sizes, tp_results, ep_results):
    """根据 tp/ep 在各 batch_size 下的 tps，给出一个粗略“切换阈值”."""
    threshold = None
    for bs in sorted(batch_sizes):
        tp_tps = tp_results.get(bs)
        ep_tps = ep_results.get(bs)
        if tp_tps is None or ep_tps is None:
            continue
        if ep_tps >= tp_tps:
            threshold = bs
            break
    return threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="MoE 模型路径，比如 /home/data/Qwen3-30B-A3B",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        required=True,
        help="tensor_parallel_size，TP 和 EP 共用这一值。",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64",
        help="逗号分隔的一组 batch_size，用来近似“每步 decode 的 token 数”。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="每条序列生成的新 token 上限，取大一点让 decode 占主导。",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="vLLM 的 max_model_len 配置。",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=512,
        help="vLLM 的 max_num_seqs 配置，需 ≥ 最大 batch_size。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="推理精度，比如 bfloat16 / float16 / auto。",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.9,
        help="gpu_memory_utilization。",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="是否打开 enforce_eager 方便 debug。",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="bench_tp_vs_ep.csv",
        help="测试结果输出的 CSV 文件路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    batch_sizes = sorted(batch_sizes)

    print("当前配置：")
    print(f"  model            = {args.model}")
    print(f"  tp_size          = {args.tp_size}")
    print(f"  batch_sizes      = {batch_sizes}")
    print(f"  max_new_tokens   = {args.max_new_tokens}")
    print(f"  max_model_len    = {args.max_model_len}")
    print(f"  max_num_seqs     = {args.max_num_seqs}")
    print(f"  dtype            = {args.dtype}")
    print(f"  gpu_mem_util     = {args.gpu_mem_util}")
    print(f"  enforce_eager    = {args.enforce_eager}")
    print(f"  output_csv       = {args.output_csv}")

    # 先跑 TP
    tp_results, tp_rows = benchmark_mode(
        model_path=args.model,
        tp_size=args.tp_size,
        batch_sizes=batch_sizes,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        gpu_mem_util=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        enable_ep=False,
    )

    # 再跑 EP
    ep_results, ep_rows = benchmark_mode(
        model_path=args.model,
        tp_size=args.tp_size,
        batch_sizes=batch_sizes,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        gpu_mem_util=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        enable_ep=True,
    )

    # 计算一个粗略阈值
    threshold = find_threshold(batch_sizes, tp_results, ep_results)

    print("\n===== 汇总（只看 tokens/s）=====")
    print("batch_size,tps_TP,tps_EP")
    for bs in batch_sizes:
        print(f"{bs},{tp_results.get(bs, float('nan')):.2f},"
              f"{ep_results.get(bs, float('nan')):.2f}")

    if threshold is None:
        print("\n在这组 batch_size 范围内，EP 的吞吐始终没有超过 TP，"
              "说明在当前硬件 + 模型下，大概率全程用 TP 更划算。")
    else:
        print(f"\n粗略阈值：batch_size ≈ {threshold} 时，EP 的 tokens/s 开始 ≥ TP。")
        print("你可以在 VERL 的 rollout 里，用“当前活跃 seq 数 >= "
              f"{threshold}”作为 MoE 切换到 EP 的条件。")

    # ===== 写入 CSV =====
    all_rows = tp_rows + ep_rows
    fieldnames = [
        "mode",
        "batch_size",
        "tokens_per_step",
        "avg_new_tokens",
        "avg_latency_sec",
        "tokens_per_sec",
    ]
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True) if os.path.dirname(args.output_csv) else None
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n所有原始测试结果已保存到 CSV：{args.output_csv}")


if __name__ == "__main__":
    # Ascend 场景可以在这里设置必要的环境变量
    # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
