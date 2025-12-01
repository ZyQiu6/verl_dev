"""
在同一台机器上比较多种 (tp, dp) 组合和两种 MoE 策略的 decode 吞吐：

  1) 整体 dp + tp（MoE 也用 TP）:
       data_parallel_size = dp
       tensor_parallel_size = tp
       enable_expert_parallel = False

  2) attn 层 dp + tp, MoE 层 ep:
       data_parallel_size = dp
       tensor_parallel_size = tp
       enable_expert_parallel = True

假设总卡数 world_size 固定（例如 16），约束 tp * dp = world_size。
默认遍历 tp=2,4,8，则 dp=8,4,2。

对一组 batch_size（≈ 每步 decode 的 token 数）在不同 (tp, dp) 下
测 tokens/s，结果写入 CSV 方便画图分析。

用法示例（16 卡）：
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \\
  python bench_tp_dp_moe_ep.py \\
      --model /home/data/Qwen3-30B-A3B \\
      --world-size 16 \\
      --tp-list 2,4,8 \\
      --batch-sizes 1,2,4,8,16,32,64 \\
      --max-new-tokens 128 \\
      --output-csv bench_tp_dp_moe_ep.csv
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
    dp_size: int,
    enable_ep: bool,
    max_model_len: int,
    max_num_seqs: int,
    dtype: str = "bfloat16",
    gpu_mem_util: float = 0.9,
    enforce_eager: bool = False,
):
    """构建一个 vLLM LLM 实例（兼容 vllm 0.11.0，无 expert_parallel_size 参数）。"""
    moe_mode = "moe_ep" if enable_ep else "moe_tp"
    print(f"\n=== 初始化 LLM: tp={tp_size}, dp={dp_size}, moe_mode={moe_mode} ===")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,          # vllm 0.11.0 支持 data_parallel_size
        enable_expert_parallel=enable_ep,    # 控制 MoE 层是否使用 EP
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
        # 如果你使用 vllm-ascend / external launcher，可以在此追加：
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
    base_prompt = "这是一个用于测试 MoE 并行策略（DP/TP/EP）的基准提示。"
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

    # decode 场景下，每步大致 1 token / seq，可近似为：
    tokens_per_step = batch_size

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
    dp_size: int,
    batch_sizes,
    max_new_tokens: int,
    max_model_len: int,
    max_num_seqs: int,
    dtype: str,
    gpu_mem_util: float,
    enforce_eager: bool,
    enable_ep: bool,
):
    """对某个 (tp_size, dp_size, moe_mode) 在一组 batch_sizes 上跑基准。

    返回：
      - results_tps: {batch_size: tokens_per_sec}
      - rows: List[dict]，每行带有 tp_size/dp_size/moe_mode/batch_size/...，用于写 CSV
    """
    moe_mode = "moe_ep" if enable_ep else "moe_tp"
    llm = build_llm(
        model_path=model_path,
        tp_size=tp_size,
        dp_size=dp_size,
        enable_ep=enable_ep,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        gpu_mem_util=gpu_mem_util,
        enforce_eager=enforce_eager,
    )

    print(f"[tp={tp_size}, dp={dp_size}, {moe_mode}] warmup ...")
    _ = llm.generate(
        ["warmup"] * min(2, batch_sizes[0]),
        SamplingParams(max_tokens=8, temperature=0.0),
    )

    results_tps = {}
    rows = []

    print(f"\n[tp={tp_size}, dp={dp_size}, {moe_mode}] 开始基准：")
    print("tp_size,dp_size,moe_mode,batch_size,tokens_per_step,avg_new_tokens,avg_latency_sec,tokens_per_sec")

    for bs in batch_sizes:
        r = run_one_batch(
            llm=llm,
            batch_size=bs,
            max_new_tokens=max_new_tokens,
            repeat=3,
        )
        results_tps[bs] = r["tps"]

        print(
            f"{tp_size},{dp_size},{moe_mode},{r['batch_size']},"
            f"{r['tokens_per_step']},"
            f"{r['avg_new_tokens']:.1f},"
            f"{r['avg_latency']:.4f},"
            f"{r['tps']:.2f}"
        )

        row = {
            "tp_size": tp_size,
            "dp_size": dp_size,
            "moe_mode": moe_mode,
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
    """给定同一 (tp, dp) 下 moe_tp / moe_ep 的 tps，求一个“EP 开始不亏”的粗略阈值。"""
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
        "--world-size",
        type=int,
        default=16,
        help="总卡数 world_size，要求 tp * dp = world_size。",
    )
    parser.add_argument(
        "--tp-list",
        type=str,
        default="2,4,8",
        help="逗号分隔的一组 tp 值，例如 2,4,8；对应 dp = world_size // tp。",
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
        default=4096,
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
        default="bench_tp_dp_moe_ep.csv",
        help="测试结果输出的 CSV 文件路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    batch_sizes = sorted(batch_sizes)
    tp_list = [int(x) for x in args.tp_list.split(",") if x.strip()]
    tp_list = sorted(tp_list)

    print("当前配置：")
    print(f"  model          = {args.model}")
    print(f"  world_size     = {args.world_size}")
    print(f"  tp_list        = {tp_list}")
    print(f"  batch_sizes    = {batch_sizes}")
    print(f"  max_new_tokens = {args.max_new_tokens}")
    print(f"  max_model_len  = {args.max_model_len}")
    print(f"  max_num_seqs   = {args.max_num_seqs}")
    print(f"  dtype          = {args.dtype}")
    print(f"  gpu_mem_util   = {args.gpu_mem_util}")
    print(f"  enforce_eager  = {args.enforce_eager}")
    print(f"  output_csv     = {args.output_csv}")

    all_rows = []

    for tp in tp_list:
        if args.world_size % tp != 0:
            raise ValueError(
                f"world_size={args.world_size} 不能被 tp={tp} 整除，"
                "无法保证 tp * dp = world_size。"
            )
        dp = args.world_size // tp

        print(f"\n========== 测试组合: tp={tp}, dp={dp} (tp*dp={tp*dp}) ==========")

        # 整体 dp+tp（MoE 也用 TP）
        tp_results, tp_rows = benchmark_mode(
            model_path=args.model,
            tp_size=tp,
            dp_size=dp,
            batch_sizes=batch_sizes,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            dtype=args.dtype,
            gpu_mem_util=args.gpu_mem_util,
            enforce_eager=args.enforce_eager,
            enable_ep=False,
        )

        # attn dp+tp, MoE ep
        ep_results, ep_rows = benchmark_mode(
            model_path=args.model,
            tp_size=tp,
            dp_size=dp,
            batch_sizes=batch_sizes,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            dtype=args.dtype,
            gpu_mem_util=args.gpu_mem_util,
            enforce_eager=args.enforce_eager,
            enable_ep=True,
        )

        all_rows.extend(tp_rows)
        all_rows.extend(ep_rows)

        # 对这个 (tp, dp) 求一个“EP 开始不亏”的阈值
        threshold = find_threshold(batch_sizes, tp_results, ep_results)

        print(f"\n[tp={tp}, dp={dp}] 汇总（只看 tokens/s）：")
        print("batch_size,tps_moe_tp,tps_moe_ep")
        for bs in batch_sizes:
            print(
                f"{bs},"
                f"{tp_results.get(bs, float('nan')):.2f},"
                f"{ep_results.get(bs, float('nan')):.2f}"
            )

        if threshold is None:
            print(f"\n[tp={tp}, dp={dp}] 在这组 batch_size 范围内，moe_ep 的吞吐始终没有超过 moe_tp，"
                  "说明在当前硬件 + 模型下，这个 (tp,dp) 组合大概率全程用 TP 更划算。")
        else:
            print(f"\n[tp={tp}, dp={dp}] 粗略阈值：batch_size ≈ {threshold} 时，"
                  "attn dp+tp, moe ep 的 tokens/s 开始 ≥ 整体 dp+tp。")
            print("你可以在 VERL 的 rollout 里，对这个 (tp,dp) 用："
                  f"当前活跃 seq 数 >= {threshold} → moe_ep，否则 moe_tp。")

    # ===== 写入 CSV =====
    fieldnames = [
        "tp_size",
        "dp_size",
        "moe_mode",
        "batch_size",
        "tokens_per_step",
        "avg_new_tokens",
        "avg_latency_sec",
        "tokens_per_sec",
    ]
    if os.path.dirname(args.output_csv):
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n所有原始测试结果已保存到 CSV：{args.output_csv}")


if __name__ == "__main__":
    # Ascend 场景可以在这里设置必要的环境变量，比如：
    # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()