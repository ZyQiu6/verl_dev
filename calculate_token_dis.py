#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取 JSON，汇总所有层的64专家选择次数，计算每个专家的占比（总和=1），
并将结果以“单行64列（无表头）”写入 CSV。

用法：
  python expert_share_64.py input.json --out_csv expert_share.csv
"""

import json, argparse, csv, sys
from typing import Any, Dict, List

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_layer_dict(d: Dict[str, Any]) -> bool:
    """是否形如 {layer_id(str) -> list_of_64_numbers}"""
    if not isinstance(d, dict) or not d:
        return False
    return any(isinstance(v, list) for v in d.values())

def extract_layer_dicts(data: Any) -> List[Dict[str, List[float]]]:
    """
    抽取若干“层->64专家计数列表”的字典，返回列表。
    每个元素形如：{"0":[...64...], "1":[...64...], ...}
    """
    layer_dicts: List[Dict[str, List[float]]] = []

    def try_add(obj: Any):
        if isinstance(obj, dict):
            if is_layer_dict(obj):
                layer_dicts.append(obj)
            else:
                # 可能是 {seq_id -> layer_dict}
                for v in obj.values():
                    if isinstance(v, dict) and is_layer_dict(v):
                        layer_dicts.append(v)

    if isinstance(data, list):
        for item in data:
            try_add(item)
    elif isinstance(data, dict):
        try_add(data)
    else:
        raise ValueError(f"不支持的 JSON 顶层结构：{type(data)}")

    if not layer_dicts:
        raise ValueError("未找到任何 {layer_id -> [64个数]} 结构，请检查输入 JSON。")
    return layer_dicts

def check_64(vec: List[Any]) -> None:
    if len(vec) != 64:
        raise ValueError(f"检测到专家数为 {len(vec)}（期望64）。")

def main():
    ap = argparse.ArgumentParser(description="计算64专家占比（单行CSV，无表头）")
    ap.add_argument("json_path", help="输入 JSON 文件")
    ap.add_argument("--out_csv", default="expert_share.csv", help="输出 CSV 文件（默认 expert_share.csv）")
    args = ap.parse_args()

    data = load_json(args.json_path)
    layer_dicts = extract_layer_dicts(data)

    sum_vec: List[float] = []
    init = False
    total_layers = 0

    for ld in layer_dicts:
        # 层ID排序仅为稳定，不影响数值
        for layer_id in sorted(ld.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            vec = ld[layer_id]
            check_64(vec)
            if not init:
                sum_vec = [0.0] * 64
                init = True
            for i in range(64):
                sum_vec[i] += float(vec[i])
            total_layers += 1

    if not init or total_layers == 0:
        print("未汇总到任何层的数据。", file=sys.stderr)
        sys.exit(1)

    grand_total = sum(sum_vec)
    if grand_total == 0:
        print("总计数为0，无法计算占比。", file=sys.stderr)
        sys.exit(1)

    share = [v / grand_total for v in sum_vec]  # 64列，占比之和=1

    # 写CSV：一行64列，无表头
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"{x:.8f}" for x in share])

    # 终端提示（可选）
    print(f"已写出：{args.out_csv}")
    print(f"校验占比之和：{sum(share):.6f}（应接近1）")
    print(f"汇总层数：{total_layers}")

if __name__ == "__main__":
    main()
