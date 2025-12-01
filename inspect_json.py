import json

def check_top_list_all_equal(json_path):
    # 读入 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 确认顶层是 list
    if not isinstance(data, list):
        raise TypeError(f"顶层不是 list，而是 {type(data).__name__}，无法直接对比第一层 list 元素。")

    n = len(data)
    print(f"顶层 list 长度：{n}")

    # 长度 0 或 1 的情况，认为“都相同”是 trivially True
    if n <= 1:
        print("元素数量 ≤ 1，视为所有元素相同。")
        return True

    first = data[0]
    all_equal = True
    diff_indices = []

    for i, elem in enumerate(data[1:], start=1):
        if elem != first:      # 深度比较，嵌套结构也会逐项比较
            all_equal = False
            diff_indices.append(i)

    if all_equal:
        print("结果：第一层 list 中所有元素完全相同。")
    else:
        print("结果：第一层 list 中存在不相同的元素。")
        print("与第 0 个元素不同的索引有：", diff_indices[:20], 
              "(仅显示前 20 个，如有更多请自己调整打印逻辑)")

    return all_equal


if __name__ == "__main__":
    # 改成你的 json 文件路径
    check_top_list_all_equal("moe_step_0.json")
