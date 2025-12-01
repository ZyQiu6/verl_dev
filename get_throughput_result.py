import re

def count_and_avg_timing(filepath: str):
    # 匹配 timing_per_token_ms/gen: 后面的浮点数
    pattern = re.compile(r"timing_per_token_ms/gen:([0-9\.Ee+-]+)")
    values = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            for m in pattern.finditer(line):
                v = float(m.group(1))
                values.append(v)

    count = len(values)
    avg = sum(values) / count if count > 0 else None
    return count, avg

if __name__ == "__main__":
    # 把这里换成你的 txt 文件路径
    logfile = "qwen30b-record-16npu_16dp4tp4_acl_4096.txt"

    count, avg = count_and_avg_timing(logfile)
    print(f"出现次数: {count}")
    if avg is not None:
        print(f"平均值: {avg}")
    else:
        print("未找到任何 timing_per_token_ms/gen 记录")