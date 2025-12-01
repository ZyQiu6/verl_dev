from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/home/data/Qwen3-30B-A3B",
    device_map={"": "npu"},
    torch_dtype=torch.bfloat16,   # 或 float16，看你当前环境
)

input_ids = torch.randint(0, model.config.vocab_size, (1, 16), device="npu")
out = model(input_ids)
print(out.logits.shape)
