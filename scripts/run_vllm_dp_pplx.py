# /home/weijia/verl/scripts/run_vllm_dp_pplx.py
import os
from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        "allenai/OLMoE-1B-7B-0924-Instruct",
        tensor_parallel_size=1,
        data_parallel_size=2,           # 你需要测试的 DP=2
        enable_expert_parallel=True,    # vLLM 0.11 会自动设置 EP=DP*TP
        trust_remote_code=True,
        enforce_eager=True,
    )
    out = llm.generate(["hi"], SamplingParams(max_tokens=8))
    print("the generated text is:", out[0].outputs[0].text)

if __name__ == "__main__":
    main()
