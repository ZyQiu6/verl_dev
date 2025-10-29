# client_smoke.py
from openai import OpenAI

cli = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
resp = cli.chat.completions.create(
    model="allenai/OLMoE-1B-7B-0924-Instruct",
    messages=[{"role": "user", "content": "Ping! 只要快速回一句话即可。"}],
    temperature=0.0,
    max_tokens=8,
)
print("OK:", resp.choices[0].message.content.strip())
