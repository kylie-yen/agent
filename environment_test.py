import ollama
response = ollama.generate(
    model="qwen2.5:3b",  # 模型名称
    prompt="你是谁。"  # 提示文本
)
print(response)
