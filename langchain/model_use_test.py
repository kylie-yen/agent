from langchain_openai import ChatOpenAI
import os

# 从环境变量读取 DashScope API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

# 创建 LangChain 的 ChatOpenAI 实例，指向通义千问的兼容接口
chat = ChatOpenAI(
    model="qwen-flash",                # 或 qwen-plus, qwen-max 等
    temperature=0.0,
    openai_api_key=api_key,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 测试调用
response = chat.invoke("你好，请介绍一下你自己")
print(response.content)