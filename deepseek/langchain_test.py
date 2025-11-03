from langchain.chat_models import ChatOpenAI

# 这里我们将参数temperature设置为0.0，从而减少生成答案的随机性。
# 如果你想要每次得到不一样的有新意的答案，可以尝试调整该参数。
chat = ChatOpenAI(temperature=0.0)
chat
