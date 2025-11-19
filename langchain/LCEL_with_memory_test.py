import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- 1. 配置 API Key ---
api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 2. 模拟一个数据库来存储历史记录 ---
# 在生产环境中，这里通常会连接 Redis 或 Postgres
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    根据 session_id 获取对应的历史记录。
    如果该 session_id 不存在，则创建一个新的历史记录对象。
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- 3. 定义核心组件 ---

# 初始化 Qwen 模型
model = ChatTongyi(model="qwen-plus", temperature=0.7)

# 定义 Prompt
# 关键点：必须包含一个与 history_messages_key 同名的占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的 AI 助手，你的名字叫通义。"),
    MessagesPlaceholder(variable_name="history"), # <--- 历史记录会注入到这里
    ("human", "{input}"), # <--- 用户的新输入
])

# 定义基础链 (无记忆版本)
runnable = prompt | model | StrOutputParser()

# --- 4. 添加记忆功能 (核心步骤) ---
# 使用 RunnableWithMessageHistory 包装基础链
with_message_history = RunnableWithMessageHistory(
    runnable,  # 原来的链
    get_session_history, # 获取历史记录的回调函数
    input_messages_key="input", # 指明哪个变量是用户输入
    history_messages_key="history", # 指明 Prompt 中哪里放历史记录
)

# --- 5. 运行测试 ---

print("=== 开始对话 (Session ID: user_123) ===")

# 第 1 轮对话
# config 中必须包含 session_id，用来区分不同用户
response1 = with_message_history.invoke(
    {"input": "你好，我是小明，我喜欢吃苹果。"},
    config={"configurable": {"session_id": "user_123"}}
)
print(f"AI: {response1}")

# 第 2 轮对话 (测试记忆)
print("\n=== 第二轮对话 ===")
response2 = with_message_history.invoke(
    {"input": "我刚才说我喜欢吃什么？"},
    config={"configurable": {"session_id": "user_123"}}
)
print(f"AI: {response2}")

# --- 6. 模拟另一个用户 (Session ID: user_999) ---
# 验证记忆是隔离的
print("\n=== 切换用户 (Session ID: user_999) ===")
response3 = with_message_history.invoke(
    {"input": "我刚才说我喜欢吃什么？"}, # 新用户没有之前的记忆
    config={"configurable": {"session_id": "user_999"}}
)
print(f"AI: {response3}")
