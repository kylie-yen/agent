from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
import os

# --- 配置 API KEY ---
# api_key = os.getenv("DASHSCOPE_API_KEY")

# 1. 定义组件
model = ChatTongyi(model="qwen-plus")

prompt = ChatPromptTemplate.from_template("请用一句话介绍一下：{topic}")
parser = StrOutputParser() # 自动把 Message 转成纯文本字符串

# 2. 使用 LCEL 组装链条 (流水线)
chain = prompt | model | parser

# 3. 调用
# invoke 接受一个字典，对应 prompt 中的 {topic}
result = chain.invoke({"topic": "量子力学"})

print(result)
# 输出: 量子力学是研究微观粒子运动规律的物理学分支...
