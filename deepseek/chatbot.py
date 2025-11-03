import os
import panel as pn
from openai import OpenAI

# 初始化API客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 初始化对话历史
history = [
    {"role": "system", "content": """
你是一个订餐机器人，为奶茶店自动收集订单信息。
你要首先和顾客问好，然后等待用户回复收集订单信息。收集完信息需缺人顾客是否要有其他需求。
最后询问顾客是否自取活外送，如果是外送需要收集地址信息。
最后告诉顾客订单总金额，并送上祝福。
 
你的回答应以简短、随意、友好的风格呈现。
 
菜单：
珍珠奶茶（超大，大，中）：20元，15元，10元
红茶（超大，大，中）：15元，10元，5元
绿茶（超大，大，中）：15元，10元，5元
布丁奶茶（超大，大，中）：25元，20元，15元
芒果奶昔（超大，大，中）：30元，25元，20元
 
冰度：正常冰，少冰，去冰，常温
甜度：全糖，七分糖，五分糖，三分糖，无糖
规格：超大，大，中
"""}
]

# 渲染对话历史
def render_history():
    text = ""
    for msg in history:
        if msg["role"] == "system":
            continue  # 跳过系统消息
        role = "你" if msg["role"] == "user" else "机器人"
        text += f"**{role}**: {msg['content']}\n\n"
    return text

# 创建Markdown显示区域
chat_display = pn.pane.Markdown(render_history(), sizing_mode="stretch_width", height=500)

# 创建输入组件
user_input = pn.widgets.TextInput(
    placeholder="输入你的订单需求...",
    name="订单输入",
    width=400
)

# 发送按钮
send_button = pn.widgets.Button(
    name="发送",
    button_type="primary",
    width=100
)

# 处理发送事件
def on_send(event):
    user_msg = user_input.value.strip()
    if not user_msg:
        return
    
    # 添加用户消息到历史
    history.append({"role": "user", "content": user_msg})
    chat_display.object = render_history()
    
    # 添加AI占位消息
    history.append({"role": "assistant", "content": ""})
    chat_display.object = render_history()
    
    # 调用API并处理流式响应
    try:
        completion = client.chat.completions.create(
            model="qwen-flash",
            messages=history,
            extra_body={"enable_thinking": True},
            stream=True
        )
        
        # 处理流式响应
        for chunk in completion:
            if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
                # 更新思考过程
                history[-1]["content"] += chunk.choices[0].delta.reasoning_content
                chat_display.object = render_history()
            
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                # 更新回复内容
                history[-1]["content"] += chunk.choices[0].delta.content
                chat_display.object = render_history()
                
    except Exception as e:
        # 处理API错误
        history[-1]["content"] = f"系统错误: {str(e)}"
        chat_display.object = render_history()
    
    # 清空输入框
    user_input.value = ""

# 绑定按钮事件
send_button.on_click(on_send)

# 创建应用布局
app = pn.Column(
    pn.pane.Markdown("# 🧋 奶茶店订餐机器人", sizing_mode="stretch_width", margin=10),
    chat_display,
    pn.Row(
        user_input,
        send_button,
        sizing_mode="stretch_width"
    ),
    sizing_mode="stretch_width",
    margin=20
)

# 启动应用
if __name__ == "__main__":
    app.servable().show()
