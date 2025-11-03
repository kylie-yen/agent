:::tips
链接：[datawhale翻译](https://github.com/datawhalechina/llm-cookbook)  [英文原版课程](https://learn.deeplearning.ai/)

状态：已完成

时间：20251103

:::

prompt：输入给大模型的内容 

completion：大模型返回的输出

## 原则 guidelines
prompt设计的两个原则：清晰具体、思考时间充足

### 指令清晰、具体
提供丰富的上下文、细节，不带歧义

#### 使用分隔符
将不同的指令、上下文、输入隔开，避免意外的混淆，可以使用''' """ <> <tag> </tag> , ;等

可以防止提示词注入 prompt rejection（ 用户输入的文本可能包含与你的预设 Prompt 相冲突的内容  ）

```python
from tool import get_completion

text = f"""
您应该提供尽可能清晰、具体的指示，以表达您希望模型执行的任务。\
这将引导模型朝向所需的输出，并降低收到无关或不正确响应的可能性。\
不要将写清晰的提示词与写简短的提示词混淆。\
在许多情况下，更长的提示词可以为模型提供更多的清晰度和上下文信息，从而导致更详细和相关的输出。
"""
# 需要总结的文本内容
prompt = f"""
把用三个反引号括起来的文本总结成一句话。
```{text}```
"""
# 指令内容，使用 ``` 来分隔指令和待总结的内容
response = get_completion(prompt)
print(response)

```

#### 结构化输出
如json html

```python
prompt = f"""
请生成包括书名、作者和类别的三本虚构的、非真实存在的中文书籍清单，\
并以 JSON 格式提供，其中包含以下键:book_id、title、author、genre。
"""
response = get_completion(prompt)
print(response)

{
    "books": [
        {
            "book_id": 1,
            "title": "迷失的时光",
            "author": "张三",
            "genre": "科幻"
        },
        {
            "book_id": 2,
            "title": "幻境之门",
            "author": "李四",
            "genre": "奇幻"
        },
        {
            "book_id": 3,
            "title": "虚拟现实",
            "author": "王五",
            "genre": "科幻"
        }
    ]
}

```

#### 要求模型检查是否满足要求
```python
# 满足条件的输入（text中提供了步骤）
text_1 = f"""
泡一杯茶很容易。首先，需要把水烧开。\
在等待期间，拿一个杯子并把茶包放进去。\
一旦水足够热，就把它倒在茶包上。\
等待一会儿，让茶叶浸泡。几分钟后，取出茶包。\
如果您愿意，可以加一些糖或牛奶调味。\
就这样，您可以享受一杯美味的茶了。
"""
prompt = f"""
您将获得由三个引号括起来的文本。\
如果它包含一系列的指令，则需要按照以下格式重新编写这些指令：

第一步 - ...
第二步 - …
…
第N步 - …

如果文本中不包含一系列的指令，则直接写“未提供步骤”。"
\"\"\"{text_1}\"\"\"
"""
response = get_completion(prompt)
print("Text 1 的总结:")
print(response)

# 不满足条件的输入（text中未提供预期指令）
text_2 = f"""
今天阳光明媚，鸟儿在歌唱。\
这是一个去公园散步的美好日子。\
鲜花盛开，树枝在微风中轻轻摇曳。\
人们外出享受着这美好的天气，有些人在野餐，有些人在玩游戏或者在草地上放松。\
这是一个完美的日子，可以在户外度过并欣赏大自然的美景。
"""
prompt = f"""
您将获得由三个引号括起来的文本。\
如果它包含一系列的指令，则需要按照以下格式重新编写这些指令：

第一步 - ...
第二步 - …
…
第N步 - …

如果文本中不包含一系列的指令，则直接写“未提供步骤”。"
\"\"\"{text_2}\"\"\"
"""
response = get_completion(prompt)
print("Text 2 的总结:")
print(response)

```

#### 提供少量示例 few-shot prompting
在要求模型执行实际任务之前，给模型一两个已完成的样例，让模型了解我们的要求和期望的输出样式

```python
prompt = f"""
您的任务是以一致的风格回答问题。

<孩子>: 请教我何为耐心。

<祖父母>: 挖出最深峡谷的河流源于一处不起眼的泉眼；最宏伟的交响乐从单一的音符开始；最复杂的挂毯以一根孤独的线开始编织。

<孩子>: 请教我何为韧性。
"""
response = get_completion(prompt)
print(response)

```

### 给模型推理时间
在 Prompt 中添加逐步推理的要求，能让语言模型投入更多时间逻辑思维，输出结果也将更可靠准确。

#### 指定完成任务的步骤
指定输出格式，避免问题

```python
prompt_2 = f"""
1-用一句话概括下面用<>括起来的文本。
2-将摘要翻译成英语。
3-在英语摘要中列出每个名称。
4-输出一个 JSON 对象，其中包含以下键：English_summary，num_names。

请使用以下格式：
文本：<要总结的文本>
摘要：<摘要>
翻译：<摘要的翻译>
名称：<英语摘要中的名称列表>
输出 JSON：<带有 English_summary 和 num_names 的 JSON>

Text: <{text}>
"""
response = get_completion(prompt_2)
print("\nprompt 2:")
print(response)

```

#### 让模型自主思考，再和自己的答案对比
```python
prompt = f"""
请判断学生的解决方案是否正确，请通过如下步骤解决这个问题：

步骤：

    首先，自己解决问题。
    然后将您的解决方案与学生的解决方案进行比较，对比计算得到的总费用与学生计算的总费用是否一致，并评估学生的解决方案是否正确。
    在自己完成问题之前，请勿决定学生的解决方案是否正确。

使用以下格式：

    问题：问题文本
    学生的解决方案：学生的解决方案文本
    实际解决方案和步骤：实际解决方案和步骤文本
    学生计算的总费用：学生计算得到的总费用
    实际计算的总费用：实际计算出的总费用
    学生计算的费用和实际计算的费用是否相同：是或否
    学生的解决方案和实际解决方案是否相同：是或否
    学生的成绩：正确或不正确

问题：

    我正在建造一个太阳能发电站，需要帮助计算财务。 
    - 土地费用为每平方英尺100美元
    - 我可以以每平方英尺250美元的价格购买太阳能电池板
    - 我已经谈判好了维护合同，每年需要支付固定的10万美元，并额外支付每平方英尺10美元;

    作为平方英尺数的函数，首年运营的总费用是多少。

学生的解决方案：

    设x为发电站的大小，单位为平方英尺。
    费用：
    1. 土地费用：100x美元
    2. 太阳能电池板费用：250x美元
    3. 维护费用：100,000+100x=10万美元+10x美元
    总费用：100x美元+250x美元+10万美元+100x美元=450x+10万美元

实际解决方案和步骤：
"""
response = get_completion(prompt)
print(response)

```

### limits
幻觉 hallucination

可以先让语言模型直接引用文本中的原句，然后再进行解答。这可以追踪信息来源，降低虚假内容的风险。

## 迭代优化 iterative
![prompt迭代优化流程](https://cdn.nlark.com/yuque/0/2025/png/42911247/1762137577650-cdb962c6-2ec4-4f21-9f3b-e2c6bb63624d.png)

+ 文本长度限制
+ 训练语言模型**根据不同目标受众关注不同的方面，输出风格和内容上都适合的文本**
+ 添加表格描述
    - 指定表格的列、表名和格式；再将所有内容格式化为可以在网页使用的 HTML

```python
# 要求它抽取信息并组织成表格，并指定表格的列、表名和格式
prompt = f"""
您的任务是帮助营销团队基于技术说明书创建一个产品的零售网站描述。

根据```标记的技术说明书中提供的信息，编写一个产品描述。

该描述面向家具零售商，因此应具有技术性质，并侧重于产品的材料构造。

在描述末尾，包括技术规格中每个7个字符的产品ID。

在描述之后，包括一个表格，提供产品的尺寸。表格应该有两列。第一列包括尺寸的名称。第二列只包括英寸的测量值。

给表格命名为“产品尺寸”。

将所有内容格式化为可用于网站的HTML格式。将描述放在<div>元素中。

技术规格：```{fact_sheet_chair}```
"""

response = get_completion(prompt)
print(response)

```

## 文本概括 summarizing
LLM文本摘要功能的巨大优势：**节省时间**，**提高效率**，以及**精准获取信息**

### 单一文本概括
+ 限制文本输出长度
+ 设置关键角度侧重

```python
prompt = f"""
您的任务是从电子商务网站上生成一个产品评论的简短摘要。

请对三个反引号之间的评论文本进行概括，最多30个词汇，并且侧重在产品价格和质量上。

评论: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)

```

+ 关键信息提取

```python
prompt = f"""
您的任务是从电子商务网站上的产品评论中提取相关信息。

请从以下三个反引号之间的评论文本中提取产品运输相关的信息，最多30个词汇。

评论: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)

```

### 同时概括多条文本
```python
for i in range(len(reviews)):
    prompt = f"""
    你的任务是从电子商务网站上的产品评论中提取相关信息。

    请对三个反引号之间的评论文本进行概括，最多20个词汇。

    评论文本: ```{reviews[i]}```
    """
    response = get_completion(prompt)
    print(f"评论{i+1}: ", response, "\n")

```

## 推断 inferring
### 情感倾向分析
```python
prompt = f"""
以下用三个反引号分隔的产品评论的情感是什么？

用一个单词回答：「正面」或「负面」。

评论文本: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)

```

+ 识别情感类型

```python
# 中文
prompt = f"""
识别以下评论的作者表达的情感。包含不超过五个项目。将答案格式化为以逗号分隔的单词列表。

评论文本: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)

```

+ 识别愤怒

```python
# 中文
prompt = f"""
以下评论的作者是否表达了愤怒？评论用三个反引号分隔。给出是或否的答案。

评论文本: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)

```

### 信息提取
信息提取是自然语言处理（NLP）的重要组成部分，它帮助我们从文本中抽取特定的、我们关心的信息。

#### 综合情感推断与信息提取
```python
# 中文
prompt = f"""
从评论文本中识别以下项目：
- 情绪（正面或负面）
- 审稿人是否表达了愤怒？（是或否）
- 评论者购买的物品
- 制造该物品的公司

评论用三个反引号分隔。将你的响应格式化为 JSON 对象，以 “情感倾向”、“是否生气”、“物品类型” 和 “品牌” 作为键。
如果信息不存在，请使用 “未知” 作为值。
让你的回应尽可能简短。
将 “是否生气” 值格式化为布尔值。

评论文本: ```{lamp_review}```
"""
response = get_completion(prompt)
print(response)

```

### 主题推断
+ 推断主题

```python
# 中文
prompt = f"""
确定以下给定文本中讨论的五个主题。

每个主题用1-2个词概括。

请输出一个可解析的Python列表，每个元素是一个字符串，展示了一个主题。

给定文本: ```{story}```
"""
response = get_completion(prompt)
print(response)

```

## 文本转换 transforming
### 文本翻译
 通过在大规模高质量平行语料上进行 Fine-Tune，大语言模型可以深入学习不同语言间的词汇、语法、语义等层面的对应关系，模拟双语者的转换思维，进行意义传递的精准转换，而非简单的逐词替换。

+ 翻译为目标语言
+ 识别语种
+ 多语种翻译
+ 同时语气转换
+ 通用翻译器
    - 自动识别不同语言文本的语种，无需人工指定。然后它可以将这些不同语言的文本翻译成目标用户的母语。

```python
user_messages = [
  "La performance du système est plus lente que d'habitude.",  # System performance is slower than normal         
  "Mi monitor tiene píxeles que no se iluminan.",              # My monitor has pixels that are not lighting
  "Il mio mouse non funziona",                                 # My mouse is not working
  "Mój klawisz Ctrl jest zepsuty",                             # My keyboard has a broken control key
  "我的屏幕在闪烁"                                             # My screen is flashing
]


import time
for issue in user_messages:
    time.sleep(20)
    prompt = f"告诉我以下文本是什么语种，直接输出语种，如法语，无需输出标点符号: ```{issue}```"
    lang = get_completion(prompt)
    print(f"原始消息 ({lang}): {issue}\n")

    prompt = f"""
    将以下消息分别翻译成英文和中文，并写成
    中文翻译：xxx
    英文翻译：yyy
    的格式：
    ```{issue}```
    """
    response = get_completion(prompt)
    print(response, "\n=========================================")

```

### 语气与写作风格调整
```python
prompt = f"""
将以下文本翻译成商务信函的格式: 
```小老弟，我小羊，上回你说咱部门要采购的显示器是多少寸来着？```
"""
response = get_completion(prompt)
print(response)

```

### 文件格式转换
实现 JSON 到 HTML、XML、Markdown 等格式的相互转化

```python
data_json = { "resturant employees" :[ 
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}

prompt = f"""
将以下Python字典从JSON转换为HTML表格，保留表格标题和列名：{data_json}
"""
response = get_completion(prompt)
print(response)

```

### 拼写及语法纠正
```python
for i in range(len(text)):
    time.sleep(20)
    prompt = f"""请校对并更正以下文本，注意纠正文本保持原始语种，无需输出原始文本。
    如果您没有发现任何错误，请说“未发现错误”。
    
    例如：
    输入：I are happy.
    输出：I am happy.
    ```{text[i]}```"""
    response = get_completion(prompt)
    print(i, response)

```

```python
from redlines import Redlines
from IPython.display import display, Markdown

diff = Redlines(text,response)
display(Markdown(diff.output_markdown))

```

### 综合
同时对一段文本进行翻译、拼写纠正、语气调整和格式转换等操作。

```python
prompt = f"""
针对以下三个反引号之间的英文评论文本，
首先进行拼写及语法纠错，
然后将其转化成中文，
再将其转化成优质淘宝评论的风格，从各种角度出发，分别说明产品的优点与缺点，并进行总结。
润色一下描述，使评论更具有吸引力。
输出结果格式为：
【优点】xxx
【缺点】xxx
【总结】xxx
注意，只需填写xxx部分，并分段输出。
将结果输出成Markdown格式。
```{text}```
"""
response = get_completion(prompt)
display(Markdown(response))

```

## 文本扩展 expanding
temperature 控制文本生成的多样性

### 回复邮件
输入客户的评论文本和对应的情感分析结果(正面或者负面)。然后构造一个 Prompt，要求大语言模型基于这些信息来生成一封定制的回复电子邮件

```python
from tool import get_completion

prompt = f"""
你是一位客户服务的AI助手。
你的任务是给一位重要客户发送邮件回复。
根据客户通过“```”分隔的评价，生成回复以感谢客户的评价。提醒模型使用评价中的具体细节
用简明而专业的语气写信。
作为“AI客户代理”签署电子邮件。
客户评论：
```{review}```
评论情感：{sentiment}
"""
response = get_completion(prompt)
print(response)

```

### temperature
大语言模型中的 “温度”(temperature) 参数可以控制生成文本的随机性和多样性。temperature 的值越大，语言模型输出的多样性越大；temperature 的值越小，输出越倾向高概率的文本。

![](https://cdn.nlark.com/yuque/0/2025/png/42911247/1762140126647-ce31ad85-7bb5-4fbb-b2cb-d76a0fb593e4.png)

## 聊天机器人 chatbot
两个方法：

get_completion()

适用于单论对话



get_completion_from_message()

传入一个消息列表，来自不同的角色roles 

+ system message 系统消息，提供总体的指令，设置助手的行为和角色，作为对话的高级指示
+ user message
+ assistant message

```python
import openai

# 下文第一个函数即tool工具包中的同名函数，此处展示出来以便于读者对比
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # 控制模型输出的随机程度
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # 控制模型输出的随机程度
    )
#     print(str(response.choices[0].message))
    return response.choices[0].message["content"]

```

### 给定身份
讲笑话

```python
# 中文
messages =  [  
{'role':'system', 'content':'你是一个像莎士比亚一样说话的助手。'},    
{'role':'user', 'content':'给我讲个笑话'},   
{'role':'assistant', 'content':'鸡为什么过马路'},   
{'role':'user', 'content':'我不知道'}  ]

response = get_completion_from_messages(messages, temperature=1)
print(response)

```

### 构建上下文 context
```python
# 中文
messages =  [  
{'role':'system', 'content':'你是个友好的聊天机器人。'},
{'role':'user', 'content':'Hi, 我是Isa'},
{'role':'assistant', 'content': "Hi Isa! 很高兴认识你。今天有什么可以帮到你的吗?"},
{'role':'user', 'content':'是的，你可以提醒我, 我的名字是什么?'}  ]
response = get_completion_from_messages(messages, temperature=1)
print(response)

```

### 订餐机器人构建
自动收集用户信息，接受订单

```python
def collect_messages(_):
    prompt = inp.value_input
    inp.value = ''
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context) 
    context.append({'role':'assistant', 'content':f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))
 
    return pn.Column(*panels)

```



通过包panel完成界面可视化

```python
# 中文
import panel as pn  # GUI
pn.extension()

panels = [] # collect display 

context = [{'role':'system', 'content':"""
你是订餐机器人，为披萨餐厅自动收集订单信息。
你要首先问候顾客。然后等待用户回复收集订单信息。收集完信息需确认顾客是否还需要添加其他内容。
最后需要询问是否自取或外送，如果是外送，你要询问地址。
最后告诉顾客订单总金额，并送上祝福。

请确保明确所有选项、附加项和尺寸，以便从菜单中识别出该项唯一的内容。
你的回应应该以简短、非常随意和友好的风格呈现。

菜单包括：

菜品：
意式辣香肠披萨（大、中、小） 12.95、10.00、7.00
芝士披萨（大、中、小） 10.95、9.25、6.50
茄子披萨（大、中、小） 11.95、9.75、6.75
薯条（大、小） 4.50、3.50
希腊沙拉 7.25

配料：
奶酪 2.00
蘑菇 1.50
香肠 3.00
加拿大熏肉 3.50
AI酱 1.50
辣椒 1.00

饮料：
可乐（大、中、小） 3.00、2.00、1.00
雪碧（大、中、小） 3.00、2.00、1.00
瓶装水 5.00
"""} ]  # accumulate messages


inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard

```

# 
## 


