import json
import os
import random
from openai import OpenAI 

# =================配置区=================
# 1. 调试阶段使用性价比最高的模型
MODEL_NAME = "qwen-long" 
# 2. 关闭 Debug 模式，让 AI 真正接管
DEBUG_MODE = False 
# =======================================

class LLMBrain:
    def __init__(self):
        # 先声明引用全局变量，再做逻辑判断
        global DEBUG_MODE 
        
        # 从环境变量获取 Key
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # 检查 Key 是否存在 (非 Debug 模式下必须有 Key)
        if not DEBUG_MODE and not self.api_key:
            print("⚠️ 严重错误: 环境变量 'DASHSCOPE_API_KEY' 未配置！")
            print("Windows: set DASHSCOPE_API_KEY=sk-xxx")
            print("Mac/Linux: export DASHSCOPE_API_KEY=sk-xxx")
            
            # 强制回退到 Debug 模式
            print("🔄 已自动切换回 DEBUG 模式 (使用随机数模拟)")
            DEBUG_MODE = True
    
    def get_decision(self, system_prompt, user_prompt):
        """发送 Prompt 给 LLM 并获取结构化 JSON 决策"""
        
        # --- A. 调试模式 (随机数据) ---
        if DEBUG_MODE:
            return self._mock_decision()
        
        # --- B. 真实调用模式 ---
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # 稍微降低一点温度，让决策更稳定一点
                temperature=0.5, 
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print(f"⚠️ JSON 解析失败，原始返回: {content[:50]}...")
                return {"thought": "思维混乱，无法决策", "decision_area": 0}

            if "decision_area" not in data:
                data["decision_area"] = 0
                
            return data

        except Exception as e:
            print(f"❌ API 调用出错: {e}")
            return {"thought": "连接中断，维持现状", "decision_area": 0}

    def _mock_decision(self):
        """仅供无 Key 时的代码流程测试"""
        scenarios = [
            {"thought": "DEBUG: 没什么动力。", "area_factor": 0.0},
            {"thought": "DEBUG: 试一试吧。", "area_factor": 0.2},
            {"thought": "DEBUG: 全力投入！", "area_factor": 0.8}
        ]
        choice = random.choice(scenarios)
        return {
            "thought": choice["thought"],
            "decision_area": round(choice["area_factor"] * 10, 1)
        }