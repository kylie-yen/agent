import mesa
import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime

# ==========================================
# 1. 配置与全局参数
# ==========================================
# 基础农业参数
RICE_INCOME_PER_MU = 1000      # 水稻保底净收益 (元/亩)
CASH_CROP_COST_PER_MU = 2000   # 经济作物投入成本 (元/亩)
CASH_CROP_BASE_YIELD = 300     # 经济作物基准产量 (kg/亩)
BASE_MARKET_PRICE = 10         # 经济作物初始收购价 (元/kg) -> 初始利润 = 300*10 - 2000 = 1000 (33%利润率)

# 系统约束
TAX_RATE = 0.15                # 税率/回流给政府的比例 (15%)
LABOR_CAPACITY = 5.0           # 每个劳动力能照看的最大亩数 (超过需雇人)
HIRE_COST = 800                # 雇佣劳动力成本 (元/亩)

# LLM 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"      # 确保你本地已 `ollama pull qwen2.5:3b`

# 日志文件
LOG_FILE = "agent_thoughts.log"
# 清空旧日志
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"Simulation Start: {datetime.now()}\n{'='*50}\n")

# ==========================================
# 2. 工具函数：调用本地 LLM
# ==========================================
def query_ollama(prompt, context=""):
    """
    发送Prompt给Ollama，并强制要求返回JSON格式
    """
    full_prompt = f"""
    {context}
    
    任务：{prompt}
    
    【重要限制】
    1. 你必须扮演上述描述的农户。
    2. 请只输出标准的 JSON 格式，不要包含任何其他解释性文字或Markdown标记。
    3. JSON格式必须如下：
    {{
        "plant_cash_crop_ratio": 0.0到1.0之间的浮点数 (表示种植经济作物的土地比例),
        "loan_amount": 整数 (申请贷款金额，若不贷则为0),
        "reasoning": "一句话解释你的决策逻辑"
    }}
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "format": "json" # 强制JSON模式 (Ollama新版特性)
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()['response']
        return json.loads(result)
    except Exception as e:
        print(f"LLM调用失败: {e}")
        # 失败时的保守兜底策略
        return {"plant_cash_crop_ratio": 0.1, "loan_amount": 0, "reasoning": "Error fallback"}

# ==========================================
# 3. Agent 定义
# ==========================================
class FarmerAgent(mesa.Agent):
    def __init__(self, unique_id, model, row_data):
        # 【修复点1】Mesa 3.0 父类初始化只接受 model，不要传 unique_id
        super().__init__(model) 
        
        # 【修复点2】手动设置 unique_id
        self.unique_id = unique_id 
        
        # 从CSV加载的属性
        self.family_size = row_data['Family_Size']
        self.labor = row_data['Labor_Count']
        self.savings = row_data['Initial_Savings_Yuan']
        self.land_area = row_data['Land_Area_Mu']
        self.risk_prompt = row_data['Risk_Profile_Prompt']
        self.debt = row_data['Debt_Yuan']
        
        # 动态属性
        self.cash_crop_area = 0
        self.rice_area = 0
        self.net_income = 0
        self.is_bankrupt = False

    def step(self):
        if self.is_bankrupt:
            return

        # --- A. 感知环境 (Perception) ---
        subsidy_per_mu = self.model.current_subsidy_per_mu
        market_price_last_year = self.model.market_price
        
        # 构建 Prompt Context
        context = f"""
        你是一名中国农户。
        【个人资产】
        - 存款: {self.savings:.2f} 元
        - 负债: {self.debt:.2f} 元
        - 土地: {self.land_area:.2f} 亩
        - 劳动力: {self.labor} 人 (超过 {self.labor * LABOR_CAPACITY} 亩的种植需雇人，成本{HIRE_COST}元/亩)
        
        【性格画像】
        {self.risk_prompt}
        
        【市场环境】
        - 政府经济作物补贴: {subsidy_per_mu:.2f} 元/亩
        - 去年经济作物市价: {market_price_last_year:.2f} 元/kg (基准成本约 {CASH_CROP_COST_PER_MU/CASH_CROP_BASE_YIELD:.1f}元/kg)
        - 种植水稻是保底选择，风险极低但收益有限。
        """

        prompt = "基于你的性格和当前资产，决定今年种植多少比例的经济作物？是否需要申请小额贷款（用于投入成本）？"

        # --- B. 决策 (Decision by LLM) ---
        # 只有在关键时刻或每隔几年调用一次LLM以节省时间，这里简化为每年调用
        # 或者是简单规则 + LLM微调。为了演示，我们直接调用。
        decision = query_ollama(prompt, context)
        
        # 解析决策
        ratio = float(decision.get('plant_cash_crop_ratio', 0))
        loan_req = float(decision.get('loan_amount', 0))
        reason = decision.get('reasoning', '')

        # 记录思维链
        self.log_thought(context, decision)

        # 执行种植决策
        self.cash_crop_area = self.land_area * ratio
        self.rice_area = self.land_area * (1 - ratio)
        
        # 处理贷款 (简单逻辑：如果存款不够投入，必须借贷；或者主动借贷)
        input_cost = self.cash_crop_area * CASH_CROP_COST_PER_MU
        
        # 如果主动申请贷款
        if loan_req > 0:
            self.debt += loan_req
            self.savings += loan_req
            
        # 支付成本
        if self.savings >= input_cost:
            self.savings -= input_cost
        else:
            # 钱不够，被迫借贷补足
            shortfall = input_cost - self.savings
            self.debt += shortfall
            self.savings = 0

    def calculate_income(self, final_market_price):
        if self.is_bankrupt:
            return

        # 1. 产出计算
        # 经济作物收入
        cash_yield = self.cash_crop_area * CASH_CROP_BASE_YIELD # 简化，未加天气随机性
        cash_income = cash_yield * final_market_price
        
        # 水稻收入 (保底)
        rice_income = self.rice_area * RICE_INCOME_PER_MU
        
        # 补贴收入
        subsidy_income = self.cash_crop_area * self.model.current_subsidy_per_mu
        
        # 2. 成本计算 (雇佣成本)
        labor_needed = self.land_area / LABOR_CAPACITY
        hire_cost = 0
        if labor_needed > self.labor:
            hire_cost = (labor_needed - self.labor) * LABOR_CAPACITY * HIRE_COST
            
        # 3. 债务利息 (假设5%利息)
        interest = self.debt * 0.05
        self.debt += interest
        
        # 4. 总核算
        gross_income = cash_income + rice_income + subsidy_income
        total_expense = hire_cost + interest # 种植成本已在step中扣除
        
        self.net_income = gross_income - total_expense
        
        # 税收/生活支出 (流回政府)
        tax = max(0, self.net_income * TAX_RATE)
        self.model.govt_balance += tax
        
        # 更新资产
        self.savings += (self.net_income - tax)
        
        # 偿还债务 logic
        if self.savings > 0 and self.debt > 0:
            repay = min(self.savings, self.debt)
            self.savings -= repay
            self.debt -= repay
            
        # 破产判定 (连续两年负债且无存款，简化判定)
        if self.debt > self.land_area * 5000: # 资不抵债
            self.is_bankrupt = True

    def log_thought(self, context, decision):
        """将Agent的思考过程写入本地文件"""
        log_entry = f"\n[Agent {self.unique_id} - Year {self.model.year}]\n"
        log_entry += f"Risk Profile: {self.risk_prompt[:50]}...\n"
        log_entry += f"Decision: Cash Crop Ratio {decision.get('plant_cash_crop_ratio')}, Loan {decision.get('loan_amount')}\n"
        log_entry += f"Reasoning: {decision.get('reasoning')}\n"
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)

# ==========================================
# 4. 系统模型 (Model)
# ==========================================
class VillageModel(mesa.Model):
    def __init__(self, agent_df):
        super().__init__()
        self.year = 0
        
        # 【修正点1】改名为 self.farmers，避免和 Mesa 3.0 的 self.agents 冲突
        self.farmers = [] 
        
        # 初始化政府
        self.govt_budget = 1000000 # 初始预算 100万
        self.govt_balance = self.govt_budget
        self.base_subsidy = 300    # 初始补贴
        self.current_subsidy_per_mu = self.base_subsidy
        
        # 初始化市场
        self.market_price = BASE_MARKET_PRICE
        
        # 创建Agents
        for i, row in agent_df.iterrows():
            a = FarmerAgent(i, self, row)
            # 【修正点2】添加到自定义列表
            self.farmers.append(a)
            
        # 数据收集器
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Market_Price": "market_price",
                "Govt_Balance": "govt_balance",
                # 【修正点3】lambda函数里也要改名字
                "Total_Cash_Crop_Area": lambda m: sum([a.cash_crop_area for a in m.farmers]),
                "Bankrupt_Count": lambda m: sum([1 for a in m.farmers if a.is_bankrupt]),
                "Average_Savings": lambda m: np.mean([a.savings for a in m.farmers])
            }
        )

    def step(self):
        self.year += 1
        print(f"--- Year {self.year} Start ---")
        
        # 1. 农户决策 (Agent Decision)
        # 随机激活顺序
        # 【修正点4】洗牌和遍历都用 self.farmers
        np.random.shuffle(self.farmers)
        for agent in self.farmers:
            agent.step()
            
        # 2. 市场反馈机制 (Market Mechanism)
        total_cash_area = sum([a.cash_crop_area for a in self.farmers])
        total_land = sum([a.land_area for a in self.farmers])
        supply_ratio = total_cash_area / total_land if total_land > 0 else 0
        
        # 价格曲线
        self.market_price = BASE_MARKET_PRICE * (1.3 - 1.0 * supply_ratio)
        self.market_price = max(self.market_price, 3.0) 
        
        print(f"Market: Supply Ratio {supply_ratio:.2%}, New Price {self.market_price:.2f}")

        # 3. 政府财政结算 (Policy Feedback)
        total_subsidy_needed = total_cash_area * self.current_subsidy_per_mu
        
        if total_subsidy_needed > self.govt_balance:
            print("Warning: Government Budget Deficit! Subsidy will be cut next year.")
            self.govt_balance = 0
            self.current_subsidy_per_mu *= 0.8 
        else:
            self.govt_balance -= total_subsidy_needed
            if self.govt_balance > 500000:
                self.current_subsidy_per_mu = min(self.current_subsidy_per_mu * 1.05, 500)
                
        # 4. 农户核算当年收入
        for agent in self.farmers:
            agent.calculate_income(self.market_price)
            
        self.datacollector.collect(self)

# ==========================================
# 5. 运行脚本
# ==========================================
if __name__ == "__main__":
    # 读取数据
    try:
        df = pd.read_csv(".\win_win\data\Farmer_Agents_Initialized.csv")
    except FileNotFoundError:
        print("Error: 找不到 Farmer_Agents_Initialized.csv，请确保文件在同一目录下。")
        exit()

    # 初始化模型
    # 为了测试速度，可以先只取前10个农户测试
    # df_test = df.head(10) 
    model = VillageModel(df) 
    
    # 运行模拟 (例如运行 5 年)
    params = {"years": 5}
    print(f"Starting simulation for {params['years']} years with {len(df)} agents...")
    
    for _ in range(params['years']):
        model.step()
        
    # 保存结果
    results = model.datacollector.get_model_vars_dataframe()
    results.to_csv(".\win_win\data\Simulation_Results.csv")
    print("\nSimulation Finished. Results saved to 'Simulation_Results.csv'.")
    print("Agent thoughts saved to 'agent_thoughts.log'.")
    
    # 简单打印结果预览
    print(results)
