import mesa
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime
from collections import defaultdict

# ==========================================
# 1. 配置与全局参数
# ==========================================
MATURITY_YEARS = 5             # 经济作物成熟期 (年)
RICE_INCOME_PER_MU = 1000      # 水稻保底净收益 (元/亩)
CASH_CROP_COST_PER_MU = 2000   # 经济作物投入成本 (元/亩，每年都要投)
CASH_CROP_BASE_YIELD = 300     # 经济作物基准产量 (kg/亩)
BASE_MARKET_PRICE = 10         # 初始收购价

# 系统约束
TAX_RATE = 0.15                # 税率
LABOR_CAPACITY = 5.0           # 劳动力覆盖能力
HIRE_COST = 800                # 雇佣成本
LOAN_INTEREST_RATE = 0.05      # 贷款利率

# LLM 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"
LOG_FILE = "agent_thoughts.log"

# 初始化日志
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"Simulation Start: {datetime.now()}\n{'='*50}\n")

# ==========================================
# 2. 工具函数
# ==========================================
def query_ollama(prompt, context=""):
    full_prompt = f"""
    {context}
    
    任务：{prompt}
    
    【输出限制】
    只输出标准的 JSON 格式：
    {{
        "target_cash_crop_ratio": 0.0-1.0 (希望经济作物占总土地的比例),
        "loan_amount": 整数 (申请贷款金额),
        "reasoning": "简短的一句决策理由"
    }}
    """
    payload = {
        "model": MODEL_NAME, "prompt": full_prompt, "stream": False, "format": "json"
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        return json.loads(response.json()['response'])
    except:
        return {"target_cash_crop_ratio": 0.1, "loan_amount": 0, "reasoning": "Fallback"}

# ==========================================
# 3. Agent 定义 (引入作物年龄)
# ==========================================
class FarmerAgent(mesa.Agent):
    def __init__(self, unique_id, model, row_data):
        super().__init__(model) # Mesa 3.0 fix
        self.unique_id = unique_id
        
        # 属性初始化
        self.family_size = row_data['Family_Size']
        self.labor = row_data['Labor_Count']
        self.savings = row_data['Initial_Savings_Yuan']
        self.land_area = row_data['Land_Area_Mu']
        self.risk_prompt = row_data['Risk_Profile_Prompt']
        self.debt = row_data['Debt_Yuan']
        
        # 作物记录：{种植年份: 面积}
        # eg {1: 2.0, 3: 1.5} 表示第1年种了2亩，第3年种了1.5亩
        self.crop_schedule = defaultdict(float) 
        
        self.net_income = 0
        self.is_bankrupt = False

    @property
    def total_cash_area(self):
        """当前经济作物总面积"""
        return sum(self.crop_schedule.values())

    @property
    def mature_cash_area(self):
        """当前已成熟（可收获）的面积"""
        mature_area = 0
        for plant_year, area in self.crop_schedule.items():
            age = self.model.year - plant_year
            if age >= MATURITY_YEARS:
                mature_area += area
        return mature_area

    def step(self):
        if self.is_bankrupt: return

        # --- 1. 感知与LLM决策 ---
        mature_area = self.mature_cash_area
        immature_area = self.total_cash_area - mature_area
        
        context = f"""
        【年份: 第 {self.model.year} 年】
        你是一名农户。需注意：经济作物种植后需 {MATURITY_YEARS} 年才能收获！
        在此期间，你每年每亩需投入 {CASH_CROP_COST_PER_MU} 元维护，但没有卖树收入，只能靠政府补贴和积蓄。
        
        【资产状况】
        - 存款: {self.savings:.0f} 元 | 负债: {self.debt:.0f} 元
        - 总土地: {self.land_area:.1f} 亩
        - 现有经济作物: {self.total_cash_area:.1f} 亩 (其中 {mature_area:.1f} 亩已成熟可赚钱，{immature_area:.1f} 亩只有投入无产出)
        - 剩余种植水稻: {self.land_area - self.total_cash_area:.1f} 亩 (保底收益)
        
        【外部环境】
        - 补贴: {self.model.current_subsidy_per_mu:.0f} 元/亩 (针对所有经济作物)
        - 市场价: {self.model.market_price:.2f} 元/kg (仅成熟作物有产出)
        
        【性格】{self.risk_prompt}
        """

        decision = query_ollama("基于长期投资视角，决定今年的目标种植比例？", context)
        
        # --- 2. 调整种植结构 ---
        target_ratio = float(decision.get('target_cash_crop_ratio', 0))
        target_cash_area = self.land_area * target_ratio
        current_cash_area = self.total_cash_area
        
        diff = target_cash_area - current_cash_area
        
        if diff > 0.01:
            # 扩种：记录当前年份为种植年份
            self.crop_schedule[self.model.year] += diff

        elif diff < -0.01:
                    # 缩减（砍树）逻辑
                    remove_amount = abs(diff)
                    sorted_years = sorted(self.crop_schedule.keys(), reverse=True)
                    for y in sorted_years:
                        if remove_amount <= 0: break
                        available = self.crop_schedule[y]
                        if available >= remove_amount:
                            self.crop_schedule[y] -= remove_amount
                            remove_amount = 0
                        else:
                            self.crop_schedule[y] = 0
                            remove_amount -= available

        # --- 3. 资金处理 (投入阶段) ---
        # 只有维护成本，没有产出收入。产出在 calculate_income 算
        total_maintenance_cost = self.total_cash_area * CASH_CROP_COST_PER_MU
        
        # 处理贷款
        loan_req = float(decision.get('loan_amount', 0))
        if loan_req > 0:
            self.debt += loan_req
            self.savings += loan_req
            
        # 扣除成本
        if self.savings >= total_maintenance_cost:
            self.savings -= total_maintenance_cost
        else:
            shortfall = total_maintenance_cost - self.savings
            self.savings = 0
            self.debt += shortfall # 钱不够被迫负债
            
        # 记录日志
        self.log_thought(context, decision)

    def calculate_income(self, market_price):
        if self.is_bankrupt: return
        
        # 1. 收入计算
        # 只有成熟的树才有产量！
        yield_kg = self.mature_cash_area * CASH_CROP_BASE_YIELD
        cash_income = yield_kg * market_price
        
        rice_area = self.land_area - self.total_cash_area
        rice_income = rice_area * RICE_INCOME_PER_MU
        
        # 补贴是针对所有种植面积的（不仅是成熟的），这是政府的激励手段
        subsidy_income = self.total_cash_area * self.model.current_subsidy_per_mu
        
        # 2. 支出计算 (雇佣)
        labor_needed = self.land_area / LABOR_CAPACITY
        hire_cost = 0
        if labor_needed > self.labor:
            hire_cost = (labor_needed - self.labor) * LABOR_CAPACITY * HIRE_COST
            
        interest_cost = self.debt * LOAN_INTEREST_RATE
        self.debt += interest_cost
        
        # 3. 结算
        gross_income = cash_income + rice_income + subsidy_income
        total_expense = hire_cost + interest_cost # 种植投入已在step扣除
        
        self.net_income = gross_income - total_expense
        
        # 4. 税收与资产更新
        tax = max(0, self.net_income * TAX_RATE)
        self.model.govt_balance += tax
        
        self.savings += (self.net_income - tax)
        
        # 还债逻辑
        if self.savings > 0 and self.debt > 0:
            repay = min(self.savings, self.debt)
            self.savings -= repay
            self.debt -= repay
            
        # 破产判定
        if self.debt > self.land_area * 8000: 
            self.is_bankrupt = True

    def log_thought(self, context, decision):
        entry = f"\n[Agent {self.unique_id} - Year {self.model.year}]\n"
        entry += f"Decision: Target Ratio {decision.get('target_cash_crop_ratio')}, Loan {decision.get('loan_amount')}\n"
        entry += f"Reasoning: {decision.get('reasoning')}\n"
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)

# ==========================================
# 4. 系统模型
# ==========================================
class VillageModel(mesa.Model):
    def __init__(self, agent_df):
        super().__init__()
        self.year = 0
        self.farmers = []  # 自定义Agent列表
        
        self.govt_budget = 2000000 # 初始预算，否则撑不过5年
        self.govt_balance = self.govt_budget
        self.current_subsidy_per_mu = 400 # 提高初始补贴，否则没人敢种
        self.market_price = BASE_MARKET_PRICE
        
        for i, row in agent_df.iterrows():
            a = FarmerAgent(i, self, row)
            self.farmers.append(a)
            
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Year": "year",
                "Market_Price": "market_price",
                "Govt_Balance": "govt_balance",
                "Total_Cash_Area": lambda m: sum([a.total_cash_area for a in m.farmers]),
                "Mature_Area": lambda m: sum([a.mature_cash_area for a in m.farmers]), # 新增指标
                "Avg_Debt": lambda m: np.mean([a.debt for a in m.farmers])
            }
        )

    def step(self):
        self.year += 1
        print(f"--- Year {self.year} ---")
        
        # 1. 农户决策
        np.random.shuffle(self.farmers)
        for agent in self.farmers:
            agent.step()
            
        # 2. 市场反馈 (只看成熟作物的供应量)
        total_mature_area = sum([a.mature_cash_area for a in self.farmers])
        total_land = sum([a.land_area for a in self.farmers])
        
        # 只有成熟的才会冲击市场
        supply_ratio = total_mature_area / total_land if total_land > 0 else 0
        
        if supply_ratio < 0.05:
            # 市场上几乎没货，价格维持高位
            self.market_price = BASE_MARKET_PRICE
        else:
            # 货多了，价格下跌
            self.market_price = BASE_MARKET_PRICE * (1.5 - 1.2 * supply_ratio)
            self.market_price = max(self.market_price, 2.0)
            
        print(f"Market: Mature Supply {supply_ratio:.1%}, Price {self.market_price:.2f}")

        # 3. 财政结算 (补贴是发给所有种植面积的)
        total_planted_area = sum([a.total_cash_area for a in self.farmers])
        subsidy_needed = total_planted_area * self.current_subsidy_per_mu
        
        self.govt_balance -= subsidy_needed
        
        # 简单的财政调整
        if self.govt_balance < 0:
            print("Govt Deficit! Cutting Subsidy.")
            self.current_subsidy_per_mu *= 0.7 # 大幅削减
            self.govt_balance = 0 # 假设举债归零
            
        # 4. 结算收入
        for agent in self.farmers:
            agent.calculate_income(self.market_price)
            
        self.datacollector.collect(self)

# ==========================================
# 5. 运行入口
# ==========================================
if __name__ == "__main__":
    try:
        df = pd.read_csv(".\win_win_v2\data\Farmer_Agents_Initialized.csv")
        # df = df.head(10) 
        
        model = VillageModel(df)
        
        # 跑 10 年
        for _ in range(10):
            model.step()
            
        res = model.datacollector.get_model_vars_dataframe()
        res.to_csv("Simulation_Results_v2.csv")
        print("\nFinished. Check 'Simulation_Results_v2.csv'")
        print(res[['Year', 'Govt_Balance', 'Total_Cash_Area', 'Mature_Area', 'Market_Price']].tail(10))
        
    except FileNotFoundError:
        print("Missing CSV file.")
