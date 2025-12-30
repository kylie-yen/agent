from mesa import Agent

class FarmerAgent(Agent):
    def __init__(self, unique_id, model, profile, total_area, survival_area):
        super().__init__(model)
        self.unique_id = unique_id
        self.profile = profile
        self.total_area = total_area
        self.survival_area = survival_area
        
        # 状态量
        self.eco_crop_area = 0.0  # 经济作物面积
        self.rice_area = total_area
        self.cash = 50000  # 初始资金（设高一点防止第一年破产）
        self.cumulative_income = 0 # 十年累计收益
        self.memory = [] 

    def calculate_annual_income(self, year, subsidy_amount):
        """
        核心经济学逻辑：分阶段计算收益
        假设：
        1. 粮食收益：稳赚 500元/亩
        2. 经作成本：投入期(前2年) 成本 800元/亩；盛果期(3年+) 维护费 500元/亩
        3. 经作产出：投入期 0产出；盛果期 产值 3500元/亩
        """
        # 基础参数
        income_grain_per_mu = 500
        cost_eco_early = 800    # 幼树期投入
        cost_eco_late = 500     # 盛果期维护
        revenue_eco_late = 3500 # 盛果期产值
        
        # 1. 计算粮食收益
        grain_income = self.rice_area * income_grain_per_mu
        
        # 2. 计算经作收益 (分阶段)
        # 注意：model.steps 从 0 开始，所以 year 1 是 steps=0
        if year <= 2: 
            # --- 投入期 (Year 1, 2) ---
            # 净收益 = 补贴 - 投入成本
            eco_net_per_mu = subsidy_amount - cost_eco_early
            eco_total_income = self.eco_crop_area * eco_net_per_mu
        else:
            # --- 收获期 (Year 3+) ---
            # 假设第3年开始就有收益 (无补贴)
            eco_net_per_mu = revenue_eco_late - cost_eco_late
            eco_total_income = self.eco_crop_area * eco_net_per_mu
            
        total_income = grain_income + eco_total_income
        return total_income, eco_net_per_mu

    def get_social_context(self):
        """获取邻居去年的行为"""
        farmers = [a for a in self.model.agents if isinstance(a, FarmerAgent) and a != self]
        if not farmers: return "你是村里第一个考虑这件事的人。"
        avg_area = sum(a.eco_crop_area for a in farmers) / len(farmers)
        planting_rate = len([f for f in farmers if f.eco_crop_area > 0]) / len(farmers)
        return f"村里有 {planting_rate:.0%} 的人种了经作，平均种植面积 {avg_area:.1f} 亩。"

    def step(self):
        current_year = self.model.steps + 1
        subsidy = self.model.subsidy_amount
        max_allowed = self.total_area - self.survival_area
        
        # --- 1. 构建 机会成本 (Opportunity Cost) Prompt ---
        # 计算如果种经作，比种粮食“亏”多少？
        # 基准：粮食赚 500
        # 现状：经作赚 (补贴 - 800)
        grain_income = 500
        eco_cost = 800
        
        if current_year <= 2:
            eco_actual_income = subsidy - eco_cost
            gap = eco_actual_income - grain_income # 负数代表比种粮亏
            
            economy_desc = f"""
            【当前阶段】：产业投入期（第 {current_year} 年）。
            【账本分析】：
            1. 种粮食：每亩稳赚 {grain_income} 元。
            2. 种经作：需要投入成本 {eco_cost} 元，政府补贴 {subsidy} 元。
               -> 实际每亩净收益：{eco_actual_income} 元。
            【机会成本对比】：
            相比种粮食，现在改种经作每亩会 {'多赚' if gap>=0 else '少赚/亏损'} {abs(gap)} 元。
            （注意：必须忍受这两年的{'微利' if eco_actual_income>0 else '亏损'}，才能等到第3年的高收益。）
            """
        else:
            economy_desc = f"""
            【当前阶段】：产业收获期（第 {current_year} 年）。
            【账本分析】：
            你的经作树已经长大，不再需要补贴。
            预计每亩净收益可达 3000 元（远超粮食的500元）。
            """

        system_prompt = f"""
        你扮演{self.profile['name']}。
        性格：{self.profile['character']}。风险偏好：{self.profile['risk']}。
        拥有土地{self.total_area}亩，其中{self.survival_area}亩必须种粮食保命。
        """

        user_prompt = f"""
        {economy_desc}
        
        【外部信息】：{self.get_social_context()}
        【你的状态】：去年种了 {self.eco_crop_area} 亩经作。
        【决策】：今年打算维持、增加还是减少经作面积？(0 ~ {max_allowed} 亩)
        
        请权衡眼下的收支落差和未来的收益，输出JSON: {{"thought": "...", "decision_area": float}}
        """

        # --- 2. LLM 决策 ---
        decision_data = self.model.llm_brain.get_decision(system_prompt, user_prompt)
        proposed_area = float(decision_data.get("decision_area", 0))
        
        # 执行决策
        final_area = max(0, min(proposed_area, max_allowed))
        self.eco_crop_area = final_area
        self.rice_area = self.total_area - final_area
        
        # --- 3. 结算当年收益 ---
        income, eco_per_mu = self.calculate_annual_income(current_year, subsidy)
        self.cash += income
        self.cumulative_income += income
        
        # 记录记忆
        self.memory.append({
            "year": current_year,
            "decision": final_area,
            "income": income,
            "thought": decision_data.get("thought")
        })

        # --- 新增：写入全局模型日志 (用于导出 Excel) ---
        self.model.journal.append({
            "Scenario_Subsidy": subsidy,      # 当前实验组补贴额
            "Year": current_year,             # 年份
            "Farmer": self.profile['name'],   # 农户姓名
            "Risk_Profile": self.profile['risk'], # 风险偏好
            "Decision_Area": final_area,      # 决策面积
            "Thought": decision_data.get("thought") # LLM 的具体想法
        })
        
        # print(f"[{self.profile['name']}] ...") # 可以注释掉print，因为我们有Excel了


class GovernmentAgent(Agent):
    """
    政府代理人：在本实验中，它不再动态调整，而是作为一个静态的政策发布者。
    保留这个类是为了保持架构完整性，方便未来扩展。
    """
    def __init__(self, unique_id, model):
        super().__init__(model)
    
    def step(self):
        pass # 政策已在 Model 初始化时定死，无需每一步调整