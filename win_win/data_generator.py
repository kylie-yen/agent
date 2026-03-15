import pandas as pd
import numpy as np

def generate_farmer_agents(num_agents=100, seed=42):
    """
    基于人口普查数据生成农户智能体种群
    """
    np.random.seed(seed)
    
    # ==========================================
    # 1. 参数提取
    # ==========================================
    
    # [家庭户情况]：1人户到10人及以上户的分布
    # 数据来源：人口普查数据.xlsx - 家庭户情况.csv
    family_counts = [106967, 142031, 112559, 94733, 47438, 28467, 10015, 3411, 1461, 1475]
    total_families = sum(family_counts)
    family_size_probs = [c / total_families for c in family_counts]
    family_size_range = list(range(1, 11)) # 1 to 10+
    
    # [基本情况]：劳动力占比 (15-64岁)
    # 数据来源：人口普查数据.xlsx - 基本情况.csv (68.6%)
    labor_ratio = 0.686 
    
    # [收入情况]：五等分组人均可支配收入 (元)
    # 数据来源：人口普查数据.xlsx - 农村居民五等分组收入情况.csv
    # 对应：低收入, 中偏下, 中间, 中偏上, 高收入
    income_means = [5410, 13298, 19337, 27060, 53805] 
    income_labels = ['Low', 'Mid-Low', 'Mid', 'Mid-High', 'High']
    
    # [土地情况]：人均耕地面积 (亩)
    # 数据来源：人口普查数据.xlsx - 农村家庭土地经营情况.csv (取2012年数据作为参考)
    land_per_capita = 2.34
    
    # ==========================================
    # 2. 生成 Agent 循环
    # ==========================================
    agents_data = []
    
    for i in range(num_agents):
        # --- A. 家庭人口结构 ---
        # 依据概率分布抽取家庭人口
        fam_size = np.random.choice(family_size_range, p=family_size_probs)
        # 依据二项分布生成劳动力数量 (模拟真实世界的随机性)
        labor_num = np.random.binomial(n=fam_size, p=labor_ratio)
        # 修正：如果家庭有人，至少保证有1个劳动力（或者设为户主）
        labor_num = max(1, labor_num) if fam_size > 0 else 0
        
        # --- B. 经济与资产 (初始化) ---
        # 随机分配一个收入组 (假设五等分各占20%)
        inc_group_idx = np.random.randint(0, 5)
        base_income = income_means[inc_group_idx]
        
        # 生成具体人均收入：引入 25% 的标准差波动，模拟组内差异
        real_per_capita_inc = max(1000, np.random.normal(base_income, base_income * 0.25))
        
        # 初始存款 (Assets)：假设为家庭 1~3 年的可支配收入积累
        # 逻辑：人均收入 * 人口 * 随机年份系数
        initial_savings = real_per_capita_inc * fam_size * np.random.uniform(0.5, 3.0)
        
        # --- C. 土地资源 ---
        # 基础面积 = 人口 * 人均指标
        base_land = fam_size * land_per_capita
        # 引入 +/- 20% 的地块差异 (有的地多，有的地少)
        land_area = base_land * np.random.uniform(0.8, 1.2)
        
        # --- D. 风险偏好 (LLM 驱动核心) ---
        # 生成 0~1 的风险系数 (正态分布，均值0.5)
        risk_score = np.clip(np.random.normal(0.5, 0.15), 0.0, 1.0)
        
        # 映射为自然语言 Prompt
        if risk_score < 0.35:
            risk_desc = "保守型 (Conservative): 你厌恶风险，优先考虑资产安全，只有在较高确定性下才考虑投资。"
        elif risk_score < 0.65:
            risk_desc = "稳健型 (Balanced): 你会在风险和收益之间进行权衡，愿意在可控范围内尝试新技术或小额借贷。"
        else:
            risk_desc = "进取型 (Aggressive): 你追求高收益，愿意承担较大风险，倾向于扩大规模或尝试高风险高回报的作物。"
            
        # --- E. 汇总数据 ---
        agent = {
            "Agent_ID": i,
            "Family_Size": int(fam_size),
            "Labor_Count": int(labor_num),
            "Income_Level": income_labels[inc_group_idx],
            "Initial_Savings_Yuan": round(initial_savings, 0), # 初始资金
            "Land_Area_Mu": round(land_area, 2),               # 耕地面积
            "Risk_Score": round(risk_score, 3),                # 风险系数(数学计算用)
            "Risk_Profile_Prompt": risk_desc,                  # 风险描述(LLM用)
            "Debt_Yuan": 0                                     # 初始负债默认为0
        }
        agents_data.append(agent)
        
    return pd.DataFrame(agents_data)

# 运行并保存
df_agents = generate_farmer_agents(100)
print(df_agents.head())
df_agents.to_csv(".\win_win\data\Farmer_Agents_Initialized.csv", index=False)
