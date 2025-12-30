import pandas as pd
import matplotlib.pyplot as plt
import os
from model import VillageModel

# --- 1. 设置输出路径 ---
# 获取当前脚本所在目录 (farmer文件夹)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 拼接 simulation 文件夹路径
OUTPUT_DIR = os.path.join(BASE_DIR, "simulation")

# 确保文件夹存在，不存在则创建
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📂 输出目录已锁定: {OUTPUT_DIR}")

def run_single_scenario(scenario_name, subsidy_val):
    print(f"\n🚀 开始运行实验: {scenario_name} (补贴 {subsidy_val} 元/亩)")
    
    model = VillageModel(subsidy_amount=subsidy_val)
    
    for year in range(10):
        model.step()
        print(f"   Year {year+1} 完成...", end="\r")
    
    # 获取定量数据 (DataFrame)
    df = model.datacollector.get_model_vars_dataframe()
    df['Scenario'] = scenario_name
    df['Year'] = range(1, 11)
    
    # 获取定性数据 (日志列表)
    logs = model.journal
    # 给日志加上场景标签
    for log in logs:
        log['Scenario'] = scenario_name
    
    final_area = df.iloc[-1]['TotalEcoArea']
    final_wealth = df.iloc[-1]['AvgWealth']
    print(f"\n✅ 结束。第10年总种植面积: {final_area:.1f} 亩, 户均累计财富: {final_wealth:.0f} 元")
    
    return df, logs

if __name__ == "__main__":
    scenarios = [
        ("Group A (Low)", 400),
        ("Group B (Mid)", 800),
        ("Group C (High)", 1300)
    ]
    
    all_results_df = [] # 存画图数据
    all_thoughts_log = [] # 存Excel想法数据
    
    # 批量运行
    for name, subsidy in scenarios:
        df, logs = run_single_scenario(name, subsidy)
        all_results_df.append(df)
        all_thoughts_log.extend(logs) # 合并日志
        
    # --- 2. 导出 Excel (农户想法) ---
    thoughts_df = pd.DataFrame(all_thoughts_log)
    
    # 调整列顺序，好看一点
    cols = ['Scenario', 'Year', 'Farmer', 'Risk_Profile', 'Decision_Area', 'Thought']
    thoughts_df = thoughts_df[cols]
    
    excel_path = os.path.join(OUTPUT_DIR, "farmer_thoughts.xlsx")
    thoughts_df.to_excel(excel_path, index=False)
    print(f"\n📖 农户思考日志已保存: {excel_path}")
        
    # --- 3. 导出图片 ---
    all_data = pd.concat(all_results_df)
    
    # 图1: 种植规模
    plt.figure(figsize=(10, 5))
    for name in all_data['Scenario'].unique():
        subset = all_data[all_data['Scenario'] == name]
        plt.plot(subset['Year'], subset['TotalEcoArea'], marker='o', label=name)
    
    plt.title("Impact of Subsidy on Economic Crop Scale (10 Years)")
    plt.xlabel("Year")
    plt.ylabel("Total Planted Area (Mu)")
    plt.grid(True)
    plt.legend()
    save_path_scale = os.path.join(OUTPUT_DIR, "result_scale_comparison.png")
    plt.savefig(save_path_scale)
    print(f"📊 规模对比图已保存: {save_path_scale}")
    
    # 图2: 累计收益
    plt.figure(figsize=(10, 5))
    for name in all_data['Scenario'].unique():
        subset = all_data[all_data['Scenario'] == name]
        plt.plot(subset['Year'], subset['AvgWealth'], marker='s', linestyle='--', label=name)
        
    plt.title("Impact of Subsidy on Farmer's Accumulated Wealth")
    plt.xlabel("Year")
    plt.ylabel("Avg Accumulated Income (CNY)")
    plt.grid(True)
    plt.legend()
    save_path_wealth = os.path.join(OUTPUT_DIR, "result_wealth_comparison.png")
    plt.savefig(save_path_wealth)
    print(f"📊 财富对比图已保存: {save_path_wealth}")