import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 0. 全局绘图风格设置 (关键：中文字体)
# ==========================================
# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("deep")

# 【核心修改】设置中文字体
# Windows系统通常使用 'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑)
# 如果论文严格要求宋体，可以改为 'SimSun'
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['font.family'] = 'sans-serif'

# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False      

def plot_simulation_results(file_path=r".\win_win_v2\data\Simulation_Results_v2.csv"):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    # 1. 读取数据
    df = pd.read_csv(file_path)
    years = df['Year']

    # 创建一个 2x1 的画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # ==========================================
    # 子图 1: 市场供需与价格动态 (中文版)
    # ==========================================
    color_area = 'tab:green'
    color_price = 'tab:red'
    
    # 左轴：种植面积
    ax1.set_xlabel('年份 (Year)') # 底部子图共享x轴，这里可以不显示，但留着也没事
    ax1.set_ylabel('种植面积 (亩)', color=color_area, fontsize=12, fontweight='bold')
    
    # 绘制堆叠面积图
    ax1.fill_between(years, df['Total_Cash_Area'], color=color_area, alpha=0.3, label='投入期面积 (未成熟)')
    ax1.fill_between(years, df['Mature_Area'], color=color_area, alpha=0.8, label='供应量 (成熟面积)')
    ax1.tick_params(axis='y', labelcolor=color_area)
    
    # 右轴：市场价格
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('市场收购价 (元/公斤)', color=color_price, fontsize=12, fontweight='bold')
    # 绘制价格曲线
    ax1_twin.plot(years, df['Market_Price'], color=color_price, linewidth=2.5, marker='o', label='市场价格')
    ax1_twin.tick_params(axis='y', labelcolor=color_price)
    
    # 添加基准价格辅助线
    ax1_twin.axhline(y=10.0, color='gray', linestyle='--', alpha=0.5, label='基准价格 (10元)')
    
    # 图例合并 (处理双轴图例)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', prop={'size': 10})
    
    ax1.set_title('(a) 经济作物供需结构与价格演变', loc='left', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ==========================================
    # 子图 2: 政府财政与农户风险 (中文版)
    # ==========================================
    color_govt = 'tab:blue'
    color_debt = 'tab:orange'
    
    # 左轴：政府财政余额
    ax2.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax2.set_ylabel('政府财政余额 (百万元)', color=color_govt, fontsize=12, fontweight='bold')
    # 绘制财政曲线 (除以100万换算单位)
    ax2.plot(years, df['Govt_Balance'] / 1000000, color=color_govt, linewidth=2.5, marker='s', label='财政余额')
    ax2.tick_params(axis='y', labelcolor=color_govt)
    
    # 标记“财政最低点” (关键事件标注)
    min_balance = df['Govt_Balance'].min() / 1000000
    min_year = df.loc[df['Govt_Balance'].idxmin(), 'Year']
    
    # 动态调整标注位置，防止遮挡
    offset_y = -0.05 if min_balance > 0.5 else 0.05
    
    ax2.annotate(f'财政低点: {min_balance:.2f}百万', 
                 xy=(min_year, min_balance), 
                 xytext=(min_year, min_balance + offset_y + 0.1), #稍微抬高一点
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 fontsize=10, color='black')

    # 右轴：农户平均负债
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('农户平均负债 (元)', color=color_debt, fontsize=12, fontweight='bold')
    # 绘制柱状图
    ax2_twin.bar(years, df['Avg_Debt'], color=color_debt, alpha=0.6, width=0.5, label='户均负债')
    ax2_twin.tick_params(axis='y', labelcolor=color_debt)
    
    # 设置刻度
    ax2.set_xticks(years)
    ax2.set_title('(b) 政府财政收支与农户信贷风险', loc='left', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 合并下方图例
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', prop={'size': 10}, ncol=2)

    # 保存图片
    plt.tight_layout()
    save_path = r".\win_win_v2\data\Simulation_Analysis_Plot.png"
    
    # 自动创建目录防止报错
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已成功保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_simulation_results()
    