import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置学术风格绘图
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("deep")
plt.rcParams['font.family'] = 'Times New Roman' # 论文常用字体
plt.rcParams['axes.unicode_minus'] = False      # 解决负号显示问题

def plot_simulation_results(file_path=".\win_win_v2\data\Simulation_Results_v2.csv"):
    # 1. 读取数据
    df = pd.read_csv(file_path)
    years = df['Year']

    # 创建一个 2x1 的画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # ==========================================
    # 子图 1: 市场供需与价格机制 (Market Dynamics)
    # ==========================================
    color_area = 'tab:green'
    color_price = 'tab:red'
    
    # 左轴：种植面积
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Planting Area (Mu)', color=color_area, fontsize=12, fontweight='bold')
    
    # 绘制总面积（浅色）和成熟面积（深色）堆叠效果
    ax1.fill_between(years, df['Total_Cash_Area'], color=color_area, alpha=0.3, label='Immature Area')
    ax1.fill_between(years, df['Mature_Area'], color=color_area, alpha=0.8, label='Mature Area (Supply)')
    ax1.tick_params(axis='y', labelcolor=color_area)
    
    # 右轴：市场价格
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Market Price (Yuan/kg)', color=color_price, fontsize=12, fontweight='bold')
    line_price = ax1_twin.plot(years, df['Market_Price'], color=color_price, linewidth=2.5, marker='o', label='Market Price')
    ax1_twin.tick_params(axis='y', labelcolor=color_price)
    
    # 添加基准价格线
    ax1_twin.axhline(y=10.0, color='gray', linestyle='--', alpha=0.5, label='Base Price')
    
    # 图例合并
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    ax1.set_title('(a) Market Supply & Price Dynamics', loc='left', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ==========================================
    # 子图 2: 政府财政与农户负债 (Fiscal & Debt)
    # ==========================================
    color_govt = 'tab:blue'
    color_debt = 'tab:orange'
    
    # 左轴：政府财政余额
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Govt Balance (Million Yuan)', color=color_govt, fontsize=12, fontweight='bold')
    ax2.plot(years, df['Govt_Balance'] / 1000000, color=color_govt, linewidth=2.5, marker='s', label='Govt Balance')
    ax2.tick_params(axis='y', labelcolor=color_govt)
    
    # 标记“回血点”
    min_balance = df['Govt_Balance'].min() / 1000000
    min_year = df.loc[df['Govt_Balance'].idxmin(), 'Year']
    ax2.annotate(f'Lowest Point: {min_balance:.2f}M', 
                 xy=(min_year, min_balance), 
                 xytext=(min_year, min_balance - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # 右轴：农户平均负债
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Avg Farmer Debt (Yuan)', color=color_debt, fontsize=12, fontweight='bold')
    ax2_twin.bar(years, df['Avg_Debt'], color=color_debt, alpha=0.6, width=0.5, label='Avg Debt')
    ax2_twin.tick_params(axis='y', labelcolor=color_debt)
    
    # 设置刻度
    ax2.set_xticks(years)
    ax2.set_title('(b) Government Fiscal Balance & Farmer Risk', loc='left', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('.\win_win_v2\data\Simulation_Analysis_Plot.png', dpi=300)
    print("图表已保存为 Simulation_Analysis_Plot.png")
    plt.show()

if __name__ == "__main__":
    plot_simulation_results()
