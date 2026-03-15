import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Simulation_Results.csv')

# Create a figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: Government Balance
axes[0].plot(df.index + 1, df['Govt_Balance'], marker='o', color='green', linewidth=2)
axes[0].set_title('Government Fiscal Balance Over Time', fontsize=14)
axes[0].set_ylabel('Balance (Yuan)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Market Price & Cash Crop Area
ax2 = axes[1]
line1 = ax2.plot(df.index + 1, df['Market_Price'], marker='s', color='blue', label='Market Price', linewidth=2)
ax2.set_ylabel('Price (Yuan/kg)', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Create a twin axis for Area
ax2_twin = ax2.twinx()
line2 = ax2_twin.plot(df.index + 1, df['Total_Cash_Crop_Area'], marker='^', color='orange', linestyle='--', label='Cash Crop Area')
ax2_twin.set_ylabel('Total Area (Mu)', fontsize=12, color='orange')
ax2_twin.tick_params(axis='y', labelcolor='orange')

# Legend for dual axis
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper center')
ax2.set_title('Market Price vs. Planting Scale', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)

# Plot 3: Average Savings
axes[2].plot(df.index + 1, df['Average_Savings'], marker='D', color='purple', linewidth=2)
axes[2].set_title('Average Farmer Savings Over Time', fontsize=14)
axes[2].set_xlabel('Year', fontsize=12)
axes[2].set_ylabel('Savings (Yuan)', fontsize=12)
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('.\win_win\data\simulation_analysis.png')
