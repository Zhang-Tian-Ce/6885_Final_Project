import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取两个数据集
df_pretrain = pd.read_excel('/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/Pretrain_multi_agent/pretrain_multi_agent.xlsx')
df_new = pd.read_excel('/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/new_multi_agent/multi_agent.xlsx')

# 限制Episode最大值为2000
df_pretrain = df_pretrain[df_pretrain['Episode'] <= 2000]
df_new = df_new[df_new['Episode'] <= 2000]

# 计算滑动窗口平均（window_size=10）
window_size = 10
df_pretrain['Reward_MA'] = df_pretrain['Max_Reward'].rolling(window=window_size).mean()
df_new['Reward_MA'] = df_new['Max_Reward'].rolling(window=window_size).mean()
df_pretrain['Score_MA'] = df_pretrain['Best_Score'].rolling(window=window_size).mean()
df_new['Score_MA'] = df_new['Best_Score'].rolling(window=window_size).mean()

# 计算统计数据
stats = {
    'Pretrained Model': {
        'Reward': {
            'Mean': df_pretrain['Max_Reward'].mean(),
            'Std': df_pretrain['Max_Reward'].std(),
            'Max': df_pretrain['Max_Reward'].max(),
            'Min': df_pretrain['Max_Reward'].min(),
            'Median': df_pretrain['Max_Reward'].median()
        },
        'Score': {
            'Mean': df_pretrain['Best_Score'].mean(),
            'Std': df_pretrain['Best_Score'].std(),
            'Max': df_pretrain['Best_Score'].max(),
            'Min': df_pretrain['Best_Score'].min(),
            'Median': df_pretrain['Best_Score'].median()
        }
    },
    'New Model': {
        'Reward': {
            'Mean': df_new['Max_Reward'].mean(),
            'Std': df_new['Max_Reward'].std(),
            'Max': df_new['Max_Reward'].max(),
            'Min': df_new['Max_Reward'].min(),
            'Median': df_new['Max_Reward'].median()
        },
        'Score': {
            'Mean': df_new['Best_Score'].mean(),
            'Std': df_new['Best_Score'].std(),
            'Max': df_new['Best_Score'].max(),
            'Min': df_new['Best_Score'].min(),
            'Median': df_new['Best_Score'].median()
        }
    }
}

# 打印统计结果
print("\n=== Statistical Analysis ===")
for model in ['Pretrained Model', 'New Model']:
    print(f"\n{model}:")
    print("\nReward Statistics:")
    for stat, value in stats[model]['Reward'].items():
        print(f"{stat}: {value:.2f}")
    print("\nScore Statistics:")
    for stat, value in stats[model]['Score'].items():
        print(f"{stat}: {value:.2f}")

# 计算收敛后的表现（最后500个episode）
print("\n=== Performance After Convergence (Last 500 Episodes) ===")
for model, df in [('Pretrained Model', df_pretrain), ('New Model', df_new)]:
    last_500 = df.tail(500)
    print(f"\n{model}:")
    print(f"Average Reward: {last_500['Max_Reward'].mean():.2f} ± {last_500['Max_Reward'].std():.2f}")
    print(f"Average Score: {last_500['Best_Score'].mean():.2f} ± {last_500['Best_Score'].std():.2f}")

# 设置图表样式
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# 创建图1：Reward对比
plt.figure()
plt.plot(df_pretrain['Episode'], df_pretrain['Reward_MA'], 
         label='Pretrained Model', color='#2ecc71', linewidth=1.5)
plt.plot(df_new['Episode'], df_new['Reward_MA'], 
         label='New Model', color='#e74c3c', linewidth=1.5)
plt.title('Training Reward over Episodes (Moving Average, Window=10)', fontsize=14, pad=15)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('training_reward_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 创建图2：Score对比
plt.figure()
plt.plot(df_pretrain['Episode'], df_pretrain['Score_MA'], 
         label='Pretrained Model', color='#2ecc71', linewidth=1.5)
plt.plot(df_new['Episode'], df_new['Score_MA'], 
         label='New Model', color='#e74c3c', linewidth=1.5)
plt.title('Training Score over Episodes (Moving Average, Window=10)', fontsize=14, pad=15)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('training_score_comparison.png', dpi=300, bbox_inches='tight')
plt.close()