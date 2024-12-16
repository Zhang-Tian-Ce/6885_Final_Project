#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curves():
    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv('reward.csv')
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制两条曲线
    raw_data = df[df['Method'] == 'PPO-Raw']
    smoothed_data = df[df['Method'] == 'PPO-Smoothed']
    
    plt.plot(raw_data['episode'], raw_data['episode reward'], 
             alpha=0.3, label='PPO-Raw', color='blue')
    plt.plot(smoothed_data['episode'], smoothed_data['episode reward'], 
             linewidth=2, label='PPO-Smoothed', color='red')
    
    # 计算统计信息
    raw_mean = raw_data['episode reward'].mean()
    raw_max = raw_data['episode reward'].max()
    raw_min = raw_data['episode reward'].min()
    
    # 设置图表属性
    plt.title(f'Snake Game Training Process\nMean: {raw_mean:.1f}, Max: {raw_max:.1f}, Min: {raw_min:.1f}', 
              fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 training_curve.png")
    
    # 打印详细统计信息
    print("\n========= 训练统计信息 =========")
    print(f"平均奖励: {raw_mean:.2f}")
    print(f"最高奖励: {raw_max:.2f}")
    print(f"最低奖励: {raw_min:.2f}")

if __name__ == "__main__":
    plot_training_curves()