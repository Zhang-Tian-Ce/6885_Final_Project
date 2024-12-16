# train_newsnake.py

import os
import numpy as np
import torch
import pandas as pd
from collections import deque
from datetime import datetime
from slither_env import SlitherEnv
from sac import SAC

def save_data_to_excel(new_data):
    """将新数据追加到Excel文件中"""
    excel_path = 'results/newsnake.xlsx'
    
    try:
        # 如果文件存在，读取现有数据
        existing_df = pd.read_excel(excel_path)
        # 创建新数据的DataFrame
        new_df = pd.DataFrame([new_data])
        # 合并数据
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    except FileNotFoundError:
        # 如果文件不存在，创建新的DataFrame
        updated_df = pd.DataFrame([new_data])
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    # 保存更新后的数据
    updated_df.to_excel(excel_path, index=False)

def train():
    # 创建保存目录
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化环境和智能体
    env = SlitherEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC(state_dim, action_dim)
    
    # 训练参数
    max_episodes = 10000
    max_steps = 2000
    batch_size = 256
    
    # 记录数据
    score_window = deque(maxlen=100)  # 使用100轮的滑动窗口
    best_avg_score = 0
    
    # 开始训练
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < max_steps:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)
            
            # 如果有足够的样本，进行学习
            if len(agent.memory) > batch_size:
                agent.update(batch_size)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # 更新分数窗口
        current_score = info['length']
        score_window.append(current_score)
        current_avg_score = np.mean(score_window) if score_window else 0
        
        # 记录并保存本轮数据
        episode_data = {
            'Episode': episode + 1,
            'Reward': float(episode_reward),
            'Score': current_score,
            'Avg_Score': current_avg_score,
            'Kills': info['kills'],
            'Steps': steps
        }
        save_data_to_excel(episode_data)
        
        # 如果平均分数更高，保存模型
        if current_avg_score > best_avg_score and len(score_window) == score_window.maxlen:
            best_avg_score = current_avg_score
            torch.save({
                'episode': episode + 1,
                'policy_state_dict': agent.policy.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'average_score': best_avg_score,
                'window_size': score_window.maxlen
            }, os.path.join(save_dir, 'newsanke_best_model.pth'))
        
        # 打印训练进度
        print(f"Episode: {episode+1}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Score: {current_score}")
        print(f"Average Score: {current_avg_score:.2f}")
        print(f"Best Average Score: {best_avg_score:.2f}")
        print(f"Kills: {info['kills']}")
        print(f"Steps: {steps}")
        print("------------------------")

if __name__ == '__main__':
    train()