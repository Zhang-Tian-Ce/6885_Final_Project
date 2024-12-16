import torch
import pygame
import time
from slither_env import SlitherEnv
from sac import SAC

def visualize_episode(model_path, delay=0.01):
    """
    渲染一个完整的episode
    delay: 每步之间的延迟时间（秒），用于放慢观察
    """
    # 初始化环境和智能体
    env = SlitherEnv()
    agent = SAC(state_dim=6, action_dim=1)
    
    # 加载模型
    checkpoint = torch.load(model_path)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    print(f"Loaded model from episode {checkpoint['episode']} with average score {checkpoint['avg_score']:.1f}")
    
    # 运行一个episode
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    done = False
    
    while not done:
        # 选择动作
        with torch.no_grad():
            action = agent.select_action(state)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 渲染当前帧
        env.render()
        
        # 更新状态
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        # 打印当前步信息
        print(f"\rStep {episode_steps}: Score = {info['length']}, Reward = {reward:.2f}", end="")
        
        # 延迟
        time.sleep(delay)
        
    print(f"\nEpisode finished:")
    print(f"Final Score: {info['length']}")
    print(f"Total Steps: {episode_steps}")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Efficiency: {info['length']/episode_steps:.4f}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    model_path = '/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s1/models/sac_model_avg68.2_e2955.pth'
    visualize_episode(model_path)