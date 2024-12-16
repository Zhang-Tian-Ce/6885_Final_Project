import numpy as np
import torch
import time
from multi_agent_slither_env import MultiAgentSlitherEnv
from ctde_sac import CTDESAC
import pygame



# 渲染游戏
def render_game():
    pretrained_model_path = "/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/results/best_pretrained_model.pt"

    # 环境和模型参数
    num_agents = 6
    hidden_dim = 256
    buffer_size = 300000

    # 初始化环境
    env = MultiAgentSlitherEnv(num_agents=num_agents)
    state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    # 初始化模型
    agent = CTDESAC(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        buffer_size=buffer_size
    )

    # 加载预训练模型
    load_pretrained_model(agent, pretrained_model_path, state_dim)

    # 开始游戏
    print("开始渲染游戏...")
    states = env.reset()
    done = False

    while not done:
        actions = agent.select_actions(states, evaluate=True)
        states, _, done, _ = env.step(actions)
        env.render()
        pygame.time.delay(50)  # 控制渲染速度

    env.close()
    print("游戏结束！")

if __name__ == "__main__":
    render_game()