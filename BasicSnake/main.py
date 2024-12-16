#!/usr/bin/python
# -*- coding: utf-8 -*-
from Agent import AgentDiscretePPO
from core import ReplayBuffer
from draw import Painter
from env4Snake import Snake
import random
import pygame
import numpy as np
import torch
import matplotlib.pyplot as plt


def testAgent(test_env,agent,episode):
    ep_reward = 0
    o = test_env.reset()
    for _ in range(650):
        if episode % 100 == 0:
            test_env.render()
        for event in pygame.event.get():  # 不加这句render要卡，不清楚原因
            pass
        a_int, a_prob = agent.select_action(o)
        o2, reward, done, _ = test_env.step(a_int)
        ep_reward += reward
        if done: break
        o = o2
    return ep_reward

if __name__ == "__main__":
    env = Snake()
    test_env = Snake()
    act_dim = 4
    obs_dim = 6
    agent = AgentDiscretePPO()
    agent.init(512,obs_dim,act_dim,if_use_gae=True)
    agent.state = env.reset()
    buffer = ReplayBuffer(2**12,obs_dim,act_dim,True)
    MAX_EPISODE = 1000
    batch_size = 256
    rewardList = []
    maxReward = -np.inf
    
    # 添加平均奖励跟踪
    running_reward = 0
    running_length = 100  # 计算最近100个episode的平均值

    for episode in range(MAX_EPISODE):
        with torch.no_grad():
            trajectory_list = agent.explore_env(env,2**12,1,0.99)
        buffer.extend_buffer_from_list(trajectory_list)
        agent.update_net(buffer,batch_size,1,2**-8)
        ep_reward = testAgent(test_env, agent, episode)
        
        # 更新奖励列表
        rewardList.append(ep_reward)
        
        # 计算平均奖励
        if len(rewardList) > running_length:
            running_reward = np.mean(rewardList[-running_length:])
            print(f'Episode: {episode}, 当前奖励: {ep_reward:.2f}, 最近{running_length}回合平均奖励: {running_reward:.2f}')
            
            # 更新最高平均奖励并保存模型
            if running_reward > maxReward:
                maxReward = running_reward
                print('保存模型！')
                print(f'新的最高平均奖励！Episode {episode}, 最近{running_length}回合平均奖励: {maxReward:.2f}')
                torch.save(agent.act.state_dict(),'act_weight.pkl')
        else:
            running_reward = np.mean(rewardList)
            print(f'Episode: {episode}, 当前奖励: {ep_reward:.2f}, 目前平均奖励: {running_reward:.2f}')

    # 绘图部分
    painter = Painter(load_csv=True, load_dir='reward.csv')
    
    # 添加原始数据和平滑后的数据
    painter.addData(rewardList, 'PPO-Raw', smooth=False)  # 原始数据
    painter.addData(rewardList, 'PPO-Smoothed', smooth=True)  # 平滑后的数据
    
    # 保存数据
    painter.saveData('reward.csv')
    
    # 设置图表属性
    painter.setTitle('Snake Game Training Process')
    painter.setXlabel('Episode')
    painter.setYlabel('Reward')
    
    # 添加统计信息到标题
    max_reward = max(rewardList)
    avg_reward = np.mean(rewardList)
    final_avg = np.mean(rewardList[-100:])  # 最后100回合的平均值
    title = f'Snake Game Training (Max:{max_reward:.1f}, Avg:{avg_reward:.1f}, Final Avg:{final_avg:.1f})'
    painter.setTitle(title)
    
    # 绘制图表
    painter.drawFigure(style="whitegrid")


