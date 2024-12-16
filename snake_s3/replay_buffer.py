import numpy as np
from collections import deque
import random

class MultiAgentReplayBuffer:
    def __init__(self, capacity=300000, num_agents=6):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = deque(maxlen=capacity)
        
    def push(self, states, actions, rewards, next_states, done):
        """存储一个时间步的多智能体经验"""
        # 确保所有输入都是numpy数组
        states = np.asarray(states)       # [num_agents, state_dim]
        actions = np.asarray(actions)     # [num_agents, action_dim]
        rewards = np.asarray(rewards)     # [num_agents]
        next_states = np.asarray(next_states)  # [num_agents, state_dim]
        
        # 将标量done转换为每个智能体的done标志
        dones = np.full(self.num_agents, done)  # [num_agents]
        
        # 确保维度正确
        assert states.shape[0] == self.num_agents, f"states shape: {states.shape}"
        assert actions.shape[0] == self.num_agents, f"actions shape: {actions.shape}"
        assert rewards.shape[0] == self.num_agents, f"rewards shape: {rewards.shape}"
        assert next_states.shape[0] == self.num_agents, f"next_states shape: {next_states.shape}"
        
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size):
        """采样一个批次的多智能体经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为numpy数组并调整维度
        # [batch_size, num_agents, dim]
        states = np.stack(states)         # [batch_size, num_agents, state_dim]
        actions = np.stack(actions)       # [batch_size, num_agents, action_dim]
        rewards = np.stack(rewards)       # [batch_size, num_agents]
        next_states = np.stack(next_states)  # [batch_size, num_agents, state_dim]
        dones = np.stack(dones)           # [batch_size, num_agents]
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)