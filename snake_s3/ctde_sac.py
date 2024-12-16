import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from replay_buffer import MultiAgentReplayBuffer

class CTDESAC:
    """
    Centralized Training with Decentralized Execution SAC
    中心化训练、分散执行的SAC算法
    """
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=256,
                 gamma=0.99, tau=0.005, alpha=0.2, buffer_size=300000):
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 每个智能体的独立策略网络（用于分散执行）
        self.policies = [
            PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(num_agents)
        ]
        
        # 中心化的Q网络（用于训练）
        # 输入包括所有智能体的状态和动作
        self.critic = QNetwork(
            state_dim * num_agents,
            action_dim * num_agents,
            hidden_dim
        ).to(self.device)
        
        self.critic_target = QNetwork(
            state_dim * num_agents,
            action_dim * num_agents,
            hidden_dim
        ).to(self.device)
        
        # 复制参数到目标网络
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(param.data)
        
        # 优化器设置
        self.policy_optimizers = [
            optim.Adam(policy.parameters(), lr=3e-4)
            for policy in self.policies
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 温度参数（可以是每个智能体独立的）
        self.log_alphas = [
            torch.zeros(1, requires_grad=True, device=self.device)
            for _ in range(num_agents)
        ]
        self.alpha_optimizers = [
            optim.Adam([log_alpha], lr=3e-4)
            for log_alpha in self.log_alphas
        ]
        
        # 目标熵（可以根据动作空间调整）
        self.target_entropy = -action_dim
        
        # 其他超参数
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # 经验回放缓冲区
        self.memory = MultiAgentReplayBuffer(buffer_size, num_agents)
        
    def select_actions(self, states, evaluate=False):
        """分散执行：每个智能体独立选择动作"""
        states = torch.FloatTensor(states).to(self.device)
        actions = []
        
        for i, policy in enumerate(self.policies):
            if evaluate:
                # 评估时使用均值
                action = policy.get_action(states[i])
            else:
                # 训练时使用采样
                action, _ = policy.sample(states[i].unsqueeze(0))
                action = action.squeeze(0)
            actions.append(action.detach().cpu().numpy())
            
        return np.array(actions)
    
    def train_step(self, batch_size):
        """执行一步训练"""
        if len(self.memory) < batch_size:
            return
            
        # 采样经验
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # 转换为tensor，并确保维度正确
        # [batch_size, num_agents, dim]
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 重塑为联合状态和动作
        joint_states = states.reshape(batch_size, -1)
        joint_actions = actions.reshape(batch_size, -1)
        joint_next_states = next_states.reshape(batch_size, -1)
        
        # 更新中心化Critic
        with torch.no_grad():
            next_actions = []
            next_log_probs = []
            
            for i in range(self.num_agents):
                next_action, next_log_prob = self.policies[i].sample(next_states[:, i])
                next_actions.append(next_action)
                next_log_probs.append(next_log_prob)
            
            next_actions = torch.cat(next_actions, dim=1)
            next_log_probs = torch.cat(next_log_probs, dim=1)
            
            q1_next, q2_next = self.critic_target(joint_next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            # 计算每个智能体的温度加权日志概率
            weighted_log_probs = torch.zeros_like(rewards)
            for i in range(self.num_agents):
                weighted_log_probs[:, i] = torch.exp(self.log_alphas[i].detach()) * next_log_probs[:, i]
            
            # 使用mean代替sum来计算整体奖励和done状态
            q_target = rewards.mean(dim=1, keepdim=True) + \
                    (1 - dones.mean(dim=1, keepdim=True)) * \
                    self.gamma * (q_next - weighted_log_probs.mean(dim=1, keepdim=True))
        
        # 当前Q值和critic损失
        q1, q2 = self.critic(joint_states, joint_actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新每个智能体的Policy
        for i in range(self.num_agents):
            new_actions = []
            log_probs = []
            
            for j in range(self.num_agents):
                if i == j:
                    action, log_prob = self.policies[i].sample(states[:, i])
                else:
                    with torch.no_grad():
                        action, log_prob = self.policies[j].sample(states[:, j])
                new_actions.append(action)
                log_probs.append(log_prob)
            
            joint_new_actions = torch.cat(new_actions, dim=1)
            q1_new, q2_new = self.critic(joint_states, joint_new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            alpha = torch.exp(self.log_alphas[i].detach())
            policy_loss = (alpha * log_probs[i] - q_new).mean()
            
            self.policy_optimizers[i].zero_grad()
            policy_loss.backward()
            self.policy_optimizers[i].step()
            
            # 更新温度参数
            alpha_loss = -(self.log_alphas[i] * (log_probs[i].detach() + self.target_entropy)).mean()
            self.alpha_optimizers[i].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizers[i].step()
        
        # 软更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
    def save_models(self, path):
        """保存模型"""
        torch.save({
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'policies': [policy.state_dict() for policy in self.policies],
            'log_alphas': [log_alpha.item() for log_alpha in self.log_alphas]
        }, path)
        
    def load_models(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        for i, policy in enumerate(self.policies):
            policy.load_state_dict(checkpoint['policies'][i])
        for i, alpha in enumerate(checkpoint['log_alphas']):
            self.log_alphas[i].data.fill_(alpha)

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
        
    def sample(self, state):
        """使用重参数化采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
        
    def get_action(self, state):
        """获取确定性动作（用于评估）"""
        mean, _ = self.forward(state)
        return torch.tanh(mean)

class QNetwork(nn.Module):
    """Q网络"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        
        # Q1
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1_net(x), self.q2_net(x)