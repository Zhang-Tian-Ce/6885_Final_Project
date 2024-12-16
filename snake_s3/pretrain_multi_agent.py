import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

from multi_agent_slither_env import MultiAgentSlitherEnv
from ctde_sac import CTDESAC

def load_pretrained_model(agent, pretrained_model_path, current_input_dim):
    """加载预训练模型，并动态调整策略和价值网络的所有层"""
    checkpoint = torch.load(pretrained_model_path)

    # 加载策略网络
    print("加载策略网络的预训练状态...")
    if 'policy_state_dict' in checkpoint:
        pretrained_policy_state = checkpoint['policy_state_dict']
    else:
        # 如果没有policy_state_dict，尝试直接使用checkpoint
        pretrained_policy_state = checkpoint

    policy_network = agent.policies[0]  # 假设第一个策略网络是参考

    # 获取当前网络的参数名称
    current_policy_state = policy_network.state_dict()

    # 新的参数映射关系
    policy_layer_map = {
        # 如果预训练模型使用net开头的命名
        'net.0.weight': 'net.0.weight',
        'net.0.bias': 'net.0.bias',
        'net.2.weight': 'net.2.weight',
        'net.2.bias': 'net.2.bias',
        'mean.weight': 'mean.weight',
        'mean.bias': 'mean.bias',
        'log_std.weight': 'log_std.weight',
        'log_std.bias': 'log_std.bias',
        # 如果预训练模型使用linear开头的命名
        'linear1.weight': 'net.0.weight',
        'linear1.bias': 'net.0.bias',
        'linear2.weight': 'net.2.weight',
        'linear2.bias': 'net.2.bias',
        'mean_linear.weight': 'mean.weight',
        'mean_linear.bias': 'mean.bias',
        'log_std_linear.weight': 'log_std.weight',
        'log_std_linear.bias': 'log_std.bias'
    }

    adjusted_policy_state = {}
    for pretrained_key, value in pretrained_policy_state.items():
        if pretrained_key in policy_layer_map:
            current_key = policy_layer_map[pretrained_key]
            if current_key in current_policy_state:
                target_shape = current_policy_state[current_key].shape
                if value.shape != target_shape:
                    print(f"调整策略网络层: {current_key} 从 {value.shape} -> {target_shape}")
                    # 创建新的张量并部分填充
                    adjusted_value = torch.zeros(target_shape)
                    slices = tuple(slice(0, min(s, t)) for s, t in zip(value.shape, target_shape))
                    adjusted_value[slices] = value[slices]
                    adjusted_policy_state[current_key] = adjusted_value
                else:
                    adjusted_policy_state[current_key] = value
                print(f"成功映射参数: {pretrained_key} -> {current_key}")
            else:
                print(f"当前模型中未找到对应参数: {current_key}")
        else:
            print(f"未找到参数映射: {pretrained_key}")

    # 更新所有策略网络
    missing_keys = set(current_policy_state.keys()) - set(adjusted_policy_state.keys())
    if missing_keys:
        print("警告：以下参数未能从预训练模型加载，将使用随机初始化：")
        for key in missing_keys:
            print(f"- {key}")

    for policy in agent.policies:
        policy.load_state_dict(adjusted_policy_state, strict=False)
    print("策略网络加载完成！")

    # 加载价值网络
    print("\n加载价值网络的预训练状态...")
    if 'critic_state_dict' in checkpoint:
        pretrained_critic_state = checkpoint['critic_state_dict']
    else:
        # 如果没有critic_state_dict，尝试其他可能的键名
        critic_keys = [k for k in checkpoint.keys() if 'critic' in k.lower()]
        if critic_keys:
            pretrained_critic_state = checkpoint[critic_keys[0]]
        else:
            print("警告：未找到价值网络参数，将使用随机初始化")
            return

    critic_network = agent.critic

    # 价值网络的参数映射
    critic_layer_map = {
        # q1网络
        'linear1.weight': 'q1_net.0.weight',
        'linear1.bias': 'q1_net.0.bias',
        'linear2.weight': 'q1_net.2.weight',
        'linear2.bias': 'q1_net.2.bias',
        'linear3.weight': 'q1_net.4.weight',
        'linear3.bias': 'q1_net.4.bias',
        # q2网络
        'linear4.weight': 'q2_net.0.weight',
        'linear4.bias': 'q2_net.0.bias',
        'linear5.weight': 'q2_net.2.weight',
        'linear5.bias': 'q2_net.2.bias',
        'linear6.weight': 'q2_net.4.weight',
        'linear6.bias': 'q2_net.4.bias'
    }

    adjusted_critic_state = {}
    for pretrained_key, value in pretrained_critic_state.items():
        if pretrained_key in critic_layer_map:
            current_key = critic_layer_map[pretrained_key]
            if current_key in critic_network.state_dict():
                target_shape = critic_network.state_dict()[current_key].shape
                if value.shape != target_shape:
                    print(f"调整价值网络层: {current_key} 从 {value.shape} -> {target_shape}")
                    adjusted_value = torch.zeros(target_shape)
                    slices = tuple(slice(0, min(s, t)) for s, t in zip(value.shape, target_shape))
                    adjusted_value[slices] = value[slices]
                    adjusted_critic_state[current_key] = adjusted_value
                else:
                    adjusted_critic_state[current_key] = value
                print(f"成功映射参数: {pretrained_key} -> {current_key}")
            else:
                print(f"当前模型中未找到对应参数: {current_key}")
        else:
            print(f"未找到参数映射: {pretrained_key}")

    missing_critic_keys = set(critic_network.state_dict().keys()) - set(adjusted_critic_state.keys())
    if missing_critic_keys:
        print("警告：以下价值网络参数未能从预训练模型加载，将使用随机初始化：")
        for key in missing_critic_keys:
            print(f"- {key}")

    critic_network.load_state_dict(adjusted_critic_state, strict=False)
    print("价值网络加载完成！")
    print("预训练模型加载并调整完成！")

class PretrainedMultiAgentTrainer:
    def __init__(self, 
                 pretrained_model_path,
                 num_agents=6,
                 max_episodes=5000,
                 max_steps=2000,
                 batch_size=256,
                 evaluate_freq=100,
                 save_freq=500,
                 hidden_dim=256,
                 buffer_size=300000):
        
        # 初始化环境
        self.env = MultiAgentSlitherEnv(num_agents=num_agents)
        self.eval_env = MultiAgentSlitherEnv(num_agents=num_agents)
        
        # 获取环境参数
        self.num_agents = num_agents
        self.state_dim = self.env.observation_space.shape[1]  # 31
        self.action_dim = self.env.action_space.shape[1]  # 1
        
        # 初始化CTDE-SAC算法
        self.agent = CTDESAC(
            num_agents=num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            buffer_size=buffer_size
        )
        
        # 加载预训练模型
        print(f"加载预训练模型: {pretrained_model_path}")
        load_pretrained_model(self.agent, pretrained_model_path, self.state_dim)
        
        # 训练参数
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.evaluate_freq = evaluate_freq
        self.save_freq = save_freq
        
        # 保存路径
        self.save_dir = "/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/results"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Excel文件路径
        self.excel_path = os.path.join(self.save_dir, "pretrain_multi_agent.xlsx")
        if os.path.exists(self.excel_path):
            self.df = pd.read_excel(self.excel_path)
        else:
            self.df = pd.DataFrame(columns=[
                'Episode',
                'Average_Reward',
                'Max_Reward',
                'Min_Reward',
                'Survival_Rate',
                'Food_Collection_Rate',
                'Average_Length',
                'Best_Score',
                'Episode_Length',
                'Alive_Agents',
                'Training_Time',
                'Eval_Reward'
            ])
        
        # 模型性能追踪
        self.best_avg_score = -float('inf')  # 保存最高平均 `Best Score` 的模型路径
        self.best_model_path = os.path.join(self.save_dir, "best_pretrained_model.pt")
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.survival_rates = []
        self.food_collection_rates = []
        self.recent_best_scores = deque(maxlen=100)

    def update_excel(self, episode_data):
        """更新Excel文件中的训练数据"""
        new_row = pd.DataFrame({
            'Episode': [episode_data['Episode']],
            'Average_Reward': [episode_data['Average_Reward']],
            'Max_Reward': [episode_data['Max_Reward']],
            'Min_Reward': [episode_data['Min_Reward']],
            'Survival_Rate': [episode_data['Survival_Rate']],
            'Food_Collection_Rate': [episode_data['Food_Collection_Rate']],
            'Average_Length': [episode_data['Average_Length']],
            'Best_Score': [episode_data['Best_Score']],
            'Episode_Length': [episode_data['Episode_Length']],
            'Alive_Agents': [episode_data['Alive_Agents']],
            'Training_Time': [episode_data['Training_Time']],
            'Eval_Reward': [episode_data.get('Eval_Reward', np.nan)]
        })
        
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_excel(self.excel_path, index=False)
    
    def train(self):
        """执行完整的训练循环"""
        print("开始训练...")
        training_start_time = time.time()
        
        # 保存预训练模型的状态
        pretrained_policy_states = [policy.state_dict() for policy in self.agent.policies]
        pretrained_critic_state = self.agent.critic.state_dict()
        
        # 保存最佳状态
        best_policy_states = pretrained_policy_states
        best_critic_state = pretrained_critic_state
        best_total_reward = float('-inf')
        
        # 用于追踪性能
        performance_window = deque(maxlen=5)
        restore_cooldown = 0
        
        for episode in range(1, self.max_episodes + 1):
            episode_start_time = time.time()
            states = self.env.reset()
            episode_reward = np.zeros(self.num_agents)
            
            # 记录每条蛇的得分
            snake_max_scores = [0] * self.num_agents
            snake_max_lengths = [0] * self.num_agents
                
            for step in range(self.max_steps):
                # 使用较低的探索率
                actions = self.agent.select_actions(states, evaluate=True)  # 减少探索
                next_states, rewards, done, info = self.env.step(actions)
                
                self.agent.memory.push(states, actions, rewards, next_states, done)
                episode_reward += rewards
                
                # 更新每条蛇的得分和最大长度
                for i, snake in enumerate(self.env.game.snakes):
                    snake_max_scores[i] = max(snake_max_scores[i], snake.food_count)
                    snake_max_lengths[i] = max(snake_max_lengths[i], len(snake.body))
                
                states = next_states
                
                if done:
                    break
            
            # 计算统计数据
            current_mean_reward = episode_reward.mean()
            max_reward = episode_reward.max()
            min_reward = episode_reward.min()
            best_score = max(snake_max_scores)
            
            # 更新最佳状态
            if current_mean_reward > best_total_reward:
                best_total_reward = current_mean_reward
                best_policy_states = [policy.state_dict() for policy in self.agent.policies]
                best_critic_state = self.agent.critic.state_dict()
                print(f"更新最佳性能状态，新的最佳奖励：{best_total_reward:.2f}")
            
            # 记录统计数据
            self.episode_rewards.append(current_mean_reward)
            self.episode_lengths.append(step + 1)
            self.recent_best_scores.append(best_score)
            
            # 计算存活率和食物收集率
            survival_rate = info['alive_agents'] / self.num_agents
            self.survival_rates.append(survival_rate)
            
            # 计算食物收集率（使用所有蛇收集的食物总数）
            food_collection_rate = sum(snake_max_scores) / self.num_agents
            self.food_collection_rates.append(food_collection_rate)
            
            # 更新最高平均 Best Score 模型
            avg_best_score = np.mean(self.recent_best_scores)
            if avg_best_score > self.best_avg_score:
                self.best_avg_score = avg_best_score
                self.agent.save_models(self.best_model_path)
                print(f"新最佳模型已保存，最近 100 轮平均 Best Score: {self.best_avg_score:.2f}")
            
            # 评估当前策略
            eval_reward = None
            if episode % self.evaluate_freq == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
            
            # 更新Excel数据
            episode_data = {
                'Episode': episode,
                'Average_Reward': current_mean_reward,
                'Max_Reward': max_reward,
                'Min_Reward': min_reward,
                'Survival_Rate': survival_rate,
                'Food_Collection_Rate': food_collection_rate,
                'Average_Length': np.mean(snake_max_lengths),  # 使用最大长度计算平均值
                'Best_Score': best_score,
                'Episode_Length': step + 1,
                'Alive_Agents': info['alive_agents'],
                'Training_Time': time.time() - episode_start_time,
                'Eval_Reward': eval_reward
            }
            self.update_excel(episode_data)
            
            # 打印训练信息
            print(f"\nEpisode {episode}")
            print(f"Recent 100 episodes average Best Score: {avg_best_score:.2f}")
            print(f"Episode reward: {current_mean_reward:.2f}")
            print(f"Survival rate: {survival_rate:.2f}")
            print(f"Food collection rate: {food_collection_rate:.2f}")
            print(f"Episode length: {step+1}")
            print(f"Best Score: {best_score}")
            print("Individual snake max scores:", [f"{score:.1f}" for score in snake_max_scores])
            print("Individual snake max lengths:", [f"{length}" for length in snake_max_lengths])
            if eval_reward is not None:
                print(f"Evaluation reward: {eval_reward:.2f}")
            print("--------------------")
            
            # 定期保存模型和绘制曲线
            if episode % self.save_freq == 0:
                model_path = os.path.join(self.save_dir, f"model_episode_{episode}.pt")
                self.agent.save_models(model_path)
                print(f"模型已保存至: {model_path}")
                self.plot_curves()
    
    def evaluate(self, num_episodes=5):
        """评估当前策略"""
        eval_rewards = []
        
        for _ in range(num_episodes):
            states = self.eval_env.reset()
            episode_reward = np.zeros(self.num_agents)
            done = False
            
            while not done:
                actions = self.agent.select_actions(states, evaluate=True)
                next_states, rewards, done, _ = self.eval_env.step(actions)
                episode_reward += rewards
                states = next_states
            
            eval_rewards.append(episode_reward.mean())
        
        return np.mean(eval_rewards)
    
    def plot_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.savefig(os.path.join(self.save_dir, 'training_rewards.png'))
        plt.close()
        
        if self.eval_rewards:
            plt.figure(figsize=(10, 6))
            plt.plot(self.eval_rewards)
            plt.title('Evaluation Rewards')
            plt.xlabel('Evaluation')
            plt.ylabel('Average Reward')
            plt.savefig(os.path.join(self.save_dir, 'eval_rewards.png'))
            plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.survival_rates)
        plt.title('Survival Rates')
        plt.xlabel('Episode')
        plt.ylabel('Survival Rate')
        plt.savefig(os.path.join(self.save_dir, 'survival_rates.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.food_collection_rates)
        plt.title('Food Collection Rates')
        plt.xlabel('Episode')
        plt.ylabel('Food Collection Rate')
        plt.savefig(os.path.join(self.save_dir, 'food_collection_rates.png'))
        plt.close()

def main():
    pretrained_model_path = "/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/snake_s1.pth"
    torch.manual_seed(0)
    np.random.seed(0)
    
    # 创建训练器
    trainer = PretrainedMultiAgentTrainer(
        pretrained_model_path=pretrained_model_path,
        num_agents=6,
        max_episodes=5000,
        max_steps=2000,
        batch_size=256,
        evaluate_freq=100,
        save_freq=500
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()