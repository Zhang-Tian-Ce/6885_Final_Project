import numpy as np
import torch
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

from multi_agent_slither_env import MultiAgentSlitherEnv
from ctde_sac import CTDESAC

class MultiAgentTrainer:
    def __init__(self, 
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
        
        # 训练参数
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.evaluate_freq = evaluate_freq
        self.save_freq = save_freq
        
        # 存储路径
        self.save_dir = "/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/models"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Excel文件路径
        self.excel_path = os.path.join(self.save_dir, "multi_agent.xlsx")
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
        self.best_avg_score = -float('inf')  # 最近 100 轮内平均 `Best Score` 的最高值
        self.best_model_path = os.path.join(self.save_dir, "best_model.pt")
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.survival_rates = []
        self.food_collection_rates = []
        self.recent_best_scores = deque(maxlen=100)  # 移动窗口

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
            
            for episode in range(1, self.max_episodes + 1):
                episode_start_time = time.time()
                states = self.env.reset()
                episode_reward = np.zeros(self.num_agents)
                
                # 记录每条蛇的最高得分
                snake_max_scores = [0] * self.num_agents
                snake_max_lengths = [0] * self.num_agents
                
                for step in range(self.max_steps):
                    actions = self.agent.select_actions(states)
                    next_states, rewards, done, info = self.env.step(actions)
                    
                    self.agent.memory.push(states, actions, rewards, next_states, done)
                    episode_reward += rewards
                    
                    # 更新每条蛇的得分和最大长度
                    for i, snake in enumerate(self.env.game.snakes):
                        snake_max_scores[i] = max(snake_max_scores[i], snake.food_count)
                        snake_max_lengths[i] = max(snake_max_lengths[i], len(snake.body))
                    
                    if len(self.agent.memory) > self.batch_size:
                        self.agent.train_step(self.batch_size)
                    
                    states = next_states
                    
                    if done:
                        break
                
                # 记录统计数据
                best_score = max(snake_max_scores)  # 使用所有蛇中吃到的最多食物数
                mean_reward = episode_reward.mean()
                max_reward = episode_reward.max()
                min_reward = episode_reward.min()
                self.episode_rewards.append(mean_reward)
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
                    'Average_Reward': mean_reward,
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
                print(f"Episode reward: {mean_reward:.2f}")
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
                
            print("训练完成!")
            print(f"总训练时间: {(time.time() - training_start_time) / 3600:.2f} 小时")
            
            # 保存最终模型
            final_model_path = os.path.join(self.save_dir, "final_model.pt")
            self.agent.save_models(final_model_path)
            print(f"最终模型已保存至: {final_model_path}")
    
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
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)
    
    # 创建训练器
    trainer = MultiAgentTrainer(
        num_agents=6,
        max_episodes=10000,
        max_steps=2000,
        batch_size=256,
        evaluate_freq=100,
        save_freq=500
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()