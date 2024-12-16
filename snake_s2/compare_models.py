import torch
import numpy as np
from slither_env import SlitherEnv
from sac import SAC
import matplotlib.pyplot as plt
from collections import defaultdict
import time

class ModelTrendAnalyzer:
    def __init__(self):
        self.env = SlitherEnv()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        self.model_paths = {
            'pretrain': '/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s2/results/pretrain_snake_best_model.pth',
            'new': '/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s2/results/newsanke_best_model.pth'
        }
        
        self.colors = {
            'pretrain': '#2E86C1',
            'new': '#E74C3C'
        }
        
        self.window_size = 10
        self.num_episodes = 100
        
    def load_model(self, model_type):
        agent = SAC(self.state_dim, self.action_dim)
        checkpoint = torch.load(self.model_paths[model_type])
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        return agent
        
    def evaluate_model(self, agent, model_type):
        stats = defaultdict(list)
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            state = self.env.reset()
            steps = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                steps += 1
            
            stats['scores'].append(info['length'])
            stats['steps'].append(steps)
            stats['episode'].append(episode + 1)
            
            if (episode + 1) % 50 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_ep = elapsed_time / (episode + 1)
                remaining_episodes = self.num_episodes - (episode + 1)
                estimated_time = remaining_episodes * avg_time_per_ep
                
                print(f"Model: {model_type}")
                print(f"Episode: {episode + 1}/{self.num_episodes}")
                print(f"Recent avg score: {np.mean(stats['scores'][-50:]):.2f}")
                print(f"Estimated remaining time: {estimated_time/60:.1f} minutes")
                print("-" * 50)
                
        return stats
        
    def evaluate_both_models(self):
        all_stats = {}
        for model_type in ['pretrain', 'new']:
            print(f"\nEvaluating {model_type} model...")
            agent = self.load_model(model_type)
            all_stats[model_type] = self.evaluate_model(agent, model_type)
        return all_stats

    def plot_evaluation_trends(self, all_stats):
        metrics = {
            'scores': 'Score',
            'steps': 'Steps'
        }
        
        for metric, ylabel in metrics.items():
            plt.figure(figsize=(15, 10))
            
            for model_type in ['pretrain', 'new']:
                episodes = all_stats[model_type]['episode']
                values = all_stats[model_type][metric]
                
                # Plot raw data with low alpha
                plt.plot(episodes, values, color=self.colors[model_type],
                        alpha=0.2, linewidth=1)
                
                # Plot moving average
                moving_avg = np.convolve(values, 
                                       np.ones(self.window_size)/self.window_size, 
                                       mode='valid')
                episodes_avg = episodes[self.window_size-1:]
                
                plt.plot(episodes_avg, moving_avg, 
                        color=self.colors[model_type],
                        label=f'{model_type.capitalize()} Model',
                        linewidth=2.5)
            
            plt.title(f'Model Evaluation - {ylabel}\n(Moving Average Window={self.window_size})', 
                     fontsize=14, pad=20)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10, loc='upper left')
            
            # Add statistics box
            stats_text = self._generate_stats_text(all_stats, metric)
            plt.text(0.02, 0.02, stats_text, 
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=10,
                    verticalalignment='bottom')
            
            filename = f'evaluation_{metric}_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved {filename}")
            plt.close()

    def _generate_stats_text(self, all_stats, metric):
        stats_text = ""
        for model_type in ['pretrain', 'new']:
            data = all_stats[model_type][metric]
            stats_text += f"{model_type.capitalize()} Model:\n"
            stats_text += f"Mean: {np.mean(data):.2f} ± {np.std(data):.2f}\n"
            stats_text += f"Max: {np.max(data):.2f}\n"
            stats_text += f"Min: {np.min(data):.2f}\n"
            stats_text += f"Median: {np.median(data):.2f}\n\n"
        return stats_text

    def print_statistics(self, all_stats):
        metrics = ['scores', 'steps']
        
        print("\n===== Evaluation Statistics =====")
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            for model_type in ['pretrain', 'new']:
                data = all_stats[model_type][metric]
                print(f"\n{model_type.capitalize()} Model:")
                print(f"  Mean: {np.mean(data):.2f} ± {np.std(data):.2f}")
                print(f"  Max: {np.max(data):.2f}")
                print(f"  Min: {np.min(data):.2f}")
                print(f"  Median: {np.median(data):.2f}")
                print(f"  25th percentile: {np.percentile(data, 25):.2f}")
                print(f"  75th percentile: {np.percentile(data, 75):.2f}")

def main():
    analyzer = ModelTrendAnalyzer()
    all_stats = analyzer.evaluate_both_models()
    analyzer.plot_evaluation_trends(all_stats)
    analyzer.print_statistics(all_stats)
    analyzer.env.close()

if __name__ == '__main__':
    main()