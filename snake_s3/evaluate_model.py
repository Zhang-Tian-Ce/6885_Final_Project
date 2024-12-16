import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multi_agent_slither_env import MultiAgentSlitherEnv
from ctde_sac import CTDESAC

def evaluate_models(pretrained_path, new_model_path, num_episodes=100):
    env = MultiAgentSlitherEnv(num_agents=6)
    results = {
        'Pretrained Model': {'rewards': [], 'scores': []},
        'New Model': {'rewards': [], 'scores': []}
    }
    
    # 评估两个模型
    for model_path, model_name in [(pretrained_path, 'Pretrained Model'), 
                                  (new_model_path, 'New Model')]:
        # 初始化模型
        agent = CTDESAC(
            num_agents=6,
            state_dim=env.observation_space.shape[1],
            action_dim=env.action_space.shape[1],
            hidden_dim=256
        )
        
        # 加载模型
        checkpoint = torch.load(model_path)
        for i, policy in enumerate(agent.policies):
            policy.load_state_dict(checkpoint['policies'][i])
        agent.critic.load_state_dict(checkpoint['critic'])
        
        print(f"\nEvaluating {model_name}...")
        
        # 运行评估episodes
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            snake_scores = [0] * env.num_agents
            
            while True:
                action = agent.select_actions(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward.mean()
                for i, snake in enumerate(env.game.snakes):
                    snake_scores[i] = max(snake_scores[i], snake.food_count)
                
                state = next_state
                if done:
                    break
            
            results[model_name]['rewards'].append(episode_reward)
            results[model_name]['scores'].append(max(snake_scores))
            
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Score={max(snake_scores)}")
    
    # 计算统计数据
    stats = {}
    for model_name in results:
        stats[model_name] = {
            'Reward': {
                'Mean': np.mean(results[model_name]['rewards']),
                'Std': np.std(results[model_name]['rewards']),
                'Max': np.max(results[model_name]['rewards']),
                'Min': np.min(results[model_name]['rewards']),
                'Median': np.median(results[model_name]['rewards'])
            },
            'Score': {
                'Mean': np.mean(results[model_name]['scores']),
                'Std': np.std(results[model_name]['scores']),
                'Max': np.max(results[model_name]['scores']),
                'Min': np.min(results[model_name]['scores']),
                'Median': np.median(results[model_name]['scores'])
            }
        }
    
    # 打印统计数据
    print("\n=== Statistical Analysis ===")
    for model_name in stats:
        print(f"\n{model_name}:")
        for metric in ['Reward', 'Score']:
            print(f"\n{metric} Statistics:")
            for stat, value in stats[model_name][metric].items():
                print(f"{stat}: {value:.2f}")
    
    # 创建数据帧用于绘图
    df_list = []
    for model_name in results:
        df = pd.DataFrame({
            'Episode': range(1, num_episodes + 1),
            'Reward': results[model_name]['rewards'],
            'Score': results[model_name]['scores'],
            'Model': model_name
        })
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    
    # 计算移动平均
    window_size = 10
    for model_name in results:
        mask = df['Model'] == model_name
        df.loc[mask, 'Reward_MA'] = df.loc[mask, 'Reward'].rolling(window=window_size).mean()
        df.loc[mask, 'Score_MA'] = df.loc[mask, 'Score'].rolling(window=window_size).mean()
    
    # 绘制图表
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    
    # Reward图
    plt.figure()
    for model_name, color in [('Pretrained Model', '#2ecc71'), ('New Model', '#e74c3c')]:
        mask = df['Model'] == model_name
        plt.plot(df[mask]['Episode'], df[mask]['Reward_MA'], 
                label=model_name, color=color, linewidth=1.5)
    plt.title('Evaluation Reward over Episodes (Moving Average, Window=10)', fontsize=14, pad=15)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('evaluation_reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Score图
    plt.figure()
    for model_name, color in [('Pretrained Model', '#2ecc71'), ('New Model', '#e74c3c')]:
        mask = df['Model'] == model_name
        plt.plot(df[mask]['Episode'], df[mask]['Score_MA'], 
                label=model_name, color=color, linewidth=1.5)
    plt.title('Evaluation Score over Episodes (Moving Average, Window=10)', fontsize=14, pad=15)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('evaluation_score_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    pretrained_path = "/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/Pretrain_multi_agent/best_pretrained_model.pt"
    new_model_path = "/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/new_multi_agent/new_best_model.pt"
    evaluate_models(pretrained_path, new_model_path, num_episodes=100)