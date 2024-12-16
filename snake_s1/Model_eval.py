import torch
import numpy as np
from slither_env import SlitherEnv
from sac import SAC
import matplotlib.pyplot as plt
import pandas as pd
import os

def create_score_plot(episodes, scores, save_path, avg_score):
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, scores, 'b-', alpha=0.3, label='Raw Score')
    plt.plot(episodes, pd.Series(scores).rolling(10).mean(), 'r-', 
             linewidth=2, label='10-Episode Moving Average')
    
    plt.axhline(y=np.mean(scores), color='g', linestyle='--', 
                label=f'Mean Score: {np.mean(scores):.1f}')
    
    plt.title('Score Evaluation', fontsize=14, pad=20)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score (Snake Length)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    stats_text = f'Mean: {np.mean(scores):.1f}\nStd: {np.std(scores):.1f}\n'
    stats_text += f'Max: {np.max(scores)}\nMin: {np.min(scores)}'
    
    plt.text(0.95, 0.05, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.savefig(os.path.join(save_path, f'score_evaluation_avg{avg_score:.1f}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_steps_plot(episodes, steps, save_path, avg_score):
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, steps, 'c-', alpha=0.3, label='Raw Steps')
    plt.plot(episodes, pd.Series(steps).rolling(10).mean(), 'r-', 
             linewidth=2, label='10-Episode Moving Average')
    
    plt.axhline(y=np.mean(steps), color='g', linestyle='--', 
                label=f'Mean Steps: {np.mean(steps):.1f}')
    
    plt.title('Steps Evaluation', fontsize=14, pad=20)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps per Episode', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    stats_text = f'Mean: {np.mean(steps):.1f}\nStd: {np.std(steps):.1f}\n'
    stats_text += f'Max: {np.max(steps)}\nMin: {np.min(steps)}'
    
    plt.text(0.95, 0.05, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.savefig(os.path.join(save_path, f'steps_evaluation_avg{avg_score:.1f}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model_path, num_episodes=100):
    env = SlitherEnv()
    agent = SAC(state_dim=6, action_dim=1)
    
    checkpoint = torch.load(model_path)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    print(f"Loaded model from episode {checkpoint['episode']} with average score {checkpoint['avg_score']:.1f}")
    
    scores = []
    steps = []
    episodes = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_steps = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_steps += 1
        
        score = info.get('length', 0)
        
        scores.append(score)
        steps.append(episode_steps)
        episodes.append(episode)
        
        print(f"Episode {episode}: Score = {score}, Steps = {episode_steps}")
    
    save_path = os.path.dirname(model_path)
    avg_score = np.mean(scores)
    
    create_score_plot(episodes, scores, save_path, avg_score)
    create_steps_plot(episodes, steps, save_path, avg_score)
    
    print("\n=== Evaluation Results ===")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Steps: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Max Steps: {max(steps)}")
    print(f"Min Score: {min(scores)}")
    print(f"Min Steps: {min(steps)}")
    
    stats_df = pd.DataFrame({
        'Metric': ['Score', 'Steps'],
        'Mean': [np.mean(scores), np.mean(steps)],
        'Std': [np.std(scores), np.std(steps)],
        'Min': [np.min(scores), np.min(steps)],
        'Max': [np.max(scores), np.max(steps)],
        'Median': [np.median(scores), np.median(steps)],
        '25%': [np.percentile(scores, 25), np.percentile(steps, 25)],
        '75%': [np.percentile(scores, 75), np.percentile(steps, 75)]
    })
    
    stats_path = os.path.join(save_path, f'evaluation_stats_avg{avg_score:.1f}.csv')
    stats_df.to_csv(stats_path, index=False)
    
    return {
        'scores': scores,
        'steps': steps,
        'stats_df': stats_df
    }

if __name__ == "__main__":
    model_path = '/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s1/models/S1_Best_Model.pth'
    results = evaluate_model(model_path, num_episodes=100)