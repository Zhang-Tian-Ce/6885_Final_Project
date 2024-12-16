# render_game.py
import torch
import pygame
import os
from multi_agent_slither_env import MultiAgentSlitherEnv
from ctde_sac import CTDESAC

def render_game(model_path, num_episodes=5, max_steps=2000):
    env = MultiAgentSlitherEnv(num_agents=6)
    agent = CTDESAC(
        num_agents=6,
        state_dim=env.observation_space.shape[1],
        action_dim=env.action_space.shape[1],
        hidden_dim=256
    )
    
    # 正确加载模型
    checkpoint = torch.load(model_path)
    for i, policy in enumerate(agent.policies):
        if i < len(checkpoint['policies']):
            policy.load_state_dict(checkpoint['policies'][i])
    agent.critic.load_state_dict(checkpoint['critic'])
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}")
        state = env.reset()
        episode_reward = 0
        snake_scores = [0] * env.num_agents
        
        for step in range(max_steps):
            env.render()
            pygame.time.wait(50)
            
            action = agent.select_actions(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward.mean()
            state = next_state
            
            # 更新和打印每条蛇的得分
            for i, snake in enumerate(env.game.snakes):
                snake_scores[i] = max(snake_scores[i], snake.food_count)
            
            print(f"Step: {step}, Reward: {reward.mean():.2f}, "
                  f"Alive: {info['alive_agents']}, "
                  f"Scores: {snake_scores}")
            
            if done:
                break
        
        print(f"Episode {episode + 1} finished - Total reward: {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    model_path = "/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/models/best_model.pt"
    render_game(model_path)