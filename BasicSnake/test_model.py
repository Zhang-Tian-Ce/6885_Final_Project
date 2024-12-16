#!/usr/bin/python
# -*- coding: utf-8 -*-
from Agent import AgentDiscretePPO
from env4Snake import Snake
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def moving_average(data, window_size=5):
    """Calculate moving average with specified window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

class ModelEvaluator:
    def __init__(self, model_path, n_episodes=100):
        """Initialize the evaluator"""
        self.model_path = model_path
        self.n_episodes = n_episodes
        self.test_env = Snake()
        self.test_env.snake_speed = 0  # Disable visualization delay
        
        # Initialize agent
        self.agent = AgentDiscretePPO()
        self.agent.init(512, 6, 4, if_use_gae=True)
        
        # Load model
        self.agent.act.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))
        )
        
        # Initialize metrics storage
        self.scores = []
    
    def run_episode(self):
        """Run a single test episode"""
        o = self.test_env.reset()
        episode_score = 0
        done = False
        
        while not done:
            a_int, _ = self.agent.select_action(o)
            o2, reward, done, _ = self.test_env.step(a_int)
            episode_score += reward
            o = o2
            
        return episode_score
    
    def evaluate(self):
        """Run full evaluation and generate statistics"""
        print(f"\nStarting evaluation over {self.n_episodes} episodes...")
        
        # Run episodes
        for episode in range(self.n_episodes):
            score = self.run_episode()
            self.scores.append(score)
            if (episode + 1) % 10 == 0:
                print(f"Completed {episode + 1} episodes")
        
        # Calculate statistics
        mean_score = np.mean(self.scores)
        max_score = np.max(self.scores)
        min_score = np.min(self.scores)
        std_score = np.std(self.scores)
        
        # Print results
        print("\n========= Evaluation Results =========")
        print(f"Average Score: {mean_score:.2f} ± {std_score:.2f}")
        print(f"Maximum Score: {max_score:.2f}")
        print(f"Minimum Score: {min_score:.2f}")
        
        # Generate and save plot
        self.create_moving_average_plot()
        
        return mean_score, max_score, min_score

    def create_moving_average_plot(self):
        """Create and save moving average plot"""
        # Calculate moving average
        ma_scores = moving_average(self.scores, window_size=5)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(ma_scores)), ma_scores, 'b-', linewidth=2)
        plt.title('Moving Average of Scores (Window Size = 5)')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        plt.grid(True)
        
        # Add mean line
        mean_score = np.mean(self.scores)
        plt.axhline(y=mean_score, color='r', linestyle='--', 
                   label=f'Overall Mean: {mean_score:.2f}')
        plt.legend()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(f'scores_ma_plot_{timestamp}.png')
        plt.close()

def main():
    evaluator = ModelEvaluator('act_weight.pkl', n_episodes=100)
    mean_score, max_score, min_score = evaluator.evaluate()

if __name__ == "__main__":
    main()