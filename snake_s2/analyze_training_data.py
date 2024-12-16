import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_process_data():
    new_snake = pd.read_excel('/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s2/results/newsnake.xlsx')
    pretrain_snake = pd.read_excel('/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s2/results/pretrain_snake.xlsx')
    
    # Limit episodes to 2000
    new_snake = new_snake[new_snake['Episode'] <= 2000]
    pretrain_snake = pretrain_snake[pretrain_snake['Episode'] <= 2000]
    
    return new_snake, pretrain_snake

def calculate_moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def print_statistics(new_data, pretrain_data, metric):
    print(f"\n===== {metric} Statistics =====")
    
    print(f"\nNew Model {metric}:")
    print(f"  Mean: {new_data[metric].mean():.2f} ± {new_data[metric].std():.2f}")
    print(f"  Max: {new_data[metric].max():.2f}")
    print(f"  Min: {new_data[metric].min():.2f}")
    print(f"  Median: {new_data[metric].median():.2f}")
    
    print(f"\nPretrain Model {metric}:")
    print(f"  Mean: {pretrain_data[metric].mean():.2f} ± {pretrain_data[metric].std():.2f}")
    print(f"  Max: {pretrain_data[metric].max():.2f}")
    print(f"  Min: {pretrain_data[metric].min():.2f}")
    print(f"  Median: {pretrain_data[metric].median():.2f}")

def plot_metric_comparison(new_data, pretrain_data, metric, window_size=10):
    plt.figure(figsize=(15, 10))
    
    # Calculate moving averages
    new_ma = calculate_moving_average(new_data[metric], window_size)
    pretrain_ma = calculate_moving_average(pretrain_data[metric], window_size)
    
    # Adjust episodes array
    episodes_new = new_data['Episode'][window_size-1:]
    episodes_pretrain = pretrain_data['Episode'][window_size-1:]
    
    # Plot curves
    plt.plot(episodes_new, new_ma, 
             label='New Model', 
             color='#E74C3C',
             linewidth=2.5)
    plt.plot(episodes_pretrain, pretrain_ma, 
             label='Pretrain Model', 
             color='#2E86C1',
             linewidth=2.5)
    
    plt.title(f'{metric} Comparison Over Training Episodes\n(Moving Average, Window={window_size})', 
             fontsize=14, pad=20)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left')
    
    # Add statistics box
    stats_text = (f"New Model - Mean: {new_data[metric].mean():.2f}, Max: {new_data[metric].max():.2f}\n"
                 f"Pretrain Model - Mean: {pretrain_data[metric].mean():.2f}, Max: {pretrain_data[metric].max():.2f}")
    plt.text(0.02, 0.02, stats_text, 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), 
             fontsize=10)
    
    # Set x-axis limit
    plt.xlim(0, 2000)
    
    # Save plot
    filename = f'training_{metric.lower()}_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

def main():
    # Load and process data
    new_snake, pretrain_snake = load_and_process_data()
    
    # Analyze and plot metrics
    metrics = ['Steps', 'Score']
    for metric in metrics:
        print_statistics(new_snake, pretrain_snake, metric)
        plot_metric_comparison(new_snake, pretrain_snake, metric)

if __name__ == '__main__':
    main()