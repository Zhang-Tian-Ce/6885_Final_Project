import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

plt.style.use('default')
sns.set_theme(style="whitegrid")

def load_data():
    """Load data from specific local paths and limit to 2000 episodes"""
    single_agent_path = '/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s1/single_agent.xlsx'
    multi_agent_path = '/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s3/new_multi_agent/multi_agent.xlsx'
    
    # Load and limit both datasets to 2000 episodes
    single_data = pd.read_excel(single_agent_path).head(2000)
    multi_data = pd.read_excel(multi_agent_path).head(2000)
    
    return single_data, multi_data

def analyze_convergence(data, window_size=50, threshold=0.05, stability_window=30):
    """Analyze convergence by detecting when the moving average stabilizes"""
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()
    
    relative_std = rolling_std / rolling_mean.abs()
    stable_points = relative_std < threshold
    
    for i in range(len(stable_points) - stability_window):
        if all(stable_points[i:i+stability_window]):
            return i
    
    return len(data)

def calculate_performance_stats(data):
    """Calculate key performance statistics"""
    return {
        'Mean': np.mean(data),
        'Std Dev': np.std(data),
        'Max': np.max(data),
        'Min': np.min(data),
        'Median': np.median(data),
        'CV (%)': stats.variation(data) * 100
    }

def plot_metric_comparison(single_data, multi_data, metric_single, metric_multi, title, ylabel, window=50):
    """Plot comparison with confidence intervals"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Calculate rolling statistics
    single_rolling = single_data[metric_single].rolling(window=window).mean()
    single_std = single_data[metric_single].rolling(window=window).std()
    
    multi_rolling = multi_data[metric_multi].rolling(window=window).mean()
    multi_std = multi_data[metric_multi].rolling(window=window).std()
    
    # Create episode range
    episodes = np.arange(2000)
    
    # Plot means
    ax.plot(episodes, single_rolling, label='Single Agent', color='blue', linewidth=2)
    ax.plot(episodes, multi_rolling, label='Multi Agent', color='red', linewidth=2)
    
    # Plot confidence intervals
    ax.fill_between(episodes, 
                   single_rolling - single_std, 
                   single_rolling + single_std, 
                   color='blue', alpha=0.2)
    ax.fill_between(episodes, 
                   multi_rolling - multi_std, 
                   multi_rolling + multi_std, 
                   color='red', alpha=0.2)
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits explicitly
    ax.set_xlim(0, 2000)
    
    plt.tight_layout()
    plt.show()

def print_statistics(stats_dict, name):
    """Print statistics in a formatted way"""
    print(f"\n{name}:")
    print("-" * 50)
    for metric, value in stats_dict.items():
        if metric == 'CV (%)':
            print(f"{metric:<15}: {value:>15.2f}%")
        else:
            print(f"{metric:<15}: {value:>15.2f}")

def analyze_phases(data, conv_point, metric):
    """Analyze performance in different training phases"""
    early_phase = data[metric][:conv_point]
    late_phase = data[metric][conv_point:]
    
    return {
        'Early Phase': calculate_performance_stats(early_phase),
        'Late Phase': calculate_performance_stats(late_phase)
    }

def main():
    # Load data
    print("Loading data...")
    print("Analyzing first 2000 episodes for both agents")
    single_data, multi_data = load_data()
    
    # Analyze convergence
    print("\nAnalyzing convergence points...")
    single_conv_score = analyze_convergence(single_data['Score'])
    multi_conv_score = analyze_convergence(multi_data['Best_Score'])
    
    print(f"Single Agent score converges at episode: {single_conv_score}")
    print(f"Multi Agent score converges at episode: {multi_conv_score}")
    
    # Plot comparisons
    print("\nGenerating plots...")
    plot_metric_comparison(single_data, multi_data, 
                         'Score', 'Best_Score',
                         'Score Comparison (First 2000 Episodes)', 
                         'Score')
    
    plot_metric_comparison(single_data, multi_data, 
                         'Steps', 'Episode_Length',
                         'Steps Comparison (First 2000 Episodes)', 
                         'Steps per Episode')
    
    # Calculate statistics for both metrics
    print("\nCalculating statistics...")
    
    # Score statistics
    single_score_stats = calculate_performance_stats(single_data['Score'])
    multi_score_stats = calculate_performance_stats(multi_data['Best_Score'])
    
    # Steps statistics
    single_steps_stats = calculate_performance_stats(single_data['Steps'])
    multi_steps_stats = calculate_performance_stats(multi_data['Episode_Length'])
    
    # Print detailed statistics
    print("\n=== SCORE STATISTICS (2000 Episodes) ===")
    print_statistics(single_score_stats, "Single Agent Score")
    print_statistics(multi_score_stats, "Multi Agent Score")
    
    print("\n=== STEPS STATISTICS (2000 Episodes) ===")
    print_statistics(single_steps_stats, "Single Agent Steps")
    print_statistics(multi_steps_stats, "Multi Agent Steps")
    
    # Analyze and print phase-specific statistics
    print("\n=== PHASE ANALYSIS ===")
    
    single_score_phases = analyze_phases(single_data, single_conv_score, 'Score')
    multi_score_phases = analyze_phases(multi_data, multi_conv_score, 'Best_Score')
    
    print("\nSingle Agent Score Phases:")
    print("Early Phase (Before Convergence):")
    print_statistics(single_score_phases['Early Phase'], "Statistics")
    print("\nLate Phase (After Convergence):")
    print_statistics(single_score_phases['Late Phase'], "Statistics")
    
    print("\nMulti Agent Score Phases:")
    print("Early Phase (Before Convergence):")
    print_statistics(multi_score_phases['Early Phase'], "Statistics")
    print("\nLate Phase (After Convergence):")
    print_statistics(multi_score_phases['Late Phase'], "Statistics")

if __name__ == "__main__":
    main()