import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel('/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s1/single_agent.xlsx')

# Configure the overall style
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Figure 1: Score vs Episode
plt.figure(figsize=(10, 6))
plt.plot(df['Episode'], df['Score'], linewidth=2, color='#1f77b4', label='Score')
plt.title('Single-Agent Score vs Episode', fontsize=14, pad=10)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Score', fontsize=12)

# Add mean line for Score
mean_score = df['Score'].mean()
plt.axhline(y=mean_score, color='r', linestyle='--', alpha=0.5, 
            label=f'Mean Score: {mean_score:.2f}')
plt.legend()

# Set x-axis limit to 2000
plt.xlim(0, 2000)

# Save the first figure
plt.savefig('single_agent_score.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Steps vs Episode
plt.figure(figsize=(10, 6))
plt.plot(df['Episode'], df['Steps'], linewidth=2, color='#2ecc71', label='Steps')
plt.title('Single-Agent Steps vs Episode', fontsize=14, pad=10)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Steps', fontsize=12)

# Add mean line for Steps
mean_steps = df['Steps'].mean()
plt.axhline(y=mean_steps, color='r', linestyle='--', alpha=0.5, 
            label=f'Mean Steps: {mean_steps:.2f}')
plt.legend()

# Set x-axis limit to 2000
plt.xlim(0, 2000)

# Save the second figure
plt.savefig('single_agent_steps.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print("==================")
print("\nScore Statistics:")
print(df['Score'].describe())
print("\nSteps Statistics:")
print(df['Steps'].describe())

# Calculate and print additional metrics
print("\nAdditional Metrics:")
print("==================")
print(f"Score Coefficient of Variation: {df['Score'].std() / df['Score'].mean() * 100:.2f}%")
print(f"Steps Coefficient of Variation: {df['Steps'].std() / df['Steps'].mean() * 100:.2f}%")