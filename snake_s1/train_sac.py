import os
import torch
import numpy as np
from sac import SAC
from slither_env import SlitherEnv
import matplotlib.pyplot as plt

# 初始化环境和智能体
env = SlitherEnv()
agent = SAC(state_dim=6, action_dim=1)

# 训练设置
max_episodes = 5000
batch_size = 256
save_path = "./models/"
os.makedirs(save_path, exist_ok=True)

# 记录历史
rewards_history = []
scores_history = []
steps_history = []
eating_efficiency = []  # 吃食物效率（长度/步数）

# 记录最佳表现
window_size = 100  # 使用最近100个episodes的平均分数
best_avg_score = 0

# 训练循环
for episode in range(max_episodes):
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    done = False

    while True:  # 改用 break 控制循环
        # 强制步数检查
        if episode_steps >= 2000:  # 硬编码最大步数
            break

        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        episode_steps += 1
        episode_reward += reward

        # 存储经验
        agent.memory.push(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = next_state

        if done:
            break

    # 确保记录的步数不超过限制
    episode_steps = min(episode_steps, 2000)

    # 记录结果
    current_score = info.get('length', 0)
    current_efficiency = current_score / episode_steps if episode_steps > 0 else 0

    rewards_history.append(episode_reward)
    scores_history.append(current_score)
    steps_history.append(episode_steps)
    eating_efficiency.append(current_efficiency)

    # 每个episode的信息打印
    print(f"Episode {episode:4d}: "
          f"Total Reward = {episode_reward:8.2f}, "
          f"Score = {current_score:3d}, "
          f"Steps = {episode_steps:5d}, "
          f"Efficiency = {current_efficiency:.4f}")

    # 基于移动平均score保存模型
    if episode >= window_size:
        current_avg_score = np.mean(scores_history[-window_size:])
        if current_avg_score > best_avg_score:
            best_avg_score = current_avg_score
            model_path = os.path.join(
                save_path,
                f'sac_model_avg{current_avg_score:.1f}_e{episode+1}.pth'
            )
            torch.save({
                'episode': episode,
                'policy_state_dict': agent.policy.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'avg_score': current_avg_score,
                'current_score': current_score,
                'window_size': window_size
            }, model_path)
            print(f"New best average score: {current_avg_score:.1f}, Model saved")

# 训练结束后的分析
def plot_training_results(rewards_history, scores_history, eating_efficiency):
    plt.figure(figsize=(15, 5))

    # 总 Reward 曲线（带移动平均）
    plt.subplot(1, 3, 1)
    rewards_ma = np.convolve(rewards_history,
                            np.ones(window_size)/window_size,
                            mode='valid')
    plt.plot(rewards_history, alpha=0.3, label='Raw')
    plt.plot(rewards_ma, label=f'{window_size}-Episode MA')
    plt.title('Total Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # 总 Score 曲线（带移动平均）
    plt.subplot(1, 3, 2)
    scores_ma = np.convolve(scores_history,
                           np.ones(window_size)/window_size,
                           mode='valid')
    plt.plot(scores_history, alpha=0.3, label='Raw')
    plt.plot(scores_ma, label=f'{window_size}-Episode MA')
    plt.title('Snake Length Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.legend()

    # 吃食物效率曲线（带移动平均）
    plt.subplot(1, 3, 3)
    efficiency_ma = np.convolve(eating_efficiency,
                               np.ones(window_size)/window_size,
                               mode='valid')
    plt.plot(eating_efficiency, alpha=0.3, label='Raw')
    plt.plot(efficiency_ma, label=f'{window_size}-Episode MA')
    plt.title('Eating Efficiency Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Efficiency (Length/Steps)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'final_training_results.png'))
    plt.close()

# 训练结束后的统计分析
print("\n=== Final Training Statistics ===")
print(f"Total Episodes Trained: {max_episodes}")
print(f"Best Average Score: {best_avg_score:.1f}")
print(f"Max Score Reached: {max(scores_history)}")
print(f"Average Score (last 1000 ep): {np.mean(scores_history[-1000:]):.1f}")
print(f"Average Reward (last 1000 ep): {np.mean(rewards_history[-1000:]):.1f}")
print(f"Average Efficiency (last 1000 ep): {np.mean(eating_efficiency[-1000:]):.4f}")
print(f"Score Standard Deviation (last 1000 ep): {np.std(scores_history[-1000:]):.1f}")

# 保存完整训练历史
np.savez(os.path.join(save_path, 'training_history.npz'),
         rewards=rewards_history,
         scores=scores_history,
         efficiency=eating_efficiency,
         steps=steps_history)

# 绘制最终结果图表
plot_training_results(rewards_history, scores_history, eating_efficiency)