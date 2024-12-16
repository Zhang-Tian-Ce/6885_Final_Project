# test_slither_env.py
import pygame
from slither_env import SlitherEnv
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 确保pygame系统被正确初始化
if not pygame.get_init():
    pygame.init()
if not pygame.font.get_init():
    pygame.font.init()

def test_env_visualization():
    """测试环境可视化和基本功能"""
    env = SlitherEnv()
    state = env.reset()
    
    # 初始化记录
    stats = {
        'length': [],
        'rewards': [],
        'total_reward': 0,
    }
    
    # 不需要再次初始化pygame，因为已经在文件开头初始化了
    clock = pygame.time.Clock()
    running = True
    step = 0
    
    print("Starting environment test...")
    print(f"Initial state shape: {state.shape}")
    
    try:
        while running and step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
            
            action = [np.random.uniform(-1, 1)]
            next_state, reward, done, info = env.step(action)
            
            stats['length'].append(info['length'])
            stats['rewards'].append(reward)
            stats['total_reward'] += reward
            
            if step % 100 == 0:
                print(f"\nStep {step}:")
                print(f"Snake Length: {info['length']}")
                print(f"Total Reward: {stats['total_reward']:.2f}")
            
            if len(stats['length']) > 1 and stats['length'][-1] > stats['length'][-2]:
                print(f"\n[Food Eaten] at Step {step}:")
                print(f"New Length: {info['length']}")
                print(f"Reward: {reward}")
            
            env.render()
            pygame.display.flip()
            
            if done:
                print("\nEpisode ended!")
                print(f"Final Length: {info['length']}")
                print(f"Total Steps: {step}")
                print(f"Total Reward: {stats['total_reward']:.2f}")
                time.sleep(2)
                break
                
            step += 1
            clock.tick(30)
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        env.close()
        
    # 保存统计数据的绘图放在pygame关闭之后
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(stats['length'], label='Snake Length')
    plt.title('Snake Length Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Length')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(stats['rewards'], label='Step Reward')
    plt.title('Rewards Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_basic_functions():
    """测试环境的基本功能"""
    env = SlitherEnv()
    
    print("\nTesting reset...")
    state = env.reset()
    assert state is not None, "Reset should return a state"
    print(f"Reset successful. State shape: {state.shape}")
    
    print("\nTesting step function...")
    action = [0.0]
    state, reward, done, info = env.step(action)
    print(f"Step successful:")
    print(f"- State shape: {state.shape}")
    print(f"- Reward: {reward}")
    print(f"- Done: {done}")
    print(f"- Info: {info}")
    
    print("\nTesting reward ranges...")
    rewards = []
    for _ in range(100):
        action = [np.random.uniform(-1, 1)]
        _, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            env.reset()
    print(f"Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    
    env.close()

if __name__ == "__main__":
    try:
        # 运行测试
        print("Running basic function tests...")
        test_basic_functions()
        
        print("\nRunning visualization test...")
        test_env_visualization()
    finally:
        pygame.quit()  # 确保最后清理pygame