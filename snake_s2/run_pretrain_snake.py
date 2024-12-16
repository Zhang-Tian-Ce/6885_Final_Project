import torch
from slither_env import SlitherEnv
from sac import SAC

def load_model(agent, model_path):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"成功加载模型：{model_path}")

def observe_pretrain_snake():
    """运行 pretrain_snake_best_model.pth 模型并观察游戏"""
    # 模型路径
    model_path = '/Users/tz/Downloads/Reinforcement-Learning/FinalProject/snake_s2/results/pretrain_snake_best_model.pth'
    
    # 创建环境
    env = SlitherEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化智能体
    agent = SAC(state_dim, action_dim)
    load_model(agent, model_path)  # 加载模型

    # 开始游戏
    state = env.reset()
    done = False
    total_reward = 0

    print("运行 pretrain_snake_best_model.pth，按 Ctrl+C 退出观察。")
    try:
        while not done:
            # 使用训练好的策略选择动作，不传递 evaluate 参数
            action = agent.select_action(state)
            
            # 执行动作并更新环境
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # 渲染游戏画面
            env.render()

        print(f"游戏结束，总奖励: {total_reward:.2f}, 最终长度: {info['length']}")
    except KeyboardInterrupt:
        print("退出游戏观察。")
    finally:
        env.close()

if __name__ == '__main__':
    observe_pretrain_snake()