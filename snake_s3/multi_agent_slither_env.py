import gym
from gym import spaces
import numpy as np
import pygame
import math
from game import Game
from snake import Snake
from constants import *


class MultiAgentSlitherEnv(gym.Env):
    def __init__(self, num_agents=6):
        super().__init__()
        self.num_agents = num_agents

        # 动作空间：每个智能体的转向角度 [-1, 1]
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(num_agents, 1),
            dtype=np.float32
        )

        # 观察空间：每个智能体的状态向量 (31维)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(num_agents, 31),
            dtype=np.float32
        )

        # 初始化游戏组件
        self.game = None

        # 性能追踪变量
        self.episode_steps = 0
        self.max_episode_steps = 2000

        # 智能体统计信息
        self.agent_stats = [{
            'length': 0,
            'max_length': 0,
            'kills': 0,
            'food_collected': 0,
            'steps_without_food': 0
        } for _ in range(num_agents)]

        # 用于跟踪击杀的标记
        self.dead_snakes_set = set()

    def reset(self):
        """重置环境到初始状态"""
        self.game = Game()
        self.game.snakes = []  # 清空蛇列表

        # 在圆形阵列中生成多个智能体
        radius = min(SCREEN_WIDTH, SCREEN_HEIGHT) * 0.3
        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2

        for i in range(self.num_agents):
            angle = (2 * np.pi * i) / self.num_agents
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            snake = Snake(Vector(x, y))
            self.game.snakes.append(snake)

        # 重置统计信息
        for stats in self.agent_stats:
            stats.update({
                'length': 0,
                'max_length': 0,
                'kills': 0,
                'food_collected': 0,
                'steps_without_food': 0
            })

        self.dead_snakes_set.clear()
        self.episode_steps = 0
        return self._get_all_states()

    def _get_all_states(self):
        """获取所有智能体的状态"""
        states = []
        for i, snake in enumerate(self.game.snakes):
            state = self._get_agent_state(i)
            states.append(state)
        return np.array(states)

    def _get_agent_state(self, agent_idx):
        """获取单个智能体的状态"""
        snake = self.game.snakes[agent_idx]
        if snake.state == 'dead':
            return np.zeros(31, dtype=np.float32)

        # 获取最近的食物
        nearest_food = None
        min_distance = float('inf')
        for food in self.game.food:
            dist = (food.position - snake.position).mag()
            if dist < min_distance:
                min_distance = dist
                nearest_food = food

        # 食物相对位置
        if nearest_food:
            food_rel_x = (nearest_food.position.x - snake.position.x) / SCREEN_WIDTH
            food_rel_y = (nearest_food.position.y - snake.position.y) / SCREEN_HEIGHT
        else:
            food_rel_x, food_rel_y = 0, 0

        # 基础状态
        base_state = [
            food_rel_x,                    # 食物相对x位置
            food_rel_y,                    # 食物相对y位置
            snake.direction.x,             # 当前x方向
            snake.direction.y,             # 当前y方向
            min(snake.position.x, SCREEN_WIDTH - snake.position.x) / SCREEN_WIDTH,  # 到x边界的距离
            min(snake.position.y, SCREEN_HEIGHT - snake.position.y) / SCREEN_HEIGHT # 到y边界的距离
        ]

        # 其他智能体信息
        other_snakes = [s for i, s in enumerate(self.game.snakes) if i != agent_idx]
        other_snakes.sort(key=lambda s: (s.position - snake.position).mag_squared())
        other_states = []

        # 获取最近的5条蛇的信息
        for other in other_snakes[:5]:
            if other.state != 'dead':
                rel_x = (other.position.x - snake.position.x) / SCREEN_WIDTH
                rel_y = (other.position.y - snake.position.y) / SCREEN_HEIGHT
                rel_dx = other.direction.x - snake.direction.x
                rel_dy = other.direction.y - snake.direction.y
                rel_size = len(other.body) / len(snake.body) if len(snake.body) > 0 else 1
                other_states.extend([rel_x, rel_y, rel_dx, rel_dy, rel_size])
            else:
                other_states.extend([0, 0, 0, 0, 0])

        # 填充到5条蛇
        while len(other_states) < 25:
            other_states.extend([0, 0, 0, 0, 0])

        return np.array(base_state + other_states, dtype=np.float32)

    def _get_rewards(self):
        """计算所有智能体的奖励"""
        rewards = []
        for i, snake in enumerate(self.game.snakes):
            reward = 0.0
            stats = self.agent_stats[i]

            # 如果蛇刚刚死亡（状态从活着变为死亡），给予一次性死亡惩罚
            if snake.state == 'dead':
                if stats['length'] > 0:  # 说明上一步还活着
                    reward = -10.0  # 给予一次性死亡惩罚
                    stats['length'] = 0  # 重置长度记录
                else:
                    reward = 0.0  # 已经死亡的蛇不再获得任何奖励
            else:
                # 当前长度
                current_length = len(snake.body)

                # 1. 吃到食物奖励
                length_diff = current_length - stats['length']
                if length_diff > 0:
                    reward += 10.0  # 吃到食物奖励
                    stats['steps_without_food'] = 0
                else:
                    stats['steps_without_food'] += 1

                # 2. 生存奖励
                reward += 0.005  # 微弱正向奖励，鼓励存活

                # 3. 击杀奖励和危险规避
                for other_snake in self.game.snakes:
                    opp_id = id(other_snake)
                    if other_snake.state == 'dead' and opp_id not in self.dead_snakes_set:
                        # 击杀奖励，确保距离足够近
                        if opp_id in stats and stats[opp_id] < 100:
                            reward += 15.0
                            stats['kills'] += 1
                        self.dead_snakes_set.add(opp_id)
                    elif other_snake.state != 'dead':
                        # 更新距离信息
                        distance = (other_snake.position - snake.position).mag()
                        stats[opp_id] = distance

                        # 根据相对大小奖励或惩罚
                        size_ratio = len(snake.body) / len(other_snake.body)
                        if size_ratio > 1.2:  # 当前蛇较大
                            if distance < 100:  # 接近较小对手
                                reward += 0.1
                        elif size_ratio < 0.8:  # 当前蛇较小
                            if distance < 100:  # 进入危险区域
                                reward -= 0.1
                            elif 100 < distance < 200:  # 成功逃离危险区域
                                reward += 0.05

                # 4. 距离奖励（接近食物给予正奖励）
                nearest_food = min(
                    self.game.food,
                    key=lambda f: (f.position - snake.position).mag_squared(),
                    default=None
                )
                if nearest_food:
                    current_dist = (nearest_food.position - snake.position).mag()
                    if not hasattr(snake, 'last_food_distance'):
                        snake.last_food_distance = current_dist

                    # 奖励接近食物的行为
                    dist_reward = (snake.last_food_distance - current_dist) * 0.01
                    reward += dist_reward
                    snake.last_food_distance = current_dist

                # 5. 边界惩罚
                border_distance = min(
                    snake.position.x,
                    SCREEN_WIDTH - snake.position.x,
                    snake.position.y,
                    SCREEN_HEIGHT - snake.position.y
                )
                if border_distance < 50:  # 距离边界太近时给予惩罚
                    reward -= 0.1

                # 6. 长时间不吃食物的惩罚
                if stats['steps_without_food'] > 500:
                    penalty = min((stats['steps_without_food'] - 500) * 0.0005, 0.05)
                    reward -= penalty

                # 更新长度信息
                stats['length'] = current_length
                stats['max_length'] = max(stats['max_length'], current_length)

            rewards.append(reward)

        return np.array(rewards)

    def step(self, actions):
        """执行环境步进"""
        self.episode_steps += 1

        # 更新所有智能体
        for i, (snake, action) in enumerate(zip(self.game.snakes, actions)):
            if snake.state != 'dead':
                angle = (action[0] + 1) * np.pi
                distance = 10
                target_x = snake.position.x + np.cos(angle) * distance
                target_y = snake.position.y + np.sin(angle) * distance
                snake.target = Vector(target_x, target_y)

        # 游戏状态更新
        self.game.update()

        # 获取新状态和奖励
        next_states = self._get_all_states()
        rewards = self._get_rewards()

        # 检查是否结束
        all_snakes_dead = all(snake.state == 'dead' for snake in self.game.snakes)
        done = self.episode_steps >= self.max_episode_steps or all_snakes_dead

        # 准备信息字典
        info = {
            'alive_agents': sum(1 for snake in self.game.snakes if snake.state != 'dead'),
            'agent_stats': self.agent_stats,
            'episode_steps': self.episode_steps
        }

        return next_states, rewards, done, info

    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            self.game.render()
            pygame.display.update()

    def close(self):
        """关闭环境"""
        pygame.quit()