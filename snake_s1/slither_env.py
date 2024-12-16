import gym
from gym import spaces
import numpy as np
import pygame
import math
import random
from game import Game  
from snake import Snake  
from constants import *

class SlitherEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = Game()
        self._last_length = 0
        self.max_length = 0
        self.episode_steps = 0
        self.max_episode_steps = 2000
        self.steps_without_food = 0  

        # 动作空间：[-1,1]表示转向角度
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        # 状态空间：6维向量
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(6,),
            dtype=np.float32
        )

    def reset(self):
      self.game = Game()
      self._last_length = 0
      self.max_length = 0
      self.episode_steps = 0  # 确保这里重置
      self.steps_without_food = 0  # 重置这个计数器
      return self._get_state()

    def _get_state(self):
        """获取状态表示"""
        if not self.game.snake or self.game.state == 'end':
            return np.zeros(6, dtype=np.float32)

        snake = self.game.snake

        # 找到最近的食物
        nearest_food = None
        min_distance = float('inf')
        for food in self.game.food:
            dist = (food.position - snake.position).mag()
            if dist < min_distance:
                min_distance = dist
                nearest_food = food

        if nearest_food:
            # 计算相对位置并归一化
            rel_x = (nearest_food.position.x - snake.position.x) / SCREEN_WIDTH
            rel_y = (nearest_food.position.y - snake.position.y) / SCREEN_HEIGHT
        else:
            rel_x, rel_y = 0, 0

        # 计算到边界的距离（归一化）
        border_distances = {
            'up': snake.position.y / SCREEN_HEIGHT,
            'down': (SCREEN_HEIGHT - snake.position.y) / SCREEN_HEIGHT,
            'left': snake.position.x / SCREEN_WIDTH,
            'right': (SCREEN_WIDTH - snake.position.x) / SCREEN_WIDTH
        }

        return np.array([
            rel_x,                          # 食物相对x位置
            rel_y,                          # 食物相对y位置
            snake.direction.x,              # 当前方向x
            snake.direction.y,              # 当前方向y
            min(border_distances.values()), # 最近边界距离
            len(snake.body) / 100.0        # 归一化的长度
        ], dtype=np.float32)

    def _get_reward(self):
        if not self.game.snake or self.game.state == 'end':
            return -5.0  # 降低死亡惩罚（从-10改为-5）

        reward = 0.0
        current_length = len(self.game.snake.body)

        # 1. 吃到食物的奖励（增加）
        length_diff = current_length - self._last_length
        if length_diff > 0:
            reward += 10.0  # 显著增加吃食物的奖励（从5.0增加到10.0）
            self.steps_without_food = 0
        else:
            self.steps_without_food += 1

        # 2. 距离奖励（新增）
        snake = self.game.snake
        nearest_food = min(self.game.food,
                          key=lambda f: (f.position - snake.position).mag_squared(),
                          default=None)

        if nearest_food:
            current_dist = (nearest_food.position - snake.position).mag()
            if not hasattr(self, 'last_food_distance'):
                self.last_food_distance = current_dist

            # 接近食物给予正奖励
            dist_reward = (self.last_food_distance - current_dist) * 0.01
            reward += dist_reward
            self.last_food_distance = current_dist

        # 3. 生存奖励（新增）
        reward += 0.005  # 小的正向生存奖励

        # 4. 惩罚过长时间不吃食物（优化）
        if self.steps_without_food > 500:  # 降低阈值
            penalty = min((self.steps_without_food - 500) * 0.0005, 0.05)  # 降低惩罚强度
            reward -= penalty

        self._last_length = current_length
        return reward

    def step(self, action):
        # 强制检查步数
        if self.episode_steps >= self.max_episode_steps:
            done = True
            return (self._get_state(),
                    0,
                    done,
                    {'length': len(self.game.snake.body) if self.game.snake else 0,
                    'max_length': self.max_length,
                    'steps': self.max_episode_steps})

        # 记录步数并更新
        self.episode_steps += 1
        self.game.update(action)

        # 获取状态和奖励
        new_state = self._get_state()
        reward = self._get_reward()

        # 检查游戏是否结束
        done = (self.game.state == 'end' or
                self.episode_steps >= self.max_episode_steps)

        info = {
            'length': len(self.game.snake.body) if self.game.snake else 0,
            'max_length': self.max_length,
            'steps': self.episode_steps
        }

        return new_state, reward, done, info

    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            self.game.render()  # 只有在需要时才调用渲染
            pygame.display.update()

    def close(self):
        """关闭环境"""
        pygame.quit()