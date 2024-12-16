# slither_env.py

import gym
from gym import spaces
import numpy as np
import pygame
import math
from game import Game
from snake import Snake
from simple_ai_snake import SimpleAISnake
from constants import Vector, SCREEN_WIDTH, SCREEN_HEIGHT

class SlitherEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 动作空间：[-1,1]表示转向角度
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        # 观察空间：基础状态(6维) + 每个对手的信息(5*5=25维) = 31维
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(31,),
            dtype=np.float32
        )
        
        # 初始化游戏组件
        self.game = None
        self.player = None
        self.opponents = []
        
        # 性能追踪变量
        self._last_length = 0
        self.max_length = 0
        self.episode_steps = 0
        self.max_episode_steps = 2000
        self.steps_without_food = 0
        
        # 战斗统计
        self.kills = 0
        self.near_death_count = 0
        self._last_distances = {}
        self._dead_opponents = set()  # 记录已处理的死亡对手

    def reset(self):
        """重置环境到初始状态"""
        self.game = Game()
        self.player = self.game.snake
        self.opponents = []
        
        # 添加5个AI对手
        for _ in range(5):
            opponent = SimpleAISnake()
            self.opponents.append(opponent)
            self.game.snakes.append(opponent)
        
        # 重置所有追踪变量
        self._last_length = 0
        self.max_length = 0
        self.episode_steps = 0
        self.steps_without_food = 0
        self.kills = 0
        self.near_death_count = 0
        self._last_distances = {}
        self._dead_opponents = set()
        
        return self._get_state()

    def _get_state(self):
        """获取当前状态表示"""
        if not self.player or self.player.state == 'dead':
            return np.zeros(31, dtype=np.float32)
        
        # 获取基础状态
        snake = self.player
        
        # 找到最近的食物
        nearest_food = None
        min_distance = float('inf')
        for food in self.game.food:
            dist = (food.position - snake.position).mag()
            if dist < min_distance:
                min_distance = dist
                nearest_food = food
        
        # 计算食物相对位置
        if nearest_food:
            food_rel_x = (nearest_food.position.x - snake.position.x) / SCREEN_WIDTH
            food_rel_y = (nearest_food.position.y - snake.position.y) / SCREEN_HEIGHT
        else:
            food_rel_x, food_rel_y = 0, 0
        
        # 基础状态向量
        base_state = [
            food_rel_x,                    # 食物相对x位置
            food_rel_y,                    # 食物相对y位置
            snake.direction.x,             # 当前x方向
            snake.direction.y,             # 当前y方向
            min(snake.position.x, SCREEN_WIDTH - snake.position.x) / SCREEN_WIDTH,  # 到x边界的距离
            min(snake.position.y, SCREEN_HEIGHT - snake.position.y) / SCREEN_HEIGHT # 到y边界的距离
        ]
        
        # 对手信息
        opponent_states = []
        for opponent in self.opponents:
            if opponent.state != 'dead':
                # 相对位置
                rel_x = (opponent.position.x - snake.position.x) / SCREEN_WIDTH
                rel_y = (opponent.position.y - snake.position.y) / SCREEN_HEIGHT
                
                # 相对速度（方向）
                rel_dx = opponent.direction.x - snake.direction.x
                rel_dy = opponent.direction.y - snake.direction.y
                
                # 相对大小（长度比）
                rel_size = len(opponent.body) / len(snake.body) if len(snake.body) > 0 else 1
                
                opponent_states.extend([rel_x, rel_y, rel_dx, rel_dy, rel_size])
            else:
                opponent_states.extend([0, 0, 0, 0, 0])
        
        return np.array(base_state + opponent_states, dtype=np.float32)

    def _get_reward(self):
        """计算奖励"""
        if not self.player or self.player.state == 'dead':
            return -10.0  # 死亡惩罚

        reward = 0.0
        current_length = len(self.player.body)

        # 1. 吃到食物奖励
        length_diff = current_length - self._last_length
        if length_diff > 0:
            reward += 10.0  # 吃到食物奖励增加
            self.steps_without_food = 0
        else:
            self.steps_without_food += 1

        # 2. 生存奖励
        reward += 0.005  # 微弱正向奖励，鼓励存活

        # 3. 击杀奖励和危险规避
        for opponent in self.opponents:
            opp_id = id(opponent)

            if opponent.state == 'dead' and opp_id not in self._dead_opponents:
                # 击杀奖励，且需要距离足够近（击杀时）
                if opp_id in self._last_distances and self._last_distances[opp_id] < 100:
                    reward += 15.0
                    self.kills += 1
                self._dead_opponents.add(opp_id)
            elif opponent.state != 'dead':
                # 更新对手距离记录
                distance = (opponent.position - self.player.position).mag()
                self._last_distances[opp_id] = distance

                # 根据相对大小计算奖励或惩罚
                size_ratio = len(self.player.body) / len(opponent.body)

                if size_ratio > 1.2:  # 当前蛇较大
                    if distance < 100:  # 接近较小对手
                        reward += 0.1
                elif size_ratio < 0.8:  # 当前蛇较小
                    if distance < 100:  # 进入危险区域
                        reward -= 0.1
                        self.near_death_count += 1
                    elif 100 < distance < 200:  # 成功逃离危险区域
                        reward += 0.05

        # 4. 距离奖励（接近食物给予正奖励）
        nearest_food = min(
            self.game.food,
            key=lambda f: (f.position - self.player.position).mag_squared(),
            default=None
        )
        if nearest_food:
            current_dist = (nearest_food.position - self.player.position).mag()
            if not hasattr(self, 'last_food_distance'):
                self.last_food_distance = current_dist

            # 奖励接近食物的行为
            dist_reward = (self.last_food_distance - current_dist) * 0.01
            reward += dist_reward
            self.last_food_distance = current_dist

        # 5. 边界惩罚
        border_distance = min(
            self.player.position.x,
            SCREEN_WIDTH - self.player.position.x,
            self.player.position.y,
            SCREEN_HEIGHT - self.player.position.y
        )
        if border_distance < 50:  # 距离边界太近时给予惩罚
            reward -= 0.1

        # 6. 长时间不吃食物的惩罚
        if self.steps_without_food > 500:
            penalty = min((self.steps_without_food - 500) * 0.0005, 0.05)
            reward -= penalty

        # 更新当前长度记录
        self._last_length = current_length
        self.max_length = max(self.max_length, current_length)

        return reward

    def step(self, action):
        """环境步进"""
        self.episode_steps += 1
        
        # 更新AI对手
        game_state = {
            'food': self.game.food,
            'snakes': self.game.snakes
        }
        
        for opponent in self.opponents:
            if opponent.state != 'dead':
                opponent.get_action(game_state)
        
        # 检查是否超过最大步数
        if self.episode_steps >= self.max_episode_steps:
            return (self._get_state(), 0, True, {
                'length': len(self.player.body) if self.player else 0,
                'max_length': self.max_length,
                'steps': self.max_episode_steps,
                'kills': self.kills,
                'near_death_count': self.near_death_count
            })
        
        # 游戏更新
        self.game.update(action)
        
        # 获取新状态和奖励
        new_state = self._get_state()
        reward = self._get_reward()
        
        # 检查游戏是否结束
        done = (self.player.state == 'dead' or 
                self.episode_steps >= self.max_episode_steps)
        
        # 准备信息字典
        info = {
            'length': len(self.player.body) if self.player else 0,
            'max_length': self.max_length,
            'steps': self.episode_steps,
            'kills': self.kills,
            'near_death_count': self.near_death_count,
            'alive_opponents': sum(1 for opp in self.opponents if opp.state != 'dead')
        }
        
        return new_state, reward, done, info

    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            self.game.render()
            pygame.display.update()

    def close(self):
        """关闭环境"""
        pygame.quit()