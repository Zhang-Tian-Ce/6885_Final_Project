import pygame
from pygame.locals import *
import random
import math
import numpy as np
from snake import Snake  # 添加这行
from constants import *

pygame.init()

def new_food():
    """生成新的食物对象"""
    return Circle(random.randint(0, SCREEN_WIDTH-1), random.randint(0, SCREEN_HEIGHT-1), FOOD_RADIUS)

class Game:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Slither.io')

        # 立即初始化字体
        pygame.font.init()
        self._font = pygame.font.SysFont('arial', 24)

        self.snake = None  # 确保蛇的引用被初始化
        self.snakes = []
        self.food = []
        self.state = 'play'

        self.reset()

    def reset(self):
        """重置游戏状态"""
        # 初始化基本属性
        start_pos = Vector(
            random.randint(BASE_SIZE*2, SCREEN_WIDTH-BASE_SIZE*2),
            random.randint(BASE_SIZE*2, SCREEN_HEIGHT-BASE_SIZE*2)
        )
        self.snake = Snake(start_pos)
        self.snakes = [self.snake]
        self.food = []
        self.state = 'play'

        # 初始化食物
        for _ in range(FOOD_INIT):
            self.spawn_food()

        return self._get_state()

    def spawn_food(self):
        attempts = 0
        while attempts < 20:  # 增加尝试次数
            new_pos = Vector(
                random.randint(BASE_SIZE*2, SCREEN_WIDTH-BASE_SIZE*2),
                random.randint(BASE_SIZE*2, SCREEN_HEIGHT-BASE_SIZE*2)
            )

            # 降低食物生成的限制条件
            too_close = False
            for existing_food in self.food:
                if (existing_food.position - new_pos).mag() < FOOD_RADIUS * 2:  # 减小距离要求
                    too_close = True
                    break

            if not too_close:
                self.food.append(Circle(new_pos.x, new_pos.y, FOOD_RADIUS))
                break
            attempts += 1

    def check_collision(self):
        """检查碰撞"""
        # 检查边界碰撞
        if (self.snake.position.x < 0 or
            self.snake.position.x > SCREEN_WIDTH or
            self.snake.position.y < 0 or
            self.snake.position.y > SCREEN_HEIGHT):
            return True
        return False

    def update(self, action=None):
        """更新游戏状态"""
        if self.state == 'end':
            return self.state

        if action is not None:
            angle = (action[0] + 1) * math.pi
            distance = 10

            # 计算目标位置
            target_x = self.snake.position.x + math.cos(angle) * distance
            target_y = self.snake.position.y + math.sin(angle) * distance

            self.snake.target = Vector(target_x, target_y)

        # 更新蛇的移动
        self.snake.move()

        # 检查边界碰撞
        if self.check_collision():
            self.state = 'end'
            return self.state

        # 检查食物收集
        for food in self.food[:]:
            if self.snake.collide_circle(food):
                self.snake.grow()
                self.food.remove(food)
                if random.random() < FOOD_RESPAWN_RATE:
                    self.spawn_food()

        # 维持食物数量
        min_food_count = FOOD_INIT // 3
        while len(self.food) < min_food_count:
            self.spawn_food()

        return self.state

    def render(self):
        self.screen.fill(BLACK)

        # 渲染食物
        for food in self.food:
            pygame.draw.circle(self.screen, food.colour,
                             (food.position.x, food.position.y), food.radius)

        # 渲染蛇
        for snake in self.snakes[::-1]:
            snake.render(self.screen)

        # 渲染长度信息
        if self.state == 'play' and self.snake:
            length_text = f'Length: {len(self.snake.body)}'
            self.screen.blit(self._font.render(length_text, True, GREEN), TEXT)

    def get_length(self):
        """获取当前长度"""
        return len(self.snake.body) if self.snake else 0

    def _get_state(self):
        """获取状态表示"""
        if not self.snake or self.state == 'end':
            return np.zeros(6, dtype=np.float32)

        state = [
            self.snake.position.x / SCREEN_WIDTH,
            self.snake.position.y / SCREEN_HEIGHT,
            self.snake.direction.x,
            self.snake.direction.y
        ]

        # 找到最近的食物
        nearest_food = min(self.food,
                          key=lambda f: (f.position - self.snake.position).mag_squared(),
                          default=None)

        if nearest_food:
            dx = (nearest_food.position.x - self.snake.position.x) / SCREEN_WIDTH
            dy = (nearest_food.position.y - self.snake.position.y) / SCREEN_HEIGHT
        else:
            dx, dy = 0, 0

        state.extend([dx, dy])

        return np.array(state, dtype=np.float32)