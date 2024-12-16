import pygame
from pygame.locals import *
import random
import math
import numpy as np
from snake import Snake
from constants import *

pygame.init()

class Game:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Slither.io')

        pygame.font.init()
        self._font = pygame.font.SysFont('arial', 24)

        self.snake = None
        self.snakes = []
        self.food = []
        self.state = 'play'
        self.reset()

    def reset(self):
        """重置游戏状态"""
        start_pos = Vector(
            random.randint(BASE_SIZE * 2, SCREEN_WIDTH - BASE_SIZE * 2),
            random.randint(BASE_SIZE * 2, SCREEN_HEIGHT - BASE_SIZE * 2)
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
        """生成新的食物"""
        attempts = 0
        while attempts < 20:
            new_pos = Vector(
                random.randint(BASE_SIZE * 2, SCREEN_WIDTH - BASE_SIZE * 2),
                random.randint(BASE_SIZE * 2, SCREEN_HEIGHT - BASE_SIZE * 2)
            )

            too_close = False
            for existing_food in self.food:
                if (existing_food.position - new_pos).mag() < FOOD_RADIUS * 3:
                    too_close = True
                    break

            if not too_close:
                self.food.append(Circle(new_pos.x, new_pos.y, FOOD_RADIUS, 
                                     colour=FOOD_COLOUR, is_death_food=False))
                break
            attempts += 1

    def _handle_snake_death(self, dead_snake):
        """处理蛇的死亡，将其转化为食物"""
        if dead_snake.state == 'dead':
            return  # 确保只处理一次死亡逻辑

        dead_snake.die()  # 标记蛇为死亡状态

        # 将死亡的蛇转换为食物
        food_count = dead_snake.food_count
        if food_count > 0:
            food_positions = self._generate_death_food_positions(dead_snake.position, food_count)
            
            # 添加死亡食物，并标记它们
            for pos in food_positions:
                self.food.append(Circle(pos.x, pos.y, FOOD_RADIUS, 
                                     colour=FOOD_COLOUR, is_death_food=True))

    def _generate_death_food_positions(self, center: Vector, count: int):
        """生成死亡时的食物位置"""
        positions = []
        radius = 30  # 食物分散的半径
        
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius)
            
            x = center.x + math.cos(angle) * distance
            y = center.y + math.sin(angle) * distance
            
            # 确保食物在边界内
            x = max(FOOD_RADIUS, min(SCREEN_WIDTH - FOOD_RADIUS, x))
            y = max(FOOD_RADIUS, min(SCREEN_HEIGHT - FOOD_RADIUS, y))
            
            positions.append(Vector(x, y))
        
        return positions

    def check_collision(self):
        """检查碰撞并处理死亡"""
        # 检查所有蛇之间的碰撞
        for i, snake1 in enumerate(self.snakes):
            if snake1.state == 'dead':
                continue

            # 检查边界碰撞
            if (snake1.position.x < 0 or
                snake1.position.x > SCREEN_WIDTH or
                snake1.position.y < 0 or
                snake1.position.y > SCREEN_HEIGHT):
                self._handle_snake_death(snake1)
                continue

            # 检查与其他蛇的碰撞
            for j, snake2 in enumerate(self.snakes):
                if i != j and snake2.state != 'dead':
                    # 计算两蛇头部的距离
                    head_distance = (snake1.position - snake2.position).mag()

                    # 如果发生碰撞
                    if head_distance < (snake1.radius + snake2.radius):
                        self._handle_snake_death(snake1)
                        break

                    # 检查头部与身体的碰撞
                    for segment in snake2.body[1:]:
                        if (snake1.position - segment).mag() < (snake1.radius + snake2.radius):
                            self._handle_snake_death(snake1)
                            break

        # 判断主蛇是否死亡
        return self.snake.state == 'dead'

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

        # 更新所有蛇的移动
        for snake in self.snakes:
            if snake.state != 'dead':
                snake.move()
            else:
                # 确保死蛇不再移动
                snake.target = snake.position.copy()

        # 检查碰撞和处理死亡
        if self.check_collision():
            self.state = 'end'
            return self.state

        # 检查食物收集（对所有蛇）
        for snake in self.snakes:
            if snake.state == 'dead':
                continue

            for food in self.food[:]:
                if snake.collide_circle(food):
                    snake.grow()
                    self.food.remove(food)
                    # 只有非死亡食物才会触发重生
                    if not food.is_death_food and random.random() < FOOD_RESPAWN_RATE:
                        self.spawn_food()

        return self.state

    def render(self):
        """渲染游戏画面"""
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