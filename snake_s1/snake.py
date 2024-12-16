# snake.py 的内容
import pygame
import random
import math
from constants import *

class Snake:
    def __init__(self, start: Vector = START):
        # 初始化身体，从起始点开始生成3个连续的段
        base_direction = Vector(-MIN_DISTANCE, 0)  # 初始向左
        self.body = []
        for i in range(3):
            pos = start.copy() + base_direction * i
            self.body.append(pos)

        # 基本属性
        self.speed_factor = 1.0
        self.colour = BLUE
        self.direction = Vector()
        self.target = start.copy()
        self.state = 'main'

        # 加速相关属性
        self.boost_cooldown = 0
        self.boost_energy = 100

    @property
    def speed(self):
        """获取当前速度"""
        return BASE_SPEED * self.speed_factor

    @property
    def radius(self):
        """获取当前半径"""
        return BASE_SIZE + len(self.body) // SIZE_INC

    @property
    def position(self):
        """获取头部位置"""
        return self.body[0]

    def gradient(self, i):
        """计算蛇身体颜色渐变"""
        boost = BOOST_OFFSET if self.speed_factor == BOOST_FACTOR else 0
        if i // PING_PONG % 2 == 1:
            i = PING_PONG - i % PING_PONG
        else:
            i %= PING_PONG
        r = min(max(self.colour[0] + i + boost, 0), 255)
        g = min(max(self.colour[1] + i + boost, 0), 255)
        b = min(max(self.colour[2] + i + boost, 0), 255)
        return r, g, b

    def render(self, screen):
        """渲染蛇"""
        if self.state == 'dead':
            return

        # 渲染蛇身
        for i, vector in list(enumerate(self.body))[::-1]:
            pygame.draw.circle(screen, self.gradient(i),
                             (round(vector.x), round(vector.y)),
                             self.radius)

        # 渲染眼睛
        direct = self.position - (self.body[1] if len(self.body) > 1 else self.target)
        for b in (True, False):
            offset = direct.perpendicular(b) * (self.radius // 2)
            eye_pos = (self.position + offset).tuple()
            pygame.draw.circle(screen, WHITE, eye_pos,
                             EYE_SIZE + len(self.body) // EYE_INC)
            pygame.draw.circle(screen, BLACK, eye_pos,
                             PUPIL_SIZE + len(self.body) // PUPIL_INC)

        # 渲染加速能量条
        if self.speed_factor > 1.0:
            energy_width = 30
            energy_height = 4
            energy_x = round(self.position.x) - energy_width // 2
            energy_y = round(self.position.y) - self.radius - 10

            pygame.draw.rect(screen, (100, 100, 100),
                           (energy_x, energy_y, energy_width, energy_height))
            energy_level = self.boost_energy / 100
            pygame.draw.rect(screen, (0, 255, 0),
                           (energy_x, energy_y, energy_width * energy_level, energy_height))

    def move(self):
        """移动蛇"""
        if self.state == 'dead':
            return

        # 处理加速状态
        if self.state == 'boost':
            if self.boost_energy > 0:
                if pygame.time.get_ticks() % BOOST_DCR == 0:
                    if len(self.body) > 1:  # 确保至少保留一个身体段
                        self.body.pop()
                    self.boost_energy -= 1
                self.speed_factor = BOOST_FACTOR
            else:
                self.state = 'main'
                self.boost_cooldown = 50
        else:
            self.speed_factor = 1.0
            if self.boost_cooldown > 0:
                self.boost_cooldown -= 1
            if self.boost_energy < 100:
                self.boost_energy += 0.2

        # 计算移动
        moved, self.direction = self.body[0].lerp(self.target, self.speed)

        # 更新身体各段位置
        for i in range(1, len(self.body)):
            _, _ = self.body[i].lerp(self.body[i-1], moved, MAX_DISTANCE)

    def grow(self):
        """增长身体"""
        if self.state == 'dead':
            return

        growth_amount = 1
        for _ in range(growth_amount * (len(self.body) // GROWTH_INC + 1)):
            self.body.append(self.body[-1] - self.direction.normalized() * MIN_DISTANCE)

    def boost(self):
        """启动加速"""
        if self.state == 'dead' or self.boost_cooldown > 0:
            return

        if len(self.body) > BOOST_MIN and self.boost_energy > 20:
            self.state = 'boost'

    def collide_circle(self, other):
        """检测与圆形（食物）的碰撞"""
        min_length = self.radius + other.radius
        difference = (self.position - other.position).mag_squared()
        return min_length ** 2 >= difference

    def die(self):
        """死亡处理"""
        if self.state == 'dead':
            return
        self.state = 'dead'