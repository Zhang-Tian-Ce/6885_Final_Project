# simple_ai_snake.py

import random
import math
import numpy as np
from snake import Snake
from constants import Vector, SCREEN_WIDTH, SCREEN_HEIGHT

class SimpleAISnake(Snake):
    """简单的AI控制蛇，只追踪食物和避开墙壁"""
    def __init__(self, start: Vector = None):
        if start is None:
            start = Vector(
                random.randint(50, SCREEN_WIDTH-50),
                random.randint(50, SCREEN_HEIGHT-50)
            )
        super().__init__(start)
        
    def get_action(self, game_state):
        """生成动作：追踪最近的食物，同时避开墙壁"""
        # 检查是否太靠近墙壁
        border_distance = min(
            self.position.x,
            SCREEN_WIDTH - self.position.x,
            self.position.y,
            SCREEN_HEIGHT - self.position.y
        )
        
        if border_distance < 30:  # 如果太靠近墙壁，转向中心
            target_angle = math.atan2(
                SCREEN_HEIGHT/2 - self.position.y,
                SCREEN_WIDTH/2 - self.position.x
            )
        else:  # 正常追踪食物
            # 找到最近的食物
            nearest_food = None
            min_dist = float('inf')
            
            for food in game_state['food']:
                dist = (food.position - self.position).mag()
                if dist < min_dist:
                    min_dist = dist
                    nearest_food = food
                    
            if nearest_food:
                # 计算追踪角度
                target_angle = math.atan2(
                    nearest_food.position.y - self.position.y,
                    nearest_food.position.x - self.position.x
                )
            else:
                # 如果没有食物，随机移动
                target_angle = random.uniform(0, 2 * math.pi)
        
        # 根据角度设置目标位置
        distance = 10  # 设置一个固定的目标距离
        target_x = self.position.x + math.cos(target_angle) * distance
        target_y = self.position.y + math.sin(target_angle) * distance
        
        # 设置目标位置
        self.target = Vector(target_x, target_y)