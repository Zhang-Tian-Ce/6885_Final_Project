import math

# 屏幕设置
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
TEXT = (5, 5)
FPS = 60

# 蛇的基本参数
BASE_SIZE = 10          # 基础大小
EYE_SIZE = 4
PUPIL_SIZE = EYE_SIZE - 2
BASE_SPEED = 3         # 基础速度
MIN_DISTANCE = 1       # 身体段之间的最小距离
MAX_DISTANCE = 4       # 身体段之间的最大距离
SIZE_INC = 15          # 大小增长系数
EYE_INC = SIZE_INC * 4
PUPIL_INC = SIZE_INC * 8
GROWTH_INC = SIZE_INC * 20

# 加速机制参数
BOOST_MIN = 10         # 最小可加速长度
BOOST_FACTOR = 2       # 加速倍率
BOOST_DCR = 5         # 加速消耗率
BOOST_OFFSET = 40     # 加速时的颜色偏移

# 食物参数
FOOD_RADIUS = 5
FOOD_INIT = 50        # 降低初始食物数量
FOOD_RESPAWN_RATE = 0.8
FOOD_COLOUR = (240, 40, 40)

# 颜色定义
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)    # 蛇的颜色
WHITE = (220, 220, 220)
GREEN = (30, 180, 30)  # 文字颜色

# 渲染参数
PING_PONG = 100      # 颜色渐变周期

# 向量类
class Vector:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    @staticmethod
    def t(vector):
        return Vector(vector[0], vector[1])

    def tuple(self):
        return round(self.x), round(self.y)

    def copy(self):
        return Vector(self.x, self.y)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __mul__(self, other: float):
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other: float):
        return Vector(self.x / other, self.y / other)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def mag_squared(self):
        return self.x ** 2 + self.y ** 2

    def mag(self):
        return math.sqrt(self.mag_squared())

    def normalized(self):
        mag = self.mag()
        return Vector(self.x / mag, self.y / mag) if mag > 0 else Vector()

    def normalize(self):
        mag = self.mag()
        if mag > 0:
            self.x /= mag
            self.y /= mag
        return self

    def perpendicular(self, first=True):
        return Vector(-self.y if first else self.y,
                     self.x if first else -self.x).normalized()

    def lerp(self, target, distance, gap=0):
        direction = target - self
        mag = direction.mag()

        if gap > 0:
            mag -= gap
            direction.normalize()
            direction *= mag

        if mag <= 0:
            return 0, Vector()
        elif mag < distance:
            self.x += direction.x
            self.y += direction.y
            return mag, direction
        else:
            direction *= distance / mag
            self.x += direction.x
            self.y += direction.y
            return distance, direction

    def __str__(self):
        return f'({self.x}, {self.y})'

    __repr__ = __str__

# 圆形类（用于食物）
class Circle:
    def __init__(self, x=0.0, y=0.0, radius=1, position: Vector = None,
                 colour=FOOD_COLOUR):
        self.position = position if position else Vector(x, y)
        self.radius = radius
        self.colour = colour

    def __str__(self):
        return f'Circle(position={self.position}, radius={self.radius})'

    __repr__ = __str__

# 起始位置
START = Vector(BASE_SIZE * 2, SCREEN_HEIGHT // 2)