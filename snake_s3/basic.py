import pygame
from pygame.locals import *
import random
import sys

pygame.init()

WIDTH = 40
HEIGHT = 40
TILE_WIDTH = 20
TILE_HEIGHT = 20
SCREEN_WIDTH = WIDTH * TILE_WIDTH
SCREEN_HEIGHT = HEIGHT * TILE_HEIGHT
START = (2, HEIGHT // 2)
TEXT = (5, 5)

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

DIRECTIONS = {K_UP: (0, -1), K_RIGHT: (1, 0), K_DOWN: (0, 1), K_LEFT: (-1, 0)}


class Game:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Snek.exe')
        self.font = pygame.font.SysFont('arial', 45)

        self.direction = [K_RIGHT, K_LEFT]
        self.snake = [to_screen(*START)]
        self.food = self.new_food()

        self.state = 'start'
        self.score = 0

    def init_game(self):
        self.direction = [K_RIGHT, K_LEFT]
        self.snake = [to_screen(*START)]
        self.food = self.new_food()

        self.state = 'start'
        self.score = 0

    def run(self):
        while True:
            self.handle_input()

            if self.state == 'play':
                self.update()
            self.render()

            pygame.display.update()
            self.clock.tick(10)

    def update(self):
        future = move(self.snake[0], DIRECTIONS[self.direction[0]])
        if len(self.snake) > 1 and future == self.snake[1]:
            self.direction[0] = self.direction[1]
            future = move(self.snake[0], DIRECTIONS[self.direction[0]])

        if future == self.food:
            self.snake.insert(0, self.food)
            self.food = self.new_food()
            self.score += 1
        else:
            if future in self.snake or not Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT).contains(
                    Rect(future[0], future[1], TILE_WIDTH, TILE_HEIGHT)):
                self.state = 'end'

            self.snake.insert(0, future)
            self.snake.pop()

    def render(self):
        self.screen.fill(BLACK)
        for x, y in self.snake:
            pygame.draw.rect(self.screen, BLUE, Rect(x, y, TILE_WIDTH, TILE_HEIGHT))
        pygame.draw.rect(self.screen, RED, Rect(self.food[0], self.food[1], TILE_WIDTH, TILE_HEIGHT))

        if self.state == 'play':
            self.screen.blit(self.font.render(f'Score: {self.score}', 1, GREEN), TEXT)
        elif self.state == 'start':
            self.screen.blit(self.font.render('Press Any Key to Start...', 1, GREEN), TEXT)
        elif self.state == 'end':
            self.screen.blit(self.font.render('You Failed lol...', 1, RED), TEXT)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key in DIRECTIONS.keys():
                    self.direction[1] = self.direction[0]
                    self.direction[0] = event.key
                if self.state == 'start':
                    self.state = 'play'
                elif self.state == 'end':
                    self.init_game()
            elif event.type == QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit()

    def new_food(self):
        food = to_screen(random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1))
        while food in self.snake:
            food = to_screen(random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1))

        return food


def to_screen(x, y):
    return x * TILE_WIDTH, y * TILE_HEIGHT


def move(pos, vel):
    vel = to_screen(*vel)
    return pos[0] + vel[0], pos[1] + vel[1]


game = Game()
game.run()
