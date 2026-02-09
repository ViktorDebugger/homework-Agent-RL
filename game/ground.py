import random
import pygame

from game.constants import GROUND_SPEED, SCREEN_WIDTH
from game.bird import Bird


class Ground:
    def __init__(self, y: float) -> None:
        self.y = y
        self.height = 100
        self.speed = GROUND_SPEED
        self.width = SCREEN_WIDTH
        self.x1 = 0.0
        self.x2 = float(SCREEN_WIDTH)
        self.image = self._create_ground_image()

    def _create_ground_image(self) -> pygame.Surface:
        ground_surface = pygame.Surface((self.width, self.height))
        base_color = (222, 216, 149)
        ground_surface.fill(base_color)
        grass_color = (87, 166, 57)
        pygame.draw.rect(ground_surface, grass_color, (0, 0, self.width, 20))
        dark_grass = (70, 140, 45)
        pygame.draw.rect(ground_surface, dark_grass, (0, 0, self.width, 4))
        dirt_color = (200, 190, 120)
        for i in range(0, self.width, 40):
            for j in range(25, self.height, 30):
                pygame.draw.rect(
                    ground_surface,
                    dirt_color,
                    (
                        i + random.randint(-5, 5),
                        j + random.randint(-3, 3),
                        random.randint(15, 25),
                        random.randint(8, 15),
                    ),
                )
        stone_color = (180, 170, 110)
        for _ in range(30):
            x = random.randint(0, self.width)
            y = random.randint(25, self.height - 10)
            radius = random.randint(2, 5)
            pygame.draw.circle(ground_surface, stone_color, (x, y), radius)
        return ground_surface

    def update(self) -> None:
        self.x1 -= self.speed
        self.x2 -= self.speed
        if self.x1 <= -self.width:
            self.x1 = self.x2 + self.width
        if self.x2 <= -self.width:
            self.x2 = self.x1 + self.width

    def draw(self, screen: pygame.Surface) -> None:
        screen.blit(self.image, (self.x1, self.y))
        screen.blit(self.image, (self.x2, self.y))

    def collides_with(self, bird: Bird) -> bool:
        bird_rect = bird.get_rect()
        return bird_rect.bottom >= self.y

    def reset(self) -> None:
        self.x1 = 0
        self.x2 = SCREEN_WIDTH
