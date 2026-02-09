import pygame

from game.constants import (
    BIRD_SIZE,
    FLAP_STRENGTH,
    GRAVITY,
    SCREEN_HEIGHT,
    BLACK,
    WHITE,
)


class Bird:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.velocity = 0.0
        self.gravity = GRAVITY
        self.flap_strength = FLAP_STRENGTH
        self.width = BIRD_SIZE
        self.height = BIRD_SIZE
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill((255, 255, 0))
        pygame.draw.circle(self.image, BLACK, (24, 12), 4)
        pygame.draw.circle(self.image, WHITE, (24, 12), 2)
        self.rotation = 0
        self.max_rotation_up = 25
        self.max_rotation_down = -90
        self.rotation_velocity = 3

    def flap(self) -> None:
        self.velocity = self.flap_strength
        self.rotation = self.max_rotation_up

    def update(self) -> None:
        self.velocity += self.gravity
        if self.velocity > 10:
            self.velocity = 10
        self.y += self.velocity
        if self.velocity < 0:
            self.rotation = self.max_rotation_up
        else:
            if self.rotation > self.max_rotation_down:
                self.rotation -= self.rotation_velocity
        if self.y < 0:
            self.y = 0
            self.velocity = 0

    def draw(self, screen: pygame.Surface) -> None:
        rotated_image = pygame.transform.rotate(self.image, self.rotation)
        rotated_rect = rotated_image.get_rect(center=(self.x, self.y))
        screen.blit(rotated_image, rotated_rect)

    def get_rect(self) -> pygame.Rect:
        hitbox_width = self.width * 0.8
        hitbox_height = self.height * 0.8
        return pygame.Rect(
            self.x - hitbox_width / 2,
            self.y - hitbox_height / 2,
            hitbox_width,
            hitbox_height,
        )

    def reset(self) -> None:
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.rotation = 0
