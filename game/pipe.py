import random
import pygame

from game.constants import (
    PIPE_GAP,
    PIPE_SPEED,
    PIPE_SPAWN_INTERVAL,
    PIPE_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from game.bird import Bird


class Pipe:
    def __init__(self, x: float) -> None:
        self.x = x
        self.width = PIPE_WIDTH
        self.speed = PIPE_SPEED
        min_gap_y = 150
        max_gap_y = SCREEN_HEIGHT - PIPE_GAP - 150
        self.gap_y = random.randint(min_gap_y, max_gap_y)
        self.top_pipe = pygame.Rect(self.x, 0, self.width, self.gap_y)
        self.bottom_pipe = pygame.Rect(
            self.x,
            self.gap_y + PIPE_GAP,
            self.width,
            SCREEN_HEIGHT - (self.gap_y + PIPE_GAP),
        )
        self.passed = False
        self.pipe_color = (34, 139, 34)
        self.pipe_border_color = (25, 100, 25)
        self.highlight_color = (50, 180, 50)

    def update(self) -> None:
        self.x -= self.speed
        self.top_pipe.x = self.x
        self.bottom_pipe.x = self.x

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, self.pipe_color, self.top_pipe)
        pygame.draw.rect(screen, self.pipe_border_color, self.top_pipe, 3)
        highlight_rect_top = pygame.Rect(self.x, 0, 8, self.gap_y)
        pygame.draw.rect(screen, self.highlight_color, highlight_rect_top)

        pygame.draw.rect(screen, self.pipe_color, self.bottom_pipe)
        pygame.draw.rect(screen, self.pipe_border_color, self.bottom_pipe, 3)
        highlight_rect_bottom = pygame.Rect(
            self.x,
            self.gap_y + PIPE_GAP,
            8,
            SCREEN_HEIGHT - (self.gap_y + PIPE_GAP),
        )
        pygame.draw.rect(screen, self.highlight_color, highlight_rect_bottom)

        cap_height = 30
        cap_extra_width = 10
        top_cap = pygame.Rect(
            self.x - cap_extra_width // 2,
            self.gap_y - cap_height,
            self.width + cap_extra_width,
            cap_height,
        )
        pygame.draw.rect(screen, self.pipe_color, top_cap)
        pygame.draw.rect(screen, self.pipe_border_color, top_cap, 3)

        bottom_cap = pygame.Rect(
            self.x - cap_extra_width // 2,
            self.gap_y + PIPE_GAP,
            self.width + cap_extra_width,
            cap_height,
        )
        pygame.draw.rect(screen, self.pipe_color, bottom_cap)
        pygame.draw.rect(screen, self.pipe_border_color, bottom_cap, 3)

    def is_off_screen(self) -> bool:
        return self.x + self.width < 0

    def collides_with(self, bird: Bird) -> bool:
        bird_rect = bird.get_rect()
        if bird_rect.colliderect(self.top_pipe):
            return True
        if bird_rect.colliderect(self.bottom_pipe):
            return True
        return False

    def get_rects(self) -> tuple[pygame.Rect, pygame.Rect]:
        return (self.top_pipe, self.bottom_pipe)

    def bird_passed(self, bird: Bird) -> bool:
        if self.passed:
            return False
        if bird.x > self.x + self.width:
            self.passed = True
            return True
        return False


class PipeManager:
    def __init__(self) -> None:
        self.pipes: list[Pipe] = []
        self.spawn_timer = 0
        self.spawn_interval = PIPE_SPAWN_INTERVAL

    def update(self, dt: int) -> None:
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_pipe()
            self.spawn_timer = 0
        for pipe in self.pipes:
            pipe.update()
        self.pipes = [pipe for pipe in self.pipes if not pipe.is_off_screen()]

    def spawn_pipe(self) -> None:
        new_pipe = Pipe(SCREEN_WIDTH)
        self.pipes.append(new_pipe)

    def draw(self, screen: pygame.Surface) -> None:
        for pipe in self.pipes:
            pipe.draw(screen)

    def check_collisions(self, bird: Bird) -> bool:
        for pipe in self.pipes:
            if pipe.collides_with(bird):
                return True
        return False

    def check_score(self, bird: Bird) -> int:
        score_increment = 0
        for pipe in self.pipes:
            if pipe.bird_passed(bird):
                score_increment += 1
        return score_increment

    def reset(self) -> None:
        self.pipes = []
        self.spawn_timer = 0
