import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pygame

from game.constants import (
    BIRD_X_POSITION,
    FPS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SKY_BLUE,
    WHITE,
    BLACK,
)
from game.bird import Bird
from game.pipe import PipeManager
from game.ground import Ground
from game.score import Score


class Game:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird Clone")
        self.clock = pygame.time.Clock()
        self.game_state = "MENU"
        self.running = True
        self.bird = Bird(BIRD_X_POSITION, SCREEN_HEIGHT // 2)
        self.pipe_manager = PipeManager()
        self.ground = Ground(SCREEN_HEIGHT - 100)
        self.score = Score()
        self.title_font = pygame.font.Font(None, 70)
        self.menu_font = pygame.font.Font(None, 35)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_state == "MENU":
                        self.start_game()
                    elif self.game_state == "PLAYING":
                        self.bird.flap()
                    elif self.game_state == "GAME_OVER":
                        self.restart_game()
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.game_state == "MENU":
                        self.start_game()
                    elif self.game_state == "PLAYING":
                        self.bird.flap()
                    elif self.game_state == "GAME_OVER":
                        self.restart_game()

    def start_game(self) -> None:
        self.game_state = "PLAYING"
        self.bird.reset()
        self.pipe_manager.reset()
        self.ground.reset()
        self.score.reset()

    def restart_game(self) -> None:
        self.game_state = "PLAYING"
        self.bird.reset()
        self.pipe_manager.reset()
        self.ground.reset()
        self.score.reset()

    def update(self) -> None:
        if self.game_state != "PLAYING":
            return
        dt = self.clock.get_time()
        self.bird.update()
        self.pipe_manager.update(dt)
        self.ground.update()
        score_increment = self.pipe_manager.check_score(self.bird)
        if score_increment > 0:
            self.score.increment()
        if self.check_collisions():
            self.game_state = "GAME_OVER"

    def check_collisions(self) -> bool:
        if self.pipe_manager.check_collisions(self.bird):
            return True
        if self.ground.collides_with(self.bird):
            return True
        if self.bird.y <= 0:
            return True
        return False

    def draw(self) -> None:
        self.screen.fill(SKY_BLUE)
        if self.game_state == "MENU":
            self.draw_menu()
        else:
            self.pipe_manager.draw(self.screen)
            self.ground.draw(self.screen)
            self.bird.draw(self.screen)
            self.score.draw(self.screen, self.game_state)
        pygame.display.flip()

    def draw_menu(self) -> None:
        title_text = self.title_font.render("FLAPPY BIRD", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        title_shadow = self.title_font.render("FLAPPY BIRD", True, BLACK)
        shadow_rect = title_shadow.get_rect(
            center=(SCREEN_WIDTH // 2 + 3, SCREEN_HEIGHT // 3 + 3)
        )
        self.screen.blit(title_shadow, shadow_rect)
        self.screen.blit(title_text, title_rect)
        bird_demo = Bird(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        bird_demo.y += 10 * (pygame.time.get_ticks() % 1000) / 1000 - 5
        bird_demo.draw(self.screen)
        start_text = self.menu_font.render(
            "Press SPACE or CLICK to start", True, WHITE
        )
        start_rect = start_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT * 2 // 3)
        )
        self.screen.blit(start_text, start_rect)
        controls_text = self.menu_font.render("SPACE / CLICK = Flap", True, WHITE)
        controls_rect = controls_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT * 3 // 4)
        )
        self.screen.blit(controls_text, controls_rect)
        if self.score.high_score > 0:
            high_score_text = self.menu_font.render(
                f"Best: {self.score.high_score}", True, (255, 215, 0)
            )
            high_score_rect = high_score_text.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50)
            )
            self.screen.blit(high_score_text, high_score_rect)

    def run(self) -> None:
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()


def main() -> None:
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
