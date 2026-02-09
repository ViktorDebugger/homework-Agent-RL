import pygame

from game.constants import SCREEN_HEIGHT, SCREEN_WIDTH, WHITE, BLACK


class Score:
    def __init__(self) -> None:
        self.score = 0
        self.high_score = 0
        self.font_large = pygame.font.Font(None, 80)
        self.font_medium = pygame.font.Font(None, 40)
        self.font_small = pygame.font.Font(None, 30)
        self.text_color = WHITE
        self.shadow_color = (50, 50, 50)

    def increment(self) -> None:
        self.score += 1
        if self.score > self.high_score:
            self.high_score = self.score

    def reset(self) -> None:
        self.score = 0

    def draw(self, screen: pygame.Surface, game_state: str = "PLAYING") -> None:
        if game_state == "PLAYING":
            self._draw_text_with_shadow(
                screen,
                str(self.score),
                self.font_large,
                SCREEN_WIDTH // 2,
                50,
            )
        elif game_state == "GAME_OVER":
            self._draw_game_over_scores(screen)

    def _draw_text_with_shadow(
        self,
        screen: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        x: int,
        y: int,
    ) -> None:
        shadow_surface = font.render(text, True, self.shadow_color)
        shadow_rect = shadow_surface.get_rect(center=(x + 3, y + 3))
        screen.blit(shadow_surface, shadow_rect)
        text_surface = font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect(center=(x, y))
        screen.blit(text_surface, text_rect)

    def _draw_game_over_scores(self, screen: pygame.Surface) -> None:
        center_x = SCREEN_WIDTH // 2
        self._draw_text_with_shadow(
            screen,
            "GAME OVER",
            self.font_large,
            center_x,
            SCREEN_HEIGHT // 3,
        )
        box_width = 300
        box_height = 150
        box_x = center_x - box_width // 2
        box_y = SCREEN_HEIGHT // 2 - 50
        box_surface = pygame.Surface((box_width, box_height))
        box_surface.set_alpha(200)
        box_surface.fill((50, 50, 50))
        screen.blit(box_surface, (box_x, box_y))
        pygame.draw.rect(screen, WHITE, (box_x, box_y, box_width, box_height), 3)
        self._draw_text_with_shadow(
            screen,
            f"Score: {self.score}",
            self.font_medium,
            center_x,
            box_y + 40,
        )
        self._draw_text_with_shadow(
            screen,
            f"Best: {self.high_score}",
            self.font_small,
            center_x,
            box_y + 90,
        )
        self._draw_text_with_shadow(
            screen,
            "Press SPACE to restart",
            self.font_small,
            center_x,
            SCREEN_HEIGHT - 80,
        )
