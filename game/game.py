import pygame
import random
import sys

# ==================== CONSTANTS ====================

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
SKY_BLUE = (135, 206, 235)
BLACK = (0, 0, 0)

# Physics
GRAVITY = 0.5
FLAP_STRENGTH = -10

# Game settings
PIPE_GAP = 200  # Vertical space between top and bottom pipes
PIPE_SPEED = 3
PIPE_SPAWN_INTERVAL = 1500  # milliseconds
GROUND_SPEED = 3

# Bird settings
BIRD_SIZE = 34  # Width/height if using a square sprite
BIRD_X_POSITION = 80  # Fixed horizontal position

# Pipe settings
PIPE_WIDTH = 70

# FPS
FPS = 60


# ==================== BIRD CLASS ====================
# [Insert complete Bird class from Prompt 2 here]

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
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
        
    def flap(self):
        self.velocity = self.flap_strength
        self.rotation = self.max_rotation_up
        
    def update(self):
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
            
    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, self.rotation)
        rotated_rect = rotated_image.get_rect(center=(self.x, self.y))
        screen.blit(rotated_image, rotated_rect)
        
    def get_rect(self):
        hitbox_width = self.width * 0.8
        hitbox_height = self.height * 0.8
        return pygame.Rect(
            self.x - hitbox_width / 2,
            self.y - hitbox_height / 2,
            hitbox_width,
            hitbox_height
        )
    
    def reset(self):
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.rotation = 0


# ==================== PIPE CLASS ====================
# [Insert complete Pipe class from Prompt 3 here]

class Pipe:
    def __init__(self, x):
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
            SCREEN_HEIGHT - (self.gap_y + PIPE_GAP)
        )
        self.passed = False
        self.pipe_color = (34, 139, 34)
        self.pipe_border_color = (25, 100, 25)
        self.highlight_color = (50, 180, 50)
        
    def update(self):
        self.x -= self.speed
        self.top_pipe.x = self.x
        self.bottom_pipe.x = self.x
        
    def draw(self, screen):
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
            SCREEN_HEIGHT - (self.gap_y + PIPE_GAP)
        )
        pygame.draw.rect(screen, self.highlight_color, highlight_rect_bottom)
        
        cap_height = 30
        cap_extra_width = 10
        top_cap = pygame.Rect(
            self.x - cap_extra_width // 2,
            self.gap_y - cap_height,
            self.width + cap_extra_width,
            cap_height
        )
        pygame.draw.rect(screen, self.pipe_color, top_cap)
        pygame.draw.rect(screen, self.pipe_border_color, top_cap, 3)
        
        bottom_cap = pygame.Rect(
            self.x - cap_extra_width // 2,
            self.gap_y + PIPE_GAP,
            self.width + cap_extra_width,
            cap_height
        )
        pygame.draw.rect(screen, self.pipe_color, bottom_cap)
        pygame.draw.rect(screen, self.pipe_border_color, bottom_cap, 3)
        
    def is_off_screen(self):
        return self.x + self.width < 0
    
    def collides_with(self, bird):
        bird_rect = bird.get_rect()
        if bird_rect.colliderect(self.top_pipe):
            return True
        if bird_rect.colliderect(self.bottom_pipe):
            return True
        return False
    
    def get_rects(self):
        return (self.top_pipe, self.bottom_pipe)
    
    def bird_passed(self, bird):
        if self.passed:
            return False
        if bird.x > self.x + self.width:
            self.passed = True
            return True
        return False


# ==================== PIPE MANAGER CLASS ====================

class PipeManager:
    def __init__(self):
        self.pipes = []
        self.spawn_timer = 0
        self.spawn_interval = PIPE_SPAWN_INTERVAL
        
    def update(self, dt):
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_pipe()
            self.spawn_timer = 0
        for pipe in self.pipes:
            pipe.update()
        self.pipes = [pipe for pipe in self.pipes if not pipe.is_off_screen()]
    
    def spawn_pipe(self):
        new_pipe = Pipe(SCREEN_WIDTH)
        self.pipes.append(new_pipe)
    
    def draw(self, screen):
        for pipe in self.pipes:
            pipe.draw(screen)
    
    def check_collisions(self, bird):
        for pipe in self.pipes:
            if pipe.collides_with(bird):
                return True
        return False
    
    def check_score(self, bird):
        score_increment = 0
        for pipe in self.pipes:
            if pipe.bird_passed(bird):
                score_increment += 1
        return score_increment
    
    def reset(self):
        self.pipes = []
        self.spawn_timer = 0


# ==================== GROUND CLASS ====================
# [Insert complete Ground class from Prompt 4 here]

class Ground:
    def __init__(self, y):
        self.y = y
        self.height = 100
        self.speed = GROUND_SPEED
        self.width = SCREEN_WIDTH
        self.x1 = 0
        self.x2 = SCREEN_WIDTH
        self.image = self._create_ground_image()
        
    def _create_ground_image(self):
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
                pygame.draw.rect(ground_surface, dirt_color, 
                               (i + random.randint(-5, 5), 
                                j + random.randint(-3, 3), 
                                random.randint(15, 25), 
                                random.randint(8, 15)))
        stone_color = (180, 170, 110)
        for _ in range(30):
            x = random.randint(0, self.width)
            y = random.randint(25, self.height - 10)
            radius = random.randint(2, 5)
            pygame.draw.circle(ground_surface, stone_color, (x, y), radius)
        return ground_surface
    
    def update(self):
        self.x1 -= self.speed
        self.x2 -= self.speed
        if self.x1 <= -self.width:
            self.x1 = self.x2 + self.width
        if self.x2 <= -self.width:
            self.x2 = self.x1 + self.width
    
    def draw(self, screen):
        screen.blit(self.image, (self.x1, self.y))
        screen.blit(self.image, (self.x2, self.y))
    
    def collides_with(self, bird):
        bird_rect = bird.get_rect()
        return bird_rect.bottom >= self.y
    
    def reset(self):
        self.x1 = 0
        self.x2 = SCREEN_WIDTH


# ==================== SCORE CLASS ====================
# [Insert complete Score class from Prompt 4 here]

class Score:
    def __init__(self):
        self.score = 0
        self.high_score = 0
        self.font_large = pygame.font.Font(None, 80)
        self.font_medium = pygame.font.Font(None, 40)
        self.font_small = pygame.font.Font(None, 30)
        self.text_color = WHITE
        self.shadow_color = (50, 50, 50)
        
    def increment(self):
        self.score += 1
        if self.score > self.high_score:
            self.high_score = self.score
    
    def reset(self):
        self.score = 0
    
    def draw(self, screen, game_state="PLAYING"):
        if game_state == "PLAYING":
            self._draw_text_with_shadow(
                screen,
                str(self.score),
                self.font_large,
                SCREEN_WIDTH // 2,
                50
            )
        elif game_state == "GAME_OVER":
            self._draw_game_over_scores(screen)
    
    def _draw_text_with_shadow(self, screen, text, font, x, y):
        shadow_surface = font.render(text, True, self.shadow_color)
        shadow_rect = shadow_surface.get_rect(center=(x + 3, y + 3))
        screen.blit(shadow_surface, shadow_rect)
        text_surface = font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect(center=(x, y))
        screen.blit(text_surface, text_rect)
    
    def _draw_game_over_scores(self, screen):
        center_x = SCREEN_WIDTH // 2
        self._draw_text_with_shadow(
            screen,
            "GAME OVER",
            self.font_large,
            center_x,
            SCREEN_HEIGHT // 3
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
            box_y + 40
        )
        self._draw_text_with_shadow(
            screen,
            f"Best: {self.high_score}",
            self.font_small,
            center_x,
            box_y + 90
        )
        self._draw_text_with_shadow(
            screen,
            "Press SPACE to restart",
            self.font_small,
            center_x,
            SCREEN_HEIGHT - 80
        )


# ==================== GAME CLASS ====================

class Game:
    def __init__(self):
        """Initialize the game and all game objects."""
        # Initialize Pygame
        pygame.init()
        
        # Create screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird Clone")
        
        # Create clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Game state
        self.game_state = "MENU"  # MENU, PLAYING, GAME_OVER
        self.running = True
        
        # Create game objects
        self.bird = Bird(BIRD_X_POSITION, SCREEN_HEIGHT // 2)
        self.pipe_manager = PipeManager()
        self.ground = Ground(SCREEN_HEIGHT - 100)
        self.score = Score()
        
        # Fonts for menu
        self.title_font = pygame.font.Font(None, 70)
        self.menu_font = pygame.font.Font(None, 35)
        
    def handle_events(self):
        """Handle all pygame events."""
        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                self.running = False
                
            # Key press events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_state == "MENU":
                        # Start game
                        self.start_game()
                    elif self.game_state == "PLAYING":
                        # Bird flap
                        self.bird.flap()
                    elif self.game_state == "GAME_OVER":
                        # Restart game
                        self.restart_game()
                        
                # ESC key to quit
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    
            # Mouse click (alternative to space)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if self.game_state == "MENU":
                        self.start_game()
                    elif self.game_state == "PLAYING":
                        self.bird.flap()
                    elif self.game_state == "GAME_OVER":
                        self.restart_game()
    
    def start_game(self):
        """Start a new game from menu."""
        self.game_state = "PLAYING"
        self.bird.reset()
        self.pipe_manager.reset()
        self.ground.reset()
        self.score.reset()
        
    def restart_game(self):
        """Restart game after game over."""
        self.game_state = "PLAYING"
        self.bird.reset()
        self.pipe_manager.reset()
        self.ground.reset()
        self.score.reset()
    
    def update(self):
        """Update all game objects."""
        if self.game_state != "PLAYING":
            return
        
        # Get delta time in milliseconds
        dt = self.clock.get_time()
        
        # Update bird
        self.bird.update()
        
        # Update pipes
        self.pipe_manager.update(dt)
        
        # Update ground
        self.ground.update()
        
        # Check for scoring
        score_increment = self.pipe_manager.check_score(self.bird)
        if score_increment > 0:
            self.score.increment()
        
        # Check collisions
        if self.check_collisions():
            self.game_state = "GAME_OVER"
    
    def check_collisions(self):
        """
        Check all collision conditions.
        
        Returns:
            bool: True if any collision detected
        """
        # Check collision with pipes
        if self.pipe_manager.check_collisions(self.bird):
            return True
        
        # Check collision with ground
        if self.ground.collides_with(self.bird):
            return True
        
        # Check collision with ceiling
        if self.bird.y <= 0:
            return True
        
        return False
    
    def draw(self):
        """Draw all game elements."""
        # Draw sky background
        self.screen.fill(SKY_BLUE)
        
        if self.game_state == "MENU":
            self.draw_menu()
        else:
            # Draw pipes
            self.pipe_manager.draw(self.screen)
            
            # Draw ground
            self.ground.draw(self.screen)
            
            # Draw bird
            self.bird.draw(self.screen)
            
            # Draw score
            self.score.draw(self.screen, self.game_state)
        
        # Update display
        pygame.display.flip()
    
    def draw_menu(self):
        """Draw the main menu screen."""
        # Draw title
        title_text = self.title_font.render("FLAPPY BIRD", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        
        # Draw shadow for title
        title_shadow = self.title_font.render("FLAPPY BIRD", True, BLACK)
        shadow_rect = title_shadow.get_rect(center=(SCREEN_WIDTH // 2 + 3, SCREEN_HEIGHT // 3 + 3))
        self.screen.blit(title_shadow, shadow_rect)
        self.screen.blit(title_text, title_rect)
        
        # Draw bird in center (animated)
        bird_demo = Bird(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        bird_demo.y += 10 * (pygame.time.get_ticks() % 1000) / 1000 - 5  # Bobbing animation
        bird_demo.draw(self.screen)
        
        # Draw instructions
        start_text = self.menu_font.render("Press SPACE or CLICK to start", True, WHITE)
        start_rect = start_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT * 2 // 3))
        self.screen.blit(start_text, start_rect)
        
        # Draw controls
        controls_text = self.menu_font.render("SPACE / CLICK = Flap", True, WHITE)
        controls_rect = controls_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT * 3 // 4))
        self.screen.blit(controls_text, controls_rect)
        
        # Draw high score if available
        if self.score.high_score > 0:
            high_score_text = self.menu_font.render(f"Best: {self.score.high_score}", True, (255, 215, 0))
            high_score_rect = high_score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
            self.screen.blit(high_score_text, high_score_rect)
    
    def run(self):
        """Main game loop."""
        while self.running:
            # Handle events
            self.handle_events()
            
            # Update game state
            self.update()
            
            # Draw everything
            self.draw()
            
            # Cap frame rate at 60 FPS
            self.clock.tick(FPS)
        
        # Quit pygame
        pygame.quit()
        sys.exit()


# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point for the game."""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()