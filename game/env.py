from __future__ import annotations

import os
from typing import Any

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from game.constants import (
    BIRD_X_POSITION,
    FPS,
    PIPE_GAP,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from game.bird import Bird
from game.pipe import PipeManager
from game.ground import Ground
from game.score import Score


def _get_next_pipe(pipe_manager: PipeManager, bird_x: float):
    for pipe in pipe_manager.pipes:
        if pipe.x + pipe.width >= bird_x:
            return pipe
    return None


class FlappyBirdEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        reward_pass: float = 1.0,
        reward_death: float = -10.0,
        reward_alive: float = 0.0,
        reward_per_step: float = 0.0,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.reward_pass = reward_pass
        self.reward_death = reward_death
        self.reward_alive = reward_alive
        self.reward_per_step = reward_per_step
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._dt = 1000 // FPS
        self._bird: Bird | None = None
        self._pipe_manager: PipeManager | None = None
        self._ground: Ground | None = None
        self._score: Score | None = None
        self._pygame_initialized = False

    def _init_pygame(self) -> None:
        if self._pygame_initialized:
            return
        if self.render_mode is None:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self._pygame_initialized = True
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird RL")
        elif self.render_mode == "rgb_array":
            self._screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self._clock = pygame.time.Clock()

    def _create_objects(self) -> None:
        self._bird = Bird(BIRD_X_POSITION, SCREEN_HEIGHT // 2)
        self._pipe_manager = PipeManager()
        self._ground = Ground(SCREEN_HEIGHT - 100)
        self._score = Score()

    def _get_obs(self) -> np.ndarray:
        if self._bird is None or self._pipe_manager is None:
            return np.zeros(5, dtype=np.float32)
        bird = self._bird
        pipe_manager = self._pipe_manager
        y_norm = float(np.clip(bird.y / SCREEN_HEIGHT, 0.0, 1.0))
        vel_norm = float(np.clip((bird.velocity + 10.0) / 20.0, 0.0, 1.0))
        next_pipe = _get_next_pipe(pipe_manager, bird.x)
        if next_pipe is None:
            pipe_dist = 1.0
            gap_center_norm = 0.5
            bird_to_gap = 0.0
        else:
            pipe_dist = float(
                np.clip((next_pipe.x - bird.x) / SCREEN_WIDTH, 0.0, 2.0)
            )
            gap_center = next_pipe.gap_y + PIPE_GAP / 2.0
            gap_center_norm = gap_center / SCREEN_HEIGHT
            bird_to_gap = (gap_center - bird.y) / SCREEN_HEIGHT
            bird_to_gap = float(np.clip(bird_to_gap, -1.0, 1.0))
        return np.array(
            [y_norm, vel_norm, pipe_dist, gap_center_norm, bird_to_gap],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {"score": self._score.score if self._score else 0}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._init_pygame()
        self._create_objects()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._bird is None or self._pipe_manager is None or self._ground is None or self._score is None:
            raise RuntimeError("Env not initialized. Call reset() first.")
        vel_before = self._bird.velocity
        y_before = self._bird.y / SCREEN_HEIGHT
        if action == 1:
            self._bird.flap()
        self._bird.update()
        self._pipe_manager.update(self._dt)
        self._ground.update()
        score_inc = self._pipe_manager.check_score(self._bird)
        if score_inc > 0:
            for _ in range(score_inc):
                self._score.increment()
        collision = (
            self._pipe_manager.check_collisions(self._bird)
            or self._ground.collides_with(self._bird)
            or self._bird.y <= 0
        )
        if collision:
            reward = self.reward_death + self.reward_per_step
            terminated = True
        else:
            reward = self.reward_alive + self.reward_pass * score_inc + self.reward_per_step
            y_norm = self._bird.y / SCREEN_HEIGHT
            if y_norm < 0.25:
                reward -= 0.12
            elif y_norm > 0.82:
                reward -= 0.12
            if action == 1 and vel_before > 0:
                reward -= 0.1
            if action == 1 and y_before < 0.45:
                reward -= 0.14
            terminated = False
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, False, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        self._init_pygame()
        if self._screen is None:
            return None
        from game.constants import SKY_BLUE

        self._screen.fill(SKY_BLUE)
        if self._pipe_manager is not None:
            self._pipe_manager.draw(self._screen)
        if self._ground is not None:
            self._ground.draw(self._screen)
        if self._bird is not None:
            self._bird.draw(self._screen)
        if self.render_mode == "human":
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
        )

    def close(self) -> None:
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
        self._screen = None
        self._clock = None
