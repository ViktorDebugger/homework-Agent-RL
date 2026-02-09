from game.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FPS,
)
from game.bird import Bird
from game.pipe import Pipe, PipeManager
from game.ground import Ground
from game.score import Score
from game.game import Game, main

__all__ = [
    "SCREEN_WIDTH",
    "SCREEN_HEIGHT",
    "FPS",
    "Bird",
    "Pipe",
    "PipeManager",
    "Ground",
    "Score",
    "Game",
    "main",
]

try:
    from game.env import FlappyBirdEnv
    __all__.append("FlappyBirdEnv")
except ImportError:
    pass
