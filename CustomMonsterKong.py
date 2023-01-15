from ple.games.flappybird import FlappyBird
from ple.games.monsterkong import MonsterKong
from ple import PLE
import numpy as np
import cv2
import gym
from gym import spaces


class CustomMonsterKong(gym.Env):
    def __init__(self, force_fps=True):
        super(CustomMonsterKong, self).__init__()
        self.game = MonsterKong()
        self.p = PLE(self.game, fps=30, display_screen=True, force_fps=force_fps)
        self.p.init()
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(500, 465, 3), dtype=np.uint8
        )
        self.reward_range = (-25, 50)
        self.action_set = [None,97, 100, 32, 119, 115]

    def _step(self, action):
        if self.p.game_over():  # check if the game is over
            self.p.reset_game()
        high_level_obs = self.p.getScreenRGB()
        obs = high_level_obs
        act_action = self.action_set[action]

        reward = self.p.act(act_action)
        return obs, reward, self.p.game_over(), {}

    def get_action_meanings(self):
        return ["NOOP","left","right","jump", "up","down"]

    def step(self, action):
        return self._step(action)

    def _reset(self):
        self.p.reset_game()
        return self.p.getScreenRGB()

    def reset(self):
        return self._reset()

    def render(self, mode="human", close=False):
        return self.p.getScreenRGB()

    def _seed(self, seed=None):
        return [0]

    def _close(self):
        return

    def _configure(self):
        return
