import gym
from gym.spaces import Discrete, Box
import numpy as np
import random

from ray.rllib.env.env_context import EnvContext


class MysteriousCorridor(gym.Env):
    """Example of a custom env in which you walk down a mysterious corridor.

    You can configure the reward of the destination state via the env config.

    A mysterious corridor has 7 cells and looks like this:
    -------------------------------------------
    |  1  |  2  |  3  |  4  |  3  |  5  |  6  |
    -------------------------------------------
    You always start from state 1 (left most) or 6 (right most).
    The goal is to get to the destination state 4 (in the middle).
    There are only 2 actions, 0 means go left, 1 means go right.

    The mysterious part about this corridor, as you may have seen, is that
    it is designed so that once you get to state 3 (either side of goal),
    you would not be able to tell whether you are to the left or right of state 4.

    Thus, any deterministic policy will result in us not being able to reach
    goal state for 50% of the time.
    We need a stochastic policy to handle this environment.
    """

    def __init__(self, config: EnvContext):
        self.seed(random.randint(0, 1000))

        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 6.0, shape=(1, ), dtype=np.float32)
        self.reward = config["reward"]

        self.reset()

    def reset(self):
        # cur_pos is the actual postion of the player. not the state a player
        # sees from outside of the environemtn.
        # E.g., when cur_pos is 1, the returned state is 3.
        # Start from either side of the corridor, 0 or 4.
        self.cur_pos = random.choice([0, 6])
        return [self.cur_pos]

    def _pos_to_state(self, pos):
        ptos = [1, 2, 3, 4, 3, 5, 6]
        return ptos[pos]

    def step(self, action):
        assert action in [0, 1], action

        if action == 0:
            self.cur_pos = max(0, self.cur_pos - 1)
        if action == 1:
            self.cur_pos = min(6, self.cur_pos + 1)

        done = (self.cur_pos == 3)
        reward = self.reward if done else -0.1

        return [self._pos_to_state(self.cur_pos)], reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)

    def render(self):
        def symbol(i):
            if i == self.cur_pos:
                return "o"
            elif i == 3:
                return "x"
            elif i == 2 or i == 4:
                return "_"
            else:
                return " "
        return "| " + " | ".join([symbol(i) for i in range(7)]) + " |"
