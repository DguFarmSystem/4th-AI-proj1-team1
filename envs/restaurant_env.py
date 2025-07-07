# Restaurant Recommendation Environment 구현 파일 
import gym
import numpy as np
from gym import spaces

class RestaurantRecEnv(gym.Env):
    """
    상태(state): 현재 user id
    행동(action): 추천할 restaurant index
    보상(reward): 클릭 여부 (1 or 0)
    """
    def __init__(self, n_users, n_items):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.action_space = spaces.Discrete(n_items)
        self.observation_space = spaces.Discrete(n_users)
        self.current_user = None

    def reset(self, user_id=None):
        self.current_user = user_id if user_id is not None else np.random.randint(0, self.n_users)
        return self.current_user

    def step(self, action, reward):
        done = True
        return self.current_user, reward, done, {} 