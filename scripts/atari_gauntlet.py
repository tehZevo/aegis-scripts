import gym
from gym import spaces
import retro
import random
import numpy as np

#TODO: move to separate repo
#TODO: optional padding would require a variable observation_space shape

class AtariGauntlet(gym.Env):
  def __init__(self, step_limit=None, obs_type=retro.Observations.IMAGE, allowed_games=None, debug=False):
    """allowed_games should be a list of games with "-Atari2600" removed"""
    super().__init__()
    self.step_limit = step_limit
    self.steps = 0
    self.obs_type = obs_type
    self.game = None
    self.debug = debug

    self.games = AtariGauntlet.get_games()
    if allowed_games is not None:
      self.games = [g for g in self.games if g.replace("-Atari2600", "") in allowed_games]

    if self.obs_type == retro.Observations.IMAGE:
      #TODO: assumption on low/high
      self.observation_space = spaces.Box(shape=[250, 160, 3], low=0, high=255, dtype=np.uint8)
    else:
      self.observation_space = spaces.Box(shape=[128], low=0, high=1, dtype=np.uint8)

    self.action_space = spaces.Discrete(18)

  def step(self, action):
    state, reward, done, info = self.game.step(action)

    if self.obs_type == retro.Observations.IMAGE:
      new_state = np.zeros([250, 160])
      new_state[:state.shape[0], :state.shape[1]] = state

    self.steps += 1

    if self.step_limit is not None and self.steps >= self.step_limit:
      if self.debug:
        print("Game over (step limit reached)")
      done = True
    elif done and self.debug:
        print("Game over ({} steps)".format(self.steps))

    return state, reward, done, info

  def render(self, **kwargs):
    self.game.render(**kwargs)

  def get_games():
    games = retro.data.list_games()
    games = list(filter(lambda x: "atari2600" in x.lower(), games))
    return games

  def reset(self):
    #close old emulator
    if self.game is not None:
      self.game.render(close=True) #close window?
      self.game.close()

    self.game_name = random.choice(self.games)
    if self.debug:
      print("Starting game:", self.game_name)

    self.game = retro.make(
      self.game_name,
      obs_type=self.obs_type,
      use_restricted_actions=retro.Actions.DISCRETE)

    self.steps = 0
    return self.game.reset()

#print list of atari games
if __name__ == "__main__":
  games = AtariGauntlet.get_games()

  for game in games:
    env = retro.make(game, obs_type=retro.Observations.IMAGE,
      use_restricted_actions=retro.Actions.DISCRETE)
    print(game)
    print(env.observation_space, "->", env.action_space)
    print()
    env.close()
