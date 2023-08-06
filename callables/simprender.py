import numpy as np
from kaggle_environments import evaluate, make, utils
env = make("connectx", debug=True)

def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

env.run([my_agent, "random"])
rnd = env.render(mode="ansi")
print(rnd)
