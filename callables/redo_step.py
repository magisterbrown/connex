import numpy as np
from kaggle_environments import evaluate, make, utils
env = make("connectx", debug=True)
env.step([0,0],[''])
res = env.render(mode="ansi")
print(res)
