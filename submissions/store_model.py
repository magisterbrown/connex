import pickle
import base64
from nets.basic import ConnNet

save_file = 'alpha_boss/submission.py'
net = ConnNet(cols=7,rows=6)
weights = base64.encodebytes(pickle.dumps(net.state_dict())).decode()
variable=f"""weights=b'''{weights}'''

dec = base64.decodebytes(weights)
unpick = pickle.loads(dec)
model.load_state_dict(unpick)
model.eval()

def alpha_boss(observation, configuration):
    return model.play(observation.board)
"""

with open(save_file, 'a') as f:
    f.write(variable)
