from nets.basic import ConnNet
from train.trainer import Trainer

net = ConnNet(cols=5,rows=4)     
tr = Trainer(net, 'points/test4x5_too')
tr.train()

