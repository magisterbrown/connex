import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
from decider.player import PlayRandomized
import torch

def alpha_loss(output, target):
    crocs_entorpy = F.cross_entropy(output[0],target[0])
    mse = F.mse_loss(output[1],target[1])
    return crocs_entorpy+mse
    
class Trainer:
    def __init__(self, net, save_folder):
        self.epochs = 5
        self.batch_size = 6#32
        self.batches = 5000
        self.net = net
        self.optim = torch.optim.Adam(self.net.parameters(), weight_decay=1e-4)
        self.loss_fn = alpha_loss
        self.save_folder = save_folder
        self.buffer = None
        
    def train(self):
        for i in range(self.batches):
            st = time.time()
            X_train, y_train = self.get_buffer()#self.collect_data(self.batch_size)
            self.net.train()
            losses = list()
            for j in range(self.epochs):
                self.optim.zero_grad()
                
                pred = self.net(X_train)
                loss = self.loss_fn(pred, y_train)

                loss.backward()
                losses.append(loss.item())
                
                self.optim.step()
            draws = torch.sum(y_train[1]==0).item()
            no_draws = torch.sum(y_train[1]!=0).item()
            print(f'{i} Avg loss {np.mean(losses)} Time: {time.time()-st} Draws {draws} NoDraws {no_draws}') 
            if(i%50==0):
                torch.save(self.net.state_dict(), f'{self.save_folder}/{i}_step.pth')

    
    def get_buffer(self):
        update_size = 8
        if self.buffer is None:
            buffer = []
            for i in tqdm(range(self.batch_size), desc='Loading buffer'):
                buffer.append(self.collect_data(update_size))
            self.buffer = tuple(map(torch.concat, zip(*buffer)))
            
        else:
            new_batch = self.collect_data(update_size)
            new_turns = new_batch[0].shape
            old_turns = self.buffer[0].shape
            if new_turns[0]>=old_turns[0]:
                self.buffer = new_batch
            else:
                reps = torch.full((old_turns[0],),False)
                indices = torch.randperm(old_turns[0])
                reps[indices[:new_turns[0]]]=True
                for b,nb in zip(self.buffer, new_batch):
                    b[reps]=nb
                
        return self.buffer[:2], self.buffer[2:]
    
    def collect_data(self, size):
        def collect_game():
            pl = PlayRandomized(40,self.net)
            return pl.play()
        res = list()
        for i in range(size):
            res.append(collect_game())
        allgames = tuple(map(torch.concat, zip(*res)))
        return allgames
