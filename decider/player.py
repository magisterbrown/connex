from kaggle_environments import evaluate, make, utils
from decider.mcts import MCTS
import numpy as np
import torch

class Play:
    def __init__(self, palyouts: int, net):
        self.palyouts = palyouts
        net.eval()
        env = make("connectx", debug=False, configuration={'rows':net.rows,'columns': net.cols,'inarow':3})
        self.mcts = MCTS(net, env)
    
    def play(self):
        """
        Returns tuple of (Boards, player, mcts probs, winner)
        """
        game_is_live = True
        res = ([], [], [])
        while game_is_live:
            for i in range(self.palyouts):
                self.mcts.playout()
            list(stack.append(el) for stack,el in zip(res,self.mcts.get_state()))
            
            probs = self.mcts.get_move_probs()
            step = self.choose_move(probs)
            game_is_live = self.mcts.make_move(step)
            
        res = tuple(map(torch.concat,res))
        turns = self.mcts._root.get_player()
        rewards = torch.empty_like(res[1])
        rewards[res[1]==1] = self.mcts.env.state[1]['reward']
        rewards[res[1]==2] = self.mcts.env.state[0]['reward']
        return res + (rewards.to(dtype=torch.float32),)
    
    @staticmethod        
    def choose_move(probs:dict):
        return max(probs,key=probs.get)
    
class PlayRandomized(Play):
    
    @staticmethod        
    def choose_move(probs:dict):
        prvl = np.array(list(probs.values()))
        randomized_probs = 0.75*prvl + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
        return int(np.random.choice(
                    list(probs.keys()),
                    p=randomized_probs
                ))
