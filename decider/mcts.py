import numpy as np
import torch
import scipy
import copy

class TreeNode:
    count=0
    def __init__(self, parent, prob, player=False):
        self.children = dict()
        self.Q = 0
        self._visit_count = 0
        self._P = prob
        self.parent = parent
        self.ct = TreeNode.count
        self.player = player
        TreeNode.count+=1
        
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_player(self):
        return 1+int(self.player)
    
    def get_next(self):
        nst = max(self.children, key=self.children.get)
        return [nst,nst], self.children[nst]
    
    def get_value(self):
        if self.parent is not None:
            pv = np.sqrt(self.parent._visit_count)
        else:
            pv = 1
        return self.Q+(pv*self._P)/(1+self._visit_count)
    
    def __lt__(self, other):
        return self.get_value() < other.get_value()
    
    def expand(self, probs):
        for sc, prob in enumerate(probs):
            if prob>0:
                self.children[sc] = TreeNode(self,prob,player=not self.player)
    
    def update(self,value: int):
        self._visit_count+=1
        self.Q+=(value - self.Q) / self._visit_count
        
        if self.parent is not None:
            self.parent.update(-1*value)
    
class MCTS:
    def __init__(self, model, env):
        self.model = model
        self._root = TreeNode(None, 0)
        self.env = env
    
    def playout(self):
        env = copy.deepcopy(self.env)
        
        node = self._root
        while(not node.is_leaf()):
            step, node = node.get_next()
            env.step(step)
        board = env.state[0].observation.board
        probs, value = self.model.get_probs(board, mark=node.get_player())
        if env.done:
            value = env.state[1-node.player]['reward']
        else:
            node.expand(probs)
        
        node.update(value)
    
    def get_state(self, temp=1e-3):
        board = self.env.state[0].observation.board
        board = self.model.convert(board,self._root.get_player())
        player = torch.tensor([[self._root.get_player()]])
        move_probs = self.get_move_probs(temp)
        ind = torch.tensor(list(move_probs.keys()))
        prb = torch.tensor(list(move_probs.values()),dtype=torch.float32)
        mcts_probs = torch.zeros((1,self.model.cols),dtype=torch.float32)
        mcts_probs[0,ind] = prb
        return board, player, mcts_probs
    
    def make_move(self, mv: int) -> bool:
        self.env.step([mv,mv])
        if mv in self._root.children:
            self._root = self._root.children[mv]
            self._root.parent = None
        else:
            self._root = TreeNode(None, 0)
        
        return not self.env.done
            
    def get_move_probs(self, temp=1e-3):
        nodes = self._root.children.values()
        moves = self._root.children.keys()
        visits = [n._visit_count for n in nodes]
        act_probs = scipy.special.softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        
        return dict(zip(moves, act_probs))
