import numpy as np
import game

class MCTSNode:
    def __init__(self, state):
        self.state = state
        if state.isEnd:
            return
        self.children = {}
        self.N = np.zeros(state.shape, dtype=int)
        self.P = None
        self.Q = np.zeros(state.shape, dtype=float)
        self.W = np.zeros(state.shape, dtype=float)
        self.n = 0

    def next(self, move):
        if self.state.isEnd:
            return self
        if move not in self.children:
            self.children[move] = MCTSNode(game.transition(self.state, move))
        return self.children[move]

    def update(self, move, value):
        self.N[move] += 1
        self.W[move] += value
        self.Q[move] = self.W[move] / self.N[move]
        self.n += 1

