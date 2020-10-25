import networks
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import is_in_range, list_add, print_board
import random

def choice2d(p):
    i = np.random.choice(np.prod(p.shape), p = p.flatten())
    return np.unravel_index(i, p.shape)

class Agent:
    def __init__(self, num_mcts, board_size):
        self.num_mcts = num_mcts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = networks.NN(in_channels = 3,
            feature_size = 50,
            num_residual = 1,
            board_size = board_size,
            value_hidden_size = 256).to(self.device)
        #self.network = networks.ConvNN(in_channels = 3,
        #    feature_size = 64,
        #    num_layer = 2,
        #    value_hidden_size = 64).to(self.device)
        self.optim = optim.Adam(self.network.parameters(), 1e-4, weight_decay=1e-4)
        self.temperature = 1 # TODO
        self.UCBConstant = 1
        self.batchSize = 100

    def get_action(self, treeNode):
        while treeNode.n < self.num_mcts:
            self._dfs(treeNode)
        policy = self.get_mcts_policy(treeNode, 0.1)
        return choice2d(policy)

    def _dfs(self, treeNode):
        if treeNode.state.isEnd:
            return -abs(treeNode.state.endResult)
        if treeNode.P is None:
            p, v = self.get_network_policy_and_value(treeNode)
            treeNode.P = p
            return v
        ucb = self.get_mcts_ucb_value(treeNode) 
        move = np.unravel_index(np.argmax(ucb), ucb.shape)
        v = -self._dfs(treeNode.next(move))
        treeNode.update(move, v)
        return v

    def get_network_policy_and_value(self, treeNode):
        inp = self._construct_tensors((treeNode.state, ))
        policy_logit, value = self.network(inp)
        policy_logit = policy_logit.cpu().detach().numpy()[0]
        policy_logit = np.where(treeNode.state.validMoves, policy_logit, float("-inf"))
        exp_logit = np.exp(policy_logit - policy_logit.max())
        policy = exp_logit / exp_logit.sum()
        value = value.item()
        return policy, value

    def get_mcts_ucb_value(self, treeNode):
        # Q + U
        # U = c P \sqrt{n} / (1+N)
        # For invalid move, value = -inf
        U = np.where(treeNode.state.validMoves, \
            self.UCBConstant * np.sqrt(treeNode.n + 1) * treeNode.P / (treeNode.N+1), \
            float("-inf"))
        return treeNode.Q + U  

    def get_mcts_policy(self, treeNode, temperature = None):
        if temperature is None:
            temperature = self.temperature
        poweredN = (treeNode.N / np.max(treeNode.N)) ** (1/temperature if temperature > 0 else float("inf"))
        return poweredN / poweredN.sum()

    def update(self, experience):
        #random.shuffle(experience)
        states, policies, zs = list(zip(*experience))
        inputs = self._construct_tensors(states)
        valids = torch.stack([torch.as_tensor(s.validMoves, dtype=int, device=self.device) for s in states])
        policies = torch.stack([torch.as_tensor(p, dtype=torch.float, device=self.device) for p in policies])
        zs = torch.tensor(zs, dtype=torch.float, device=self.device).unsqueeze(-1) 
        neginf = torch.tensor(-1e8, dtype=torch.float, device=self.device)

        # Rotate and flip
        inputs = torch.cat([inputs, torch.flip(inputs, [1])])
        inputs = torch.cat([torch.rot90(inputs, i, (2,3)) for i in range(4)])
        valids = torch.cat([valids, torch.flip(valids, [1])])
        valids = torch.cat([torch.rot90(valids, i, (1,2)) for i in range(4)])
        policies = torch.cat([policies, torch.flip(policies, [1])])
        policies = torch.cat([torch.rot90(policies, i, (1,2)) for i in range(4)])
        zs = zs.repeat(8,1)

        # shuffle
        index = torch.randperm(valids.shape[0])
        inputs = inputs[index]
        valids = valids.to(dtype=bool)[index]
        policies = policies[index]
        zs = zs[index]

        total_loss = 0
        num_it = 0
        for i in range(inputs.shape[0] //self.batchSize ):
            beg, end = i*self.batchSize, (i+1)*self.batchSize
            inp = inputs[beg:end]
            val = valids[beg:end]
            pol = policies[beg:end]
            z = zs[beg:end]

            policy_logit, value = self.network(inp)
            policy_logit = torch.where(val, policy_logit, neginf)
            loss = - torch.mean(torch.sum(pol * policy_logit, dim=(1,2))) \
                + torch.mean(torch.logsumexp(policy_logit, dim=(1,2))) \
                + torch.mean((value - z)**2)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.item()
            num_it += 1
        print(f"num iter: {num_it}, mean loss: {total_loss/num_it:.2f}")

    def copy(self, other):
        if self is not other:
            self.network.load_state_dict(other.network.state_dict())

    def _construct_tensors(self, states):
        tensors = [self._construct_tensor(state) for state in states]
        return torch.stack(tensors)

    def _construct_tensor(self, state):
        board = torch.as_tensor(state.board, dtype=torch.float, device=self.device)
        isBlack = torch.full((1,) + state.shape, int(state.isNextBlack), dtype=torch.float, device=self.device)
        tensor = torch.cat((board, isBlack))
        return tensor.to(self.device)
        
    
class BaselineAgent0:
    def __init__(self):
        pass

    def get_action(self, treeNode):
        state = treeNode.state
        if not np.any(state.board[0]):
            action = tuple(s//2 for s in state.shape)
            treeNode.update(action, 1)
            return action
        i = int(state.isNextBlack)
        myBoard = state.board[i]
        oppoBoard = state.board[(i+1)%2]
        scores = np.zeros(state.shape, dtype=int)
        # 1,2,4,100
        mapScores = [0, 1, 4, 8, 1000, 0]
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                for d in [(1,0), (0,1), (1,1), (1, -1)]:
                    hasFive, n = self._check_five(myBoard, oppoBoard, (i,j), d)
                    if hasFive:
                        score = mapScores[n]
                        for step in range(5):
                            pos = list_add((i,j), d, step)
                            if state.validMoves[pos]:
                                scores[pos] += score
                    hasFive, n = self._check_five(oppoBoard, myBoard, (i,j), d)
                    if hasFive:
                        score = mapScores[n]
                        for step in range(5):
                            pos = list_add((i,j), d, step)
                            if state.validMoves[pos]:
                                scores[pos] += score
        maxScore = np.max(scores)
        indices = list(zip(*np.where(np.logical_and(scores == maxScore, state.validMoves))))
        
        action = random.choice(indices)
        treeNode.update(action, 1)
        return action

    def end(self, state):
        pass

    def _check_five(self, board, oppoboard, start, direction):
        num = 0
        for step in range(5):
            pos = list_add(start, direction, step)
            if not is_in_range(pos, board.shape):
                return False, None
            if oppoboard[pos]:
                return False, None
            if board[pos]:
                num += 1
        return True, num
        

    def get_mcts_policy(self, treeNode):
        return treeNode.N / np.sum(treeNode.N)

class InputAgent:
    def __init__(self):
        pass

    def get_action(self, treeNode):
        state = treeNode.state
        print_board(state)
        while True:
            inp = input("Input position: ")
            try:
                lst = inp.split(",")
                assert(len(lst)==2)
                pos = tuple(int(l) for l in lst)
                break
            except:
                print()
        return pos

    def get_mcts_policy(self, treeNode):
        return None

    def end(self, state):
        pass

import asyncio
import json

class WebsocketAgent:
    def __init__(self, ws):
        self.ws = ws

    def state_to_json(self, state):
        shape = state.shape
        board = [[0]*shape[1] for _ in range(shape[0])]
        validMoves = [[0]*shape[1] for _ in range(shape[0])]
        for i in range(shape[0]):
            for j in range(shape[1]):
                if state.board[0, i, j]:
                    board[i][j] = 1
                elif state.board[1, i, j]:
                    board[i][j] = -1
                validMoves[i][j] = 1 if state.validMoves[i, j] else 0

        s = dict(
            shape = shape,
            board = board,
            isNextBlack = state.isNextBlack,
            validMoves = validMoves,
            isEnd = state.isEnd,
            endResult = state.endResult,
        )
        return json.dumps(s)

    async def get_action(self, treeNode):
        await self.ws.send(self.state_to_json(treeNode.state))
        rec = await self.ws.recv()
        action = tuple(json.loads(rec))
        print(action)
        return action

    def get_mcts_policy(self, treeNode):
        return None
    
    async def end(self, state):
        await self.ws.send(self.state_to_json(state))
        await self.ws.close()

