import game
from mcts import MCTSNode

def is_in_range(position, shape):
    if len(position) != len(shape):
        return False
    for p, s in zip(position, shape):
        if p<0 or p>=s:
            return False
    return True

def list_add(position, delta, scale = 1):
    t = type(position)
    return t(p + d*scale for p, d in zip(position, delta))

def _format(x):
    x = str(x)
    return "{:>2}".format(x)

def print_board(state):
    h, w = state.shape
    print("  " + "".join(_format(i) for i in range(w)))
    for i in range(h):
        row = [_format(i)]
        for j in range(w):
            if state.board[0, i, j]:
                c = "O"
            elif state.board[1, i, j]:
                c = "X"
            else:
                c = "·"
            row.append(_format(c))
        print("".join(row))

def play(agentBlack, agentWhite, vizualize=False):
    state = game.State.init()
    agents = [agentBlack, agentWhite]
    mctsNodes = [MCTSNode(state)] * 2 if agentBlack is agentWhite else [MCTSNode(state), MCTSNode(state)]

    experience = []
    end = False
    i = 0
    while not end:
        turn = i%2
        agent = agents[turn]
        mctsNode = mctsNodes[turn]
        action = agent.get_action(mctsNode)
        mcts_policy = agent.get_mcts_policy(mctsNode)
        mctsNodes[0] = mctsNodes[0].next(action)
        mctsNodes[1] = mctsNodes[1].next(action)
    
        experience.append([state, mcts_policy])
        state = game.transition(state, action)
        if vizualize:
            print_board(state)
        end = state.isEnd
        i += 1
    z = state.endResult
    return experience, z

import asyncio

async def async_play(agentBlack, agentWhite, vizualize=False):
    state = game.State.init()
    agents = [agentBlack, agentWhite]
    mctsNodes = [MCTSNode(state)] * 2 if agentBlack is agentWhite else [MCTSNode(state), MCTSNode(state)]

    experience = []
    end = False
    i = 0
    while not end:
        turn = i%2
        agent = agents[turn]
        mctsNode = mctsNodes[turn]
        if asyncio.iscoroutinefunction(agent.get_action):
            action = await agent.get_action(mctsNode)
        else:
            action = agent.get_action(mctsNode)

        mcts_policy = agent.get_mcts_policy(mctsNode)
        mctsNodes[0] = mctsNodes[0].next(action)
        mctsNodes[1] = mctsNodes[1].next(action)
    
        experience.append([state, mcts_policy])
        state = game.transition(state, action)
        if vizualize:
            print_board(state)
        end = state.isEnd
        i += 1
    
    for agent in agents:
        if asyncio.iscoroutinefunction(agent.end):
            await agent.end(state)
        else:
            agent.end(state) 
    z = state.endResult
    return experience, z