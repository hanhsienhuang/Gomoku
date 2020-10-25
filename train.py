import game
from agent import Agent, BaselineAgent0
from mcts import MCTSNode
import numpy as np
import utils
import time

num_epoch = 100000
num_episode = 20
num_testing = 20
num_mcts = 500

agent = Agent(num_mcts = num_mcts, board_size=game.DEFAULT_SHAPE)
best_agent = Agent(num_mcts = num_mcts, board_size=game.DEFAULT_SHAPE)
best_agent.copy(agent)
baseline = BaselineAgent0()

def get_selfplay_data(agent, num_episode):
    experiences = []
    for i in range(num_episode):
        experience, z = utils.play(agent, agent)
        for i, e in enumerate(experience):
            experiences.append(e + [z if i % 2 == 0 else -z])
    return experiences

def get_interplay_data(agent1, agent2, num_episode):
    experiences = []
    for i in range(num_episode//2):
        experience, z = utils.play(agent1, agent2)
        for i, e in enumerate(experience):
            experiences.append(e + [z if i % 2 == 0 else -z])

        experience, z = utils.play(agent2, agent1)
        for i, e in enumerate(experience):
            experiences.append(e + [z if i % 2 == 0 else -z])

    return experiences

def get_win_rate(agent1, agent2, num):
    sum_z = 0
    for i in range(num//2):
        _, z = utils.play(agent1, agent2)
        sum_z += z
        _, z = utils.play(agent2, agent1)
        sum_z -= z
    if num%2 == 1:
        if np.random.random() > 0.5:
            _, z = utils.play(agent1, agent2)
            sum_z += z
        else:
            _, z = utils.play(agent2, agent1)
            sum_z -= z
    return sum_z / num / 2 + 0.5


for i in range(num_epoch):
    # self play
    print("Epoch: {}".format(i))
    t1 = time.time()
    #data = get_selfplay_data(best_agent, num_episode)
    data = get_interplay_data(agent, baseline, num_episode)
    t2 = time.time()
    print(f"Data collected, {len(data)} steps, {t2-t1:.2f} sec")

    # update policy 
    agent.update(data)
    t3 = time.time()
    print(f"Updated. {t3-t2:.2f} sec")

    # evaluation
    winRate = get_win_rate(agent, best_agent, num_testing)
    utils.play(agent, best_agent, True)
    t4 = time.time()
    print(f"Win rate over best: {winRate*100:.1f}%. {t4-t3:.2f} sec")
    if winRate > 0.55:
        best_agent.copy(agent)
    winRate = get_win_rate(agent, baseline, num_testing)
    utils.play(agent, baseline, True)
    t5 = time.time()
    print(f"Win rate over baseline: {winRate*100:.1f}%. {t5-t4:.2f} sec")
