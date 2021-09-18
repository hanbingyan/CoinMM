# Theoretical limit with multi-agents and multi-actions on a/b side

import numpy as np
from scipy.optimize import fsolve
from itertools import product
from scipy.special import comb

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
DELTA = 0.0

n_agents = 2
tick_num = 4

space = np.linspace(0.1, 0.5, tick_num)
# space = np.zeros(tick_num)
# space[0] = 0.5
# space[1] = 0.9
# space[2] = 1.5


# weights = np.zeros(tick_num)
weights = np.linspace(0.0, 0.1, tick_num)

# weights[0] = 0.02
# weights[1] = 0.04
sig = 0.1
inven_factor = 0.1


def margin(action):
    price = space[action]
    kernel = np.sum(weights[action])
    kernel = kernel / sig / n_agents
    arrival_prob = np.exp(-kernel)
    winner = np.where(price == price.min())
    return arrival_prob, winner, price.min() / len(winner[0])


def expected_reward(ask_action, bid_action):
    # the first item in all actions is the player
    ask_prob, ask_winner, ask_reward = margin(ask_action)
    bid_prob, bid_winner, bid_reward = margin(bid_action)

    total_reward = np.zeros(n_agents)
    total_reward[ask_winner] += ask_reward * ask_prob
    total_reward[bid_winner] += bid_reward * bid_prob

    inven_risk = np.zeros(n_agents)
    # Case 1: both ask and bid orders arrived
    inventory = np.zeros(n_agents)
    inventory[ask_winner] -= 1 / len(ask_winner[0])
    inventory[bid_winner] += 1 / len(bid_winner[0])
    inven_risk += inven_factor * ask_prob * bid_prob * inventory ** 2

    # Case 2: only ask order arrived
    inventory = np.zeros(n_agents)
    inventory[ask_winner] -= 1 / len(ask_winner[0])
    # inventory[bid_winner] += 1 / len(bid_winner[0])
    inven_risk += inven_factor * ask_prob * (1 - bid_prob) * inventory ** 2

    # Case 3: only bid order arrived
    inventory = np.zeros(n_agents)
    # inventory[ask_winner] -= 1 / len(ask_winner[0])
    inventory[bid_winner] += 1 / len(bid_winner[0])
    inven_risk += inven_factor * (1 - ask_prob) * bid_prob * inventory ** 2

    total_reward -= inven_risk

    return total_reward[0]


def Boltz(x):
    #     prob = x/np.sum(x)
    prob = np.exp(x / temper)
    prob = prob / np.sum(prob)
    #     prob = np.ones(tick_num**2)/tick_num**2
    return prob

# Convert to the joint action vector, ask_tick and bid_tick are for agent 1
def gen_act(C, ask_tick, bid_tick):
    ind = np.cumsum(C)
    action = np.zeros(n_agents)
    for k in reversed(range(len(ind))):
        action[1:ind[k] + 1] = k
    action[0] = ask_tick * tick_num + bid_tick
    return action

# Enumerate all possible actions for other agents beside self
act_count = [np.array(val) for val in product(range(n_agents), repeat=tick_num ** 2) if sum(val) == (n_agents - 1)]

def F(Q):
    # Q is the Q function with shape tick_num**2
    Prob = Boltz(Q)
    # reward multiplied by multinomial probabilities
    ER = np.zeros(tick_num ** 2)
    for ask_tick in range(tick_num):
        for bid_tick in range(tick_num):
            # Loop over actions of other agents
            for k in range(len(act_count)):
                C = act_count[k]
                # temp is multinomial probability
                temp = comb(n_agents - 1, C[0]) * np.power(Prob[0], C[0])
                for ab in range(1, tick_num ** 2):
                    temp = temp * comb(n_agents - 1 - np.sum(C[:ab]), C[ab]) * np.power(Prob[ab], C[ab])
                action = gen_act(C, ask_tick, bid_tick)
                ask_action = (action // tick_num).astype(int)
                bid_action = (action % tick_num).astype(int)
                ER[ask_tick * tick_num + bid_tick] += temp * expected_reward(ask_action, bid_action)
    return ER + DELTA * np.max(Q) - Q

temper = 0.1
init = np.zeros(tick_num**2)
init[0] = 0.8
sol = fsolve(F, init)
print('Temperature', temper)
print('Q-values q*', sol)
print('Probability p*', Boltz(sol))
print('Residuals of contraction map', F(sol))
