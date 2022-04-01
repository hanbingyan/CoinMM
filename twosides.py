# Simulation under multi-agent and ask/bid setting with inventory
import copy
import numpy as np
import pickle
np.random.seed(12345)

class bcolors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PINK = '\033[35m'
    GREY = '\033[36m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

lr = 100000

# Use 2000000 for Fig. 2 and 3
n_periods = 2000000

eps = 1e-5
DELTA = 0.0

n_agents = 2
tick_num = 2

sig = 0.1
# Set to 0.0 for Fig.3
inven_factor = 0.0
temper = 0.1

# Comment out to replicate ask/bid spreads case in Fig.2 and 3
weights = np.zeros(tick_num)
space = np.zeros(tick_num)
space[0] = 0.1
space[1] = 0.8

# space = np.linspace(0.1, 0.5, tick_num)
# weights = np.linspace(0.0, 0.1, tick_num)

n_instance = 10

def reward_cal(action):
    # Compute profits for all agents
    reward = np.zeros(n_agents)
    action = action.astype(int)
    price = space[action]

    # Only the lowest bid gets the order
    kernel = np.sum(weights[action])
    kernel = kernel / sig / n_agents
    arrival_prob = np.exp(-kernel)

    if np.random.binomial(1, arrival_prob, 1)[0]:
        winner = np.where(price == price.min())
        reward[winner] = price.min() / len(winner[0])
    else:
        winner = -1

    return winner, reward


def boltz_select(agent, Q):
    prob = np.exp(Q[agent] / temper)
    prob = prob / np.sum(prob)
    return np.random.choice(tick_num ** 2, 1, p=prob)


# Q_hist = np.zeros((n_instance, n_periods+1, n_agents, tick_num ** 2))
Q_hist = []
Q_last = np.zeros((n_instance, n_agents, tick_num ** 2))
# inventory_hist = np.zeros((n_instance, n_periods+1, n_agents))
inventory_hist = []

for sess in range(n_instance):

    steps_done = 0
    # Counter for variations in heat
    count = 0
    inventory = np.zeros(n_agents)

    Q = np.zeros((n_agents, tick_num ** 2))

    epiQ_hist = []
    epiI_hist = []
    for i_episode in range(n_periods + 1):
        # For each agent, select and perform an action
        action = np.zeros(n_agents, dtype=int)

        for i in range(n_agents):
            if inventory[i] > 100:
                action[i] = 1
            elif inventory[i] < -100:
                action[i] = 2
            else:
                action[i] = boltz_select(i, Q)

        ask_action = (action // tick_num).astype(int)
        bid_action = (action % tick_num).astype(int)

        steps_done += 1

        old_inventory = copy.deepcopy(inventory)

        bid_winner, bid_reward = reward_cal(bid_action)
        ask_winner, ask_reward = reward_cal(ask_action)
        if bid_winner != -1:
            inventory[bid_winner] += 1 / len(bid_winner[0])
        if ask_winner != -1:
            inventory[ask_winner] -= 1 / len(ask_winner[0])

        inventory_change = inventory - old_inventory
        # Inventory risk
        reward_total = bid_reward + ask_reward - inven_factor * inventory_change**2

        old_heat = Q.argmax(1)

        if i_episode%10000 == 0:
            epiQ_hist.append(copy.deepcopy(Q))
            epiI_hist.append(copy.deepcopy(inventory))

        alpha = lr/(lr + steps_done)

        for i in range(n_agents):
            Q[i, action[i]] = (1 - alpha)*Q[i, action[i]] + alpha*(reward_total[i] + DELTA*Q[i].max())

        new_heat = Q.argmax(1)

        if np.sum(np.abs(old_heat - new_heat)) == 0:
            count += 1
        else:
            count = 0

        if i_episode % 100000 == 0:
            print(bcolors.GREEN + 'Session', sess, 'Step:', steps_done, bcolors.ENDC)
            print('Greedy policy', Q.argmax(1))
            print('Q', Q)
            print('Bid', space[bid_action], 'Ask', space[ask_action])
            print('Inventory', inventory, 'Count', count)

        if count == 1000000:
            print(bcolors.RED + 'Terminate condition satisfied.' + bcolors.ENDC)
            print('Q', Q)
            # break

    Q_last[sess, :, :] = Q
    Q_hist.append(epiQ_hist)
    inventory_hist.append(epiI_hist)

Q_mean = Q_last.mean(axis=(0, 1))
prob = np.exp(Q_mean / temper)
prob = prob / prob.sum()
print('Mean values of Q', Q_mean)
print('Variance of Q', Q_last.var(0))
print('Probability of actions', prob)


# For Fig.2
# with open('Q_stag.pickle', 'wb') as fp:
#     pickle.dump(np.array(Q_hist), fp)
#
# with open('inven_stag.pickle', 'wb') as fp:
#     pickle.dump(np.array(inventory_hist), fp)

# For Fig. 3
with open('Q_hard.pickle', 'wb') as fp:
    pickle.dump(np.array(Q_hist), fp)

with open('inven_hard.pickle', 'wb') as fp:
    pickle.dump(np.array(inventory_hist), fp)

# For Fig. 7
# with open('Q_AB.pickle', 'wb') as fp:
#     pickle.dump(np.array(Q_hist), fp)
#
# with open('inven_AB.pickle', 'wb') as fp:
#     pickle.dump(np.array(inventory_hist), fp)
