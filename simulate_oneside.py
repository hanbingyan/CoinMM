# Simulation of Ask or Bid with multi-agents
import numpy as np
import copy
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

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

lr = 100000
n_periods = 4000000


DELTA = 0.0

n_agents = 2
tick_num = 10
space = np.linspace(0.1, 1, tick_num)
weights = np.linspace(0.0, 0.1, tick_num)

# space = np.zeros(tick_num)
# space[0] = 0.5
# space[1] = 0.9

# weights = np.zeros(tick_num)
# weights[0] = 0.02
# weights[1] = 0.04
sig = 0.1
temper = 0.1

def reward_cal(action):
    # Compute profits for all agents
    reward = np.zeros(n_agents)
    action = action.astype(int)
    price = space[action]

    # Only lowest bid gets the order
    kernel = np.sum(weights[action])
    kernel = kernel / sig / n_agents
    arrival_prob = np.exp(-kernel)

    if np.random.binomial(1, arrival_prob, 1)[0]:
        winner = np.where(price == price.min())
        reward[winner] = price.min() / len(winner[0])
    else:
        winner = -1

    return winner, reward


def boltz_select(agent, Q, temperature):
    prob = np.exp(Q[agent] / temper)
    prob = prob / np.sum(prob)

    return np.random.choice(tick_num, 1, p=prob)

n_instance = 10
# Q_hist = np.zeros((n_instance, n_agents, tick_num))

Q_hist = []
Q_last = np.zeros((n_instance, n_agents, tick_num))

for sess in range(n_instance):

    # R_avg = np.zeros(2)
    steps_done = 0
    # Counter for variations in heat
    count = 0

    Q = np.zeros((n_agents, tick_num))
    # Q[:, 1] = 0.0

    epiQ_hist = []

    # vol_counter = np.zeros(n_periods)
    # hist = np.zeros((n_periods, n_agents))

    for i_episode in range(n_periods + 1):
        # For each agent, select and perform an action
        action = np.zeros(n_agents, dtype=int)
        for i in range(n_agents):
            action[i] = boltz_select(i, Q, temper)
            # action[i] = eps_greedy(i, Q, steps_done)

        steps_done += 1

        winner, reward = reward_cal(action)
        old_heat = Q.argmax(1)

        if i_episode % 10000 == 0:
            epiQ_hist.append(copy.deepcopy(Q))

        alpha = lr / (lr + steps_done)
        # alpha = 1e-7
        for i in range(n_agents):
            Q[i, action[i]] = (1 - alpha) * Q[i, action[i]] + alpha * (reward[i] + DELTA * Q[i].max())

        new_heat = Q.argmax(1)

        #         if steps_done >=2:
        #             LHS = np.max(np.abs(Q_hist[steps_done] - Q_hist[steps_done - 1]))
        #             RHS = np.max(np.abs(Q_hist[steps_done - 1] - Q_hist[steps_done - 2]))
        #             # if LHS < RHS:
        #             vol_counter[steps_done] = LHS/RHS
        #             # else:
        #             #     vol_counter[steps_done] = -1

        #         if steps_done%10000 == 0:
        #             print('LHS/RHS', LHS, RHS, LHS/RHS)


        # hist[int(steps_done % n_periods), :] = action

        if np.sum(np.abs(old_heat - new_heat)) == 0:
            count += 1
        else:
            count = 0

        if i_episode % 50000 == 0:
            print(bcolors.GREEN + 'Session', sess, 'Step:', steps_done, bcolors.ENDC)
            print('Count', count)
            print('Greedy policy', Q.argmax(1))
            print('Q', Q)
            print('Action', space[action])

        # if steps_done > 0 and steps_done%n_periods == 0:
        #             act_max = np.unique(hist, return_counts=True, axis = 0)[1].max()

        #             print("act_max", act_max)
        #             print("ask_dist", np.unique(hist, return_counts=True, axis = 0)[1])

        # if act_max/n_periods > 0.9:
        if count == 1000000:
            print(bcolors.RED, sess, 'Terminate condition satisfied.' + bcolors.ENDC)
            print('Q', Q)
            break

    Q_last[sess, :, :] = Q
    Q_hist.append(epiQ_hist)
            # print(np.sum(vol_counter == 1)/steps_done)
            # print(np.sum(vol_counter[int(0.9*steps_done):] == 1) /0.9/steps_done)
with open('Q_oneside.pickle', 'wb') as fp:
    pickle.dump(np.array(Q_hist), fp)

Q_mean = Q_last.mean(axis=(0,1))
prob = np.exp(Q_mean/temper)
prob = prob/prob.sum()
print('Mean values of Q', Q_mean)
print('Variance of Q', Q_last.var(0))
print('Probability of actions', prob)

