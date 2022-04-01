# Simulation with Q states and bid/ask spreads
import copy
import numpy as np
import pickle
import os

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
n_periods = 4000000

DELTA = 0.9
n_agents = 2
tick_num = 4

sig = 0.1
inven_factor = 0.1
temper = 0.01

space = np.linspace(0.1, 0.5, tick_num)
weights = np.linspace(0.0, 0.1, tick_num)


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
        order = 1
        winner = np.where(price == price.min())
        reward[winner] = price.min() / len(winner[0])
    else:
        winner = -1
        order = 0

    return order, winner, reward



def boltz_select(agent, Q, state):
    prob = np.exp(Q[agent, state[0], state[1]]/temper)
    prob = prob / np.sum(prob)
    return np.random.choice(tick_num ** 2, 1, p=prob)


def train():

    Q_hist = []
    Q_last = np.zeros((n_instance, n_agents, tick_num ** 2, tick_num ** 2, tick_num ** 2))
    inventory_hist = []
    state_hist = []
    reward_hist = []
    order_hist = []

    for sess in range(n_instance):

        steps_done = 0
        # Counter for variations in heat
        count = 0
        inventory = np.zeros(n_agents)

        # Q is in the shape of (agents, state_axis1, state_axis2, actions)
        Q = np.zeros((n_agents, tick_num ** 2, tick_num ** 2, tick_num ** 2))
        # Q[:, :, :, 0] = 0.1
        # Q = np.random.rand(n_agents, tick_num**2)

        epiQ_hist = []
        epiI_hist = []
        epistate_hist = []
        epireward_hist = []
        epiorder_hist = []

        state = np.random.randint(tick_num**2, size=2)
        for i_episode in range(n_periods + 1):
            # For each agent, select and perform an action

            action = np.zeros(n_agents, dtype=int)

            for i in range(n_agents):
                action[i] = boltz_select(i, Q, state)

            next_state = action.copy()

            ask_action = (action // tick_num).astype(int)
            bid_action = (action % tick_num).astype(int)

            steps_done += 1

            old_inventory = copy.deepcopy(inventory)

            bid_order, bid_winner, bid_reward = reward_cal(bid_action)
            ask_order, ask_winner, ask_reward = reward_cal(ask_action)


            if bid_winner != -1:
                inventory[bid_winner] += 1 / len(bid_winner[0])
            if ask_winner != -1:
                inventory[ask_winner] -= 1 / len(ask_winner[0])

            inventory_change = inventory - old_inventory
            # Inventory risk
            reward_total = bid_reward + ask_reward - inven_factor * inventory_change**2

            if i_episode % 100 == 0:
                epistate_hist.append(state)
                epireward_hist.append(reward_total)
                epiorder_hist.append(ask_order+bid_order)

            old_heat = Q.argmax(3)

            if i_episode%10000 == 0:
                epiQ_hist.append(copy.deepcopy(Q))
                epiI_hist.append(copy.deepcopy(inventory))

            alpha = lr/(lr + steps_done)

            for i in range(n_agents):
                Q[i, state[0], state[1], action[i]] = (1 - alpha)*Q[i, state[0], state[1], action[i]] + \
                                                        alpha*(reward_total[i] + DELTA*Q[i, next_state[0], next_state[1]].max())

            state = next_state
            new_heat = Q.argmax(3)

            if np.sum(np.abs(old_heat - new_heat)) < 20:
                count += 1
            else:
                count = 0

            if i_episode % 10000 == 0:
                print(bcolors.GREEN + 'Session', sess, 'Step:', steps_done, bcolors.ENDC)
                # print('Greedy policy', Q.argmax(3))
                # print('Q', Q)
                print('Bid', space[bid_action], 'Ask', space[ask_action])
                print('Inventory', inventory, 'Count', count)

            if count == 500000:
                print(bcolors.RED + 'Terminate condition satisfied.' + bcolors.ENDC)
                # print('Q', Q)
                break

        Q_last[sess, :, :, :, :] = Q
        Q_hist.append(epiQ_hist)
        inventory_hist.append(epiI_hist)
        state_hist.append(epistate_hist)
        reward_hist.append(epireward_hist)
        order_hist.append(epiorder_hist)

    return Q_last, Q_hist, inventory_hist, state_hist, reward_hist, order_hist


if __name__ == '__main__':
    Q_last, Q_hist, inventory_hist, state_hist, reward_hist, order_hist = train()

    # sub_folder = '{}_{}.{}.{}'.format('state', args.thres, datetime.now().strftime('%M'), datetime.now().strftime('%S'))
    sub_folder = '{}_{}'.format('mem_longsighted', temper)
    log_dir = './logs/{}'.format(sub_folder)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save params configuration
    with open('{}/params.txt'.format(log_dir), 'w') as fp:
        fp.write('Params setting \n')
        fp.write('Delta {}, inven_factor {}, temper {}\n'.format(DELTA, inven_factor, temper))

    with open('{}/Q.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(Q_hist, fp)

    with open('{}/inven.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(inventory_hist, fp)

    with open('{}/state.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(state_hist, fp)

    with open('{}/reward.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(reward_hist, fp)

    with open('{}/order.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(order_hist, fp)
