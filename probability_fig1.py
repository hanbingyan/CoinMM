# Theoretical limit with two actions on one side; Multi-agents allowed
import random
import numpy as np

import matplotlib.pyplot as plt
from scipy.special import comb

random.seed(12345)
np.random.seed(12345)

n_agents = 2
tick_num = 2

space = np.zeros(tick_num)

weights = np.zeros(tick_num)
sig = 0.1

def expected_reward(action):
    action = action.astype(int)
    price = space[action]

    kernel = np.sum(weights[action])
    kernel = kernel / sig / n_agents
    arrival_prob = np.exp(-kernel)
    winner = np.where(price == price.min())
    return arrival_prob*price.min()/len(winner[0])

def poly(x):
    val = 0
    for i in range(n_agents):
        action = np.ones(n_agents)
        action[:i+1] = 0
        val += expected_reward(action)*comb(n_agents-1, i)*np.power(x, i)*np.power(1-x, n_agents-1-i)
    val -= expected_reward(np.ones(n_agents))*np.power(1-x, n_agents-1)
    return val

x = np.linspace(1e-15, 0.999999999, 5000)

space[0] = 0.1
space[1] = 0.8
temper = 0.1
SHOne = np.zeros_like(x)
for k in range(len(x)):
    SHOne[k] = poly(x[k])/temper

space[0] = 0.1
space[1] = 0.8
temper = 0.01
SHTwo = np.zeros_like(x)
for k in range(len(x)):
    SHTwo[k] = poly(x[k])/temper

space[0] = 0.41
space[1] = 0.8
temper = 0.01


PD = np.zeros_like(x)
for k in range(len(x)):
    PD[k] = poly(x[k])/temper

# lin = a12 - a22 + (a11 - a12 + a22)*x

lnf = np.log(x/(1 - x))
# plt.figure(figsize=(8,6))
fig, ax = plt.subplots()
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax.plot(x, SHOne, linestyle='dashed', label=r'SH, $\lambda$ = 0.1')
ax.plot(x, SHTwo, linestyle='dotted', label=r'SH, $\lambda$ = 0.01')
ax.plot(x, PD, linestyle='dashdot', label=r'PD, $\lambda$ = 0.01')
ax.plot(x, lnf, color='black')
ax.set_xlabel('Probability', fontsize=14)
ax.legend(loc='best', fontsize=14)
plt.savefig('TwoAgents.pdf', format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)
# plt.show()
