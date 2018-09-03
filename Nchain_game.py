import gym
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, InputLayer
# import matplotlib.pylab as plt
# import pandas as pd
import pickle

env = gym.make('NChain-v0')

# ########################################################################################

def naive_table(env, num_actions, num_states, epsilon = 0.01, decay = 0.1, num_episode = 5000):
    table = np.zeros((num_states, num_actions))
    for e in range(num_episode):
        s = env.reset()
        done = False
        while not done:
            if np.sum(table[s, :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(table[s, :])

            new_state, r, done, _ = env.step(a)
            table[s, a] = table[s, a] + (1/(1+e))*(r - table[s, a])
            s = new_state
    return table

# ########################################################################################

def Q_table(env, num_states, num_actions, num_eps=500):
    q_table = np.zeros((num_states, num_actions))
    alpha = 0.8 # learning rate
    gama = 0.95 # discount factor
    for i in range(num_eps):
        s = env.reset()
        done = False
        while not done:
            if np.sum (q_table[s, :]) == 0:
                a = np.random.randint (0, 2)
            else:
                a = np.argmax (q_table[s, :])

            new_state, r, done, _ = env.step(a)
            q_table[s, a] += r + alpha*(gama* np.max(q_table[new_state, :]) - q_table[s, a])
            s = new_state
    return q_table

# ########################################################################################

def greedy_Q_table(env, num_states, num_actions, eps, episode = 500):
    greedy_table = np.zeors((num_states, num_actions))
    alpha = 0.9
    gama = 0.95

    for i in range(episode):
        s = env.reset()
        done = False
        eps *=eps
        while not done:
            if np.random.random()< eps or np.sum(greedy_table[s, :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(greedy_table[s, :])
            new_state, r, done, _ = env.step(a)
            # greedy_table[s, a] += r +


# ########################################################################################
def game_run(env, q_table, episode = 10):
    render = True
    for i in range(episode):
        s = env.reset ()
        done = False
        total_reward = 0
        horizon = 0
        while not done:
            # if render:
            #     env.render()
            a = np.argmax(q_table[s, :])
            new_state, r, done, _ =env.step(a)
            s = new_state
            total_reward += r
            horizon +=1
        print("total reward in the "+ str(i) + " = " + str(total_reward) + ", number of step = "+ str(horizon))

# ########################################################################################
# ########################################################################################
# ################# naive table result
# table = naive_table(env, 2, 5)
# PIK = "naive_table1.dat"
#
# # with open(PIK, "wb") as f:
# #     pickle.dump(table, f)
# with open(PIK, "rb") as f:
#     tab = pickle.load(f)
#     print(tab)
#     game_run(env, tab)

# ########################################################################################
# ########################################################################################
# ################# Q table result
table = Q_table(env, 5, 2)
PIK = "q_table.dat"

with open(PIK, "wb") as f:
    pickle.dump(table, f)
with open(PIK, "rb") as f:
    tab = pickle.load(f)
    print(tab)
    game_run(env, tab)




