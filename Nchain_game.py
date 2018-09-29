import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.models import model_from_json, load_model
import matplotlib.pylab as plt
import pandas as pd
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
    greedy_table = np.zeros((num_states, num_actions))
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
            greedy_table[s, a] += r + alpha*(gama*np.max(greedy_table[new_state, :]) - greedy_table[s, a])
            s = new_state

    return greedy_table

# ########################################################################################

# def model()
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

# table = Q_table(env, 5, 2)
# PIK = "q_table.dat"
#
# with open(PIK, "wb") as f:
#     pickle.dump(table, f)
# with open(PIK, "rb") as f:
#     tab = pickle.load(f)
#     print(tab)
#     game_run(env, tab)

# ########################################################################################
# ########################################################################################
# ################# greedy Q table result

# table = greedy_Q_table(env, 5, 2, 0.7)
# PIK = "greedy_table.dat"
#
# with open(PIK, 'wb') as f:
#     pickle.dump(table, f)
# with open(PIK, 'rb') as f:
#     tab = pickle.load(f)
#     print(tab)
#     game_run(env, tab)
# ########################################################################################
# ########################################################################################
# ################# compare of naive table, Q table, and greedy Q table performance





# ########################################################################################
# ########################################################################################
# ################# keras deep model

num_episode = 15
gama = 0.9
eps = 0.7
r_avg = []

model = Sequential()

model.add(InputLayer(batch_input_shape=(1, 5)))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(2, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics = ['mae'])

for i in range(num_episode):
    s = env.reset()
    eps *= eps
    if i%5 == 0:
        print("Episode {} of {}".format(i+1, num_episode))
    done = False
    r_sum = 0
    while not done:
        if np.random.random() < eps:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(model.predict(np.identity(5)[s:s+1]))
            # S = []
            # for i in range(5):
            #     if i == s:
            #         S.append(s)
            #     else:
            #         S.append(0)
            #
            # a = np.argmax(model.predict(S))
        new_state, r, done, _ = env.step(a)
        target = r + gama*(np.max(model.predict(np.identity(5)[new_state: new_state +1])))
        target_vec = model.predict(np.identity(5)[s:s+1])[0]
        target_vec[a] = target
        model.fit(np.identity(5)[s:s+1], target_vec.reshape(-1,2), epochs=1, verbose=0)
        s = new_state
        r_sum += r
    r_avg.append(r_sum/1000)


PIK = 'average_reward.dat'

with open(PIK, 'wb') as f:
    pickle.dump(r_avg, f)

# Creates a HDF5 file 'my_model.h5'
model.save ('my_model.h5')

# Deletes the existing model
del model

# Returns a compiled model identical to the previous one
model2 = load_model ('my_model.h5')



# for i in range(10):
#     a = np.argmax(loaded_model.predict(np.identity(5)[s:s+1]))
#     new_state, r, done, _ = env.step(a)
#     print('in state {} action is {} and new state is {}'.format(s, a, new_state))
#     s = new_state


# with open('average_reward.dat', 'rb') as f:
#     tab = pickle.load(f)
#     print(tab)
#
# plt.plot(np.arange(0, len(tab), 1), tab)
# plt.show()