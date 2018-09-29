# import IPython.display
# import numpy as np
# import gym
# import random
# import pickle
# import skimage
#
#
# def run_game(env, table):
#     for i in range(10):
#         s = env.reset()
#         done = False
#         while not done:
#             env.render()
#             a = np.argmax(table[s, :])
#             new_state , reward, done, info = env.step(a)
#             s = new_state
#         print("episode {} of 10".format(i))
#
# env = gym.make('Taxi-v2')
# env.reset()
#
# action_size = env.action_space.n
# state_space = env.observation_space.n
#
# qtable = np.zeros((state_space, action_size))
# total_episode = 2000
# learning_rate = 0.9
# max_steps = 100
# gamma = 0.9
#
# epsilon = 1.0
# max_epsilon = 1.0
# min_epsilon = 0.01
# decay_rate = 0.006
#
# rewards = []
# for episode in range(total_episode):
#     s = env.reset()
#     done = False
#     total_reward = 0
#     step = 0
#     while not done:
#         randm = random.uniform(0, 1)
#         if randm < episode:
#             a = env.action_space.sample()
#         else:
#             a = np.argmax(qtable[s, :])
#         state, reward, done, _ = env.step(a)
#         qtable[s, a] = qtable[s, a] + learning_rate*(reward + gamma*np.max(qtable[state, :]) - qtable[s, a])
#         total_reward += reward
#         step += 1
#         if step> max_steps:
#             break
#         s = state
#     epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
#     rewards.append(total_reward)
#
# PIK = "qtable_taxi-v0.dat"
#
# with open(PIK, "wb") as f:
#     pickle.dump(qtable, f)
# with open(PIK, "rb") as f:
#     tab = pickle.load(f)
#     print(tab)
#     run_game(env, tab)
#
#
#
#

import numpy as np
import keras




