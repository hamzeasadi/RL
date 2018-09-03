import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math

class Model():
    '''This class is a general model for reinforcement purpose'''
    def __init__(self, num_states, num_actions, batch_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self._states = None
        self._actions = None
        self._logits = None
        self._optimizer = None
        self._vare_init = None
        self._define_model()

    def _define_model(self):
        self.states = tf.placeholder(shape=[None, self.num_states], dtype=tf.float32)
        self._Q_s_a = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)

        fc1 = tf.layers.dense(self._states, 50, activation= tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation= tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self.num_actions)
        loss = tf.losses.mean_squared_error(self._Q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._vare_init = tf.global_variables_initializer()

    def _predict_one(self, state, sess):
        output = sess.run(self._logits, feed_dict = {self._states: state.reshape(1, self.num_states)})
        return output

    def predict_batch(self, states, sess):
        output = sess.run(self._logits, feed_dic = {self._states: states})
        return output

    def batch_train(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dic = {self._states: x_batch, self._Q_s_a: y_batch})


class Memory():
    '''this class save different sample of interaction between agent and environment to prevent of overfitting'''
    def __init__(self, max_memory):

        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples)> self._max_memory:
            self._samples.pop()

    def sample(self, num_samples):
        if num_samples> len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, num_samples)



class GameRunner():
    '''running the game using our predefined model'''
    def __init__(self, sess, model, env, memory, max_eps, min_eps, decay, render = True):
        self._sess = sess
        self._env = env
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._model = model
        self._memory = memory
        self._render = render
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        total_reward = 0
        max_x = -100
        while True:
            if self._render:
                self._env.render()

            action = self._choose_action(state)
            next_state, reward, done, info = self._env.step(action)
            if next_state[0] > 0.1:
                reward += 10
            elif next_state[0] > 0.25:
               reward += 20
            elif next_state[0] > 0.5:
                reward += 100
            if next_state[0]> max_x:
                max_x = next_state[0]

            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            self._reply()






























