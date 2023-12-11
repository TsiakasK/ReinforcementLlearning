# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:19:43 2023

@author: Anniek Jansen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:58:10 2023

@author: Anniek Jansen
"""

#!/usr/bin/python


import numpy as np
import random as rnd
import itertools
import matplotlib.pyplot as plt
import csv
from extended_user_model_np import *


def maxs(seq):
    max_indices = []
    # if seq:
    max_val = seq[0]
    for i, val in ((i, val) for i, val in enumerate(seq) if val >= max_val):
        if val == max_val:
            max_indices.append(i)
        else:
            max_val = val
            max_indices = [i]

    return rnd.choice(max_indices)

# return a random action based on the Q-probabilities


def cdf(seq):
    r = rnd.random()
    for i, s in enumerate(seq):
        if r <= s:
            return i

# create state space as combinatin of features


def state_action_space():
    # 0: neutral/none, 1: content, 2: happy, 3: surprised, 4: sad, 5: angry
    current_action = [0, 1, 2, 3, 4, 5]
    #pain_presence = [0, 1]  # 0: no pain, 1: pain
    # 0: neutral/none, 1: content, 2: happy, 3: surprised, 4: sad, 5: angry
    previous_action = [0, 1, 2, 3, 4, 5]
    # 0: no user_interaction, 1: any user_interaction (petting, clapping, shaking)
    previous_success = [0, 1]
    combs = (current_action,
             previous_action, previous_success)
    states = list(itertools.product(*combs))

    '''
    actions 0: neutral/none, 1: content, 2: happy, 3: surprised, 4: sad, 5: angry

    '''
    actions = [0, 1, 2, 3, 4, 5]
    return states, actions


def get_next_state(state, states, action, previous_action, user_model):
    '''
    STATE SPACE
    state[0]: current action {0: neutral/none, 1: content, 2: happy, 3: surprised, 4: sad, 5: angry}
    state[1]: previous action {0: neutral/none, 1: content, 2: happy, 3: surprised, 4: sad, 5: angry}
    state[2]: previous succes 0: no user_interaction, 1: any user_interaction (petting, clapping, shaking)

    STATE last 10 seconds

    At the end of the state, sample pain and interaction and get reward

    Question:: is current state needed,
    '''

    next_state = state[:]
    
    interaction = user_interaction_model2(action, state, user_model, previous_action)
    # if there is any user_interaction, set previous succes to 1
    if interaction != 0:
        next_state[2] = 1
    else:
        next_state[2] = 0
    next_state[0] = action
    #next_state[1] = pain_model(action, state, user_model)
    next_state[1] = previous_action

# REWARDS
# TODO change reward signal to discourage switching to often between behaviours, look at s, s' and a
# Reward of interaction should be higher then punishment for changing action

# using -1 for switching state resulted in a policy that never changed state

    score = 0
    if interaction == 1:
        score = 10
    if (action == previous_action and interaction == 1):
        score += 2
    # ensures that the robot still does some switches in behaviour
    elif (action == previous_action and interaction == 0 ):
        score -= 2
               
    else:
        score = 0

    reward = score
    return reward, next_state

# define the MDP


class MDP:
    def __init__(self, init_state, actlist, terminals=[], gamma=.9):
        self.init = init_state
        self.actlist = actlist
        self.terminals = terminals
        if not (0 <= gamma < 1):
            raise ValueError("An MDP must have 0 <= gamma < 1")
        self.gamma = gamma
        self.states = set()
        self.reward = 0

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
list of actions, except for terminal states. Override this
 if you need to specialize by state."""
        if state in self.terminals:
            return None
        else:
            return self.actlist

# define Policy (softmax or e-greedy)


class Policy:

    def __init__(
        self,
        name,
        param,
        Q_state=[],
        Q_next_state=[],
    ):
        self.name = name
        self.param = param
        self.Q_state = Q_state
        self.Q_next_state = Q_next_state

    def return_action(self):

        if self.name == 'egreedy':
            values = self.Q_state
            maxQ = max(values)
            e = self.param
            if rnd.random() < e:  # exploration
                return rnd.randint(0, len(values) - 1)
            else:
               # exploitation
                return maxs(values)

# representation is Qtable only


class Representation:
    # qtable, neural network, policy function, function approximation
    def __init__(self, name, params):
        self.name = name
        self.params = params
        if self.name == 'qtable':
            [self.actlist, self.states] = self.params
            self.Q = [[0.0] * len(self.actlist)
                      for x in range(len(self.states))]

# learning algorithms: sarsa and q-learning


class Learning:
    # qlearning, sarsa, traces, actor critic, policy gradient
    def __init__(self, name, params):
        self.name = name
        self.params = params
        if self.name == 'qlearn' or self.name == 'sarsa':
            self.alpha = self.params[0]
            self.gamma = self.params[1]

    def decrease_alpha(self, alpha):
        self.alpha = alpha

    def update(self, state, action, next_state, next_action, reward, Q_state, Q_next_state, done):
        if done:
            Q_state[action] = Q_state[action] + \
                self.alpha*(reward - Q_state[action])
            error = reward - Q_state[action]
        else:
            if self.name == 'qlearn':
                Q_state[action] += self.alpha * \
                    (reward + self.gamma*max(Q_next_state) - Q_state[action])
                error = reward + self.gamma*max(Q_next_state) - Q_state[action]

            if self.name == 'sarsa':
                learning = self.alpha * \
                    (reward + self.gamma *
                     Q_next_state[next_action] - Q_state[action])
                Q_state[action] = Q_state[action] + learning
                error = reward + self.gamma * \
                    Q_next_state[next_action] - Q_state[action]
        return Q_state, error


def simulate(ALPHA, GAMMA, num_interactions, egreedy_param, num_episodes, user_model):
    # get state-action space
    states, actions = state_action_space()
    # exploring starts, both with and without pain
    start_state = [0, 0, 0]
 
    m = MDP(start_state, actions)
    m.states = states

    alabel = ["neutral/none", "content",  "happy", "surprised", "sad", "angry"]

    # initialize Q-table
    table = Representation('qtable', [m.actlist, m.states])
    Q = np.asarray(table.Q)


    # this can change to suit the problem -- number of learning episodes - games
    episodes = num_episodes
    episode = 1

    # q-values --> q.Q
    egreedy = Policy('egreedy', 1.0, table.Q)

    # State transitions happen every # seconds, e.g. every 10 seconds. Then it will check for interactions, pain levels, previous success and update state

    # alpha = float(ALPHA) #learning rate
    alpha = float(ALPHA)
    gamma = float(GAMMA)  # discount factor
    learning = Learning('sarsa', [alpha, gamma])
    # this can change to suit the problem -- number of rounds
    interactions = num_interactions
    attempts = []
    errors = []
    returns = []

    while (episode <= episodes):
        previous_action = 0  # start with neutral as "last" action
        interaction = 1
        done = 0
        state = [0,  0, 0]
        # state = start_state
        if (episode % 499 == 0):
            print("Episode: " + str(episode))
        r = 0
        e = 0
        egreedy.param *= egreedy_param  # this can change to suit the problem
        alpha *= 0.99
        if alpha < 0.1:
            alpha = 0.1
        learning.decrease_alpha(alpha)

        if egreedy.param < 0.1:
            egreedy.param = 0.0

        while (not done):
            state_index = states.index(tuple(state))

            # human actions not available for the agent (actions 5-10)
            egreedy.Q_state = Q[state_index][:]

            action = egreedy.return_action()

            # wait 10 seconds and evaluate reward and get next state

            # or get next state and reward from interaction for online learning
            reward, next_state = get_next_state(
                state, states, action, previous_action, user_model)
            next_state_index = states.index(tuple(next_state))
            r += (learning.gamma**(interaction-1))*reward

            # sarsa
            # again only choose from actions that the agent can do
            egreedy.Q_next_state = Q[next_state_index][:]
            next_action = egreedy.return_action()
            if interaction == interactions:
                done = 1
                attempts.append(interaction)

            # LEARNING
            Q[state_index][:], error = learning.update(
                state_index, action, next_state_index, next_action, reward, Q[state_index][:], Q[next_state_index][:], done)
            e += error
            if (episode % 499 == 0):
                # print ("Episode: " + str(episode))
                print(interaction, state, alabel[action],
                      next_state, reward, egreedy.param)
            state = next_state
            previous_action = action
            interaction += 1

        episode += 1
        returns.append(r)
        errors.append(error)
    # print(Q)
    if user_model == 1:
        with open('q_table_um1_np.py', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(Q)
    if user_model ==2:
         with open('q_table_um2_np.py', 'w', newline='') as f:
             writer = csv.writer(f,delimiter=' ')
             writer.writerows(Q)
    if user_model ==3:
         with open('q_table_um3_np.py', 'w', newline='') as f:
             writer = csv.writer(f,delimiter=' ')
             writer.writerows(Q)
    if user_model ==4:
         with open('q_table_um4_np.py', 'w', newline='') as f:
             writer = csv.writer(f,delimiter=' ')
             writer.writerows(Q)


    return returns, errors


def moving_average(a, n=20):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


run1_returns, run1_errors = simulate(ALPHA=0.25, GAMMA=0.9, num_interactions= 180, egreedy_param=0.95, num_episodes=500, user_model=1)
run2_returns, run2_errors = simulate(ALPHA=0.25, GAMMA=0.9, num_interactions= 180, egreedy_param=0.95, num_episodes=500, user_model=2)
run3_returns, run3_errors = simulate(ALPHA=0.25, GAMMA=0.9, num_interactions= 180, egreedy_param=0.95, num_episodes=500, user_model=3)
run4_returns, run4_errors = simulate(ALPHA=0.25, GAMMA=0.9, num_interactions= 180, egreedy_param=0.95, num_episodes=500, user_model=4)


plt.plot(moving_average(run1_returns), 'b', moving_average(run2_returns), 'r', moving_average(run3_returns), 'g', moving_average(run4_returns), 'c')
plt.legend(['user model 1', 'user model 2', 'user model 3','user model 4'])
plt.title("Total return")
plt.show()

plt.plot(moving_average(run1_errors), 'b', moving_average(run2_errors), 'r', moving_average(run3_errors), 'g', moving_average(run4_errors), 'c')
plt.legend(['user model 1', 'user model 2', 'user model 3', 'user model 4'])
plt.title("Total error")
plt.show()


