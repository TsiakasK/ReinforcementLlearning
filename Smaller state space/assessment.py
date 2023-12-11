# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:12:21 2023

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
    
    d = [action, interaction, state[2], previous_action]

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
    return reward, next_state, d

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
        

def get_action(interaction, total_interactions):
# =============================================================================
#     actions = [0, 1, 2, 3,4,5]
#     num_actions = len(actions)
#     period = total_interactions // num_actions  # Use integer division to get whole periods
# 
#     # Calculate the current period and return the corresponding action
#     current_period = interaction // period
# 
# 
#     # Make sure the current period is within the valid range
#     current_period = min(current_period, num_actions - 1)
# 
#     return actions[current_period]
# 
# =============================================================================
    actions = [0, 1, 2, 3, 4, 5]
    num_actions = len(actions)
    pairs = [(i, j) for i in range(num_actions) for j in range(i + 1, num_actions)]
    pairs_count = len(pairs)
    period = total_interactions // pairs_count  # Use integer division to get whole periods

    # Calculate the current period and return the corresponding action pair
    current_pair = interaction // period

    # Make sure the current pair is within the valid range
    current_pair = min(current_pair, pairs_count - 1)

    # Get the actions from the current pair
    current_actions = pairs[current_pair]

    # Alternate between the two actions in the pair for each half of the period
    action = current_actions[interaction % (period // 2) >= period // 4]

    return action
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

    def return_action(self, interaction, num_interactions):

        if self.name == 'egreedy':
            values = self.Q_state
            maxQ = max(values)
            e = self.param
            if rnd.random() < e:  # exploration
                return rnd.randint(0, len(values) - 1)
            else:
               # exploitation
                return maxs(values)
        elif self.name =="fixed":
            return get_action(interaction, num_interactions)
            
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


def simulate(ALPHA, GAMMA, num_interactions, egreedy_param, num_episodes, user_model, basemodel):
    # get state-action space
    states, actions = state_action_space()
    # exploring starts, both with and without pain
    start_state = [0,  0, 0, 0]
    #start_state = [0, 1, 0, 0, 0]

    m = MDP(start_state, actions)
    m.states = states

    alabel = ["neutral/none", "content",  "happy", "surprised", "sad", "angry"]

    # initialize Q-table
    table = Representation('qtable', [m.actlist, m.states])
    Q = np.asarray(table.Q)

    q = True

    # A Q-table can be written and loaded as a file
    # you can load it like this:
    if q:
        match basemodel:
            case "um1":
                ins = open("q_table_um1_np.py", 'r') 

            case "um2":
                ins = open("q_table_um2_np.py", 'r') 
      
            case "um3":
                ins = open("q_table_um3_np.py", 'r') 

            case "um4":
                ins = open("q_table_um4_np.py", 'r') 

        Q = [[float(n) for n in line.split()] for line in ins]
        ins.close()
        table.Q = Q
    # this can change to suit the problem -- number of learning episodes - games
    episodes = num_episodes
    episode = 1

    # q-values --> q.Q
    egreedy = Policy('egreedy', 1.0, table.Q)
    fixed = Policy('fixed', 1.0, table.Q)

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
    d1= []

    while (episode <= episodes):
        d1=d2=d3=d4=d5= []
        previous_action = 0  # start with neutral as "last" action
        interaction = 1
        done = 0
        state = [0, 0, 0]
        # state = start_state
# =============================================================================
#         if (episode % 1 == 0):
#             print("Episode: " + str(episode))
# =============================================================================
        r = 0
        e = 0
        egreedy.param = egreedy_param
        alpha = float(ALPHA)
        egreedy.param *= egreedy_param  # this can change to suit the problem
        alpha *= 0.90
        if alpha < 0.1:
            alpha = 0.1
        learning.decrease_alpha(alpha)

        if egreedy.param < 0.1:
            egreedy.param = 0.0

        while (not done):
            state_index = states.index(tuple(state))

            fixed.Q_state = Q[state_index][:]

            action = fixed.return_action(interaction, interactions)

            # wait 10 seconds and evaluate reward and get next state

            # or get next state and reward from interaction for online learning
            reward, next_state, d = get_next_state(
                state, states, action, previous_action, user_model)
            if user_model==1:
                d1.append(d)
            elif user_model==2:
                d2.append(d)
            elif user_model==3:
                d3.append(d)
            elif user_model==4:
                d4.append(d)
            elif user_model ==5:
                d5.append(d)
            next_state_index = states.index(tuple(next_state))
            r += (learning.gamma**(interaction-1))*reward

            # sarsa
            # again only choose from actions that the agent can do
            fixed.Q_next_state = Q[next_state_index][:]
            next_action = fixed.return_action(interaction, interactions)
            if interaction == interactions:
                done = 1
                attempts.append(interaction)

            # LEARNING
            Q[state_index][:], error = learning.update(
                state_index, action, next_state_index, next_action, reward, Q[state_index][:], Q[next_state_index][:], done)
            e += error
# =============================================================================
#             if (interaction % 1  == 0):
#                 # print ("Episode: " + str(episode))
#                 print(interaction, state, alabel[action],
#                       next_state, reward, egreedy.param, alpha)
# =============================================================================
            state = next_state
            previous_action = action
            interaction += 1
            returns.append(r)
            errors.append(error)
            if interaction % 5 == 0:
                egreedy.param *= egreedy_param  # this can change to suit the problem
                alpha *= 0.90
                if alpha < 0.1:
                    alpha = 0.1
                learning.decrease_alpha(alpha)

                if egreedy.param < 0.1:
                    egreedy.param = 0.0

        episode += 1
        if user_model==1:
            with open('interactions_um1_episodes.csv', 'a', newline='') as f:
                         writer = csv.writer(f, delimiter=',')
                         writer.writerow(d1)
        elif user_model==2:
           with open('interactions_um2_episodes.csv', 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerow(d2)
        elif user_model==3:
           with open('interactions_um3_episodes.csv', 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerow(d3)
        elif user_model==4:
           with open('interactions_um4_episodes.csv', 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerow(d4)
        elif user_model==5:
            with open('interactions_um5_episodes.csv', 'a', newline='') as f:
                         writer = csv.writer(f, delimiter=',')
                         writer.writerow(d5)

    return returns, errors


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

bm = "um4"
a=0.5
g= 0.9
num_int = 180
eg_param=0.1
num_eps= 1

# run1_returns, run1_errors = simulate(ALPHA=a, GAMMA=g, num_interactions= num_int, egreedy_param= eg_param, num_episodes=num_eps, user_model=1,basemodel="um1")
# run2_returns, run2_errors = simulate(ALPHA=a, GAMMA=g, num_interactions= num_int, egreedy_param= eg_param, num_episodes=num_eps, user_model=2,basemodel="um2")
# run3_returns, run3_errors = simulate(ALPHA=a, GAMMA=g, num_interactions= num_int, egreedy_param= eg_param, num_episodes=num_eps, user_model=3,basemodel="um3")
# run4_returns, run4_errors = simulate(ALPHA=a, GAMMA=g, num_interactions= num_int, egreedy_param= eg_param, num_episodes=num_eps, user_model=4,basemodel="um4")
                                     
run5_returns, run5_errors = simulate(ALPHA=a, GAMMA=g, num_interactions= num_int, egreedy_param= eg_param, num_episodes=num_eps, user_model=5,basemodel="um1")

# =============================================================================
# plt.plot(moving_average(run1_returns), 'b', moving_average(run2_returns), 'r', moving_average(run3_returns), 'g', moving_average(run4_returns), 'c')
# plt.legend(['base model 1', 'base model 2', 'base model 3','base model 4'])
# plt.title("Total return ")
# plt.show()
# 
# plt.plot(moving_average(run1_errors), 'b', moving_average(run2_errors), 'r', moving_average(run3_errors), 'g', moving_average(run4_errors), 'c')
# plt.legend(['base model 1', 'base model 2', 'base model 3', 'base model 4'])
# plt.title("Total error")
# plt.show()
# 
# 
# =============================================================================
