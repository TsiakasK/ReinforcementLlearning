# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:45:26 2023

@author: Anniek Jansen
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:01:38 2023

@author: Anniek Jansen
"""




import numpy as np
import random as rnd
def pain_model(action, state , user_model):
    # pain_model
    match user_model:
        case 1:
            prob_topain = 0.1
            prob_tonopain = 0.2
        case 2:
            prob_topain = 0.2
            prob_tonopain = 0.1
        case 3:
            prob_topain = 0.01
            prob_tonopain = 0.05
        case 4:
            prob_topain = 0.05
            prob_tonopain = 0.1

    if state[0] == 0:  # e.g. if not in pain, transition with 0.1 prob to pain
        if rnd.random() <= prob_topain:
            pain = 1
        else:
            pain = 0
    else:  # if in pain, transition with 0.20 prob to no pain
        if rnd.random() <= prob_tonopain:
            pain = 1
        else:
            pain = 0

    return pain


def user_interaction_model2(pain, action, state, user_model, prev_action):
    match user_model:
        case 1:
            # no interaction , interaction
            if action == 0:  # neutral
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # neutral, no success, no pain
                        probability = np.array([0.9, 0.1])
                    else:  # pain
                        # neutral, no success, pain
                        probability = np.array([0.95, 0.05])
                else:  # previous success
                    if pain == 0:  # no pain
                        # neutral, success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # neutral, success, pain
                        probability = np.array([0.6, 0.4])

            if action == 1:  # content
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # content, no success, no pain
                        probability = np.array([0.8, 0.2])
                    else:  # pain
                        # content, no success, pain
                        probability = np.array([0.9, 0.1])
                else:  # previous success
                    if pain == 0:  # no pain
                        # content, success, no pain
                        probability = np.array([0.3, 0.7])
                    else:  # pain
                        # content, success, pain
                        probability = np.array([0.4, 0.6])

            if action == 2:  # happy
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # happy, no success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # happy, no success, pain
                        probability = np.array([0.5, 0.5])
                else:  # previous success
                    if pain == 0:  # no pain
                        # happy, success, no pain
                        probability = np.array([0.1, 0.9])
                    else:  # pain
                        # happy, success, pain
                        probability = np.array([0.25, 0.75])

            if action == 3:  # surprised
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # surprised, no success, no pain
                        probability = np.array([0.7, 0.3])
                    else:  # pain
                        # surprised, no success, pain
                        probability = np.array([0.8, 0.2])
                else:  # previous success
                    if pain == 0:  # no pain
                        # surprised, success, no pain
                        probability = np.array([0.45, 0.55])
                    else:  # pain
                        # surprised, success, pain
                        probability = np.array([0.55, 0.45])

            if action == 4:  # sad
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # sad, no success, no pain
                        probability = np.array([0.35, 0.65])
                    else:  # pain
                        # sad, no success, pain
                        probability = np.array([0.4, 0.6])
                else:  # previous success
                    if pain == 0:  # no pain
                        # sad, success, no pain
                        probability = np.array([0.2, 0.8])
                    else:  # pain
                        # sad, success, pain
                        probability = np.array([0.45, 0.55])

            if action == 5:  # angry
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # angry, no success, no pain
                        probability = np.array([0.9, 0.1])
                    else:  # pain
                        # angry, no success, pain
                        probability = np.array([0.9, 0.1])
                else:  # previous success
                    if pain == 0:  # no pain
                        # angry, success, no pain
                        probability = np.array([0.8, 0.2])
                    else:  # pain
                        # angry, success, pain
                        probability = np.array([0.85, 0.15])

        case 2:
            # no interaction , interaction
            if action == 0:  # neutral
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # neutral, no success, no pain
                        probability = np.array([0.5, 0.5])
                    else:  # pain
                        # neutral, no success, pain
                        probability = np.array([0.6, 0.4])
                else:  # previous success
                    if pain == 0:  # no pain
                        # neutral, success, no pain
                        probability = np.array([0.3, 0.7])
                    else:  # pain
                        # neutral, success, pain
                        probability = np.array([0.35, 0.65])

            if action == 1:  # content
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # content, no success, no pain
                        probability = np.array([0.3, 0.7])
                    else:  # pain
                        # content, no success, pain
                        probability = np.array([0.4, 0.6])
                else:  # previous success
                    if pain == 0:  # no pain
                        # content, success, no pain
                        probability = np.array([0.2, 0.8])
                    else:  # pain
                        # content, success, pain
                        probability = np.array([0.3, 0.7])

            if action == 2:  # happy
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # happy, no success, no pain
                        probability = np.array([0.3, 0.7])
                    else:  # pain
                        # happy, no success, pain
                        probability = np.array([0.4, 0.6])
                else:  # previous success
                    if pain == 0:  # no pain
                        # happy, success, no pain
                        probability = np.array([0.1, 0.9])
                    else:  # pain
                        # happy, success, pain
                        probability = np.array([0.1, 0.9])

            if action == 3:  # surprised
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # surprised, no success, no pain
                        probability = np.array([0.2, 0.8])
                    else:  # pain
                        # surprised, no success, pain
                        probability = np.array([0.3, 0.7])
                else:  # previous success
                    if pain == 0:  # no pain
                        # surprised, success, no pain
                        probability = np.array([0.25, 0.75])
                    else:  # pain
                        # surprised, success, pain
                        probability = np.array([0.35, 0.65])

            if action == 4:  # sad
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # sad, no success, no pain
                        probability = np.array([0.1, 0.9])
                    else:  # pain
                        # sad, no success, pain
                        probability = np.array([0.2, 0.8])
                else:  # previous success
                    if pain == 0:  # no pain
                        # sad, success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # sad, success, pain
                        probability = np.array([0.45, 0.55])

            if action == 5:  # angry
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # angry, no success, no pain
                        probability = np.array([0.6, 0.4])
                    else:  # pain
                        # angry, no success, pain
                        probability = np.array([0.55, 0.45])
                else:  # previous success
                    if pain == 0:  # no pain
                        # angry, success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # angry, success, pain
                        probability = np.array([0.5, 0.5])

        case 3:
            # no interaction , interaction
            if action == 0:  # neutral
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # neutral, no success, no pain
                        probability = np.array([0.9, 0.1])
                    else:  # pain
                        # neutral, no success, pain
                        probability = np.array([0.95, 0.05])
                else:  # previous success
                    if pain == 0:  # no pain
                        # neutral, success, no pain
                        probability = np.array([0.5, 0.5])
                    else:  # pain
                        # neutral, success, pain
                        probability = np.array([0.7, 0.3])

            if action == 1:  # content
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # content, no success, no pain
                        probability = np.array([0.9, 0.1])
                    else:  # pain
                        # content, no success, pain
                        probability = np.array([0.95, 0.05])
                else:  # previous success
                    if pain == 0:  # no pain
                        # content, success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # content, success, pain
                        probability = np.array([0.5, 0.5])

            if action == 2:  # happy
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # happy, no success, no pain
                        probability = np.array([0.6, 0.4])
                    else:  # pain
                        # happy, no success, pain
                        probability = np.array([0.8, 0.2])
                else:  # previous success
                    if pain == 0:  # no pain
                        # happy, success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # happy, success, pain
                        probability = np.array([0.45, 0.55])

            if action == 3:  # surprised
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # surprised, no success, no pain
                        probability = np.array([0.8, 0.2])
                    else:  # pain
                        # surprised, no success, pain
                        probability = np.array([0.9, 0.1])
                else:  # previous success
                    if pain == 0:  # no pain
                        # surprised, success, no pain
                        probability = np.array([0.6, 0.4])
                    else:  # pain
                        # surprised, success, pain
                        probability = np.array([0.7, 0.3])

            if action == 4:  # sad
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # sad, no success, no pain
                        probability = np.array([0.65, 0.35])
                    else:  # pain
                        # sad, no success, pain
                        probability = np.array([0.8, 0.2])
                else:  # previous success
                    if pain == 0:  # no pain
                        # sad, success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # sad, success, pain
                        probability = np.array([0.45, 0.55])

            if action == 5:  # angry
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # angry, no success, no pain
                        probability = np.array([0.9, 0.1])
                    else:  # pain
                        # angry, no success, pain
                        probability = np.array([0.9, 0.1])
                else:  # previous success
                    if pain == 0:  # no pain
                        # angry, success, no pain
                        probability = np.array([0.8, 0.2])
                    else:  # pain
                        # angry, success, pain
                        probability = np.array([0.95, 0.05])

        case 4:
            # no interaction , interaction
            if action == 0:  # neutral
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # neutral, no success, no pain
                        probability = np.array([0.95, 0.05])
                    else:  # pain
                        # neutral, no success, pain
                        probability = np.array([0.6, 0.4])
                else:  # previous success
                    if pain == 0:  # no pain
                        # neutral, success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # neutral, success, pain
                        probability = np.array([0.65, 0.35])

            if action == 1:  # content
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # content, no success, no pain
                        probability = np.array([0.85, 0.15])
                    else:  # pain
                        # content, no success, pain
                        probability = np.array([0.9, 0.1])
                else:  # previous success
                    if pain == 0:  # no pain
                        # content, success, no pain
                        probability = np.array([0.2, 0.8])
                    else:  # pain
                        # content, success, pain
                        probability = np.array([0.4, 0.6])

            if action == 2:  # happy
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # happy, no success, no pain
                        probability = np.array([0.45, 0.55])
                    else:  # pain
                        # happy, no success, pain
                        probability = np.array([0.5, 0.5])
                else:  # previous success
                    if pain == 0:  # no pain
                        # happy, success, no pain
                        probability = np.array([0.1, 0.9])
                    else:  # pain
                        # happy, success, pain
                        probability = np.array([0.2, 0.8])

            if action == 3:  # surprised
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # surprised, no success, no pain
                        probability = np.array([0.8, 0.2])
                    else:  # pain
                        # surprised, no success, pain
                        probability = np.array([0.9, 0.1])
                else:  # previous success
                    if pain == 0:  # no pain
                        # surprised, success, no pain
                        probability = np.array([0.55, 0.45])
                    else:  # pain
                        # surprised, success, pain
                        probability = np.array([0.6, 0.4])

            if action == 4:  # sad
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # sad, no success, no pain
                        probability = np.array([0.9, 0.1])
                    else:  # pain
                        # sad, no success, pain
                        probability = np.array([0.9, 0.1])
                else:  # previous success
                    if pain == 0:  # no pain
                        # sad, success, no pain
                        probability = np.array([0.9, 0.1])
                    else:  # pain
                        # sad, success, pain
                        probability = np.array([0.9, 0.1])

            if action == 5:  # angry
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # angry, no success, no pain
                        probability = np.array([0.9, 0.1])
                    else:  # pain
                        # angry, no success, pain
                        probability = np.array([0.95, 0.05])
                else:  # previous success
                    if pain == 0:  # no pain
                        # angry, success, no pain
                        probability = np.array([0.6, 0.4])
                    else:  # pain
                        # angry, success, pain
                        probability = np.array([0.9, 0.1])
                        
        case 5:
            # no interaction , interaction
            if action == 0:  # neutral
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # neutral, no success, no pain
                        probability = np.array([0.8, 0.2])
                    else:  # pain
                        # neutral, no success, pain
                        probability = np.array([0.7, 0.3])
                else:  # previous success
                    if pain == 0:  # no pain
                        # neutral, success, no pain
                        probability = np.array([0.6, 0.4])
                    else:  # pain
                        # neutral, success, pain
                        probability = np.array([0.5, 0.5])
        
            if action == 1:  # content
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # content, no success, no pain
                        probability = np.array([0.7, 0.3])
                    else:  # pain
                        # content, no success, pain
                        probability = np.array([0.6, 0.4])
                else:  # previous success
                    if pain == 0:  # no pain
                        # content, success, no pain
                        probability = np.array([0.5, 0.5])
                    else:  # pain
                        # content, success, pain
                        probability = np.array([0.4, 0.6])
        
            if action == 2:  # happy
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # happy, no success, no pain
                        probability = np.array([0.6, 0.4])
                    else:  # pain
                        # happy, no success, pain
                        probability = np.array([0.5, 0.5])
                else:  # previous success
                    if pain == 0:  # no pain
                        # happy, success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # happy, success, pain
                        probability = np.array([0.3, 0.7])
        
            if action == 3:  # surprised
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # surprised, no success, no pain
                        probability = np.array([0.5, 0.5])
                    else:  # pain
                        # surprised, no success, pain
                        probability = np.array([0.4, 0.6])
                else:  # previous success
                    if pain == 0:  # no pain
                        # surprised, success, no pain
                        probability = np.array([0.3, 0.7])
                    else:  # pain
                        # surprised, success, pain
                        probability = np.array([0.2, 0.8])
        
            if action == 4:  # sad
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # sad, no success, no pain
                        probability = np.array([0.4, 0.6])
                    else:  # pain
                        # sad, no success, pain
                        probability = np.array([0.3, 0.7])
                else:  # previous success
                    if pain == 0:  # no pain
                        # sad, success, no pain
                        probability = np.array([0.2, 0.8])
                    else:  # pain
                        # sad, success, pain
                        probability = np.array([0.1, 0.9])
        
            if action == 5:  # angry
                # if previous no success
                if state[3] == 0:
                    if pain == 0:  # no pain
                        # angry, no success, no pain
                        probability = np.array([0.9, 0.1])
                    else:  # pain
                        # angry, no success, pain
                        probability = np.array([0.8, 0.2])
                else:  # previous success
                    if pain == 0:  # no pain
                        # angry, success, no pain
                        probability = np.array([0.7, 0.3])
                    else:  # pain
                        # angry, success, pain
                        probability = np.array([0.8, 0.2])


    #make it less likely that previous action will succeed if it is the same action and it did not succeed the last time
    #print(prev_action, action)
    if prev_action == action and state[3] == 0:
        probability[0] = probability[0] + 0.8*probability[1]
        probability[1] = 0.2*probability[1]
        #print(probability)

    interaction = np.random.choice([0, 1], p=probability, size=1)[0]

    return interaction
