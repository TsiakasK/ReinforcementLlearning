# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:01:38 2023

@author: Anniek Jansen
"""

import numpy as np
import random as rnd


def pain_model(action, state ,user_model):
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


def user_interaction_model(pain, action, state, user_model):
    match user_model:
        case 1:
            #no interaction , interaction
            if action == 0:  # neutral
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.9, 0.1]) #neutral, no success, no pain
                    else:  # pain
                        probability = np.array([0.95, 0.05]) #neutral, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.4, 0.6]) #neutral, success, no pain
                    else:  # pain
                        probability = np.array([0.6, 0.4]) #neutral, success, pain
        
            if action == 1:  # content
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.8, 0.2]) #content, no success, no pain
                    else:  # pain
                        probability = np.array([0.9, 0.1]) #content, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.3, 0.7]) #content, success, no pain
                    else:  # pain
                        probability = np.array([0.4, 0.6]) #content, success, pain
        
            if action == 2:  # happy
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.4, 0.6]) #happy, no success, no pain
                    else:  # pain
                        probability = np.array([0.5, 0.5]) #happy, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.1, 0.9]) #happy, success, no pain
                    else:  # pain
                        probability = np.array([0.25, 0.75]) #happy, success, pain
        
            if action == 3:  # surprised
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.7, 0.3]) #surprised, no success, no pain
                    else:  # pain
                        probability = np.array([0.8, 0.2]) #surprised, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.45, 0.55]) #surprised, success, no pain
                    else:  # pain
                        probability = np.array([0.55, 0.45]) #surprised, success, pain
        
            if action == 4:  # sad
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.35, 0.65]) #sad, no success, no pain
                    else:  # pain
                        probability = np.array([0.4, 0.6]) #sad, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.2, 0.8]) #sad, success, no pain
                    else:  # pain
                        probability = np.array([0.45, 0.55]) #sad, success, pain
        
        
            if action == 5:  # angry
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.9, 0.1]) #angry, no success, no pain
                    else:  # pain
                        probability = np.array([0.9, 0.1]) #angry, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.8, 0.2]) #angry, success, no pain
                    else:  # pain
                        probability = np.array([0.85, 0.15]) #angry, success, pain
                        
        case 2:
            #no interaction , interaction
            if action == 0:  # neutral
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.5, 0.5]) #neutral, no success, no pain
                    else:  # pain
                        probability = np.array([0.6, 0.4]) #neutral, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.3, 0.7]) #neutral, success, no pain
                    else:  # pain
                        probability = np.array([0.35, 0.65]) #neutral, success, pain
        
            if action == 1:  # content
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.3, 0.7]) #content, no success, no pain
                    else:  # pain
                        probability = np.array([0.4, 0.6]) #content, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.2, 0.8]) #content, success, no pain
                    else:  # pain
                        probability = np.array([0.3, 0.7]) #content, success, pain
        
            if action == 2:  # happy
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.3, 0.7]) #happy, no success, no pain
                    else:  # pain
                        probability = np.array([0.4, 0.6]) #happy, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.1, 0.9]) #happy, success, no pain
                    else:  # pain
                        probability = np.array([0.1, 0.9]) #happy, success, pain
        
            if action == 3:  # surprised
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.2, 0.8]) #surprised, no success, no pain
                    else:  # pain
                        probability = np.array([0.3, 0.7]) #surprised, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.25, 0.75]) #surprised, success, no pain
                    else:  # pain
                        probability = np.array([0.35, 0.65]) #surprised, success, pain
        
            if action == 4:  # sad
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.1, 0.9]) #sad, no success, no pain
                    else:  # pain
                        probability = np.array([0.2, 0.8]) #sad, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.4, 0.6]) #sad, success, no pain
                    else:  # pain
                        probability = np.array([0.45, 0.55]) #sad, success, pain
        
        
            if action == 5:  # angry
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.6, 0.4]) #angry, no success, no pain
                    else:  # pain
                        probability = np.array([0.55, 0.45]) #angry, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.4, 0.6]) #angry, success, no pain
                    else:  # pain
                        probability = np.array([0.5, 0.5]) #angry, success, pain

        case 3:
            #no interaction , interaction
            if action == 0:  # neutral
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.9, 0.1]) #neutral, no success, no pain
                    else:  # pain
                        probability = np.array([0.95, 0.05]) #neutral, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.5, 0.5]) #neutral, success, no pain
                    else:  # pain
                        probability = np.array([0.7, 0.3]) #neutral, success, pain
        
            if action == 1:  # content
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.9, 0.1]) #content, no success, no pain
                    else:  # pain
                        probability = np.array([0.95, 0.05]) #content, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.4, 0.6]) #content, success, no pain
                    else:  # pain
                        probability = np.array([0.5, 0.5]) #content, success, pain
        
            if action == 2:  # happy
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.6, 0.4]) #happy, no success, no pain
                    else:  # pain
                        probability = np.array([0.8, 0.2]) #happy, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.4, 0.6]) #happy, success, no pain
                    else:  # pain
                        probability = np.array([0.45, 0.55]) #happy, success, pain
        
            if action == 3:  # surprised
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.8, 0.2]) #surprised, no success, no pain
                    else:  # pain
                        probability = np.array([0.9, 0.1]) #surprised, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.6, 0.4]) #surprised, success, no pain
                    else:  # pain
                        probability = np.array([0.7, 0.3]) #surprised, success, pain
        
            if action == 4:  # sad
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.65, 0.35]) #sad, no success, no pain
                    else:  # pain
                        probability = np.array([0.8, 0.2]) #sad, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.4, 0.6]) #sad, success, no pain
                    else:  # pain
                        probability = np.array([0.45, 0.55]) #sad, success, pain
        
        
            if action == 5:  # angry
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.9, 0.1]) #angry, no success, no pain
                    else:  # pain
                        probability = np.array([0.9, 0.1]) #angry, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.8, 0.2]) #angry, success, no pain
                    else:  # pain
                        probability = np.array([0.95, 0.05]) #angry, success, pain
                                
        case 4:
            #no interaction , interaction
            if action == 0:  # neutral
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.95, 0.05]) #neutral, no success, no pain
                    else:  # pain
                        probability = np.array([0.6, 0.4]) #neutral, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.4, 0.6]) #neutral, success, no pain
                    else:  # pain
                        probability = np.array([0.65, 0.35]) #neutral, success, pain
        
            if action == 1:  # content
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.85, 0.15]) #content, no success, no pain
                    else:  # pain
                        probability = np.array([0.9, 0.1]) #content, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.2, 0.8]) #content, success, no pain
                    else:  # pain
                        probability = np.array([0.4, 0.6]) #content, success, pain
        
            if action == 2:  # happy
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.45, 0.55]) #happy, no success, no pain
                    else:  # pain
                        probability = np.array([0.5, 0.5]) #happy, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.1, 0.9]) #happy, success, no pain
                    else:  # pain
                        probability = np.array([0.2, 0.8]) #happy, success, pain
        
            if action == 3:  # surprised
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.8, 0.2]) #surprised, no success, no pain
                    else:  # pain
                        probability = np.array([0.9, 0.1]) #surprised, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.55, 0.45]) #surprised, success, no pain
                    else:  # pain
                        probability = np.array([0.6, 0.4]) #surprised, success, pain
        
            if action == 4:  # sad
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.9, 0.1]) #sad, no success, no pain
                    else:  # pain
                        probability = np.array([0.9, 0.1]) #sad, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.9, 0.1]) #sad, success, no pain
                    else:  # pain
                        probability = np.array([0.9, 0.1]) #sad, success, pain
        
        
            if action == 5:  # angry
                #if previous no success
                if state[4]==0:
                    if pain==0: #no pain
                        probability = np.array([0.9, 0.1]) #angry, no success, no pain
                    else:  # pain
                        probability = np.array([0.95, 0.05]) #angry, no success, pain
                else: #previous success
                    if pain==0: #no pain
                        probability = np.array([0.6, 0.4]) #angry, success, no pain
                    else:  # pain
                        probability = np.array([0.9, 0.1]) #angry, success, pain
                        

       
        
    interaction =np.random.choice([0,1], p= probability, size=1)[0]

    return interaction
