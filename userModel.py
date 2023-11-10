# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:31:41 2023

@author: Anniek Jansen
"""
import numpy as np
import random as rnd


def pain_model(action, state):
    # pain_model
    if state[0] == 0:  # e.g. if not in pain, transition with 0.1 prob to pain
        if rnd.random() <= 0.1:
            pain = 1
        else:
            pain = 0
    else:  # if in pain, transition with 0.20 prob to no pain
        if rnd.random() <= 0.2:
            pain = 1
        else:
            pain = 0
            
    return pain


def user_interaction_model(pain, action, state):
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

       
        
    interaction =np.random.choice([0,1], p= probability, size=1)[0]

    return interaction
