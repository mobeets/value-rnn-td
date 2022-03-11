#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:22:16 2022

@author: mobeets
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class PavlovTiming(Dataset):
    def __init__(self, ncues=4, ntrials_per_cue=50, include_reward=True):
        self.include_reward = include_reward
        self.ncues = ncues
        self.reward_times = 10 + 5*np.arange(ncues)
        self.ntrials_per_cue = ntrials_per_cue
        self.ntrials = self.ncues * self.ntrials_per_cue
        self.make_trials()

    def make_trial(self, cue, iti):
        isi = self.reward_times[cue]
        trial = np.zeros((iti + isi + 2, self.ncues + 1))
        trial[iti, cue] = 1.0 # encode stimulus
        trial[iti + isi, -1] = 1.0 # encode reward
        return trial

    def make_trials(self):
        cues = np.tile(np.arange(self.ncues), self.ntrials_per_cue)
        
        # ITI per trial
        ITIs = np.random.geometric(p=0.5, size=self.ntrials)
        
        # make trials
        self.trials = [self.make_trial(cue, iti) for cue, iti in zip(cues, ITIs)]
    
    def __getitem__(self, index):
        X = self.trials[index][:,:-1]
        y = self.trials[index][:,-1:]
        
        # augment X with previous y
        if self.include_reward:
            X = np.hstack([X, y])

        return (torch.from_numpy(X), torch.from_numpy(y))

    def __len__(self):
        return len(self.trials)
