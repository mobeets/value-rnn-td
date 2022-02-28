#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:01:56 2022

@author: mobeets
"""
from experiment import PavlovTiming
from model import ValueRNN
from train import make_dataloader, train_model, probe_model
from plotting import plot_trials, plot_loss, plot_predictions, plot_hidden_activity

#%% create experiment

E = PavlovTiming(ncues=2, include_reward=True)
plot_trials(E.trials[:15])

#%% create model

hidden_size = 2 # number of hidden neurons
gamma = 0.5 # discount rate
model = ValueRNN(input_size=E.ncues + int(E.include_reward),
            hidden_size=hidden_size, gamma=gamma)
print('model # parameters: {}'.format(model.n_parameters()))

#%% train model

lr = 0.003
batch_size = 12
dataloader = make_dataloader(E, batch_size=batch_size)
scores = train_model(model, dataloader, lr=lr)
plot_loss(scores)

#%% visualize results on example trials

responses = probe_model(model, dataloader)
plot_predictions(responses[:2], 'value', gamma=model.gamma)
plot_hidden_activity(responses[:2])
