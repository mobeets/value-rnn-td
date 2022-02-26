#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:43:03 2022

@author: mobeets
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

class ValueRNN(nn.Module):
    def __init__(self, input_size=4, output_size=1, hidden_size=15, 
                 num_layers=1, gamma=0.9):
      super(ValueRNN, self).__init__()

      self.gamma = gamma
      self.input_size = input_size
      self.output_size = output_size
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
      self.value = lambda x: torch.sum(x,2)[:,:,None]
      # self.value = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x):
        x, (hx, cx) = self.rnn(x)
        if type(x) is torch.nn.utils.rnn.PackedSequence:
            x, output_lengths = pad_packed_sequence(x, batch_first=False)
        x = F.relu(x)
        return self.value(x), (hx, cx)

    def freeze_weights(self):
        for name, p in self.named_parameters():
            p.requires_grad = False
    
    def unfreeze_weights(self):
        for name, p in self.named_parameters():
            p.requires_grad = True

    def n_parameters(self):
        return sum([p.numel() for p in self.parameters()])
    
    def get_features(self, name):
        def hook(mdl, input, output):
            self.features[name] = output
        return hook
    
    def prepare_to_gather_activity(self):
        if hasattr(self, 'handle'):
            self.handle.remove()
        self.features = {}
        self.hook = self.get_features('hidden')
        self.handle = self.rnn.register_forward_hook(self.hook)
