#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:43:03 2022

@author: mobeets
"""
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    X = pad_sequence(xx, batch_first=True, padding_value=0)
    y = pad_sequence(yy, batch_first=True, padding_value=0)

    X = torch.transpose(X, 1, 0) # n.b. no longer batch_first
    y = torch.transpose(y, 1, 0)
    X = X.float()
    y = y.float()
    return X, y, x_lens

def make_dataloader(experiment, batch_size):
    return DataLoader(experiment, batch_size=batch_size, collate_fn=pad_collate)

def train_epoch(model, dataloader, loss_fn, optimizer=None, handle_padding=True):
    " if optimizer is None, no gradient steps are taken "
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    train_loss = 0
    n = 0

    for batch, (X, y, xls) in enumerate(dataloader):
        if handle_padding:
            # handle sequences with different lengths
            X = pack_padded_sequence(X, xls, enforce_sorted=False)
        
        # train TD learning
        V, (hx,cx) = model(X)

        # Compute prediction error
        V_hat = V[:-1,:,:]
        V_next = V[1:,:,:]
        V_target = y[:-1,:,:] + model.gamma*V_next.detach()

        if handle_padding:
            # do not compute loss on padded values
            loss = 0.0
            for i,l in enumerate(xls):
                loss += loss_fn(V_hat[:,i][:l], V_target[:,i][:l])
            loss /= len(xls)
        else:
            loss = loss_fn(V_hat, V_target)

        # Backpropagation
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.item()
        train_loss += loss
        n += 1
    train_loss /= n
    return train_loss

def train_model(model, dataloader, lr, nchances=4, epochs=5000, handle_padding=True, print_every=10):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scores = np.nan * np.ones((epochs,))
    best_score = np.inf
    best_weights = None
    nsteps_increase = 0
    try:
        for t in range(epochs):
            scores[t] = train_epoch(model, dataloader, loss_fn, optimizer, handle_padding)
            if t % print_every == 0:
                print(f"Epoch {t+1}, {scores[t]:0.4f}")
            if scores[t] < best_score:
                best_score = scores[t]
                best_weights = deepcopy(model.state_dict())
            if t > 0 and scores[t] > scores[t-1]:
                if nsteps_increase > nchances:
                    print("Stopping.")
                    break
                nsteps_increase += 1
            else:
                nsteps_increase = 0
    except KeyboardInterrupt:
        pass
    finally:
        scores = scores[~np.isnan(scores)]
        model.load_state_dict(best_weights)
        print(f"Done! Best loss: {best_score}")
        return scores

def probe_model(model, dataloader):
    responses = []
    model.prepare_to_gather_activity()
    with torch.no_grad():
      for batch, (X, y, xls) in enumerate(dataloader):
        X_batch = X.numpy()
        y_batch = y.numpy()
        V_batch, _ = model(X)
        V_batch = V_batch.numpy()
        Z_batch = model.features['hidden'][0].detach().numpy()
    
        # for each item in batch
        for j in range(X_batch.shape[1]):
            X = X_batch[:xls[j],j,:]
            Z = Z_batch[:xls[j],j,:]
            y = y_batch[:xls[j],j,:]
            V = V_batch[:xls[j],j,:]
            
            V_hat = V[:-1,:]
            V_next = V[1:,:]
            r = y[:-1,:]
            V_target = r + model.gamma*V_next
            rpe = V_target - V_hat
            
            # recover trial info
            cue = np.where(X[:,:-1].sum(axis=0))[0][0]
            iti = np.where(X.sum(axis=1))[0][0]
            if r.sum() > 0:
                isi = np.where(r)[0][0] - iti
            else:
                isi = None
            
            data = {'cue': cue, 'iti': iti, 'isi': isi,
                    'X': X, 'y': y, 'value': V, 'rpe': rpe,
                    'Z': Z}
            responses.append(data)
    return sorted(responses, key=lambda data: data['iti'])
