#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from dataset import ViTacDataset
from torch import nn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import tqdm
import argparse


class FLAGS():
    def __init__(self):
        self.data_dir = '/home/tasbolat/some_python_examples/data_VT_SNN/'
        self.batch_size = 8
        self.sample_file = 5
args = FLAGS()

# parser = argparse.ArgumentParser("Train model.")
# parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)

# parser.add_argument(
#     "--sample_file", type=int, help="Sample number to train from.", required=True
# )
# parser.add_argument(
#     "--batch_size", type=int, help="Batch Size.", required=True
# )
# args = parser.parse_args()


# In[29]:


train_dataset = ViTacDataset(
    path=args.data_dir, sample_file=f"train_80_20_{args.sample_file}.txt"
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_dataset = ViTacDataset(
    path=args.data_dir, sample_file=f"test_80_20_{args.sample_file}.txt"
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)


big_train_dataset = []#torch.zeros(240, 156, 325)
labels = []#torch.zeros(240)
for i, (in_tact, _, _, label) in enumerate(train_loader, 0):
    big_train_dataset.append(in_tact.squeeze())
    labels.append(label)

    

big_test_dataset = []#torch.zeros(240, 156, 325)
labels_test = []#torch.zeros(240)
for i, (in_tact, _, _, label) in enumerate(test_loader, 0):
    big_test_dataset.append(in_tact.squeeze())
    labels_test.append(label)
    
big_train_dataset = torch.cat(big_train_dataset,0)
big_test_dataset = torch.cat(big_test_dataset,0)
big_train_dataset.shape, big_test_dataset.shape


lengths_ = np.linspace(10,320,32, dtype=int)
lengths = lengths_.tolist()
lengths.append(325)


y = torch.cat(labels).cpu().numpy()
y_test = torch.cat(labels_test).cpu().numpy()

# prepare parameters
skf = StratifiedKFold(n_splits=12)
param_grid = {'C':np.linspace(0.000001, 2.1, 3000),} #10000

# collect statistics
cv_std = []
cv_mean = []
test_accs = []
C_params = []

for length in tqdm.tqdm(lengths):
    
    # get data
    X = torch.sum(big_train_dataset[...,:length], dim=2).cpu().numpy()
    X_test = torch.sum(big_test_dataset[...,:length], dim=2).cpu().numpy()
    
    # define clf
    svc = SVC(tol=0.00001, max_iter=5000, kernel='linear')
    search = GridSearchCV(svc, param_grid, cv=skf, n_jobs=-1, scoring='accuracy')
    search.fit(X, y)
    
    # collect statistics
    cv_std.append( search.cv_results_['std_test_score'][search.best_index_] )
    cv_mean.append(search.cv_results_['mean_test_score'][search.best_index_])
    C_params.append(search.best_estimator_.C)
    
    # get test result based on the best estimator
    y_test_pred = search.best_estimator_.predict(X_test)
    test_accs.append(accuracy_score(y_test, y_test_pred))

import pickle
pickle.dump([lengths, cv_std, cv_mean, test_accs, C_params], open('svm_results_' + str(args.sample_file) + '.pkl', 'wb'))