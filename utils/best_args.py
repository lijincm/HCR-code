# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

best_args = {
    'seq-cifar10': {'sgd': {-1: {'lr': 0.1,
                                 'batch_size': 32,
                                 'n_epochs': 50}},
                    'er': {200: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 50},
                           500: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 50},
                           5120: {'lr': 0.1,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 50}},
                    'hcr': {200: {'lr': 0.01,
                                  'minibatch_size': 32,
                                  'alpha': 0.01,
                                  'beta': 0.9,
                                  'gamma': 0.5,
                                  'sigmoid': 0.01,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            500: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 0.01,
                                  'beta': 0.3,
                                  'gamma': 0.5,
                                  'sigmoid': 0.01,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            5120: {'lr': 0.03,
                                   'minibatch_size': 32,
                                   'alpha': 0.5,
                                  'beta': 0.3,
                                  'gamma': 0.5,
                                  'sigmoid': 0.5,
                                   'batch_size': 32,
                                   'n_epochs': 50}}
                   
                }
