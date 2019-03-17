import os

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np
import time
from operator import itemgetter

TRAINED_MODELS_DIR = "trained_models"

class Coref_Basic(nn.Module):
    def __init__(self, X_data, Y_data, model_name, hidden_size=100, n_classes=3, dropout_prob=0.5, epochs=20):
        super().__init__()
        self.X_data = X_data
        self.Y_data = Y_data
        self.model_name = model_name
        self.n_features = len(self.X_data[0])
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.epochs = epochs
        self.embed_to_hidden = nn.Linear(self.n_features, self.hidden_size)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, X):
        h = F.relu(self.embed_to_hidden(X))
        d = self.dropout(h)
        logits = self.hidden_to_logits(d)
        return F.softmax(logits, dim=1)


class MentionPairScore(nn.Module):
    def __init__(self, X_data, Y_data, model_name, m1_dim=1000, m2_dim=500, dropout_prob=0.5):
        super().__init__()
        self.model_name = model_name
        self.X_data = X_data
        self.Y_data = Y_data
        print(self.X_data[0].size()[1])
        self.score = nn.Sequential(
            nn.Linear(self.X_data[0].size()[1], m1_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_prob),
            nn.Linear(m1_dim, m2_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_prob),
            nn.Linear(m2_dim, 1)
        )

    def forward(self, X):
        # for a specific sentence we find the scores of all of the 
        # noun chunks
        X_tensorized = torch.cat(X)
        score = self.score(X_tensorized.float())
        logits_list = []
        while X:
            sent_len = X[0].size()[0]
            epsilon = torch.tensor([[0.]]).requires_grad_()
            score_with_epsilon = torch.cat((score[:sent_len, :], epsilon))
            sent_logits = F.softmax(score_with_epsilon, dim=0)
            logits_list.append(sent_logits)
            X = X[1:]
            score = score[sent_len:, :]
        logits = torch.cat(logits_list)
        return logits_list


class Trainer:
    def __init__(self, model, lr=1e-3, l2_reg=1e-3, num_epochs=20):

        self.__dict__.update(locals())
        self.num_epochs = num_epochs
        self.model = model

        self.optimizer = torch.optim.RMSprop(params=[p for p in self.model.parameters()
                                            if p.requires_grad],
                                    lr=lr, weight_decay=l2_reg)

    def train(self, batch_size=100, *args, **kwargs):
        """ Train a model """
        self.model.train()
        for epoch in range(self.num_epochs):
            shuffled_idx = list(np.random.permutation(len(self.model.X_data)))
            X_data = self.model.X_data
            Y_data = self.model.Y_data
            X_data_shuffled = [X_data[i] for i in shuffled_idx]
            Y_data_shuffled = [Y_data[i] for i in shuffled_idx]
            minibatch_idxs = np.array_split(shuffled_idx, len(shuffled_idx)/batch_size)
            for batch_no, batch_idxs in enumerate(minibatch_idxs):
                getter = itemgetter(*batch_idxs)
                X_batch = getter(X_data_shuffled)
                # Y_batch = torch.unsqueeze(torch.cat(getter(Y_data_shuffled)), dim=0)
                Y_batch = getter(Y_data_shuffled)
                y_pred = self.model(X_batch)
                eps = 1e-8
                loss = -1*torch.sum(torch.cat([torch.log(torch.matmul(torch.unsqueeze(Y_batch[i], dim=0), y_pred[i]).clamp(eps, 1-eps)) for i, s in enumerate(Y_batch)]))
                print('Epoch {epoch}, Example {batch_no}, Loss {loss}'.format(epoch=epoch, batch_no=batch_no, loss=loss.detach()))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        output_path = os.path.join(TRAINED_MODELS_DIR, self.model.model_name, time.strftime("%Y-%m-%d-%H:%M:%S"))
        torch.save(self.model, output_path)