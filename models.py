import os

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd
from operator import itemgetter

TRAINED_MODELS_DIR = "trained_models"

class Coref_Basic(nn.Module):
    def __init__(self, X_data, Y_data, model_name, hidden_size=100, n_classes=3, dropout_prob=0.5, epochs=10):
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
    def __init__(self, X_data, Y_data, model_name, m1_dim=1000, m2_dim=500, dropout_prob=0.6):
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
            nn.Linear(m2_dim, m2_dim),
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
    def __init__(self, model, lr=1e-3, num_epochs=10, batch_size=50):

        self.__dict__.update(locals())
        self.num_epochs = num_epochs
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(params=[p for p in self.model.parameters()
                                            if p.requires_grad], lr=lr)

    def train(self, *args, **kwargs):
        """ Train a model """
        self.model.train()
        for epoch in range(self.num_epochs):
            shuffled_idx = list(np.random.permutation(len(self.model.X_data)))
            X_data = self.model.X_data
            Y_data = self.model.Y_data
            X_data_shuffled = [X_data[i] for i in shuffled_idx]
            Y_data_shuffled = [Y_data[i] for i in shuffled_idx]
            minibatch_idxs = np.array_split(shuffled_idx, len(shuffled_idx)/self.batch_size)
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
        output_path = os.path.join(TRAINED_MODELS_DIR, str(self.model.model_name) + time.strftime("%Y-%m-%d-%H:%M:%S"))
        torch.save(self.model, output_path)
    
    def evaluate(self, X_data, Y_data, ents, mentions, ids):
        self.model.eval()
        df_list = {'ID': [], 'A-coref': [], 'B-coref': []}
        y_pred = self.model.forward(X_data)
        correct = 0
        for i, y_pred_i in enumerate(y_pred):
            _, predicted = torch.max(y_pred_i, 0)
            ent_candidates = ents[i] + ['NULL']
            pred_candidate = ent_candidates[predicted[0]]
            if Y_data[i][predicted[0]].data == 1:
                correct += 1
                print("correct")
            else:
                print("incorrect")
            print("===")
            df_list['ID'].append(ids[i])
            print(pred_candidate)
            if pred_candidate == mentions[i][0]:
                df_list['A-coref'].append(True)
                df_list['B-coref'].append(False)
            elif pred_candidate == mentions[i][1]:
                df_list['A-coref'].append(False)
                df_list['B-coref'].append(True)
            else:
                df_list['A-coref'].append(False)
                df_list['B-coref'].append(False)
        df = pd.DataFrame(df_list)
        df.to_csv('{modelname}_epoch={epochs}_lr={lr}_batch={batch_size}_labels.tsv'.format(
            modelname=self.model.model_name,
            epochs=self.num_epochs,
            lr=self.lr,
            batch_size=self.batch_size
        ), header=False, index=False, sep='\t')
            # confusion_matrix = np.zeros((3,3))
            # for i in range(predicted.size()[0]):
            #     confusion_matrix[predicted[i],Y_data[i]] += 1
            # print(confusion_matrix)
        total = len(Y_data)
        accuracy = 100 * correct / total
        print('accuracy: {accuracy}'.format(accuracy=accuracy))