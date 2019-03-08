import os

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

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

def train(model, batch_size=100):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(model.epochs):
        shuffled_idx = torch.randperm(model.X_data.size()[0])
        X_data = model.X_data[shuffled_idx]
        Y_data = model.Y_data[shuffled_idx]
        minibatches_x = torch.split(X_data, batch_size)
        minibatches_y = torch.split(Y_data, batch_size)
        minibatches = [(minibatches_x, minibatches_y) for i,d in enumerate(minibatches_x)]
        for batch in minibatches:
            X_batch = batch[0]
            Y_batch = batch[1]
        # y_pred = model(self.X_data)
            y_pred = model(X_data)
            loss = criterion(y_pred, Y_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {epoch}, loss: {loss}'.format(epoch=epoch, loss=loss))
        _, predicted = torch.max(y_pred.data, 1)
        total = len(model.Y_data)
        correct = (predicted == Y_data).sum()
        accuracy = 100 * correct / total
        print('accuracy: {accuracy}'.format(accuracy=accuracy))
    output_path = os.path.join(TRAINED_MODELS_DIR, model.model_name)
    torch.save(model, output_path)