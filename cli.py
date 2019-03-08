import os

import fire
import torch
import pandas as pd
import numpy as np

from models import Coref_Basic, MLPAttention, train, TRAINED_MODELS_DIR
import utils

class CLI(object):
    """CLI object for gap-coreference project."""
    
    def prepare_data(self, dataset, reload=False):
        utils.prepare_data(dataset, reload=reload)

    def load_data(dataset):
        print(utils.load_data(dataset))

    def train_basic(self, dataset):
        model_name = "Coref_Basic-{}.pt".format(dataset)
        ids, X_data, Y_data = utils.load_data(dataset)
        model = Coref_Basic(X_data, Y_data, model_name)
        train(model)

    def label_to_boolean(self, mention, val):
        if ((mention == 'A' and val==0) or (mention == 'B' and val==1)):
            return True
        else:
            return False

    def print_to_file(self, y_pred, ids):
        df = pd.DataFrame()
        _, predicted = torch.max(y_pred.data, 1)
        A_coref = [self.label_to_boolean('A', p) for p in predicted]
        B_coref = [self.label_to_boolean('B', p) for p in predicted]
        df['ID'] = ids
        df['A-coref'] = A_coref
        df['B-coref'] = B_coref
        df.to_csv('Coref_Basic-test_system_labels.tsv', header=False, index=False, sep='\t')


    def evaluate_basic(self):
        ids, X_data, Y_data = utils.load_data("test")
        model = torch.load(os.path.join(TRAINED_MODELS_DIR, "Coref_Basic-development.pt"))
        model.eval()
        y_pred = model.forward(X_data) 
        print(y_pred)
        print(Y_data)
        _, predicted = torch.max(y_pred.data, 1)
        print("===")
        confusion_matrix = np.zeros((3,3))
        for i in range(predicted.size()[0]):
            confusion_matrix[predicted[i],Y_data[i]] += 1
        print(confusion_matrix)
        total = len(Y_data)
        correct = (predicted == Y_data).sum()
        accuracy = 100 * correct / total
        print('accuracy: {accuracy}'.format(accuracy=accuracy))
        self.print_to_file(y_pred, ids)


if __name__ == '__main__':
    fire.Fire(CLI)