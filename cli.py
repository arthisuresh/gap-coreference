import os

import fire
import torch
import pandas as pd
import numpy as np
import pickle

from models import Coref_Basic, MentionPairScore, Trainer, TRAINED_MODELS_DIR
from gap_scorer import *
from collections import defaultdict
import utils

class CLI(object):
    """CLI object for gap-coreference project."""
    
    def prepare_data(self, dataset, setting='pairs', reload=False):
        print("Hitting here")
        utils.prepare_data(dataset, setting, reload=reload)

    def load_data(dataset, setting):
        print(utils.load_data(dataset, setting))
        
    def train_basic(self, dataset, modelname, setting, lr, n_epochs, batch_size):
        ids, X_data, Y_data, ents, gold_mentions = utils.load_data(dataset, setting)
        # we need to predict mention pairs
        # in evaluation we need to output the top scoring mention
        model = MentionPairScore(X_data, Y_data, model_name=modelname)
        trainer = Trainer(model, lr=lr, num_epochs=n_epochs, batch_size=batch_size)
        trainer.train()
        ids_val, X_data_val, Y_data_val, ents_val, gold_mentions_val = utils.load_data("test", setting)
        trainer.evaluate(X_data_val, Y_data_val, ents_val, gold_mentions_val, ids_val)

    def label_to_boolean(self, mention, val):
        if ((mention == 'A' and val==0) or (mention == 'B' and val==1)):
            return True
        else:
            return False

    def safe_divide(self, x, y):
        if y == 0:
            return 0
        return x / y

    def print_to_file(self, y_pred, ids):
        df = pd.DataFrame()
        _, predicted = torch.max(y_pred.data, 1)
        A_coref = [self.label_to_boolean('A', p) for p in predicted]
        B_coref = [self.label_to_boolean('B', p) for p in predicted]
        df['ID'] = ids
        df['A-coref'] = A_coref
        df['B-coref'] = B_coref
        df.to_csv('Coref_Basic-test_system_labels.tsv', header=False, index=False, sep='\t')
        
    def tune_hyper_parameters(self, modelname):
        lr_options = [0.01, 0.001, 0.0005]
        epoch_options = [5, 10, 20]
        batch_size_options = [5, 10, 20, 50, 100, 200]
        # lr_options = [0.001, 0.005]
        # epoch_options = [2]
        # batch_size_options = [100]
        scores_dict = defaultdict()
        best_combo = []
        best_overall_f1 = 0
        for lr in lr_options:
            scores_dict[lr] = defaultdict()
            for n_epochs in epoch_options:
                scores_dict[lr][n_epochs] = defaultdict()
                for batch_size in batch_size_options:
                    filename = '{modelname}_epoch={epochs}_lr={lr}_batch={batch_size}_labels.tsv'.format(
                        modelname=modelname,
                        epochs=n_epochs,
                        lr=lr,
                        batch_size=batch_size
                    )
                    if not os.path.exists(filename):
                        self.train_basic('development', modelname, 'pairs', lr, n_epochs, batch_size)
                        scorecard, scores = run_scorer('gap-validation.tsv', filename)
                        print(scores)
                        f_gender_scores = scores.get(Gender.FEMININE, Scores())
                        m_gender_scores = scores.get(Gender.MASCULINE, Scores())
                        overall_scores = scores.get(None, Scores())
                        f_recall = f_gender_scores.recall()
                        f_precision = f_gender_scores.precision()
                        f_f1 = f_gender_scores.f1()
                        m_recall = m_gender_scores.recall()
                        m_precision = m_gender_scores.precision()
                        m_f1 = m_gender_scores.f1()
                        o_recall = overall_scores.recall()
                        o_precision = overall_scores.precision()
                        o_f1 = overall_scores.f1()
                        bias = self.safe_divide(f_f1, m_f1)
                        scores_dict[lr][n_epochs][batch_size] = {
                            'Recall (F)': f_recall,
                            'Recall (M)': m_recall,
                            'Recall (O)': o_recall,
                            'Precision (F)': f_precision,
                            'Precision (M)': m_precision,
                            'Precision (O)': o_precision,
                            'F1 (F)': f_f1,
                            'F1 (M)': m_f1,
                            'F1 (O)': o_f1,
                            'Bias': bias
                        }
                        if o_f1 > best_overall_f1:
                            best_overall_f1 = o_f1
                            best_combo = [lr, n_epochs, batch_size]
        pickle.dump(scores_dict, open("hyperparameter_tuning_scores.p", "wb"))
        print(best_combo)




    def evaluate_basic(self, modelpath, dataset):
        ids, X_data, Y_data = utils.load_data(dataset)
        print(len(X_data))
        print(X_data[0].size())
        # model = MentionPairScore(X_data, Y_data, model_name='MentionPairScore')
        # model = torch.nn.DataParallel(model)
        model = torch.load(os.path.join(TRAINED_MODELS_DIR, "MentionPairScore2019-03-17-00:56:16"))
        # model.load_state_dict(state_dict['model'])
        model.eval()
        y_pred = model.forward(X_data)
        correct = 0
        for i, y_pred_i in enumerate(y_pred):
            _, predicted = torch.max(y_pred_i, 0)
            print(Y_data[i])
            print(predicted)
            if Y_data[i][predicted[0]].data == 1:
                correct += 1
                print("correct")
            else:
                print("incorrect")
            print("===")
            # confusion_matrix = np.zeros((3,3))
            # for i in range(predicted.size()[0]):
            #     confusion_matrix[predicted[i],Y_data[i]] += 1
            # print(confusion_matrix)
        total = len(Y_data)
        accuracy = 100 * correct / total
        print('accuracy: {accuracy}'.format(accuracy=accuracy))
        # self.print_to_file(y_pred, ids)


if __name__ == '__main__':
    fire.Fire(CLI)

# to load data:
# python cli.py prepare_data --dataset=small --setting=pairs