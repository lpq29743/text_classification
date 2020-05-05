import random
import torch
import torch.nn as nn
import numpy as np
from utils import data_utils, vectorize

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(1)


class TextRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, max_seq_len, num_layers):
        super(TextRNN, self).__init__()
        self.embedding_size = embedding_matrix.shape[1]
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.embedding_layer = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1]).cuda()
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.rnn = nn.LSTM(self.embedding_size, hidden_size, num_layers, bidirectional=True, batch_first=True).cuda()
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size * 2, 2).cuda()
        self.loss = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, data_X, data_y, dropout=0.0):
        data_X = torch.cuda.LongTensor(data_X)
        data_y = torch.cuda.LongTensor(data_y)
        word_embeddings = self.embedding_layer(data_X)
        dropout_layer = nn.Dropout(dropout)
        word_embeddings = dropout_layer(word_embeddings)

        all_h, _ = self.rnn(word_embeddings)
        h = torch.mean(all_h, 1)
        h_output = self.output_layer(h)

        predict_labels = torch.argmax(h_output, -1)
        cost = self.loss(h_output, data_y)
        correct_num = (predict_labels.eq(data_y)).sum()

        return cost, correct_num


class Classifier:
    def __init__(self, vectorizer, para, data_fname='../../data/imdb.npz'):
        self.data_fname = data_fname
        self.vectorizer = vectorizer
        self.para = para

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname, is_raw=False)
        embedding_matrix = self.vectorizer.get_embedding_matrix()
        max_acc, step, stop_num = 0.0, -1, 0

        # Model Training
        model = TextRNN(embedding_matrix, self.para['hidden_size'], self.para['max_seq_len'], self.para['num_layers'])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.para['learning_rate'],
                                     weight_decay=self.para['l2_reg'])
        for i in range(self.para['epoch_num']):
            train_cost, train_acc = 0.0, 0
            for j in range(int(len(X_train) / self.para['batch_size'])):
                data_X = X_train[j * self.para['batch_size']: (j + 1) * self.para['batch_size']]
                data_y = y_train[j * self.para['batch_size']: (j + 1) * self.para['batch_size']]
                loss, correct_num = model.forward(data_X, data_y)
                train_cost += loss.item() * self.para['batch_size']
                train_acc += correct_num.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_cost, test_acc = 0.0, 0
            for j in range(int(len(X_test) / self.para['batch_size'])):
                data_X = X_test[j * self.para['batch_size']: (j + 1) * self.para['batch_size']]
                data_y = y_test[j * self.para['batch_size']: (j + 1) * self.para['batch_size']]
                loss, correct_num = model.forward(data_X, data_y)
                test_cost += loss.item() * self.para['batch_size']
                test_acc += correct_num.item()
            print('Epoch %d; Training Loss: %.3f; Training Acc: %.3f. Testing Loss: %.3f; Testing Acc: %.3f' % (
                i, train_cost / X_train.shape[0], train_acc / X_train.shape[0], test_cost / X_test.shape[0],
                test_acc / X_test.shape[0]))

            if test_acc / X_test.shape[0] > max_acc:
                max_acc = test_acc / X_test.shape[0]
                step = i
            else:
                stop_num += 1

            if stop_num == self.para['stop_num']:
                break

        print('Best Performance: %.3f at Epoch %d' % (max_acc, step))


if __name__ == '__main__':
    vectorizer = vectorize.Vectorizer('Word2Vec', emb_fname='/data/linpq/Word2Vec/glove.840B.300d.txt',
                                      word_index_fname='../../data/imdb_word_index.json')
    para = {'learning_rate': 0.001, 'l2_reg': 1e-5, 'hidden_size': 300, 'max_seq_len': 500, 'num_layers': 1,
            'epoch_num': 100, 'batch_size': 200, 'stop_num': 10}
    classifier = Classifier(vectorizer=vectorizer, para=para)
    classifier.run()
