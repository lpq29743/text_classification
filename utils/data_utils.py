import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def load_raw_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    sentences = df['sentences'].values
    labels = df['labels'].values
    return sentences, labels


def load_processed_data(file_path, seq_len=500, seed=42):
    data = np.load(file_path)
    X_train, X_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']

    rng = np.random.RandomState(seed)
    indices = np.arange(len(X_train))
    rng.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    indices = np.arange(len(X_test))
    rng.shuffle(indices)
    X_test = X_test[indices]
    y_test = y_test[indices]

    X_train = pad_sequences(X_train, maxlen=seq_len, value=0)
    X_test = pad_sequences(X_test, maxlen=seq_len, value=0)
    X_train = X_train.reshape(-1, seq_len)
    X_test = X_test.reshape(-1, seq_len)
    return X_train, X_test, y_train, y_test


def load_train_test_data(file_path, is_raw=True, test_size=0.2, random_state=42):
    if is_raw:
        sentences, labels = load_raw_data(file_path)
        return train_test_spilt(sentences, labels, test_size, random_state)
    else:
        return load_processed_data(file_path)


def train_test_spilt(sentences, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test
