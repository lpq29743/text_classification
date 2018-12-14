import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    sentences = df['sentences'].values
    labels = df['labels'].values
    return sentences, labels


def load_train_test_data(file_path, test_size=0.2, random_state=42):
    sentences, labels = load_raw_data(file_path)
    return train_test_spilt(sentences, labels, test_size, random_state)


def train_test_spilt(sentences, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test
