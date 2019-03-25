from sklearn.linear_model import Perceptron
from utils import data_utils, vectorize


class MyPerceptron:
    def __init__(self, vectorizer, data_fname='../../data/imdb.csv'):
        self.data_fname = data_fname
        self.vectorizer = vectorizer

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname)
        train_features, test_features = self.vectorizer.feature_extraction(X_train, X_test)
        perceptron = Perceptron()
        perceptron.fit(train_features, y_train)
        print(perceptron.score(test_features, y_test))


if __name__ == '__main__':
    vectorizer = vectorize.Vectorizer('BOW', ngram_range=(1, 1))
    model = MyPerceptron(vectorizer=vectorizer)
    model.run()
