from sklearn.ensemble import RandomForestClassifier
from utils import data_utils, vectorize


class RandomForest:
    def __init__(self, data_fname='../data/imdb.csv', n_estimators=10, criterion='gini', vectorize_method='BOW'):
        self.data_fname = data_fname
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.vectorize_method = vectorize_method

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname)
        if self.vectorize_method == 'BOW':
            train_features, test_features = vectorize.createBagOfWords(X_train, X_test)
        elif self.vectorize_method == 'TFIDF':
            train_features, test_features = vectorize.createTFIDF(X_train, X_test)
        dt = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion)
        dt.fit(train_features, y_train)
        print(dt.score(test_features, y_test))


if __name__ == '__main__':
    model = RandomForest(vectorize_method='BOW')
    model.run()
