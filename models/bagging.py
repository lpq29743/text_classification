from sklearn.ensemble import BaggingClassifier
from utils import data_utils, vectorize


class Bagging:
    def __init__(self, vectorizer, data_fname='../data/imdb.csv', base_estimator=None, n_estimators=10):
        self.data_fname = data_fname
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.vectorizer = vectorizer

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname)
        train_features, test_features = self.vectorizer.feature_extraction(X_train, X_test)
        bc = BaggingClassifier(base_estimator=self.base_estimator, n_estimators=self.n_estimators)
        bc.fit(train_features, y_train)
        print(bc.score(test_features, y_test))


if __name__ == '__main__':
    vectorizer = vectorize.Vectorizer('BOW', ngram_range=(1, 1))
    model = Bagging(vectorizer=vectorizer)
    model.run()
