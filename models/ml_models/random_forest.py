from sklearn.ensemble import RandomForestClassifier
from utils import data_utils, vectorize


class RandomForest:
    def __init__(self, vectorizer, data_fname='../../data/imdb.csv', n_estimators=10, criterion='gini'):
        self.data_fname = data_fname
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.vectorizer = vectorizer

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname)
        train_features, test_features = self.vectorizer.feature_extraction(X_train, X_test)
        rf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion)
        rf.fit(train_features, y_train)
        print(rf.score(test_features, y_test))


if __name__ == '__main__':
    vectorizer = vectorize.Vectorizer('BOW', ngram_range=(1, 1))
    model = RandomForest(vectorizer=vectorizer)
    model.run()
