from sklearn.tree import DecisionTreeClassifier
from utils import data_utils, vectorize


class DecisionTree:
    def __init__(self, vectorizer, data_fname='../../data/imdb.csv', criterion='gini'):
        self.data_fname = data_fname
        self.criterion = criterion
        self.vectorizer = vectorizer

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname)
        train_features, test_features = vectorizer.feature_extraction(X_train, X_test)
        dt = DecisionTreeClassifier(criterion=self.criterion)
        dt.fit(train_features, y_train)
        print(dt.score(test_features, y_test))


if __name__ == '__main__':
    vectorizer = vectorize.Vectorizer('BOW', ngram_range=(1, 1))
    model = DecisionTree(vectorizer=vectorizer)
    model.run()
