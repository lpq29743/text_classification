from sklearn.tree import DecisionTreeClassifier
from utils import data_utils, vectorize


class DecisionTree:
    def __init__(self, data_fname='../data/imdb.csv', criterion='gini', vectorize_method='BOW'):
        self.data_fname = data_fname
        self.criterion = criterion
        self.vectorize_method = vectorize_method

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname)
        if self.vectorize_method == 'BOW':
            train_features, test_features = vectorize.createBagOfWords(X_train, X_test)
        elif self.vectorize_method == 'TFIDF':
            train_features, test_features = vectorize.createTFIDF(X_train, X_test)
        dt = DecisionTreeClassifier(criterion=self.criterion)
        dt.fit(train_features, y_train)
        print(dt.score(test_features, y_test))


if __name__ == '__main__':
    model = DecisionTree(vectorize_method='TFIDF')
    model.run()
