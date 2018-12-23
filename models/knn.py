from sklearn.neighbors import KNeighborsClassifier
from utils import data_utils, vectorize


class KNN:
    def __init__(self, data_fname='../data/imdb.csv', n_neighbors=10, vectorize_method='BOW'):
        self.data_fname = data_fname
        self.n_neighbors = n_neighbors
        self.vectorize_method = vectorize_method

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname)
        if self.vectorize_method == 'BOW':
            train_features, test_features = vectorize.createBagOfWords(X_train, X_test)
        elif self.vectorize_method == 'TFIDF':
            train_features, test_features = vectorize.createTFIDF(X_train, X_test)
        neigh = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        neigh.fit(train_features, y_train)
        print(neigh.score(test_features, y_test))


if __name__ == '__main__':
    model = KNN(vectorize_method='TFIDF')
    model.run()
