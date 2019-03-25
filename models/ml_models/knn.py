from sklearn.neighbors import KNeighborsClassifier
from utils import data_utils, vectorize


class KNN:
    def __init__(self, vectorizer, data_fname='../../data/imdb.csv', n_neighbors=10):
        self.data_fname = data_fname
        self.n_neighbors = n_neighbors
        self.vectorizer = vectorizer

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname)
        train_features, test_features = self.vectorizer.feature_extraction(X_train, X_test)
        neigh = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        neigh.fit(train_features, y_train)
        print(neigh.score(test_features, y_test))


if __name__ == '__main__':
    vectorizer = vectorize.Vectorizer('BOW', ngram_range=(1, 1))
    model = KNN(vectorizer=vectorizer)
    model.run()
