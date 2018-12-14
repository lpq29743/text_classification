from sklearn.neighbors import KNeighborsClassifier
from utils import data_utils, vectorize


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_utils.load_train_test_data('../data/imdb.csv')
    bag_of_words_train, bag_of_words_test = vectorize.createBagOfWords(X_train, X_test)
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(bag_of_words_train, y_train)
    print(neigh.score(bag_of_words_test, y_test))
