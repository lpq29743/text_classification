from sklearn.svm import SVC
from utils import data_utils, vectorize


class SVM:
    def __init__(self, vectorizer, data_fname='../data/imdb.csv'):
        self.data_fname = data_fname
        self.vectorizer = vectorizer

    def run(self):
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.data_fname)
        train_features, test_features = self.vectorizer.feature_extraction(X_train, X_test)
        svc = SVC()
        svc.fit(train_features, y_train)
        print(svc.score(test_features, y_test))


if __name__ == '__main__':
    vectorizer = vectorize.Vectorizer('TFIDF', ngram_range=(3, 3))
    model = SVM(vectorizer=vectorizer)
    model.run()
