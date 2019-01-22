from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Vectorizer:
    def __init__(self, method='BOW', ngram_range=(1, 1), max_features=300):
        if method == 'BOW':
            self.vectorizer = CountVectorizer(analyzer='word', input='content', stop_words='english', ngram_range=ngram_range, max_features=max_features)
        elif method == 'TFIDF':
            self.vectorizer = TfidfVectorizer(analyzer='word', input='content', stop_words='english', max_features=max_features)

    def feature_extraction(self, X_train, X_test):
        bag_of_words_train = self.vectorizer.fit_transform(X_train).toarray()
        bag_of_words_test = self.vectorizer.transform(X_test).toarray()
        return bag_of_words_train, bag_of_words_test
