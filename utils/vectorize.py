from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def createBagOfWords(X_train, X_test, max_features=300):
    vectorizer = CountVectorizer(analyzer='word', input='content', stop_words='english', max_features=max_features)
    bag_of_words_train = vectorizer.fit_transform(X_train).toarray()
    bag_of_words_test = vectorizer.transform(X_test).toarray()
    return bag_of_words_train, bag_of_words_test


def createTFIDF(X_train, X_test, max_features=300):
    vectorizer = TfidfVectorizer(analyzer='word', input='content', stop_words='english', max_features=max_features)
    tfidf_train = vectorizer.fit_transform(X_train).toarray()
    tfidf_test = vectorizer.transform(X_test).toarray()
    return tfidf_train, tfidf_test
