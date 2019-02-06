# Text Classification

The goal of this repository is to implement text classification in traditional machine learning methods and deep learning methods (in Pytorch).

## Models

### Feature Extraction

- [x] BOW (Bag of Words)
- [x] TFIDF (Term Frequency-Inverse Document Frequency)
- [x] N-gram

### Traditional Machine Learning Methods

- [x] KNN (K-Nearest Neighbor)
- [x] Decision Tree
- [x] Perceptron
- [x] Random Forest
- [x] Naive Bayes
- [x] SVM (Support Vector Machine)
- [x] Logistic Regression

### Deep Learning Methods

- [ ] FastText
- [ ] TextCNN
- [ ] TextRNN
- [ ] RCNN
- [ ] Transformer
- [ ] Elmo
- [ ] BERT

## Experiments

### Effects of Feature Extraction Methods

| Methods | KNN | Decision Tree | Perceptron | Random Forest | Naive Bayes | SVM | Logistic Regression |
| :---: | :---: | :---: | :---:| :---: | :---: | :---: | :---: |
| BoW (ngram-range=(1, 1))| 0.665 | 0.707 | 0.742 | 0.757 | 0.800 | 0.823 | 0.826 |
| BoW (ngram-range=(1, 2))| 0.666 | 0.699 | 0.750 | 0.751 | 0.795 | 0.819 | 0.823 |
| BoW (ngram-range=(1, 3))| 0.667 | 0.700 | 0.712 | 0.757 | 0.795 | 0.818 | 0.824 |
| BoW (ngram-range=(2, 2))| 0.579 | 0.628 | 0.652 | 0.652 | 0.669 | 0.671 | 0.692 |
| BoW (ngram-range=(2, 3))| 0.578 | 0.625 | 0.625 | 0.648 | 0.662 | 0.667 | 0.684 |
| BoW (ngram-range=(3, 3))| 0.536 | 0.572 | 0.525 | 0.581 | 0.561 | 0.576 | 0.590 |
| TFIDF | 0.714 | 0.705 | 0.594 | 0.760 | 0.807 | 0.804 | 0.824

## To-do List

- Class `Vectorizer` lazy initialization
- Cross Validation
- Grid Search
