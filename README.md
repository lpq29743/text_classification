# Text Classification

The goal of this repository is to implement text classification in traditional machine learning methods and deep learning methods (in Pytorch).

## Models

### Feature Extraction

- [x] BOW (Bag of Words)
- [x] TFIDF (Term Frequency-Inverse Document Frequency)
- [x] N-gram
- [ ] TextRank
- [ ] LDA (Latent Dirichlet Allocation)

### Traditional Machine Learning Methods

- [x] KNN (K-Nearest Neighbor)
- [x] Decision Tree
- [x] Random Forest
- [ ] SVM (Support Vector Machine)

### Deep Learning Methods

- [ ] fastText
- [ ] TextCNN
- [ ] TextRNN
- [ ] RCNN
- [ ] Transformer
- [ ] Elmo
- [ ] Bert

## Experiments

### Effects of Feature Extraction Methods

| Methods | KNN | Decision Tree | Random Forest |
| :---: | :---: | :---: | :---:|
| BoW (ngram-range=(1, 1))| 0.665 | 0.707 | 0.757 |
| BoW (ngram-range=(1, 2))| 0.666 | 0.699 | 0.751 |
| BoW (ngram-range=(1, 3))| 0.667 | 0.700 | 0.757 |
| BoW (ngram-range=(2, 2))| 0.579 | 0.628 | 0.652 |
| BoW (ngram-range=(2, 3))| 0.578 | 0.625 | 0.648 |
| BoW (ngram-range=(3, 3))| 0.536 | 0.572 | 0.581 |
| TFIDF | 0.714 | 0.705 | 0.760 |

## To-do List

- Class `Vectorizer` lazy initialization
- Cross Validation
- Grid Search
