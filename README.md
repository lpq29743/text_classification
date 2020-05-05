# Text Classification

The goal of this repository is to implement text classification in traditional machine learning methods and deep learning methods (in Pytorch).

Note: Since I have to spend more time in conducting research, I stop reproducing classic papers on the task of text classification. However, I will still maintain this project. Any contributions lke issues and pull requests are welcome.

## Models

### Feature Extraction

- [x] BOW (Bag of Words)
- [x] TFIDF (Term Frequency-Inverse Document Frequency)
- [x] N-gram

### Traditional Machine Learning Methods

- [x] KNN (K-Nearest Neighbor)
- [x] Decision Tree
- [x] Perceptron
- [x] Bagging
- [x] Random Forest
- [x] AdaBoost
- [x] Gradient Boosting
- [x] Naive Bayes
- [x] SVM (Support Vector Machine)
- [x] Logistic Regression

### Deep Learning Methods

- [x] DAN: [Deep Unordered Composition Rivals Syntactic Methods for Text Classification](http://www.aclweb.org/anthology/P15-1162)
- [x] TextCNN: [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181)
- [x] TextRNN: [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
- [ ] <del>RCNN</del>
- [ ] <del>Capsule Network</del>
- [ ] <del>Transformer</del>
- [ ] <del>Elmo</del>
- [ ] <del>BERT</del>

## Experiments

### Effects of Feature Extraction Methods

| Methods | KNN | Decision Tree | Perceptron | Bagging | Random Forest | AdaBoost | Gradient Boosting | Naive Bayes | SVM | Logistic Regression |
| :---: | :---: | :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BoW (ngram-range=(1, 1))| 0.665 | 0.707 | 0.742 | 0.749 | 0.757 | 0.724 | 0.713 | 0.800 | 0.823 | 0.826 |
| BoW (ngram-range=(1, 2))| 0.666 | 0.699 | 0.750 | 0.744 | 0.751 | 0.724 | 0.712 | 0.795 | 0.819 | 0.823 |
| BoW (ngram-range=(1, 3))| 0.667 | 0.700 | 0.712 | 0.748 | 0.757 | 0.724 | 0.712 | 0.795 | 0.818 | 0.824 |
| BoW (ngram-range=(2, 2))| 0.579 | 0.628 | 0.652 | 0.646 | 0.652 | 0.584 | 0.600 | 0.669 | 0.671 | 0.692 |
| BoW (ngram-range=(2, 3))| 0.578 | 0.625 | 0.625 | 0.636 | 0.648 | 0.584 | 0.600 | 0.662 | 0.667 | 0.684 |
| BoW (ngram-range=(3, 3))| 0.536 | 0.572 | 0.525 | 0.578 | 0.581 | 0.532 | 0.539 | 0.561 | 0.576 | 0.590 |
| TFIDF | 0.714 | 0.705 | 0.594 | 0.759 | 0.760 | 0.723 | 0.714 | 0.807 | 0.804 | 0.824

### Effects of Word2Vec Methods 
| Methods | DAN | TextCNN | TextRNN |  
| :---: | :---: | :---: | :---: |
| Trained Using Corpus | 0.880 | 0.893 | 0.903 |
| Glove | 0.886 | 0.900 | 0.890 |

## To-do List

- Word2Vec Methods (Pre-trained and Trained Using Corpus) Comparision
- Class `Vectorizer` lazy initialization
- Cross Validation
- Grid Search
- Visualize text feature
- Shared Classifier Module for Deep Learning
