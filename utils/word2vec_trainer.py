import pandas as pd
from gensim.models.word2vec import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train(file_path, save_file_path, method='Word2Vec', embed_size=300):
    df = pd.read_csv(file_path, sep='\t')
    raw_sentences = df['sentences'].values
    sentences = []
    for s in raw_sentences:
        sentences.append(s.split())
    w2vModel = Word2Vec(sentences=sentences, hs=0, negative=5, min_count=5, window=5, workers=4, size=embed_size)
    w2vModel.wv.save_word2vec_format(save_file_path, binary=False)


if __name__ == '__main__':
    train('../data/imdb.csv', '../data/imdb.vec')
