import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.test.utils import datapath
from gensim import utils


class MyCorpus(object):
    def __iter__(self):
        for line in open('article.txt', encoding='utf-8'):
            yield utils.simple_preprocess(line)


# Perhatikan bahwa setting yang digunakan adalah setting default, sepertiL min_count atau jumlah kemunculan kata = 5. Pada eksperimen yang berbeda, sesuaikan setting.

import gensim.models

sentences = MyCorpus()
model_1 = gensim.models.Word2Vec(sentences, min_count=1, window=2, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007,
                                 negative=20)

sentences = MyCorpus()
model_2 = gensim.models.Word2Vec(sentences, min_count=5, window=2, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007,
                                 negative=20)

word_1 = 'pemerintah'
word_2 = 'kami'
word_3 = 'negara'
word_4 = 'stimulus'
word_5 = 'virus'

vec_positif = model_1.wv[word_1]
print(vec_positif)

vec_positif = model_2.wv[word_2]
print(vec_positif)

# Periksa 5 kata yang similarity-nya paling tinggi dengan sebuah kata.
# dan 1 kata yang similarity-nya paling tinggi dengan sebuah kata. 
# 
# Perhitungan similarity apa yang digunakan sebagai default di Gensim?

# Akses representasi vektor/embedding jumlah minimal kemunculan kata = 5 buah kata
print(model_1.wv.most_similar(positive=[word_2], topn=5))

# Akses representasi vektor/embedding jumlah minimal kemunculan kata = 5 buah kata
print(model_1.wv.most_similar(positive=[word_1], topn=5))

# Point 2. Tes nilai similarity antar 2 kata.

# tes similiarity antar 2 kata dengan mi_count = 1
print(model_1.wv.similarity(word_1, word_2))

# tes similiarity antar 2 kata dengan mi_count = 1
print(model_1.wv.similarity(word_4, word_5))

# tes similiarity antar 2 kata dengan mi_count = 1
print(model_1.wv.similarity(word_3, word_1))

# tes similiarity antar 2 kata dengan mi_count = 5
print(model_2.wv.similarity(word_3, word_1))

# tes similiarity antar 2 kata dengan mi_count = 5
print(model_2.wv.similarity(word_4, word_5))

# tes similiarity antar 2 kata dengan mi_count = 5
print(model_2.wv.similarity(word_3, word_1))

# Persiapan visualisasi embeddings, import library yang diperlukan

from sklearn.decomposition import IncrementalPCA  # inital reduction
from sklearn.manifold import TSNE  # final reduction
import numpy as np  # array handling

# Fungsi untuk reduksi dimensi, supaya lebih mudah dimengerti.
# Pada contoh, dimensi vektor direduksi menjadi 2.

def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = []  # positions in vector space
    labels = []  # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


# Lakukan proses reduksi dimensi

x_vals, y_vals, labels = reduce_dimensions(model_1)


def reduce_dimensions_mod1(model1):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = []  # positions in vector space
    labels1 = []  # keep track of words to label our data again later
    for word in model1.wv.vocab:
        vectors.append(model1.wv[word])
        labels1.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels1 = np.asarray(labels1)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_val = [v[0] for v in vectors]
    y_val = [v[1] for v in vectors]
    return x_val, y_val, labels1


x_val, y_val, labels1 = reduce_dimensions_mod1(model_2)


# Coba tampilkan dengan plotly. Image akan disimpan sebagai file .html.
# Perhatikan path penyimpanan file, sesuaikan dengan komputer/drive masing-masing.


def plot_with_plotly(x_vals, y_vals, labels):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    plot(data, filename='word-embedding-plot.html')


plot_with_plotly(x_vals, y_vals, labels)


def plot_with_plotly_lab1(x_val, y_val, labels1):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_val, y=y_val, mode='text', text=labels1)
    data = [trace]

    iplot(data, filename='word-embedding-plot-model1.html')


plot_with_plotly_lab1(x_vals, y_vals, labels1)


# Fungsi untuk visualisasi dengan matplotlib

def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))


# Tes plot dengan matplotlib


plot_with_matplotlib(x_vals, y_vals, labels)


def plot_with_matplotlib_lab1(x_val, y_val, labels1):
    import matplotlib.pyplot as plot
    import random

    random.seed(0)
    plot.figure(figsize=(12, 12))
    plot.scatter(x_val, y_val)

    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels1)))
    selected_indices = random.sample(indices, 25)
    for j in selected_indices:
        plot.annotate(labels1[j], (x_val[j], y_val[j]))


plot_with_matplotlib_lab1(x_val, y_val, labels1)
