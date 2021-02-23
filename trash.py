import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def loadIds(fn):
    pair = dict()
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            pair[int(th[0])] = th[1]
    return pair

def twoDims():
    vec = np.load("out_ae.npy")
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(vec)
    np.save("out_ae_2", principalComponents)

    fig = plt.figure()  # Создание объекта Figure
    for el in vec:
        plt.scatter(el[0], el[1])  # scatter - метод для нанесения маркера в точке (1.0, 1.0

    # После нанесения графического элемента в виде маркера
    # список текущих областей состоит из одной области
    print(fig.axes)
    # смотри преамбулу
    save(name='pic_1_4_1', fmt='pdf')
    save(name='pic_1_4_1', fmt='png')

    plt.show()

def tsna():
    vectors = np.load("out_ae.npy")
    dic_1 = loadIds("data/ru_en/ent_ids_1")
    dic_2 = loadIds("data/ru_en/ent_ids_2")
    dic_1.update(dic_2)
    dic_ids = dic_1
    words = []
    embeddings = []
    i = 0
    for vec in vectors:
        embeddings.append(vec)
        words.append(dic_ids[i])
        i+=1

    tsne_ak_2d = TSNE(n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings)

def main():
    tsna()

if __name__ == "__main__":
    main()