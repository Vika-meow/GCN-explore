import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def loadIds(fn):
    pair = dict()
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            pair[int(th[0])] = th[1]
    return pair

def tsne_plot_2d(label, embeddings, words=[], a=1):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig(label+".png", format='png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize():
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
    tsne_plot_2d('smth', embeddings_ak_2d, a=0.1)

def main():
    visualize()

if __name__ == "__main__":
    main()