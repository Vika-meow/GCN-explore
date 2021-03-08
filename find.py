import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import math
import scipy

def loadIds(fn):
        pair = dict()
        with open(fn, encoding='utf-8') as f:
                for line in f:
                        th = line[:-1].split('\t')
                        pair[int(th[0])]=th[1]
        return pair


def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing $num integers in each line."""
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret

def findCloseFromDifferentLang(num=10):
    vec = np.load("out_ae.npy")
    print("vectors loaded")
    test_pair = loadfile("data/ru_en/ref_ent_ids", 2)
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    print("sim counted")
    dic_1 = loadIds("data/ru_en/ent_ids_1")
    dic_2 = loadIds("data/ru_en/ent_ids_2")
    dic_1.update(dic_2)
    dic_ids = dic_1
    print("ids loaded")
    while(1):
        print("print entity id")
        a = int(input())
        print(dic_ids[a])
        rank = sim[a, :].argsort()
        rank = rank[:num]
        for el in rank:
            print(dic_ids[13500 + el] + "\t")

def findCloseFromAll(num=10):
    vec = np.load("out_ae.npy")
    print("vectors loaded")
    test_pair = loadfile("data/ru_en/ref_ent_ids", 2)
    Lvec = np.copy(vec)
    Rvec = np.copy(vec)
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    print("sim counted")
    dic_1 = loadIds("data/ru_en/ent_ids_1")
    dic_2 = loadIds("data/ru_en/ent_ids_2")
    dic_1.update(dic_2)
    dic_ids = dic_1
    print("ids loaded")
    while(1):
        print("print entity id")
        a = int(input())
        print(dic_ids[a])
        rank = sim[a, :].argsort()
        rank = rank[:num]
        for el in rank:
            print(dic_ids[el] + "\t")


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('graph', 'one', 'Graphs for find nearest entities') #one - in another lang graph; two - from two graphs

if __name__ == "__main__":
    if(FLAGS.graph == 'one'):
        findCloseFromDifferentLang()
    else:
        findCloseFromAll()