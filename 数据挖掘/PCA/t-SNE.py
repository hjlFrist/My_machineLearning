#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
from pandas import DataFrame
import pandas as pd
import  numpy as np

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
print(iris)

def T_sne(data):
    print(type(data))
    X_tsne = TSNE(learning_rate=100).fit_transform(data)
    X_pca = PCA().fit_transform(data)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.show()

if __name__ == '__main__':
    print('start......')
    df = np.loadtxt('C:/Users/Administrator/Desktop/tar.csv', dtype=np.str, delimiter=",")
    #df_new = df[1:,1:].astype(np.float)
    T_sne(df)
