import numpy as np
from scipy.linalg import sqrtm
from scipy.special import softmax
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import matplotlib.pyplot as plt
from matplotlib import animation

# Adjacency matrix
A = np.array(
    [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
)

# feature vector
feats = np.arange(A.shape[0]).reshape((-1,1))+1

# sum of feature vector by connection
# H[i] : i번째 노드와 연결된 다른 노드들의 feature vector의 합
H = A @ feats

# D : diagonal matrix
# 각각의 diagonal element는 각 노드의 연결된 이웃 노드 수의 합
D = np.zeros(A.shape)
np.fill_diagonal(D, A.sum(axis=0))
D_inv = np.linalg.inv(D)

H_avg = D_inv @ A @ feats







