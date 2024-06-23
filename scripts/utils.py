import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

# 最適輸送に使うライブラリをインストールします。
import subprocess
subprocess.check_call(["pip", "install", "pot"])
import ot


### functions
def show_RDM(matrix, labels=[], title=None):
  plt.figure(figsize=(12, 10))
  sns.heatmap(matrix, annot=False, cmap='coolwarm', cbar=True, xticklabels=labels, yticklabels=labels)
  if title is not None:
    plt.title(title)
  plt.xticks(rotation=90) # x軸のラベルを90度回転して見やすくする
  # colorbarのラベルを設定する
  cbar = plt.gca().collections[0].colorbar
  #cbar.set_label('Similarity', rotation=270, labelpad=20, fontsize=20)
  plt.show()

def MDS(matrix, n_dimensions=2):
  mds = sklearn.manifold.MDS(n_components=n_dimensions, random_state=30)
  embedding = mds.fit_transform(matrix)
  return embedding

def RSA(matrix1, matrix2):
  upper_tri_1 = matrix1[np.triu_indices(matrix1.shape[0], k=1)]
  upper_tri_2 = matrix2[np.triu_indices(matrix2.shape[0], k=1)]
  corr, _ = spearmanr(upper_tri_1, upper_tri_2)
  return corr

def plot_2d_embedding(points, colors=None, title="2D Embedding Plot"):
  points = np.array(points)
    
  if points.shape[1] != 2:
      raise ValueError("Input points should have shape (n_samples, 2).")
  
  if colors is None:
        colors = np.linspace(0, 1, len(points))  # Default color based on an even gradient
  
  plt.figure(figsize=(8, 6))
  plt.scatter(points[:, 0], points[:, 1], c=colors, cmap='viridis', s=50, alpha=0.7)
  
  plt.title(title)
  plt.xlabel('Component 1')
  plt.ylabel('Component 2')
  plt.grid(True)
  plt.show()

def plot_3d_embedding(points, colors=None, title="3D Embedding Plot"):
  points = np.array(points)
    
  if points.shape[1] != 3:
      raise ValueError("Input points should have shape (n_samples, 3).")
  
  if colors is None:
        colors = np.linspace(0, 1, len(points))  # Default color based on an even gradient
      
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap='viridis', s=50, alpha=0.7)
  
  ax.set_title(title)
  ax.set_xlabel('Component 1')
  ax.set_ylabel('Component 2')
  ax.set_zlabel('Component 3')
  plt.show()

def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,T=None,
                                max_iter=500, tol=1e-9, verbose=False):
    C1, C2, p, q = ot.utils.list_to_array(C1, C2, p, q)
    nx = ot.backend.get_backend(C1, C2, p, q)
    # add T as an input
    if T is None:
      T = nx.outer(p, q)
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)
    cpt = 0
    err = 1
    while (err > tol and cpt < max_iter):
        Tprev = T
        # compute the gradient
        tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
        T = ot.bregman.sinkhorn(p, q, tens, epsilon, method='sinkhorn')
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt += 1
    
    GWD = ot.gromov.gwloss(constC, hC1, hC2, T)
    return T, GWD    
        

def matching_rate(OT_plan):
  # add doc string
  """
  Calculate the matching rate of the OT plan assuming that the correct matching is the diagonal of the OT plan.
  Args:
    OT_plan (np.ndarray): Optimal transport plan.
  Returns:
    float: Matching rate.
  """
  
  corresponding_indices = np.argmax(OT_plan, axis=1)
  match_count = 0
  for i in range(OT_plan.shape[0]):
    if corresponding_indices[i] == i:
      match_count += 1
  matching_rate = match_count / OT_plan.shape[0]
  return matching_rate

def norm_T(n_iters,n,m,mu=1,v=1,seed=None):
  if seed != None:
    np.random.seed(seed)
  T = np.abs(np.random.normal(mu,v,size=(n,m,n_iters)))
  return (T/np.sum(T,axis=(0,1))).T
