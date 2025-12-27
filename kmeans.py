
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

X,y_true = make_blobs(n_samples = 500,centers = 6,cluster_std= 1.0, random_state= 43)

def initialize_centroid(X,k):
  np.random.seed(42)
  random_incidices = np.random.permutation(X.shape[0])[:k]
  centroids = X[random_incidices]
  return centroids
  
def assign_clusters(X,centroids):
  clusters = []
  for x in X:
    distances = np.linalg.norm(x - centroids,axis = 1)
    cluster = np.argmin(distances)
    clusters.append(clusters)
  return np.array(clusters)
  
def update_centroids(X,clusters,k):
  centroids = np.zeros((k, X.shape[1]))
  for i in range(k):
    points = X[clusters == i]
    if len(points) > 0:
      centroids[i] = points.mean(axis = 0)
  return centroids    
  
def kmeans(X,k,max_iters = 50, tol = 1e-4):
      centroids = initialize_centroid(X,k)
      for m in range(max_iters):
        clusters = assign_clusters(X,centroids)
        new_centroids = update_centroids(X,clusters,k)
        if np.all(np.abs(new_centroids - centroids) < tol):
          break
        centroids = new_centroids
      return clusters,centroids    
  
 
clusters, centroids = kmeans(X,k=3)
  