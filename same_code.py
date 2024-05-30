import os
os.environ["OMP_NUM_THREADS"] = "2"

# Force re-import to ensure the setting is taken into account
import sys
if 'sklearn' in sys.modules:
    del sys.modules['sklearn']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.io import arff

# Load custom data from ARFF file
# Replace 'your_dataset.arff' with the path to your actual ARFF file
data, meta = arff.loadarff('your_dataset.arff')

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Assuming your data has two columns 'x' and 'y' for the coordinates
X_aniso = df[['x', 'y']].values

# Apply KMeans clustering with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=170, n_init=10)
kmeans_labels = kmeans.fit_predict(X_aniso)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_aniso)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# KMeans plot
ax1.scatter(X_aniso[:, 0], X_aniso[:, 1], c=kmeans_labels, cmap='viridis')
ax1.set_title('KMeans Clustering (n=6)')

# DBSCAN plot
unique_labels = np.unique(dbscan_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for label, col in zip(unique_labels, colors):
    if label == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = (dbscan_labels == label)
    xy = X_aniso[class_member_mask]
    ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
ax2.set_title('DBSCAN Clustering')

plt.show()
