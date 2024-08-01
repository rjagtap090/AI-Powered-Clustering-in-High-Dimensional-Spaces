# AI-Powered Clustering in High-Dimensional Spaces

## Overview

This project focuses on enhancing clustering accuracy in high-dimensional datasets by addressing the challenges posed by the curse of dimensionality. The project employs advanced dimensionality reduction techniques, feature selection methods, and specialized algorithms to improve the performance of clustering methods such as k-means. 
## Features

- **Dimensionality Reduction:** Applied techniques to reduce the number of random variables under consideration.
- **Feature Selection:** Identified the most significant features to improve clustering accuracy.
- **Specialized Algorithms:** Utilized algorithms optimized for high-dimensional data.
- **Cluster Analysis:** Performed detailed cluster analysis to validate results.

## Skills Demonstrated

- **Feature Selection:** Techniques to identify and retain the most relevant features in a dataset.
- **Cluster Analysis:** Methods to analyze and validate the clusters formed.
- **k-means Clustering:** Implementation of k-means clustering on high-dimensional data.
- **Dimensionality Reduction:** Techniques such as PCA (Principal Component Analysis) and t-SNE (t-distributed Stochastic Neighbor Embedding).

## Methodology

1. **Dimensionality Reduction:**
   - Applied PCA and t-SNE to reduce the dataset's dimensionality.
   - Evaluated the effectiveness of each technique in preserving data variance.

2. **Feature Selection:**
   - Used statistical methods to select the most impactful features.
   - Reduced computational complexity while maintaining data integrity.

3. **Clustering Algorithms:**
   - Implemented k-means and other clustering algorithms.
   - Optimized algorithm parameters for high-dimensional data.

4. **Cluster Analysis:**
   - Analyzed clusters for cohesion and separation.
   - Validated results using metrics such as silhouette score and Davies-Bouldin index.

Code(More in python file above)

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(reduced_data)
labels = kmeans.labels_

# Evaluation
silhouette_avg = silhouette_score(reduced_data, labels)
print(f'Silhouette Score: {silhouette_avg}')

# Plotting the clusters
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering on Reduced Data')
plt.show()
```

Results

- **Enhanced Clustering Accuracy:** Improved accuracy through effective dimensionality reduction and feature selection.
- **Scalable Solutions:** Developed scalable algorithms suitable for large, high-dimensional datasets.
- **Validated Methods:** Robust cluster validation metrics ensuring reliability of the results.

 Authors

- Rishabh Jagtap [Email](mailto:rjagtap.1999@gmail.com))
- Team Members:Collaborated with peers at the University of Delaware.

Acknowledgments

- Professors and Mentors at the University of Delaware
- Department of Business Analytics and Information Management
