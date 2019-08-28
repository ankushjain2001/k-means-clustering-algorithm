# K-Means Clustering Algorithm
This is an implementation of K-Means clustering algorithm along with the K-Means++ seeding technique using NumPy. Further, a comparison of the manual implementation is made against the SK Learn implementation of the algorithm. The Iris dataset is used for this comparison.

## Clustering Graphs
### Iris Dataset True Clusters

<p align="center"><img src="readme/01_truth.png" title="Iris Truth" width=70%></p>

### SK Learn K-Means Clusters

<p align="center"><img src="readme/03_sklearn_output.png" title="SK Learn Output" width=70%></p>

### Manual Implementation Clusters

<p align="center"><img src="readme/02_manual_output.png" title="Manual Output" width=70%></p>

### Confusion Matrix - Iris True Clusters v/s Manual Implementation Clusters

||0|1|2|
|:---:|:---:|:---:|:---:|
|0|50|0|0|
|1|0|48|2|
|2|0|14|36|

### Confusion Matrix - SK Learn Clusters v/s Manual Implementation Clusters

||0|1|2|
|:---:|:---:|:---:|:---:|
|0|50|0|0|
|1|0|62|0|
|2|0|0|38|
