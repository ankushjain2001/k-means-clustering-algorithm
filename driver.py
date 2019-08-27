# =============================================================================
# === K-Means Clustering Algorithm - Driver ===================================
# =============================================================================

import kmeans
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# Importing the datasets
iris = datasets.load_iris()

# Extracting the data and ground truth labels 
data = iris.data
y_truth = iris.target

# Setting the parameters for the algorithm
k = 3
maxIters = 100
initMethod = 'kmeans++'

# Executing the MANUAL ALGORITHM
y_pred = kmeans.KMeans(data, k, maxIters, initMethod)

# Executing the SKLEARN KMEANS ALGORITHM
kmeans_sklearn = KMeans(n_clusters=3, init='k-means++').fit(data)
y_sklearn_pred = kmeans_sklearn.labels_

# Plotting the output and ground truth
kmeans.drawPlot(data, y_truth, 'Truth')
kmeans.drawPlot(data, y_pred, 'Manual Output')
kmeans.drawPlot(data, y_sklearn_pred, 'SKLearn Output')

# Confusion Matrix
print('Confusion Matrix of True Labels v/s Predicted Labels:')
print(confusion_matrix(y_truth, y_pred))

print('Confusion Matrix of True Labels v/s SK Learn Predicted Labels:')
print(confusion_matrix(y_truth, y_sklearn_pred))

print('Confusion Matrix of SK Learn Predicted Labels v/s Predicted Labels:')
print(confusion_matrix(y_sklearn_pred, y_pred))

# -----------------------------------------------------------------------------

# NOTE: Since KMeans is a heuristics based unsupervized learning algorithm, it 
# cannot be guaranted that the Y_MANUAL_PRED and Y_SKLEARN_PRED will have same
# label names (numbers 0, 1, 2) as Y_TRUTH.
#
# The clusters will be similar but labels can change. The number and name 
# mapping for Y_TRUTH is as follows
# 0 : 'setosa' : VIOLET in plot
# 1 : 'versicolor' : BLUE in plot
# 2 : 'virginica' : YELLOW in plot
#
# Use the following code snippet to change the label names/numbers manually.

'''
for i in range(len(y_pred)):
    if y_pred[i]==0:
        y_pred[i] = 2
    elif y_pred[i]==1:
        y_pred[i] = 0
    elif y_pred[i]==2:
        y_pred[i] = 1
'''

# -----------------------------------------------------------------------------

