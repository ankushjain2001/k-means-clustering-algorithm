# =============================================================================
# === K-Means Clustering Algorithm ============================================
# =============================================================================

import numpy as np

#d = data = np.array([[1, 2, 3], [1, 4,3], [1, 0,3], [10, 2,3], [10, 4,3], [10, 0,3]])
d = data = np.array([[1, 2, 3,7,4], [1, 4,3,3,9], [1, 0,3,0,2], [10, 2,3,1,6], [10, 4,3,4,4], [10, 0,3,3,3]])
initMethod = 'kmeans++'
maxIters = 100
k = 2
docs = data.shape[0]
features = data.shape[1]

#def euclideanDistance(c, d):
#    """
#    c : Centroid coordinates
#    d : Dataset
#    """
#    eucDistances = []
#    for i in c: # a centroid tuple is selected
#        tupleSum = 0
#        for j in range(len(i)): # elements of row selected
#            tupleSum += (i[j] - d[j])**2
#        eucDistances.append(tupleSum**0.5)
#    return np.array(eucDistances)

#def squareDistance(c, d):
#    """
#    c : Centroid coordinates
#    d : Dataset
#    """
#    sqDistances = []
#    for i in d: # row is selected
#        tupleSum = 0
#        for j in range(len(i)): # elements of row selected
#            tupleSum += (c[j] - i[j])**2
#        sqDistances.append(tupleSum)
#    return np.array(sqDistances)

def getNextSeedIndex(dc, w):
    idx = np.searchsorted(dc, w, side="left")
    return dc[idx]

        
def getInitCentroids(method, k, d, docs, features):
    if method == 'random':
        # Columns of seed matrix initially represent the centroid coordinates
        seeds = np.empty(shape = [0, k])
        for i in range(features):
            values = []
            # Restricts the random values to the range of each feature.
            minVal = min(d[:, i])
            maxVal = max(d[:, i])
            rangeVal = maxVal - minVal
            # Handles features with zero range i.e. constant values.
            # PERFORMANCE NOTE: A function could have been defined to execute
            # the following expression only once as a function. 
            # However, since function calls in python are costlier than 
            # conditional statements (due to execution frame change) its better
            # to have K iterations of conditional statements than function calls.
            for j in range(k):
                if rangeVal != 0:
                    # NOTE: Do we need this zero?
                    values.append(np.random.randint(minVal, maxVal) + np.random
                                  .random_sample())
                else:
                    values.append(float(minVal))
            #seeds.append(np.array(values))
            seeds = np.vstack((seeds, values))
        # Transposing the matrix so that each row represent centroid coordinates
        seeds = seeds.transpose()
    
    elif method == 'kmeans++':
        seeds = np.empty(shape = [0, features])
        # Generating the first random seed out of the given data points.
        firstSeedIndex = np.random.randint(0, docs)
        firstSeed = d[firstSeedIndex]
        seeds = np.vstack((seeds, firstSeed))
        seedsIndexList = [firstSeedIndex]
        # Calculating square distance of all points from the first centroid
        # (firstSeed). Variable "distSeeds"  will also be used as a global list
        # of distances for the subsequent centroids.
        
#        # Poor implementation with loops: Takes 20-30 seconds for 10^6 vectors of 5 features.
#        distSeeds = squareDistance(firstSeed, d)
        
        # NumPy functions, take about 8-9 seconds.
        distSeeds = ((d-firstSeed)**2).sum(axis=1)
        
        # Calculating the cumulative distance/sum for all the points, which 
        # will act as weights for the probability distribution
        distCumulative = np.cumsum(distSeeds)
        # Calculating the subsequent 'K' centroids
        for _ in range(k - 1):
            # Extracting the greatest cumulative sum
            greatestCumVal = int(distCumulative[-1])
            # Determining the random weight from the greatest cum sum.
            nextSeedRandWeight = np.random.randint(0, greatestCumVal)
            # Determining the next seed index on the basis of probability of 
            # nextSeedRandWeight in the probability distribution.
            nextSeedIndex = np.searchsorted(distCumulative, nextSeedRandWeight, side="left")
            nextSeed = d[nextSeedIndex]
            #  Repeat the above 3 statements if the seed already exists.
            while (nextSeedIndex in seedsIndexList):
                nextSeedRandWeight = np.random.randint(0, greatestCumVal)
                nextSeedIndex = np.searchsorted(distCumulative, nextSeedRandWeight, side="left")
                nextSeed = d[nextSeedIndex]
                print(seedsIndexList)
                print(nextSeedIndex)
            seedsIndexList.append(nextSeedIndex)
            # Appending new centroids to 'seeds'
            seeds = np.vstack((seeds, nextSeed))
            # Calculating squared distance of all points from the next centroid 
            # (nextSeed).
            
            #distNextSeed = squareDistance(nextSeed, d)
            distNextSeed = ((d-nextSeed)**2).sum(axis=1)
            
            # Appending distances to the global distSeeds variable. Following 
            # is also a row major matrix.
            distSeeds = np.vstack((distSeeds, distNextSeed))
            # Determining the smallest square distance for each point wrt each 
            # centroid
            distNewSeeds = []
            for i in range(docs):
                distNewSeeds.append(min(distSeeds[:,i]))
                distCumulative = np.cumsum(distNewSeeds)
        
    return seeds


def shouldStop(oldCentroids, centroids, iterations, maxIters):
    if iterations > maxIters:
        return True
    return np.array_equal(oldCentroids, centroids)


def getLabels(data, centroids):
    labels = []
    for i in data:
#        # Poor implementation with loops: Takes 20-30 seconds for 10^6 vectors of 5 features.
#        t=time.time()
#        for _ in range(1000000):
#            distances = euclideanDistance(centroids, i)
#        print(time.time()-t)
        
        # NumPy functions, take about 8-9 seconds.
        distances = np.sqrt(((centroids-i)**2).sum(axis=1))
        label = np.where(distances == min(distances))
        labels.append(label[0][0])
    return np.array(labels)

def getNewCentroids(data, labels, k, features):
    newCentroids= np.empty(shape = [0, features])
    for i in range(k):
        indices = np.where(labels == i)
        centroid = np.mean(data[indices[0], :], axis=0)
        newCentroids = np.vstack((newCentroids, centroid))
    return newCentroids
        
        
    
#def KMeans(data, k, maxIters, initMethod):
    """
    
    data : Can only be a 2-dimension np.array where each column of the array
    represents an axis of the n-dimension space.
    
    k : Number of desired clusters.
    
    maxIters : Maximum number of iterations to stop the centroid calculation if
    the final centroids are not determined in the given iterations.
    
    initMethod : Method to determine the initial centroid. Can be - "kmeans++"
    or "random".
    
    """

# Number of docs or data points
numDocs = data.shape[0]
# Number of features
numFeatures = data.shape[1]
# Generating initial centroids
centroids = getInitCentroids(initMethod, k, data, numDocs, numFeatures)
# Variables for counting iterations and holding old centroids
iterations = 0
oldCentroids = np.empty(shape = [0, numFeatures])

# Executing the k-means algorithm
while not shouldStop(oldCentroids, centroids, iterations, maxIters):
    iterations += 1
    oldCentroids = centroids
    # Assign labels to each datapoint based on centroids
    labels = getLabels(data, centroids)
    while set(labels) != set(range(k)) and iterations == 1:
        centroids = getInitCentroids(initMethod, k, data, numDocs, numFeatures)
        oldCentroids = centroids
        labels = getLabels(data, centroids)
    # Assign centroids based on datapoint labels
    centroids = getNewCentroids(data, labels, k, numFeatures)
    # Implement the function and improve the euclidean distance calculation using stanford

print(labels)



'''
# =============================================================================
# === Scratch Pad =============================================================
# =============================================================================
# Needs testing and maybe variable change in function names
# Implement classes











# ROUGH
x = np.array([1, 2, 3, 4, 5])
y = np.array([8, 8, 8, 8, 8])
z = np.ones((5, 9))

np.sqrt(sum((x - y) ** 2))

np.sqrt(((z-x)**2).sum(axis=0))










from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

dat = dataset.data
dat = ['Story of a cat cat cat cat and a dog',
       'Story of a dog',
       'Story of dog and mouse',
       'Life of a cat and a dog',
       'Life of a mouse']

vectorizer = TfidfVectorizer(max_features=10000,
                                 stop_words='english',
                                 use_idf=True)

X = vectorizer.fit_transform(dat)
vectorizer.get_feature_names()
vectorizer.idf_

table = X.todense()

table= X.data
'''
