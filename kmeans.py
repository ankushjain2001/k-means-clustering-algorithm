# =============================================================================
# === K-Means Clustering Algorithm ============================================
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Setting the PyPlot style
plt.style.use('ggplot')


# Function to set the init centroids as per the chosen seeding method
def getInitCentroids(method, k, d, docs, features):
    # Random seeds initiation function
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
            # to have K iterations of conditional statements than function 
            # calls.
            for j in range(k):
                if rangeVal != 0:
                    # NOTE: Do we need this zero?
                    values.append(np.random.randint(minVal, maxVal) + np.random
                                  .random_sample())
                else:
                    values.append(float(minVal))
            #seeds.append(np.array(values))
            seeds = np.vstack((seeds, values))
        # Transposing the matrix so that each row represent centroid 
        # coordinates.
        seeds = seeds.transpose()
    # KMeans++ seeds initiation function
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
            nextSeedIndex = np.searchsorted(distCumulative, nextSeedRandWeight,
                                            side="left")
            nextSeed = d[nextSeedIndex]
            #  Repeat the above 3 statements if the seed already exists.
            while (nextSeedIndex in seedsIndexList):
                nextSeedRandWeight = np.random.randint(0, greatestCumVal)
                nextSeedIndex = np.searchsorted(distCumulative,
                                                nextSeedRandWeight, side="left"
                                                )
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


# Function to determine the stop condition for cluster iterations
def shouldStop(oldCentroids, centroids, iterations, maxIters):
    if iterations > maxIters:
        return True
    return np.array_equal(oldCentroids, centroids)


# Function to get updated labels as per the centroids
def getLabels(data, centroids):
    labels = []
    for i in data:
        # NumPy functions, take about 8-9 seconds.
        distances = np.sqrt(((centroids-i)**2).sum(axis=1))
        label = np.where(distances == min(distances))
        labels.append(label[0][0])
    return np.array(labels)


# Function to get new centroids
def getNewCentroids(data, labels, k, features):
    newCentroids= np.empty(shape = [0, features])
    for i in range(k):
        indices = np.where(labels == i)
        centroid = np.mean(data[indices[0], :], axis=0)
        newCentroids = np.vstack((newCentroids, centroid))
    return newCentroids

        
# Scatter plot of kmeans output
def drawPlot(X, y, figName):
    plt.figure(figName, figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title('K-Means ' + figName + ' Scatter Plot')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    #plt.xticks(())
    #plt.yticks(())
    plt.show()    
    
    
def KMeans(data, k, maxIters, initMethod):
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
            centroids = getInitCentroids(initMethod, k, data, numDocs,
                                         numFeatures)
            oldCentroids = centroids
            labels = getLabels(data, centroids)
        # Assign centroids based on datapoint labels
        centroids = getNewCentroids(data, labels, k, numFeatures)
    return labels
