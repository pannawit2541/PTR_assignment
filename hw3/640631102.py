import numpy as np
import matplotlib.pyplot as plt


def readfile():
    filename = 'ionosphere.csv'

    features = np.genfromtxt(filename, delimiter=',')[1:, :-1]
    targets = np.unique(np.genfromtxt(filename, delimiter=',', dtype=str)[
        1:, -1], return_inverse=True)[1]  # g = 1 , b = 0

    return features, targets,

def mean_abs_diff(X):
    return np.mean(np.abs(X-np.mean(X, axis=0)), axis=0)

def plotBar(X):
    plt.bar(np.arange(X.shape[1]), mean_abs_diff(X), color='red')
    plt.xlabel("feature")
    plt.ylabel("value")
    plt.title("mean absolute difference")
    plt.show()

def euclidean_distance(p1,p2,label):
    distance = []

    for i in range(p1.shape[0]):
        dis = np.sqrt(np.sum((p1[i,:]-p2)**2,axis=1))
        distance.append(list(zip(dis,label)))
    
    return np.array(distance)

def knn(distance,k=3):
    k_nearest = []
    for i in range(distance.shape[0]):
        # sorted
        dis_sorted =  distance[i][distance[i][:, 0].argsort()]
        
        # k-nearest distances
        unique, counts = np.unique(dis_sorted[:k,:][:,1], return_counts=True)
        k_nearest.append(max(list(zip(unique, counts)))[0])

    return np.array(k_nearest)
def cross_validations_split(shape,folds):
    fold_size = int(shape * folds/100)
    k = 0
    index = []
    for i in range(1,folds+1):
        index.append([k,i*fold_size]) if (i < folds) else index.append([k,shape])
        k = i*fold_size
    return index

if __name__ == "__main__":
    X, Y = readfile()
    id_features = np.where(mean_abs_diff(X) > 0.4)
    X = X[:, id_features[0]]
