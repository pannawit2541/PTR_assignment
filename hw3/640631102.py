import numpy as np
import matplotlib.pyplot as plt


def readfile():
    filename = 'ionosphere.csv'

    features = np.genfromtxt(filename, delimiter=',')[1:, :-1]

    targets = np.unique(np.genfromtxt(filename, delimiter=',', dtype=str)[
        1:, -1], return_inverse=True)[1]  # g = 1 , b = 0

    data = np.concatenate((features, targets.reshape(-1, 1)), axis=1)
    
    features = data[:,:-1]
    targets = data[:,-1]
    return features, targets


def mean_abs_diff(X):
    return np.mean(np.abs(X-np.mean(X, axis=0)), axis=0)


def plotBar(X):
    plt.bar(np.arange(X.shape[1]), mean_abs_diff(X), color='red')
    plt.xlabel("feature")
    plt.ylabel("value")
    plt.title("mean absolute difference")
    plt.show()


def euclidean_distance(p1, p2, label):
    distance = []

    for i in range(p1.shape[0]):
        dis = np.sqrt(np.sum((p1[i, :]-p2)**2, axis=1))
        distance.append(list(zip(dis, label)))

    return np.array(distance)


def knn(distance, k=3):
    k_nearest = []
    for i in range(distance.shape[0]):
        # sorted
        dis_sorted = distance[i][distance[i][:, 0].argsort()]

        # k-nearest distances
        unique, counts = np.unique(dis_sorted[:k, :][:, 1], return_counts=True)
        k_nearest.append(max(list(zip(unique, counts)))[0])

    return np.array(k_nearest)


def cross_validations_split(shape, folds):
    fold_size = int(shape * folds/100)
    k = 0
    index = []
    for i in range(1, folds+1):
        index.append([k, i*fold_size]) if (i <
                                           folds) else index.append([k, shape])
        k = i*fold_size
    return index


def confusion_matrix(y_pred, y_true):

    matrix = np.zeros((int(max(y_true))+1, int(max(y_true))+1))
    for i in range(y_pred.shape[0]):
        matrix[int(y_true[i]), int(y_pred[i])] += 1
    return matrix


if __name__ == "__main__":
    X,Y = readfile()

    # id_features = np.where(mean_abs_diff(X) > 0.4)
    # print(id_features)
    # data = np.concatenate((X[:, id_features[0]], Y.reshape(-1, 1)), axis=1)
    data = np.concatenate((X, Y.reshape(-1, 1)), axis=1)

    conf_arr = []
    acc_arr = []
    accuracy = 0
    for i, j in cross_validations_split(data.shape[0], 10):
        train = np.concatenate((data[:i], data[j:])).copy()
        test = data[i:j].copy()
        x_train, y_train = train[:, :-1], train[:, -1]
        x_test, y_test = test[:, :-1], test[:, -1]

        # find eculidean distance
        distance = euclidean_distance(x_test, x_train, y_train)

        # calculate KNN
        pred = knn(distance, 3)

        result = confusion_matrix(pred, y_test)
        conf_arr.append(result)
        acc_arr.append(np.trace(result)*100/np.sum(result))
        accuracy += np.trace(result)*100/np.sum(result)

    print("----------------------------------------------------------------")
    print(accuracy/10)
    conf_arr = np.array(conf_arr)
    acc_arr = np.array(acc_arr)

    np.savetxt("conf.csv", conf_arr.reshape(
        10*2, 2), delimiter=",", fmt='%d')

