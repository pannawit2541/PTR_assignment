import numpy as np
import os
from numba import jit

@jit
def edit_distance(str1="", str2=""):

    edit_distance = np.zeros((len(str1)+1, len(str2)+1))
    edit_distance[0, :], edit_distance[:, 0] = np.arange(
        (len(str2)+1)), np.arange((len(str1)+1))

    for i in range(1, edit_distance.shape[0]):
        for j in range(1, edit_distance.shape[1]):
            cost = min(edit_distance[i-1, j-1],  # replace
                       edit_distance[i-1, j],  # delete
                       edit_distance[i, j-1])  # insert
            if str1[i-1] != str2[j-1]:
                cost += 1
            edit_distance[i, j] = cost
    return edit_distance, edit_distance[-1, -1]

def min_index(arr):
    mean_arr = np.mean(arr, axis=1)
    id = np.argmin(mean_arr)
    return id

def update_Et(arr1, arr2):
    distance = []
    for i in range(arr1.shape[0]):
        _, d = edit_distance(arr1[i], arr2[i])
        distance.append(d)
    return max(distance)

def predict(U,y_test):
    pre_cluster = []
    for col in U.T:
        cluster = np.where(col == 1)[0]
        pre_cluster.append(cluster+1)
    
    pre_cluster = np.array(pre_cluster)
    return np.concatenate((y_test,pre_cluster),axis=1)

def generate_U(distanc_vector):

    U = np.zeros(distanc_vector.shape)
    minEachCols = np.argmin(distanc_vector, axis=0)

    for i in range(U.shape[1]):
        U[minEachCols[i], i] = 1

    return U

def calculate_distance(X1, X2):
    """calculate each distance of X1,X2 when |X1| >= |X2|

    Args:
        X1 (Array): Sameple of data 
        X2 (Array): K-mean cluster or any sameple of data 

    Returns:
        Array: Metrix has size : (|X2|,|X1|)
    """
    distance = []

    for i, c in enumerate(X2):
        d = []
        for x in X1:
            _, edit_dis = edit_distance(c, x)
            d.append(edit_dis)
        distance.append(d)
        print(i+1, "/", X2.shape[0])

    return np.array(distance).reshape(X2.shape[0], X1.shape[0])

def update_k_means(X, U):
    V_new = []
    for c in range(U.shape[0]):
        id = np.where(U[c, :] == 1)
        str_arr = []
        for j in id:
            str_arr.append(X[j])
        str_arr = np.array(str_arr).reshape(-1)
        print("cluster ", c+1, "size is :", len(str_arr))
        distance = calculate_distance(str_arr, str_arr)
        min_id = min_index(distance)
        V_new.append(str_arr[min_id])

    return np.array(V_new)

def preprocess_data(path):
    data = []
    for filename in os.listdir(path):
        with open(path+filename, "r") as f:
            for line in f:
                txt = line.split('/')
                type = txt[1].split(' ')[3]
                index = txt[1].split(' ')[1]
                chromosome = txt[-1][1:-1]
                data.append([chromosome, index, type])

    data = np.array(data)
    np.random.shuffle(data)
    return data[:, 0], data[:, 1:].astype(int)


if __name__ == "__main__":

    X_train, _ = preprocess_data('./chrom/train_set/')  # preprocess data
    x_test, y_test = preprocess_data('./chrom/test_set/')
    k = 22  # number of k-means
    t_max = 10  # maximum number of iteration
    threshold = 0  # termination theshold
    Et = float('inf')
    V = np.random.choice(X_train, k)  # pick up k-mean cluster from X

    # * Train cluster
    for i in range(t_max):
        print("-------------------------------- epoch:",
              i+1, "--------------------------------")
        distance = calculate_distance(X_train, V)
        U = generate_U(distance)

        V_new = update_k_means(X_train, U)
        Et = update_Et(V, V_new)
        print("-------------------------------- Et:",
              Et, "--------------------------------")
        if Et <= threshold:
            break
        V = V_new

    # * Test cluster
    distance_pre = calculate_distance(x_test, V)
    U_test = generate_U(distance_pre)
    y_pre = predict(U_test,y_test)


    np.savetxt('./chrom/predict/values.csv', y_pre, delimiter=",",fmt='%d')
    np.savetxt('./chrom/predict/label.csv', y_test, delimiter=",",fmt='%d')
