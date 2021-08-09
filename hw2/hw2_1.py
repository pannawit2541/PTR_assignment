import numpy as np

# cross_validations
def cross_validations_split(shape,folds):
    fold_size = int(shape * folds/100)
    k = 0
    index = []
    for i in range(1,folds+1):
        index.append([k,i*fold_size]) if (i < folds) else index.append([k,shape])
        k = i*fold_size
    return index

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

if __name__ == "__main__":
    # split features and classes to two classes
    data = np.genfromtxt('TWOCLASS.csv',delimiter=',')[1:,:]
    np.random.shuffle(data) # shuffle data

    for i,j in cross_validations_split(data.shape[0],10):

        # collect data
        train = np.concatenate((data[:i],data[j:])).copy()
        test = data[i:j].copy()
        x_train,y_train = train[:,:-1],train[:,-1]
        x_test,y_test = test[:,:-1],test[:,-1]
        distance = euclidean_distance(x_test,x_train,y_train)
        print(distance.shape)
        # print(distance[0])
        # print(knn(distance,3))
        print(knn(distance,3))
        print(y_test)
        break


       