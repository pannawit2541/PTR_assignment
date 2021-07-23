import numpy as np
from numpy import linalg as LA

# mean
def mean(x): return x.mean(axis=0)

# std
def std(x): return x.std(axis=0)

# covarian-matrix
def cov_matrix(x):
    fact = x.shape[0] - 1
    return np.dot((x-mean(x)).T,(x-std(x)))*(1/fact)

# multivariate normal distribution
def multi_distribution(x,cov,mean): 
    const = ((2*np.pi)**(cov.shape[1]/2))
    cov_norm = LA.norm(cov)**(0.5)
    exp = np.exp(-0.5*np.dot(np.dot((x-mean),LA.inv(cov)),(x-mean).T))

    return ((1/(const*cov_norm))*exp).diagonal().reshape(-1,1) # return only diagonal values

def cross_validations_split(shape,folds):
    fold_size = int(shape * folds/100)
    k = 0
    index = []
    for i in range(1,folds+1):
        index.append([k,i*fold_size]) if (i < folds) else index.append([k,shape])
        k = i*fold_size
    return index

# probability of Wi
def prob_of_p(n,N):
    return n/N


# for 2 classes
def bayes_rules(f1,f2,p1,p2):
    likelihood_ratio = f1/f2
    threshold = p2/p1
    decision_matrix =  (likelihood_ratio > threshold)
    
    # unique, counts = np.unique(y_pred, return_counts=True)
    # print(dict(zip(unique, counts)))

    return np.where(decision_matrix,np.float64(1),np.float64(2)).reshape(-1)

# confusion matrix
def confusion_matrix(y_pred,y_true):

    if y_true.shape != y_pred.shape : return

    def _condition(y_pred,y_true):
        if y_pred == y_true and y_true == 1:
            return "TN"
        elif y_pred != y_true and y_true == 2:
            return "FP"
        elif y_pred != y_true and y_true == 1:
            return "FN"
        return "TP"
    
    matrix = np.array([[0, 0], [0, 0]])

    for i in range(y_true.shape[0]):
        result = _condition(y_pred[i],y_true[i])
        if result == "TN":
           matrix[0][0] += 1
        elif result == "FN":
            matrix[0][1] += 1
        elif result == "FP":
            matrix[1][0] += 1
        else:
            matrix[1][1] += 1

    return matrix


if __name__ == "__main__":

    # split features and classes to two classes
    data = np.genfromtxt('TWOCLASS.csv',delimiter=',')[1:,:]
    np.random.shuffle(data) # shuffle data

    for i,j in cross_validations_split(data.shape[0],10):
        
        x_population =  np.concatenate((data[:i],data[j:]))
        x_samples =  data[i:j]

        # separate the data to two class
        features_1 = x_population[x_population[:,-1] == 1][:,:-1]
        features_2 = x_population[x_population[:,-1] == 2][:,:-1]
  
        f1 = multi_distribution(x_samples[:,:-1],cov_matrix(features_1),mean(features_1))
        f2 = multi_distribution(x_samples[:,:-1],cov_matrix(features_2),mean(features_2))

        p1 = prob_of_p(features_1.shape[0],x_population.shape[0])
        p2 = prob_of_p(features_2.shape[0],x_population.shape[0])
        y_pred = bayes_rules(f1,f2,p1,p2)
        y_true = x_samples[:,-1]
     
        print(confusion_matrix(y_pred,y_true))


    

