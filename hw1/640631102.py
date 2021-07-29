import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt

# mean
def mean(x): return x.mean(axis=0)

# std
def std(x): return x.std(axis=0)

# covarian-matrix
def cov_matrix(x):
    fact = x.shape[0] - 1
    return np.dot((x-mean(x)).T,(x-std(x)))*(1/fact)

# multivariate normal distribution
def multi_distribution(X,cov,mean): 
    const = ((2*np.pi)**(cov.shape[1]/2))
    cov_norm = LA.norm(cov)**(0.5)

    # ? if err : use this below
    # exp = np.exp(-0.5*np.dot(np.dot((X-mean),LA.inv(cov)),(X-mean).T))
    # return ((1/(const*cov_norm))*exp).diagonal().reshape(-1,1) # return only diagonal values
    # ? ------------------------

    exp = np.array(list(map(lambda x: np.exp(-0.5*np.dot(np.dot((x-mean),LA.inv(cov)),(x-mean).T)),X)))
    return ((1/(const*cov_norm))*exp) 
    

# cross_validations
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
    
    return np.where(decision_matrix,np.float64(1),np.float64(2)).reshape(-1)

# confusion matrix
def confusion_matrix(y_pred,y_true,err = False):

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

    if err: 
        return matrix,100-(matrix[0][0]+matrix[1][1])*100/y_true.shape[0]
    return matrix

def preprocess_data(data,i,j):

    population = np.concatenate((data[:i],data[j:]))
    samples = data[i:j]

    # calculate P(Wi)
    p1 = prob_of_p(population[population[:,-1] == 1][:,:-1].shape[0],population.shape[0])
    p2 = prob_of_p(population[population[:,-1] == 2][:,:-1].shape[0],population.shape[0])
    
    pre_data = {
        'population' : population,
        'x_sample' : samples[:,:-1],
        'x_class1' : population[population[:,-1] == 1][:,:-1], # separate the data to class 1
        'x_class2' : population[population[:,-1] == 2][:,:-1], # separate the data to class 2
        'p1': p1,
        'p2': p2,
        'y_sample': samples[:,-1]
    }
    return pre_data

# main
if __name__ == "__main__":

    # split features and classes to two classes
    data = np.genfromtxt('TWOCLASS.csv',delimiter=',')[1:,:]
    np.random.shuffle(data) # shuffle data

    for i,j in cross_validations_split(data.shape[0],10):

        # * --------------- preprocess data ---------------
        x = preprocess_data(data,i,j)

        # calculate multivariate normal distribution
        f1 = multi_distribution(x['x_sample'],cov_matrix(x['x_class1']),mean(x['x_class1']))
        f2 = multi_distribution(x['x_sample'],cov_matrix(x['x_class2']),mean(x['x_class2']))

        # evaluate
        y_pred = bayes_rules(f1,f2,x['p1'],x['p2'])
        y_true = x['y_sample']
        print(confusion_matrix(y_pred,y_true,err=True))


    

