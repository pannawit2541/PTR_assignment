import numpy as np
from numpy import linalg as LA
import pandas as pd

# mean
def mean(x): return np.round(x.mean(axis=0),3)

# std
def std(x): return np.round(x.std(axis=0),3)

# covarian-matrix
def cov_matrix(x):
    fact = x.shape[0] - 1
    return np.round(np.dot((x-mean(x)).T,(x-std(x)))*(1/fact),3) 

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

    x_class1 = population[population[:,-1] == 1][:,:-1]
    x_class2 = population[population[:,-1] == 2][:,:-1]

    # calculate P(Wi)
    p1 = prob_of_p(population[population[:,-1] == 1][:,:-1].shape[0],population.shape[0])
    p2 = prob_of_p(population[population[:,-1] == 2][:,:-1].shape[0],population.shape[0])

    # calculate COV(Wi)
    cov_1 = cov_matrix(x_class1)
    cov_2 = cov_matrix(x_class2)

    # calculate mean(Wi)
    mean_1 = mean(x_class1)
    mean_2 = mean(x_class2)
    
    pre_data = {
        'population' : population,
        'x_sample' : samples[:,:-1],
        'x_class1' : x_class1, # separate the data to class 1
        'x_class2' : x_class2, # separate the data to class 2
        'p1': p1,
        'p2': p2,
        'y_sample': samples[:,-1],
        'cov1': cov_1,
        'cov2': cov_2,
        'mean1': mean_1,
        'mean2': mean_2,
    }
    return pre_data

# main
if __name__ == "__main__":

    # split features and classes to two classes
    data = np.genfromtxt('TWOCLASS.csv',delimiter=',')[1:,:]
    np.random.shuffle(data) # shuffle data

    # mean1_t1 = []
    # mean2_t1 = []
    # cov1_t1 = []
    # cov2_t1 = []
    # conf_t1 = [] 

    # mean1_t2 = []
    # mean2_t2 = []
    # cov1_t2 = []
    # cov2_t2 = []
    # conf_t2 = [] 

    k = 1
    for i,j in cross_validations_split(data.shape[0],10):

        # * --------------- preprocess data ---------------
        x1 = preprocess_data(data,i,j) # for test 1
        x2 = preprocess_data(data[:,[0,1,-1]],i,j) # for test 2
        
        # calculate multivariate normal distribution test 1
        fx1_1 = multi_distribution(x1['x_sample'],x1['cov1'],x1['mean1'])
        fx1_2 = multi_distribution(x1['x_sample'],x1['cov2'],x1['mean2'])
        
        # calculate multivariate normal distribution test 2
        fx2_1 = multi_distribution(x2['x_sample'],x2['cov1'],x2['mean1'])
        fx2_2 = multi_distribution(x2['x_sample'],x2['cov2'],x2['mean2'])

        # evaluate test 1
        y_pred1 = bayes_rules(fx1_1,fx1_2,x1['p1'],x1['p2'])
        y_true1 = x1['y_sample']

        # evaluate test 1
        y_pred2 = bayes_rules(fx2_1,fx2_2,x2['p1'],x2['p2'])
        y_true2 = x2['y_sample']

        
        print("############### K=", k ," #################")
        k+=1
    
        # conf_t1.append(confusion_matrix(y_pred1,y_true1).astype(int))
        # mean1_t1.append(x1['mean1'])
        # mean2_t1.append(x1['mean2'])
        # cov1_t1.append(x1['cov1'])
        # cov2_t1.append(x1['cov2'])

        # print("--------------------------------")
        # conf_t2.append(confusion_matrix(y_pred2,y_true2).astype(int))
        # mean1_t2.append(x2['mean1'])
        # mean2_t2.append(x2['mean2'])
        # cov1_t2.append(x2['cov1'])
        # cov2_t2.append(x2['cov2'])
        # print("--------------------------------")


    # mean1_t1 = np.array(mean1_t1).reshape(-1,4)
    # np.savetxt("mean1_t1.csv",mean1_t1,delimiter=",")
    # mean2_t1 = np.array(mean2_t1).reshape(-1,4)
    # np.savetxt("mean2_t1.csv",mean2_t1,delimiter=",")

    # cov1_t1 = np.array(cov1_t1).reshape(-1,4)
    # np.savetxt("cov1_t1.csv",cov1_t1,delimiter=",")
    
    # print(len(cov2_t1))
    # cov2_t1 = np.array(cov2_t1).reshape(-1,4)
    # np.savetxt("cov2_t1.csv",cov2_t1,delimiter=",")
    
    # conf_t1 = np.array(conf_t1).reshape(-1,2)
    # np.savetxt("conf_t1.csv",conf_t1,delimiter=",")

    # mean1_t2 = np.array(mean1_t2).reshape(-1,2)
    # np.savetxt("mean1_t2.csv",mean1_t2,delimiter=",")
    # mean2_t2 = np.array(mean2_t2).reshape(-1,2)
    # np.savetxt("mean2_t2.csv",mean2_t2,delimiter=",")
    # cov1_t2 = np.array(cov1_t2).reshape(-1,2)
    # np.savetxt("cov1_t2.csv",cov1_t2,delimiter=",")
    # cov2_t2 = np.array(cov2_t2).reshape(-1,2)
    # np.savetxt("cov2_t2.csv",cov2_t2,delimiter=",")
    # conf_t2 = np.array(conf_t2).reshape(-1,2)
    # np.savetxt("conf_t2.csv",conf_t2,delimiter=",")
    

