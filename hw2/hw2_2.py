import numpy as np
import os

def edit_distance(str1="",str2=""):

    edit_distance = np.zeros((len(str1)+1,len(str2)+1))
    edit_distance[0,:],edit_distance[:,0] = np.arange((len(str2)+1)),np.arange((len(str1)+1))

    for i in range(1,edit_distance.shape[0]):
        for j in range(1,edit_distance.shape[1]):
            cost =  min(edit_distance[i-1,j-1], # replace
                    edit_distance[i-1,j], # delete
                    edit_distance[i,j-1]) # insert
            if str1[i-1] != str2[j-1]:
                cost += 1
            edit_distance[i,j] =  cost

    return edit_distance,edit_distance[-1,-1]

def generate_U(arr):
    i = arr.shape[1]-1
    for col in arr.T:
        if i < 0 :
            id = np.random.randint(0, arr.shape[1]-1)
            col[id] = 1
        else : 
            col[i] = 1
            i -= 1
        
    
    return arr

def calculate_distance(X,k_means):
    distance = []
    for x in X:
        d = []
        for c in k_means:
            _,edit_dis = edit_distance(x,c)
            d.append(edit_dis)
        print(d)
        distance.append(d)

    return np.array(distance).reshape(X.shape[0],k_means.shape[0])


def preprocess_data(path):
    data = []
    for filename in os.listdir(path):
        with open(path+filename, "r") as f:
            for line in f:
                txt = line.split('/')
                type = txt[1].split(' ')[5]
                chromosome = txt[-1][1:-1]
                data.append([chromosome,type])

    data = np.array(data)
    np.random.shuffle(data)
    return data[:,0],data[:,1].astype(float)
                
if __name__ == "__main__":
    
    chromosome,type = preprocess_data('./chrom/') # preprocess data

    k = 22 # number of k-means
    t_max =  1000000 # maximum number of iteration
    threshold = 10e-6 # termination theshold 
    k_means = np.random.choice(chromosome, k) # random pick up k means 
  
    U = generate_U(np.zeros((type.shape[0],chromosome.shape[0])))
    print(U)


    # TODO: calculate EditDistance between mean and X
    # ---------------------------------------------
    # distance = calculate_distance(chromosome,k_means)
    # print(distance)



    # ---------------------------------------------




        