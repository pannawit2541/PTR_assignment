import numpy as np

a = np.array([[1,0,0,1],[0,1,1,0]])
print(a)
id = np.argwhere(a == 1)
print(id[0])
