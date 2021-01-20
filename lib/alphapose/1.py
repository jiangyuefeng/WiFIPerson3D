import numpy as np
b1 =[]
a = np.random.random((4,17,2))
print(a.shape[1])
b = np.random.random((4,17))
c = np.expand_dims(b, axis=2)
b1 = np.concatenate((a,c),axis=2)
print(b1.shape)