import numpy as np

np.random.seed(1)
data=np.random.normal(10,2,1000)

mean=np.mean(data)
var=np.var(data)

print("Estimated Mean:",round(mean,2))
print("Estimated Variance:",round(var,2))
