import numpy as np
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])
Y=np.array([2,4,6,8,10])

w,b,lr=0,0,0.01
loss=[]

for i in range(100):
    y=w*X+b
    dw=(-2/len(X))*sum(X*(Y-y))
    db=(-2/len(X))*sum(Y-y)
    w-=lr*dw
    b-=lr*db
    loss.append(np.mean((Y-y)**2))

print("Weight=",round(w,2))
print("Bias=",round(b,2))

plt.plot(loss)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
