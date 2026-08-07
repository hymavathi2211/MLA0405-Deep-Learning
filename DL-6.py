from sklearn.linear_model import Perceptron

X=[[0,0],[0,1],[1,0],[1,1]]
Y=[0,0,0,1]

model=Perceptron()
model.fit(X,Y)

print(model.predict([[1,1]]))
print(model.predict([[0,1]]))
