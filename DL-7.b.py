import math

# Inputs
x = 1
target = 1

# Initial weight and bias
w = 0.5
b = 0.1
lr = 0.1

# Forward propagation
z = w*x + b
y = 1/(1+math.exp(-z))

# Backpropagation
error = target - y
gradient = error * y * (1-y)

w = w + lr * gradient * x
b = b + lr * gradient

print("Updated Weight =", round(w,4))
print("Updated Bias =", round(b,4))
