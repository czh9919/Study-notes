import sys,os
sys.path.append(os.path.abspath('F:/notes'))
import numpy as np
from common.functions import softmax#输出层，不记得了。。。还回去看了

def sigmoid(x):
    return 1/(1+np.exp(-x))
def cross_entropy_error(y,t):
    if t.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad
class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)
    def predict(self,x):
        return np.dot(x,self.W)
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)
        return loss
net=simpleNet()
print(net.W)
x=np.array([0.6,0.9])
print(net.predict(x))
p=net.predict(x)
np.argmax(p)
t=np.array([0,0,1])
f=lambda w:net.loss(x,t)
dW=numerical_gradient(f,net.W)
print(dW)