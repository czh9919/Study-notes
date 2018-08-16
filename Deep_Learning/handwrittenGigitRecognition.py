# coding: utf-8
import sys, os
sys.path.append(os.path.abspath('F:/notes'))
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
def sigmoid(x):
    return 1/(1+np.exp(-x))
def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)
    return x_test,t_test
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network
def predict(network,x):
    w1,w2,w3=network['w1'],network['w2'],network['w3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=softmax(a3)
    return y
x,t=get_data()
network=init_network()
accuracy_cnt=0
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y)
    if p==t[i]:
        accuracy_cnt+=1
print("Accuracy"+str(float(accuracy_cnt)/len(x)))