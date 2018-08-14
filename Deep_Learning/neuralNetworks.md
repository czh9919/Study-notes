# Neural Networks

据我所知，深度学习都是需要由机器自己学习的，并不可能是用户输入$w_1w_2b$,于是我们引入神经网络(Neural Networks)

## 从感知机到神经网络

这里，反正笔记嘛，懒得找图了，电脑也懒得画了

## 阶跃函数

    def step_function(x):
        if x>0:
            return 1
        else:
            return 0
为了未来的操作，改为

    def step_function(x):
        y=x>0
        return y.astype(np.int)

## sigmoid函数

    def sigmoid(x):
        return 1/(1+np.exp(-x))

## 非线性函数

神经网络的激活函数必须使用非线性函数。对线性函数，总存非隐藏层的神经网络与之对应，那么就无法发挥出叠加层的势，所以，激活函数必须使用非线性函数。

## RELU函数

$$
y=\left\{
\begin{array}{c}
x (x>0)\\
0 (x\leq0)
\end{array}
\right.
$$
实现：

    def relu(x):
        return np.maximum(0,x)