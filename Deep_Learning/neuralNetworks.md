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

神经网络的激活函数必须使用非线性函数。对线性函数，总存非隐藏层的神经网络与之对应，那么就无法发挥出叠加层的优势，所以，激活函数必须使用非线性函数。

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

## 神经网络的内积

我前几天才刚刚看了矩阵的乘法，这里就是一个利用矩阵乘法的例子。在py里，可以利用np里的dot函数计算，这样非常方便快捷。
这里我们尝试一个三层神经网络的实现。

### 符号引入

先来第一个吧，$w^{(1)}_{1 2}$中(1)表示第一层的权重，右下角的1表示后一层的第1个神经元，2表示前一层的第2个神经元
所以$w^{(1)}_{1 2}$表示前一层的第二个神经元到后一层的第一个神经元$a^{(1)}_1$的权重
那么$a^{(1)}_1$就表示为隐藏层的第一个神经元
那么$a^{(1)}$就表示为隐藏层的第一层
那么$a^{(0)}$就表示为输入层
大写小写没啥关系……
不知道这么说有没有错:)

### 各层之间的信号传递

首先拿输入层到第一层的第一个神经元传递举例
用数学式表示$a^{(1)}_1$
$$
a^{(1)}_1=w^{(1)}_{1 1}x_1+w^{(1)}_{1 2}x_2+b^{(1)}_1
$$
那么拓展开来就是
$$
A^{(1)}=XW^{(1)}+B^{(1)}
$$
其中
$$
A^{(1)}=\begin{pmatrix} a^{(1)}_1 & a^{(1)}_2 & a^{(1)}_3  \\ \end{pmatrix}
$$
$$
X=\begin{pmatrix} x_1 &x_2 \\ \end{pmatrix}
$$
$$
B^{(1)}=\begin{pmatrix} b^{(1)}_1 &b^{(1)}_2&b^{(1)}_3  \\ \end{pmatrix}
$$
$$
W^{(1)}=\begin{pmatrix} w^{(1)}_{11} & w^{(1)}_{21}&w^{(1)}_{31} \\ w^{(1)}_{12} & w^{(1)}_{22}&w^{(1)}_{32} \\ \end{pmatrix}
$$
具体实现嘛，懒得写了,这都看不懂那就得回去找你线代老师了:)
而我就牛逼了，我没有线代老师，还没开课:)

### 代码实现小结

代码就不写在这里了，但还不是很能理解

## 输出层

分类问题，顾名思义，就是分类
回归问题，就是，呃，类似根据图像估算一些东西的问题

回归问题一般用恒等函数
分类问题一般用softmax函数

### 恒等函数和softmax函数

恒等函数相对简单
而softmax函数看起来就略难了

真糟糕，又要插入数学公式:)

$$
y_k=\frac{exp(a_k)}{\sum_{i=1}^n exp{(a_i)}}
$$
这就是softmax函数，输出层共有$n$个神经元，计算第$k$个神经元的输出$y_k$,然后他的分子是输入信号$a_k$的指数函数，分号是所有输入信号的和。呃，不是很懂啊,过了5s，行吧，懂了，大概吧，呃，看书无聊，找点乐趣
这样输出层的神经元就会收到所有输入信号的影响
在实际使用中，可以改进一下，softmax函数，防止$e^x$溢出
有了，softmax函数，我们就可以概率的判断问题

### 输出层的神经元数量:)

一般对分类问题来说，有几类，就有几个输出神经元，呃，不太对，自己知道就行了

## 手写数字识别

对其中的load_mnist函数

    (x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)#x表示图像，t表示标签
然后，我们现在已经了解怎么使用数据集

### 神经网络的推理处理

设定学习率很重要，学习率也叫超参数，学习率设定不能太大，也不能太小

## 梳理

我觉得这个地方有必要梳理一遍，先读取数据，然后设定超参数，然后取随机，利用mini-batch学习，然后计算梯度，然后更新参数
这里补充一下，如何利用梯度更新参数，这里给出公式
$$
x_0=x_0-\eta{\frac{\partial f}{\partial x_0}}
$$
$$
x_1=x_1-\eta{\frac{\partial f}{\partial x_1}}
$$
前几天太丧了，没耐心看书

### 基于测试数据的评价

