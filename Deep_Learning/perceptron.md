# 感知机(perceptron)

## 啥是感知机

$$
y=\left\{
\begin{array}{c}
0 (w_1x_1+w_2x_2\leq\theta)\\
1 (w_1x_1+w_2x_2>\theta)
\end{array}
\right.
$$

这就是感知机,$w_1w_2$是权重(weight),$x_1x_2$是输入信号

## 简单逻辑电路

### 与门

    def AND(x1,x2):
        w1,w2,theta=0.5,0.5,0.7
        tmp=x1*w1+x2*w2
        if tmp<=theta:
            return 0
        elif tmp>theta:
            return 1

在这里，我们引入偏置b，那么函数就变为

    def AND(x1,x2):
        w1,w2,theta=0.5,0.5,0.7
        b=-0.7
        tmp=x1*w1+x2*w2+b
        if tmp<=theta:
            return 0
        elif tmp>theta:
            return 1

那么，至此与门操作完成。但代码可以进行更改

    import numpy as np
    def AND(x1,x2):
        x=np.array([x1,x2])
        w=np.array([0.5,0.5])
        b=-0.7
        tmp=np.sum(w*x)+b
        if tmp<=theta:
            return 0
        elif tmp>theta:
            return 1

### 非门

    def NAND(x1,x2):
        x=np.array([x1,x2])
        w=np.array([-0.5,-0.5])
        b=0.7
        tmp=np.sum(w*x)+b
        if tmp<=0:
            return 0
        else:
            return 1
其实在这里，就是把$w_1w_2$和$b$变为相反值

### 或门

    def OR(x1,x2):
        x=np.array([x1,x2])
        w=np.array([0.5,0.5])
        b=-0.2
        tmp=np.sum(w*x)+b
        if tmp<=0:
            return 0
        else:
            return 1

## 感知机的局限性

这个有点难说，emmmm……，这么说不知道对不对，感知机只能表示直线分割的空间，难以表示曲线分割。

## 多层感知机

就是单个感知机的叠加
这里我记录下

    def XOR(x1,x2):
        s1=NAND(x1,x2)
        s2=OR(x1,x2)
        y=AND(s1,s2)
        return y
这里是个异或门  
异或门是个多层结构的神经网络  
叠加了多层的感知机也称为多层感知机
