# 神经网络的学习

## 过拟合

对某种数据集过度拟合的状态成为过拟合

## 损失函数

### 均方误差

如下所示
$$
E=\frac{1}{2}\sum_{k}{(y_k-t_k)^2}
$$
$y_k$表示神经网络的输出，$t_k$表示监督数据,$k$表示数据的维数
代码实现

    def mean_squared_error(y,t):
        return 0.5*np.sum((y-t)**2)
见86页

### 交叉熵(shang)误差

如下所示
$$
E=-\sum_{k}{t_k\ln y_k}
$$

    def cross_entropy_error(y,t):
        delta=le-7
        return -np.sum(t*np.log(y+delta))

### mini-batch学习

从训练数据中国随机选一批，再对这一批进行学习叫mini-batch学习