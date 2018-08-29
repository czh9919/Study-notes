import os
from MulLayer import MulLayer
class AddLayer:
    def __init__(self):
        pass
    def forward(self,x,y):
        out=x+y
        return out
    def backward(self,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy
#测试
apple=100
apple_num=2
orange=150
orange_num=3
tax=1.1
#layer
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=MulLayer()
#forward
apple_price=mul_apple_layer.forward(apple,apple_num)
orange_price=mul_orange_layer.forward(orange,orange_num)
all_price=add_apple_orange_layer.forward(apple_price,orange_price)
price=mul_tax_layer.forward(all_price,tax)
#backward
dprice=1
dall_price,dtax=mul_tax_layer.backward(dprice)
dapple_price,dorange_price=add_apple_orange_layer.backward(dall_price)
dorange_price,dorange_num=mul_orange_layer.backward(dorange_price)
dapple,dapple_num=mul_apple_layer.backward(dall_price)

print(price)#715
print(dapple_num,dapple,dorange_num,dtax)#110 2.2 3.3 165 650