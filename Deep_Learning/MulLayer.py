class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None
    
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y

        return out
    
    def backward(self,dout):
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy
#测试  
apple=100
apple_num=2
tax=1.1
#layer
mul_apple_layer=MulLayer()
mul_tax_layer=MulLayer()
#forward
apple_price=mul_apple_layer.forward(apple,apple_num)
price=mul_tax_layer.forward(apple_price,tax)
print(price)#220