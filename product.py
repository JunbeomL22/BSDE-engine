import math

class Product:
    def __init__(self, product_info):
        self.maturity = product_info['maturity']

    def terminal_payoff(self, x):
        pass

    
class CallOption(Product):
    def __init__(self, product_info):
        Product.__init__(self, product_info)
        self.x_price = product_info['exercise price']

    def terminal_payoff(self,  x):
        return max(x[0] - self.x_price, 0.)


class CallForward(Product):
    def __init__(self, product_info):
        Product.__init__(self, product_info)
        self.x_price = product_info['exercise price']

    def terminal_payoff(self,  x):
        return x - self.x_price

