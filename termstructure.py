from math import exp

class TermStructure:
    def __init__(self, quote):
        self.r = quote # considering constant interest rate
                       # for the time being
    def discount_factor(self, t):
        return self.pdiscountfactor(0., t)
    
    def pdiscount_factor(self, s, t):
        return exp( -self.r * (t - s) ) # should be remedied

    def forward_rate(self, s, t):
        return self.r # should be remedied

    def short_rate(self, t):
        return self.forward_rate(t, t + 0.000001)
