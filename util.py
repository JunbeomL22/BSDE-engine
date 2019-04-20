import math
from scipy.stats import norm

def bs_calloption(s0, k, vol, r, mat):
    dsf = math.exp(-r*mat)
    d1 = ( math.log(s0/k) + ( r + vol*vol*0.5  )*mat\
            ) / ( vol * math.sqrt(mat))
    d2 = d1 - vol*math.sqrt(mat)
    Nd1 = norm.cdf(d1)  
    Nd2 = norm.cdf(d2)
    return (s0 * Nd1 - dsf*k*Nd2, Nd1)


def eval_func_tuple(f_args):
    """Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])


def fn(a):
    return  a+1
