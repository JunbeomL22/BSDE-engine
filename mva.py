from engine import *
from numpy import linalg, pi
import termstructure as ts
from scipy.stats import norm

class MVABSDE(MCBSDEEngine):
    def __init__(self, all_processes, simulation_spec, product, bsde_info):
        MCBSDEEngine.__init__(self, all_processes, simulation_spec, product)
        ''''''
        self.risk_free_rate = bsde_info['OIS term structure']
        self.im_rate = bsde_info['IM term structure']
        self.alpha = bsde_info['cvar alpha']
        self.Delta = bsde_info['payment gap']
        self.c_alpha = self.get_c_alpha(self.alpha)
        self.loaded_sqrt = math.sqrt

    def get_c_alpha(self, alpha):
        x = norm.ppf(alpha)
        denominator = (1.-alpha) * math.sqrt(2. * pi)
        nominator = math.exp( - x * x / 2. )
        return nominator / denominator

    def generator(self, t, x, y, z):
        #import pdb;pdb.set_trace()
        r = self.risk_free_rate.short_rate(t)
        R = self.im_rate.short_rate(t)
        modular_z = linalg.norm(z)
        
        gap_adj = self.loaded_sqrt(min(t+self.Delta, self.product.maturity) - t)
        
        ret = -r * y + R * self.c_alpha * gap_adj * modular_z

        return ret


