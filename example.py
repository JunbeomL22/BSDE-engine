import mva
import process
import termstructure as ts
from util import *

risk_free_rate = 0.02
im_rate = 0.02
bsde_info = {}
bsde_info['OIS term structure'] = ts.TermStructure(risk_free_rate)
bsde_info['IM term structure'] = ts.TermStructure(im_rate)
bsde_info['cvar alpha'] = 0.99
bsde_info['payment gap'] = 0.02
## underlying asset information
params = {}
params['sigma'] = 0.25
params['mu'] = risk_free_rate
params['initial value'] = 20.
underlying_asset = process.ConstantGBMProcess(params)
all_underlying_process = process.AllProcesses( [underlying_asset], [1.])
## product information of a CallOption (class in product.py)
prod_info ={}
prod_info['maturity'] = 1.
prod_info['exercise price'] = 20.
call_option = product.CallOption(prod_info)

## simulation spec
sim_spec = {}
sim_spec['simulation number'] = 5000
sim_spec['seed number'] = 2
num_time_grid = 40 
time_step_length = prod_info['maturity'] / num_time_grid
sim_spec['time grid'] = list(map(lambda t: time_step_length * t, range(1, num_time_grid +1)))
'''  <++>  '''
# pricing using MVABSDE class
mc_bsde_engine = mva.MVABSDE(all_underlying_process, sim_spec, call_option, bsde_info)
mc_bsde_engine.backward_iteration()
## black scholes price and delta
(bs_price, bs_delta) = bs_calloption(params['initial value'],
                                    prod_info['exercise price'],
                                    params['sigma'],
                                    risk_free_rate,
                                    prod_info['maturity'])
## We get a price with R = 0 using MC.
## This value will be used for adjust the numerical value so as to reduce the error.
bsde_info_bs = {i:j for i, j in bsde_info.items()}
bsde_info_bs['IM term structure'] = ts.TermStructure(0.)
mc_bsde_bs = mva.MVABSDE(all_underlying_process, sim_spec, call_option, bsde_info_bs)
mc_bsde_bs.backward_iteration()
mc_bs_y = mc_bsde_bs.Y0
mc_bs_delta = mc_bsde_bs.Z0[0] / (params['sigma'] * params['initial value'])
# 
adj_price = bs_price - mc_bs_y
adj_delta = bs_delta - mc_bs_delta
#
print()
print('Call option price with MVA:            ', mc_bsde_engine.Y0 + adj_price)
print('call Option delta with MVA:            ',
      mc_bsde_engine.Z0[0] / (params['sigma'] * params['initial value']) + adj_delta)
print()
print('Black-Scholes call option price:       ', bs_price)
print('Black-Scholes call option delta:       ', bs_delta)
#
#
