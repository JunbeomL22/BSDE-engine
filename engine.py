import math, process, time, functools, itertools
import product, regression, util
from partition_clstyle import partition_clstyle as parting
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

tt = time.time

def select_column(vec, col):
    return [row[col] for row in vec]

def average(vec):
    return sum(vec) /len(vec)

class Engine:
    def __init__(self, all_processes, simulation_spec, product):
        self.all_processes = all_processes
        # Note that the first element in time_grid (vector, e.g., [0.5, 1.]) is not zero.
        self.time_grid = simulation_spec['time grid']
        # The number of time grids.
        self.tg_num = len(self.time_grid) 
        # The number of process
        self.pr_num = all_processes.p_num
        # The correlation of the multiple process. Recall that corr = [[1]]
        # Note that self.correlation is already cholesky decomposed.
        self.correlation = self.all_processes.corr
        # Initial values of processes
        self.all_iv = [p.initial_value for p in all_processes.processes]

        self.product = product
        

class MCEngine(Engine):
    '''Monte Carlo simulation engin only with Brownian motions.'''
    def __init__(self, all_processes, simulation_spec, product):
        Engine.__init__(self, all_processes, simulation_spec, product)
        self.sim_num = simulation_spec['simulation number']

        num_of_rand_num = self.tg_num * self.pr_num * self.sim_num
        self.seed = simulation_spec['seed number']
        '''Extracting random number using multiprocessing does not outperform.  
        When the number of random numbers is very large, e.g., more than 4 millions,
        comment the line below, and use the next multiprocessing code. However,
        note that multiprocessing approach is only 1 sec efficient than single
        core computing. 
        '''
        
        st1 = time.time()
        np.random.seed(self.seed)
        nm = np.random.normal
        self.random_numbers = list(map(lambda a: nm(0., 1.), range(num_of_rand_num)))
        ed1 = time.time()
        #import pdb;pdb.set_trace()
        '''
        st2 = time.time()        
        pool = multiprocessing.Pool()
        self.random_numbers = list(pool.map(<+fill up+>), range(num_of_rand_num))
        ed2 = time.time()
 
        print('single: ', ed1 - st1, 'multi: ', ed2 -  st2)
        '''
        #vectorize self.random_numbers
        pr = parting(self.random_numbers, self.pr_num)
        self.random_numbers = parting(pr, self.tg_num)
        #self.random_numbers = parting(tm, self.sim_num)
        
 
        self.path =[[[]]]
        # random rnumbers are generated in the following functions
        self.tg_with_zero = [0.]+self.time_grid[:-1] #no last element
        sqrt = math.sqrt
        self.dt = list(map(lambda r: r[0]-r[1],   zip(self.time_grid, self.tg_with_zero)))
        self.sqrt_dt = list(map(sqrt, self.dt))

        self.corr = self.all_processes.corr
        
        self.make_path()
   
    def make_path(self):
        '''path of underlying processes are generated'''
        # self.random_numbers are generated below        
        #import pdb; pdb.set_trace()       
        # To make self.random_numbers Brownian motion, i.e., random_numbers = dW
        #import pdb;pdb.set_trace()
        
        #import pdb; pdb.set_trace()
        def std_to_brownian(vec):
            ret = [] 
            for t in range(self.tg_num):
                tmp = [v*self.sqrt_dt[t] for v in vec[t]]
                ret.append(tmp)
            return ret
        
        # INDEPENDENT brownian motion
        self.brownian_path = np.array(list(map(std_to_brownian, self.random_numbers)))
        
        self.random_numbers = list(map(std_to_brownian, self.random_numbers))

        # To generate path of underlying process
        all_drift = [p.drift for p in self.all_processes.processes]
        all_diffusion = [p.diffusion for p in self.all_processes.processes]
        dot = np.dot
        
        def simulating_und(sliced_rand):
            '''note that output's length is self.tg_num + 1'''
            #import pdb; pdb.set_trace()        
            ret = [self.all_iv]
            x  = []
            diff = [0] * self.pr_num
            
            # correlize the random_numbers
            if len(self.corr) != 1:
                map_corr = functools.partial(dot, self.corr)
                sliced_rand = map(map_corr, sliced_rand)
 
            for t in range(self.tg_num):
                tm = self.tg_with_zero[t]
                dt_t = self.dt[t]
                x = ret[-1]
                for p in range(self.pr_num):
                    diff[p] = all_drift[p](x[p], tm) * dt_t\
                              + all_diffusion[p](x[p], tm) * sliced_rand[t][p]
                    ret.append( [i+j for i, j in zip(x, diff)] )
                    
            return ret
        
        self.path = list(map(simulating_und, self.random_numbers))                


class MCBSDEEngine(MCEngine):
    def __init__(self, all_processes, simulation_spec, product):
        MCEngine.__init__(self, all_processes, simulation_spec, product)
        last_und = [p[-1] for p in self.path]
        map_terminal_values = map(self.product.terminal_payoff, last_und)
        # Y is an array onwards
        self.Y = np.array(list(map_terminal_values))
        self.Z = np.zeros((self.sim_num, self.pr_num))
        self.Z0 = np.zeros(self.pr_num)
        self.Y0 = 0.
        # random_numbers is an array onwards
        self.random_numbers = np.array(self.random_numbers)

        # x, z are vector
    def generator(self, t, x, y, z):
        return 0.

    def generator_array(self, t, arr_x, arr_y, arr_z):
        ret = np.zeros(self.sim_num)
        for s in range(self.sim_num):
            ret[s] = self.generator(t, arr_x[s], arr_y[s], arr_z[s])
        return ret
        
    def backward_iteration(self):
        basis_generator = regression.PolyBasesGenerator(3)
        sliced_path = []
        sliced_bases = []
        #rgr = RandomForestRegressor(n_jobs = 3)
        rgr = linear_model.LinearRegression(n_jobs = 3)
        #import pdb;pdb.set_trace()
        # 
        for t in range(self.tg_num-2, 0, -1):
            #import pdb;pdb.set_trace()
            dt_t = self.dt[t+1] 
            # path is a list
            # sliced_bases: [[1, x, x^2], ...
            sliced_path = select_column(self.path, t)
            sliced_bases = map(basis_generator.all_bases, sliced_path)
            
            for p in range(0, self.pr_num):
                # tdata := y(t+1)*dw(t+1)
                # Be careful about the index t here
                z_data= self.Y * self.brownian_path[:, t, p] / dt_t
                rgr.fit(sliced_path, z_data)                
                self.Z[:,p] = rgr.predict(sliced_path)
            
            y_data = self.Y + self.generator_array(self.time_grid[t],
                                                   sliced_path, self.Y, self.Z) * dt_t
            rgr.fit(sliced_path, y_data)
            self.Y = rgr.predict(sliced_path)

        #import pdb; pdb.set_trace()

        self.Y0 = average(self.Y)
        for p in range(0, self.pr_num):
            self.Z0[p] = average(self.Y * self.brownian_path[:,0,p]) / self.time_grid[0]
 
    
