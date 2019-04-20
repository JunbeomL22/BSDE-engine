import numpy as np

class Process:
    """1-dimensional process. 'Process' class will be passed to 'AllProcesses' class.
    To the pricing engine, 'AllProcesses' class should be passed, not 'Process' class,
    even when the pricing is conducted only on a single asset"""
    def __init__(self, parameters):
        self.initial_value = parameters['initial value']
    #The following two functions will be overwritten over inheritence.
    #t (resp. x) stands for time (resp. spatial) variables
    def drift(self, t, x):
        pass
    def diffusion(self, t, x):
        pass

    
class ConstantGBMProcess(Process):
    """Geometric Brownian motnion process with constant parameters. """
    def __init__(self, parameters):
        ## ts : termstructure
        ## The class ts should be changed onwards
        Process.__init__(self, parameters)
        self.volatility = parameters['sigma']
        self.mu = parameters['mu']
    def drift(self, x, t):
        return self.mu * x
    def diffusion(self, x, t):
        return self.volatility * x

    
class AllProcesses:
    def __init__(self, processes, corr=[1.]):
        self.processes = processes
        # 'something_num' stands for number of something.
        # Here, p_num means the number of process
        self.p_num = len(processes)
        
        if len(corr)> 1:
            self.corr = np.linalg.cholesky(corr)
        else:
            self.corr = corr

