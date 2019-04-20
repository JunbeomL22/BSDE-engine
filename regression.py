from functools import reduce

class PolyBasesGenerator:
    '''When degree = 1, expand_basis:x -> 1, x.
    When degree =2, expand_basis: x -> 1, x, x**2'''
    def __init__(self, degree): #inputdata and target data
        assert isinstance(degree, int)
        assert degree > 0
        
        self.degree = degree


    def index_non_repreated(self, vec_length, sliced_deg):
        '''
        index_non_repeated receives the length of a vector and one spesific degree.
        Then it produces non-repeated incremental indices.
        ----------------------------------------------------------------
        ex1) When vec_length = 2 and sliced_deg = 2, 
               the output should be [[0,0], [0,1], [1,1]]
        ex2)When vec_length = 2 and sliced_deg = 3, 
               the output should be [[0,0,0], [0,0,1], [0,1,1], [1,1,1]]
        ----------------------------------------------------------------
        This set of indices will be used to generate regression basis. This is
        mainly for avoiding repeatition among multiplication. This part may be
        rewritten more elegantly.
        '''
        
        def appending_indices(vec, tail_vec):
            #This function will be used for 'functools.reduce'
            #All elements of the second argument should be singleton.
            assert not list(filter(lambda i: len(i) != 1, tail_vec))
            #Append the second argument only when they are incremental at tail. 
            return [i+j  for j in tail_vec for i in vec if i[-1] <= j[0]]
        
        # Let vec_length = 3. Then listified = [[0], [1], [2]]
        listified = [[i] for i in range(vec_length)]
        #Multiply listified so that we can generate indices using 'reduce'
        multiple_listified = [listified for i in range(sliced_deg)]
        if sliced_deg == 1:
            return listified
        else:
            return reduce(appending_indices, multiple_listified)

    def one_level_basis(self, vec, sliced_deg):
        '''
        note that the output is iterator. Use list, tuple, etc, to evaluate.
        '''
        assert isinstance(sliced_deg, int) and sliced_deg >=1
        v_length = len(vec)
        indices = self.index_non_repreated(v_length, sliced_deg)
        v_ind = [[vec[j] for j in i] for i in indices ]
        ret = map( lambda x: reduce( lambda a, b: a*b, x), v_ind)
        return ret

    def all_bases(self, vec):
        '''This functions generate all bases using the instance attribute
        'self.degree' including 1.'''
        ret = [bs for i in range(1, self.degree+1)
               for bs in self.one_level_basis(vec, i)]
        ret.insert(0, 1.)
        return ret
