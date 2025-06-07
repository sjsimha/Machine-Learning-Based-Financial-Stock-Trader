import numpy as np
from DTLearner import DTLearner


class RTLearner(DTLearner):
    """
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """

    def __init__(self, leaf_size=1, verbose=False):
        super().__init__(leaf_size, verbose)

    def get_feature(self, in_data):
        return np.random.randint(0, high=in_data.shape[1]-1)
