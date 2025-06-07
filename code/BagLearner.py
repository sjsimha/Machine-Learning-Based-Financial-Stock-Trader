import numpy as np
from scipy import stats as st


class BagLearner(object):
    """

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        self.learners = []
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))

        if verbose:
            print(f'Created {self.bags} learners')

        """
        Constructor method  		  	   		  		 			  		 			     			  	 
        """

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "ssimha31"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
  		  	   		  		 			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        n = data_x.shape[0]
        for learner in self.learners:
            indexes = np.random.choice(n, n, replace=True)
            learner.add_evidence(data_x[indexes], data_y[indexes])

    def query(self, points):
        """  		  	   		  		 			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		  		 			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        """

        pred_list = []
        for learner in self.learners:
            pred_list.append(learner.query(points))

        n = np.array(pred_list)
        pred_y = st.mode(n, axis=0).mode
        pred_y = np.squeeze(pred_y, axis=0)

        return pred_y
