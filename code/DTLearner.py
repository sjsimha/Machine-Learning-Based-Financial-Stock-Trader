import numpy as np
from scipy import stats as st

class DTLearner(object):
    """

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

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
        train_data = np.concatenate((data_x, data_y[:, np.newaxis]), axis=1)
        self.dtree = self.build_dtree(train_data)

        if self.verbose:
            print(f'Done building decision tree with size {self.dtree.shape}')

    def query(self, points):
        """  		  	   		  		 			  		 			     			  	 
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        pred_y = np.array([self.predict_y(0, x) for x in points])
        return pred_y

    def predict_y(self, row, x):
        f, sv, l, r = int(self.dtree[row][0]), self.dtree[row][1], int(self.dtree[row][2]), int(self.dtree[row][3])

        if f == -1:
            return sv

        if x[f] <= sv:
            return self.predict_y(row + l, x)
        else:
            return self.predict_y(row + r, x)
    """
        Computes correlation coeeficient for all feature (or X) columns with the label (or Y column).
        Returns the feature with the highest computable absolute correlation. In the event correlation 
        could not be computed for any feature, -1 is returned 
    """
    def get_feature(self, in_data):
        f = in_data[:, 0:-1]
        y = in_data[:, -1][:, np.newaxis]
        c = np.abs(np.corrcoef(f, y, rowvar=False)[0:-1, -1])

        # check for NaNs; if all NaN slices, then return -1
        try:
            feature_index = np.nanargmax(c)
        except ValueError as VE:
            return -1

        return feature_index

    def generate_base_case_return(self, y_value):
        return np.array([[-1, y_value, -1, -1]])

    def get_mean_y(self, in_data):
        return np.mean(in_data[:, -1])

    def get_mode_y(self, in_data):
        md = st.mode(in_data[:, -1])
        return md.mode[0]

    def build_dtree(self, in_data):
        if in_data.shape[0] <= self.leaf_size:
            # Note: this also takes care of the base case for reaching last node (leaf)
            return self.generate_base_case_return(self.get_mode_y(in_data))
        else:
            y = in_data[:, -1]
            if np.all(y == y[0]):
                # all labels are same
                return self.generate_base_case_return(y[0])

        feature = self.get_feature(in_data) # Note: this method is overridden in RTLearner
        if feature == -1:
            # All NaN slices encountered,compress to leaf
            if self.verbose:
                print('No correlation found for any features, compressing to leaf')
            return self.generate_base_case_return(self.get_mode_y(in_data))

        # TODO change median to random split_val
        split_val = np.median(in_data[:, feature])
        right_data = in_data[in_data[:, feature] > split_val]

        if right_data.shape[0] == 0:
            # Median could not produce split, compress to leaf
            if self.verbose:
                print('Cant split, compressing to leaf')
            return self.generate_base_case_return(self.get_mode_y(in_data))

        left_data = in_data[in_data[:, feature] <= split_val]
        left_tree = self.build_dtree(left_data)
        right_tree = self.build_dtree(right_data)
        root_node = np.array([[feature, split_val, 1, left_tree.shape[0] + 1]])

        return np.concatenate((root_node, left_tree, right_tree))
